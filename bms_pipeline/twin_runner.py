"""Bounded, state-preserving replay of measured Imperial M50T C10 input.

The preparation is a declared CC/CV/rest protocol, not exact measured-current
replay. The observed C10 current is linearly interpolated at its original times.
Numerical connection checks and observed voltage errors are separate results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ("cell_id", "session_id", "time_s", "voltage_V", "current_A",
          "surface_temperature_C", "source_row", "phase")
MESH = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 40, "r_p": 40}
RTOL, ATOL = 1e-8, 1e-10
STATE_RELATIVE_TOL = 1e-5
VOLTAGE_TOL_V = 1e-5
CHARGE_TOL_AH = 2e-5


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Canonical eight input columns required")
        rows = list(reader)
    if len(rows) < 2 or len({(r['cell_id'], r['session_id']) for r in rows}) != 1:
        raise ValueError("One nonempty cell/session is required")
    data = np.asarray([[float(r[k]) for k in ("time_s", "voltage_V", "current_A")]
                       for r in rows], dtype=float)
    if not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("Finite strictly increasing input times required")
    if np.any(data[:, 1] <= 0) or np.max(abs(data[:, 2])) > 5:
        raise ValueError("Input outside the bounded 5 A exploratory domain")
    return rows, data


def rest_anchor(rows):
    candidates = [r for r in rows if r["phase"] == "preparation_step_0_rest"
                  and 0 <= float(r["time_s"]) <= 120
                  and abs(float(r["current_A"])) <= 1e-6]
    if len(candidates) < 3:
        raise ValueError("Three interior initial-rest samples required")
    chosen = candidates[-3:]
    return float(np.mean([float(r["voltage_V"]) for r in chosen])), {
        "source_rows": [int(r["source_row"]) for r in chosen],
        "source_times_s": [float(r["time_s"]) for r in chosen],
        "voltage_V": [float(r["voltage_V"]) for r in chosen],
        "rule": "last three zero-current observations inside the declared initial 120 s rest",
        "assumption": "homogeneous initial concentration; 120 s rest does not identify equilibrium or prior gradients",
    }


def make_parameters(config, calibration, target_voltage):
    import pybamm
    from scipy.optimize import brentq
    if config["parameter_set"] != "OKane2022" or config["options"]["thermal"] != "isothermal":
        raise ValueError("This bounded implementation requires frozen isothermal OKane2022")
    params = pybamm.ParameterValues(config["parameter_set"])
    params.update(calibration["material_overrides"])
    if (float(params["Lower voltage cut-off [V]"]) != float(config["voltage_min_V"])
            or float(params["Upper voltage cut-off [V]"]) != float(config["voltage_max_V"])):
        raise ValueError("Frozen voltage bounds differ from the actual parameter set")
    faraday = float(pybamm.constants.F.value)
    area = params["Electrode width [m]"] * params["Electrode height [m]"]
    q = {}
    for side in ("Negative", "Positive"):
        q[side] = (faraday * area * params[f"{side} electrode thickness [m]"] *
                   params[f"{side} electrode active material volume fraction"] *
                   params[f"Maximum concentration in {side.lower()} electrode [mol.m-3]"] / 3600)
    inventory = float(calibration["initial_particle_lithium_Ah"])
    lo = max(1e-6, (inventory - q["Positive"] * (1 - 1e-6)) / q["Negative"])
    hi = min(1 - 1e-6, (inventory - q["Positive"] * 1e-6) / q["Negative"])

    def voltage(x):
        y = (inventory - q["Negative"] * x) / q["Positive"]
        value = params["Positive electrode OCP [V]"](y) - params["Negative electrode OCP [V]"](x)
        return float(np.asarray(value.evaluate() if hasattr(value, "evaluate") else value).item())

    x = float(brentq(lambda n: voltage(n) - target_voltage, lo, hi, xtol=1e-12))
    y = (inventory - q["Negative"] * x) / q["Positive"]
    concentrations = {
        "Initial concentration in negative electrode [mol.m-3]": x * params["Maximum concentration in negative electrode [mol.m-3]"],
        "Initial concentration in positive electrode [mol.m-3]": y * params["Maximum concentration in positive electrode [mol.m-3]"],
    }
    params.update(concentrations)
    params.update({"Initial temperature [K]": 298.15, "Ambient temperature [K]": 298.15})
    residual = q["Negative"] * x + q["Positive"] * y - inventory
    if abs(residual) > 1e-10 or abs(voltage(x) - target_voltage) > 1e-8:
        raise ValueError("Initial voltage/inventory construction failed")
    return params, {"target_voltage_V": target_voltage, "initial_equilibrium_model_voltage_V": voltage(x),
                    "negative_stoichiometry": x, "positive_stoichiometry": y,
                    "initial_concentrations": concentrations, "particle_lithium_Ah": inventory,
                    "inventory_residual_Ah": residual, "initial_soc_remapping_used": False}


def fingerprint(state):
    values = np.asarray(state.all_ys[-1], dtype=float).reshape(-1)
    return hashlib.sha256(values.tobytes() + np.asarray([state.t[-1]], dtype=float).tobytes()).hexdigest()


def simulation(params, config, steps):
    import pybamm
    return pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config["options"]),
        parameter_values=params.copy(), var_pts=MESH,
        experiment=pybamm.Experiment([tuple(steps)], temperature=298.15),
        solver=pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL))


def prepare_state(params, config, preparation_rows):
    import pybamm
    # Native declared controls. Actual measured current and control timings are
    # retained in the separate preparation CSV and are not silently substituted.
    steps = [pybamm.step.rest(120, period=10, skip_ok=False),
             pybamm.step.current(-1.5, duration=20000, termination="4.2 V", period=10, skip_ok=False),
             pybamm.step.voltage(4.2, duration=20000, termination="0.05 A", period=10, skip_ok=False),
             pybamm.step.rest(7230, period=10, skip_ok=False)]
    started = time.perf_counter()
    sol = simulation(params, config, steps).solve(calc_esoh=False)
    pieces = sol.cycles[-1].steps
    expected = ["final time", "event: Voltage > 4.2 [V] [experiment]",
                "event: abs(Current [A]) < 0.05 [A] [experiment]", "final time"]
    if len(pieces) != 4 or [str(p.termination) for p in pieces] != expected:
        raise ValueError("Preparation controls did not reach their declared endpoints")
    records = []
    for name, piece in zip(("initial_rest", "constant_current_charge", "constant_voltage_charge", "rest_before_C10"), pieces):
        t = np.asarray(piece["Time [s]"].entries, dtype=float)
        v = np.asarray(piece["Voltage [V]"].entries, dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("Nonfinite preparation voltage")
        records.append({"phase": name, "duration_s": float(t[-1] - t[0]),
                        "termination": str(piece.termination), "end_voltage_V": float(v[-1])})
    observed_span = float(preparation_rows[-1]["time_s"]) - float(preparation_rows[0]["time_s"])
    report = {"wall_s": time.perf_counter() - started, "controls": records,
              "model_duration_s": float(sol.t[-1] - sol.t[0]), "observed_preparation_sample_span_s": observed_span,
              "model_minus_observed_sample_span_s": float(sol.t[-1] - sol.t[0]) - observed_span,
              "raw_preparation_current_forced": False,
              "method": "declared 120 s rest; -1.5 A to 4.2 V; 4.2 V to 0.05 A; 7230 s rest",
              "state_carried_through_all_controls": True,
              "state_fingerprint": fingerprint(sol.last_state)}
    return sol.last_state.copy(), report


def slice_table(table, start, end):
    table = np.asarray(table, dtype=float)
    if not table[0, 0] <= start < end <= table[-1, 0]:
        raise ValueError("Slice must be inside available current history")
    middle = table[(table[:, 0] > start) & (table[:, 0] < end)]
    return np.vstack(([start, np.interp(start, table[:, 0], table[:, 1])], middle,
                      [end, np.interp(end, table[:, 0], table[:, 1])])) - np.array([start, 0.])


VARIABLES = {"voltage_V": "Voltage [V]", "current_A": "Current [A]",
             "discharge_capacity_Ah": "Discharge capacity [A.h]",
             "SEI_lithium_loss_Ah": "Loss of capacity to negative SEI [A.h]",
             "plating_inventory_Ah": "Loss of capacity to negative lithium plating [A.h]",
             "dead_lithium_mol_m3": "X-averaged negative dead lithium concentration [mol.m-3]",
             "particle_lithium_mol": "Total lithium in particles [mol]",
             "model_temperature_K": "Volume-averaged cell temperature [K]"}


def observed_step(params, config, initial, table):
    import pybamm
    before = fingerprint(initial)
    started = time.perf_counter()
    step = pybamm.step.current(np.asarray(table, dtype=float), duration=float(table[-1, 0]), period=1, skip_ok=False)
    solution = simulation(params, config, [step]).solve(starting_solution=initial.copy(), calc_esoh=False)
    piece = solution.cycles[-1].steps[-1]
    t = np.asarray(piece["Time [s]"].entries, dtype=float)
    local_t = t - t[0]
    data = {name: np.asarray(piece[key].entries, dtype=float) for name, key in VARIABLES.items()}
    if str(piece.termination) != "final time" or abs(local_t[-1] - table[-1, 0]) > 1e-6:
        raise ValueError("Observed replay stopped before the recorded endpoint")
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite observed replay output")
    expected = np.interp(local_t, table[:, 0], table[:, 1])
    current_error = float(np.max(abs(data["current_A"] - expected)))
    charge = float(data["discharge_capacity_Ah"][-1] - data["discharge_capacity_Ah"][0])
    source_charge = float(np.trapezoid(table[:, 1], table[:, 0]) / 3600)
    q_error = charge - source_charge
    model = piece.all_models[0]
    n = model.len_rhs
    before_y = np.asarray(initial.all_ys[-1], dtype=float).reshape(-1)
    start_y = np.asarray(piece.all_ys[0], dtype=float)[:, 0]
    continuity = float(np.max(abs(before_y[:n] - start_y[:n])))
    if current_error > 1e-8 or abs(q_error) > CHARGE_TOL_AH or continuity > 1e-7:
        raise ValueError(f"Observed input/state/charge check failed: {current_error}, {q_error}, {continuity}")
    if fingerprint(initial) != before:
        raise ValueError("Observed replay mutated its input checkpoint")
    audit = {"wall_s": time.perf_counter() - started, "duration_s": float(local_t[-1]),
             "termination": str(piece.termination), "current_max_error_A": current_error,
             "charge_integral_source_Ah": source_charge, "charge_integral_model_Ah": charge,
             "charge_residual_Ah": q_error, "initial_differential_state_max_jump": continuity,
             "input_fingerprint": before, "terminal_fingerprint": fingerprint(solution.last_state)}
    return solution.last_state.copy(), {"time_s": local_t, **data}, audit


def save_checkpoint(directory, state, identity):
    import pybamm
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "state.pkl"
    with payload.open("wb") as stream:
        pickle.dump(state.last_state.copy(), stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(directory / "state.npz", y=np.asarray(state.all_ys[-1], dtype=float), t=np.asarray(state.t))
    receipt = {"pickle_sha256": digest(payload), "arrays_sha256": digest(directory / "state.npz"),
               "state_fingerprint": fingerprint(state), "identity": identity,
               "pybamm_version": pybamm.__version__, "python_version": platform.python_version(),
               "platform": platform.platform(), "trusted_local_output_only": True}
    write_json(directory / "receipt.json", receipt)
    return receipt


def load_checkpoint(directory, identity):
    """Load only this runner's local, integrity-checked output in the same runtime."""
    import pybamm
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    if receipt["identity"] != identity or receipt["pybamm_version"] != pybamm.__version__ or receipt["python_version"] != platform.python_version() or receipt["platform"] != platform.platform():
        raise ValueError("Checkpoint source/config/runtime identity differs")
    if digest(directory / "state.pkl") != receipt["pickle_sha256"] or digest(directory / "state.npz") != receipt["arrays_sha256"]:
        raise ValueError("Checkpoint integrity mismatch")
    with (directory / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    with np.load(directory / "state.npz", allow_pickle=False) as arrays:
        if not np.array_equal(np.asarray(state.all_ys[-1], dtype=float), arrays["y"]) or not np.array_equal(np.asarray(state.t), arrays["t"]):
            raise ValueError("Checkpoint disagrees with independently saved state arrays")
    if fingerprint(state) != receipt["state_fingerprint"]:
        raise ValueError("Restored physical state differs")
    return state


def compare_states(a, b):
    av = np.asarray(a.all_ys[-1], dtype=float).reshape(-1)
    bv = np.asarray(b.all_ys[-1], dtype=float).reshape(-1)
    if av.shape != bv.shape:
        raise ValueError("Compared states use different layouts")
    scaled = np.abs(av - bv) / np.maximum(1., np.abs(av))
    return {"max_absolute_state_difference": float(np.max(abs(av - bv))),
            "max_scaled_state_difference": float(np.max(scaled)),
            "state_elements": int(len(av)),
            "state_scale_definition": "abs(a-b)/max(1,abs(a)) elementwise"}


def future_branch(params, config, initial, current_A, duration_s=120.):
    import pybamm
    before = fingerprint(initial)
    started = time.perf_counter()
    try:
        step = pybamm.step.current(-abs(current_A), duration=duration_s,
                                  termination="4.2 V", period=1, skip_ok=False)
        solution = simulation(params, config, [step]).solve(starting_solution=initial.copy(), calc_esoh=False)
        piece = solution.cycles[-1].steps[-1]
        t = np.asarray(piece["Time [s]"].entries, dtype=float)
        actual_current = np.asarray(piece["Current [A]"].entries, dtype=float)
        v = np.asarray(piece["Voltage [V]"].entries, dtype=float)
        if not np.isfinite(v).all() or np.max(abs(actual_current + abs(current_A))) > 1e-8:
            raise ValueError("Future current/finite-output guard failed")
        termination = str(piece.termination)
        if termination not in ("final time", "event: Voltage > 4.2 [V] [experiment]"):
            raise ValueError("Unexpected future endpoint: " + termination)
        result = {"status": "completed", "duration_s": float(t[-1] - t[0]),
                  "termination": termination, "end_voltage_V": float(v[-1]),
                  "maximum_voltage_V": float(v.max()), "minimum_voltage_V": float(v.min())}
        for name in ("SEI_lithium_loss_Ah", "plating_inventory_Ah", "dead_lithium_mol_m3", "discharge_capacity_Ah"):
            value = np.asarray(piece[VARIABLES[name]].entries, dtype=float)
            result[name + "_increment"] = float(value[-1] - value[0])
        expected_charge = -abs(current_A) * result["duration_s"] / 3600
        result["charge_residual_Ah"] = result["discharge_capacity_Ah_increment"] - expected_charge
        if abs(result["charge_residual_Ah"]) > 1e-7:
            raise ValueError("Future branch charge balance failed")
    except pybamm.SolverError as error:
        message = str(error)
        if "initial" not in message.lower() or not any(s in message.lower() for s in ("voltage", "infeasible")):
            raise
        result = {"status": "blocked_by_initial_voltage_condition", "duration_s": 0.,
                  "termination": message, "end_voltage_V": None}
    if fingerprint(initial) != before:
        raise ValueError("Future branch mutated the shared judgment-time state")
    result.update(requested_charging_current_A=abs(current_A), requested_duration_s=duration_s,
                  input_state_fingerprint=before, input_state_unchanged=True,
                  wall_s=time.perf_counter() - started, conditional_model_projection=True)
    return result


def run_twin(input_csv, decisions_csv, outdir, configdir=None, preparation_csv=None):
    import pybamm
    began = time.perf_counter()
    input_csv, outdir = Path(input_csv).resolve(), Path(outdir).resolve()
    configdir = Path(configdir or HERE / "configs").resolve()
    preparation_csv = Path(preparation_csv or input_csv.with_name(input_csv.name.replace("_prefix.csv", "_preparation.csv"))).resolve()
    if preparation_csv == input_csv:
        raise ValueError("Explicit preparation CSV required for non-prefix input filenames")
    rows, observed = read_input(input_csv)
    preparation_rows, _ = read_input(preparation_csv)
    if rows[0]["cell_id"] != preparation_rows[0]["cell_id"]:
        raise ValueError("Preparation belongs to a different cell")
    if float(preparation_rows[-1]["time_s"]) >= observed[0, 0]:
        raise ValueError("Preparation must end before the observed C10 window")
    duration = float(observed[-1, 0] - observed[0, 0])
    if not 0 < duration <= 600.001:
        raise ValueError("Prototype observation budget is at most 600 s")
    config = read_json(configdir / "frozen_model.json")
    calibration = read_json(configdir / "cell_calibration.json")
    identities = {"input_sha256": digest(input_csv), "preparation_sha256": digest(preparation_csv),
                  "model_sha256": digest(configdir / "frozen_model.json"),
                  "calibration_sha256": digest(configdir / "cell_calibration.json"),
                  "runner_sha256": digest(__file__), "decisions_sha256": digest(decisions_csv),
                  "cell_id": rows[0]["cell_id"],
                  "session_id": rows[0]["session_id"], "mesh": MESH, "rtol": RTOL, "atol": ATOL}
    with Path(decisions_csv).open(encoding="utf-8-sig", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    selected = [r for r in decisions if r["cell_id"] == rows[0]["cell_id"]
                and r["session_id"] == rows[0]["session_id"]
                and r["selected_by_model"].strip().lower() in ("true", "1")]
    for decision in selected:
        if not observed[0, 0] <= float(decision["issue_time_s"]) < float(decision["target_time_s"]) <= observed[-1, 0] + 1e-6:
            raise ValueError("Selected interval lies outside the available observed history")
        if not float(decision["target_time_s"]) <= float(decision["decision_cutoff_s"]) <= observed[-1, 0] + 1e-6:
            raise ValueError("Residual-based selection cannot precede its observed target")
    outdir.mkdir(parents=True, exist_ok=False)
    target, anchor = rest_anchor(preparation_rows)
    params, initialization = make_parameters(config, calibration, target)
    initial, preparation = prepare_state(params, config, preparation_rows)
    table = np.column_stack((observed[:, 0] - observed[0, 0], observed[:, 2]))
    continuous, continuous_trace, continuous_audit = observed_step(params, config, initial, table)
    split_at = float(table[len(table) // 2, 0])
    first_state, first_trace, first_audit = observed_step(params, config, initial, slice_table(table, 0., split_at))
    save_checkpoint(outdir / "midpoint_checkpoint", first_state, identities)
    reloaded = load_checkpoint(outdir / "midpoint_checkpoint", identities)
    split_state, second_trace, second_audit = observed_step(params, config, reloaded, slice_table(table, split_at, duration))
    comparison = compare_states(continuous, split_state)
    split_time = np.concatenate((first_trace["time_s"], second_trace["time_s"][1:] + split_at))
    split_voltage = np.concatenate((first_trace["voltage_V"], second_trace["voltage_V"][1:]))
    grid = observed[:, 0] - observed[0, 0]
    continuous_voltage = np.interp(grid, continuous_trace["time_s"], continuous_trace["voltage_V"])
    split_voltage_grid = np.interp(grid, split_time, split_voltage)
    comparison.update(split_time_s=split_at, observed_duration_s=duration,
                      maximum_voltage_difference_V=float(np.max(abs(continuous_voltage - split_voltage_grid))),
                      state_scaled_tolerance=STATE_RELATIVE_TOL, voltage_tolerance_V=VOLTAGE_TOL_V,
                      restored_midpoint_state_exact=True)
    comparison["passed"] = (comparison["max_scaled_state_difference"] <= STATE_RELATIVE_TOL and
                            comparison["maximum_voltage_difference_V"] <= VOLTAGE_TOL_V)
    if not comparison["passed"]:
        write_json(outdir / "failed_continuity_check.json", comparison)
        raise ValueError("Continuous versus split/checkpoint replay failed")
    save_checkpoint(outdir / "terminal_checkpoint", split_state, identities)
    terminal = load_checkpoint(outdir / "terminal_checkpoint", identities)
    future = [future_branch(params, config, terminal, current) for current in (.5, 1.)]
    if len({r["input_state_fingerprint"] for r in future}) != 1:
        raise ValueError("Future conditions did not start from one common state")
    trace_rows = []
    for index, source in enumerate(rows):
        result = {"cell_id": source["cell_id"], "session_id": source["session_id"],
                  "source_row": source["source_row"], "source_time_s": float(source["time_s"]),
                  "elapsed_s": float(grid[index]), "observed_voltage_V": float(source["voltage_V"]),
                  "observed_surface_temperature_C": source["surface_temperature_C"]}
        for name, values in continuous_trace.items():
            if name != "time_s":
                result["model_" + name] = float(np.interp(grid[index], continuous_trace["time_s"], values))
        result["voltage_error_mV"] = 1000 * (result["model_voltage_V"] - result["observed_voltage_V"])
        trace_rows.append(result)
    write_csv(outdir / "observed_replay.csv", trace_rows, list(trace_rows[0]))
    selected_rows = []
    for decision in selected:
        a = float(decision["issue_time_s"]) - observed[0, 0]
        b = float(decision["target_time_s"]) - observed[0, 0]
        mask = (grid >= a - 1e-6) & (grid <= b + 1e-6)
        error = continuous_voltage[mask] - observed[mask, 1]
        record = {"cell_id": rows[0]["cell_id"], "issue_time_s": float(decision["issue_time_s"]),
                  "target_time_s": float(decision["target_time_s"]), "decision_cutoff_s": float(decision["decision_cutoff_s"]),
                  "prefix_elapsed_s": a, "interval_voltage_RMSE_mV": float(np.sqrt(np.mean(error ** 2)) * 1000),
                  "compared_rows": int(mask.sum()), "selection_reason": decision.get("selection_reason", ""),
                  "state_reset_at_window_start": False}
        for name in ("SEI_lithium_loss_Ah", "plating_inventory_Ah", "dead_lithium_mol_m3"):
            values = continuous_trace[name]
            record[name + "_increment"] = float(np.interp(b, continuous_trace["time_s"], values) - np.interp(a, continuous_trace["time_s"], values))
        selected_rows.append(record)
    selected_fields = list(selected_rows[0]) if selected_rows else ["cell_id", "issue_time_s", "target_time_s", "selection_reason"]
    write_csv(outdir / "selected_window_results.csv", selected_rows, selected_fields)
    errors = continuous_voltage - observed[:, 1]
    report = {"status": "completed", "scope": "measured-current connection prototype with conditional model outputs",
              "identity": identities, "source_time_origin_s": float(observed[0, 0]),
              "source_time_end_s": float(observed[-1, 0]), "input_rows": len(rows),
              "selected_observed_windows": len(selected_rows), "selected_windows_evaluated": len(selected_rows),
              "initialization": {**initialization, "anchor": anchor}, "preparation": preparation,
              "observed_current_interpolation": "piecewise linear at original source sample times; no pack scaling",
              "temperature": {"model": "assumed isothermal 25 C based on source base-cooling setpoint; not measured cell temperature", "surface_measurements_preserved": True,
                              "surface_temperature_used_as_ambient": False, "temperature_reproduction_validated": False},
              "continuous_replay": continuous_audit, "split_replay": [first_audit, second_audit],
              "continuity_verification": comparison, "future_conditions": future,
              "numerical_connection_checks_passed": True, "numerical_validation_passed": True,
              "source_hashes_unchanged": True,
              "physical_accuracy": {"voltage_RMSE_mV": float(np.sqrt(np.mean(errors ** 2)) * 1000),
                                    "voltage_max_abs_error_mV": float(np.max(abs(errors)) * 1000),
                                    "voltage_bias_mV": float(np.mean(errors) * 1000),
                                    "compared_rows": len(rows), "physical_acceptance_threshold_predeclared": False,
                                    "physical_model_validation_passed": False},
              "degradation_cause_identified": False, "lifetime_validation_passed": False,
              "model_quantity_definitions": {
                  "SEI_lithium_loss_Ah": "model lithium consumed by the negative-electrode SEI reaction, expressed as charge",
                  "plating_inventory_Ah": "model metallic plus dead-lithium inventory in charge units; may decrease by stripping and is not irreversible capacity loss",
                  "dead_lithium_mol_m3": "model x-averaged negative dead-lithium concentration; no direct experimental measurement or identified cause",
                  "increments": "end minus start over the stated interval; compare only with matched exposure and endpoint definitions"},
              "uses_observed_future_voltage_as_physics_input": False,
              "judgment_state_time_s": float(observed[-1, 0]),
              "runtime": {"pybamm_version": pybamm.__version__, "python_version": platform.python_version(),
                          "platform": platform.platform(), "solver": "IDAKLU", "threads_per_library": 1},
              "timing": {"preparation_s": preparation["wall_s"], "observed_reference_s": continuous_audit["wall_s"],
                         "split_verification_s": first_audit["wall_s"] + second_audit["wall_s"],
                         "future_conditions_s": sum(x["wall_s"] for x in future)},
              "simulation_solve_calls": 6,
              "compute_savings_demonstrated": False,
              "limitations": ["Selection extracts observed-window results from a continuous prefix replay; unselected prefix computation is still counted.",
                              "The continuous and split replays are numerical verification work, not a measured algorithm speedup.",
                              "Preparation reproduces declared controls rather than the raw preparation-current waveform; its duration can differ.",
                              "Homogeneous initial concentrations and C-frozen material parameters do not identify true cell history or unique degradation causes.",
                              "Future charge projections are model examples from the final observed state, not measured future outcomes.",
                              "Future conditions may stop at different times at 4.2 V; their raw cumulative losses do not establish a safe-current ranking.",
                              "SEI/plating quantities are model internal outputs, not measured irreversible lifetime loss or identified causes.",
                              "Checkpoint pickle is for this trusted output and identical runtime only; regenerate preparation on a different platform."]}
    for name, path in (("input_sha256", input_csv), ("preparation_sha256", preparation_csv),
                       ("model_sha256", configdir / "frozen_model.json"),
                       ("calibration_sha256", configdir / "cell_calibration.json"), ("runner_sha256", Path(__file__)),
                       ("decisions_sha256", Path(decisions_csv))):
        if digest(path) != identities[name]:
            raise ValueError("Source/config changed during replay: " + name)
    report["observed_replay"] = {"rows": len(rows), "path": "observed_replay.csv", **report["physical_accuracy"]}
    report["future_candidates"] = future
    report["checkpoints"] = {"midpoint": "midpoint_checkpoint", "terminal": "terminal_checkpoint",
                             "midpoint_restoration_exact": True, "terminal_restoration_exact": True}
    report["timing"]["runner_before_final_report_write_s"] = time.perf_counter() - began
    write_json(outdir / "twin_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configdir", type=Path)
    parser.add_argument("--preparation", type=Path)
    args = parser.parse_args()
    result = run_twin(args.input, args.decisions, args.output, args.configdir, args.preparation)
    print(json.dumps({"status": result["status"], "selected_windows": result["selected_observed_windows"],
                      "numerical_connection_checks_passed": result["numerical_connection_checks_passed"],
                      "physical_accuracy": result["physical_accuracy"], "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
