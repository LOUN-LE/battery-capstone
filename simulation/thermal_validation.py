"""Validate measured pulse temperature with a separate, explicit thermal boundary.

Run with ``python -m simulation.thermal_validation --observed FILE --output DIR``.
The fitted effective heat-transfer coefficient is a candidate for this fixture.
It is never installed into the production model or used as an abuse/fire label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

import numpy as np
import pybamm
from scipy.integrate import trapezoid

from .joint_spme_bol_fit import derive_updates, homogeneous_state

ROOT = Path(__file__).resolve().parents[1]


def thermal_parameters(initial_cell_C, ambient_C, heat_transfer_W_m2_K):
    """Keep initial state, external boundary and cooling strength independent."""
    values = (initial_cell_C, ambient_C, heat_transfer_W_m2_K)
    if any(isinstance(value, bool) or not np.isscalar(value) for value in values):
        raise ValueError("Thermal inputs must be finite numeric scalars")
    if not np.isfinite(values).all() or min(values[:2]) <= -273.15 or values[2] < 0:
        raise ValueError("Temperatures must exceed absolute zero and cooling must be nonnegative")
    return {"Initial temperature [K]": float(initial_cell_C) + 273.15,
            "Ambient temperature [K]": float(ambient_C) + 273.15,
            "Total heat transfer coefficient [W.m-2.K-1]": float(heat_transfer_W_m2_K)}


def pulse_window(data, number, recovery_s=60):
    t, current, voltage, temperature, loaded = (
        np.asarray(data[key]) for key in ("t", "i", "v", "temperature_C", "loaded"))
    if t.ndim != 1 or any(a.shape != t.shape for a in (current, voltage, temperature, loaded)):
        raise ValueError("Input traces must share one time axis")
    if not all(np.isfinite(a).all() for a in (t, current, voltage, temperature, loaded)):
        raise ValueError("Input contains missing values")
    if not np.all(np.diff(t) > 0) or not np.isin(loaded, (0, 1)).all():
        raise ValueError("Invalid time axis or pulse mask")
    bounds = np.flatnonzero(np.diff(loaded.astype(int), prepend=0, append=0))
    starts, stops = bounds[::2], bounds[1::2]
    if not 2 <= number < len(starts):
        raise ValueError("Pulse needs a preceding and subsequent rest")
    start, stop = int(starts[number - 1]), int(stops[number - 1])
    first = start - 1
    anchor = int(np.searchsorted(t, t[first] - 10))
    last = int(np.searchsorted(t, t[stop] + recovery_s, side="right"))
    if anchor < stops[number - 2] or last > starts[number] or last <= stop + 1:
        raise ValueError("Insufficient clean rest around pulse")
    if np.max(np.abs(current[anchor:start])) > .01:
        raise ValueError("OCV anchor contains load")
    rows = slice(first, last)
    return {"t": t[rows] - t[first], "current_A": current[rows], "voltage_V": voltage[rows],
            "surface_C": temperature[rows], "loaded": loaded[rows].astype(bool),
            "initial_cell_C": float(temperature[first]),
            "anchor_V": float(np.median(voltage[anchor:start])), "source_rows": [first, last]}


def metrics(t, predicted, observed):
    error = np.asarray(predicted) - observed
    if len(t) < 2 or not np.all(np.diff(t) > 0) or not np.isfinite(error).all():
        raise ValueError("Invalid metric support")
    return {"time_weighted_rmse": float(np.sqrt(trapezoid(error**2, t) / (t[-1] - t[0]))),
            "sample_rmse": float(np.sqrt(np.mean(error**2))),
            "max_abs": float(np.max(abs(error)))}


def replay(window, config, profile, heat_transfer, ambient_C=25.0, mesh_scale=1):
    started = time.perf_counter()
    params = pybamm.ParameterValues(config["parameter_set"])
    params.update(profile["material_overrides"])
    _, inventory = derive_updates(params, 1.0, 1.0, 0.0)
    if not np.isclose(inventory["Q_Li_Ah"], profile["initial_particle_lithium_Ah"], rtol=0, atol=1e-8):
        raise ValueError("Material profile lithium inventory differs")
    state, _ = homogeneous_state(params, inventory, window["anchor_V"])
    params.update(state)
    params.update(thermal_parameters(window["initial_cell_C"], ambient_C, heat_transfer))
    params.update({"Current function [A]": pybamm.Interpolant(window["t"], window["current_A"], pybamm.t)})
    model = pybamm.lithium_ion.SPMe(options={**config["options"], "thermal": "lumped"})
    points = {"x_n": 20 * mesh_scale, "x_s": 20 * mesh_scale, "x_p": 20 * mesh_scale,
              "r_n": 30 * mesh_scale, "r_p": 30 * mesh_scale}
    sim = pybamm.Simulation(model, parameter_values=params, var_pts=points,
                           solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10,
                                                     on_extrapolation="error", options={"dt_max": 1.0}))
    sol = sim.solve(t_eval=[0, float(window["t"][-1])], t_interp=window["t"])
    if sol.termination != "final time" or abs(sol.t[-1] - window["t"][-1]) > 1e-8:
        raise ValueError(f"Incomplete pulse: {sol.termination}")
    names = {"model_C": "Volume-averaged cell temperature [C]", "model_V": "Voltage [V]",
             "model_A": "Current [A]", "capacity_Ah": "Discharge capacity [A.h]"}
    trace = {key: np.asarray(sol[name](window["t"]), dtype=float) for key, name in names.items()}
    if any(a.shape != window["t"].shape or not np.isfinite(a).all() for a in trace.values()):
        raise ValueError("Invalid solver observations")
    initial_error = abs(trace["model_C"][0] - window["initial_cell_C"])
    current_error = float(np.max(abs(trace["model_A"] - window["current_A"])))
    charge = float(trace["capacity_Ah"][-1] - trace["capacity_Ah"][0])
    residual = charge - float(trapezoid(window["current_A"], window["t"]) / 3600)
    integrity = initial_error < 1e-8 and current_error < 1e-10 and abs(residual) < 1e-6
    result = {"heat_transfer_W_m2_K": heat_transfer, "ambient_C": ambient_C,
              "initial_temperature_error_C": float(initial_error), "current_error_A": current_error,
              "charge_residual_Ah": residual, "integrity_passed": bool(integrity),
              "wall_s": time.perf_counter() - started}
    for phase, mask in (("loaded", window["loaded"]),
                        ("recovery", (~window["loaded"]) & (window["t"] > 0))):
        result[phase] = {"temperature_C": metrics(window["t"][mask], trace["model_C"][mask], window["surface_C"][mask]),
                         "voltage_mV": metrics(window["t"][mask], trace["model_V"][mask] * 1000, window["voltage_V"][mask] * 1000)}
    return result, trace


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/model.json"
    profile_path = ROOT / "configs/m50_c_joint_v3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    hashes = {str(path.resolve()): file_sha(path) for path in (args.observed, config_path, profile_path, Path(__file__))}
    with np.load(args.observed, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train_pulses, test_pulses = (2, 3), tuple(range(4, 13))
    candidates = (5.0, 10.0, 20.0, 40.0)
    report = {"status": "running", "source_sha256": hashes, "pybamm_version": pybamm.__version__,
              "train_pulses": train_pulses, "test_pulses": test_pulses,
              "candidate_heat_transfer_W_m2_K": candidates, "fit_objective": "mean of train pulse loaded and recovery time-weighted temperature RMSE",
              "measurement": "Volume-average cell prediction compared with measured surface temperature",
              "boundary_source": "Nominal 25 C fixture setting; time-resolved cooler boundary unavailable",
              "data_split": "Test pulses excluded from this fit; previously inspected in other diagnostics",
              "initialization": "Independent pulse states from preceding rest voltage and initial observed temperature",
              "thermal_validation_scope": "Short GITT pulses in one 25 C LG M50T fixture",
              "full_temperature_range_validated": False, "thermal_runaway_validated": False, "fire_validated": False,
              "training": [], "validation": []}
    traces = {}
    for cooling in candidates:
        rows = []
        for number in train_pulses:
            result, _ = replay(pulse_window(data, number), config, profile, cooling)
            rows.append(result)
            print(f"train pulse={number} h={cooling} completed", flush=True)
        objective = float(np.mean([row[phase]["temperature_C"]["time_weighted_rmse"]
                                   for row in rows for phase in ("loaded", "recovery")]))
        report["training"].append({"heat_transfer_W_m2_K": cooling, "objective_C": objective, "runs": rows})
    chosen = min(report["training"], key=lambda row: row["objective_C"])["heat_transfer_W_m2_K"]
    report["chosen_heat_transfer_W_m2_K"] = chosen
    for number in test_pulses:
        window = pulse_window(data, number)
        for cooling in sorted({10.0, chosen}):
            result, arrays = replay(window, config, profile, cooling)
            report["validation"].append({"pulse": number, **result})
            prefix = f"pulse{number}_h{cooling:g}"
            for key, values in {**arrays, "t": window["t"], "observed_C": window["surface_C"], "observed_V": window["voltage_V"]}.items():
                traces[f"{prefix}_{key}"] = values
            print(f"validation pulse={number} h={cooling} completed", flush=True)
    report["summary"] = {str(cooling): {phase: float(np.mean([
        row[phase]["temperature_C"]["time_weighted_rmse"] for row in report["validation"]
        if row["heat_transfer_W_m2_K"] == cooling])) for phase in ("loaded", "recovery")}
        for cooling in sorted({10.0, chosen})}
    source_unchanged = all(file_sha(path) == value for path, value in hashes.items())
    report["integrity_passed"] = source_unchanged and all(row["integrity_passed"] for row in report["validation"]) and all(
        row["integrity_passed"] for fit in report["training"] for row in fit["runs"])
    report["source_hashes_unchanged"] = source_unchanged
    report["status"] = "complete_diagnostic" if report["integrity_passed"] else "failed_integrity"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.output / "traces.npz", **traces)
    rows = [{"pulse": row["pulse"], "heat_transfer_W_m2_K": row["heat_transfer_W_m2_K"], "phase": phase,
             "temperature_rmse_C": row[phase]["temperature_C"]["time_weighted_rmse"],
             "voltage_rmse_mV": row[phase]["voltage_mV"]["time_weighted_rmse"], "integrity_passed": row["integrity_passed"]}
            for row in report["validation"] for phase in ("loaded", "recovery")]
    with (args.output / "temperature_validation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], "chosen_h": chosen, "summary": report["summary"]}), flush=True)
    if not report["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
