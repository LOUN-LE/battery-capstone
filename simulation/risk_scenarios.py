"""Bounded SPMeT scenario datasets with explicit simulation-only labels.

Run with ``python -m simulation.risk_scenarios --output PATH --workers 8``.
Each independent case starts from a homogeneous fresh-cell SOC, charges at CC
until the requested horizon or 4.2 V, then rests. Thermal/runaway/fire ground
truth requires additional validated physics and measurements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"

import numpy as np
import pybamm

from simulation.rebuilt_lifetime import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "spmet-risk-scenarios-v1"
VARIABLES = {
    "time_s": "Time [s]",
    "current_A": "Current [A]",
    "voltage_V": "Voltage [V]",
    "temperature_K": "Volume-averaged cell temperature [K]",
    "discharge_capacity_Ah": "Discharge capacity [A.h]",
    "plating_net_current_A_m2": "X-averaged negative electrode lithium plating interfacial current density [A.m-2]",
    "metallic_lithium_mol_m3": "X-averaged negative lithium plating concentration [mol.m-3]",
    "dead_lithium_mol_m3": "X-averaged negative dead lithium concentration [mol.m-3]",
    "plating_inventory_Ah": "Loss of capacity to negative lithium plating [A.h]",
    "sei_loss_Ah": "Loss of capacity to negative SEI [A.h]",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    initial_soc: float
    initial_temperature_C: float
    ambient_temperature_C: float
    charge_current_A: float
    heat_transfer_W_m2K: float
    charge_horizon_s: float = 1800.0
    rest_s: float = 300.0
    voltage_limit_V: float = 4.2
    temperature_stop_C: float = 60.0
    sample_period_s: float = 1.0
    plating_threshold_A_m2: float = 1e-3
    plating_persistence_s: float = 5.0
    mesh_scale: int = 8
    rtol: float = 1e-7
    atol: float = 1e-9

    def validate(self):
        for key, value in asdict(self).items():
            if key != "scenario_id" and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{key} must be a finite number")
        if not self.scenario_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.scenario_id):
            raise ValueError("Invalid scenario_id")
        if not 0 < self.initial_soc < 1:
            raise ValueError("initial_soc must be inside (0, 1)")
        if min(self.initial_temperature_C, self.ambient_temperature_C) <= -273.15:
            raise ValueError("Invalid absolute temperature")
        if self.initial_temperature_C >= self.temperature_stop_C:
            raise ValueError("Initial temperature must be below the research stop")
        for key in ("charge_horizon_s", "rest_s", "heat_transfer_W_m2K", "sample_period_s", "plating_threshold_A_m2", "plating_persistence_s", "rtol", "atol"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.charge_current_A < 0 or self.mesh_scale not in (1, 2, 4, 8, 16):
            raise ValueError("Charge magnitude must be nonnegative; supported mesh scales are 1, 2, 4, 8, 16")
        if self.sample_period_s > self.plating_persistence_s:
            raise ValueError("Sample interval exceeds persistence window")
        if not 2.5 < self.voltage_limit_V <= 4.2:
            raise ValueError("Requested voltage limit outside the model protocol")


def pilot_scenarios():
    combinations = itertools.product((0.0, 10.0, 25.0, 40.0), (0.5, 2.5), (0.2, 0.6), (5.0, 10.0))
    return [Scenario(f"case_{i:03d}", soc, temp, temp, current, cooling)
            for i, (temp, current, soc, cooling) in enumerate(combinations, 1)]


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance():
    sources = (Path(__file__), ROOT / "simulation/rebuilt_lifetime.py",
               ROOT / "configs/model.json", ROOT / "configs/m50_c_joint_v3.json")
    return {"sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            "pybamm_version": pybamm.__version__, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "dependencies": {
                name: importlib.metadata.version(name) for name in ("pybammsolvers", "casadi", "scipy")}}


def persistent_onset(time_s, net_current, threshold, persistence):
    """First sampled start with negative net plating flux sustained for duration."""
    start = None
    for stamp, current in zip(time_s, net_current):
        if current < -threshold:
            if start is None:
                start = float(stamp)
            if stamp - start >= persistence:
                return start
        else:
            start = None
    return None


def csv_atomic(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty table")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def simulate(scenario):
    scenario.validate()
    model_config = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "configs/m50_c_joint_v3.json").read_text(encoding="utf-8"))
    options = {**model_config["options"], "thermal": "lumped"}
    model = pybamm.lithium_ion.SPMe(options=options)
    parameters = pybamm.ParameterValues(model_config["parameter_set"])
    parameters.update(calibration["material_overrides"])
    parameters.update({"Initial temperature [K]": scenario.initial_temperature_C + 273.15,
                       "Ambient temperature [K]": scenario.ambient_temperature_C + 273.15,
                       "Total heat transfer coefficient [W.m-2.K-1]": scenario.heat_transfer_W_m2K})
    stop = pybamm.step.CustomTermination("research temperature stop", lambda v:
        scenario.temperature_stop_C + 273.15 - v["Volume-averaged cell temperature [K]"])
    charge = pybamm.step.current(-scenario.charge_current_A, duration=scenario.charge_horizon_s,
                                termination=[f"{scenario.voltage_limit_V} V", stop],
                                period=scenario.sample_period_s)
    rest = pybamm.step.rest(duration=scenario.rest_s, termination=[stop], period=scenario.sample_period_s)
    mesh = {key: int(value * scenario.mesh_scale) for key, value in model.default_var_pts.items()}
    solver = pybamm.IDAKLUSolver(rtol=scenario.rtol, atol=scenario.atol,
                               options={"num_threads": 1})
    simulation = pybamm.Simulation(model, parameter_values=parameters, var_pts=mesh, solver=solver,
                                   experiment=pybamm.Experiment([(charge, rest)]))
    solution = simulation.solve(initial_soc=scenario.initial_soc, calc_esoh=False)
    steps = solution.cycles[0].steps
    rows, phases = [], []
    charge_balance_error = 0.0
    for phase, step in zip(("charge", "rest"), steps):
        if isinstance(step, pybamm.EmptySolution):
            continue
        arrays = {key: np.asarray(step[name].entries, dtype=float).reshape(-1) for key, name in VARIABLES.items()}
        t = arrays["time_s"]
        if len(t) < 2 or np.any(np.diff(t) <= 0) or not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Invalid or nonfinite solution trace")
        residual = arrays["discharge_capacity_Ah"][-1] - arrays["discharge_capacity_Ah"][0] - np.trapezoid(arrays["current_A"], t) / 3600
        charge_balance_error = max(charge_balance_error, abs(float(residual)))
        if charge_balance_error > 1e-6:
            raise ValueError(f"Charge conservation residual {charge_balance_error} Ah")
        phases.append({"phase": phase, "termination": str(step.termination),
                       "start_s": float(t[0]), "end_s": float(t[-1])})
        for index in range(len(t)):
            row = {"scenario_id": scenario.scenario_id, "phase": phase,
                   **{key: float(values[index]) for key, values in arrays.items()}}
            row["temperature_C"] = row.pop("temperature_K") - 273.15
            rows.append(row)
    if not phases or phases[0]["phase"] != "charge":
        raise ValueError("Charge step infeasible at the requested initial state")
    if len(phases) < 2 and "research temperature stop" not in phases[0]["termination"]:
        raise ValueError("Solver returned partial experiment before required rest; no valid complete label")
    initial_li_Ah = float(solution["Total lithium in particles [mol]"].entries[0]) * pybamm.constants.F.value / 3600
    if abs(initial_li_Ah - calibration["initial_particle_lithium_Ah"]) > 1e-7:
        raise ValueError("SOC initialization changed calibrated lithium inventory")
    if abs(rows[0]["temperature_C"] - scenario.initial_temperature_C) > 1e-6:
        raise ValueError("Experiment overwrote the requested initial cell temperature")
    # Phase derivatives are kept separate to avoid a tiny step-boundary dt.
    rates = []
    for phase in ("charge", "rest"):
        selected = [r for r in rows if r["phase"] == phase]
        if len(selected) > 1:
            rates.extend(np.gradient([r["temperature_C"] for r in selected], [r["time_s"] for r in selected]))
    onset = persistent_onset([r["time_s"] for r in rows], [r["plating_net_current_A_m2"] for r in rows],
                             scenario.plating_threshold_A_m2, scenario.plating_persistence_s)
    complete = (len(phases) == 2 and phases[-1]["termination"] == "final time"
                and all("research temperature stop" not in p["termination"] for p in phases)
                and (phases[0]["termination"] == "final time" or "Voltage" in phases[0]["termination"]))
    summary = {**asdict(scenario), "status": "completed" if complete else "censored",
               "label_basis": "SPMeT_simulation", "plating_event": True if onset is not None else (False if complete else None),
               "plating_onset_s": onset, "plating_event_definition": "x_averaged_net_current_below_negative_threshold_for_persistence",
               "thermal_runaway_label": None, "fire_label": None, "physical_risk_validated": False,
               "thermal_runaway_validation": "unavailable_physics_and_abuse_test_data",
               "maximum_temperature_C": max(r["temperature_C"] for r in rows),
               "research_temperature_limit_reached": max(r["temperature_C"] for r in rows) >= scenario.temperature_stop_C - 1e-5,
               "maximum_temperature_rise_C_s": float(max(rates)),
               "initial_particle_lithium_Ah": initial_li_Ah,
               "charge_balance_error_Ah": charge_balance_error,
               "charge_end_s": phases[0]["end_s"], "total_duration_s": rows[-1]["time_s"],
               "charge_termination": phases[0]["termination"], "final_termination": phases[-1]["termination"],
               "sample_count": len(rows)}
    for quantity in ("metallic_lithium_mol_m3", "dead_lithium_mol_m3", "plating_inventory_Ah", "sei_loss_Ah"):
        values = [r[quantity] for r in rows]
        summary.update({f"{quantity}_start": values[0], f"{quantity}_peak": max(values),
                        f"{quantity}_end": values[-1], f"{quantity}_net_change": values[-1] - values[0],
                        f"{quantity}_peak_increase": max(values) - values[0]})
    return summary, rows, {"phases": phases, "options": options, "mesh": mesh,
                           "calibration": calibration, "initialization": "fresh_homogeneous_SOC_each_case"}


def run_case(payload):
    scenario_data, output_str, provenance_data, retry_failed = payload
    scenario = Scenario(**scenario_data)
    scenario.validate()
    output = Path(output_str) / scenario.scenario_id
    output.mkdir(parents=True, exist_ok=True)
    identity = content_hash({"scenario": scenario_data, "provenance": provenance_data})
    result_path, trace_path = output / "result.json", output / "trace.csv"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["identity_sha256"] != identity:
            raise ValueError(f"Existing case has different code/config: {scenario.scenario_id}")
        valid_trace = trace_path.exists() and previous.get("trace_sha256") == sha(trace_path)
        if previous["summary"]["status"] == "failed" and not retry_failed:
            return previous["summary"]
        if previous["summary"]["status"] != "failed" and valid_trace:
            return previous["summary"]
    lock_path = output / "running.lock"
    # A lock prevents two launchers writing the same case. After interruption the
    # operator may remove it only after confirming its recorded process stopped.
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "started": time.time()}))
    started = time.perf_counter()
    result = {"schema": SCHEMA, "identity_sha256": identity, "provenance": provenance_data}
    try:
        summary, rows, details = simulate(scenario)
        csv_atomic(trace_path, rows)
        result.update(summary=summary, details=details, trace_sha256=sha(trace_path))
    except Exception as error:
        result["summary"] = {**scenario_data, "status": "failed", "error": f"{type(error).__name__}: {error}",
                             "plating_event": None, "thermal_runaway_label": None, "fire_label": None,
                             "physical_risk_validated": False, "label_basis": "SPMeT_simulation"}
    finally:
        result["summary"]["wall_seconds"] = time.perf_counter() - started
        result["summary"]["identity_sha256"] = identity
        write_json(result_path, result)
        lock_path.unlink()
    return result["summary"]


def run_dataset(scenarios, output, workers=8, retry_failed=False):
    if workers not in range(1, 9):
        raise ValueError("workers must be 1 through 8")
    if not scenarios or len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("At least one scenario and unique IDs required")
    for scenario in scenarios:
        scenario.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = provenance()
    cases = [asdict(s) for s in scenarios]
    manifest = {"schema": SCHEMA, "provenance": source, "scenarios": cases,
                "design": "bounded_CC_to_4.2V_or_horizon_then_rest; fresh_cell_each_case",
                "threshold_scope": "numerical_screening_convention; no universal safety cutoff",
                "temperature_scope": "lumped_cell_average; core_hotspots_and_fire_unmodeled"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output manifest differs; use a new output directory")
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    payloads = [(s, str(output), source, retry_failed) for s in cases]
    if workers == 1:
        summaries = [run_case(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(run_case, payloads))
    csv_atomic(output / "scenario_summary.csv", summaries)
    if provenance() != source:
        raise RuntimeError("Source or dependencies changed during dataset execution; results require review")
    report = {"schema": SCHEMA, "scenario_count": len(summaries), "workers": workers,
              "wall_seconds": time.perf_counter() - started,
              "status_counts": {status: sum(s["status"] == status for s in summaries)
                                for status in ("completed", "censored", "failed")},
              "source": source, "summary_sha256": sha(output / "scenario_summary.csv")}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    scenarios = pilot_scenarios()
    if args.config:
        scenarios = [Scenario(**value) for value in json.loads(args.config.read_text(encoding="utf-8"))]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        scenarios = scenarios[:args.limit]
    report = run_dataset(scenarios, args.output, args.workers, args.retry_failed)
    print(json.dumps(report, indent=2))
    return 1 if report["status_counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
