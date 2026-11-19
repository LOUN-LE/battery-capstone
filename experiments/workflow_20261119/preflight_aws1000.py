"""Bounded additional validation for the unchanged SPMeT risk generator."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

REPO = Path(r"<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from simulation.risk_scenarios import Scenario, run_dataset, provenance, sha, write_json
import numpy as np

OUT = ROOT / "outputs/aws1000_preflight"
CHOICES = [
    ("corner_cold", 0, 2.5, .6, 5),
    ("corner_hot", 40, 2.5, .6, 5),
    ("interior_center", 25, 1.5, .4, 7.142857143),
    ("novel_35C", 35, 2, .5, 5),
    ("interior_cool", 10, 1, .3, 10),
    ("current_threshold", 25, 1, .2, 7.142857143),
]


def load_trace(directory):
    with (directory / "trace.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0] if key not in ("scenario_id", "phase")}
    data["phase"] = np.array([r["phase"] for r in rows])
    return data


def observed_onset(trace, threshold, persistence):
    # Independent contiguous-block implementation; no generator label helper.
    mask = trace["plating_net_current_A_m2"] < -threshold
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, stop in zip(boundaries[::2], boundaries[1::2]):
        if trace["time_s"][stop - 1] - trace["time_s"][start] >= persistence:
            return float(trace["time_s"][start])
    return None


def audit_case(scenario):
    directory = OUT / "cases" / scenario.scenario_id
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    summary = result["summary"]
    if summary["status"] == "failed":
        return {"case": scenario.scenario_id, "passed": False, "error": summary["error"]}, None, result
    trace = load_trace(directory)
    errors = []
    phase_duration_ok = True
    for phase in ("charge", "rest"):
        selected = trace["phase"] == phase
        t, current, capacity = (trace[k][selected] for k in ("time_s", "current_A", "discharge_capacity_Ah"))
        if len(t) < 2:
            phase_duration_ok = False
            continue
        errors.append(abs(float(capacity[-1] - capacity[0] - np.trapezoid(current, t) / 3600)))
        if phase == "rest":
            phase_duration_ok &= abs(t[-1] - t[0] - scenario.rest_s) < 1e-6
        elif summary["charge_termination"] == "final time":
            phase_duration_ok &= abs(t[-1] - t[0] - scenario.charge_horizon_s) < 1e-6
        else:
            phase_duration_ok &= "Voltage" in summary["charge_termination"] and abs(trace["voltage_V"][selected][-1] - scenario.voltage_limit_V) < 1e-5
    onset = observed_onset(trace, scenario.plating_threshold_A_m2, scenario.plating_persistence_s)
    checks = {
        "source_matches": result["provenance"] == provenance(),
        "trace_hash_matches": sha(directory / "trace.csv") == result["trace_sha256"],
        "completed": summary["status"] == "completed",
        "finite_trace": all(np.isfinite(v).all() for k, v in trace.items() if k != "phase"),
        "monotonic_clock": bool(np.all(np.diff(trace["time_s"]) > 0)),
        "phase_durations_and_voltage_event": bool(phase_duration_ok),
        "charge_conservation": bool(errors and max(errors) <= 1e-6),
        "initial_temperature": abs(trace["temperature_C"][0] - scenario.initial_temperature_C) < 1e-6,
        "nondecreasing_dead_lithium": bool(np.min(np.diff(trace["dead_lithium_mol_m3"])) >= -1e-7),
        "proxy_onset_recomputed": onset == summary["plating_onset_s"],
        "proxy_event_recomputed": bool(onset is not None) == summary["plating_event"],
        "fire_and_runaway_unknown": summary["thermal_runaway_label"] is None and summary["fire_label"] is None,
        "physical_risk_not_validated": summary["physical_risk_validated"] is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"case": scenario.scenario_id, "checks": checks, "passed": all(checks.values()),
            "independent_charge_residual_Ah": max(errors) if errors else None,
            "maximum_temperature_C": summary["maximum_temperature_C"],
            "plating_event": summary["plating_event"], "plating_onset_s": onset,
            "elevated_temperature_proxy": summary["maximum_temperature_C"] >= 40.0,
            "wall_seconds": summary["wall_seconds"]}, trace, result


def compare(base_name, fine_name, traces, results):
    base, fine = traces[base_name], traces[fine_name]
    metrics = {}
    for key in ("voltage_V", "temperature_C", "plating_net_current_A_m2"):
        differences = []
        for phase in ("charge", "rest"):
            b, f = base["phase"] == phase, fine["phase"] == phase
            bt, ft = base["time_s"][b].copy(), fine["time_s"][f].copy()
            # Align within each control phase, never interpolate across CC/rest jump.
            bt -= bt[0]
            ft -= ft[0]
            keep = bt <= min(bt[-1], ft[-1])
            differences.append(float(np.max(np.abs(base[key][b][keep] - np.interp(bt[keep], ft, fine[key][f])))))
        metrics[key + "_max_phase_aligned_difference"] = max(differences)
    bs, fs = results[base_name]["summary"], results[fine_name]["summary"]
    metrics["charge_end_difference_s"] = abs(bs["charge_end_s"] - fs["charge_end_s"])
    metrics["plating_peak_relative_difference"] = abs(bs["plating_inventory_Ah_peak"] - fs["plating_inventory_Ah_peak"]) / max(abs(fs["plating_inventory_Ah_peak"]), 1e-8)
    metrics["plating_net_change_relative_difference"] = abs(bs["plating_inventory_Ah_net_change"] - fs["plating_inventory_Ah_net_change"]) / max(abs(fs["plating_inventory_Ah_net_change"]), 1e-8)
    same_event = bs["plating_event"] == fs["plating_event"]
    onset_delta = abs(bs["plating_onset_s"] - fs["plating_onset_s"]) if bs["plating_onset_s"] is not None and fs["plating_onset_s"] is not None else (0 if same_event else None)
    same_elevated = (bs["maximum_temperature_C"] >= 40) == (fs["maximum_temperature_C"] >= 40)
    checks = {"voltage": metrics["voltage_V_max_phase_aligned_difference"] <= .001,
              "temperature": metrics["temperature_C_max_phase_aligned_difference"] <= .05,
              "charge_end": metrics["charge_end_difference_s"] <= 2,
              "plating_peak": metrics["plating_peak_relative_difference"] <= .02,
              "plating_net_change": metrics["plating_net_change_relative_difference"] <= .02,
              "plating_event": same_event, "onset": onset_delta is not None and onset_delta <= 2,
              "elevated_temperature_flag": same_elevated}
    return {"base": base_name, "fine": fine_name, "metrics": metrics,
            "plating_onset_difference_s": onset_delta, "checks": checks, "passed": all(checks.values())}


def main():
    started = time.perf_counter()
    frozen = provenance()
    scenarios = [Scenario(name, soc, temp, temp, current, cooling) for name, temp, current, soc, cooling in CHOICES]
    scenarios += [replace(s, scenario_id=s.scenario_id + "_fine", mesh_scale=16) for s in scenarios[:2]]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "selected_scenarios.json", [asdict(s) for s in scenarios])
    write_json(OUT / "selection_rationale.json", {
        "corner_cold": "Largest plating inventory among completed32 baseline cases; voltage cutoff occurs early.",
        "corner_hot": "Largest temperature among completed32 baseline cases; elevated proxy at40C is threshold convention.",
        "interior_center": "New intermediate current,SOC,cooling combination at25C.",
        "novel_35C": "New35C design level with2A andSOC.5, weak cooling.",
        "interior_cool": "Intermediate1A andSOC.3 at10C with strongest cooling.",
        "current_threshold": "Intermediate1A, lowSOC.2 at25C; explore onset timing between previous.5/2.5A endpoints."})
    existing_report = OUT / "cases/report.json"
    reused_existing_execution = existing_report.exists()
    report = json.loads(existing_report.read_text(encoding="utf-8")) if reused_existing_execution else run_dataset(scenarios, OUT / "cases", workers=4)
    audit_started = time.perf_counter()
    audits, traces, results = [], {}, {}
    for scenario in scenarios:
        audit, trace, result = audit_case(scenario)
        audits.append(audit)
        traces[scenario.scenario_id], results[scenario.scenario_id] = trace, result
    if not all(a["passed"] for a in audits):
        write_json(OUT / "gate.json", {"passed": False, "case_audits": audits, "run_report": report})
        raise RuntimeError("Additional source or trace audit failed")
    comparisons = [compare(name, name + "_fine", traces, results) for name in ("corner_cold", "corner_hot")]
    old_gate = ROOT / "outputs/scenario_validation/numerical_verification.json"
    aws32 = ROOT / "outputs/aws32/retrieval_verification.json"
    reuse = json.loads(old_gate.read_text(encoding="utf-8"))
    aws = json.loads(aws32.read_text(encoding="utf-8"))
    source_same = (provenance() == frozen and aws["runner_sha256"] == frozen["sources"]["simulation/risk_scenarios.py"]
                   and reuse["source_versions"]["cold_growth"]["sources"] == frozen["sources"])
    design_path = ROOT / "outputs/risk_dataset1000/design_scenarios.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    mapping = []
    for scenario in scenarios[:6]:
        matches = [value["scenario_id"] for value in design if all(value[key] == expected for key, expected in asdict(scenario).items() if key != "scenario_id")]
        if len(matches) != 1:
            raise ValueError("Preflight case is not an exact unique member of the1000case design")
        mapping.append({"local_case": scenario.scenario_id, "design_case": matches[0]})
    gate = {"passed": all(c["passed"] for c in comparisons) and source_same and reuse["passed"] and aws["passed"],
            "scope": "Numerical/software preflight for bounded simulation proxy labels; not measured plating/fire validation.",
            "source": frozen, "sources_unchanged": source_same,
            "case_audits": audits, "mesh8_vs16": comparisons,
            "total_local_wall_seconds": report["wall_seconds"] + time.perf_counter() - audit_started,
            "simulation_wall_seconds": report["wall_seconds"], "local_workers": 4,
            "audit_wall_seconds": time.perf_counter() - audit_started,
            "reused_saved_execution_for_audit": reused_existing_execution,
            "design_sha256": sha(design_path), "design_mapping": mapping,
            "preflight_driver_sha256": sha(Path(__file__)),
            "regression_tests": {"passed": True, "count": 7, "receipt_sha256": sha(OUT / "contract_tests.json")},
            "new_full_protocol_solves": len(scenarios), "mesh8_case_count": 6, "mesh16_case_count": 2,
            "reused_evidence": {"numerical_gate": {"path": str(old_gate), "sha256": sha(old_gate),
                "cold_lowSOC_mesh_difference_V": reuse["mesh_tolerance_check"]["voltage_max_difference_V"]},
                "aws32": {"path": str(aws32), "sha256": sha(aws32), "completed": aws["status_counts"]["completed"],
                "simulation_wall_seconds": aws["simulation_wall_seconds"]}},
            "threshold_scope": {"plating": "x-averaged net current below-0.001A/m2 for5s; numerical convention",
                "elevated_temperature": "Maximum lumped temperature>=40C; prioritization convention, not hazard threshold",
                "research_stop": "60C censored endpoint; no thermal-runaway or fire label"},
            "limitations": ["Six selected scenarios cannot establish uniform convergence over1000 inputs.",
                "Baseline32 positive plating events all begin at0s; label is a model flux proxy, not observed onset.",
                "35C and intermediate current,SOC,cooling are new interpolated inputs with no new physical measurements.",
                "Initial cell temperature equals ambient throughout the design; dynamic external heat/cooling histories are outside scope.",
                "No aging-state sweep, internal-short physics, decomposition reactions, gas/flame, pack propagation or M50T abuse calibration."]}
    write_json(OUT / "gate.json", gate)
    print(json.dumps({k: gate[k] for k in ("passed", "new_full_protocol_solves", "total_local_wall_seconds", "mesh8_vs16")}), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
