"""Exactly 1000 distinct bounded scenarios, with simulation screening labels.

The two flags and their count-based priority are an explicit research policy.
They are not measured fire outcomes, physical danger levels, or safety labels.
Operational SPMeT physics remain in the unchanged risk_scenarios module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import time

from simulation import risk_scenarios as physics
from simulation.rebuilt_lifetime import sha, write_json


SCHEMA = "bounded-screening-dataset1000-v1"
AXES = {
    "ambient_temperature_C": [0., 10., 25., 35., 40.],
    "charge_current_A": [.5, 1., 1.5, 2., 2.5],
    "initial_soc": [.2, .3, .4, .5, .6],
    "heat_transfer_W_m2K": [round(5 + 5*i/7, 9) for i in range(8)],
}
FEATURES = list(AXES)
INDICATORS = ("none", "plating", "elevated_temperature", "both")
PRIORITIES = ("low", "medium", "high")
POLICY = {
    "schema": SCHEMA, "requested_scenarios": 1000, "axes": AXES,
    "initial_temperature": "equals ambient temperature in each scenario",
    "initial_state": "fresh homogeneous SOC; no prior degradation history; independent reset for every case",
    "domain": "Within earlier pilot bounds; interpolated conditions have not all been individually validated",
    "features": FEATURES,
    "plating_flag": "Existing model net plating current < -0.001 A/m2 for at least 5 sampled seconds",
    "temperature_flag": "Maximum sampled lumped mean cell temperature >= 40.0 C, no persistence requirement",
    "temperature_threshold_C": 40.0,
    "temperature_flag_limits": "Initial 40 C scenarios can satisfy this flag at t=0. It is not heat-generation onset, a core hotspot, runaway onset or a universal safety threshold.",
    "indicator_type": {"none": "neither defined flag observed", "plating": "plating only",
                       "elevated_temperature": "temperature only", "both": "both defined flags"},
    "screening_priority": {"low": "0 flags", "medium": "1 flag", "high": "2 flags"},
    "priority_scope": "Count-based research triage policy only; not a physical hazard ordering. Low/none does not mean safe.",
    "incomplete_policy": "Failed, censored and pending cases have no indicator_type or priority target; exclude from supervised targets",
    "label_basis": "SPMeT_simulation_policy_proxy", "physical_risk_validated": False,
    "thermal_runaway_label": None, "fire_label": None,
    "split": {"method": "sha256-ranked SOC/temperature/current groups; all 8 cooling settings stay together",
              "seed": "risk1000-fixed-split-v1", "train_groups": 87, "validation_groups": 19, "test_groups": 19,
              "requested_counts": {"train": 696, "validation": 152, "test": 152},
              "limits": "Same-model interpolation test; no empirical or new-cell-family generalization claim. No post-label balancing or fold search."},
}


def scenarios():
    values = itertools.product(*AXES.values())
    result = [physics.Scenario(f"risk1000_{i:04d}", soc, temp, temp, current, cooling)
              for i, (temp, current, soc, cooling) in enumerate(values, 1)]
    if len(result) != 1000 or len({physical_key(s) for s in result}) != 1000:
        raise AssertionError("Design must contain exactly 1000 distinct physical conditions")
    for scenario in result:
        scenario.validate()
    return result


def physical_key(scenario):
    return tuple(getattr(scenario, name) for name in FEATURES)


def group_key(scenario):
    return json.dumps([scenario.ambient_temperature_C, scenario.charge_current_A,
                       scenario.initial_soc], separators=(",", ":"))


def split_assignments(design):
    groups = sorted({group_key(s) for s in design}, key=lambda group:
                    physics.content_hash({"seed": POLICY["split"]["seed"], "group": group}))
    if len(groups) != 125:
        raise ValueError("Expected 125 distinct temperature/current/SOC groups")
    mapping = {group: "train" if i < 87 else "validation" if i < 106 else "test"
               for i, group in enumerate(groups)}
    return {s.scenario_id: {"dataset_split": mapping[group_key(s)], "protocol_group": group_key(s)}
            for s in design}


def classify(summary):
    """Only complete model runs receive targets; absent targets are never low."""
    result = {"training_eligible": False, "indicator_type": None, "screening_priority": None,
              "screening_flag_count": None, "plating_flag": None, "elevated_temperature_flag": None,
              "classification_basis": "SPMeT_simulation_policy_proxy", "physical_risk_validated": False,
              "thermal_runaway_label": None, "fire_label": None}
    if summary.get("status") not in {"completed", "censored", "failed", "pending"}:
        raise ValueError("Unknown simulation status")
    if summary["status"] != "completed":
        return result
    plating = summary.get("plating_event")
    temperature = summary.get("maximum_temperature_C")
    if type(plating) is not bool or isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("Completed case requires a finite maximum temperature and explicit boolean plating event")
    if summary.get("physical_risk_validated", False) or summary.get("thermal_runaway_label") is not None or summary.get("fire_label") is not None:
        raise ValueError("This model cannot supply physical fire/runaway labels")
    elevated = temperature >= POLICY["temperature_threshold_C"]
    count = int(plating) + int(elevated)
    indicator = "both" if count == 2 else "plating" if plating else "elevated_temperature" if elevated else "none"
    result.update(training_eligible=True, indicator_type=indicator, screening_priority=PRIORITIES[count],
                  screening_flag_count=count, plating_flag=plating, elevated_temperature_flag=elevated)
    return result


def execution_provenance(design):
    return {**physics.provenance(), "dataset_generator": {
        "schema": SCHEMA, "module_sha256": sha(Path(__file__)),
        "policy_sha256": physics.content_hash(POLICY),
        "scenario_definitions_sha256": physics.content_hash([asdict(s) for s in design])}}


def ensure_json(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Existing {path.name} differs; use a new output directory")
    write_json(path, value)


def write_design(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = scenarios()
    assignments = split_assignments(design)
    ensure_json(output / "design_plan.json", POLICY)
    ensure_json(output / "design_scenarios.json", [asdict(s) for s in design])
    physics.csv_atomic(output / "features.csv", [
        {"scenario_id": s.scenario_id, **assignments[s.scenario_id],
         **{key: getattr(s, key) for key in FEATURES}} for s in design])
    return design, assignments


def status_counts(summaries, requested=1000):
    counts = Counter(row["status"] for row in summaries)
    return {**{name: counts[name] for name in ("completed", "censored", "failed")},
            "pending": requested - sum(counts[name] for name in ("completed", "censored", "failed"))}


def save_progress(output, summaries, workers, started, state="running", error=None):
    counts = status_counts(summaries)
    finished = 1000 - counts["pending"]
    elapsed = time.perf_counter() - started
    data = {"schema": SCHEMA, "state": state, "updated_UTC": datetime.now(timezone.utc).isoformat(),
            "requested_scenarios": 1000, "finished_scenarios": finished,
            "workers": workers, "status_counts": counts, "launcher_elapsed_seconds": elapsed,
            "rough_remaining_runtime_seconds": elapsed * counts["pending"] / finished if finished else None,
            "eta_scope": "Linear estimate from current launcher only; cached resumes and case variability can invalidate it",
            "all_1000_completed": counts["completed"] == 1000}
    if error:
        data["error"] = error
    write_json(output / "progress.json", data)
    if summaries:
        physics.csv_atomic(output / "partial_summary.csv", sorted(summaries, key=lambda row: row["scenario_id"]))


def collect_results(design, output, source, skip_locked=False):
    """Read atomic case records and verify identity, scenario and trace digest."""
    summaries = []
    for scenario in design:
        path = output / scenario.scenario_id / "result.json"
        locked = path.with_name("running.lock").exists()
        if not path.exists() or (skip_locked and locked):
            summaries.append({**asdict(scenario), "status": "pending", "label_basis": "SPMeT_simulation",
                              "plating_event": None, "physical_risk_validated": False,
                              "thermal_runaway_label": None, "fire_label": None,
                              "pending_reason": "running_or_interrupted_case_lock" if locked else "no_atomic_result_record"})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = physics.content_hash({"scenario": asdict(scenario), "provenance": source})
        if result.get("identity_sha256") != expected_identity or result.get("provenance") != source:
            raise ValueError(f"Case source identity mismatch: {scenario.scenario_id}")
        summary = result["summary"]
        if any(summary.get(key) != value for key, value in asdict(scenario).items()):
            raise ValueError(f"Case physical conditions mismatch: {scenario.scenario_id}")
        if summary.get("status") not in {"completed", "censored", "failed"}:
            raise ValueError(f"Unknown recorded status: {scenario.scenario_id}")
        if summary["status"] != "failed":
            trace = path.with_name("trace.csv")
            if not trace.exists() or sha(trace) != result.get("trace_sha256"):
                raise ValueError(f"Trace missing or checksum mismatch: {scenario.scenario_id}")
            if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in summary.values()):
                raise ValueError(f"Nonfinite summary value: {scenario.scenario_id}")
        classify(summary)
        summaries.append(summary)
    return summaries


def finalize(output, design, assignments, source, workers, wall_seconds, finalization_mode):
    if execution_provenance(design) != source:
        raise RuntimeError("Source, policy, design or dependency changed; finalization refused")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["provenance"] != source or manifest["scenarios"] != [asdict(s) for s in design]
            or manifest.get("policy") != POLICY or manifest.get("features_sha256") != sha(output / "features.csv")):
        raise ValueError("Execution manifest changed")
    if json.loads((output / "design_plan.json").read_text(encoding="utf-8")) != POLICY:
        raise ValueError("Classification policy file changed")
    if json.loads((output / "design_scenarios.json").read_text(encoding="utf-8")) != [asdict(s) for s in design]:
        raise ValueError("Scenario design file changed")
    summaries = collect_results(design, output, source, skip_locked=finalization_mode == "existing_records_only")
    labels = [{"scenario_id": s["scenario_id"], "status": s["status"], **classify(s)} for s in summaries]
    combined = [{**s, **assignments[s["scenario_id"]], **classify(s)} for s in summaries]
    physics.csv_atomic(output / "scenario_summary.csv", combined)
    physics.csv_atomic(output / "labels.csv", labels)
    training_rows = [{"scenario_id": s["scenario_id"], "dataset_split": assignments[s["scenario_id"]]["dataset_split"],
                      **{key: s[key] for key in FEATURES},
                      **{key: label[key] for key in ("indicator_type", "screening_priority", "training_eligible")},
                      "status": s["status"]} for s, label in zip(summaries, labels)]
    physics.csv_atomic(output / "training_dataset.csv", training_rows)
    counts = status_counts(summaries)
    eligible = [row for row in labels if row["training_eligible"]]
    report = {"schema": SCHEMA, "scenario_count": 1000, "requested_scenarios": 1000,
              "unique_physical_conditions": 1000, "workers": workers, "wall_seconds": wall_seconds,
              "finalization_mode": finalization_mode, "status_counts": counts,
              "execution_complete": counts["pending"] == 0,
              "all_1000_completed": counts["completed"] == 1000,
              "locked_cases_excluded": sum(s.get("pending_reason") == "running_or_interrupted_case_lock" for s in summaries),
              "supervised_target_rows": len(eligible),
              "indicator_counts": {name: sum(row["indicator_type"] == name for row in eligible) for name in INDICATORS},
              "screening_priority_counts": {name: sum(row["screening_priority"] == name for row in eligible) for name in PRIORITIES},
              "requested_split_counts": dict(Counter(a["dataset_split"] for a in assignments.values())),
              "eligible_split_counts": dict(Counter(assignments[row["scenario_id"]]["dataset_split"] for row in eligible)),
              "source": source, "sources_unchanged_after_execution": True,
              "label_basis": POLICY["label_basis"], "physical_risk_validated": False,
              "thermal_runaway_prediction_validated": False, "fire_prediction_validated": False,
              "summary_sha256": sha(output / "scenario_summary.csv"),
              "features_sha256": sha(output / "features.csv"), "labels_sha256": sha(output / "labels.csv"),
              "training_dataset_sha256": sha(output / "training_dataset.csv"),
              "policy_sha256": source["dataset_generator"]["policy_sha256"],
              "warning": "Screening priority counts defined model flags. It is not measured physical danger or fire truth; low is not safe."}
    if execution_provenance(design) != source:
        raise RuntimeError("Source changed while exporting dataset")
    write_json(output / "dataset_report.json", report)
    write_json(output / "report.json", report)
    return report


def run(output, workers=8, retry_failed=False, finalize_only=False):
    if type(workers) is not int or workers not in range(1, 9):
        raise ValueError("workers must be an integer from 1 through 8")
    output = Path(output)
    design, assignments = write_design(output)
    source = execution_provenance(design)
    expected_manifest = {"schema": SCHEMA, "provenance": source,
                         "scenarios": [asdict(s) for s in design], "policy": POLICY,
                         "features_sha256": sha(output / "features.csv")}
    if finalize_only and not (output / "manifest.json").exists():
        raise ValueError("No execution manifest exists to finalize")
    ensure_json(output / "manifest.json", expected_manifest)
    if finalize_only:
        return finalize(output, design, assignments, source, workers, None, "existing_records_only")
    started = time.perf_counter()
    received = []
    save_progress(output, received, workers, started)
    payloads = [(asdict(s), str(output), source, retry_failed) for s in design]
    try:
        if workers == 1:
            for payload in payloads:
                received.append(physics.run_case(payload))
                if len(received) % 8 == 0:
                    save_progress(output, received, workers, started)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(physics.run_case, payload) for payload in payloads]
                for future in as_completed(futures):
                    received.append(future.result())
                    if len(received) % 8 == 0:
                        save_progress(output, received, workers, started)
        report = finalize(output, design, assignments, source, workers, time.perf_counter() - started, "after_execution")
        save_progress(output, received, workers, started,
                      "completed" if report["all_1000_completed"] else "finished_with_incomplete_targets")
        return report
    except Exception as error:
        save_progress(output, received, workers, started, "error", f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-failed", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--finalize-only", action="store_true")
    modes.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    if args.design_only:
        design, _ = write_design(args.output)
        print(json.dumps({"scenario_count": len(design), "design_only": True,
                          "policy_sha256": physics.content_hash(POLICY)}, indent=2))
        return 0
    report = run(args.output, args.workers, args.retry_failed, args.finalize_only)
    print(json.dumps(report, indent=2))
    return 0 if report["all_1000_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
