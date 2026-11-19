"""Compare local and cloud artifacts without executing or changing frozen code.

All tolerances below are numerical reproduction tolerances. They are NOT
acceptance criteria for agreement with a real battery, lifetime, or safety.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time
import unittest


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "bms_pipeline_20261116/data/initial_low_rate_prefix.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def safe_member(root, value):
    path = (root / value.replace("\\", "/")).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Artifact path escapes result directory")
    return path


def numeric_comparison(left, right, columns, *, relative_tolerance=0.0):
    """Same ordered observations; units and absolute tolerances are explicit."""
    if len(left) != len(right) or not left:
        return {"passed": False, "reason": "different or empty row counts",
                "local_rows": len(left), "cloud_rows": len(right)}
    results = {}
    for name, tolerance in columns.items():
        maximum, failures = 0.0, 0
        for a, b in zip(left, right):
            x, y = float(a[name]), float(b[name])
            finite = math.isfinite(x) and math.isfinite(y)
            error = abs(x-y) if finite else None
            if not finite or error > tolerance + relative_tolerance * max(abs(x), abs(y)):
                failures += 1
            if error is not None:
                maximum = max(maximum, error)
        results[name] = {"absolute_tolerance": tolerance,
                         "relative_tolerance": relative_tolerance,
                         "maximum_absolute_difference": maximum,
                         "failed_rows": failures, "passed": failures == 0}
    return {"passed": all(x["passed"] for x in results.values()), "rows": len(left), "columns": results}


def decision_key(row):
    return (row["cell_id"], row["session_id"], Decimal(row["issue_time_s"]), Decimal(row["target_time_s"]))


def selected_d(rows):
    return [r for r in rows if r["cell_id"].endswith("_D")
            and r["selected_by_model"].lower() in ("true", "1")]


def compare(local, cloud, source, voltage_tolerance_mV=0.01):
    began = time.perf_counter()
    local, cloud, source = Path(local).resolve(), Path(cloud).resolve(), Path(source).resolve()
    if not math.isfinite(voltage_tolerance_mV) or voltage_tolerance_mV <= 0:
        raise ValueError("Positive finite voltage tolerance required")
    voltage_V = voltage_tolerance_mV / 1000.0
    outputs = []
    for root in (local, cloud):
        main = read_json(root / "pipeline_report.json")
        manifest = read_json(root / "dataset_manifest.json")
        learning = read_json(root / "machine_learning/training_report.json")
        twin = read_json(root / "twin/twin_report.json")
        outputs.append(dict(root=root, main=main, manifest=manifest, learning=learning, twin=twin,
                            verification=read_json(root / "verification.json")))
    a, b = outputs
    checks, details = {}, {}
    original_header, original_rows = read_csv(source)
    source_sha = sha(source)
    checks["both_pipelines_completed"] = all(x["main"]["status"] == "complete" for x in outputs)
    checks["both_integration_verifications_passed"] = all(x["verification"]["all_checks_passed"] is True for x in outputs)
    checks["source_input_identical"] = all(x["main"]["input_sha256"] == source_sha and
        x["learning"]["provenance"]["input"]["sha256"] == source_sha for x in outputs)
    checks["frozen_pipeline_code_hashes_identical"] = a["main"]["code_sha256"] == b["main"]["code_sha256"]
    identity_keys = ("input_sha256", "preparation_sha256", "model_sha256", "calibration_sha256", "runner_sha256")
    checks["physical_input_config_code_hashes_identical"] = all(
        a["twin"]["identity"][k] == b["twin"]["identity"][k] for k in identity_keys)
    settings = ("cell_id", "session_id", "mesh", "rtol", "atol")
    checks["physical_settings_identical"] = all(a["twin"]["identity"][k] == b["twin"]["identity"][k] for k in settings)
    checks["selector_policy_identical"] = a["learning"]["policy"] == b["learning"]["policy"]
    checks["same_pybamm_version"] = a["twin"]["runtime"]["pybamm_version"] == b["twin"]["runtime"]["pybamm_version"]
    for name in ("before", "after"):
        paths = [safe_member(x["root"], x["manifest"][name]["file"]) for x in outputs]
        frames = [read_csv(path) for path in paths]
        checks[name + "_CSV_bytes_identical"] = paths[0].read_bytes() == paths[1].read_bytes()
        checks[name + "_CSV_values_identical"] = frames[0] == frames[1]
        checks[name + "_manifest_hashes_match_files"] = all(
            x["manifest"][name]["sha256"] == sha(path) for x, path in zip(outputs, paths))
        checks[name + "_manifest_counts_match_files"] = all(
            x["manifest"][name]["rows"] == len(frame[1]) for x, frame in zip(outputs, frames))
        details[name] = {"local_rows": len(frames[0][1]), "cloud_rows": len(frames[1][1]),
                         "local_sha256": sha(paths[0]), "cloud_sha256": sha(paths[1])}
        if name == "before":
            checks["both_before_values_equal_original_source"] = all(
                frame == (original_header, original_rows) for frame in frames)
    decisions = []
    for x in outputs:
        path = x["root"] / "machine_learning/selection_decisions.csv"
        _, rows = read_csv(path)
        rows.sort(key=decision_key)
        decisions.append(rows)
        bound = sha(path)
        checks[("local" if x is a else "cloud") + "_decisions_bound_to_twin_and_model"] = (
            x["twin"]["identity"]["decisions_sha256"] == bound ==
            x["learning"]["artifacts_sha256"]["selection_decisions.csv"] ==
            x["manifest"]["selection_decisions_sha256"])
        copied = safe_member(x["root"], x["manifest"]["after"]["file"]).parent / "selection_decisions.csv"
        checks[("local" if x is a else "cloud") + "_decision_sidecar_identical"] = copied.read_bytes() == path.read_bytes()
    keys = [[decision_key(row) for row in rows] for rows in decisions]
    checks["all_decision_times_identical_and_unique"] = keys[0] == keys[1] and all(len(k) == len(set(k)) for k in keys)
    fields = ("split", "residual_exceeded", "uncertainty_exceeded", "support_exceeded",
              "selected_by_model", "candidate_review_reason", "selection_reason", "ml_adopted")
    checks["all_selection_decisions_identical"] = len(decisions[0]) == len(decisions[1]) and all(
        all(x[k] == y[k] for k in fields) and Decimal(x["decision_cutoff_s"]) == Decimal(y["decision_cutoff_s"])
        for x, y in zip(*decisions))
    chosen = [selected_d(rows) for rows in decisions]
    checks["same_58_D_selected_windows"] = len(chosen[0]) == len(chosen[1]) == 58 and [decision_key(r) for r in chosen[0]] == [decision_key(r) for r in chosen[1]]
    details["decision_counts"] = {"local_total": len(decisions[0]), "cloud_total": len(decisions[1]),
                                 "local_D_selected": len(chosen[0]), "cloud_D_selected": len(chosen[1])}
    details["voltage_prediction"] = numeric_comparison(*decisions,
        {"predicted_voltage_V": voltage_V, "observed_voltage_V": 0.0,
         "residual_mV": voltage_tolerance_mV, "predictive_std_mV": voltage_tolerance_mV})
    checks["voltage_predictions_reproduced"] = details["voltage_prediction"]["passed"]
    traces = [read_csv(x["root"] / "twin/observed_replay.csv") for x in outputs]
    trace_keys = lambda rows: [(r["cell_id"], r["session_id"], r["source_row"], Decimal(r["source_time_s"])) for r in rows]
    checks["physical_trace_observations_aligned"] = traces[0][0] == traces[1][0] and trace_keys(traces[0][1]) == trace_keys(traces[1][1])
    tolerances = {"model_voltage_V": voltage_V, "voltage_error_mV": voltage_tolerance_mV,
                  "model_current_A": 1e-8, "model_discharge_capacity_Ah": 1e-7,
                  "model_SEI_lithium_loss_Ah": 1e-8, "model_plating_inventory_Ah": 1e-7,
                  "model_dead_lithium_mol_m3": 1e-5, "model_particle_lithium_mol": 1e-8,
                  "model_model_temperature_K": 1e-6, "observed_voltage_V": 0.0,
                  "observed_surface_temperature_C": 0.0, "elapsed_s": 1e-8}
    details["physical_replay"] = numeric_comparison(traces[0][1], traces[1][1], tolerances)
    checks["physical_replay_reproduced"] = details["physical_replay"]["passed"]
    windows = [read_csv(x["root"] / "twin/selected_window_results.csv")[1] for x in outputs]
    window_key = lambda r: (r["cell_id"], Decimal(r["issue_time_s"]), Decimal(r["target_time_s"]))
    for rows in windows:
        rows.sort(key=window_key)
    checks["selected_physical_windows_identical"] = [window_key(r) for r in windows[0]] == [window_key(r) for r in windows[1]]
    details["selected_physical_results"] = numeric_comparison(*windows,
        {"interval_voltage_RMSE_mV": voltage_tolerance_mV, "SEI_lithium_loss_Ah_increment": 1e-8,
         "plating_inventory_Ah_increment": 1e-7, "dead_lithium_mol_m3_increment": 1e-5,
         "compared_rows": 0.0, "decision_cutoff_s": 0.0})
    checks["selected_physical_results_reproduced"] = details["selected_physical_results"]["passed"]
    branches = [sorted(x["twin"]["future_candidates"], key=lambda r: r["requested_charging_current_A"]) for x in outputs]
    checks["future_branch_protocols_and_endpoints_identical"] = len(branches[0]) == len(branches[1]) == 2 and all(
        all(x[k] == y[k] for k in ("status", "termination", "requested_charging_current_A", "requested_duration_s"))
        for x, y in zip(*branches))
    details["future_projection"] = numeric_comparison(*branches,
        {"duration_s": 1e-3, "end_voltage_V": voltage_V, "maximum_voltage_V": voltage_V,
         "minimum_voltage_V": voltage_V, "SEI_lithium_loss_Ah_increment": 1e-8,
         "plating_inventory_Ah_increment": 1e-7, "dead_lithium_mol_m3_increment": 1e-5,
         "discharge_capacity_Ah_increment": 1e-7, "charge_residual_Ah": 1e-7})
    checks["future_projection_reproduced"] = details["future_projection"]["passed"]
    checks["no_physical_lifetime_or_savings_promotion"] = all(
        x["main"]["lifetime_prediction_validated"] is False and
        x["twin"]["physical_accuracy"]["physical_model_validation_passed"] is False and
        x["twin"]["compute_savings_demonstrated"] is False and
        x["verification"]["physical_accuracy_threshold_passed"] is False for x in outputs)
    return {"schema": "bms-local-cloud-reproduction-v1", "status": "passed" if all(checks.values()) else "failed",
            "all_checks_passed": all(checks.values()), "checks": checks, "details": details,
            "local_directory": str(local), "cloud_directory": str(cloud), "source_input": str(source),
            "source_sha256": source_sha, "comparator_sha256": sha(__file__),
            "voltage_reproduction_tolerance_mV": voltage_tolerance_mV,
            "tolerance_scope": "Numerical reproduction only. Not a real-battery accuracy acceptance threshold, lifetime validation, causal validation, or safety certification.",
            "physical_accuracy_validated_by_this_comparison": False,
            "model_hash_policy": "Fitted model/checkpoint hashes may differ by floating-point and serialization details; predictions and physical outputs are compared numerically. Frozen source/config hashes must match.",
            "fitted_model_hashes": {"local": a["learning"]["model_sha256"], "cloud": b["learning"]["model_sha256"]},
            "runtimes": {"local": a["twin"]["runtime"], "cloud": b["twin"]["runtime"]},
            "self_comparison_only": local == cloud,
            "elapsed_wall_seconds": time.perf_counter()-began}


class ComparatorTests(unittest.TestCase):
    def test_voltage_millivolt_conversion_boundary(self):
        self.assertTrue(numeric_comparison([{"v": 0}], [{"v": .000009}], {"v": .01/1000})["passed"])
        self.assertFalse(numeric_comparison([{"v": 0}], [{"v": .000011}], {"v": .01/1000})["passed"])

    def test_missing_or_nonfinite_rows_fail(self):
        self.assertFalse(numeric_comparison([], [], {"v": 1})["passed"])
        self.assertFalse(numeric_comparison([{"v": 1}], [{"v": float("nan")}], {"v": 1})["passed"])

    def test_decision_times_preserve_decimal_identity(self):
        row = {"cell_id": "D", "session_id": "s", "issue_time_s": "17282.27682086406", "target_time_s": "17292.27682133904"}
        changed = dict(row, target_time_s="17292.27682133905")
        self.assertNotEqual(decision_key(row), decision_key(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--cloud", type=Path)
    parser.add_argument("--source-input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--voltage-tolerance-mV", type=float, default=.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparatorTests))
        return 0 if outcome.wasSuccessful() else 2
    if not all((args.local, args.cloud, args.output)):
        parser.error("--local, --cloud and --output are required for a comparison")
    try:
        result = compare(args.local, args.cloud, args.source_input, args.voltage_tolerance_mV)
    except Exception as error:
        result = {"status": "comparison_error", "all_checks_passed": False,
                  "error": f"{type(error).__name__}: {error}", "physical_accuracy_validated_by_this_comparison": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "receipt": str(args.output.resolve()),
                      "failed_checks": [k for k, v in result.get("checks", {}).items() if not v]}, ensure_ascii=True))
    return 0 if result["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
