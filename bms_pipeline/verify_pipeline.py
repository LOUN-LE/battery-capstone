"""Independent artifact/identity checks for the measured-input pipeline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def verify(output, source=None):
    output = Path(output)
    source = Path(source or Path(__file__).parent / "data/initial_low_rate_prefix.csv")
    main = load(output / "pipeline_report.json")
    manifest = load(output / "dataset_manifest.json")
    twin = load(output / "twin/twin_report.json")
    learning = load(output / "machine_learning/training_report.json")
    before = csv_rows(output / manifest["before"]["file"])
    after = csv_rows(output / manifest["after"]["file"])
    original = csv_rows(source)
    key = lambda r: (r["cell_id"], r["session_id"], r["source_row"])
    indexed = {key(r): r for r in original}
    decisions_path = output / "machine_learning/selection_decisions.csv"
    decisions = csv_rows(decisions_path)
    d = learning["cell_summaries"]["D"]
    selected = [r for r in decisions if r["cell_id"] == d["cell_id"]
                and r["selected_by_model"].lower() in ("true", "1")]
    cutoff = manifest["after"]["decision_cutoff_s"]
    branches = twin["future_candidates"]
    checks = {
        "pipeline_completed": main["status"] == "complete",
        "source_hash_unchanged": sha(source) == main["input_sha256"],
        "before_exactly_matches_source": before == original,
        "source_ids_unique": len(indexed) == len(original),
        "after_exact_subset": all(indexed.get(key(r)) == r for r in after),
        "after_only_evaluation_cell": bool(after) and all(r["cell_id"] == d["cell_id"] for r in after),
        "after_within_judgment_cutoff": all(float(r["time_s"]) <= cutoff + 1e-8 for r in after),
        "selected_decisions_match_twin": len(selected) == twin["selected_windows_evaluated"],
        "replayed_exact_export": twin["identity"]["input_sha256"] == sha(output / manifest["after"]["file"]),
        "decision_provenance_bound": twin["identity"]["decisions_sha256"] == sha(decisions_path),
        "replay_row_coverage": twin["observed_replay"]["rows"] == len(after),
        "numerical_state_and_charge_checks": twin["numerical_validation_passed"] is True,
        "checkpoint_restore_exact": twin["checkpoints"]["midpoint_restoration_exact"] and twin["checkpoints"]["terminal_restoration_exact"],
        "future_branches_share_same_input": len(branches) == 2 and len({r["input_state_fingerprint"] for r in branches}) == 1,
        "branches_preserve_state": all(r["input_state_unchanged"] and r["status"] == "completed" for r in branches),
        "no_D_training_or_threshold_fitting": learning["D_was_used_to_fit_or_calibrate"] is False,
        "no_future_sensor_features": learning["future_sensors_used_as_features"] is False,
        "no_lifetime_or_causal_claim": main["lifetime_prediction_validated"] is False and twin["degradation_cause_identified"] is False,
        "source_and_configuration_guards": twin["source_hashes_unchanged"] is True,
        "no_compute_saving_claim": twin["compute_savings_demonstrated"] is False,
    }
    result = {"checks": checks, "all_checks_passed": all(checks.values()),
              "ready_for_cloud_integration_reproduction": all(checks.values()),
              "ready_for_lifetime_risk_or_cause_claim": False,
              "before_rows": len(before), "after_rows": len(after), "selected_windows": len(selected),
              "voltage_forecast_rmse_mV": d["metrics"]["voltage_rmse_mV"],
              "physical_replay_rmse_mV": twin["observed_replay"]["voltage_RMSE_mV"],
              "physical_accuracy_threshold_passed": False,
              "scope": "Functional, identity, data-leakage, state-transfer and numerical checks only"}
    (output / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["all_checks_passed"]:
        raise AssertionError([k for k,v in checks.items() if not v])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.output, args.source), ensure_ascii=False))
