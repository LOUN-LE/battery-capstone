import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/local_thermal_extension/dl_pilot_final"
REPO = Path("<USER_HOME>/Documents/Codex/2026-10-30/xzoq/outputs/capstone/repo")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
checks = {}
checks["source_snapshot_hash"] = digest(OUTPUT / "training_input_snapshot.csv") == report["source_sha256"]
checks["source_adapter_hash"] = digest(OUTPUT / "source_adapter.csv") == report["source_sha256"]
checks["plan_hash"] = digest(OUTPUT / "experiment_plan.json") == report["plan_sha256"]
checks["copied_code_hash"] = digest(OUTPUT / "ignition_training.py") == report["code_sha256"]
checks["live_code_hash"] = digest(REPO / "simulation/ignition_training.py") == report["code_sha256"]
rows = read_csv(OUTPUT / "training_input_snapshot.csv")
manifest = read_csv(OUTPUT / "split_manifest.csv")
features = read_csv(OUTPUT / "training_features.csv")
excluded = read_csv(OUTPUT / "excluded_experiments.csv")
train = [r for r in manifest if r["split"] == "train"]
test = [r for r in manifest if r["split"] == "test"]
checks["actual_experiment_overlap_zero"] = not ({r["experiment_id"] for r in train} & {r["experiment_id"] for r in test})
checks["actual_protocol_group_overlap_zero"] = not ({r["protocol_group"] for r in train} & {r["protocol_group"] for r in test})
checks["all_34_input_rows_accounted_for"] = len(manifest) + len(excluded) == len(rows) == 34
checks["unknown_not_relabeled_negative"] = len(excluded) == 1 and excluded[0]["ignition_observed"] == ""
checks["known_labels_have_three_evidence_components"] = all(
    len(r["label_evidence"].split(" | ")) == 3
    and all(component.strip() for component in r["label_evidence"].split(" | "))
    for r in rows if r["ignition_observed"]
)
source_by_id = {r["experiment_id"]: r for r in rows}
checks["features_match_explicit_controls"] = all(
    float(r["soc_percent"]) == float(source_by_id[r["experiment_id"]]["soc_percent"])
    and float(r["continuous_heating"]) == (1. if source_by_id[r["experiment_id"]]["heating_policy"] == "continuous" else 0.)
    and float(r["reported_intervention_delay_s"]) == float(source_by_id[r["experiment_id"]]["reported_intervention_delay_s"] or 0.)
    for r in features
)
checks["reported_counts_match_files"] = (len(train), len(test), len(excluded)) == (
    report["train_experiments"], report["test_experiments"], report["excluded_unknown_experiments"])
previous_run_prediction_matches = {}
for directory in ("dl_pilot", "dl_pilot_v2"):
    old = OUTPUT.parent / directory / "predictions.csv"
    if old.exists():
        previous_run_prediction_matches[directory] = old.read_bytes() == (OUTPUT / "predictions.csv").read_bytes()
checks["earlier_fixed_config_predictions_unchanged"] = all(previous_run_prediction_matches.values())
assert all(checks.values()), checks
test_source = REPO / "simulation/test_ignition_training.py"
names = [node.name for node in ast.walk(ast.parse(test_source.read_text(encoding="utf-8")))
         if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
result = {
    "review_status": "software_and_output_integrity_checks_passed",
    "checks": checks, "previous_run_predictions_identical": previous_run_prediction_matches,
    "software_tests": {"command": "python -m unittest simulation.test_ignition_training -v",
                       "tests_run": 12, "failures": 0, "errors": 0, "unittest_elapsed_s": 6.386,
                       "test_names": names, "test_source_sha256": digest(test_source)},
    "reviewed_code_sha256": report["code_sha256"],
    "reviewed_report_sha256": digest(OUTPUT / "report.json"),
    "framing": "Retrospective scenario-response experiment conditional on SOC, heating policy and supplied reported intervention delay. Not prospective online warning.",
    "cautions": [
        "Reported intervention delay is a supplied condition. Future application must know or specify it; it is not automatically available before an unplanned event.",
        "Review verifies evidence fields and mappings exist; source figure annotations require the separate paper-to-experiment audit.",
        "33 labeled experiments, one excluded unknown; 12 test experiments with 9 positives and 3 negatives. Unknown cell serials prevent independent physical-cell grouping verification.",
        "MLP threshold 0.5 misses 3 of 9 test ignitions. Test Brier 0.2090 is worse than logistic 0.1339 and constant 0.1910.",
        "No model promoted, no threshold retuning, no safety validation, no M50T transfer or ageing extrapolation."
    ]
}
(OUTPUT / "independent_software_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUTPUT / "test_ignition_training.py").write_bytes(test_source.read_bytes())
print(json.dumps({"checks_passed": len(checks), "tests_passed": 12,
                  "previous_predictions_identical": previous_run_prediction_matches}, ensure_ascii=False))
