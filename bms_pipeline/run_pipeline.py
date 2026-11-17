"""Measured cell records -> frozen voltage-review model -> physical replay.

This is a reproducible integration experiment, not a lifetime-risk classifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

HERE = Path(__file__).resolve().parent
CANONICAL = ["cell_id", "session_id", "time_s", "voltage_V", "current_A",
             "surface_temperature_C", "source_row", "phase"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=CANONICAL):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_datasets(input_csv, selection, output):
    """Export observations unchanged; selection reasons remain in a sidecar."""
    fields, rows = read_csv(input_csv)
    if fields != CANONICAL:
        raise ValueError("Expected source-preserving canonical cell observations")
    first, second = output / "01_before_machine_learning", output / "02_after_machine_learning_for_twin"
    first.mkdir()
    second.mkdir()
    before = first / "01_머신러닝_전_입력.csv"
    after = second / "02_머신러닝_후_디지털트윈_입력.csv"
    write_csv(before, rows)
    summary = selection["cell_summaries"]["D"]
    cid = summary["cell_id"]
    matching = [r for r in rows if r["cell_id"] == cid]
    model_selected = summary["selected"]
    cutoff = summary["decision_cutoff_s"] if model_selected else max(float(r["time_s"]) for r in matching)
    selected = [r for r in matching if float(r["time_s"]) <= cutoff + 1e-8]
    if not selected:
        raise ValueError("No observation prefix available for the selected evaluation cell")
    write_csv(after, selected)
    shutil.copyfile(selection["decisions_path"], second / "selection_decisions.csv")
    manifest = {
        "before": {"file": str(before.relative_to(output)), "rows": len(rows), "sha256": sha(before),
                   "roles": {"C": "training and threshold calibration", "D": "frozen evaluation"}},
        "after": {"file": str(after.relative_to(output)), "rows": len(selected), "sha256": sha(after),
                  "cell_id": cid, "decision_cutoff_s": cutoff,
                  "selected_by_model": bool(model_selected),
                  "inclusion_reason": "model_review_selection" if model_selected else "explicit_integration_control",
                  "prefix_preserved_for_state_initialization": True,
                  "voltage_and_temperature_are_observations_not_replaced_predictions": True},
        "selection_decisions_sha256": sha(second / "selection_decisions.csv"),
        "interpretation": "Row reduction excludes training cell C; it is not evidence of reduced digital-twin calls.",
        "additional_twin_requirements": ["data/cell_D_preparation.csv", "configs/frozen_model.json", "configs/cell_calibration.json"],
        "risk_classification_validated": False, "lifetime_prediction_validated": False,
    }
    # The second dataset must be a literal subset, including unchanged decimals.
    source = {(r["cell_id"], r["session_id"], r["source_row"]): r for r in rows}
    if len(source) != len(rows):
        raise ValueError("Observation identity is not unique")
    _, persisted = read_csv(after)
    if any(source[(r["cell_id"], r["session_id"], r["source_row"])] != r for r in persisted):
        raise AssertionError("Selected input changed source values")
    manifest["checks"] = {"unique_observation_ids": True, "after_is_exact_subset": True,
                          "after_excludes_training_cell": all(r["cell_id"] == cid for r in persisted),
                          "after_has_no_future_rows": all(float(r["time_s"]) <= cutoff + 1e-8 for r in persisted)}
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def run(input_csv, output, *, skip_twin=False):
    from ml_selector import run_selector
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_csv = Path(input_csv).resolve()
    original_hash = sha(input_csv)
    code_hashes = {name: sha(HERE / name) for name in
                   ("run_pipeline.py", "data_adapter.py", "ml_selector.py", "twin_runner.py")}
    started = time.perf_counter()
    report = {"status": "running", "scope": "measured-voltage review and physical integration validation",
              "input_sha256": original_hash, "code_sha256": code_hashes, "python_version": platform.python_version(),
              "risk_classification_validated": False, "lifetime_prediction_validated": False}
    try:
        stage = time.perf_counter()
        selection = run_selector(input_csv, output / "machine_learning")
        report["machine_learning_wall_s"] = time.perf_counter() - stage
        report["machine_learning"] = selection
        report["datasets"] = export_datasets(input_csv, selection, output)
        if not skip_twin:
            from twin_runner import run_twin
            stage = time.perf_counter()
            report["twin"] = run_twin(
                output / report["datasets"]["after"]["file"], selection["decisions_path"], output / "twin",
                configdir=HERE / "configs", preparation_csv=HERE / "data/cell_D_preparation.csv")
            report["twin_wall_s"] = time.perf_counter() - stage
        report["input_unchanged"] = sha(input_csv) == original_hash
        if not report["input_unchanged"]:
            raise AssertionError("Input changed during execution")
        if code_hashes != {name: sha(HERE / name) for name in code_hashes}:
            raise AssertionError("Pipeline implementation changed during execution")
        report["status"] = "selection_only" if skip_twin else "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["total_wall_s"] = time.perf_counter() - started
        write_json(output / "pipeline_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/initial_low_rate_prefix.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-twin", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, skip_twin=args.skip_twin)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "total_wall_s": result["total_wall_s"]}, ensure_ascii=False))
