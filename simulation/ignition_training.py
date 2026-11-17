"""Small, reproducible local NIST ignition learning pilot, not a safety model.

The input is an explicit adapter table of published experiment outcomes. No
temperature trace, event time, or simulation-generated fire label is consumed.
Reported intervention timing is a CONDITION of the retrospective scenario,
not an early-warning measurement known before an unplanned intervention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


FEATURES = ["soc_percent", "continuous_heating", "reported_intervention_delay_s"]
PLAN = {
    "experiment": "nist_nmc21700_ignition_local_pilot_v1",
    "task": "Retrospective outcome conditional on specified heating protocol and reported intervention timing",
    "target": "Paper-confirmed ignition observed vs paper-confirmed no ignition during the experiment's observation period",
    "features": FEATURES,
    "intervention_timing_semantics": "Recorded experiment condition; a future use must specify this control in advance. Not a live early-warning model.",
    "continuous_delay_encoding": "0 is a structural placeholder paired with continuous_heating=1, not an observed immediate stop",
    "excluded_inputs": ["experiment identifier", "full temperature trace", "maximum temperature", "ignition time", "SVB time", "observation duration", "paper label evidence"],
    "unknown_policy": "exclude unknown outcomes, retain source row in excluded_experiments.csv",
    "group": "identical SOC, heating policy and reported interruption delay stay in one split",
    "cell_identity": "No unique physical cell serial numbers supplied; experiment is the available independent test unit. Physical-cell grouping cannot be verified.",
    "split": {"method": "first fold of StratifiedGroupKFold", "n_splits": 3, "shuffle": True, "seed": 20261115},
    "preprocessing": "StandardScaler fitted only on training experiments",
    "model": {"layers": [3, 8, 4, 1], "activation": "ReLU", "optimizer": "Adam", "learning_rate": 0.01,
              "weight_decay": 0.001, "epochs": 500, "loss": "unweighted BCEWithLogitsLoss", "seed": 20261115},
    "baselines": {"constant": "training positive fraction", "logistic": "C=1, max_iter=1000, same training-only scaling"},
    "classification_threshold": 0.5,
    "selection_policy": "All choices fixed before scoring. No seed/fold/model/threshold search or test-driven tuning. Neither model promoted for deployment.",
    "scope": "Small same-study exploratory pipeline run; no external validation, safe-use threshold, M50T identity, ageing matrix or generalization claim",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_examples(path: Path):
    required = {"experiment_id", "soc_percent", "heating_policy", "reported_intervention_delay_s", "ignition_observed", "label_evidence"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    ids = [r["experiment_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated experiment identifiers are not independent samples")
    included, excluded, vectors, labels, groups = [], [], [], [], []
    for row in rows:
        if row["ignition_observed"] == "":
            excluded.append(row)
            continue
        if row["ignition_observed"] not in {"0", "1"} or not row["label_evidence"].strip():
            raise ValueError("Every nonempty label must be binary and have observation evidence")
        soc = float(row["soc_percent"])
        if not np.isfinite(soc) or soc not in {25., 50., 75., 100.}:
            raise ValueError("Unexpected SOC for this source-specific pilot")
        policy = row["heating_policy"]
        if policy == "continuous":
            if row["reported_intervention_delay_s"] != "":
                raise ValueError("Continuous experiment must not have a stop delay")
            continuous, delay = 1., 0.
        elif policy == "interrupted":
            continuous, delay = 0., float(row["reported_intervention_delay_s"])
            if not np.isfinite(delay) or delay < 0:
                raise ValueError("Invalid intervention condition")
        else:
            raise ValueError(f"Unknown policy {policy}")
        included.append(row)
        vectors.append([soc, continuous, delay])
        labels.append(int(row["ignition_observed"]))
        groups.append(f"{soc:g}|{policy}|{delay:g}")
    return included, excluded, np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(groups)


def split_examples(x, y, groups):
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 3:
        raise ValueError("Pilot requires both observed classes and at least three experiments per class")
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261115)
    train, test = next(split.split(x, y, groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Protocol replicate group leakage")
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Predeclared split lacks both classes; do not search folds for a favourable score")
    return train, test


def network():
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))


def score(y, probability):
    pred = np.asarray(probability) >= 0.5
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {"n": len(y), "positives": int(sum(y)), "negatives": int(len(y)-sum(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "confusion_matrix_true_rows_0_1_predicted_columns_0_1": matrix.tolist(),
            "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def adapt_nist(source: Path, destination: Path):
    """Explicitly adapt the audited experiment table; never infer unknown labels."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 34 or {r["experiment_id"] for r in original} != {str(i) for i in range(49,83)}:
        raise ValueError("Expected the complete 34-experiment NMC21700 table")
    adapted = []
    for row in original:
        if row["dataset_id"] != "NIST_mds2-4027" or row["cell_format"] != "21700" or row["cathode"] != "NMC":
            raise ValueError("Do not mix battery chemistries or synthetic outcomes into this pilot")
        label = {"true": "1", "false": "0", "": ""}[row["ignition_observed"]]
        if label and any(not row[key].strip() for key in ["outcome_source", "outcome_locator", "outcome_annotation_method"]):
            raise ValueError("Every known outcome requires source URL, source locator and annotation method")
        adapted.append({"experiment_id": row["experiment_id"], "soc_percent": row["soc_pct"],
                        "heating_policy": row["heating_mode"], "reported_intervention_delay_s": row["intervention_after_svb_s"],
                        "ignition_observed": label,
                        "label_evidence": " | ".join([row["outcome_source"], row["outcome_locator"], row["outcome_annotation_method"]]),
                        "outcome_note": row["outcome_note"], "parent_csv_sha256": digest(source)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(destination, adapted, list(adapted[0]))
    return destination


def run(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["report.json", "experiment_plan.json", "ignition_mlp.pt", "predictions.csv"]):
        raise FileExistsError("Use a new run directory to preserve prior scores")
    source_hash = digest(source)
    code_hash = digest(Path(__file__))
    snapshot = output / "training_input_snapshot.csv"
    if snapshot.exists():
        raise FileExistsError("Existing input snapshot must not be overwritten")
    snapshot.write_bytes(source.read_bytes())
    if digest(snapshot) != source_hash:
        raise RuntimeError("Input changed while snapshot was made")
    (output / "ignition_training.py").write_bytes(Path(__file__).read_bytes())
    # Persist the declared choices before reading outcomes or fitting a model.
    write_json(output / "experiment_plan.json", PLAN)
    plan_hash = digest(output / "experiment_plan.json")
    rows, excluded, x, y, groups = read_examples(snapshot)
    train, test = split_examples(x, y, groups)
    train_ids, test_ids = {rows[i]["experiment_id"] for i in train}, {rows[i]["experiment_id"] for i in test}
    assert not train_ids & test_ids
    split_rows = [{"experiment_id": r["experiment_id"], "split": "train" if i in train else "test",
                   "protocol_group": groups[i], "ignition_observed": int(y[i])} for i, r in enumerate(rows)]
    write_csv(output / "split_manifest.csv", split_rows, ["experiment_id", "split", "protocol_group", "ignition_observed"])
    write_csv(output / "excluded_experiments.csv", excluded, list(rows[0]))
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x).astype(np.float32)
    torch.set_num_threads(1)
    torch.manual_seed(20261115)
    torch.use_deterministic_algorithms(True)
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    inputs = torch.from_numpy(z[train])
    targets = torch.from_numpy(y[train].astype(np.float32)).reshape(-1, 1)
    history = []
    for epoch in range(1, 501):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            history.append({"epoch": epoch, "training_loss_before_update": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(z))).numpy().reshape(-1)
    logistic = LogisticRegression(C=1., max_iter=1000, random_state=20261115).fit(z[train], y[train])
    logistic_probability = logistic.predict_proba(z)[:, 1]
    constant_probability = np.full(len(y), float(y[train].mean()))
    probabilities = {"mlp": probability, "logistic": logistic_probability, "constant": constant_probability}
    write_csv(output / "training_history.csv", history, list(history[0]))
    predictions = []
    for i, row in enumerate(rows):
        predictions.append({"experiment_id": row["experiment_id"], "split": split_rows[i]["split"],
                            "ignition_observed": int(y[i]), **{f"{k}_probability": float(v[i]) for k,v in probabilities.items()},
                            "mlp_prediction_threshold_0_5": int(probability[i] >= .5)})
    write_csv(output / "predictions.csv", predictions, list(predictions[0]))
    feature_rows = [{"experiment_id": r["experiment_id"], **dict(zip(FEATURES, map(float, x[i]))),
                     "ignition_observed": int(y[i]), "split": split_rows[i]["split"], "label_evidence": r["label_evidence"]} for i,r in enumerate(rows)]
    write_csv(output / "training_features.csv", feature_rows, list(feature_rows[0]))
    checkpoint = {"state_dict": model.state_dict(), "feature_names": FEATURES, "scaler_mean": scaler.mean_.tolist(),
                  "scaler_scale": scaler.scale_.tolist(), "architecture": [3,8,4,1], "plan": PLAN,
                  "source_sha256": source_hash, "train_experiments": sorted(train_ids), "test_experiments": sorted(test_ids)}
    checkpoint_path = output / "ignition_mlp.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = network()
    restored.load_state_dict(loaded["state_dict"])
    restored.eval()
    restored_z = ((x - np.asarray(loaded["scaler_mean"])) / np.asarray(loaded["scaler_scale"])).astype(np.float32)
    with torch.no_grad():
        restored_probability = torch.sigmoid(restored(torch.from_numpy(restored_z))).numpy().reshape(-1)
    restore_error = float(np.max(np.abs(restored_probability - probability)))
    if restore_error > 1e-6:
        raise AssertionError("Saved checkpoint did not reproduce inference")
    if digest(source) != source_hash or digest(snapshot) != source_hash or digest(Path(__file__)) != code_hash or digest(output / "experiment_plan.json") != plan_hash:
        raise RuntimeError("Source, snapshot, code or plan changed during the run; no success report emitted")
    report = {"status": "local_training_and_inference_completed", "deployment_validated": False,
              "source_sha256": source_hash, "code_sha256": code_hash,
              "plan_sha256": plan_hash,
              "included_experiments": len(rows), "excluded_unknown_experiments": len(excluded),
              "protocol_groups": len(set(groups)), "train_experiments": len(train), "test_experiments": len(test),
              "independent_cell_serials_available": False, "parameter_count": sum(p.numel() for p in model.parameters()),
              "model_selection_on_test": False,
              "metrics": {name: {"train": score(y[train], p[train]), "test": score(y[test], p[test])} for name,p in probabilities.items()},
              "integrity": {"experiment_overlap": 0, "protocol_group_overlap": 0, "scaler_fit_train_only": True,
                            "unknown_labels_excluded": True, "no_temperature_or_observed_ignition_svb_time_input": True,
                            "reported_intervention_delay_is_supplied_condition": True,
                            "source_snapshot_code_plan_unchanged_during_run": True,
                            "checkpoint_max_probability_difference": restore_error},
              "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "scikit_learn": sklearn.__version__},
              "limitations": PLAN["scope"]}
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path)
    sources.add_argument("--nist-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any((args.output / name).exists() for name in ["report.json", "experiment_plan.json", "ignition_mlp.pt", "predictions.csv", "source_adapter.csv"]):
        raise FileExistsError("Use a new run directory to preserve prior scores")
    source = adapt_nist(args.nist_input, args.output / "source_adapter.csv") if args.nist_input else args.input
    print(json.dumps(run(source, args.output), ensure_ascii=False, indent=2))
