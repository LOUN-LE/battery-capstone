"""C-trained ten-second voltage forecast and retrospective review routing.

This is a connection pilot, not a cell hazard/lifetime classifier. Every forecast
uses sensors available at issue time. The subsequent observation is used only to
score that forecast and make a review decision at the observation's timestamp.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge


SCHEMA = "measured_voltage_review_v1"
REQUIRED = ("cell_id", "session_id", "time_s", "voltage_V", "current_A", "surface_temperature_C")
FEATURES = ("issue_voltage_V", "issue_current_A", "issue_surface_temperature_C",
            "past_voltage_slope_V_per_s", "past_current_slope_A_per_s")
POLICY = {
    "nominal_horizon_s": 10.0,
    "maximum_actual_horizon_s": 11.25,
    "history_s": 10.0,
    "maximum_history_s": 11.25,
    "maximum_sample_gap_s": 2.0,
    "training_target_end_relative_s": 360.0,
    "calibration_issue_start_relative_s": 370.0,
    "maximum_session_relative_s": 600.0,
    "evaluation_issue_stride_s": 10.0,
    "calibration_quantile": 0.95,
    "threshold_rule": "strictly greater than C calibration empirical 95th percentile (higher order statistic)",
    "adoption_rule": "C calibration voltage RMSE strictly lower than issue-voltage persistence RMSE",
    "selection_requires_adoption": True,
    "support_distance": "nearest C training feature row, Euclidean after C training-only scaling",
    "scale_rule": "C training standard deviation; features with std < 1e-8 use 1 original unit",
    "scope": "nominal 10-second voltage forecast; unknown future current is not an input",
    "coverage_guarantee": False,
    "risk_classification_validated": False,
    "lifetime_prediction_validated": False,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def canonicalize(rows):
    """Sort within each session, reject ambiguity; ignore unrelated metadata."""
    groups = {}
    for number, row in enumerate(rows):
        if any(key not in row for key in REQUIRED):
            raise ValueError(f"Row {number}: missing a required canonical column")
        cell, session = str(row["cell_id"]), str(row["session_id"])
        if not cell or not session or not cell.endswith(("_C", "_D")):
            raise ValueError("Expected identified Imperial cell IDs ending _C or _D")
        try:
            values = [float(row[key]) for key in REQUIRED[2:]]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row {number}: missing or nonnumeric measured sensor value") from exc
        if not np.isfinite(values).all():
            raise ValueError("Missing/nonfinite measurements must not be imputed for selection")
        groups.setdefault((cell, session), []).append(values)
    if len(groups) != 2 or {cell[-1] for cell, _ in groups} != {"C", "D"}:
        raise ValueError("This fixed pilot requires one C session and one D session")
    result = {}
    for (cell, session), values in sorted(groups.items()):
        a = np.asarray(sorted(values, key=lambda item: item[0]), dtype=float)
        if len(a) < 80 or np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("Each session needs at least 80 unique, increasing timestamps")
        relative = a[:, 0] - a[0, 0]
        if relative[-1] > POLICY["maximum_session_relative_s"] + 1e-6:
            raise ValueError("Only the predeclared first 600 seconds may enter this pilot")
        if np.max(np.diff(relative)) > POLICY["maximum_sample_gap_s"]:
            raise ValueError("A recording gap exceeds the declared continuity limit")
        result[cell[-1]] = {"cell_id": cell, "session_id": session,
                              "array": a, "relative": relative}
    return result


def make_samples(session):
    """Use an as-of history row and next observation after the fixed horizon.

    Actual target time/horizon affect scoring only, never the input features.
    There is no interpolation across a future measurement and no full-run feature.
    """
    a, t = session["array"], session["relative"]
    samples = []
    for issue in range(len(t)):
        history = int(np.searchsorted(t, t[issue] - POLICY["history_s"], side="right") - 1)
        target = int(np.searchsorted(t, t[issue] + POLICY["nominal_horizon_s"], side="left"))
        if history < 0 or target >= len(t):
            continue
        history_dt, horizon = t[issue] - t[history], t[target] - t[issue]
        if (history_dt > POLICY["maximum_history_s"] or
                horizon > POLICY["maximum_actual_horizon_s"]):
            continue
        features = [a[issue, 1], a[issue, 2], a[issue, 3],
                    (a[issue, 1] - a[history, 1]) / history_dt,
                    (a[issue, 2] - a[history, 2]) / history_dt]
        samples.append({
            "cell_id": session["cell_id"], "session_id": session["session_id"],
            "issue_time_s": float(a[issue, 0]), "target_time_s": float(a[target, 0]),
            "feature_history_start_time_s": float(a[history, 0]),
            "issue_relative_s": float(t[issue]), "target_relative_s": float(t[target]),
            "history_relative_s": float(t[history]), "actual_horizon_s": float(horizon),
            "features": features, "observed_voltage_V": float(a[target, 1]),
            "delta_voltage_V": float(a[target, 1] - a[issue, 1]),
        })
    return samples


def _stride(samples):
    result, last = [], -float("inf")
    for sample in samples:
        if sample["issue_relative_s"] >= last + POLICY["evaluation_issue_stride_s"] - 1e-7:
            result.append(sample)
            last = sample["issue_relative_s"]
    return result


def predict_frozen(model, features):
    """Portable prediction from JSON, including the fitted Bayesian covariance."""
    x = (np.asarray(features, dtype=float) - model["feature_mean"]) / model["feature_scale"]
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("Prediction needs finite rows of the declared five features")
    delta = x @ np.asarray(model["coefficient"]) + model["intercept"]
    variance = np.einsum("ij,jk,ik->i", x, np.asarray(model["coefficient_covariance"]), x)
    variance += 1.0 / model["noise_precision"]
    std = np.sqrt(np.maximum(variance, 0.0))
    training = np.asarray(model["scaled_training_features"])
    distance = np.linalg.norm(x[:, None, :] - training[None, :, :], axis=2).min(axis=1)
    return delta, std, distance


def _metrics(samples, predictions):
    observed = np.array([sample["observed_voltage_V"] for sample in samples])
    persistence = np.array([sample["features"][0] for sample in samples])
    error = 1000.0 * (predictions - observed)
    baseline = 1000.0 * (persistence - observed)
    return {"samples": len(samples), "voltage_rmse_mV": float(np.sqrt(np.mean(error ** 2))),
            "voltage_mae_mV": float(np.mean(abs(error))),
            "voltage_max_abs_error_mV": float(np.max(abs(error))),
            "persistence_rmse_mV": float(np.sqrt(np.mean(baseline ** 2))),
            "persistence_mae_mV": float(np.mean(abs(baseline)))}


def fit_c_only(samples):
    train = [s for s in samples if s["target_relative_s"] <= POLICY["training_target_end_relative_s"]]
    calibration = _stride([s for s in samples
                           if s["issue_relative_s"] >= POLICY["calibration_issue_start_relative_s"]
                           and s["history_relative_s"] > POLICY["training_target_end_relative_s"]])
    if len(train) < 50 or len(calibration) < 10:
        raise ValueError("Insufficient C data for the fixed training/calibration split")
    x = np.asarray([s["features"] for s in train])
    y = np.asarray([s["delta_voltage_V"] for s in train])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    estimator = BayesianRidge(max_iter=500, tol=1e-8).fit(standardized, y)
    model = {
        "schema": SCHEMA, "method": "BayesianRidge", "features": list(FEATURES),
        "target": "observed target voltage minus issue voltage, in V",
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "coefficient": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
        "coefficient_covariance": estimator.sigma_.tolist(),
        "noise_precision": float(estimator.alpha_), "weight_precision": float(estimator.lambda_),
        "iterations": int(estimator.n_iter_), "estimator_parameters": estimator.get_params(),
        "scaled_training_features": standardized.tolist(),
        "training_numeric_sha256": _digest({"x": x.tolist(), "y": y.tolist()}),
        "policy": POLICY,
        "uncertainty_scope": "Bayesian linear model predictive std; empirical serial calibration, not a safety probability or guaranteed coverage",
    }
    cal_x = np.asarray([s["features"] for s in calibration])
    delta, std, distance = predict_frozen(model, cal_x)
    prediction = cal_x[:, 0] + delta
    residual = abs(prediction - np.asarray([s["observed_voltage_V"] for s in calibration]))
    q = lambda values: float(np.quantile(values, POLICY["calibration_quantile"], method="higher"))
    model["thresholds"] = {"absolute_residual_V": q(residual),
                           "predictive_std_V": q(std), "support_distance": q(distance)}
    model["calibration_numeric_sha256"] = _digest({"x": cal_x.tolist(),
                                                    "y": [s["delta_voltage_V"] for s in calibration]})
    metrics = _metrics(calibration, prediction)
    model["adopted"] = metrics["voltage_rmse_mV"] < metrics["persistence_rmse_mV"]
    split = {"training_count": len(train), "calibration_count": len(calibration),
             "training_last_target_relative_s": max(s["target_relative_s"] for s in train),
             "calibration_first_history_relative_s": min(s["history_relative_s"] for s in calibration),
             "purged_history_and_target_overlap": True,
             "calibration_metrics": metrics,
             "calibration_is_independent_test": False,
             "serial_correlation_remains": True}
    return model, calibration, split


def decide(model, samples, split):
    features = np.asarray([s["features"] for s in samples])
    delta, std, distance = predict_frozen(model, features)
    predictions = features[:, 0] + delta
    thresholds = model["thresholds"]
    decisions = []
    for sample, predicted, uncertainty, support in zip(samples, predictions, std, distance):
        residual = sample["observed_voltage_V"] - predicted
        flags = {"residual_exceeded": abs(residual) > thresholds["absolute_residual_V"],
                 "uncertainty_exceeded": uncertainty > thresholds["predictive_std_V"],
                 "support_exceeded": support > thresholds["support_distance"]}
        reason = ";".join(name for name, active in flags.items() if active) or "no_review_trigger"
        selected = bool(any(flags.values()) and model["adopted"])
        record = {key: sample[key] for key in ("cell_id", "session_id", "issue_time_s", "target_time_s",
                  "issue_relative_s", "target_relative_s", "feature_history_start_time_s", "actual_horizon_s")}
        record.update({"split": split, "nominal_horizon_s": POLICY["nominal_horizon_s"],
                       "predicted_voltage_V": float(predicted),
                       "observed_voltage_V": sample["observed_voltage_V"],
                       "residual_mV": float(1000 * residual), "predictive_std_mV": float(1000 * uncertainty),
                       "training_support_distance": float(support), **{k: bool(v) for k, v in flags.items()},
                       "candidate_review_reason": reason, "selected_by_model": selected,
                       "selection_reason": reason if model["adopted"] else "model_not_adopted",
                       "decision_cutoff_s": sample["target_time_s"], "ml_adopted": bool(model["adopted"]),
                       "risk_classification_validated": False, "lifetime_prediction_validated": False})
        decisions.append(record)
    metrics = _metrics(samples, predictions)
    metrics.update({"review_trigger_count_before_adoption_gate": sum(any(r[k] for k in
                       ("residual_exceeded", "uncertainty_exceeded", "support_exceeded")) for r in decisions),
                    "selected_count": sum(r["selected_by_model"] for r in decisions),
                    "residual_exceeded_count": sum(r["residual_exceeded"] for r in decisions),
                    "uncertainty_exceeded_count": sum(r["uncertainty_exceeded"] for r in decisions),
                    "support_exceeded_count": sum(r["support_exceeded"] for r in decisions)})
    return decisions, metrics


def evaluate_rows(rows):
    sessions = canonicalize(rows)
    c_samples = make_samples(sessions["C"])
    model, calibration, split = fit_c_only(c_samples)
    d_samples = _stride(make_samples(sessions["D"]))
    if len(d_samples) < 10:
        raise ValueError("Insufficient D observations for frozen evaluation")
    decisions_c, metrics_c = decide(model, calibration, "C_calibration_reused_for_threshold")
    decisions_d, metrics_d = decide(model, d_samples, "D_frozen_evaluation")
    decisions = decisions_c + decisions_d
    summaries = {}
    for label, records, metrics in (("C", decisions_c, metrics_c), ("D", decisions_d, metrics_d)):
        chosen = [r for r in records if r["selected_by_model"]]
        summaries[label] = {"cell_id": sessions[label]["cell_id"], "selected": bool(chosen),
                           "selected_windows": len(chosen), "evaluated_windows": len(records),
                           "selection_reason": "review_trigger" if chosen else
                           ("no_review_trigger" if model["adopted"] else "model_not_adopted"),
                           "decision_cutoff_s": max((r["decision_cutoff_s"] for r in chosen), default=None),
                           "metrics": metrics}
    report = {
        "schema": SCHEMA, "status": "complete", "policy": POLICY, "split": split,
        "model_sha256": _digest(model), "ml_adopted": bool(model["adopted"]),
        "selection_scope": "retrospective observed-voltage disagreement or uncertainty review, not abnormal-cell ground truth",
        "D_was_used_to_fit_or_calibrate": False, "D_is_new_blind_project_dataset": False,
        "future_sensors_used_as_features": False, "issued_before_target_observation": True,
        "calibration_thresholds_are_research_policy": True,
        "independent_cell_count": 2, "independent_training_cell_count": 1,
        "risk_classification_validated": False, "lifetime_prediction_validated": False,
        "degradation_cause_identified": False, "risk_label": None,
        "future_load_forecast_validated": False, "continuous_open_loop_forecast": False,
        "actual_horizon_range_s": [min(r["actual_horizon_s"] for r in decisions),
                                   max(r["actual_horizon_s"] for r in decisions)],
        "cell_summaries": summaries,
        "limitations": ["C and D are short fresh-cell RPT0 records, not observed fault or lifetime classes",
                        "Every forecast starts with measured issue-time voltage; it is not a recursive long-horizon forecast",
                        "Unknown future current changes can cause disagreement without degradation",
                        "Small residuals do not establish safety and slow degradation can be missed",
                        "C calibration thresholds are descriptive and serially dependent; no guaranteed 95 percent coverage",
                        "D has been inspected previously elsewhere in this project"],
    }
    return model, report, decisions


def _atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_selector_rows(rows, outdir, *, input_provenance=None):
    """Return paths + C/D summaries; never insert an integration-control flag."""
    output = Path(outdir)
    if output.exists():
        raise FileExistsError(f"Preserve previous output: {output}")
    source = Path(__file__).read_bytes()
    model, report, decisions = evaluate_rows(rows)
    if source != Path(__file__).read_bytes():
        raise RuntimeError("Selector implementation changed during evaluation")
    report["provenance"] = {"input": input_provenance,
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}}
    output.mkdir(parents=True, exist_ok=False)
    model_path, report_path = output / "frozen_model.json", output / "training_report.json"
    decisions_path = output / "selection_decisions.csv"
    _atomic_text(model_path, json.dumps(model, indent=2, allow_nan=False) + "\n")
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
    writer.writeheader()
    writer.writerows(decisions)
    _atomic_text(decisions_path, stream.getvalue())
    report["artifacts_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (model_path, decisions_path)}
    _atomic_text(report_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"status": "complete", "report_path": str(report_path.resolve()),
            "decisions_path": str(decisions_path.resolve()), "model_path": str(model_path.resolve()),
            "model_sha256": report["model_sha256"], "ml_adopted": report["ml_adopted"],
            "cell_summaries": report["cell_summaries"],
            "risk_classification_validated": False, "lifetime_prediction_validated": False}


def run_selector(input_csv, outdir):
    path = Path(input_csv)
    raw = path.read_bytes()
    import io
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    return run_selector_rows(rows, outdir, input_provenance={"path": str(path.resolve()),
                             "sha256": hashlib.sha256(raw).hexdigest()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_selector(args.input, args.output), indent=2, allow_nan=False))
