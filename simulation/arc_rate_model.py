"""Bounded empirical ARC rate(T) experiment, separate from operational SPMeT.

This is a conditional curve fit to supplied surface heating rates. It has no
reaction extent, heat capacity, energy source, clock model, or fire labels.
The reciprocal-temperature slope is an empirical coefficient, not a measured
decomposition activation energy. No prediction is allowed outside M1 support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from simulation.thermal_abuse_validation import Trace, read_trace


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RateModel:
    intercept: float
    slope_K: float
    reference_K: float
    support_min_C: float
    support_max_C: float
    baseline_rate_C_per_min: float
    train_samples: int

    def predict(self, temperature_C: np.ndarray) -> np.ndarray:
        """Return NaN outside training support; never extrapolate or clip T."""
        temperature = np.asarray(temperature_C, dtype=float)
        prediction = np.full(temperature.shape, np.nan)
        supported = (np.isfinite(temperature)
                     & (temperature >= self.support_min_C)
                     & (temperature <= self.support_max_C))
        x = 1 / self.reference_K - 1 / (temperature[supported] + 273.15)
        prediction[supported] = np.exp(self.intercept + self.slope_K * x)
        return prediction


def fit_rate(temperature_C: np.ndarray, rate_C_per_min: np.ndarray,
             window_C: tuple[float, float], reference_K: float = 443.15,
             minimum_samples: int = 20) -> RateModel:
    """Fit M1 only using centered/scaled linear least squares in ln(rate)."""
    temperature = np.asarray(temperature_C, dtype=float)
    rate = np.asarray(rate_C_per_min, dtype=float)
    if (temperature.ndim != 1 or rate.shape != temperature.shape
            or not np.isfinite(temperature).all() or not np.isfinite(rate).all()
            or np.any(temperature <= -273.15)):
        raise ValueError("Temperature/rate vectors must be finite with physical Kelvin T")
    lo, hi = window_C
    if not np.isfinite([lo, hi, reference_K]).all() or lo >= hi or reference_K <= 0:
        raise ValueError("Invalid fixed fit window or reference temperature")
    mask = (temperature >= lo) & (temperature <= hi) & (rate > 0)
    if np.count_nonzero(mask) < minimum_samples:
        raise ValueError("Insufficient positive-rate training samples in fixed window")
    temperature, rate = temperature[mask], rate[mask]
    x = 1 / reference_K - 1 / (temperature + 273.15)
    spread = float(np.std(x))
    if spread <= 1e-12:
        raise ValueError("Training temperatures cannot identify a slope")
    x_mean = float(np.mean(x))
    design = np.column_stack([np.ones(len(x)), (x - x_mean) / spread])
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.log(rate), rcond=None)
    if rank != 2:
        raise ValueError("Rank-deficient empirical fit")
    slope = float(coefficients[1] / spread)
    intercept = float(coefficients[0] - slope * x_mean)
    model = RateModel(intercept, slope, reference_K, float(temperature.min()),
                      float(temperature.max()), float(np.exp(np.mean(np.log(rate)))),
                      len(rate))
    if not np.isfinite(model.predict(temperature)).all():
        raise ValueError("Nonfinite fitted rate")
    return model


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Equal sample weights; log errors use natural logarithms, not log10."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (observed.ndim != 1 or predicted.shape != observed.shape or len(observed) == 0
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()
            or np.any(observed <= 0) or np.any(predicted <= 0)):
        raise ValueError("Metrics need nonempty finite positive paired rates")
    log_error = np.log(predicted) - np.log(observed)
    absolute_relative_error = np.abs(predicted - observed) / observed
    log_rmse = float(np.sqrt(np.mean(log_error ** 2)))
    return {
        "samples": len(observed), "log_RMSE_ln": log_rmse,
        "exp_log_RMSE_factor": float(np.exp(log_rmse)),
        "median_absolute_relative_error": float(np.median(absolute_relative_error)),
        "mean_absolute_relative_error": float(np.mean(absolute_relative_error)),
        "p90_absolute_relative_error": float(np.quantile(absolute_relative_error, 0.9)),
        "max_absolute_relative_error": float(absolute_relative_error.max()),
        "rate_RMSE_C_per_min": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "observed_rate_min_C_per_min": float(observed.min()),
        "observed_rate_max_C_per_min": float(observed.max()),
    }


def score_trace(model: RateModel, trace: Trace) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted = model.predict(trace.temperature_C)
    mask = np.isfinite(predicted) & (trace.rate_C_per_min > 0)
    if not mask.any():
        raise ValueError("Evaluation has no positive rates within M1 temperature support")
    scored_indices = np.flatnonzero(mask)
    first, last = int(scored_indices[0]), int(scored_indices[-1])
    intervals = np.diff(trace.time_s)
    # This is a clock-span audit, not a claim of continuous exothermal coverage.
    coverage = {
        "total_samples": len(trace.time_s), "scored_samples": int(mask.sum()),
        "sample_fraction": float(mask.mean()),
        "excluded_samples": int((~mask).sum()),
        "total_temperature_min_C": float(trace.temperature_C.min()),
        "total_temperature_max_C": float(trace.temperature_C.max()),
        "total_rate_min_C_per_min": float(trace.rate_C_per_min.min()),
        "total_rate_max_C_per_min": float(trace.rate_C_per_min.max()),
        "original_clock_first_s": float(trace.time_s[0]),
        "original_clock_last_s": float(trace.time_s[-1]),
        "raw_clock_span_s": float(trace.time_s[-1] - trace.time_s[0]),
        "scored_first_original_clock_s": float(trace.time_s[first]),
        "scored_last_original_clock_s": float(trace.time_s[last]),
        "scored_clock_envelope_s": float(trace.time_s[last] - trace.time_s[first]),
        "scored_temperature_min_C": float(trace.temperature_C[mask].min()),
        "scored_temperature_max_C": float(trace.temperature_C[mask].max()),
        "raw_inter_sample_gap_min_s": float(intervals.min()),
        "raw_inter_sample_gap_max_s": float(intervals.max()),
        "continuous_time_coverage_fraction": None,
        "time_coverage_note": "Clock envelopes only. Missing HWS phases/segment boundaries prevent a continuous coverage claim.",
        "trace_complete": trace.metadata["trace_complete"],
    }
    observed = trace.rate_C_per_min[mask]
    return {"coverage": coverage, "model": metrics(observed, predicted[mask]),
            "constant_baseline": metrics(observed, np.full(mask.sum(),
                                                        model.baseline_rate_C_per_min))}, predicted, mask


def validate_pair(train: Trace, evaluation: Trace) -> None:
    for trace in (train, evaluation):
        if trace.metadata.get("kind") != "measured" or trace.metadata.get("rate_source") != "provided":
            raise ValueError("Only provided experimental rates are supported")
    for key in ("cell_model", "soc_percent", "protocol", "temperature_observable", "time_origin"):
        if key not in train.metadata or train.metadata.get(key) != evaluation.metadata.get(key):
            raise ValueError(f"Incompatible or missing experimental {key}")
    if (not train.metadata.get("replicate") or not evaluation.metadata.get("replicate")
            or train.metadata["replicate"] == evaluation.metadata["replicate"]
            or train.metadata["csv_sha256"] == evaluation.metadata["csv_sha256"]):
        raise ValueError("Evaluation must use a distinct measured replicate")


def acceptance(model: RateModel, evaluation_score: dict, rule: dict) -> dict:
    score = evaluation_score["model"]
    checks = {
        "minimum_train_samples": model.train_samples >= rule["minimum_train_samples"],
        "minimum_evaluation_samples": score["samples"] >= rule["minimum_evaluation_samples"],
        "positive_finite_slope": bool(np.isfinite(model.slope_K) and model.slope_K > 0),
        "finite_parameters": bool(np.isfinite(list(asdict(model).values())).all()),
        "improves_on_constant_log_RMSE": score["log_RMSE_ln"] < evaluation_score["constant_baseline"]["log_RMSE_ln"],
        "median_relative_error": score["median_absolute_relative_error"] <= rule["evaluation_median_absolute_relative_error_max"],
        "p90_relative_error": score["p90_absolute_relative_error"] <= rule["evaluation_p90_absolute_relative_error_max"],
    }
    return {"checks": checks, "bounded_rate_fit_passed": all(checks.values()),
            "thermal_runaway_prediction_validated": False,
            "fire_prediction_validated": False, "M50T_transfer_validated": False}


def write_comparison(path: Path, trace_id: str, role: str, trace: Trace,
                     model: RateModel, prediction: np.ndarray, mask: np.ndarray) -> None:
    fields = ["trace_id", "role", "source_row_1_based", "time_s", "temperature_C",
              "rate_C_per_min", "segment_id", "scored", "exclusion_reason",
              "predicted_rate_C_per_min", "baseline_rate_C_per_min",
              "signed_relative_error", "absolute_relative_error", "log_error_ln",
              "thermal_runaway_label", "fire_label"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(trace.time_s)):
            row = {"trace_id": trace_id, "role": role, "source_row_1_based": i + 2,
                   "time_s": trace.time_s[i], "temperature_C": trace.temperature_C[i],
                   "rate_C_per_min": trace.rate_C_per_min[i], "segment_id": trace.segment[i],
                   "scored": bool(mask[i]), "thermal_runaway_label": "", "fire_label": ""}
            if mask[i]:
                relative = prediction[i] / trace.rate_C_per_min[i] - 1
                row.update(exclusion_reason="", predicted_rate_C_per_min=prediction[i],
                           baseline_rate_C_per_min=model.baseline_rate_C_per_min,
                           signed_relative_error=relative, absolute_relative_error=abs(relative),
                           log_error_ln=float(np.log(prediction[i] / trace.rate_C_per_min[i])))
            else:
                row["exclusion_reason"] = ("outside_M1_temperature_support" if not np.isfinite(prediction[i])
                                           else "nonpositive_observed_rate")
            writer.writerow(row)


def plot_comparison(path: Path, train: Trace, evaluation: Trace, model: RateModel) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for trace, label, color in ((train, "M1 (fit)", "#2878b5"),
                                (evaluation, "M2 (held out)", "#c45118")):
        # Scatter avoids drawing continuous histories through missing HWS data.
        for axis in axes:
            axis.scatter(trace.temperature_C, trace.rate_C_per_min, s=18,
                         color=color, alpha=0.75, label=label)
    grid = np.linspace(model.support_min_C, model.support_max_C, 200)
    for axis in axes:
        axis.plot(grid, model.predict(grid), color="black", lw=1.8, label="M1 empirical fit")
        axis.plot(grid, np.repeat(model.baseline_rate_C_per_min, len(grid)),
                  color="#777777", ls="--", label="M1 constant baseline")
        axis.axvspan(model.support_min_C, model.support_max_C, color="#888888", alpha=0.1)
        axis.set_yscale("log")
        axis.set_xlabel("Measured surface-center temperature (°C)")
        axis.set_ylabel("Provided heating rate (°C/min), log scale")
        axis.grid(alpha=0.2)
    axes[0].set_title("All recorded samples retained")
    axes[1].set_title("Conditional rate fit within M1 support")
    axes[1].set_xlim(148, 192)
    axes[1].set_ylim(1, 50)
    axes[0].legend(fontsize=8)
    fig.suptitle("Anonymous NMC / SOC100 / ARC — no full time trace or fire prediction", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiment(plan_path: Path, input_dir: Path, raw_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists() or (output / "frozen_model.json").exists():
        raise FileExistsError("Use a new experiment directory; do not overwrite a scored held-out run")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan["declared_before_current_fit_and_M2_scoring"]:
        raise ValueError("A declared experiment plan is required")
    ids = (plan["train_id"], plan["evaluation_id"])
    paths = [plan_path, Path(__file__), Path(__file__).with_name("thermal_abuse_validation.py")]
    for trace_id in ids:
        paths.extend([input_dir / f"{trace_id}.csv", input_dir / f"{trace_id}.metadata.json",
                      raw_dir / f"{trace_id}.txt"])
    hashes = {str(path.resolve()): sha256(path) for path in paths}
    source_copies = output / "source_data"
    source_copies.mkdir(exist_ok=True)
    for trace_id in ids:
        raw = raw_dir / f"{trace_id}.txt"
        metadata = json.loads((input_dir / f"{trace_id}.metadata.json").read_text(encoding="utf-8"))
        if (sha256(raw) != plan["raw_expected_sha256"][trace_id]
                or sha256(raw) != metadata["raw_sha256"]
                or hashlib.md5(raw.read_bytes()).hexdigest() != metadata["raw_md5"]):
            raise ValueError("Public raw file does not match frozen source identity")
        # Verify each canonical number directly against the original TXT; full
        # canonical and raw bytes are archived, including all out-of-window rows.
        raw_rows = list(csv.reader(raw.read_text(encoding="cp1252").splitlines(), delimiter="\t"))[1:]
        canonical = read_trace(input_dir / f"{trace_id}.csv", input_dir / f"{trace_id}.metadata.json", 1)
        raw_values = np.array([[float(value) for value in row] for row in raw_rows])
        expected = np.column_stack((raw_values[:, 0] * 60, raw_values[:, 1], raw_values[:, 2]))
        actual = np.column_stack((canonical.time_s, canonical.temperature_C, canonical.rate_C_per_min))
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            raise ValueError("Canonical values differ from published raw data")
        for source in (raw, input_dir / f"{trace_id}.csv", input_dir / f"{trace_id}.metadata.json"):
            shutil.copyfile(source, source_copies / source.name)
    if plan_path.resolve() != (output / "experiment_plan.json").resolve():
        shutil.copyfile(plan_path, output / "experiment_plan.json")
    started = datetime.now(timezone.utc).isoformat()
    train = read_trace(input_dir / f"{ids[0]}.csv", input_dir / f"{ids[0]}.metadata.json", 1)
    model = fit_rate(train.temperature_C, train.rate_C_per_min, tuple(plan["fit_window_C"]),
                     plan["reference_K"], plan["acceptance"]["minimum_train_samples"])
    # Freeze coefficients before evaluation rates are used for any metric.
    save_json(output / "frozen_model.json", {
        "model": asdict(model), "plan_sha256": sha256(plan_path),
        "source_module_sha256": sha256(Path(__file__)), "train_csv_sha256": train.metadata["csv_sha256"],
        "fit_uses_evaluation_data": False, "coefficient_meaning": "apparent empirical rate slope; no mechanistic activation energy claim",
    })
    frozen_hash = sha256(output / "frozen_model.json")
    evaluation = read_trace(input_dir / f"{ids[1]}.csv", input_dir / f"{ids[1]}.metadata.json", 1)
    validate_pair(train, evaluation)
    scores = {}
    for trace_id, role, trace in ((ids[0], "train", train), (ids[1], "held_out_from_fit", evaluation)):
        score, predicted, mask = score_trace(model, trace)
        scores[trace_id] = score
        write_comparison(output / f"{trace_id}_comparison.csv", trace_id, role, trace, model, predicted, mask)
    plot_comparison(output / "rate_comparison.png", train, evaluation, model)
    unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    unchanged = unchanged and sha256(output / "frozen_model.json") == frozen_hash
    if not unchanged:
        raise RuntimeError("Inputs, model, plan or code changed during evaluation")
    report = {
        "experiment_id": plan["experiment_id"], "started_UTC": started,
        "finished_UTC": datetime.now(timezone.utc).isoformat(),
        "method": plan["model"], "plan_sha256": sha256(plan_path), "frozen_model_sha256": frozen_hash,
        "model": asdict(model), "scores": scores,
        "acceptance": acceptance(model, scores[ids[1]], plan["acceptance"]),
        "input_and_source_hashes": hashes, "sources_unchanged_after_evaluation": unchanged,
        "dependencies": {"python": platform.python_version(), "numpy": version("numpy"),
                         "matplotlib": version("matplotlib")},
        "source_metadata": {ids[0]: train.metadata, ids[1]: evaluation.metadata},
        "thermal_runaway_label": None, "fire_label": None,
        "limits": [plan["prior_exposure"], plan["time_policy"], plan["physical_scope"],
                   "Single training cell and one evaluation cell; no population uncertainty or prospective external validation.",
                   "Scores use equal recorded-sample weights, not elapsed-time weights. Full raw out-of-window values remain available.",
                   "Rate is conditioned on measured temperature; evaluating a curve is not a future-temperature or self-contained thermal simulation."],
        "sources": plan["sources"],
    }
    save_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.plan, args.input_dir, args.raw_dir, args.output)
    print(json.dumps({"report": str(args.output / "report.json"), "acceptance": report["acceptance"],
                      "evaluation": report["scores"]["NMC_SOC100_M2"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
