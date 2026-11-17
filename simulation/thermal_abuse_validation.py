"""Audit thermal-abuse traces without treating a temperature limit as a fire label.

Inputs are canonical CSV files with time_s, temperature_C, and optionally
rate_C_per_min and segment_id. Metadata specifies the experiment, observable,
clock origin, source and completeness. A prediction uses the same time origin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Criterion:
    temperature_C: float
    rate_C_per_min: float
    persistence_s: float
    max_gap_s: float
    definition_source: str

    def __post_init__(self):
        values = (self.temperature_C, self.rate_C_per_min,
                  self.persistence_s, self.max_gap_s)
        if not np.isfinite(values).all():
            raise ValueError("Criterion values must be finite")
        if self.persistence_s < 0 or self.max_gap_s <= 0:
            raise ValueError("Persistence must be nonnegative and max gap positive")
        if not self.definition_source.strip():
            raise ValueError("Declare the origin and purpose of the research criterion")


@dataclass
class Trace:
    time_s: np.ndarray
    temperature_C: np.ndarray
    rate_C_per_min: np.ndarray
    segment: np.ndarray
    metadata: dict


def import_ohneseit_arc(source: Path, output_csv: Path, metadata_path: Path,
                       expected_md5: str) -> None:
    """Convert Zenodo 7707929 raw units, retaining the acquisition clock.

    The released files contain exothermal samples only. Their incomplete time
    histories do not establish a negative runaway/fire label.
    """
    payload = source.read_bytes()
    if hashlib.md5(payload).hexdigest() != expected_md5.lower():
        raise ValueError("ARC source checksum differs from the published manifest")
    rows = list(csv.reader(io.StringIO(payload.decode("cp1252")), delimiter="\t"))
    expected_header = ["time / min", "temperature / °C", "temperature rate / ( °C/min)"]
    if rows[0] != expected_header:
        raise ValueError("Unexpected ARC column names or units")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "temperature_C", "rate_C_per_min", "segment_id"])
        for row in rows[1:]:
            if len(row) != 3:
                raise ValueError("Unexpected ARC row length")
            minute, temperature, rate = map(float, row)
            writer.writerow([minute * 60, temperature, rate, "exothermal_export"])
    metadata = {
        "kind": "measured", "cell_model": "Ohneseit2023_NMC_21700_manufacturer_undisclosed",
        "source_url": f"https://zenodo.org/records/7707929/files/{source.name}",
        "publication": "https://doi.org/10.3390/batteries9050237",
        "raw_sha256": hashlib.sha256(payload).hexdigest(), "raw_md5": expected_md5.lower(),
        "protocol": "ARC_HWS_exothermal_export", "temperature_observable": "surface_center",
        "time_origin": "original_ARC_test_clock", "trace_complete": False,
        "soc_percent": int(source.stem.split("SOC")[1].split("_")[0]),
        "replicate": source.stem.split("_")[-1], "observed_events": {},
        "scope": "Public exothermal surface-temperature samples. M50T equivalence unverified; missing HWS phases and flame evidence.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_trace(csv_path: Path, metadata_path: Path, max_gap_s: float) -> Trace:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    required = ("source_url", "cell_model", "protocol", "temperature_observable",
                "time_origin", "trace_complete", "kind")
    if any(key not in metadata for key in required):
        raise ValueError(f"Metadata must provide {required}")
    if metadata["kind"] not in ("measured", "simulation", "synthetic_test"):
        raise ValueError("Unknown trace kind")
    if not isinstance(metadata["trace_complete"], bool):
        raise ValueError("trace_complete must be a boolean")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("At least two temperature samples are required")
    time = np.array([float(row["time_s"]) for row in rows])
    temperature = np.array([float(row["temperature_C"]) for row in rows])
    if not np.isfinite(time).all() or not np.isfinite(temperature).all():
        raise ValueError("Nonfinite temperatures or times")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time must be strictly increasing; repair the source explicitly")
    segment = np.array([row.get("segment_id", "0") for row in rows])
    supplied_rate = [row.get("rate_C_per_min", "") for row in rows]
    if any(supplied_rate) and not all(supplied_rate):
        raise ValueError("Supply all rate samples or omit the column")
    if all(supplied_rate):
        rate = np.array([float(value) for value in supplied_rate])
        if not np.isfinite(rate).all():
            raise ValueError("Nonfinite measured rates")
        rate_source = "provided"
    else:
        # Backward interval rates keep future observations out of onset detection.
        # Gaps and segment changes have no derivative and cannot trigger an event.
        rate = np.full(len(time), np.nan)
        valid = (np.diff(time) <= max_gap_s) & (segment[1:] == segment[:-1])
        rate[1:] = np.where(valid, np.diff(temperature) / np.diff(time) * 60, np.nan)
        rate_source = "backward_difference"
    metadata = {**metadata, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "rate_source": rate_source}
    return Trace(time, temperature, rate, segment, metadata)


def first_persistent_crossing(trace: Trace, criterion: Criterion) -> dict:
    mask = ((trace.temperature_C >= criterion.temperature_C)
            & (trace.rate_C_per_min >= criterion.rate_C_per_min))
    start = None
    for index, active in enumerate(mask):
        interrupted = index > 0 and (
            trace.time_s[index] - trace.time_s[index - 1] > criterion.max_gap_s
            or trace.segment[index] != trace.segment[index - 1])
        if interrupted or not active:
            start = None
        if active and start is None:
            start = index
        if start is not None and trace.time_s[index] - trace.time_s[start] >= criterion.persistence_s:
            return {"status": "criterion_observed", "onset_s": float(trace.time_s[start]),
                    "confirmed_at_s": float(trace.time_s[index]),
                    "onset_temperature_C": float(trace.temperature_C[start])}
    status = "not_observed_in_complete_trace" if trace.metadata["trace_complete"] else "censored"
    return {"status": status, "onset_s": None, "confirmed_at_s": None,
            "onset_temperature_C": None}


def summarize(trace: Trace, criterion: Criterion) -> dict:
    finite_rates = trace.rate_C_per_min[np.isfinite(trace.rate_C_per_min)]
    observed_labels = {}
    for event in ("thermal_runaway", "fire"):
        annotation = trace.metadata.get("observed_events", {}).get(event)
        if annotation is None:
            observed_labels[event] = {"observed": None, "evidence": None}
        else:
            if trace.metadata["kind"] != "measured":
                raise ValueError("Observed event labels require a measured trace")
            if not isinstance(annotation.get("observed"), bool) or not annotation.get("evidence"):
                raise ValueError("Event annotation requires boolean observed and source evidence")
            observed_labels[event] = annotation
    return {
        "metadata": trace.metadata,
        "samples": len(trace.time_s),
        "start_s": float(trace.time_s[0]), "end_s": float(trace.time_s[-1]),
        "recorded_peak_temperature_C": float(trace.temperature_C.max()),
        "recorded_peak_rate_C_per_min": float(finite_rates.max()) if len(finite_rates) else None,
        "research_criterion": asdict(criterion),
        "criterion_crossing": first_persistent_crossing(trace, criterion),
        "observed_events": observed_labels,
        "thermal_runaway_prediction_validated": False,
        "fire_prediction_validated": False,
    }


def compare_traces(measured: Trace, predicted: Trace, criterion: Criterion) -> dict:
    if measured.metadata["kind"] != "measured" or predicted.metadata["kind"] != "simulation":
        raise ValueError("Comparison requires measured data and a simulation")
    for field in ("cell_model", "protocol", "temperature_observable", "time_origin"):
        if measured.metadata[field] != predicted.metadata[field]:
            raise ValueError(f"Incompatible {field}; supply an explicit observation/clock mapping")
    if predicted.metadata.get("used_for_fit", True):
        raise ValueError("Declare used_for_fit=false for a held-out prediction")
    if not predicted.metadata.get("model_id") or not predicted.metadata.get("parameters_sha256"):
        raise ValueError("Prediction requires model and frozen-parameter provenance")
    t = measured.time_s
    right = np.searchsorted(predicted.time_s, t, side="left")
    indices = np.minimum(right, len(predicted.time_s) - 1)
    exact = (right < len(predicted.time_s)) & (predicted.time_s[indices] == t)
    left = np.maximum(right - 1, 0)
    inside = ((right > 0) & (right < len(predicted.time_s))
              & (predicted.time_s[indices] - predicted.time_s[left] <= criterion.max_gap_s)
              & (predicted.segment[indices] == predicted.segment[left]))
    covered = exact | inside
    if not covered.any():
        raise ValueError("No overlapping supported samples")
    interpolated = np.interp(t[covered], predicted.time_s, predicted.temperature_C)
    residual = interpolated - measured.temperature_C[covered]
    observed_event = first_persistent_crossing(measured, criterion)
    predicted_event = first_persistent_crossing(predicted, criterion)
    onset_error = None
    if observed_event["onset_s"] is not None and predicted_event["onset_s"] is not None:
        onset_error = predicted_event["onset_s"] - observed_event["onset_s"]
    return {
        "measured": summarize(measured, criterion), "predicted": summarize(predicted, criterion),
        "coverage_fraction": float(covered.mean()), "covered_samples": int(covered.sum()),
        "temperature_RMSE_C": float(np.sqrt(np.mean(residual ** 2))),
        "temperature_max_abs_error_C": float(np.max(np.abs(residual))),
        "criterion_onset_error_s": onset_error,
        "comparison_status": "complete" if covered.all() else "partial_coverage",
        "thermal_runaway_prediction_validated": False, "fire_prediction_validated": False,
        "assessment": "Temperature and specified criterion comparison only; model admission requires independent abuse validation and explicit acceptance limits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predicted", type=Path)
    parser.add_argument("--prediction-metadata", type=Path)
    parser.add_argument("--temperature-C", type=float, required=True)
    parser.add_argument("--rate-C-per-min", type=float, required=True)
    parser.add_argument("--persistence-s", type=float, required=True)
    parser.add_argument("--max-gap-s", type=float, required=True)
    parser.add_argument("--criterion-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.predicted) != bool(args.prediction_metadata):
        parser.error("Prediction CSV and metadata must be supplied together")
    criterion = Criterion(args.temperature_C, args.rate_C_per_min,
                          args.persistence_s, args.max_gap_s, args.criterion_source)
    measured = read_trace(args.measured, args.metadata, criterion.max_gap_s)
    result = summarize(measured, criterion)
    if args.predicted:
        predicted = read_trace(args.predicted, args.prediction_metadata, criterion.max_gap_s)
        result = compare_traces(measured, predicted, criterion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
