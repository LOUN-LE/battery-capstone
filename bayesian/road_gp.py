"""Chronological vehicle-level GP benchmark on KU Leuven measured road currents.

This benchmark has no driver/route labels. It deliberately does not invoke the
matched-route personal prediction API or convert different vehicles to cells.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from .driving import _fit

OFFICIAL_MD5 = "cc2d63758b411c0f8d4b61bdaa1fff1c"
SOURCE_URL = "https://rdr.kuleuven.be/api/access/datafile/272593"
DOI = "https://doi.org/10.48804/8KPDTW"
HORIZON_S = 300
GRID_S = np.arange(0, HORIZON_S + 1, 5, dtype=float)


@dataclass(frozen=True)
class RoadWindow:
    member: str
    vehicle: str
    started: datetime
    completed: datetime
    current_a: np.ndarray
    member_sha256: str
    source_rows: int


def _utc(value):
    time = datetime.fromisoformat(value)
    if time.utcoffset() is None:
        raise ValueError("timestamp_without_utc_offset")
    return time.astimezone(timezone.utc)


def parse_member(member, raw):
    """Select first 301 complete one-second samples; never fill gaps or labels."""
    match = re.search(r"/(BEV[12])_\d{4}-\d{2}-\d{2}_drive\d+\.csv$", member)
    if match is None:
        raise ValueError("unsupported_member_name")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = set(reader.fieldnames or ())
    if not {"Timestamp", "RawBattCurrent132"}.issubset(columns):
        raise ValueError("required_channels_missing")
    times, current = [], []
    for row in reader:
        try:
            stamp = _utc(row["Timestamp"])
        except (ValueError, TypeError):
            raise ValueError("invalid_timestamp") from None
        if times and stamp <= times[-1]:
            raise ValueError("nonincreasing_timestamp")
        times.append(stamp)
        # Only the predetermined forecast horizon enters feature extraction.
        if (stamp - times[0]).total_seconds() <= HORIZON_S:
            try:
                value = float(row["RawBattCurrent132"])
            except (ValueError, TypeError):
                raise ValueError("missing_current_within_horizon") from None
            if not np.isfinite(value):
                raise ValueError("nonfinite_current_within_horizon")
            current.append(value)
    if not times or (times[-1] - times[0]).total_seconds() < HORIZON_S:
        raise ValueError("shorter_than_300_seconds")
    elapsed = np.array([(t - times[0]).total_seconds() for t in times if (t - times[0]).total_seconds() <= HORIZON_S])
    if len(elapsed) != HORIZON_S + 1 or not np.array_equal(elapsed, np.arange(HORIZON_S + 1)):
        raise ValueError("noncontinuous_one_second_horizon")
    values = np.asarray(current, dtype=float)[GRID_S.astype(int)]
    values.setflags(write=False)
    return RoadWindow(member, match[1], times[0], times[-1], values,
                      hashlib.sha256(raw).hexdigest(), len(times)), columns


def load_archive(path):
    path = Path(path)
    with path.open("rb") as source:
        md5 = hashlib.file_digest(source, "md5").hexdigest()
    if md5 != OFFICIAL_MD5:
        raise ValueError("Official KU Leuven V2 ZIP MD5 mismatch")
    windows, exclusions, headers = [], [], set()
    with zipfile.ZipFile(path) as archive:
        members = sorted(n for n in archive.namelist() if "/driving sessions/" in n and n.endswith(".csv"))
        for name in members:
            raw = archive.read(name)  # zipfile checks this member's CRC.
            try:
                window, columns = parse_member(name, raw)
                windows.append(window)
                headers.update(columns)
            except ValueError as error:
                exclusions.append({"member": name, "reason": str(error), "member_sha256": hashlib.sha256(raw).hexdigest()})
    return windows, {"zip_path": str(path.resolve()), "zip_bytes": path.stat().st_size,
                     "zip_md5": md5, "source_url": SOURCE_URL, "dataset_doi": DOI,
                     "license": "CC BY 4.0", "driving_members": len(members),
                     "eligible_windows": len(windows), "exclusions": exclusions,
                     "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
                     "available_headers": sorted(headers)}


def _score(errors):
    values = np.asarray(errors, dtype=float)
    return {"rmse_a": float(np.sqrt(np.mean(values**2))), "mae_a": float(np.mean(abs(values)))}


def forecast_window(history, target, *, recent_count=5):
    """One forecast made before observing the target's current waveform."""
    if type(recent_count) is not int or recent_count < 2:
        raise ValueError("recent_count must be an integer >= 2")
    previous = sorted((w for w in history if w.vehicle == target.vehicle and w.completed <= target.started
                       and w.member != target.member), key=lambda w: (w.started, w.member))
    if len({w.member for w in previous}) != len(previous):
        raise ValueError("Duplicate historical session")
    if len(previous) < recent_count + 2:
        raise ValueError("Need recent histories and at least two older completed sessions")
    recent = previous[-recent_count:]
    older = previous[:-recent_count]
    recent_values = np.stack([w.current_a for w in recent])
    older_values = np.stack([w.current_a for w in older])
    if (recent_values.shape[1:] != GRID_S.shape or older_values.shape[1:] != GRID_S.shape
            or not np.isfinite(recent_values).all() or not np.isfinite(older_values).all()):
        raise ValueError("Finite fixed-grid historical pack currents required")
    scaled_distance = np.sqrt(3) * abs(GRID_S[:, None] - GRID_S[None, :]) / (0.08 * HORIZON_S)
    kernel = (1 + scaled_distance) * np.exp(-scaled_distance)
    _, mean, covariance = _fit(recent_values, older_values, kernel)
    interval_half = 1.64485362695 * np.sqrt(np.maximum(np.diag(covariance), 0))
    return {"gp": mean, "interval_lower": mean - interval_half, "interval_upper": mean + interval_half,
            "recent_mean": recent_values.mean(axis=0),
            "all_past_mean": np.stack([w.current_a for w in previous]).mean(axis=0),
            "last_trip": previous[-1].current_a.copy(),
            "training_members": [w.member for w in previous],
            "recent_members": [w.member for w in recent],
            "older_members": [w.member for w in older]}


def evaluate(windows, *, test_fraction=0.2, recent_count=5):
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    result = {}
    for vehicle in sorted({w.vehicle for w in windows}):
        cohort = sorted((w for w in windows if w.vehicle == vehicle), key=lambda w: (w.started, w.member))
        if len({w.member for w in cohort}) != len(cohort):
            raise ValueError("Duplicate source member")
        number_test = max(1, math.ceil(len(cohort) * test_fraction))
        targets = cohort[-number_test:]
        folds, skipped = [], []
        errors = {key: [] for key in ("gp", "recent_mean", "all_past_mean", "last_trip")}
        coverages = []
        for target in targets:
            try:
                prediction = forecast_window(cohort, target, recent_count=recent_count)
            except ValueError as error:
                skipped.append({"member": target.member, "reason": str(error)})
                continue
            observed = target.current_a
            if observed.shape != GRID_S.shape or not np.isfinite(observed).all():
                raise ValueError("Finite fixed-grid target required")
            fold = {"member": target.member, "forecast_origin_utc": target.started.isoformat(),
                    "completed_utc": target.completed.isoformat(),
                    "training_members": prediction["training_members"],
                    "recent_members": prediction["recent_members"],
                    "older_members": prediction["older_members"], "scores": {},
                    "observed_current_a": observed.tolist()}
            for key in errors:
                err = prediction[key] - observed
                errors[key].append(err)
                fold["scores"][key] = _score(err)
                fold[key + "_current_a"] = prediction[key].tolist()
            inside = (observed >= prediction["interval_lower"]) & (observed <= prediction["interval_upper"])
            coverages.append(inside)
            fold["gp_interval_90_coverage"] = float(inside.mean())
            fold["interval_lower_a"] = prediction["interval_lower"].tolist()
            fold["interval_upper_a"] = prediction["interval_upper"].tolist()
            folds.append(fold)
        metrics = {key: _score(np.stack(value)) for key, value in errors.items()} if folds else {}
        improvements = {key: 100*(metrics[key]["rmse_a"]-metrics["gp"]["rmse_a"])/metrics[key]["rmse_a"]
                        if metrics[key]["rmse_a"] else None
                        for key in errors if key != "gp"} if folds else {}
        result[vehicle] = {"eligible_sessions": len(cohort), "requested_test_sessions": number_test,
                           "initial_training_sessions": len(cohort)-number_test,
                           "evaluated_test_sessions": len(folds), "grid_points_per_session": len(GRID_S),
                           "metrics": metrics, "gp_rmse_improvement_percent_vs": improvements,
                           "gp_interval_90_coverage": float(np.mean(coverages)) if folds else None,
                           "skipped_test_sessions": skipped, "folds": folds}
    return result


def run(zip_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    windows, source = load_archive(zip_path)
    results = evaluate(windows)
    report = {"schema": "road_pack_current_gp_v1", "status": "complete", "source": source,
              "scope": "measured road data; vehicle-level unconditional session-start pack-current forecast",
              "not_claimed": ["driver-specific prediction", "same-route or traffic-conditioned prediction",
                              "M50T cell-current transfer", "battery life prediction"],
              "units": "pack amperes; source RawBattCurrent132 sign retained, discharge positive and regeneration negative",
              "protocol": {"horizon_seconds": HORIZON_S, "grid_seconds": GRID_S.tolist(),
                           "selection": "source driving sessions with first 301 timestamps exactly one-second continuous and finite current; no filling",
                           "target_population_limit": "sessions lasting at least 300 seconds with a complete first 300-second window; shorter/incomplete sessions excluded retrospectively",
                           "split": "chronological final 20 percent separately per vehicle; rolling completed-history updates",
                           "recent_count": 5, "minimum_older_histories": 2,
                           "past_only_rule": "entire historical session must end at or before forecast origin",
                           "gp": "existing functional GP _fit: recent five vehicle windows vs older windows; fixed Matern kernel length 24 s",
                           "hyperparameter_search": "none; choices fixed before viewing evaluation outcomes",
                           "score": "equal-weight errors at 61 exact 5-second nodes; not high-frequency peak accuracy",
                           "uncertainty": "90 percent marginal pointwise interval; not simultaneous whole-curve coverage"},
              "vehicles": results,
              "eligible_manifest": [{"member": w.member, "vehicle": w.vehicle,
                                     "started_utc": w.started.isoformat(), "completed_utc": w.completed.isoformat(),
                                     "member_sha256": w.member_sha256, "source_rows": w.source_rows} for w in windows]}
    for name, path in (("road_gp", Path(__file__)), ("functional_gp", Path(__file__).with_name("driving.py"))):
        report.setdefault("code_sha256", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(output / "eligible_current_windows.npz", grid_seconds=GRID_S,
                        current_a=np.stack([w.current_a for w in windows]),
                        members=np.array([w.member for w in windows]))
    return report


evaluate_archive = run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.zip, args.output)
    print(json.dumps({key: {k: v for k, v in value.items() if k != "folds"}
                      for key, value in result["vehicles"].items()}, ensure_ascii=False, indent=2))
