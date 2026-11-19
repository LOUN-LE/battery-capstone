"""Score D-cell RPT0/RPT1 capacity and C/10 curves without solving or fitting.

score_rpts({0: curve, 1: curve}, output_dir) accepts each curve as a dict with
time_s, current_A (discharge positive), voltage_V, discharge_capacity_Ah,
termination, and independent_cutoff=True. A dict may instead contain ``path``
to an NPZ plus the endpoint metadata. PyBaMM variable-name aliases are accepted.
No metric in this module establishes a degradation mechanism or full validity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from os import PathLike
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPOSITORY / "source/imperial"
DEFAULT_CRITERIA = REPOSITORY / "configs/empirical_acceptance.json"
FIELDS = {
    "time_s": ("time_s", "Time [s]", "t"),
    "current_A": ("current_A", "Current [A]", "I"),
    "voltage_V": ("voltage_V", "Voltage [V]", "V", "v"),
    "discharge_capacity_Ah": ("discharge_capacity_Ah", "Discharge capacity [A.h]", "Q_Ah", "q"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integral(y, x):
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2))


def _clean_curve(curve):
    result = {}
    for name, aliases in FIELDS.items():
        key = next((k for k in aliases if k in curve), None)
        if key is None:
            raise ValueError(f"Missing curve field {name}")
        result[name] = np.asarray(curve[key], dtype=float).reshape(-1)
    sizes = {len(a) for a in result.values()}
    if len(sizes) != 1 or next(iter(sizes)) < 3:
        raise ValueError("Curve arrays must have the same length and at least three samples")
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError("Nonfinite curve sample")
    t, q = result["time_s"], result["discharge_capacity_Ah"]
    if np.any(np.diff(t) <= 0) or np.any(np.diff(q) <= 0):
        raise ValueError("Discharge time and cumulative Ah must increase strictly; no silent sorting/deduplication")
    if np.any(result["current_A"] <= 0):
        raise ValueError("Expected a discharge-only curve with positive current")
    result["discharge_capacity_Ah"] = q - q[0]
    return result


def _curve_stats(curve):
    t, i, q, v = (curve[k] for k in ("time_s", "current_A", "discharge_capacity_Ah", "voltage_V"))
    dt = np.diff(t)
    return {"rows": len(t), "duration_s": float(t[-1] - t[0]),
            "first_voltage_V": float(v[0]), "last_voltage_V": float(v[-1]),
            "current_min_A": float(i.min()), "current_max_A": float(i.max()),
            "capacity_first_to_last_Ah": float(q[-1]),
            "current_integral_first_to_last_Ah": _integral(i, t) / 3600,
            "capacity_minus_current_integral_Ah": float(q[-1]) - _integral(i, t) / 3600,
            "sample_spacing_s": {"minimum": float(dt.min()), "median": float(np.median(dt)), "maximum": float(dt.max())}}


def load_empirical(source_dir=DEFAULT_SOURCE, summary_path=None):
    """Read the exact local D summary, existing RPT0 export, and raw RPT1 Ns5."""
    source_dir = Path(source_dir)
    summary_path = Path(summary_path or source_dir / "kirkaldy_expt2_cellD_performance_summary.csv")
    p0 = source_dir / "kirkaldy_expt2_cellD_RPT0_C10.csv"
    raw = source_dir / "raw_protocol/NDK - LG M50 deg - exp 2,2 - rig 3 - 25degC - cell D - RPT1_01_MB_CA5.mpt"
    source_files = [{"path": str(p.resolve()), "sha256": digest(p)} for p in (summary_path, p0, raw)]
    with summary_path.open(encoding="utf-8-sig", newline="") as f:
        summary = {int(float(r["Ageing Sets"])): r for r in csv.DictReader(f)}
    with p0.open(encoding="utf-8-sig", newline="") as f:
        rows0 = list(csv.DictReader(f))
    rows1, line_numbers, outside_after_start = [], [], False
    with raw.open(encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for line, row in enumerate(reader, 2):
            selected = row["Ns"] == "5" and row["mode"] == "1"
            if selected:
                if outside_after_start:
                    raise ValueError("Multiple separated Ns5/mode1 runs; explicit run selection required")
                if float(row["error"]) != 0 or float(row["I/mA"]) >= 0:
                    raise ValueError("RPT1 discharge includes an instrument error or wrong current direction")
                rows1.append(row)
                line_numbers.append(line)
            elif rows1:
                outside_after_start = True
    curves = {
        0: _clean_curve({"time_s": [r["Time (s)"] for r in rows0],
            "current_A": [-float(r["Current (mA)"]) / 1000 for r in rows0],
            "voltage_V": [r["Voltage (V)"] for r in rows0],
            "discharge_capacity_Ah": [float(r["Charge (mA.h)"]) / 1000 for r in rows0]}),
        1: _clean_curve({"time_s": [r["time/s"] for r in rows1],
            "current_A": [-float(r["I/mA"]) / 1000 for r in rows1],
            "voltage_V": [r["Ecell/V"] for r in rows1],
            "discharge_capacity_Ah": [float(r["Q discharge/mA.h"]) / 1000 for r in rows1]})}
    if any(digest(entry["path"]) != entry["sha256"] for entry in source_files):
        raise ValueError("Empirical source changed while it was being read")
    evidence = {"cell_id": "D", "experiment": "Imperial Expt 2,2; LG M50T; nominal 25 C",
        "source_files": source_files,
        "summary_rpt_ids": sorted(summary),
        "RPT1_selection": {"Ns": 5, "mode": 1, "source_line_first": line_numbers[0],
            "source_line_last": line_numbers[-1], "rows": len(rows1), "current_conversion": "-I/mA / 1000"},
        "curve_capacity_origin": "Subtract first retained discharge sample; do not invent unrecorded onset or cutoff sample",
        "raw_cutoff_policy": "Last retained loaded sample is reported as observed; no interpolation/extrapolation to an unrecorded 2.5 V crossing",
        "uncertainty": {"instrument_accuracy": "not supplied to this scorer", "repeat_cell_confidence_interval": "not estimated",
            "summary_vs_raw_difference": "reported as extraction/integration consistency; not a statistical confidence interval"}}
    return summary, curves, evidence


def _model_curve(spec):
    spec = {"path": spec} if isinstance(spec, (str, PathLike)) else dict(spec)
    source = None
    if "path" in spec:
        p = Path(spec["path"])
        with np.load(p, allow_pickle=False) as z:
            loaded = {k: z[k] for k in z.files}
        loaded.update({k: v for k, v in spec.items() if k != "path"})
        spec = loaded
        source = {"path": str(p.resolve()), "sha256": digest(p)}
    termination = str(np.asarray(spec.get("termination", "")).item())
    independent = np.asarray(spec.get("independent_cutoff", False)).item() is True
    if not independent or "Voltage < 2.5 [V]" not in termination:
        raise ValueError("Model RPT needs independent_cutoff=True and the native Voltage < 2.5 [V] endpoint, not observed-duration termination")
    curve = _clean_curve(spec)
    # Numerical endpoint sanity, not an empirical accuracy acceptance threshold.
    if abs(curve["voltage_V"][-1] - 2.5) > 1e-4:
        raise ValueError("Native endpoint metadata conflicts with terminal model voltage")
    charge_residual = _curve_stats(curve)["capacity_minus_current_integral_Ah"]
    charge_tolerance = 1e-7 + 1e-4 * curve["discharge_capacity_Ah"][-1]
    if abs(charge_residual) > charge_tolerance:
        raise ValueError("Model discharged capacity disagrees with the current integral")
    array_hash = hashlib.sha256()
    for key in FIELDS:
        array_hash.update(key.encode("utf-8"))
        array_hash.update(np.asarray(curve[key], dtype="<f8").tobytes())
    return curve, {"source": source, "termination": termination, "independent_cutoff_declared": independent,
                   "endpoint_voltage_numerical_check_tolerance_V": 1e-4,
                   "charge_integral_numerical_check_tolerance_Ah": float(charge_tolerance),
                   "normalized_curve_arrays_sha256": array_hash.hexdigest()}


def curve_metrics(model, observed, model_denominator_Ah, observed_denominator_Ah, *, grid_points=1001):
    """Uniform normalized-Ah weighting on strict intersection, never extrapolation."""
    if not np.isfinite([model_denominator_Ah, observed_denominator_Ah]).all() or min(model_denominator_Ah, observed_denominator_Ah) <= 0:
        raise ValueError("Positive finite normalization capacities required")
    mx = model["discharge_capacity_Ah"] / model_denominator_Ah
    ox = observed["discharge_capacity_Ah"] / observed_denominator_Ah
    lo, hi = max(0., float(mx[0]), float(ox[0])), min(1., float(mx[-1]), float(ox[-1]))
    valid = hi > lo
    result = {"status": "scored_on_overlap" if valid else "not_evaluated_no_overlap",
              "axis": "discharged_Ah / stated_normalization_capacity_Ah",
              "model_denominator_Ah": model_denominator_Ah, "observed_denominator_Ah": observed_denominator_Ah,
              "requested_interval": [0., 1.], "model_domain": [float(mx[0]), float(mx[-1])],
              "observed_domain": [float(ox[0]), float(ox[-1])], "valid_interval": [lo, hi] if valid else None,
              "valid_fraction_of_requested_span": max(0., hi-lo), "extrapolation": False,
              "capacity_and_duration_mismatch_removed_by_this_axis": model_denominator_Ah != observed_denominator_Ah,
              "acceptance_threshold_mV": None, "physical_validation_complete": False}
    if not valid:
        return result, []
    if grid_points < 3:
        raise ValueError("At least three comparison-grid points required")
    x = np.linspace(lo, hi, grid_points)
    mv, ov = np.interp(x, mx, model["voltage_V"]), np.interp(x, ox, observed["voltage_V"])
    error = (mv - ov) * 1000
    result.update(grid_points=grid_points, weighting="uniform normalized discharged Ah; trapezoidal integral",
        RMSE_mV=float(np.sqrt(_integral(error**2, x)/(hi-lo))),
        MAE_mV=_integral(abs(error), x)/(hi-lo), bias_mV=_integral(error, x)/(hi-lo),
        max_abs_error_mV=float(abs(error).max()),
        offset_removed_RMSE_mV=float(np.sqrt(_integral((error-error[0])**2, x)/(hi-lo))),
        removed_initial_voltage_bias_mV=float(error[0]))
    points = [{"normalized_discharge": float(a), "model_voltage_V": float(b),
               "observed_voltage_V": float(c), "residual_mV": float(d)} for a, b, c, d in zip(x, mv, ov, error)]
    return result, points


def score_rpts(model_rpts, output_dir=None, empirical_summary=None, raw_reference_dir=None, criteria_path=None):
    """Return a JSON-safe scoring report; optionally write report and aligned CSVs."""
    models = {int(k): _model_curve(v) for k, v in model_rpts.items()}
    if set(models) != {0, 1}:
        raise ValueError("This limited comparison requires exactly RPT0 and RPT1")
    summary, observed, evidence = load_empirical(raw_reference_dir or DEFAULT_SOURCE, empirical_summary)
    criteria_path = Path(criteria_path or DEFAULT_CRITERIA)
    criteria = json.loads(criteria_path.read_text(encoding="utf-8-sig"))
    evidence["criteria"] = {"path": str(criteria_path.resolve()), "sha256": digest(criteria_path), "criteria_id": criteria["criteria_id"]}
    evidence["scorer"] = {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)}
    mq = {k: float(models[k][0]["discharge_capacity_Ah"][-1]) for k in models}
    oq = {k: float(summary[k]["C/10 Capacity [mA h]"])/1000 for k in models}
    rpt_records, aligned = [], {}
    for k in (0, 1):
        curve, meta = models[k]
        shape, shape_rows = curve_metrics(curve, observed[k], mq[k], oq[k])
        common, common_rows = curve_metrics(curve, observed[k], oq[0], oq[0])
        raw_stats = _curve_stats(observed[k])
        rpt_records.append({"rpt_id": k, "model": {**meta, **_curve_stats(curve)}, "observed_raw": raw_stats,
            "measured_summary_capacity_Ah": oq[k], "capacity_error_Ah": mq[k]-oq[k],
            "capacity_error_percent_of_measured": 100*(mq[k]/oq[k]-1),
            "summary_minus_raw_first_to_last_capacity_Ah": oq[k]-raw_stats["capacity_first_to_last_Ah"],
            "voltage_at_own_normalized_capacity": shape, "voltage_at_common_measured_RPT0_capacity_axis": common})
        aligned[f"RPT{k}_own_normalized_capacity"] = shape_rows
        aligned[f"RPT{k}_common_measured_RPT0_capacity"] = common_rows
    model_soh = np.asarray([100*mq[k]/mq[0] for k in (0, 1)])
    observed_soh = np.asarray([100*oq[k]/oq[0] for k in (0, 1)])
    errors = model_soh-observed_soh
    rmse, maximum = float(np.sqrt(np.mean(errors**2))), float(abs(errors).max())
    diagnostic_limits = {name: {"SOH_RMSE_pp_max": criteria[name]["SOH_RMSE_pp_max"],
        "SOH_max_abs_error_pp_max": criteria[name]["SOH_max_abs_error_pp_max"],
        "two_point_values_within_limits": rmse <= criteria[name]["SOH_RMSE_pp_max"] and maximum <= criteria[name]["SOH_max_abs_error_pp_max"]}
        for name in ("basic", "challenge")}
    report = {"status": "partial_RPT0_RPT1_metric_scoring_complete", "physical_validation_complete": False,
        "degradation_cause_validated": False, "lifetime_validated": False, "evidence": evidence,
        "RPTs": rpt_records, "capacity_change": {"model_drop_Ah": mq[0]-mq[1], "observed_drop_Ah": oq[0]-oq[1],
            "drop_error_Ah": (mq[0]-mq[1])-(oq[0]-oq[1]),
            "model_drop_percent_of_own_RPT0": 100-model_soh[-1], "observed_drop_percent_of_own_RPT0": 100-observed_soh[-1],
            "drop_error_percentage_points": float(observed_soh[-1]-model_soh[-1]),
            "model_SOH_percent": model_soh.tolist(), "observed_SOH_percent": observed_soh.tolist(),
            "SOH_errors_pp": errors.tolist(), "SOH_RMSE_pp": rmse, "SOH_max_abs_error_pp": maximum,
            "post_baseline_only_SOH_RMSE_pp": float(abs(errors[1])),
            "RPT0_is_normalized_to_100_by_definition": True, "post_baseline_observation_count": 1},
        "baseline_screening": {"scope": "capacity-only two-point numerical screen, not full-curve acceptance",
            "status": "within_historical_numeric_limits_on_two_points" if diagnostic_limits["basic"]["two_point_values_within_limits"] else "exceeds_historical_numeric_limits_on_two_points",
            "capacity_only_basic_limits": diagnostic_limits["basic"],
            "voltage_acceptance": "not_evaluated_no_preagreed_threshold",
            "full_validation_passed": False, "post_baseline_RPT_count": 1,
            "uncertainty_quantified": False},
        "historical_acceptance": {"numeric_limits_diagnostic_only": diagnostic_limits,
            "formal_full_curve_acceptance": "not_evaluated_partial_curve", "challenge_required": False,
            "observed_total_RPT_count": len(summary), "scored_RPT_count": 2,
            "remaining_RPT_ids_not_scored": sorted(set(summary)-{0, 1}), "EOL": "not_evaluated_no_observed_80_percent_crossing_in_this_two_point_comparison",
            "voltage_shape_threshold": None, "fresh_heldout_claim": False},
        "limitations": ["One observed ageing interval; RPT0 contributes a zero SOH residual by normalization, not an independent successful prediction.",
            "Normalizing each voltage curve by its own total capacity hides capacity/elapsed-time mismatch; absolute capacities and a common-Ah axis are also reported.",
            "The raw loaded discharge samples do not include an asserted exact 2.5 V crossing; the summary capacity is the measured capacity reference.",
            "D has been examined previously. No new blinded validation or parameter freeze chronology is certified here.",
            "Instrument uncertainty and between-cell variability are not estimated; dense voltage samples are not independent replicate cells.",
            "No degradation-mode inverse fitting is performed. Published DMA estimates are not direct mechanism ground truth."]}
    for entry in evidence["source_files"] + [evidence["criteria"], evidence["scorer"]]:
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Scoring source changed during execution")
    for _, metadata in models.values():
        source = metadata["source"]
        if source is not None and digest(source["path"]) != source["sha256"]:
            raise ValueError("Model curve file changed during execution")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        targets = [out/"d_long_score.json"] + [out/(key+".csv") for key in aligned]
        if any(p.exists() for p in targets):
            raise FileExistsError("Scoring artifacts already exist; choose a new output directory")
        for key, rows in aligned.items():
            if rows:
                with (out/(key+".csv")).open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
        (out/"d_long_score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _self_test():
    curve = _clean_curve({"time_s": [0, 1, 2], "current_A": [1, 1, 1],
                          "voltage_V": [4, 3, 2.5], "discharge_capacity_Ah": [0, 1/3600, 2/3600]})
    same, _ = curve_metrics(curve, curve, 2/3600, 2/3600)
    assert same["RMSE_mV"] == 0 and same["valid_fraction_of_requested_span"] == 1
    partial, points = curve_metrics(curve, curve, 4/3600, 2/3600)
    assert partial["valid_fraction_of_requested_span"] == .5 and points[-1]["normalized_discharge"] == .5
    try:
        _model_curve({**curve, "termination": "final time", "independent_cutoff": True})
    except ValueError:
        pass
    else:
        raise AssertionError("Observed-duration endpoint was incorrectly admitted")
    try:
        _clean_curve({**curve, "time_s": [0, 1, 1]})
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate timestamps were incorrectly admitted")
    print("Four scoring/overlap/endpoint/input checks passed; no physics solver executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path, help="JSON containing model_rpts mapping; array dictionaries or NPZ descriptors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    elif args.manifest and args.output:
        request = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        result = score_rpts(request["model_rpts"], args.output, request.get("empirical_summary"), request.get("raw_reference_dir"), request.get("criteria_path"))
        print(json.dumps({"status": result["status"], "capacity_change": result["capacity_change"]}, ensure_ascii=False))
    else:
        parser.error("Use --self-test or --manifest plus --output")
