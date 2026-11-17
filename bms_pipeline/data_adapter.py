"""Read-only source adapter for two Imperial LG M50T initial low-rate tests.

Canonical CSVs are machine inputs, not reconstructed ageing states. Source times
remain in their original clock coordinates; a consumer must subtract its chosen
start explicitly. No measured temperature is substituted for ambient temperature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import unittest
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "source"
FIELDS = ("cell_id", "session_id", "time_s", "voltage_V", "current_A",
          "surface_temperature_C", "source_row", "phase")
SOURCE_FIELDS = ("Time (s)", "Voltage (V)", "Current (mA)",
                 "Charge (mA.h)", "Temperature (degC)")
MISSING = {"", "nan", "na", "n/a", "null"}
SOURCE_URL = "https://zenodo.org/records/10637534"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value, name):
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not number.is_finite():
        raise ValueError(f"Nonfinite {name}")
    return number


def surface_value(value):
    text = str(value).strip()
    if text.lower() in MISSING:
        return ""
    if finite(text, "surface temperature") <= Decimal("-273.15"):
        raise ValueError("Surface temperature below absolute zero")
    return text


def convert_records(records, cell_id, session_id):
    """Preserve source strings except current unit/sign and missing markers."""
    output = []
    previous = None
    for line, record in enumerate(records, start=2):
        stamp = finite(record["Time (s)"], "time")
        voltage = finite(record["Voltage (V)"], "voltage")
        current = -finite(record["Current (mA)"], "current") / Decimal(1000)
        finite(record["Charge (mA.h)"], "instrument charge")
        if previous is not None and stamp <= previous:
            raise ValueError(f"Time is not strictly increasing at source line {line}")
        if voltage <= 0 or current <= 0:
            raise ValueError(f"Expected positive voltage and discharge current at line {line}")
        output.append(dict(zip(FIELDS, (
            cell_id, session_id, record["Time (s)"].strip(),
            record["Voltage (V)"].strip(), format(current, "f"),
            surface_value(record["Temperature (degC)"]), line,
            "initial_low_rate_discharge"))))
        previous = stamp
    if len(output) < 2:
        raise ValueError("At least two source observations required")
    return output


def select_prefix(rows, seconds):
    limit = finite(seconds, "prefix duration")
    if limit <= 0:
        raise ValueError("Prefix duration must be positive")
    origin = finite(rows[0]["time_s"], "time")
    selected = [row for row in rows if finite(row["time_s"], "time") - origin <= limit]
    if len(selected) < 2:
        raise ValueError("Prefix contains fewer than two observations")
    return selected


def describe(rows):
    stamps = [finite(row["time_s"], "time") for row in rows]
    steps = [b-a for a, b in zip(stamps, stamps[1:])]
    result = {
        "rows": len(rows), "source_row_first": rows[0]["source_row"],
        "source_row_last": rows[-1]["source_row"],
        "source_time_origin_s": rows[0]["time_s"],
        "source_time_end_s": rows[-1]["time_s"],
        "observed_duration_s": str(stamps[-1]-stamps[0]),
        "sample_interval_s": {"minimum": float(min(steps)),
                              "median": float(statistics.median(steps)),
                              "maximum": float(max(steps))},
        "missing_temperature_source_rows": [r["source_row"] for r in rows
                                            if r["surface_temperature_C"] == ""],
    }
    for name in ("voltage_V", "current_A", "surface_temperature_C"):
        values = [float(row[name]) for row in rows if row[name] != ""]
        result[name] = {"minimum": min(values) if values else None,
                        "maximum": max(values) if values else None,
                        "first": float(rows[0][name]) if rows[0][name] else None,
                        "last": float(rows[-1][name]) if rows[-1][name] else None}
    result["trapezoidal_discharge_Ah"] = math.fsum(
        float(dt) * (float(a["current_A"])+float(b["current_A"])) / 7200
        for dt, a, b in zip(steps, rows, rows[1:]))
    return result


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream))
    if restored != [{k: str(row[k]) for k in FIELDS} for row in rows]:
        raise AssertionError("Canonical CSV did not preserve exact field values")
    return {"path": str(Path(path).resolve()), "sha256": digest(path), **describe(rows)}


def preparation_from_raw(path, target, cell_id):
    """Extract actual preceding observations; match the first discharge sample."""
    prepared, phases, match = [], [], None
    with Path(path).open(encoding="cp1252", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        temperature_keys = [key for key in reader.fieldnames if key.startswith("Temperature/")]
        if len(temperature_keys) != 1:
            raise ValueError("Ambiguous raw temperature field")
        temperature_key = temperature_keys[0]
        for line, row in enumerate(reader, start=2):
            stamp = finite(row["time/s"], "raw time")
            difference = abs(stamp-finite(target["time_s"], "processed time"))
            if difference < Decimal("0.0000001"):
                if (abs(finite(row["Ecell/V"], "raw voltage")-finite(target["voltage_V"], "voltage")) > Decimal("0.0000001")
                        or abs(-finite(row["I/mA"], "raw current")/1000-finite(target["current_A"], "current")) > Decimal("0.000000001")
                        or row["mode"] != "1" or row["Ns"] != "5"):
                    raise ValueError("Processed first sample does not match the raw discharge")
                match = {"raw_source_line": line, "processed_source_line": target["source_row"],
                         "time_difference_s": str(difference), "voltage_and_current_match": True}
                break
            if stamp > finite(target["time_s"], "target time"):
                raise ValueError("Raw file passed the processed start without a match")
            mode = {"1": "constant_current", "2": "constant_voltage", "3": "rest"}.get(row["mode"])
            if mode is None:
                raise ValueError("Unsupported preparation control mode")
            phase = f"preparation_step_{row['Ns']}_{mode}"
            current = -finite(row["I/mA"], "raw current")/Decimal(1000)
            record = dict(zip(FIELDS, (cell_id, f"{cell_id}_initial_preparation",
                                     row["time/s"], row["Ecell/V"], format(current, "f"),
                                     surface_value(row[temperature_key]), line, phase)))
            if prepared and stamp <= finite(prepared[-1]["time_s"], "previous raw time"):
                raise ValueError("Non-increasing preparation time")
            prepared.append(record)
            if not phases or phases[-1]["phase"] != phase:
                phases.append({"phase": phase, "first_source_line": line, "time_start_s": row["time/s"]})
            phases[-1].update(last_source_line=line, time_end_s=row["time/s"],
                              current_end_A=format(current, "f"), voltage_end_V=row["Ecell/V"],
                              control_end=row["control/V/mA"])
    if match is None or len(prepared) < 2:
        raise ValueError("Raw preparation is not matched to the processed test")
    return prepared, {"match": match, "phases": phases,
                      "boundary_warning": "A final rest-labelled row may already carry the next load command/voltage; do not infer resting voltage from that row.",
                      "initial_soc": None,
                      "initial_soc_basis": "CC charge followed by approximately 4.2 V CV, about 2 h plus 30 s rest supports a nominal fully charged protocol reference; exact electrode states/SOC are not measured.",
                      "ageing_history": "Initial reference test after source break-in; not a pristine manufactured cell and not reconstructed complete prior ageing history."}


def build(source_dir=DEFAULT_SOURCE, output_dir=HERE, prefix_seconds=600):
    started = time.perf_counter()
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output must be outside the immutable source folder")
    destination = output_dir / "data"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema": "imperial-m50t-measured-input-v1", "source_url": SOURCE_URL,
              "adapter_sha256": digest(__file__), "columns": list(FIELDS),
              "source_row_definition": "One-based physical CSV/TSV line, with header on line 1; each file's provenance identifies its own source.",
              "time_policy": "Preserve source decimal text; no resampling, rounding, rebasing, synthetic endpoint or gap filling.",
              "current_policy": "Source positive charge/negative discharge in mA; canonical current_A = -source_mA/1000.",
              "temperature_policy": "Measured cell surface temperature; blanks remain blanks and missing line numbers are reported. Not ambient or validated volume-average temperature.",
              "prefix_policy": "Include original samples at or before the requested duration from the first source sample; do not interpolate an endpoint.",
              "physical_model_validation_completed": False, "cells": {}}
    merged_prefix = []
    for cell, channel in (("C", "CA4"), ("D", "CA5")):
        cell_id = f"imperial_expt2_2_cell_{cell}"
        source = source_dir / f"kirkaldy_expt2_cell{cell}_RPT0_C10.csv"
        raw = source_dir / "raw_protocol" / f"NDK - LG M50 deg - exp 2,2 - rig 3 - 25degC - cell {cell} - BoL - RPT0_long_01_MB_{channel}.mpt"
        original_hashes = {str(path): digest(path) for path in (source, raw)}
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames) != SOURCE_FIELDS:
                raise ValueError("Unexpected source columns")
            rows = convert_records(reader, cell_id, f"{cell_id}_initial_low_rate")
        preparation, context = preparation_from_raw(raw, rows[0], cell_id)
        files = {}
        prefix = select_prefix(rows, prefix_seconds)
        merged_prefix.extend(prefix)
        for key, selected in (("full", rows), ("prefix", prefix),
                              ("preparation", preparation)):
            files[key] = write_csv(destination / f"cell_{cell}_{key}.csv", selected)
            files[key]["source_path"] = str(raw if key == "preparation" else source)
        for path, expected in original_hashes.items():
            if digest(path) != expected:
                raise AssertionError("Source mutated during extraction")
        report["cells"][cell] = {"cell_id": cell_id, "source_sha256": original_hashes,
                                 "requested_prefix_s": prefix_seconds, "initial_context": context,
                                 "files": files}
    merged_path = destination / "initial_low_rate_prefix.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged_prefix)
    with merged_path.open(encoding="utf-8", newline="") as stream:
        merged_read = list(csv.DictReader(stream))
    if merged_read != [{k: str(row[k]) for k in FIELDS} for row in merged_prefix]:
        raise AssertionError("Combined sessions did not survive CSV round-trip")
    report["combined_prefix"] = {
        "path": str(merged_path), "sha256": digest(merged_path),
        "rows": len(merged_prefix), "session_count": 2,
        "ordering": "C session then D session. Independent time axes: group by cell_id/session_id before calculating elapsed time or integrating current.",
        "source_mapping": "Resolve source_row against the processed source_path for the corresponding cell in cells.C/D.files.prefix."}
    report["adapter_checks"] = run_self_tests()
    report["wall_seconds"] = time.perf_counter()-started
    (output_dir / "data_provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


class AdapterTests(unittest.TestCase):
    def fixture(self):
        return [dict(zip(SOURCE_FIELDS, (str(t), "4.1", "-499.9938", "0.1", temp)))
                for t, temp in (("17111.306812743416", "24.385273"),
                                ("17112.306812790917", "NaN"),
                                ("17113.306812838418", "24.4"))]

    def test_decimal_clock_current_and_missing_preserved(self):
        rows = convert_records(self.fixture(), "C", "initial")
        self.assertEqual(rows[0]["time_s"], "17111.306812743416")
        self.assertEqual(rows[0]["current_A"], "0.4999938")
        self.assertEqual(rows[1]["surface_temperature_C"], "")
        self.assertEqual(describe(rows)["missing_temperature_source_rows"], [3])

    def test_duplicate_time_and_wrong_current_rejected(self):
        original = self.fixture()
        original[1]["Time (s)"] = original[0]["Time (s)"]
        with self.assertRaises(ValueError):
            convert_records(original, "C", "initial")
        original = self.fixture()
        original[1]["Current (mA)"] = "500"
        with self.assertRaises(ValueError):
            convert_records(original, "C", "initial")

    def test_prefix_uses_real_samples_without_interpolation(self):
        rows = convert_records(self.fixture(), "C", "initial")
        selected = select_prefix(rows, "1.5")
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[-1]["source_row"], 3)
        self.assertEqual(selected[-1]["time_s"], rows[1]["time_s"])


def run_self_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)
    result = unittest.TestResult()
    suite.run(result)
    if not result.wasSuccessful():
        raise AssertionError(str(result.errors+result.failures))
    return {"tests_run": result.testsRun, "passed": True,
            "coverage": "Exact clock/sign conversion, missing temperature preservation, invalid time/current rejection, source-only prefix selection and CSV round-trip."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--prefix-seconds", type=float, default=600)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests()))
    else:
        result = build(args.source_dir, args.output_dir, args.prefix_seconds)
        print(json.dumps({"cells": {k: {name: value["rows"] for name, value in v["files"].items()}
                                   for k, v in result["cells"].items()},
                          "checks": result["adapter_checks"], "wall_seconds": result["wall_seconds"]}))
