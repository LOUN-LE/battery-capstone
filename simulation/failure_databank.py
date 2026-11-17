"""Audit NASA/NREL FTRC summary data without inventing SOC or fire labels.

Only the standard library is used to read OOXML. Stored Excel values are read;
formulas are never evaluated and the original workbook is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

SOURCE_URL = "https://www.nlr.gov/media/docs/libraries/tsdc/battery-failure-databank-revision2-feb24.xlsx?sfvrsn=7b7f89_1"
EXPECTED_SHA256 = "18a6b0ed0bbc881295ef9c832e4dd1937b653702ec6481834e3c6ee50c86a7c4"
PAGE_URL = "https://www.nlr.gov/transportation/battery-failure"
PAPER_URL = "https://docs.nlr.gov/docs/fy22osti/82410.pdf"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MISSING = {"", "-"}


def read_xlsx(path):
    """Return cells by sheet, preserving blanks, raw values and external links."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in node.findall(".//s:t", NS))
                       for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        relationships = {r.get("Id"): r.get("Target") for r in
                         ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        prop = book.find("s:workbookPr", NS)
        date1904 = prop is not None and prop.get("date1904") in ("1", "true")
        sheets = {}
        for sheet in book.findall("s:sheets/s:sheet", NS):
            target = relationships[sheet.get("{" + REL_NS + "}id")]
            part = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            root = ET.fromstring(archive.read(part))
            rows, formulas = {}, []
            for row in root.findall("s:sheetData/s:row", NS):
                cells = {}
                for cell in row.findall("s:c", NS):
                    ref = cell.get("r")
                    column = re.match(r"[A-Z]+", ref).group()
                    value = cell.find("s:v", NS)
                    raw = value.text if value is not None and value.text is not None else ""
                    kind = cell.get("t")
                    if kind == "s":
                        raw = strings[int(raw)]
                    elif kind == "inlineStr":
                        raw = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                    formula = cell.find("s:f", NS)
                    if formula is not None:
                        formulas.append({"cell": ref, "formula": formula.text,
                                         "has_cached_value": value is not None})
                    cells[column] = raw
                rows[int(row.get("r"))] = cells
            external = {}
            rel_part = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
            if rel_part in archive.namelist():
                external = {r.get("Id"): r.get("Target") for r in
                            ET.fromstring(archive.read(rel_part)) if r.get("TargetMode") == "External"}
            links = {link.get("ref"): external.get(link.get("{" + REL_NS + "}id"))
                     for link in root.findall("s:hyperlinks/s:hyperlink", NS)}
            sheets[sheet.get("name")] = {"rows": rows, "links": links, "formulas": formulas}
    return sheets, date1904


def normalized(raw):
    if raw in MISSING:
        return None
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        raise ValueError("Nonfinite source value")
    return value


def excel_date(raw, date1904=False):
    if raw in MISSING:
        return None
    value = float(raw)
    # Current source uses post-1900 dates; reject Excel's fictional leap-day.
    if not date1904 and value < 61:
        raise ValueError("Unsupported early Excel date")
    epoch = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    return (epoch + timedelta(days=value)).date().isoformat()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_audit(path):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError("Source revision changed; inspect the new schema and evidence before importing")
    sheets, date1904 = read_xlsx(path)
    data = sheets["Battery Failure Databank"]
    headers = {column: label for column, label in data["rows"][1].items() if label}
    if len(set(headers.values())) != len(headers) or len(headers) != 43:
        raise ValueError("Unexpected or duplicate source headers")
    rows = []
    for rownum, cells in data["rows"].items():
        if rownum == 1 or not cells.get("B"):
            continue
        raw = {header: cells.get(column, "") for column, header in headers.items()}
        rows.append({"source_row": rownum, "raw": raw})
    ids = [row["raw"]["Test-ID"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Test-ID in source")
    selected = [row for row in rows if re.search(r"\bLG\b.*\bM50", row["raw"]["Cell-Description"])]
    if not selected:
        raise ValueError("No LG M50-family data found")
    records = []
    residuals = {"fractional_energy_kJ": [], "fractional_energy_percent": [], "mass_g": []}
    for item in selected:
        raw = item["raw"]
        values = {key: normalized(value) for key, value in raw.items()}
        rownum = item["source_row"]
        flags = []
        if isinstance(values["Post-Test-Mass-Unrecovered-g"], float) and values["Post-Test-Mass-Unrecovered-g"] < 0:
            flags.append("negative_unrecovered_mass_preserved")
        if any(isinstance(values[key], float) and values[key] < 0 for key in
               ("Theoretical-Energy-Positive-Unrecovered-Mass-kJ", "Theoretical-Energy-Negative-Unrecovered-Mass-kJ")):
            flags.append("negative_energy_correction_preserved")
        if raw["Trigger-Mechanism"] == "Nail" and values["Avg-Cell-Temp-At-Trigger-degC"] is not None:
            flags.append("nail_trigger_temperature_outside_readme_heater_definition")
        record = {"source_row": rownum, "source_sha256": sha, "values": values,
                  "raw": raw, "test_date_iso": excel_date(raw["Test-Date"], date1904),
                  "video_url": data["links"].get(f"B{rownum}"),
                  "soc_fraction": None, "age_cycles": None, "soh_fraction": None,
                  "thermal_runaway_label": None, "fire_label": None,
                  "model_identity_match": "unverified_M50_BV_to_M50T",
                  "quality_flags": flags}
        records.append(record)
        energy = sum(values[key] for key in ("Energy-Fraction-Cell-Body-kJ", "Energy-Fraction-Positive-Ejecta-kJ", "Energy-Fraction-Negative-Ejecta-kJ"))
        residuals["fractional_energy_kJ"].append(energy - values["Corrected-Total-Energy-Yield-kJ"])
        residuals["fractional_energy_percent"].append(sum(values[key] for key in
            ("Energy-Percent-Cell-Body-%", "Energy-Percent-Positive-Ejecta-%", "Energy-Percent-Negative-Ejecta-%")) - 100)
        mass_keys = [key for key in values if key.startswith("Post-Test-Mass-")]
        residuals["mass_g"].append(sum(values[key] for key in mass_keys) - values["Pre-Test-Cell-Mass-g"])
    readme = sheets["ReadMe"]["rows"]
    dictionary = {cells.get("A"): cells.get("B", "") for number, cells in readme.items() if number >= 27 and cells.get("A")}
    counts = {header: {"present": sum(r["raw"][header] not in MISSING for r in records),
                       "missing_marker": sum(r["raw"][header] == "-" for r in records),
                       "blank": sum(r["raw"][header] == "" for r in records)} for header in headers.values()}
    checks = {
        "source_hash_matches_pinned_revision": sha == EXPECTED_SHA256,
        "unique_test_ids": len(ids) == len(set(ids)),
        "m50_count_and_identity": len(records) == 25 and {r["values"]["Cell-Description"] for r in records} == {"LG 21700-M50 (BV)"},
        "energy_fraction_closure": max(abs(x) for x in residuals["fractional_energy_kJ"]) < 1e-8,
        "energy_percent_closure": max(abs(x) for x in residuals["fractional_energy_percent"]) < 1e-8,
        "mass_balance_closure": max(abs(x) for x in residuals["mass_g"]) < 1e-8,
        "unobserved_labels_remain_unknown": all(r[key] is None for r in records for key in
             ("soc_fraction", "age_cycles", "soh_fraction", "thermal_runaway_label", "fire_label")),
    }
    groups = []
    for series, trigger in sorted({(r["values"]["Test-Series"], r["values"]["Trigger-Mechanism"]) for r in records}):
        group = [r for r in records if (r["values"]["Test-Series"], r["values"]["Trigger-Mechanism"]) == (series, trigger)]
        energies = [r["values"]["Corrected-Total-Energy-Yield-kJ"] for r in group]
        groups.append({"test_series": series, "trigger_mechanism": trigger, "count": len(group),
                       "date_start": min(r["test_date_iso"] for r in group), "date_end": max(r["test_date_iso"] for r in group),
                       "corrected_energy_min_kJ": min(energies), "corrected_energy_max_kJ": max(energies),
                       "conditioning_note": "SOC not provided per row; no mixed-SOC pooled calibration"})
    audit = {
        "source_url": SOURCE_URL, "source_page": PAGE_URL, "source_sha256": sha,
        "source_file": str(path.resolve()), "source_sheets": list(sheets),
        "readme_title": readme[1]["A"], "revision_note": "File/download is revision2 February2024; embedded ReadMe title says revision1.",
        "all_test_records": len(rows), "selected_records": len(records), "source_columns": list(headers.values()),
        "selected_cell_names": sorted({r["values"]["Cell-Description"] for r in records}),
        "trigger_counts": dict(Counter(r["values"]["Trigger-Mechanism"] for r in records)),
        "heater_subset_coverage": {key: sum(r["values"][key] is not None for r in records
            if r["values"]["Trigger-Mechanism"] == "Heater (Non-ISC)") for key in
            ("Heater-Power-W", "Heater-Time-On-s", "Energy-Applied-to-Trigger-kJ", "Avg-Cell-Temp-At-Trigger-degC")},
        "failure_mechanism_counts": dict(Counter(r["values"]["Cell-Failure-Mechanism"] for r in records)),
        "field_coverage": counts, "formula_cells": data["formulas"],
        "value_method": "Stored OOXML values, including cached formula results if present; no Excel recalculation.",
        "missing_method": "Original '-' and blank preserved in raw CSV; normalized to null in JSON. Numeric zero preserved.",
        "checks": checks, "integrity_passed": all(checks.values()),
        "max_abs_residual": {key: max(abs(x) for x in values) for key, values in residuals.items()},
        "flagged_records": [{"test_id": r["values"]["Test-ID"], "flags": r["quality_flags"]} for r in records if r["quality_flags"]],
        "external_paper_context": {"url": PAPER_URL,
            "reported_subset": "2022 paper Table I reports LG M50(BV) 8 thermal and5 nail tests, all100%SOC; cells tested within1year of manufacture.",
            "application": "Context only. Per-record SOC, exact manufacturing age, cycling and SOH are not populated from this statement. It does not cover the entire2024 workbook."},
        "scope": {"summary_calorimetry_available": True, "temperature_time_series_available": False,
                  "heater_power_time_series_available": False, "cooling_boundary_history_available": False,
                  "decomposition_kinetics_calibrated": False, "m50t_physical_validation": False,
                  "thermal_runaway_occurrence_labels_available": False, "fire_labels_available": False},
        "interpretation": [
            "Corrected total energy includes conductive heat-loss and theoretical unrecovered-mass corrections; it is not a direct combustion HRR measurement.",
            "Heater-Time-On-s is heater-start to trigger per ReadMe. The workbook does not define a temperature-rate criterion for trigger or provide a sensor-position map.",
            "Avg-Cell-Temp-At-Trigger-degC is casing temperature for heater/ISC tests per ReadMe. Nail rows with this number are retained but not treated as self-heating onset.",
            "No Ejection is an ejection observation. It implies neither thermal-runaway=false nor fire=false.",
            "Pre-test OCV is retained as voltage and is not converted to SOC. Test date is not manufacturing date or cycle age.",
            "Different series and trigger types remain separated. Repeat counts alone do not establish an independently held-out calibration test.",
            "BV denotes bottom vent in the source/paper context; exact equivalence to the current LG M50T is not established."]}
    return records, audit, dictionary, groups, sheets["Disclaimer"]["rows"]


def requirement_rows():
    # Availability is for this exact workbook, not for every possible public source.
    return [
        ("Cell model and format", "available", "Cell-Description;Cell-Format", "25 LG21700-M50(BV) records; M50T identity match unverified"),
        ("SOC", "missing", "Pre-Test-Cell-Open-Circuit-Voltage-V only", "Retrieve per-test preparation metadata; do not infer SOC from OCV or series name"),
        ("Aging and SOH", "missing", "Test-Date only", "Need cycling/calendar history and measured capacity; test date cannot identify age"),
        ("Heat and cooling histories", "partial", "Heater-Power-W;Heater-Time-On-s;Conductive-Heat-Loss-Rate-kJs-1", "Average heater power and scalar loss correction do not supply boundary time histories"),
        ("Temperature sensor position and clock", "partial", "Avg-Cell-Temp-At-Trigger-degC;ReadMe", "Casing-at-trigger described; exact sensor map and full synchronized traces absent"),
        ("Runaway onset and event definition", "partial", "Heater-Time-On-s;Cell-Failure-Mechanism", "Trigger interval and vent outcome exist; no explicit occurrence label or onset-rate threshold"),
        ("Decomposition heat/kinetics calibration", "partial", "Corrected-Total-Energy-Yield-kJ;Energy-Fraction-*", "Can benchmark integrated heat; reaction rate/activation parameters need temperature-time and component calorimetry"),
        ("Independent repeat validation", "partial", "Test-ID;Test-Series;Test-Date;Trigger-Mechanism", "Repeat records available; reserve matching-condition experiments before fitting after SOC/fixture audit"),
        ("Flame and ignition", "missing", "No dedicated fields", "Need explicit visible-light flame/ignition observations; X-ray links show internal structure"),
        ("Gas species and release rate", "missing", "Ejecta heat and mass only", "Ejecta totals do not identify gas species or gas-release time histories"),
        ("Combustion heat-release rate", "missing", "Corrected-Total-Energy-Yield-kJ only", "Need HRR versus time with combustion measurement method; do not relabel integrated calorimeter heat"),
        ("Lithium plating truth", "missing", "No plating fields", "Requires plating-specific observed evidence; thermal data cannot create plating ground truth"),
    ]


def run(source, output):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    records, audit, dictionary, groups, disclaimer = build_audit(source)
    output.mkdir(parents=True, exist_ok=True)
    headers = audit["source_columns"]
    write_csv(output / "m50_records.csv", [{"source_row": r["source_row"], **r["raw"],
        "test_date_iso": r["test_date_iso"], "video_url": r["video_url"],
        "quality_flags": ";".join(r["quality_flags"])} for r in records],
        ["source_row", *headers, "test_date_iso", "video_url", "quality_flags"])
    write_csv(output / "field_dictionary.csv", [{"source_column": h, "source_definition": dictionary.get(h, ""),
              **audit["field_coverage"][h]} for h in headers],
              ["source_column", "source_definition", "present", "missing_marker", "blank"])
    write_csv(output / "series_summary.csv", groups, list(groups[0]))
    fields = ["requirement", "availability", "source_evidence", "next_evidence_needed"]
    write_csv(output / "requirements_mapping.csv", [dict(zip(fields, row)) for row in requirement_rows()], fields)
    (output / "m50_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    # Source's required notice is retained with every exported data collection.
    notice = "\n\n".join(cell for row in disclaimer.values() for cell in row.values() if cell)
    (output / "SOURCE_NOTICE.txt").write_text(notice, encoding="utf-8")
    audit["checks"]["source_unchanged_after_export"] = hashlib.sha256(source.read_bytes()).hexdigest() == before
    audit["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit["integrity_passed"] = all(audit["checks"].values())
    (output / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["integrity_passed"]:
        raise ValueError("Data-integrity audit failed; see source_audit.json")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source, args.output)
    print(json.dumps({key: report[key] for key in
                     ("selected_records", "trigger_counts", "integrity_passed", "max_abs_residual")}))
