"""Import public NIST mds2-4027 data without turning hot temperatures into fire labels.

Uses only the standard library. The XLSX is read-only OOXML input, never rewritten.
Outcome annotations below were visually transcribed from the cited primary paper.
Unknown experimental IDs and missing event timestamps remain unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

DATA_URL = "https://data.nist.gov/od/id/mds2-4027"
XLSX_URL = "https://data.nist.gov/od/ds/mds2-4027/Li-ion%20Battery%20Safe%20Intervention%20Time%20Temperature%20Data.xlsx"
DESCRIPTION_URL = "https://data.nist.gov/od/ds/mds2-4027/Data%20Descriptions.txt"
PAPER_URL = "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=961210"
EXPECTED_IDS = set(range(1, 83)) - {18, 19, 20, 23, 27}
CONTINUOUS_IDS = set(range(1, 15)) | set(range(49, 61))
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Paper Fig.3 (printed p.9), Fig.4 (p.11); line styles defined in section4.2.1.
# Fig.8 (p.15) independently confirms NMC low-SOC line styles.
# Exp33 is not assigned: Fig.3b shows it at75% while TableA1 says25%.
# Exp62 is not assigned: missing in Fig.4a, which instead includes excluded Exp19.
NO_IGNITION_IDS = {15, 17, 29, 45, 16, 21, 30, 47, 22, 24, 41, 25, 26,
                   65, 72, 73, 67, 74, 77, 78}
DELAYED_IGNITION_IDS = {36, 37, 44, 39, 31, 34, 40, 32, 35, 42,
                        68, 69, 82, 70, 71, 81, 66, 79, 80, 75, 76}
INTERRUPTED_IGNITION_IDS = {28, 38, 46, 43, 48, 61, 63, 64}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotation(experiment_id: int) -> dict:
    """Return observational labels, never inspecting temperature values."""
    if experiment_id not in EXPECTED_IDS:
        raise ValueError("Experiment is not in the77 published source cases")
    result = {"ignition_observed": None, "ignition_outcome": "unknown",
              "outcome_source": PAPER_URL, "outcome_locator": "",
              "outcome_annotation_method": "manual_primary_paper_visual_audit",
              "outcome_note": "", "svb_observed": True,
              "svb_time_s": None, "ignition_time_s": None,
              "heating_start_time_s": None, "heating_stop_time_s": None,
              "visual_observation_end_s": None, "hrr_kw": None,
              "gas_composition_available": False, "m50t_identity_verified": False}
    if experiment_id in CONTINUOUS_IDS:
        result.update(ignition_observed=True, ignition_outcome="ignition",
                      outcome_locator="section4.1 and AppendixA continuous experiment cohort")
    else:
        result["outcome_locator"] = "Figure3, printed p.9" if experiment_id < 49 else "Figure4, printed p.11"
        if experiment_id in NO_IGNITION_IDS:
            result.update(ignition_observed=False, ignition_outcome="no_ignition_observed",
                          outcome_note="No ignition reported during this experiment; not permanent safety. Exact visual follow-up horizon unavailable.")
        elif experiment_id in DELAYED_IGNITION_IDS:
            result.update(ignition_observed=True, ignition_outcome="delayed_ignition")
        elif experiment_id in INTERRUPTED_IGNITION_IDS:
            result.update(ignition_observed=True, ignition_outcome="ignition")
        elif experiment_id == 62:
            result["outcome_note"] = "Exp62 is absent from Figure4a; excluded Exp19 appears instead. No correction inferred."
        elif experiment_id == 33:
            result["outcome_note"] = "Figure3b places Exp33 in75%SOC; AppendixA and description give25%. No correction inferred."
    return result


def parse_description(text: str) -> dict[int, dict]:
    """Read experiment IDs/SOC/intervention rows, ignore excluded table leftovers."""
    metadata = {}
    pending_ids = []
    pending_soc = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Experiment #"):
            pending_ids = [int(v) for v in re.findall(r"Exp\s*(\d+)", line)]
        elif pending_ids and line.startswith("SOC (%)"):
            pending_soc = [int(v) for v in re.findall(r"\d+", line)]
        elif pending_ids and line.startswith("Intervention Time Relative"):
            values = re.findall(r"NA|\d+", line.split("(s)", 1)[1])
            if len(values) != len(pending_ids) or len(pending_soc) != len(pending_ids):
                raise ValueError("Description matrix row lengths disagree")
            for exp, soc, stop in zip(pending_ids, pending_soc, values):
                if exp not in EXPECTED_IDS:
                    continue
                if exp in metadata:
                    raise ValueError(f"Duplicate metadata for Exp{exp}")
                if soc not in {25, 50, 75, 100}:
                    raise ValueError("Unexpected SOC")
                metadata[exp] = {"soc_pct": soc,
                    "intervention_after_svb_s": None if stop == "NA" else float(stop)}
            pending_ids = []
            pending_soc = []
    if set(metadata) != EXPECTED_IDS:
        raise ValueError("Description must supply exactly77 expected experiments")
    for exp, meta in metadata.items():
        if (meta["intervention_after_svb_s"] is None) != (exp in CONTINUOUS_IDS):
            raise ValueError("Continuous/interrupted description mismatch")
    return metadata


def read_xlsx(path: Path) -> dict[int, list[dict]]:
    """Read source cell values and addresses, preserving missing TC values."""
    with ZipFile(path) as archive:
        shared = ["".join(t.itertext()) for t in ET.fromstring(
            archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        def cell_value(cell):
            value = cell.find("m:v", NS)
            if value is None:
                return None
            return shared[int(value.text)] if cell.get("t") == "s" else value.text
        rows = root.findall("m:sheetData/m:row", NS)
        columns = {}
        for cell in rows[0]:
            value = cell_value(cell)
            if value and re.fullmatch(r"Exp\d+", value):
                columns[re.sub(r"\d", "", cell.get("r"))] = int(value[3:])
        if len(columns) != 77 or set(columns.values()) != EXPECTED_IDS:
            raise ValueError("Workbook experiment columns differ from expected77")
        traces = {exp: [] for exp in columns.values()}
        for row in rows[2:]:
            cells = {re.sub(r"\d", "", c.get("r")): c for c in row}
            if "A" not in cells or cell_value(cells["A"]) is None:
                continue
            time_s = float(cell_value(cells["A"]))
            if not math.isfinite(time_s):
                raise ValueError("Nonfinite source time")
            for col, exp in columns.items():
                if col not in cells:
                    continue  # absent padded cells are not measurements
                raw = cell_value(cells[col])
                if raw is None:
                    continue
                temp = None if raw.strip().lower() == "na" else float(raw)
                if temp is not None and not math.isfinite(temp):
                    raise ValueError("Nonfinite source temperature")
                traces[exp].append({"experiment_id": exp, "time_s": time_s,
                    "surface_temperature_c": temp, "temperature_missing": temp is None,
                    "source_sheet": "Sheet1", "source_cell": cells[col].get("r")})
        for exp, trace in traces.items():
            if not trace:
                raise ValueError(f"No source samples for Exp{exp}")
            times = [r["time_s"] for r in trace]
            if times[0] != 0 or any(b - a != 1 for a, b in zip(times, times[1:])):
                raise ValueError(f"Exp{exp} is not contiguous source1Hz data")
        return traces


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else str(v).lower() if isinstance(v, bool) else v)
                             for k, v in row.items()})


def export_dataset(workbook: Path, description: Path, paper: Path, out: Path) -> dict:
    hashes = {"workbook": sha256(workbook), "description": sha256(description), "paper": sha256(paper)}
    metadata = parse_description(description.read_text(encoding="utf-8-sig"))
    traces = read_xlsx(workbook)
    experiments = []
    for exp in sorted(traces):
        samples = traces[exp]
        valid = [r for r in samples if r["surface_temperature_c"] is not None]
        row = {"experiment_id": exp, "dataset_id": "NIST_mds2-4027",
               "cell_format": "21700" if exp >= 49 else "18650",
               "cathode": "NMC" if exp >= 49 else "NCA", "anode": "graphite",
               "manufacturer": "unknown", "cell_model": "unknown",
               "nominal_capacity_ah": 5.0 if exp >= 49 else 3.5,
               **metadata[exp], "heating_mode": "continuous" if exp in CONTINUOUS_IDS else "interrupted",
               "thermocouple_malfunction_reported": exp in {3, 17},
               **annotation(exp), "source_sample_rows": len(samples), "valid_temperature_rows": len(valid),
               "temperature_record_start_s": samples[0]["time_s"],
               "temperature_record_end_s": samples[-1]["time_s"],
               "last_valid_temperature_time_s": valid[-1]["time_s"] if valid else None,
               "time_origin": "source_workbook_Time_column; per-case heating/SVB/ignition alignment not supplied",
               "source_workbook_sha256": hashes["workbook"], "source_description_sha256": hashes["description"],
               "source_paper_sha256": hashes["paper"]}
        experiments.append(row)
    out.mkdir(parents=True, exist_ok=True)
    all_samples = [s for exp in sorted(traces) for s in traces[exp]]
    nmc = [r for r in experiments if r["cell_format"] == "21700"]
    write_csv(out / "experiments_all77.csv", experiments)
    write_csv(out / "experiments_nmc21700.csv", nmc)
    write_csv(out / "timeseries_all77.csv", all_samples)
    write_csv(out / "timeseries_nmc21700.csv", [r for r in all_samples if r["experiment_id"] >= 49])
    events = [{k: r[k] for k in ["experiment_id", "ignition_observed", "ignition_outcome", "outcome_source",
              "outcome_locator", "outcome_annotation_method", "outcome_note", "svb_observed", "svb_time_s",
              "ignition_time_s", "heating_start_time_s", "heating_stop_time_s", "visual_observation_end_s",
              "temperature_record_end_s", "time_origin", "source_paper_sha256"]} for r in experiments]
    write_csv(out / "events.csv", events)
    report = {"schema_version": "nist_ignition_v1", "source_urls": {"catalog": DATA_URL, "workbook": XLSX_URL,
               "description": DESCRIPTION_URL, "paper": PAPER_URL}, "source_sha256": hashes,
              "source_unchanged_after_read": hashes == {"workbook": sha256(workbook), "description": sha256(description), "paper": sha256(paper)},
              "experiments": len(experiments), "nmc21700_experiments": len(nmc),
              "label_counts_all77": dict(Counter(r["ignition_outcome"] for r in experiments)),
              "label_counts_nmc21700": dict(Counter(r["ignition_outcome"] for r in nmc)),
              "binary_counts_nmc21700": dict(Counter(str(r["ignition_observed"]) for r in nmc)),
              "timeseries_rows": len(all_samples), "nmc21700_timeseries_rows": sum(r["experiment_id"] >= 49 for r in all_samples),
              "missing_temperature_rows": sum(r["temperature_missing"] for r in all_samples),
              "known_sensor_failures": [3, 17], "unknown_outcome_ids": [r["experiment_id"] for r in experiments if r["ignition_observed"] is None],
              "source_1hz_time_grid_pass": True, "interpolation_performed": False,
              "temperature_used_to_infer_ignition": False, "event_timestamps_available": False,
              "exact_negative_visual_followup_horizon_available": False,
              "m50t_identity_verified": False, "hrr_available": False, "gas_composition_available": False,
              "dimension_typo_not_propagated": "Description18x650/21x700mm omitted; format names only retained.",
              "not_a_validated_fire_predictor": True,
              "source_code_sha256": sha256(Path(__file__))}
    report["output_sha256"] = {p.name: sha256(p) for p in sorted(out.glob("*.csv"))}
    (out / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "NIST 실제 실험 점화 자료 — 관측 기반 CSV\n\n"
        "원본77개 실험(18650 NCA43개, 21700 NMC34개)을 보존했다. NMC21700은 점화26, 실험 중 미점화7, 미확정1이다.\n"
        "ignition_observed=true는 논문에서 실제 점화 또는 지연 점화가 보고된 실험이다. false는 실험 중 미점화 보고이며 영구 안전을 뜻하지 않는다. 빈칸은 미확정이다.\n"
        "표면 온도 임계값으로 화재 정답을 만들지 않았다. 점화 라벨은 논문4.1/그림3/4에서 수동 대조했으며 events.csv에 근거가 있다.\n"
        "Exp62는 그림4a의 실험 번호 불일치, Exp33은 그림3b와 표A1의 SOC 불일치 때문에 라벨을 비웠다. 제외 실험18/19/20/23/27은 수입하지 않았다.\n"
        "Exp3/17은 TC 고장 표식을 유지하고 온도값은 빈칸으로 보존했다. TC고장은 관측 점화 라벨을 지우는 근거가 아니다.\n"
        "원본1Hz 표본 그대로이며 보간하지 않았다. source_cell은 XLSX 원본 위치다. 시계열 끝의 패딩 빈칸은 표본으로 만들지 않았다.\n"
        "시간은 원본 Time열 초 단위다. 개별 SVB/점화/가열 시작·종료 절대 시각은 원본에 없어 비워 두었다. intervention_after_svb_s만 설명표의 SVB 상대 시각이다.\n"
        "미점화의 정확한 영상 관측 종료 시각은 공개되지 않았다. temperature_record_end_s는 온도 기록 끝이며 영상 추적 끝과 같다고 가정하지 않는다.\n"
        "제조사·모델이 공개되지 않아 M50/M50T와 동일한 셀로 취급하지 않는다. 외부 가열 실험이며 충전·주행 화재 확률로 직접 전이하지 않는다.\n"
        "실측 HRR(kW), 가스 조성, 전압·전류 시계열은 이 공개 파일에 없다. SVB는 가스 배출 사건이며 점화와 구분한다.\n"
        "DL 학습 시 동일 실험의 창을 train/test에 섞지 말고 실험 단위로 분리한다. 전체 시계열 최고온도·점화 후 구간·실험길이로 조기 예측 성능을 주장하지 않는다.\n"
        "중단 시간은 실험 설계 변수이며 예측 시점에 알고 있는지 명시해야 한다. 34개는 예비 연구 규모이며 실제 화재 위험 예측 검증 완료가 아니다.\n\n"
        + "\n".join([DATA_URL, XLSX_URL, DESCRIPTION_URL, PAPER_URL]) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_dataset(args.workbook, args.description, args.paper, args.output)
    print(json.dumps({k: report[k] for k in ["experiments", "binary_counts_nmc21700", "timeseries_rows"]}))


if __name__ == "__main__":
    main()
