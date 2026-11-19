"""Join one review-selected sensor trace with a numerical replay at source sample times."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/bms_pipeline_local_20261116_v2"
FOLDER = ROOT / "outputs/team_single_cell_20261117"

# One row in this table describes one CSV column. The six numbered groups
# contain eight quantities because SEI and plating each produce two outputs.
MAPPING = [
    ("cell_id", "식별", "셀 ID", "", "원자료 셀 식별자", "시퀀스 묶음"),
    ("session_id", "식별", "실험 세션 ID", "", "원자료 실험 구간", "시퀀스 묶음"),
    ("time_s", "관측 입력", "구간 시작 이후 시간", "s", "원본 시각에서 세션 첫 관측 시각을 뺀 값", "입력"),
    ("voltage_V", "관측 입력", "실측 셀 전압", "V", "시험 장비의 셀 단자 전압", "입력"),
    ("current_A", "관측 입력", "실측 전류", "A", "시험 장비 전류를 변환. 방전 양수, 충전 음수", "입력"),
    ("surface_temperature_C", "관측 입력", "실측 셀 표면 온도", "°C", "표면 센서 측정값", "입력"),
    ("soc_estimated_fraction", "추정 입력", "충전 상태 추정치", "0–1", "완충 후 방전 시작을 1.0으로 가정. 현 시점까지의 전류 적산량을 공칭 5Ah로 나누어 차감", "입력"),
    ("sei_thickness_nm", "1. 보호막(SEI)", "음극 표면 평균 보호막 두께", "nm", "X-averaged negative SEI thickness [m] × 10⁹. 균열 표면과 구분", "모델 출력"),
    ("sei_loss_ah", "1. 보호막(SEI)", "음극 표면 보호막으로 인한 누적 리튬 손실 환산량", "Ah", "Loss of capacity to negative SEI [A.h]. 준비 구간의 누적량 포함", "모델 출력"),
    ("plated_li_mol_m3", "2. 리튬 도금", "음극 금속 리튬 평균 농도", "mol/m³", "X-averaged negative lithium plating concentration [mol.m-3]", "모델 출력"),
    ("dead_li_mol_m3", "2. 리튬 도금", "음극 비활성 리튬 평균 농도", "mol/m³", "X-averaged negative dead lithium concentration [mol.m-3]", "모델 출력"),
    ("negative_crack_length_m", "3. 음극 균열", "음극 입자 평균 균열 길이", "m", "X-averaged negative particle crack length [m]", "모델 출력"),
    ("lam_ne_pct", "4. 음극 활물질 손실", "음극 활물질 손실률", "%", "Loss of active material in negative electrode [%]", "모델 출력"),
    ("lam_pe_pct", "5. 양극 활물질 손실", "양극 활물질 손실률", "%", "Loss of active material in positive electrode [%]", "모델 출력"),
    ("negative_porosity", "6. 음극 공극률", "음극 평균 공극률", "0–1", "X-averaged negative electrode porosity", "모델 출력"),
    ("model_voltage_V", "계산 확인", "모델 계산 전압", "V", "Voltage [V]", "비교용"),
    ("model_temperature_C", "계산 확인", "모델 온도", "°C", "Volume-averaged cell temperature [K] − 273.15. 이 모델은 25°C 등온", "비교용"),
    ("selection_status", "선별 정보", "추가 검토 대상으로 선별된 셀", "", "기존 전압 예측 오차·불확실성·학습 범위 기반 선별. 실측 위험도 라벨 없음", "메타데이터"),
    ("source_row", "출처", "원본 행 번호", "", "공개 자료의 해당 방전 파일 내 행 번호", "추적용"),
    ("source_time_s", "출처", "원본 측정 시각", "s", "시험 장비 원래 시간 좌표", "추적용"),
]
FIELDS = [row[0] for row in MAPPING]
TARGETS = [row[0] for row in MAPPING if row[-1] == "모델 출력"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"CSV header is empty or contains duplicate columns: {path.name}")
        rows = list(reader)
        require(rows and all(None not in row and None not in row.values() for row in rows),
                f"CSV contains malformed rows: {path.name}")
        return rows


def identity(row, time_column):
    return row["cell_id"], row["session_id"], int(row["source_row"]), Decimal(row[time_column])


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(model_path, model_report, folder, run_directory=RUN):
    manifest = json.loads((run_directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    source_path = run_directory / Path(manifest["after"]["file"])
    require(file_hash(source_path) == manifest["after"]["sha256"], "The frozen selection CSV changed")
    selection_path = run_directory / "machine_learning/selection_decisions.csv"
    require(file_hash(selection_path) == manifest["selection_decisions_sha256"], "The frozen decisions changed")
    sensor_rows, model_rows = records(source_path), records(model_path)
    cell, session, _, origin = identity(sensor_rows[0], "time_s")
    actual = [identity(row, "time_s") for row in sensor_rows]
    calculated = [identity(row, "source_time_s") for row in model_rows]
    require(actual == calculated and len(actual) == len(set(actual)), "Source rows and replay rows must align exactly")
    require(len(actual) == 591 and {(row[0], row[1]) for row in actual} == {(cell, session)}, "Expected one selected cell")
    selected = [row for row in records(selection_path) if row["cell_id"] == cell
                and row["session_id"] == session and row["selected_by_model"].lower() in ("1", "true")]
    require(len(selected) == 58, "Expected 58 previous review selections for this cell")
    cutoff = max(Decimal(row["decision_cutoff_s"]) for row in selected)
    require(actual[-1][-1] == cutoff and cutoff == Decimal(str(manifest["after"]["decision_cutoff_s"])),
            "Final row must be the last review decision time")

    elapsed = [key[-1] - origin for key in actual]
    require(all(a < b for a, b in zip(elapsed, elapsed[1:])), "Expected strictly increasing sensor times")
    cumulative_Ah = 0.0
    errors_V = []
    output = []
    last_current = None
    for i, (sensor, model) in enumerate(zip(sensor_rows, model_rows)):
        current = float(sensor["current_A"])
        estimated_temp = float(model["model_temperature_C"])
        voltage = float(sensor["voltage_V"])
        observed_temp = float(sensor["surface_temperature_C"])
        model_voltage = float(model["model_voltage_V"])
        require(all(map(math.isfinite, (current, estimated_temp, voltage, observed_temp, model_voltage))),
                "Sensor and model voltage/temperature/current must be finite")
        require(abs(current - float(model["model_current_A"])) < 1e-8, "Replay current differs from sensor current")
        require(abs(estimated_temp - 25.0) < 1e-7, "This run must use the declared 25 °C isothermal model")
        if last_current is not None:
            cumulative_Ah += (last_current + current) * float(elapsed[i] - elapsed[i - 1]) / 7200
        last_current = current
        soc = 1.0 - cumulative_Ah / 5.0
        require(0 <= soc <= 1, "SOC estimate outside the stated range")

        targets = {name: model[name] for name in TARGETS}
        require(all(s and math.isfinite(float(s)) for s in targets.values()), "Missing or nonfinite degradation quantity")
        require(0 < float(targets["negative_porosity"]) < 1, "Negative-electrode porosity outside physical bounds")
        require(all(float(targets[name]) >= -1e-9 for name in TARGETS),
                "A model quantity is negative beyond numerical roundoff")
        require(float(targets["sei_thickness_nm"]) > 0 and float(targets["negative_crack_length_m"]) > 0,
                "Expected active positive film thickness and crack state")
        errors_V.append(model_voltage - voltage)
        output.append({
            **{name: sensor[name] for name in ("cell_id", "session_id", "voltage_V", "current_A",
                                              "surface_temperature_C", "source_row")},
            "time_s": format(elapsed[i], "f"), "source_time_s": sensor["time_s"],
            "soc_estimated_fraction": repr(soc), **targets,
            "model_voltage_V": model["model_voltage_V"], "model_temperature_C": model["model_temperature_C"],
            "selection_status": "selected_for_review",
        })
    report = json.loads(model_report.read_text(encoding="utf-8"))
    require(report["status"] == "completed" and all(report["checks"].values())
            and report["primary_mesh"]["split_comparison"]["passed"],
            "Model output must come from a completed and checked replay")
    require(report["output_sha256"] == file_hash(model_path), "Replay CSV differs from its reviewed report")
    require(report["input_rows"] == len(output) and report["cell_id"] == cell
            and report["source_sha256"][source_path.name] == file_hash(source_path),
            "Replay report points to another set of sensor records")
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "추가검토_셀1개_실측입력과_열화6종.csv"
    mapping_path = folder / "CSV_열화6종_열매핑.csv"
    write_rows(csv_path, FIELDS, output)
    mapping_fields = ["csv_column", "quantity_group", "korean_name", "unit", "source_or_definition", "role"]
    write_rows(mapping_path, mapping_fields, [dict(zip(mapping_fields, row)) for row in MAPPING])
    require(output == records(csv_path), "Export failed CSV round-trip")
    replay_rmse_mV = 1000 * math.sqrt(sum(e * e for e in errors_V) / len(errors_V))
    receipt = {
        "cell_id": cell, "rows": len(output), "columns": len(FIELDS), "degradation_groups": 6,
        "degradation_columns": len(TARGETS), "selected_windows": len(selected), "risk_label_available": False,
        "scope": "coupled model output exercise; no measured degradation targets",
        "selection_scope": "selected for review based on voltage forecast uncertainty and observed residuals",
        "public_measurement_source": "https://zenodo.org/records/10637534",
        "simulation_model": "SPMe with OKane2022, 25 C isothermal, negative particle cracking, both electrode stress-driven LAM",
        "baseline_voltage_RMSE_mV": json.loads((run_directory / "verification.json").read_text(encoding="utf-8"))["physical_replay_rmse_mV"],
        "paths_sha256": {p.name: file_hash(p) for p in (source_path, selection_path, model_path, csv_path, mapping_path)},
        "measured_current_charge_Ah": cumulative_Ah, "initial_SOC_assumption": 1.0,
        "nominal_capacity_Ah": 5.0, "final_SOC_estimate": float(output[-1]["soc_estimated_fraction"]),
        "replay_voltage_RMSE_mV": replay_rmse_mV,
        "model_report_sha256": file_hash(model_report),
        "model_report_status": report.get("status"),
        "degradation_ranges": {name: [min(float(row[name]) for row in output), max(float(row[name]) for row in output)]
                               for name in TARGETS},
    }
    (folder / "검증기록.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=FOLDER)
    parser.add_argument("--pipeline-run", type=Path, default=RUN)
    options = parser.parse_args()
    export(options.model, options.model_report, options.out, options.pipeline_run)
