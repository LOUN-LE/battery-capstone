"""Package only verified, credential-free source and measured-input artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make(result_dir, delivery):
    result_dir, delivery = Path(result_dir).resolve(), Path(delivery).resolve()
    gate = json.loads((result_dir / "verification.json").read_text(encoding="utf-8"))
    if gate.get("ready_for_cloud_integration_reproduction") is not True:
        raise ValueError("Local integration verification has not passed")
    report = json.loads((result_dir / "pipeline_report.json").read_text(encoding="utf-8"))
    if report["code_sha256"] != {k: digest(HERE / k) for k in report["code_sha256"]}:
        raise ValueError("Source changed since the validated local run")
    delivery.mkdir(parents=True, exist_ok=False)
    for directory in ("01_before_machine_learning", "02_after_machine_learning_for_twin"):
        shutil.copytree(result_dir / directory, delivery / directory)
    for name in ("dataset_manifest.json", "verification.json", "pipeline_report.json"):
        shutil.copyfile(result_dir / name, delivery / name)
    intro = f'''실측 입력 → 머신러닝 검토 → 디지털 트윈 연결

1. 01_before_machine_learning: 머신러닝 전 입력 {gate['before_rows']:,}행. C 셀은 학습·보정, D 셀은 평가에 사용했습니다.
2. 02_after_machine_learning_for_twin: 선별 후 디지털 트윈 입력 {gate['after_rows']:,}행. D 셀의 판정 시점까지 기록이며, 측정값은 원본과 같습니다.

머신러닝은 약 10초 앞 전압을 예측하고, 예측과 관측의 불일치 또는 학습 범위 부족을 검토합니다. 실제 수명 위험 등급을 붙인 파일은 아닙니다.
선별 기준은 C 셀로만 정했고, D 셀에서는 바꾸지 않았습니다. D의 {gate['selected_windows']}개 평가 구간 모두 추가 검토 대상으로 선택되어 계산 절감 효과는 아직 없습니다.

확인한 것: 입력/단위/시각 보존, 미래 정답 누수 방지, 선별 파일과 물리 계산의 연결, 내부 상태 저장·재개, 전하 보존, 같은 상태에서의 미래 조건 분기.
머신러닝 전압 예측 오차: {gate['voltage_forecast_rmse_mV']:.3f}mV. 디지털 트윈 관측 전압 오차: {gate['physical_replay_rmse_mV']:.3f}mV. 두 값은 계산 목적과 예측 방식이 다르므로 서로 우열 비교하지 않습니다.
디지털 트윈은 이번에 25°C 등온 조건을 사용했습니다. 실제 온도 재현·수명·열화 원인 검증은 완료되지 않았습니다.

입력 CSV 외에 초기 상태 준비 자료와 배터리 설정도 필요합니다. 실행 패키지에 포함했습니다.
원자료: https://zenodo.org/records/10637534
'''
    (delivery / "먼저읽기.txt").write_text(intro, encoding="utf-8-sig")
    sources = ["run_pipeline.py", "verify_pipeline.py", "make_release.py", "data_adapter.py",
               "ml_selector.py", "twin_runner.py", "test_ml_selector.py", "test_twin_runner.py",
               "requirements.txt", "README.md", "data_provenance.json",
               "configs/frozen_model.json", "configs/cell_calibration.json",
               "data/initial_low_rate_prefix.csv", "data/cell_C_prefix.csv", "data/cell_D_prefix.csv",
               "data/cell_D_preparation.csv"]
    manifest = {name: digest(HERE / name) for name in sources}
    release = delivery / "bms_pipeline_release_20261116.zip"
    with zipfile.ZipFile(release, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(HERE / name, name)
        z.writestr("release_manifest.json", json.dumps(manifest, indent=2))
        z.write(result_dir / "verification.json", "local_verification.json")
    with zipfile.ZipFile(release) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    receipt = {"release": release.name, "sha256": digest(release), "bytes": release.stat().st_size,
               "packaged_files": manifest, "all_zip_contents_verified": True,
               "local_gate": "verification.json", "credentials_included": False,
               "windows_checkpoint_included": False}
    (delivery / "release_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    share = delivery.parent / "실측입력_머신러닝전후_로컬검증_20261116.zip"
    with zipfile.ZipFile(share, "w", zipfile.ZIP_DEFLATED) as z:
        for p in delivery.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(delivery).as_posix())
    print(json.dumps({"delivery": str(delivery), "release": str(release), "release_sha256": digest(release),
                      "release_bytes": release.stat().st_size, "share": str(share)}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--delivery", required=True)
    args = p.parse_args()
    make(args.results, args.delivery)
