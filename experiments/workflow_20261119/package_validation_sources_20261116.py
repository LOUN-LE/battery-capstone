"""Collect the preserved inputs and evidence named in the capstone source index."""
from hashlib import file_digest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json

root = Path(__file__).resolve().parents[1]
codex = root.parents[1]
out = root / "outputs"
cap = codex / "2026-10-30/xzoq/outputs/capstone"
prior = codex / "2026-11-09/x20"
bench = cap / "work/external_m50_benchmark"
validation = cap / "work/validation_proposal_20261103"
earlier = validation / "latest_spme_review_20261103"
thermal = out / "local_thermal_extension"
reading = codex / "2026-10-28/new-chat/outputs/battery_research_notes"
study = prior / "outputs/gp_study_bundle"
bosch = codex / "2026-11-11/rm-dl/work/sources"
items = {}


def files(destination, directory, *names):
    for name in names:
        source = directory / name
        archive_name = f"{destination}/{Path(name).name}"
        if archive_name in items:
            raise ValueError(f"Duplicate destination: {archive_name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        items[archive_name] = source


index = out / "캡스톤_검증에_쓴_논문과_데이터셋.md"
items["00_먼저읽기.md"] = index
imperial = "01_Imperial_LG_M50T"
files(f"{imperial}/논문", cap / "work/references/kirkaldy2024",
      "Kirkaldy_2024_LG_M50T_cycle_ageing.pdf", "source.json")
files(f"{imperial}/실험2_추출자료", bench, "experiment_metadata.xlsx",
      "download_manifest.json", "kirkaldy_minimal_extract_manifest.json",
      *[p.name for p in sorted(bench.glob("kirkaldy_expt2_cell*.csv"))])
initial = sorted(bench.glob("raw_protocol/*RPT0_long_01*.mpt"))
assert len(initial) == 2
files(f"{imperial}/실험2_초기시험_장비원본", bench, *[str(p.relative_to(bench)) for p in initial])
pipeline = out / "bms_pipeline_delivery_20261116"
files(f"{imperial}/최신_연결", pipeline,
      "01_before_machine_learning/01_머신러닝_전_입력.csv",
      "02_after_machine_learning_for_twin/02_머신러닝_후_디지털트윈_입력.csv",
      "02_after_machine_learning_for_twin/selection_decisions.csv",
      "dataset_manifest.json", "verification.json", "aws_completion_receipt.json")
files(f"{imperial}/최신_연결", out / "bms_pipeline_local_20261116_v2/twin",
      "observed_replay.csv", "twin_report.json")
files(f"{imperial}/최신_연결", root / "work/bms_pipeline_20261116", "data_provenance.json")
files(f"{imperial}/최신_연결/실측전체와_준비기록", root / "work/bms_pipeline_20261116/data",
      "cell_C_full.csv", "cell_D_full.csv", "cell_C_preparation.csv", "cell_D_preparation.csv")
old = f"{imperial}/이전_용량과_온도"
files(f"{old}/실험2_25도_초기용량전압", prior / "outputs/m50_c_calibration_25c_20261110",
      "comparison.json", "capacity_voltage_curves.npz", "verification.json")
files(f"{old}/실험2_장기용량", earlier / "calendar_user_criteria_20261105",
      "report.json", "heldout_predictions.csv", "cases.json")
files(f"{old}/실험2_장기용량", earlier / "calendar_SEI_SOH_v1",
      "C_targets.json", "D_targets.json", "parameters_frozen.json")
files(f"{old}/실험2_장기용량", validation / "../literature_validation_20261103/measured_targets_v2",
      "cellC_measured_targets.csv", "cellD_measured_targets.csv")
files(f"{old}/실험4_저율방전", prior / "work/m50t_fresh_cell_20261110/c10_v1",
      "report.json", "C10_observed.npz", "declared_sources.json")
files(f"{old}/실험4_전류펄스", prior / "work/m50t_fresh_cell_20261110/gitt_pulse_waveform_v3",
      "report.json", "observed.npz")
files(f"{old}/실험4_전류펄스", prior / "work/expt4_D_25C_native_20261110", "manifest.json")
files(f"{old}/실험4_펄스온도", out / "thermal_validation/pulse_calibration", "report.json")
files(f"{old}/실험5_10도40도_전압용량", prior / "outputs/m50t_expt5_10_40_measured_surface_20261110",
      "comparison.json", "capacity_voltage_curves.npz", "verification.json")
files(f"{old}/실험5_10도40도_전압용량", prior / "work/m50t_expt5_10_40_raw", "manifest.json")
files(f"{old}/실험5_열모델", prior / "outputs/m50t_thermal_boundary_20261110_v2", "report.json")
files(f"{old}/실험2와5_장기용량", prior / "outputs/multitemperature_summary_baseline_20261110_v1",
      "결과와_자료가용성.txt", "predictions.csv", "report.json", "declared_contract.json", "independent_audit.json")
files(f"{old}/실험2와5_장기용량/평가에_쓴_실측_요약", validation / "additional_conditions_v2",
      *[f"kirkaldy_expt2_cell{cell}_performance_summary.csv" for cell in "ABEF"])
files(f"{old}/실험2와5_장기용량/평가에_쓴_실측_요약", validation / "expt5_eol_search_v1",
      *[f"kirkaldy_expt5_cell{cell}_performance_summary.csv" for cell in "ABCDEFGH"])
files(f"{old}/실험5_용량80_비교", prior / "outputs/low_temperature_SOH80_diagnosis_20261110_v1",
      "원인분석_요약.txt", "original_interval_diagnostics.csv", "report.json")
files(imperial, study / "results", "validation_matrix.csv")

files("02_Chen_LG_M50", bench, "chen_cell02.csv", "chen_first_C10_benchmark_protocol.json")
files("02_Chen_LG_M50", cap / "work/model_review_20261102/independent_audit_v1",
      "report.json", "chen_SPMe_baseline.npz", "chen_DFN_baseline.npz", "chen_DFN_coupled.npz")
files("02_Chen_LG_M50", cap / "work/evidence", "m50_dataset_benchmark_plan_20261029.md")

nist = thermal / "nist_ignition"
files("03_열과발화/NIST", nist, "README.txt", "quality_report.json", "independent_verification.json",
      "events.csv", "experiments_all77.csv", "experiments_nmc21700.csv",
      "timeseries_all77.csv", "timeseries_nmc21700.csv",
      *[str(p.relative_to(nist)) for p in sorted(nist.rglob("nist_ignition*")) if p.suffix in {".pdf", ".xlsx", ".txt"}])
files("03_열과발화/NIST/예비평가", thermal / "dl_pilot_final",
      "training_features.csv", "split_manifest.csv", "predictions.csv", "report.json")
arc = thermal / "arc_rate"
files("03_열과발화/KIT_ARC", arc, "로컬_ARC_시험_설명.txt", "report.json",
      *[str(p.relative_to(arc)) for p in sorted(arc.rglob("NMC_*")) if p.is_file()])
fbd = thermal / "failure_databank"
files("03_열과발화/NASA_NLR", fbd, "m50_records.csv", "SOURCE_NOTICE.txt", "source_audit.json",
      "verification.json", *[str(p.relative_to(fbd)) for p in sorted(fbd.rglob("*.xlsx"))])
files("03_열과발화", thermal, "dataset_index.csv", "fire_data_scope.json")

files("04_이전_주행과_시뮬레이션/KU_Leuven_팩전류", study / "results/real_gp",
      "eligible_current_windows.npz", "source_sign_audit.json", "verification.json")
files("04_이전_주행과_시뮬레이션/팀_팩_시뮬레이션", study / "data", "raw_input.csv", "metadata.json")
files("04_이전_주행과_시뮬레이션/팀_팩_시뮬레이션", study / "results/validation_evidence",
      "replay_verification.json", "replay_result.json")
files("04_이전_주행과_시뮬레이션/AWS_32건", out / "aws32",
      "scenario_summary.csv", "all_traces.zip", "retrieval_verification.json")
files("04_이전_주행과_시뮬레이션/AWS_1000건", out / "aws1000_delivery_20261116",
      "딥러닝_학습용_열화_1000.csv", "scenario_summary.csv", "independent_data_audit.json", "cloud_verification.json")
files("05_구현_참고", bosch, "spme_paper.pdf", "bosch_2024.pdf")
files("05_구현_참고", validation, "model_comparison_primary_sources.json")
files("05_구현_참고/ORegan_확산함수_진단", earlier / "cutoff_transport_screen_v2",
      "수송_반증_및_공개확산함수_진단보고.txt", "scientific_input_provenance_check.json")
files("05_구현_참고", reading, "07_papers.md", "12_existing_data.md", "13_related_literature.md")

destination = out / "캡스톤_검증_논문과_데이터셋.zip"
with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
    manifest = {}
    for archive_name, source in items.items():
        with source.open("rb") as handle:
            manifest[archive_name] = {"sha256": file_digest(handle, "sha256").hexdigest(), "bytes": source.stat().st_size}
        archive.write(source, arcname=archive_name)
    archive.writestr("파일_목록_SHA256.json", json.dumps(manifest, indent=2, ensure_ascii=False))
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    for name, entry in manifest.items():
        with archive.open(name) as handle:
            assert file_digest(handle, "sha256").hexdigest() == entry["sha256"], name
print(json.dumps({"files_including_manifest": len(manifest) + 1, "bytes": destination.stat().st_size}))
