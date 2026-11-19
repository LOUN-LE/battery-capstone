from pathlib import Path
import copy
import json
import re
import shutil
import subprocess

work = Path(__file__).resolve().parent
root = work.parent
target = work / "capstone_git_20261117"
source = work / "bms_pipeline_20261116"
original_repo = root.parents[1] / "2026-10-30/xzoq/outputs/capstone/repo"


def git(*args):
    subprocess.run(["git", "-C", str(target), *args], check=True)


def replace(path, transform):
    file = target / path
    file.write_text(transform(file.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")


def write_json(path, content):
    (target / path).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


git("restore", "--source=origin/main", "--staged", "--worktree", "--",
    "scripts/finish_singlecell_setup.py", "scripts/run_rpt_precision_once.py",
    "scripts/singlecell_service.py", "scripts/test_singlecell_service.py")
notion = re.compile(r"\[([^\]]+)\]\(https://app\.notion\.com/[^)]+\)")
replace("README.md", lambda s: re.sub(r"\[최신 검증 보고서\]\(https://app\.notion\.com/[^)]+\)와 ", "", s))
for name in ["bayesian/README.md", "docs/team_pack_input.md"]:
    def clean(s):
        s = "\n".join(line for line in s.splitlines() if not line.startswith("설계 근거는 [이로운"))
        s = notion.sub(r"\1", s)
        s = re.sub(r"C:\\Users\\[^']+\\(raw_input\.csv|metadata\.json)", r"data/\1", s)
        s = re.sub(r"<USER_HOME>/]+/Documents/Codex/\d{4}-\d{2}-\d{2}/x20/outputs/", "outputs/", s)
        return s.replace("장이현의", "팀원의") + "\n"
    replace(name, clean)
pack = json.loads((target / "configs/team_pack_108s32p.json").read_text(encoding="utf-8"))
pack["source"] = "Team M50T ROM simulation report. Topology is supplied separately from raw_input.csv and metadata.json."
write_json("configs/team_pack_108s32p.json", pack)
replace("simulation/test_validate_short_spme.py", lambda s: s.replace("이로운/초기보정", "예시/초기보정"))

for subdir, names in {
    "docs": ["thermal_abuse_validation.md"],
    "simulation": ["arc_rate_model.py", "failure_databank.py", "ignition_training.py", "nist_ignition_dataset.py",
                   "risk_dataset1000.py", "risk_scenarios.py", "test_arc_rate_model.py", "test_ignition_training.py",
                   "test_risk_dataset1000.py", "test_risk_scenarios.py", "test_thermal_abuse_validation.py",
                   "thermal_abuse_validation.py", "thermal_validation.py", "verify_failure_databank.py",
                   "verify_thermal_validation.py"],
    "tests": ["test_nist_ignition_dataset.py"],
}.items():
    (target / subdir).mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(original_repo / subdir / name, target / subdir / name)
replace("requirements.txt", lambda s: s.rstrip() + "\nmatplotlib==3.11.2\n")

pipeline = target / "bms_pipeline"
for folder in ["", "data", "configs", "evidence"]:
    (pipeline / folder).mkdir(exist_ok=True)
for name in ["data_adapter.py", "make_release.py", "ml_selector.py", "README.md", "requirements.txt",
             "run_pipeline.py", "verify_pipeline.py", "test_ml_selector.py", "test_twin_runner.py", "twin_runner.py",
             "configs/frozen_model.json", "configs/cell_calibration.json",
             "data/initial_low_rate_prefix.csv", "data/cell_C_prefix.csv", "data/cell_D_prefix.csv",
             "data/cell_D_preparation.csv"]:
    shutil.copyfile(source / name, pipeline / name)
shutil.copyfile(work / "cloud_compare_20261116.py", pipeline / "compare_runs.py")
replace("bms_pipeline/compare_runs.py", lambda s: s.replace("bms_pipeline_20261116/data/", "data/"))
replace("bms_pipeline/data_adapter.py",
        lambda s: re.sub(r'^DEFAULT_SOURCE = .*$', 'DEFAULT_SOURCE = HERE / "source"', s, flags=re.M))


def relative_paths(value):
    if isinstance(value, str) and value.startswith("C:\\Users\\"):
        normalized = value.replace("\\", "/")
        for segment, prefix in [("/external_m50_benchmark/", "source/"), ("/bms_pipeline_20261116/data/", "data/")]:
            if segment in normalized:
                return prefix + normalized.split(segment, 1)[-1]
        return Path(normalized).name
    if isinstance(value, dict):
        return {relative_paths(key): relative_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_paths(item) for item in value]
    return value


provenance = relative_paths(json.loads((source / "data_provenance.json").read_text(encoding="utf-8")))
provenance["adapter_sha256_scope"] = "Original adapter used to derive the preserved data before its default path was made portable."
write_json("bms_pipeline/data_provenance.json", provenance)
deliver = root / "outputs/bms_pipeline_delivery_20261116"
for folder, name in [
    ("01_before_machine_learning", "01_머신러닝_전_입력.csv"),
    ("02_after_machine_learning_for_twin", "02_머신러닝_후_디지털트윈_입력.csv"),
    ("02_after_machine_learning_for_twin", "selection_decisions.csv"),
]:
    shutil.copyfile(deliver / folder / name, pipeline / "evidence" / name)
shutil.copyfile(deliver / "verification.json", pipeline / "evidence/local_verification.json")
result = root / "outputs/bms_pipeline_aws_20261116_v1"
shutil.copyfile(result / "results/verification.json", pipeline / "evidence/aws_verification.json")
comparison = json.loads((result / "cross_platform_comparison.json").read_text(encoding="utf-8"))
def public_comparison(value):
    if isinstance(value, dict):
        private_keys = {"local_directory", "cloud_directory", "source_input"}
        return {k: public_comparison(v) for k, v in value.items() if k not in private_keys}
    if isinstance(value, list):
        return [public_comparison(v) for v in value]
    return value
write_json("bms_pipeline/evidence/platform_comparison.json", public_comparison(comparison))

scenario = target / "examples/scenarios"
scenario.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / "outputs/aws1000_delivery_20261116/딥러닝_학습용_열화_1000.csv",
                scenario / "degradation_training_1000.csv")
shutil.copyfile(root / "outputs/aws32/scenario_summary.csv", scenario / "pilot_32_summary.csv")
shutil.copyfile(root / "outputs/aws1000_delivery_20261116/학습용CSV_설명.txt", scenario / "dataset_notes.txt")

index = (root / "outputs/캡스톤_검증에_쓴_논문과_데이터셋.md").read_text(encoding="utf-8")
index = re.split(r"\n## 압축 파일에서 먼저 볼 것", index, maxsplit=1)[0]
index = re.sub(r" · \[팀 전처리 기록\]\(https://app\.notion\.com/[^)]+\)", "", index)
index = re.sub(r" ZIP에는[^\n]+", " [재현 방법과 입력 CSV](../bms_pipeline/README.md)에 포함된 파일과 입력 규격을 확인할 수 있다.", index)
(target / "docs/validation_sources.md").write_text(index, encoding="utf-8", newline="\n")
