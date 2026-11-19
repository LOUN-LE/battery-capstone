"""Time the selected-cell replay and record which simulated mechanisms changed."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    began = perf_counter()
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-X", "utf8", str(package / "coupled_run.py"),
                    "--input", str(arguments.input.resolve()), "--output", str(output / "replay")],
                   check=True)
    replay_complete = perf_counter()
    report = json.loads((output / "replay/report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or not all(report["checks"].values()):
        raise RuntimeError("Replay did not pass its numerical checks")
    if report["input_rows"] != 591 or not report["primary_mesh"]["split_comparison"]["passed"]:
        raise RuntimeError("Unexpected replay scope")
    ranges = report["ranges"]
    columns = {
        "sei_thickness_nm": "SEI 두께 (nm)", "sei_loss_ah": "SEI 리튬 누적 소비 (Ah)",
        "plated_li_mol_m3": "금속 리튬 농도 (mol/m³)", "dead_li_mol_m3": "비활성 리튬 농도 (mol/m³)",
        "negative_crack_length_m": "음극 균열 길이 (m)", "lam_ne_pct": "음극 활물질 손실 (%p)",
        "lam_pe_pct": "양극 활물질 손실 (%p)", "negative_porosity": "음극 공극률 (0–1)",
    }
    interpretation = [
        "# 선별 셀 1개 열화 지표",
        "",
        f"셀 {report['cell_id']}, {report['input_rows']}개 관측 시점, {report['duration_s']:.3f}초 방전 이력.",
        f"관측 대비 전압 RMSE {report['observed_voltage_rmse_mV']:.3f}mV. 수치 재현 검증은 통과했으며, 물리 정확도 합격선과 열화 원인 실측은 이번 시험에 없다.",
        "",
        "| 모델 지표 | 시작 | 끝 | 변화 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in columns.items():
        values = ranges[key]
        interpretation.append(f"| {label} | {values['start']:.10g} | {values['end']:.10g} | {values['end_minus_start']:+.10g} |")
    interpretation.extend([
        "",
        "방전 구간에서 SEI 두께와 누적 손실, 비활성 리튬 농도의 변화 방향을 확인한다. 금속 리튬 농도가 줄었다면 가역 도금이 포함된 모델의 순재고 감소를 뜻한다. 금속·비활성 리튬 농도 변화와 Ah 단위 손실의 크기는 직접 순위를 매길 수 없다.",
        "균열과 활물질 손실 변화는 같은 변수의 시작·끝 차이와 격자 민감도를 같이 봐야 한다. 모델은 전체 이력에서 열화 원인을 실측으로 확정하지 않는다.",
        "",
    ])
    (output / "mechanism_review.md").write_text("\n".join(interpretation), encoding="utf-8")
    primary = report["primary_mesh"]
    finer = report["refined_mesh"]
    times = {
        "standard_preparation_s": primary["preparation"]["wall_s"],
        "standard_observed_replay_s": primary["observed_replay"]["wall_s"],
        "split_verification_s": sum(primary["split_comparison"][name]["wall_s"] for name in ("left", "right")),
        "refined_mesh_s": finer["preparation"]["wall_s"] + finer["observed_replay"]["wall_s"],
        "replay_process_wall_s": replay_complete - began,
        "interpretation_and_save_s": perf_counter() - replay_complete,
        "total_wall_s": perf_counter() - began,
    }
    dump(output / "timing.json", {
        "status": "success", "scope": "read inputs, setup, solve, split / refined mesh checks, store replay, mechanism summary",
        "excludes": ["EC2 boot", "transfer", "environment setup"],
        "includes_python_subprocess_startup": True, "cell_id": report["cell_id"], "rows": report["input_rows"],
        "observed_duration_s": report["duration_s"], "times_s": times,
        "observed_voltage_rmse_mV": report["observed_voltage_rmse_mV"],
        "platform": platform.platform(), "python": platform.python_version(),
        "thread_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "pybamm": report["versions"]["pybamm"],
    })
    print(json.dumps(times))


if __name__ == "__main__":
    main()
