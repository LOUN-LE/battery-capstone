# 선별 셀 1개 열화 시나리오 비교

LG M50T D 셀의 실측 입력과 열화 출력 6종을 연결하고, 같은 시작 상태에서 반응 조합을 비교한 코드다. 완료된 장기 기준 실행과 남은 검증은 [실험 보고서](../../docs/d_mechanism_review_20261119.md)에 기록했다.

## 실행 환경

저장소 루트에서 Python 3.11을 사용한다.

```powershell
python -m pip install -r bms_pipeline/requirements.txt
python experiments/team_six_outputs_20261117/coupled_run.py --help
python experiments/team_six_outputs_20261117/d_long_scoring.py --self-test
```

## 짧은 구간과 열화 출력

[전달용 CSV와 열 매핑](../../examples/team_single_cell_20261117)은 D 셀 591개 관측 시점에 실측 입력과 모델 출력 8개를 연결한 결과다. 보호막과 도금이 각각 두 출력을 갖기 때문에 6개 그룹에 8개 값이 들어간다.

```powershell
python experiments/team_six_outputs_20261117/coupled_run.py --output outputs/coupled_replay
python experiments/team_six_outputs_20261117/compare_mechanism_scenarios.py --output outputs/mechanism_comparison.json
```

첫 명령은 기본 격자, 분할 계산, 세밀 격자를 비교한다. 두 번째는 같은 준비 상태에서 두 반응 모두 활성, SEI 성장 중지, 도금 반응 중지, 둘 다 중지를 비교한다. 다른 균열·활물질 손실 설정은 공통이다.

CSV를 다시 만들 때는 먼저 [기본 연결 실행기](../../bms_pipeline/README.md)로 입력과 선별 결과를 생성한다.

```powershell
python experiments/team_six_outputs_20261117/export_selected_cell.py --pipeline-run outputs/pipeline_run --model outputs/coupled_replay/coupled_model_rows.csv --model-report outputs/coupled_replay/report.json --out outputs/selected_cell
```

## 장기 재현의 별도 입력

`run_d_long_scenarios.py`는 원시 운전 이력과 메타데이터를 복원한 작업 폴더를 `--root`로 받는다. 이 폴더에는 다음 자료가 필요하다.

- `repo/simulation/`의 이력 컴파일러와 `work/preprocess/common.py`, 원시 장비 파일을 읽는 `galvani` 환경
- `work/external_m50_benchmark/`의 D 셀 원시 RPT0·RPT1와 성능 요약
- `work/validation_proposal_20261103/latest_spme_review_20261103/raw_mode_reconstruction_v1/`의 순서·해시가 검증된 이력과 원시 파형 배열
- 같은 폴더 아래 `raw_exposure_scenarios_v1/D_all_unresolved_rest_25C.json` 및 10·40°C 조건
- `work/validation_proposal_20261103/metadata_control_probe_v2/decoded_programs_raw.json`

장비 원본과 복원 입력에는 경로 및 해시 참조가 있다. 외부 폴더를 옮겼다면 참조와 원본 해시를 확인해야 한다. 실행기는 입력이 바뀌거나 재개 상태와 다른 설정이면 오류를 낸다.

```powershell
python experiments/team_six_outputs_20261117/run_d_long_scenarios.py --root PATH_TO_RESTORED_CAPSTONE --compress-cc --group-cc --output outputs/D_baseline
python experiments/team_six_outputs_20261117/run_d_long_scenarios.py --root PATH_TO_RESTORED_CAPSTONE --compress-cc --group-cc --resume --output outputs/D_baseline
```

`--scenario`로 위 네 반응 조합을 선택한다. 반응 변경은 공통 초기 성능시험 이후부터 적용한다. `--gap-temperature`는 10·25·40°C 중 공백 구간에 가정할 온도이며, 관측 구간은 25°C 등온 모델이다. 공백은 실측된 운전으로 확정되지 않았다.

`d_long_scoring.py`는 두 모델 RPT의 NPZ와 외부 실측 자료를 받아 용량 및 전압 곡선을 비교한다. `--manifest`에는 `model_rpts`, `raw_reference_dir`, 선택적인 `empirical_summary`·`criteria_path`를 지정한다. 각 RPT는 자체 2.5 V 사건으로 종료된 방전 곡선이어야 한다.

## AWS 단기 타이밍

`aws_measure_20261118.py`는 단기 재현과 검증 시간을 기록한다. AWS 런처는 인증된 CloudShell에서 기존 인스턴스와 버킷을 인자로 받아 실행한다.

```text
python experiments/team_six_outputs_20261117/launch_aws_timing_20261118.py --help
```

런처의 ZIP은 저장소 루트 기준 `experiments/team_six_outputs_20261117/`, `bms_pipeline/`, `docs/imperial_lg_m50t_fields.md`를 포함해야 한다. 서버에는 `--python` 경로의 환경과 SSM 연결이 필요하다. 실행 후 8분 안전 종료 예약, 명령 종료 시 1분 뒤 종료 예약, 결과 S3 저장을 적용한다. 인스턴스·버킷·리전은 실행 인자로 지정한다.

## 보존된 증거

- [사전 시나리오 첫 실행](evidence/local_mechanism_switch_20261118.json)과 [재시도](evidence/local_mechanism_switch_retry_20261118.json)
- [장기 기준 실행](evidence/long_D_baseline/progress.json), [용량·전압 점수](evidence/long_D_baseline/d_long_score.json), [단계별 기록](evidence/long_D_baseline/steps.json)
- [병합 계산 대조](evidence/long_D_baseline/grouping_physical_audit.json)와 [공개용 파일 해시 목록](evidence/publication_manifest.json)

기록의 원본 해시는 당시 실행 소스와 파일 바이트를 가리킨다. 공개본에서는 개인 경로를 치환했고 코드 기본 경로를 저장소 기준으로 바꿨다. 두 파일의 해시는 목록에서 함께 확인할 수 있다. 원시 대용량 이력과 바이너리 계산 상태는 로컬에 보관한다.
