# 실측 입력에서 머신러닝 선별과 디지털 트윈 연결

Imperial LG M50T 초기 저율방전 C/D 셀의 실측 자료를 사용합니다. 입력에 해당하는 원본 시간·전압·전류·표면 온도를 보존하며 방전 전류를 양수 암페어로 통일했습니다.

## 실행

저장소 루트에서 아래 명령을 실행합니다. Python 3.11 환경과 저장소에 포함된 재현용 측정 데이터가 필요합니다.

```sh
python -m pip install -r bms_pipeline/requirements.txt
cd bms_pipeline
python -X utf8 -m unittest data_adapter.AdapterTests test_ml_selector test_twin_runner
python -X utf8 run_pipeline.py --output ../outputs/bms-local
python -X utf8 verify_pipeline.py --output ../outputs/bms-local
```

결과 경로는 새 경로여야 합니다. 실패나 이전 실행 결과를 덮어쓰지 않습니다. Windows의 내부 상태 파일을 Linux에서 불러오지 않으며, AWS에서는 준비 과정을 다시 계산합니다.

## 두 데이터 묶음

- 01_before_machine_learning: C 학습·보정 자료와 D 평가 관측을 포함한 공통 입력입니다.
- 02_after_machine_learning_for_twin: 머신러닝이 추가 검토 대상으로 선택한 D 관측과 그 시점까지의 상태 준비 구간입니다. 전압·온도를 예측값으로 교체하지 않았습니다.
- dataset_manifest.json에는 두 파일의 행 수·해시·판정 시점·추가 설정을 기록합니다.
- 디지털 트윈 실행에는 준비 과정 데이터와 configs의 물성·초기화 설정도 필요합니다. CSV만으로 내부 상태가 완성되는 것은 아닙니다.

## 검증의 의미

머신러닝은 C 셀로 학습·보정한 약 10초 앞 전압 예측 모델입니다. D를 보고 계수나 선별 기준을 바꾸지 않았습니다. 예측과 관측의 차이 또는 학습 범위 부족으로 추가 검토 구간을 고릅니다. 수명 위험 분류기나 보호막·도금 원인 분류기가 아닙니다.

물리 계산은 관측 이전의 초기 휴지로 시작 상태를 추정하고, 선언한 충전·휴지 준비 공정을 거칩니다. 이후 실제 기록 전류로 관측 구간을 계산합니다. 전하 보존, 전체 계산과 분할 계산의 일치, 저장 상태 복원, 같은 상태에서 두 미래 조건 분기를 검사합니다.

물리 계산은 25°C 등온 조건입니다. 실측 표면 온도는 보존하지만 온도 예측 정확도를 검증한 것은 아닙니다. 관측 전압 오차를 보고하되 물리 정확도 합격선은 아직 정하지 않았습니다. 수명·실제 열화 원인 검증은 별도입니다.

이번 자료에서 D의 모든 평가 구간이 추가 검토 대상으로 선택됐습니다. 선택 후 행 수가 줄어든 것은 학습용 C와 판정 이후 행을 제외한 결과이며 계산 절감 성과가 아닙니다.

## 재현 결과와 출처

[머신러닝 입력 1,200행](evidence/01_머신러닝_전_입력.csv), [디지털 트윈 입력 591행](evidence/02_머신러닝_후_디지털트윈_입력.csv), [선별 기록](evidence/selection_decisions.csv)을 내려받을 수 있습니다.

2026-11-16 실행에서 로컬과 AWS 모두 코드 검사 17개와 연결 검사 20항목을 통과했습니다. 환경 간 대조 32항목도 통과했습니다. 단기 전압 재현 오차는 4.423mV(RMSE)입니다. 전압 정확도 합격선은 사전에 정하지 않았습니다. [로컬 검증](evidence/local_verification.json) · [AWS 검증](evidence/aws_verification.json) · [환경 간 대조](evidence/platform_comparison.json)

`data/`와 위 입력 CSV는 Niall Kirkaldy, Mohammad Amin Samieian, Gregory Offer, Monica Marinescu, Yatish Patel(2024)의 공개 LG M50T 실험에서 발췌했습니다. 실험 2,2의 25°C 초기 저율방전과 그 전에 측정한 준비 구간을 골랐고, 열 이름을 정리하고 전류 단위와 부호를 변환했습니다. 장비 원시 파일은 Zenodo에서 받을 수 있습니다. [원자료](https://zenodo.org/records/10637534) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [전처리와 원본 해시](data_provenance.json)

`data_adapter.py`로 다시 전처리하려면 원본에서 추출한 두 방전 CSV와 두 장비 파일을 [출처 기록](data_provenance.json)의 파일명으로 배치하고 `--source-dir`로 경로를 전달합니다. `compare_runs.py --help`에서 두 실행 결과를 대조하는 방법을 확인할 수 있습니다.
