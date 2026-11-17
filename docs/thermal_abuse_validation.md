# 열폭주·화재 확장 검증

현재 SPMeT의 열 계산은 운전 중 셀 평균온도를 계산한다. 열폭주 확장에는 대상 셀의 분해반응 발열 계수와 열적 경계, 독립 시험이 추가로 필요하다. 화재 평가에는 화염·가스 배출·열방출 관측과 그에 대응하는 모델이 필요하다. [PyBaMM 열 모델](https://docs.pybamm.org/en/stable/source/examples/notebooks/models/thermal-models.html)

`simulation/thermal_abuse_validation.py`는 실측 온도·온도상승률의 지속 조건을 검사하고 예측 파형을 비교한다. 관측 위치, 시간 원점, 셀 종류와 프로토콜이 맞아야 비교하며, 자료 공백을 보간해서 채우지 않는다. 실측 파일과 메타데이터의 SHA256을 보관한다. 온도가 기준을 넘어도 열폭주·화재 라벨은 별도 실험 근거가 있을 때만 기록한다.

입력 CSV 열은 `time_s`, `temperature_C`, 선택적으로 `rate_C_per_min`, `segment_id`다. 상승률을 생략하면 공백과 시험 구간 경계를 제외하고 후방 차분으로 구한다. 메타데이터 JSON에는 `kind`, `source_url`, `cell_model`, `protocol`, `temperature_observable`, `time_origin`, `trace_complete`가 필요하다. 예측에는 `model_id`, `parameters_sha256`, `used_for_fit: false`를 추가한다. 보정에 쓰지 않은 시험과 동결 파라미터의 예측을 비교한다.

```powershell
python -m simulation.thermal_abuse_validation --measured measured.csv --metadata measured.metadata.json --predicted predicted.csv --prediction-metadata predicted.metadata.json --temperature-C 60 --rate-C-per-min 1 --persistence-s 5 --max-gap-s 2 --criterion-source "프로젝트에서 정한 연구 지표" --output comparison.json
```

명령의 수치는 실행법을 보이기 위한 예다. 실제 적용 기준은 대상 실험의 정의와 허용오차로 정한다. 출력의 온도 RMSE, 표본 커버리지, 조건 충족 시점 오차는 개별 비교 지표다. 열폭주 모델의 합격은 별도의 독립 시험과 승인된 수치 기준으로 판단한다.

## 2026-11-15 확인

공개 ARC 자료의 NMC·SOC 100% 반복 시험 두 건을 받아 공개 MD5와 대조했다. 각각 105·125개 온도·상승률 표본을 CSV로 변환했다. 원본은 발열 구간만 제공하므로 `trace_complete=false`다. 제조사는 익명이며 LG M50T와의 동일성은 미확인이다. [공개 데이터](https://zenodo.org/records/7707929)

원 논문의 열폭주 판정은 장비의 기록 간 온도 변화에 기반한다. 코드에서 예시로 추출한 1°C/min 도달 지점은 연구용 상승률 지표다. 관측 화재 라벨과 LG M50T 열폭주 검증 상태는 미확인으로 남겼다. [원 논문](https://doi.org/10.3390/batteries9050237)

NASA/NREL Battery Failure Databank에는 LG 21700-M50(BV) 열폭주 시험이 포함되어 있다. 후속 실측 보정에 사용하려면 원시 파형, 시험 조건, 셀 형식과 M50T의 대응 관계부터 확인한다. [NASA 자료](https://www.nasa.gov/wp-content/uploads/2024/01/battery-failure-databank.pdf)

실행 검사는 `python -m unittest simulation.test_thermal_abuse_validation -v`이며, 단위 변환·누락 구간·판정 지속시간·관측 근거·비교 조건·학습 누출 검사 10개가 통과했다.
