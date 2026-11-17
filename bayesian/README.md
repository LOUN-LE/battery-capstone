# 캡스톤 베이지안

| 파일 | 하는 일 |
| --- | --- |
| [driving.py](driving.py) | 같은 조건의 과거 주행으로 개인의 다음 셀 전류 곡선과 불확실성을 예측 |
| [driving_twin.py](driving_twin.py) | 기준 패턴, 예측 평균, 예측 분포에서 뽑은 곡선을 같은 SPMe 시작 상태로 평가 |
| [scenario_gp.py](scenario_gp.py) | 가상 경로·교통 조건별 기준 전류를 만들고, 대상 운전자와의 차이를 GP로 학습·평가 |
| [scenario_model_selection.py](scenario_model_selection.py) | 동일한 가상 조건에서 10개 예측법을 검증하고 새 주행으로 선택 결과를 확인 |
| [scenario_comparison.py](scenario_comparison.py) | 기준 운전, GP 예측, 학습에서 뺀 가상 주행을 공통 조건의 SPMe에 입력 |
| [habit_stress.py](habit_stress.py) | 같은 가상 도로에서 가속·제동 습관에 따른 전류 부담과 GP 예측 오차 비교 |
| [habit_stress_twin.py](habit_stress_twin.py) | 같은 초기 상태의 SPMe에서 네 가지 운전 전류를 재현하고 전압·보존 검사 |
| [habit_predictor.py](habit_predictor.py) | 이전 시험에서 오차가 가장 작았던 이차 회귀로 네 가지 전류 지표의 변화를 예측 |
| [habit_life.py](habit_life.py) | 같은 가상 운행을 반복할 때 공개 LG M50 셀 모델의 장기 용량을 비교 |
| [habit_waveform.py](habit_waveform.py) | GP와 이차 회귀로 기준 운전 대비 전류 변화량을 시간별로 예측 |
| [habit_life_benchmark.py](habit_life_benchmark.py) | 예측·원본 전류로 계산한 수명 차이를 비교하고 검증 자료로 설정 탐색 |
| [charging.py](charging.py) | 고정 SOC 목표까지 CC–CV 충전 후보를 평가하고 기존 BO 엔진으로 다음 전류 선택 |
| [pack_input.py](pack_input.py) | 팀의 팩 기준 6열 CSV와 메타데이터를 검사하고 셀 환산값을 추가 |
| [pack_replay.py](pack_replay.py) | 팩 원자료의 짧은 구간을 기존 SPMe에 입력하고 결과 JSON 저장 |
| [__main__.py](__main__.py) | 입력 준비·SPMe 재현·주행 예측·충전 탐색을 같은 명령에서 선택 |

## 팀 입력 규격 — 2026-11-09

사용자가 제공한 `raw_input.csv`와 `metadata.json`은 팩 기준 `pack_raw_six_v1` 규격이다. 시간은 시뮬레이션 초, 전압은 V, 전류는 A이며 방전이 양수, 충전·회생이 음수다. 온도는 고정 모듈 센서 `module_01_temperature_c`의 °C 값이다.

```text
vehicle_id,cycle_id,timestamp_s,voltage_v,current_a,temperature_c
```

먼저 `prepare-input`으로 검사·변환한다. 원본의 가변 시간 간격과 서로 다른 cycle의 경계행을 보존하고, 원래 6열에 셀 환산값과 cycle별 경과 시간을 추가한다. 결과는 `normalized_cell_input.csv`와 `input_report.json`이다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian prepare-input --csv 'data/raw_input.csv' --metadata 'data/metadata.json' --pack-config configs/team_pack_108s32p.json --output ..\work\team_input_20261109
```

현재 팀 팩 구성은 **108S32P**이며 별도 설정으로 전달한다. 첨부 메타데이터에는 직렬·병렬 수가 없고, 구성의 출처는 팀의 M50T ROM 결과보고서다. 전압÷108, 전류÷32는 균등 분담을 가정한 대표 셀 환산으로 실제 셀별 상태나 물성 일치를 입증하지 않는다. [입력 규격과 처리 범위](../docs/team_pack_input.md)에 자세한 기준을 정리했다.

이 자료의 `cycle_id`는 운행 기록 구분이며 EFC나 주행 예측기의 `trip_id`가 아니다. 운전자·경로·교통·절대 시작 시각이 없어 GP 학습용 주행 CSV를 자동 생성하지 않는다. `prepare-input`은 디지털 트윈 시뮬레이션이나 팀원의 학습 모델 전처리를 실행하지 않는다.

## 바로 실행

저장소 루트에서 기존 캡스톤 가상환경을 사용한다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian driving --output ..\work\my_bayesian\driving.json
& ..\.venv\Scripts\python.exe -m bayesian charging --current-a 3.0 --output ..\work\my_bayesian\charging_baseline.json
& ..\.venv\Scripts\python.exe -m bayesian charging --budget 5 --output ..\work\my_bayesian\search
```

주행과 충전 기준 실험은 새 JSON 파일에 저장한다. 충전 탐색은 출력 폴더에 저널과 결과를 저장하며 같은 경로와 설정으로 재개한다. `--budget`은 누적 평가 수다. 단일 셀 전류 A와 에너지 Wh가 보고서에 기록된다.

## 팀 원자료 → SPMe → 결과 JSON

`replay-input`은 같은 원자료·메타데이터·팩 설정을 검증한 뒤 선택한 cycle의 시작 구간을 실제로 계산한다. `prepare-input` 출력물을 다시 입력할 필요는 없다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian replay-input --csv 'data/raw_input.csv' --metadata 'data/metadata.json' --pack-config configs/team_pack_108s32p.json --cycle-id 1 --duration-s 300 --output ..\work\team_replay_20261109
```

출력 폴더는 새 경로여야 한다. 요청한 시간 이전의 마지막 실제 표본을 끝점으로 선택하며 가상 표본을 만들지 않는다. 제공 자료에서는 300초 요청에 318행·317구간, 실제 299.542407초가 선택된다.

기존 `configs/model.json`의 OKane2022 등온 SPMe를 사용한다. BMS SOC 80%를 균일한 새 모델의 초기 SOC 0.8로 대응시키고, 첫 모듈 센서값 20°C를 고정 온도로 사용한다. 이 초기 상태는 C 셀 보정 상태에서 가져온 것이 아니며, 센서 온도가 실제 셀 코어 온도와 일치한다고 가정한 물리 검증도 아니다. 전류는 구간 오른쪽 끝 값을 유지하며 구간 사이의 미분 상태와 용량 상태를 이어간다.

실제 실행은 `outputs/spme_replay_20261109_v2/result.json`에 저장됐다. 전 구간 완주·입력 전류·상태 연속성·전하 보존을 확인했다. 관측 팩 전압을 셀 기준으로 나눈 값과의 RMSE는 19.876mV, 전체 전하 잔차는 약 −5.36×10⁻¹⁶Ah다. 반복 실행에서도 317개 구간의 결과와 요약이 완전히 일치했다. 재현 8개·입력 12개·주행 5개·BO 17개를 함께 실행한 관련 검사 42개도 통과했다. 이 비교의 원자료는 팀 ROM 출력이며 실차 수명 검증이나 모델 적합성 합격을 뜻하지 않는다.

JSON에는 입력·설정·코드 해시, 모델·솔버 조건, 초기화·온도 가정, 구간별 전류·전압·용량과 검증 결과가 포함된다. SOH와 잔여 수명은 계산하지 않아 `null`로 남긴다.

## 운전 습관 예측과 비교

예측기는 경로·방향·교통 조건·거리가 비슷한 주행을 고른다. 참조 차량의 평균 전류 곡선에 개인 기록으로 추정한 차이를 더하고, 시점 간 공분산으로 전체 전류 곡선을 함께 샘플링한다. 과거 주행만 사용해 이후의 주행을 통째로 예측한 뒤, 개인 과거 평균과 RMSE·MAE 및 예측구간 포함률을 비교한다.

기본 명령은 합성 주행 자료를 사용한다. `driving --csv`는 아래의 별도 주행 이력 규격을 받는다. 팀의 팩 기준 6열 CSV나 `prepare-input` 결과를 직접 넣는 옵션이 아니다. 실제 주행 구간·운전자·경로 정보를 확보하고 같은 물성의 계산 셀 기준을 확인한 뒤 작성한다.

```text
vehicle_id,driver_id,trip_id,starts_at,route_id,direction,traffic,distance_m,cell_basis_id,elapsed_s,cell_current_A
```

`starts_at`과 `--as-of`는 UTC 오프셋을 포함한다. `elapsed_s`는 0부터 시작하며 전류는 방전이 양수, 회생이 음수다. 주행 CSV 판독기는 10초가 넘는 표본 간격을 거부한다. 이 제한은 가변 간격의 팩 원자료를 읽는 `prepare-input`과 다르다. 정규화 단계에서 차량·운전자·셀 기준과 주행 누락을 확인한다. 주행 트윈 실행기의 셀 기준은 `okane2022-spme-single-cell`이다. 108S32P 팩의 대표 셀 환산만으로 이 모델의 물성과 초기 상태가 맞는다고 간주하지 않는다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian driving --csv trips.csv --target vehicle-1 --reference vehicle-2 --route downtown --direction outbound --traffic moderate --distance-m 3000 --duration-s 320 --as-of '2026-11-06T09:00:00+09:00' --output ..\work\my_bayesian\csv_forecast.json
```

여러 운전자가 같은 차량을 쓰는 경우 `--target-driver`, `--reference-driver`를 지정한다. 트윈은 기준 패턴과 각 샘플의 셀 에너지, 전압 범위, 표준 용량 시험(RPT) 용량을 출력한다. 주행 종료 SOC가 달라지면 RPT 자체의 충전 이력도 달라진다. 용량은 진단값으로 남기고 이 데모에서는 주행 열화 순위를 매기지 않는다.

## 가상 주행 조건별 GP 재비교

```powershell
& ..\.venv\Scripts\python.exe -m bayesian compare-scenario-driving --output ..\work\scenario_comparison
```

새 출력 폴더를 지정한다. 가상 도로와 교통 조건을 먼저 고정하고, 같은 조건에서 생성한 기준 운전 이력으로 전류 평균을 만든다. 대상 운전과의 차이를 기존 GP 계산으로 학습한다. 학습에서 뺀 주행으로 조건별 기준 평균, 대상 운전자의 과거 평균, GP 보정을 비교한다. `--gp-only`를 붙이면 전류 예측 평가까지만 실행한다.

`scenario_gp.json`은 예측 오차와 시험 분할을 기록한다. SPMe 단계는 미리 정한 시험 주행 하나의 생성 전류와 GP 예측 전류를 같은 시나리오의 기준 운전과 비교한다. `scenario_comparison.json`에는 두 격자의 기준 운전 대비 RPT 용량 차이, `spme/driving_comparison.json`에는 주행·복귀·휴지 단계의 검사 결과가 있다. 순전하량이 다른 경우 기존 동일 수행량 순위는 보류한다. RPT 차이는 가상 셀의 전체 비교 절차에 해당한다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian compare-scenario-models --output ..\work\scenario_model_selection
```

이 명령은 조건별 평균·중앙값·절사평균·평활 평균·GP 등 10개 방법을 같은 학습 자료로 만든다. 검증 주행의 전류 RMSE가 가장 낮은 방법을 선택한 뒤 별도 시험 주행에서 오차를 다시 측정한다. 학습·검증·시험의 생성 시드는 서로 겹치지 않는다. `scenario_model_selection.json`에서 두 오차표, 선택 모델, 최대 방전 전류 오차와 순전하량 오차를 확인할 수 있다. 디지털 트윈에는 선택 모델의 예측을 연결한다. `--predictions-only`를 붙이면 예측법 비교만 실행한다.

## 운전 습관별 배터리 부담과 GP 예측

```powershell
& ..\.venv\Scripts\python.exe -m bayesian compare-habit-stress --output ..\work\habit_stress
```

가상 도심 도로와 보통 혼잡도에서 기준 운전, 빠른 출발, 급한 제동, 두 습관을 함께 적용한 운전을 비교한다. 한 쌍의 운행은 차량, 거리, 신호 대기 일정, 보조 전력 부하를 공유한다. 같은 거리를 같은 시간에 가도록 순항 속도는 각 가속·제동 패턴에 맞춰 조정한다.

`habit_stress.json`은 최대 방전·회생 전류, 전류 제곱 누적값, 절대 전하량을 기록한다. GP는 가속·제동 시간에서 각 지표의 기준 대비 차이를 직접 예측한다. 학습과 시험은 환경 생성 시드와 가속·제동 시간 조합을 분리한다. 전류 제곱 누적값은 고정 저항을 가정할 때 발열 부담을 비교하는 지표다.

`habit_stress_spme.json`은 미리 지정한 한 가상 운행을 초기 SOC 80%, 25°C 등온 모델의 두 격자에서 재현한 결과다. 전압 범위와 보존 검사를 저장한다. `--metrics-only`는 전류 지표와 GP 평가까지 실행한다. 출력 폴더는 새 경로를 지정한다.

운전 습관의 전류 부담을 예측하려면 다음 명령을 사용한다. 가속 8~16초, 제동 8~15초 사이에서 기준 대비 네 지표의 차이를 출력한다.

```powershell
& ..\.venv\Scripts\python.exe -m bayesian predict-habit-burden --launch-ramp-s 10 --brake-ramp-s 11 --output ..\work\habit_prediction.json
```

장기 비교는 [BLAST-Lite 1.1.0의 LG M50 셀 모델](https://github.com/NatLabRockies/BLAST-Lite/releases/tag/v1.1.0)을 사용한다. BLAST-Lite와 NumPy 2.3.5를 별도 환경에서 준비한다. 기존 SPMe 가상환경의 NumPy는 변경하지 않는다.

```powershell
python -m bayesian compare-habit-life --habit-report ../work/habit_stress/habit_stress.json --output ../work/habit_life
```

같은 32개 가상 운행을 하루에 세 번씩 순서대로 반복해 각 습관에서 93.12km를 운행한다. 온도 25°C, 출발 SOC 80%, 8시간 시점 1.5A 재충전 시작 규칙을 공통으로 둔다. 용량이 줄수록 동일 주행 전하량에 필요한 SOC 변동 폭을 늘리고, 하루 1회와 2회 갱신을 비교한다. 80%를 처음 통과하면 앞뒤 날짜로 저장하고, 20년 안에 도달하지 않으면 그 기간 안의 미도달로 남긴다.

이차 회귀의 네 지표는 부담 안내에 쓰인다. 장기 계산은 가상 전류 원곡선을 적분한 SOC 이력을 사용한다. BLAST-Lite의 M50 사이클 피팅은 낮은 방전율의 얕은 가상 반복을 외삽하며 순간 피크와 충전율의 열화 효과를 따로 모형화하지 않는다. 출력 수명은 가상 셀·운행 정책을 적용한 장기 추정치다.

GP와 이차 회귀의 **전체 파형**을 수명까지 비교하려면 같은 BLAST-Lite 환경에서 다음 명령을 실행한다. 이 실험은 기존의 피크 등 네 지표 회귀와 별개로 시점별 전류 변화량을 학습한다.

```powershell
python -m bayesian compare-habit-life-models --output ../work/habit_life_model_benchmark
```

같은 학습 기준 파형에 모델별 변화량을 더한다. 기준보다 더 끌어온 방전량과 회생까지 포함한 순전하량 변화는 따로 기록한다. 검증 조합 여섯 개의 수명 차이 RMSE를 기준으로 모델마다 설정 열 개를 평가하고, 시험 조합 열여섯 개는 설정을 고정한 뒤 계산한다.

원본 비교는 각 시험 조건의 가상 전류 32개를 하루에 세 번 반복한다. 예측 비교는 모델이 만든 대표 파형을 하루 96번 반복한다. 두 비교 모두 차량·운행 거리·온도·충전 정책을 맞추고, 각각의 기준 운전에 대한 용량 80% 도달일 차이를 계산한다. 결과 JSON과 압축 배열에 시뮬레이터 기준값, 예측 파형 및 시험 원본을 저장한다.

## 충전 전류 선택

단일 셀 예비 실험은 25°C, 시작 SOC 20%, 목표 SOC 80%, 60분 제한에서 3.0–3.3 A를 탐색한다. 충전 후 SOC 20%까지 같은 방전 조건을 적용하고, 공통 시점까지 휴지한 뒤 RPT를 수행한다. 기존 BO 엔진은 용량 손실의 기대 개선량과 충전 시간 제약 충족 가능성을 함께 써서 다음 후보를 고른다.

두 기능의 실행 예제는 저장소에 있는 OKane2022 등온 SPMe 가상 셀을 사용한다. 실차 팩 적용과 LG M50T 보정 모델 연결은 셀 물성, 시작 상태, 전류 분담, 독립 검증 자료가 맞춰진 다음 단계다. 실차 충전에서는 출발 시각·필요 에너지·충전기 한도를 입력으로 받아야 한다.


## 기존 실행 근거

`../work/bayesian_demo_20261106/`의 `driving_report.json`, `charging_search.json`, `charging_baseline.json`에 실제 실행 결과가 있다. 주행은 합성 데이터 예측과 SPMe 10개 궤적, 충전 탐색은 4개 평가 중 시간 제약을 만족한 3개 후보를 완료했다. 베이지안 관련 자동 검사 22개 통과는 당시 기록이다. 11월 5일의 ‘최적화 미실행’ 안내는 이 실행 전의 이력이다.

충전 결과는 평가한 후보 중 최선값을 나타내며 전역 최적해나 실차 권장값은 아니다. `recommendation=null`과 독립 수명 검증 미완료 상태를 유지한다.
