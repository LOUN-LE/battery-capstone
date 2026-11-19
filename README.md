# 배터리 디지털 트윈 · 수명 계산

PyBaMM의 LG M50 계열 모델을 바탕으로 배터리 상태를 재현하고, 운전 습관과 충전 전류가 배터리에 미치는 영향을 분석한다. 첫 100 EFC 데이터로 예측하는 LSTM은 팀원이 개발한다.

## 캡스톤 구현별 코드

2026-11-16에는 공개 LG M50T 실측 기록으로 머신러닝 선별부터 디지털 트윈 계산까지 연결했고, 로컬과 AWS에서 결과를 대조했다. [재현 방법과 입력 CSV](bms_pipeline/README.md), [검증에 쓴 논문과 데이터셋](docs/validation_sources.md), [AWS 시나리오 예제](examples/scenarios/README.md)에서 관련 자료를 확인할 수 있다.

| 작업 | 주요 코드 | 실행 |
| --- | --- | --- |
| **SPMeT 디지털 트윈과 GP** | [팩 입력 재현](bayesian/pack_replay.py), [전압 선별 전송](bayesian/gp_telemetry.py), [펄스 보정](simulation/gp_pulse_calibration.py), [리튬 손실 보정](simulation/gp_internal_state.py) | `python -m bayesian replay-input --help`, `python -m bayesian.gp_telemetry --help`, `python -m simulation.gp_pulse_calibration --help`, `python -m simulation.gp_internal_state --help` |
| **운전자 습관 이차 회귀** | [전류 부담 지표 예측](bayesian/habit_predictor.py), [전류 파형 예측](bayesian/habit_waveform.py), [수명 비교](bayesian/habit_life_benchmark.py) | `python -m bayesian predict-habit-burden --help`, `python -m bayesian compare-habit-life-models --help` |
| **충전 전류 베이지안 최적화** | [CC–CV 평가와 탐색](bayesian/charging.py), [베이지안 최적화 엔진](optimization/bayesian_charge.py), [반복 검증](bayesian/charging_stability.py) | `python -m bayesian charging --help`, `python -m bayesian charging-stability --help` |

팩 재현은 팀 ROM 입력을 사용한다. GP 전압 전송은 재현 결과 JSON을, 펄스 보정은 실측 GITT 자료를, 리튬 손실 보정은 셀별 기준 성능시험(RPT) 자료를 각각 읽는다. 운전 습관 이차 회귀는 같은 가상 도로·혼잡도에서 전류 부담과 파형을 비교하며, 충전 최적화는 충전 시간 제약을 만족하는 셀 전류 후보를 탐색한다. 입력 형식과 실행 예시는 [실행 안내](bayesian/README.md)에 있다.

## 2026-11-19 실측 셀 장기 재현 결과

LG M50T D 셀의 초기 성능시험부터 첫 노화 세트 뒤 성능시험까지 3,940단계 재현을 완료했다. 모델 용량 감소는 0.02337 Ah, 실측 감소는 0.10504 Ah였다. 두 성능시험의 용량 유지율 오차 RMSE는 1.198%p로 기존 기본 수치 기준을 넘었다. 실제 열화 원인과 수명 검증은 진행 중이다.

짧은 구간에서는 보호막 성장·리튬 도금의 네 조합을 여섯 번 실행했고, 장기 구간은 기준 조합 한 개를 완료했다. [실행 코드와 재현 조건](experiments/team_six_outputs_20261117/README.md), [실험 과정·시간·결과](docs/d_mechanism_review_20261119.md), [전달용 셀 CSV와 열 매핑](examples/team_single_cell_20261117)을 함께 확인할 수 있다.

## 2026-11-04 시점의 검증 기록

2026-11-04 기준, **LG M50T 실측 검증과 코드 수정 중이며 장기 수명 실험은 재개하지 않았다.** 이전 셀4 장기 결과(약1,894 EFC·SOH98.84%)는 사용자의 지시에 따라 수명 예측·재개 상태·정답 라벨에서 제외한다. 아래 10월 계산표와 장기 설정은 과거 이력이다.

- 같은 Ns 안의 CC·휴지를 하나의 CC로 축약하던 입력 오류를 수정했다. MPR164개를 원시 mode·Ns·clock 단위로 재구성했고, C 약10.092일·D 약1.21시간의 휴지를 복원했다. 시간 합·원본 해시 검증은 통과했으며 미관측 공백163곳은 남아 있다.
- 초기 휴지 마지막 경계 행의 전압을 평균하던 오류도 수정했다. 120초 제어 구간 내부 표본을 사용한 유효 초기 SOC는 retained v3 재료에서 약0.369662다. 과거0.393248 초기화는 새 진단에서 거부한다.
- 정정한 초기 상태의 두 격자 단기 SOH 차이는0.000001607%p다. 이는 짧은 수치 선별이며 전체 실측 SOH·수명 합격이 아니다. CC·CV 동특성과 저율 재충전·C/2 검증을 진행 중이다.
- 실측 합격선은 셀별 SOH RMSE≤1.0%p와 최대 절대오차≤2.0%p 모두 충족이다. 같은 조건의 다른 셀에 이어 미학습 온도 또는 SOC 조건도 평가한다. EFC80은 실제80% 교차 관측이 있을 때 별도로 판정한다.
- AWS는 11월4일09:08KST 실제 조회에서 활성17리전 실행·기동·정지 진행 EC2가0대였다. 현재 장기 실행을 시작하는 명령이나 검증된 수명 모수를 제공하지 않는다.

정정 결과는 복원 작업 공간의 `work/validation_proposal_20261103/latest_spme_review_20261103/`에 보존한다. [단기 수치 검증](docs/short_spme_validation.txt), [조건부 결과의 사용 제한](docs/conditional_calibration_review.txt)을 함께 확인한다.

## 과거 계산 이력 — 현재 수명 검증에서 제외

2026-10-28 기준으로 네 고정 전류의 100 EFC 계산을 완료하였다. 같은 기준 셀 4번, 격자 ×4, 25°C 조건의 결과다.

| CC 전류 (A) | 100 EFC SOH (%) | 20→80% 충전시간 범위 (분) | 60분 초과 (회) |
| --- | --- | --- | --- |
| 3.0 | 99.790341 | 61.418–61.503 | 166 |
| 3.1 | 99.790729 | 59.525–59.587 | 0 |
| 3.2 | 99.791091 | 57.776–57.842 | 0 |
| 3.3 | 99.791443 | 56.165–56.206 | 0 |

3.0 A는 시간 제약 위반으로 추천 대상에서 제외한다. SOH 조건 차이는 최대 약 0.0011%p로 작다. 5 EFC의 R₁₀ 격자 비교를 통과했지만, 100 EFC SOH의 ×4·×8 대조와 실제 배터리 검증은 별도로 필요하다. 위 순위를 수명 연장 효과로 확정하지 않았다.

- 원본 806개 파일의 해시를 대조하고 백업하였다. 그중 Parquet 664개·총 4,270,651행의 읽기를 확인하였다. **원본 데이터·체크포인트는 공개 저장소에 올리지 않는다.**
- 장기 실행기의 로컬 검증은 약 152.05초로 통과하였다. AWS Linux에서도 실제 신호 처리와 기존 100 EFC 체크포인트 가져오기를 검사하였고, `100.43647188455 EFC`에서 검증용 일시 중지 상태를 확인하였다.
- **장기 본 계산: 2026-10-28 19:41 KST, 셀 4번·3.2 A 한 조건으로 시작하였다.** 짧은 재개 검증의 완료와 3,000 EFC 또는 SOH 80%까지의 계산 완료를 구분한다.
- 기존 100 EFC 계산 소스는 커밋 `777153d2f8b695fa12510dcd50ac5d3ec16249c8`이다. 이 절은 당시 실행 이력이며 현재 실행 상태는 위의 2026-11-04 절을 따른다.

## 폐기된 장기 예비실험 설정 — 현재 재개에 사용하지 않음

| 항목 | 확정 조건 |
| --- | --- |
| 대상 | 기준 셀 4번·CC 3.2 A 한 조건 |
| 물리 모델 | OKane2022, SPMe, 25°C 등온, 격자 ×4 |
| 열화 | 용매 확산 제한 SEI, 분포 SEI 저항, 부분 가역 리튬 도금, SEI·도금 기공률 변화 |
| 운전 | SOC 20→80% CC–CV 충전, 4.2 V 상한, 5 A 방전, 충·방전 뒤 각 300 s 휴지 |
| 기준시험 | 0·5·10·20·…·3,000 EFC. 100 EFC 이후에도 10 EFC 간격 유지 |
| 종료 사유 | SOH 80% 관측, 최대 3,000 EFC, 누적 실행시간 20시간, 자원 상한, 실패를 구분 |
| 재개 | 기존 100 EFC의 물리 상태·Q₀·장부·환경·소스 식별자를 검증하고 새 출력 폴더에서 연장 |
| 독립 실행 | systemd 감독, 결과 저장·동기화 후 서버 중지, 별도 20시간 15분 안전 타이머 |

기준시험 사이에서 SOH 80%를 통과한 정확한 EFC를 임의로 만들지 않는다. 마지막 80% 초과 관측과 최초 80% 이하 관측의 구간을 기록한다. SOH 80%에 도달하지 않은 채 3,000 EFC나 시간 상한에 걸리면 수명 종료로 처리하지 않고 검열 상태를 남긴다.

실행 조건·저장 상태는 [장기 실행기](docs/lifetime_runner.md), 독립 실행·메일 알림은 [클라우드 실행 제어](docs/lifetime_cloud_controls.md)에 있다. 중지 메일은 서버 상태 알림이며 실험 성공 여부를 뜻하지 않는다. 실제 종료 사유는 `outcome.json`과 `cloud_status.json`에서 확인한다.

## 모듈과 연결

| 위치 | 입력 | 출력·역할 |
| --- | --- | --- |
| `configs/model.json` | 모델·온도·SOC·전류·전압 설정 | 물리 계산 공통 조건 |
| `configs/cells_pilot9.json` | 입자 반지름 m·SEI 확산계수 m²/s | 셀 1~9 파라미터 |
| `configs/diagnostics.json` | 기준시험 전류·전압·휴지 | Q₀·Qref 측정 조건 |
| `simulation/battery.py` | 설정·이전 물리 상태 | SPMe, CC–CV, SOC 경계, 기준시험·R₁₀ |
| `simulation/run_100efc.py` | 셀·전류·격자·목표 EFC | 기존 초기 100 EFC 실행기 |
| `simulation/lifetime_policy.py` | 진단 간격·임계값·기존 식별자 | 진단 일정·EOL 구간·가져오기 검사 |
| `simulation/lifetime_operation.py` | 물리 상태·진행 단계·장부 | 방전 중간부터 이어가는 장기 제어 |
| `simulation/run_lifetime.py` | 기존 저장본 또는 초기 상태·실행 상한 | Parquet·진단·체크포인트·종료 결과 |
| `simulation/lifetime_supervision.py` | 실행기·프로세스 신호 | 자식 종료 대기·잠금 정리 |
| `scripts/run_lifetime_cloud.py` | 실행 조건·별도 결과 경로 | 자원 감시·축약 상태·서버 중지 |
| `scripts/launch_lifetime_service.sh` | 관리 세션의 실행 명령 | 채팅과 분리된 systemd 서비스 |
| `scripts/configure_stop_email.py` | 대상 인스턴스·수신 주소 | EventBridge·SNS 설정 또는 읽기 전용 확인 |
| `simulation/verify_*.py`, `scripts/verify_cloud_controls.py` | 같은 환경·비교 입력 | 수치·재개·신호·감독 흐름 검사 |
| `outputs/` | 실행 중 생성 | 데이터·로그·체크포인트; Git 추적 제외 |

기존 네 전류 배치는 `simulation/run_cloud_batch.py`와 `scripts/cloud_bootstrap.sh`를 사용하였다. 이 부트스트랩의 12시간 타이머는 이전 100 EFC 배치용이며, 새 장기 실행기의 20시간 상한과 구분한다. 기존 서버를 재시작하는 것만으로 새 장기 실행이 시작되지는 않는다.

## 환경과 데이터 정의

로컬 기준은 Windows Python 3.11.9, 실제 AWS 검증 환경은 Amazon Linux 2023·Python 3.11.17이다. 물리 계산 전용 고정 패키지는 `requirements-simulation.txt`를 사용한다. 다른 OS·Python 패치 버전에서는 설치와 짧은 재개 검증을 다시 수행한다.

```text
python -m pip install -r requirements-simulation.txt
```

SOC는 현재 사용 가능한 평형 용량 창의 충전 위치다. SOH는 `100 × Qref/Q₀`, EFC는 `본 운전 누적 방전 Ah/Q₀`다. 복제 상태의 기준시험에서 사용한 전하량은 본 운전 EFC에 더하지 않는다. 100 EFC는 부분 충방전 100회를 뜻하지 않는다.

본 운전은 1초 관측값과 사건 끝점을 저장하며 단계 경계의 중복 행을 포함한다. 전환 구간의 0.1초 출력과 기준시험의 명시적 1초 출력은 아직 적용하지 않았다. 기록된 SOC·EFC·진단값을 모두 LSTM 입력으로 사용한다는 뜻도 아니다. 학습 입력·정답과 팀원 전달 형식은 별도로 맞춘다.

체크포인트는 신뢰하는 실행기가 만든 파일만 읽는다. 기존 물리 소스·설정·환경 식별 검사를 우회하지 않으며, 원본 100 EFC 결과를 덮어쓰지 않는다. Windows 체크포인트를 Linux에서 직접 이어 계산하는 경로는 검증하지 않았다. 서버를 중지해도 EBS 등 보존 자원에는 비용이 남는다.

## 검증·실행 문서

- [기준시험·모델 결정 기록](docs/modeling_decisions.md)
- [대표 셀 R₁₀ 격자 검증](docs/mesh_gate_2026-10-28.md)
- [기존 실행기 검증](docs/runner_validation_2026-10-28.md)
- [초기 100 EFC 실행 계획](docs/cloud_run_plan.md)
- [장기 실행기와 검증](docs/lifetime_runner.md)
- [독립 실행·자동 중지·메일 알림](docs/lifetime_cloud_controls.md)
