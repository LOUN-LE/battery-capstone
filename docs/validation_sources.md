# 캡스톤에서 검증에 쓴 논문과 데이터셋

2026-11-16 · 실행 결과, 입력 출처와 파일 해시가 남아 있는 작업을 기준으로 정리.

**현재 LG M50T의 짧은 방전 전압을 확인한 직접 출처는 Kirkaldy 외(2024)의 Imperial College London 실측 데이터다.** 논문은 어떤 실험을 했는지 설명한다. 공개 데이터의 방전 전 기록은 시작 상태를 추정하는 데 썼고, 방전 전류를 모델에 입력한 뒤 같은 구간의 실측 전압으로 오차를 계산했다.

## LG M50T 디지털 트윈과 초기 배터리 모델 비교

| 출처 | 프로젝트에서 실제로 한 작업 | 확인 범위 |
| --- | --- | --- |
| Kirkaldy 외(2024), *Lithium-ion battery degradation: Comprehensive cycle ageing data and analysis for commercial 21700 cells*. [논문 PDF](https://spiral.imperial.ac.uk/server/api/core/bitstreams/de5c960b-dd9a-4e19-88f8-f66604a9d760/content) · [원본 데이터](https://zenodo.org/records/10637534) | **실험 2,2의 25°C 초기 성능 시험**. 저율방전과 사전 충전·휴지 기록으로 모델을 보정하고, 같은 제품의 다른 개체에서 방전 전압을 평가. 최신 공통 센서 CSV 1,200행과 디지털 트윈 전달 CSV 591행을 이 자료에서 만들었음. | 최신 591행 구간 전압 RMSE 4.423mV. 25°C 등온 가정이며 물리 정확도 합격선은 미정. 로컬·AWS 대조는 프로그램의 재현성 확인. |
| 위 Kirkaldy 데이터, 실험 2,2 장기 성능 시험 | 정기 방전 용량 기록으로 별도의 단순화된 열화 모델이 용량 유지율(SOH)을 얼마나 재현하는지 평가. | 같은 25°C 평가 개체는 RMSE 0.573%p로 기본 기준 통과. 10°C 두 개체는 2.490·3.250%p로 기준 미달. 이 25°C 비교에서 SOH 80% 도달은 관측되지 않았음. |
| 위 Kirkaldy 데이터, 실험 4와 5 | 실험 4의 저율방전·전류 펄스로 새 개체의 전압과 단기 온도를 점검. 실험 5의 10·40°C 첫 1C 방전으로 전압·용량을 비교하고, 별도의 장기 요약 자료로 온도별 수명 기준선을 평가. | 펄스 온도 시험은 특정 냉각 장치와 구간에 한정. 실험 5의 8개 셀로 옮긴 단순 수명 기준선은 기준 미달. [원저자 실험 주석](https://zenodo.org/records/7235858) |
| Chen 외(2020), *Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models*. [논문 PDF](https://pure-oai.bham.ac.uk/ws/files/96660426/Chen_2020_J._Electrochem._Soc._167_080534.pdf) · [원본 데이터](https://zenodo.org/records/4032561) | 공개 LG M50 자료 중 **LGM50_cell02.csv**의 첫 C/10 방전을 초기 모델에 대조했음. | 당시 단순화 모델의 전압 RMSE 약 75.0mV. 위의 최신 LG M50T 연결 시험과 모델·전처리·조건이 다름. |

Imperial 원본 저장소에는 논문에 실린 **LG M50T 40개 셀**의 자료가 있다. 우리가 수행한 각 실험의 대상은 위에 나온 구간과 개체로 확인한다. [재현 방법과 입력 CSV](../bms_pipeline/README.md)에 포함된 파일과 입력 규격을 확인할 수 있다.

## 함께 진행했던 별도 검증

| 출처 | 실제로 확인한 것 |
| --- | --- |
| Tam 외(2026), NIST 외부 가열 실험. [논문 원문](https://www.nist.gov/publications/examining-safe-intervention-windows-18650-and-21700-lithium-ion-batteries-under-heat) · [실측 데이터](https://data.nist.gov/od/id/mds2-4027) | 전체 77건에서 NMC 21700의 발화 판정이 확인된 33건으로 별도 딥러닝 예비 학습·평가. 특정 외부 가열 시험의 결과다. |
| Ohneseit 외(2023), KIT 가속열량계(ARC) 실험. [논문](https://doi.org/10.3390/batteries9050237) · [실측 데이터](https://zenodo.org/records/7707929) | 익명 NMC 21700의 반복 두 건에서, 첫 시험으로 맞춘 온도 상승률 식을 두 번째 시험에 적용. 발열 구간의 온도 상승률을 비교. |
| NASA·NLR Battery Failure Databank. [데이터와 원본 워크북](https://www.nlr.gov/transportation/battery-failure) · [NASA 소개 자료](https://www.nasa.gov/wp-content/uploads/2024/01/battery-failure-databank.pdf) | LG M50(BV) 25건의 방출 에너지·질량을 추출하고 원본과 대조. M50T 동일성 및 발화 라벨은 확인되지 않았고, 프로젝트 모델 정확도 평가에는 사용하지 않았음. |
| Yasko 외(2025), KU Leuven BEV 실측. [논문](https://www.nature.com/articles/s41597-025-06148-5) · [데이터](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi%3A10.48804%2F8KPDTW) | 별도 주행 연구에서 차량 **팩 전류** 예측에 사용. 단일 셀 전압·수명 검증과 범위가 다름. |
| 팀의 M50T 팩 시뮬레이션 · 자체 AWS 32건·1,000건 | 팩 입력 변환 및 계산 연결, 시나리오 실행·파일 무결성·수치 일관성을 확인. 32건·1,000건의 온도·도금 결과는 계산으로 만든 값이다. |

## 모델 구현과 실험 설계에 참고한 원문

- **O’Kane 외(2022)** — PyBaMM `OKane2022`와 열화 모델의 출처. 우리 실행은 보호막 성장과 일부 가역 도금 등을 선택한 구성이다. [논문](https://doi.org/10.1039/D2CP00417H) · [저자 공개본](https://arxiv.org/abs/2112.02037)
- **O’Regan 외(2022)** — 공개 전극 확산 함수를 전압 오차의 원인을 점검하는 별도 진단에 시험 적용. [논문](https://doi.org/10.1016/j.electacta.2022.140700)
- **Moura 외(2017), Brosa Planella·Widanage** — 전해액 변화를 포함한 단일입자 모델과 부반응을 추가한 단순화 모델의 근거로 검토. [Moura 논문](https://doi.org/10.1109/TCST.2016.2571663) · [Brosa Planella·Widanage 논문 공개본](https://arxiv.org/abs/2206.05078)
- **Li 외(2025)** — 수명 모델에서 열화 원인별 평가를 어떻게 하는지 검토. 저자 결과·스크립트와 파라미터를 검토했고, 출판된 계수를 현재 모델에 채택하지 않았음. [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11926349/) · [공개 자료·코드](https://zenodo.org/records/14995785)
- **Severson 외(2019), Attia 외(2020), Paradis·Whitmeyer(2019)** — 초기 데이터로 수명을 예측하는 방법과 학습·시험 분할을 검토. 팀 노션에 Stanford/MATR의 LFP 124셀 전처리가 기록돼 있다. 이 문서는 팀 전처리 기록을 근거로 하며 원시 파일을 재실행한 결과까지 포함하지 않는다. [Severson 논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf) · [Attia 논문](https://doi.org/10.1038/s41586-020-1994-5) · [Paradis·Whitmeyer 논문](https://arxiv.org/abs/1910.01347) · [Stanford/MATR 데이터](https://data.matr.io/1/)
- **Bosch 공개 발표·특허** — 클라우드 연결과 셀 선별 구조를 참고. 프로젝트의 실측 비교값은 Imperial 자료에서 얻었다. [발표](https://tbse24.mapyourshow.com/mys_shared/tbse24/handouts/1430_BatteryInTheCloudCloudBased_Huelsing.pdf) · [특허](https://patents.google.com/patent/US12560654B2/en)
- **Zhang 외의 LG M50 합성 데이터** — 열화 데이터 설계 검토에서 참고. 결과는 물리 시뮬레이션으로 생성됐다. [공개 데이터](https://data.mendeley.com/datasets/ry6g9cc5bw/2)
- **열폭주 연구의 추가 출처** — LG M50T 가열·자가발열 연구 및 Warwick 21700 열남용 연구는 적용 가능성을 검토했다. [TUM 논문](https://doi.org/10.1016/j.jpowsour.2024.235064) · [Imperial 논문](https://doi.org/10.1016/j.apenergy.2025.125435) · [Warwick 논문](https://doi.org/10.1016/j.psep.2022.02.014)
