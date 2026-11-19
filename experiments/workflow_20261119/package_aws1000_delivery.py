from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'work/aws1000_retrieved/sharing'
OUT = ROOT / 'outputs/aws1000_delivery_20261116'
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads((SRC / 'dataset_report.json').read_text(encoding='utf-8'))
for name, key in [('training_dataset.csv', 'training_dataset_sha256'),
                  ('scenario_summary.csv', 'summary_sha256'),
                  ('features.csv', 'features_sha256'), ('labels.csv', 'labels_sha256')]:
    data = (SRC / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == report[key], name
    (OUT / name).write_bytes(data)

readme = '''1,000건 CSV 전달 안내 (2026-11-16)

AWS 실행: 1,000건 완료 / 실패·중단·미완료 0건. 서버 중지 확인.
배치 경과시간: 48분 7.5초, 서버 1대에서 최대 8개 병렬. 순수 풀이 시간만을 뜻하지 않습니다.

먼저 볼 파일: scenario_summary.csv
- 입력 조건과 단기 SEI 손실·비활성 리튬·도금량의 시작/끝/증가량이 들어 있습니다.
- sei_loss_Ah_net_change: 해당 실행에서 SEI로 인한 용량 손실 증가량 [Ah]
- dead_lithium_mol_m3_net_change: 비활성 리튬 농도 증가량 [mol/m³]
- plating_inventory_Ah_net_change: 종료 시점 도금 inventory의 순변화 [Ah]. 비가역 손실량과 동일하지 않습니다.
- 위 결과값을 미래 조건 선별 모델의 입력 특성으로 넣으면 정답 누출입니다. 예측 목표로 사용합니다.

features.csv: 시나리오 입력과 고정 학습/검증/시험 분할.
labels.csv / training_dataset.csv: 어제 정한 온도·도금 선별 지표와 우선순위. 수명 위험 등급이 아닙니다.

중요: 이 1,000건은 신선한 가상 셀의 조건 조합이며 실제 BMS 시계열 1,000개가 아닙니다.
low/medium/high를 수명 감소의 크기로 해석하지 마세요. 40°C 온도 기준과 도금 여부를 센 값입니다.
동일 BMS 입력으로 기존 ML+DT와 새 DL+DT를 비교한 실험은 아직 수행하지 않았습니다.
시간비교 보고서에 실측값과 미측정 항목을 구분했습니다. 원본 CSV 값·분할·라벨은 변경하지 않았습니다.
'''
(OUT / '00_먼저읽기.txt').write_text(readme, encoding='utf-8-sig')

review = '''프로젝트 방향성 비판 검토
검토 대상: Battery_Digital_Twin_Project_Plan.pdf, 6쪽 전체
기준: 발화가 아닌 수명·열화 조건 선별 / 동일 BMS 입력에서 기존 ML+DT와 새 DL+DT 비교

판단: 방향은 타당합니다. 그러나 지금 문서는 개념 기획서이며 비교 실험을 재현할 명세는 부족합니다.

1. 비교 대상의 역할부터 같아야 합니다. (1·6쪽)
기존 ML이 셀을 고르고 새 DL이 그 셀의 미래 운용 조건을 고른다면 서로 대체하는 모델이 아닙니다.
같은 BMS 원본, 같은 관측 마감 시각, 같은 셀 집합을 넣고 전체 파이프라인을 비교하세요.
선택기 자체 효과를 보려면 공통 상류 셀 선별 이후 동일 상태·후보 풀에서 비교해야 합니다.
두 비교를 별도로 기록하고, 새 모델만 DFN·기존 모델은 SPMe인 비교는 피해야 합니다.

2. ‘물고기 협력’은 아직 알고리즘이 정의되지 않았습니다. (2·4·5쪽)
공유하는 수치, 점수 결합식, 불확실성 보정, 가지 보류·복구 조건, 계산 예산과 종료 조건을 적어야 합니다.
그렇지 않으면 여러 예측기와 조건문을 묶은 것보다 무엇이 개선됐는지 증명할 수 없습니다.
기존 ML, 단일 다중출력 모델, 무작위/층화 선택, 소통 없는 전문가를 같은 예산에서 비교해야 합니다.

3. 현재 1,000건은 같은 BMS 상태에서 만든 1,000가지 미래가 아닙니다. (3·6쪽)
온도 5 × 전류 5 × 초기 SOC 5 × 냉각 8 조합이며 열화 이력 없이 매번 초기화합니다.
초기 온도와 주변 온도가 함께 바뀝니다. 초기 SOC·온도를 고정하면 미래 후보는 전류 5 × 냉각 8 = 40개입니다.
동일 BMS 실험에서는 같은 관측 이력으로 상태를 준비하고 같은 체크포인트에서 각 후보를 독립 재시작해야 합니다.
동일한 BMS 파일만 쓰고 관측 구간·전처리·상태 추정·미래 후보를 다르게 쓰면 공정한 비교가 아닙니다.

4. 현재 자료로 5개 열화 전문가를 바로 학습할 수 없습니다. (2~5쪽)
현 SPMeT는 SEI·부분 가역 도금을 포함하지만 균열·LAM 하위 모델은 활성화되어 있지 않습니다.
파라미터 세트 이름이 OKane2022라는 것만으로 모든 메커니즘이 작동하지 않습니다.
새 결합 DFN에는 새 출력 정답과 격자·오차·실측 비교 검증이 필요합니다.
우선 SEI·도금 2개 목표로 시간/성능 비교를 하는 축소안이 현실적입니다.

5. 단기 열화 지표와 장기 수명을 연결하는 검증이 없습니다. (2~4·6쪽)
최대 30분 충전+5분 휴지의 SEI 증가나 도금 사건은 잔여 수명 정답이 아닙니다.
첫 목표를 ‘정해진 기간의 모형상 SEI 손실/비활성 리튬 증가량 예측과 조건 선별’로 한정하세요.
장기 수명 감소 주장은 반복 운용·용량 검사 또는 독립 수명 실험과 연결한 뒤 해야 합니다.

6. 어제 만든 우선순위도 수명 위험 등급으로 쓰면 안 됩니다. (이번 CSV 독립 점검)
- 도금 지표 936건, 온도 지표 200건, 어느 하나라도 해당 952건입니다.
- 온도 양성 200건 모두 초기 온도가 이미 40°C인 조건입니다.
- 도금 양성 936건 중 920건(98.29%)은 사건 시작 시각이 0초입니다.
- 동일 온도·전류·SOC의 125그룹 모두 냉각 8조건을 바꿔도 분류값이 같습니다.
- 학습 구간에서 온도별 다수 분류만 외운 단순 표도 시험 정확도 94.74%입니다.
- 검증 구간에는 low가 없고, 검증/시험 구간에 온도만 양성인 분류가 없습니다.
- high의 비활성 리튬 증가 중앙값은 medium보다 작습니다. 지표 개수와 열화량 순서는 다릅니다.
따라서 이 분류에서 높은 정확도가 나와도 딥러닝의 필요성이나 수명 예측력을 입증하지 못합니다.
분류값을 보기 좋게 다시 붙이지 않았습니다. 연속 열화 지표와 목표 정의를 먼저 검토해야 합니다.

7. ‘위험을 대부분 직접 확인하면서 호출을 크게 줄인다’는 목표의 수학적 제약도 봐야 합니다.
현재 정의에서 양성 952건의 95%를 DT로 직접 확인하려면, 완벽히 선별해도 최소 905건을 계산해야 합니다.
이때 줄일 수 있는 고유 호출은 최대 9.5%입니다. 시간 절감률은 각 시나리오 비용이 달라 별도 측정해야 합니다.
이 계산은 현재 온도/도금 proxy 정의에만 해당합니다. 새 수명 위험 목표의 양성 비율을 뜻하지 않습니다.
‘각 양성을 직접 확인하는 발견률’과 ‘계산하지 않은 후보까지 포함한 예측 정확도’를 혼동하면 안 됩니다.

유지할 부분
문서는 동일 물리 모델, 실제 원인 확정 금지, 미래 정답 누출 금지, 단일 AI/무소통 비교,
총시간 및 학습 비용 분리, 미계산=미검증, SEI·도금부터 시작하는 범위를 이미 제시합니다.
이 부분은 적절합니다. 발화 예측을 못 한다거나 모든 주장이 허위라는 식의 비판은 맞지 않습니다.

당장 필요한 비교
A: 전체 DT 기준선. B: 기존 ML+동일 DT. C: 새 DL을 추가한 방식+동일 DT.
같은 BMS 파일 SHA256·셀·관측 마감·입력 특징·상태/후보/정답·하드웨어로 짝을 맞춥니다.
각각 단계별 시간, 총 벽시계 시간, 고유 DT 호출 수, 목표별 누락률과 예측 오차를 기록합니다.
학습용 데이터 생성·학습 시간은 별도 기록하고 반복 사용 시 손익분기점을 계산합니다.
기존1000의 고정 분할은 696/152/152입니다. 같은1000을 모두 학습한 뒤 같은1000으로 성능을 주장하지 않습니다.
Bosch 원문/실제 기존 모델 구현은 이번 입력으로 특정되지 않았으므로 동일 구현 재현이라고 주장하지 않습니다.

근거
계획서 1~6쪽, 실제 생성 코드/설계, AWS 완료 기록, independent_data_audit.json.
PyBaMM 결합 열화 공식 예제:
https://docs.pybamm.org/en/v25.8.0/source/examples/notebooks/models/coupled-degradation.html
Imperial 원자료(LLI/LAM 추정과 실제 전압/용량 검증 역할): https://zenodo.org/records/10637534
Zhang 합성 자료(독립 실험 정답과 구분): https://data.mendeley.com/datasets/ry6g9cc5bw/2

검증 범위
CSV 1,000건의 고유성·입력·해시·라벨 규칙·분할을 로컬에서 재확인했습니다.
AWS 검증 기록은 원시 시계열 2,079,965행 검사 통과를 보고합니다.
전체 원시 시계열 파일은 대용량 다운로드 지연으로 이번 로컬 검사에서 다시 계산하지 않았습니다.
CSV/완료 기록은 내려받았고 서버가 stopped인 것을 직접 조회했습니다.
'''
(OUT / '프로젝트_방향성_비판검토.txt').write_text(review, encoding='utf-8-sig')
shutil.copy2(ROOT / 'outputs/aws1000/independent_data_audit.json', OUT / 'independent_data_audit.json')
for name in ['dataset_report.json', 'cloud_verification.json', 'design_plan.json']:
    shutil.copy2(SRC / name, OUT / name)
print(json.dumps({'output': str(OUT), 'copied_csv_rows': 1000, 'csv_hashes_verified': True}, ensure_ascii=False))
