"""Package the verified local and AWS artifacts without changing frozen code."""
from pathlib import Path
import json
import hashlib
import shutil
import zipfile
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
delivery = root / 'outputs/bms_pipeline_delivery_20261116'
cloud = root / 'outputs/bms_pipeline_aws_20261116_v1'
comparison = json.loads((cloud / 'cross_platform_comparison.json').read_text(encoding='utf-8'))
assert comparison['all_checks_passed'] is True
assert (cloud / 'exit_code.txt').read_text().strip() == '0'
shutil.copytree(cloud, delivery / '03_AWS_검증_기록', dirs_exist_ok=True)
shutil.copy2(root / 'work/cloud_compare_20261116.py', delivery / '03_AWS_검증_기록/cloud_compare_20261116.py')
receipt = {
    'written_at_utc': datetime.now(timezone.utc).isoformat(),
    'instance_id': 'i-00000000000000000',
    'region': 'us-east-1',
    'run_command_id': '1a64d564-6fa9-4174-87a7-5513bdc60a5e',
    'aws_command_status': 'Success',
    'aws_result_archive_sha256': '8edd96c549fd825c08d5cfd1167666fed57c9eb4ff59d750598e0c6c203fd01e',
    's3_prefix': 's3://battery-capstone-backup-000000000000-us-east-1/bms-pipeline/20261116/local-validated-v1/',
    'observed_instance_state': 'stopped',
    'state_verification': 'Direct observation of AWS EC2 describe-instances output in authenticated CloudShell on 2026-11-16; written_at is the receipt creation time, not an API observation timestamp.',
    'old_risk1000_service': 'disabled and stopped to prevent the completed batch restarting at boot',
    'cloudshell_watchdog': 'terminated after stopped state was verified',
    'unit_tests_per_platform': 17,
    'integration_checks_per_platform': 20,
    'cross_platform_comparison_checks': len(comparison['checks']),
    'cross_platform_comparison_passed': True,
    'physical_lifetime_or_cause_validated': False,
}
(delivery / 'aws_completion_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
intro = '''파일은 이렇게 쓰면 됩니다.

1. 01_before_machine_learning: 머신러닝 전 실측 입력 1,200행.
2. 02_after_machine_learning_for_twin: 머신러닝 선별 후 디지털 트윈에 실제 전달한 입력 591행과 선별 근거.
3. 03_AWS_검증_기록: AWS 결과와 로컬 결과 대조 기록.
4. bms_pipeline_release_20261116.zip: 실행 코드·설정·준비 과정 자료. 압축 안 README에 실행 방법이 있습니다.

두 CSV는 실제 LG M50T 셀 실험의 시간·전압·전류·표면 온도 기록입니다. 차량에서 직접 얻은 BMS 자료는 아닙니다.
C 셀 600행으로 학습·기준을 설정하고 D 셀 600행에서 평가했습니다. 디지털 트윈에는 D의 마지막 판정 시점까지 591행을 전달했습니다.
600행(학습용 C)+9행(판정 이후 D)을 제외한 것이므로 '안전한 609행 제거'나 계산량 절감 성과가 아닙니다.

머신러닝은 약 10초 뒤 전압을 예측한 후 관측과의 차이·예측 불확실성·학습 자료 범위를 검토했습니다. 수명 위험 등급을 붙인 것은 아닙니다.
D의 58개 평가 구간이 모두 추가 검토로 선택됐습니다. 오차 기준 초과는 6개, 불확실성·자료 범위 기준 초과는 58개입니다. 계산 절감은 아직 입증되지 않았습니다.

로컬과 AWS에서 코드 검사 17개, 결과 대조 20항목 통과. 내려받은 AWS와 로컬 결과의 재현 대조 32항목도 통과했습니다.
두 환경에서 전후 CSV의 값과 파일 해시가 완전히 같고, 선별 결정도 같습니다.
확인한 범위: 데이터·단위·판정 시점, 미래 정보 누수 방지, 입력 전달, 전하 보존, 상태 저장·복원, 같은 상태에서 미래 조건 분기.
25°C 등온 가정입니다. 실제 온도·수명 위험·열화 원인 정확도 검증은 완료되지 않았습니다.

머신러닝의 10초 전압 예측 오차: 0.136mV. 디지털 트윈의 관측 전압 재현 오차: 4.423mV. 계산 목적이 달라 서로 우열 비교하지 않습니다.
연결 계산 1회: 로컬 약27.2초, AWS 약12.1초. 부팅·설치·업로드 제외, 하드웨어가 다른 단일 실행이므로 효율 비교 실험 결과가 아닙니다.
AWS 결과를 보관한 후 서버 stopped 상태를 확인했습니다.

노션: https://app.notion.com/p/41a8df1f4d7a81358726e39b03293905
원자료: https://zenodo.org/records/10637534
'''
(delivery / '먼저읽기.txt').write_text(intro, encoding='utf-8-sig')
share = root / 'outputs/실측입력_머신러닝전후_로컬_AWS검증_완료_20261116.zip'
with zipfile.ZipFile(share, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in sorted(delivery.rglob('*')):
        if file.is_file():
            z.write(file, file.relative_to(delivery).as_posix())
with zipfile.ZipFile(share) as z:
    assert z.testzip() is None
    files = z.namelist()
    assert sum(x.endswith('.csv') for x in files) >= 2
print(json.dumps({'share': str(share), 'bytes': share.stat().st_size, 'files': len(files), 'sha256': hashlib.sha256(share.read_bytes()).hexdigest()}, ensure_ascii=False))
