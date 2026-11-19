from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=Path(r'<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo')
STAGE=HERE/'risk1000_release'
UPLOAD=HERE/'risk1000_upload'
for folder in (STAGE,UPLOAD,ROOT/'outputs/aws1000'):
    folder.mkdir(parents=True,exist_ok=True)
files=['simulation/risk_scenarios.py','simulation/rebuilt_lifetime.py','simulation/test_risk_scenarios.py',
       'simulation/risk_dataset1000.py','simulation/test_risk_dataset1000.py','configs/model.json','configs/m50_c_joint_v3.json']
for name in files:
    target=STAGE/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/name,target)
(STAGE/'simulation/__init__.py').write_text('',encoding='utf-8')
shutil.copy2(HERE/'aws1000_ops.py',STAGE/'aws1000_ops.py')
(STAGE/'requirements.txt').write_text('pybamm==26.8.0.0\npybammsolvers==0.9.1\nnumpy==2.4.6\nscipy==1.17.1\ncasadi==3.7.2\nboto3==1.43.83\nbotocore==1.43.97\n',encoding='utf-8')
design=ROOT/'outputs/risk_dataset1000'
(STAGE/'reviewed_design').mkdir(exist_ok=True)
for name in ('design_plan.json','design_scenarios.json','features.csv','software_test_record.json','1000개_학습데이터_설명.txt'):
    shutil.copy2(design/name,STAGE/'reviewed_design'/name)
gate=ROOT/'outputs/aws1000_preflight/gate.json'
shutil.copy2(gate,STAGE/'preflight_gate.json')
shutil.copy2(gate,UPLOAD/'preflight_gate.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'release':'capstone-risk1000-20261115-v1','files':{
    p.relative_to(STAGE).as_posix():sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file() and p.name!='release_manifest.json' and '__pycache__' not in p.parts}}
(STAGE/'release_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=ROOT/'outputs/aws1000/aws_risk1000_code.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(STAGE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            tar.add(p,arcname='release/'+p.relative_to(STAGE).as_posix(),recursive=False)
shutil.copy2(archive,UPLOAD/archive.name)
for name in ('launch_risk1000.py','generate_aws1000_role_policy.py','aws1000_worker.sh','aws1000_finish.sh','aws1000_emergency.py'):
    target=UPLOAD/name
    target.write_text((HERE/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
bundle=ROOT/'outputs/aws1000/aws_risk1000_upload.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(UPLOAD.iterdir()):
        if p.is_file() and p.name!='user_authorization.json':
            z.write(p,'risk1000_upload/'+p.name)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for p in sorted(UPLOAD.iterdir()):
        if p.is_file() and p.name!='user_authorization.json':
            assert hashlib.sha256(z.read('risk1000_upload/'+p.name)).hexdigest()==sha(p)
receipt={'release_sha256':sha(archive),'upload_sha256':sha(bundle),'upload_bytes':bundle.stat().st_size,
         'gate_sha256':sha(gate),'generator_sha256':sha(REPO/'simulation/risk_dataset1000.py'),
         'files_sha256':{p.name:sha(p) for p in UPLOAD.iterdir() if p.is_file()},'not_launched':True}
(ROOT/'outputs/aws1000/prepared_release.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
