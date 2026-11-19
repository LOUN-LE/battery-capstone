"""Reviewed CloudShell launcher. --apply requires recorded per-action IAM approval.

Creates one bounded worker, then grants ONLY that instance temporary access to
the exact batch prefix and the existing self-email topic. No credentials copied.
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

from generate_aws1000_role_policy import build_policy

REGION='us-east-1'
BUCKET='battery-capstone-backup-000000000000-us-east-1'
PREFIX='risk-dataset1000/20261115/batch-v1/'
NAME='capstone-risk1000-20261115-v1'
ROLE='capstone-ssm-instance'
POLICY_NAME='CapstoneRisk1000SingleInstance20261115'
TOPIC='arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102'
RECIPIENT='user@example.com'
HERE=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_local():
    gate=json.loads((HERE/'preflight_gate.json').read_text())
    release=HERE/'aws_risk1000_code.tar.gz'
    if gate.get('passed') is not True or gate.get('sources_unchanged') is not True:
        raise ValueError('Numerical preflight not passed')
    with tarfile.open(release,'r:gz') as archive:
        manifest=json.load(archive.extractfile('release/release_manifest.json'))
        for name,expected in manifest['files'].items():
            payload=archive.extractfile('release/'+name).read()
            if hashlib.sha256(payload).hexdigest()!=expected:
                raise ValueError('Release file mismatch: '+name)
        for name,expected in gate['source']['sources'].items():
            if manifest['files'][name]!=expected:
                raise ValueError('Numerical gate differs from released physics')
        design_bytes=archive.extractfile('release/reviewed_design/design_scenarios.json').read()
        design=json.loads(design_bytes)
        value=hashlib.sha256(design_bytes).hexdigest()
        if value!=gate['design_sha256'] or len(design)!=1000:
            raise ValueError('Released1000design differs from local preflight')
    for name in ('aws1000_worker.sh','aws1000_finish.sh'):
        subprocess.run(['bash','-n',str(HERE/name)],check=True,capture_output=True)
    return gate,manifest,release

def user_data(bootstrap_url, bootstrap_sha):
    encoded=base64.b64encode(bootstrap_url.encode()).decode()
    script=r'''#!/bin/bash
set -Eeuo pipefail
exec >> /var/log/capstone-risk1000-bootstrap.log 2>&1
BOOT_STAGE=initial
fail_bootstrap() {
  result=$?
  trap - ERR
  set +e
  echo "bootstrap_failure stage=$BOOT_STAGE exit=$result"
  if [ -f /opt/capstone-risk1000/aws1000_emergency.py ]; then
    timeout 120 python3 /opt/capstone-risk1000/aws1000_emergency.py
  fi
  systemctl --no-block poweroff
  exit "$result"
}
trap fail_bootstrap ERR
mkdir -p /opt/capstone-risk1000
chmod 700 /opt/capstone-risk1000
BOOT_STAGE=arm-independent-stop
cat > /etc/systemd/system/capstone-risk1000-hard-stop.service <<'UNIT'
[Unit]
Description=Absolute five-hour capstone stop fallback
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl --force poweroff
UNIT
cat > /etc/systemd/system/capstone-risk1000-hard-stop.timer <<'UNIT'
[Unit]
Description=Force poweroff five hours after bootstrap timer is armed
[Timer]
OnActiveSec=18000
AccuracySec=1s
Unit=capstone-risk1000-hard-stop.service
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now capstone-risk1000-hard-stop.timer
BOOT_STAGE=ensure-python
if ! command -v python3 >/dev/null 2>&1; then
  timeout 90 dnf install -y python3
fi
BOOT_STAGE=download-bootstrap
python3 - <<'PY'
import base64,hashlib,io,os,tarfile,urllib.request
from pathlib import Path,PurePosixPath
root=Path('/opt/capstone-risk1000')
url=base64.b64decode('__URL__').decode()
with urllib.request.urlopen(url,timeout=90) as r:
    data=r.read()
if hashlib.sha256(data).hexdigest()!='__SHA__':
    raise ValueError('Bootstrap checksum mismatch')
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
    for member in archive.getmembers():
        rel=PurePosixPath(member.name)
        if rel.is_absolute() or '..' in rel.parts or not member.isfile():
            raise ValueError('Invalid bootstrap entry')
        target=root.joinpath(*rel.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
        os.chmod(target,0o600)
print('Bootstrap checksum and paths verified')
PY
BOOT_STAGE=install-detached-worker
cat > /etc/systemd/system/capstone-risk1000.service <<'UNIT'
[Unit]
Description=Capstone1000 independent simulation batch with eight workers
Wants=network-online.target
After=network-online.target
[Service]
Type=oneshot
WorkingDirectory=/opt/capstone-risk1000
ExecStart=/bin/bash /opt/capstone-risk1000/worker.sh
ExecStopPost=/bin/bash /opt/capstone-risk1000/finish.sh
TimeoutStartSec=10800
TimeoutStopSec=960
KillMode=control-group
Environment=PYBAMM_DISABLE_TELEMETRY=true
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable capstone-risk1000.service
systemctl --no-block start capstone-risk1000.service
echo "Detached worker requested $(date -u +%FT%TZ)"
'''.replace('__URL__',encoded).replace('__SHA__',bootstrap_sha)
    if len(script.encode())>16384:
        raise ValueError('EC2 user data exceeds limit')
    subprocess.run(['bash','-n'],input=script,text=True,check=True,capture_output=True)
    return script

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    gate,manifest,release=validate_local()
    if not args.apply:
        print(json.dumps({'prepared_only':True,'launch_name':NAME,'scenario_count':1000,'max_workers':8,
                          'release_sha256':sha(release),'numerical_gate':True,'requires_explicit_iam_approval':True}))
        return
    approved=json.loads((HERE/'user_authorization.json').read_text())
    if approved.get('explicit_iam_permission') is not True or approved.get('batch_name')!=NAME or approved.get('prefix')!=PREFIX:
        raise ValueError('Specific IAM permission approval not recorded')
    import boto3
    from botocore.config import Config
    session=boto3.Session()
    credentials=session.get_credentials()
    credentials.get_frozen_credentials()
    expiry=getattr(credentials,'_expiry_time',None)
    if expiry and (expiry-dt.datetime.now(dt.timezone.utc)).total_seconds()<240:
        raise RuntimeError('Bootstrap signer expires too soon; refresh console credentials before launch')
    ec2=session.client('ec2',region_name=REGION)
    s3=session.client('s3',region_name=REGION,config=Config(signature_version='s3v4'))
    sns=session.client('sns',region_name=REGION)
    iam=session.client('iam',region_name=REGION)
    current=ec2.describe_instances(Filters=[{'Name':'tag:Name','Values':[NAME]},
        {'Name':'instance-state-name','Values':['pending','running','stopping','stopped']}])
    if any(r['Instances'] for r in current['Reservations']):
        raise RuntimeError('Batch instance already exists; do not launch duplicate')
    subscribers=[]
    for page in sns.get_paginator('list_subscriptions_by_topic').paginate(TopicArn=TOPIC):
        subscribers.extend(s for s in page['Subscriptions'] if s['SubscriptionArn']!='PendingConfirmation')
    if len(subscribers)!=1 or subscribers[0]['Protocol']!='email' or subscribers[0]['Endpoint']!=RECIPIENT:
        raise RuntimeError('Completion topic recipients differ from authorized self email')
    if POLICY_NAME in iam.list_role_policies(RoleName=ROLE)['PolicyNames']:
        raise RuntimeError('Batch role policy already exists; inspect before reuse')
    launched=dt.datetime.now(dt.timezone.utc)
    cfg={'region':REGION,'bucket':BUCKET,'prefix':PREFIX,'topic_arn':TOPIC,'recipient':RECIPIENT,
         'batch_name':NAME,'policy_expiry_utc':(launched+dt.timedelta(hours=5)).isoformat(),
         'expected_scenarios':1000,'max_workers':8}
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as bundle:
        with tarfile.open(release,'r:gz') as code:
            for member in code.getmembers():
                if member.isfile():
                    bundle.addfile(member,code.extractfile(member))
        for name,payload in [('runtime_config.json',json.dumps(cfg).encode()),
                             ('worker.sh',(HERE/'aws1000_worker.sh').read_bytes()),
                             ('finish.sh',(HERE/'aws1000_finish.sh').read_bytes()),
                             ('aws1000_emergency.py',(HERE/'aws1000_emergency.py').read_bytes())]:
            info=tarfile.TarInfo(name);info.size=len(payload);info.mode=0o600
            bundle.addfile(info,io.BytesIO(payload))
    bootstrap=buffer.getvalue()
    bootstrap_sha=hashlib.sha256(bootstrap).hexdigest()
    s3.put_object(Bucket=BUCKET,Key=PREFIX+'bootstrap.tar.gz',Body=bootstrap,ContentType='application/gzip')
    s3.upload_file(str(release),BUCKET,PREFIX+release.name)
    s3.upload_file(str(HERE/'preflight_gate.json'),BUCKET,PREFIX+'preflight_gate.json')
    url=s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':PREFIX+'bootstrap.tar.gz'},ExpiresIn=900)
    boot=user_data(url,bootstrap_sha)
    response=ec2.run_instances(ImageId='ami-00000000000000000',InstanceType='m7i.2xlarge',MinCount=1,MaxCount=1,
        ClientToken='capstone-risk1000-20261115-v1',IamInstanceProfile={'Name':ROLE},
        NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':'subnet-0b92671ac264cbba7','Groups':['sg-024918075c115c588'],'AssociatePublicIpAddress':True}],
        BlockDeviceMappings=[{'DeviceName':'/dev/xvda','Ebs':{'VolumeSize':24,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],
        MetadataOptions={'HttpTokens':'required','HttpEndpoint':'enabled'},InstanceInitiatedShutdownBehavior='stop',UserData=boot,
        TagSpecifications=[{'ResourceType':r,'Tags':[{'Key':'Name','Value':NAME},{'Key':'Purpose','Value':'bounded-risk1000'}]} for r in ('instance','volume')])
    instance=response['Instances'][0]['InstanceId']
    policy=build_policy(instance,PREFIX,launched)
    (HERE/'applied_role_policy.json').write_text(json.dumps(policy,indent=2))
    try:
        iam.put_role_policy(RoleName=ROLE,PolicyName=POLICY_NAME,PolicyDocument=json.dumps(policy))
        if iam.get_role_policy(RoleName=ROLE,PolicyName=POLICY_NAME)['PolicyDocument']!=policy:
            raise RuntimeError('Role policy readback mismatch')
    except Exception:
        ec2.stop_instances(InstanceIds=[instance])
        raise
    receipt={'instance_id':instance,'name':NAME,'instance_type':'m7i.2xlarge','region':REGION,'launch_utc':launched.isoformat(),
             'estimated_finish_utc_range':[(launched+dt.timedelta(minutes=n)).isoformat() for n in (60,90)],
             'estimate_basis':'Previous32cases92.2756sec gives48.1min physics; add startup/checkpoint/verification/upload margin',
             'hard_stop':'Force poweroff18000seconds after bootstrap arms timer; normal completion/failure stops earlier',
             'physics_timeout_s':9000,'service_timeout_s':10800,'finalization_timeout_s':960,
             'scenario_count':1000,'max_workers':8,'s3':f's3://{BUCKET}/{PREFIX}',
             'release_sha256':sha(release),'bootstrap_sha256':bootstrap_sha,'user_data_sha256':hashlib.sha256(boot.encode()).hexdigest(),
             'preflight_gate_passed':True,'role_policy_name':POLICY_NAME,'role_policy_sha256':sha(HERE/'applied_role_policy.json'),
             'role_policy_scope':'This exact instanceARN + one batch prefix + existing single-self-recipient SNS topic',
             'role_policy_expires_utc':cfg['policy_expiry_utc'],'completion_email_recipient':RECIPIENT,
             'state_at_launch':response['Instances'][0]['State']['Name'],'chat_independent':True}
    (HERE/'risk1000_launch_receipt.json').write_text(json.dumps(receipt,indent=2))
    s3.put_object(Bucket=BUCKET,Key=PREFIX+'launch_receipt.json',Body=json.dumps(receipt,indent=2).encode())
    s3.put_object(Bucket=BUCKET,Key=PREFIX+'applied_role_policy.json',Body=json.dumps(policy,indent=2).encode())
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
