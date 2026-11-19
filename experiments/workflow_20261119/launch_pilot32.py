"""Run in authenticated AWS CloudShell after local numerical validation passes.

Creates one EC2 worker with a detached systemd job and an independent stop timer.
Uses one-hour S3 URLs scoped to this release and named output objects; no IAM edits.
"""
import base64
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import runpy
import shlex
import tarfile
import boto3
from botocore.config import Config

REGION = "us-east-1"
BUCKET = "battery-capstone-backup-000000000000-us-east-1"
PREFIX = "risk-pilot/20261115/mesh8-v2/"
NAME = "capstone-risk-pilot32-20261115-r2"
HERE = Path.cwd()
archive = HERE / "aws_pilot32_code.tar.gz"
gate = json.loads((HERE / "numerical_verification.json").read_text())
if gate.get("passed") is not True:
    raise RuntimeError("Numerical validation has not passed")
s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))
ec2 = boto3.client("ec2", region_name=REGION)
existing = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [NAME]},
                                         {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
if any(r["Instances"] for r in existing["Reservations"]):
    raise RuntimeError("A pilot instance already exists; inspect it before launching again")
s3.upload_file(str(archive), BUCKET, PREFIX + archive.name)
s3.upload_file(str(HERE / "numerical_verification.json"), BUCKET, PREFIX + "numerical_verification.json")
outputs = ("started.json", "completion.json", "scenario_summary.csv", "aws_pilot32_results.tar.gz", "worker.log", "bootstrap_failure.log")
config = {"release_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "get_release": s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": PREFIX + archive.name}, ExpiresIn=3600),
          "put": {name: s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": PREFIX + name}, ExpiresIn=3600)
                  for name in outputs}}
sns = boto3.client("sns", region_name=REGION)
topic = "arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102"
recipient = "user@example.com"
subscribers = []
for page in sns.get_paginator("list_subscriptions_by_topic").paginate(TopicArn=topic):
    subscribers.extend(s for s in page["Subscriptions"] if s["SubscriptionArn"] != "PendingConfirmation")
if len(subscribers) != 1 or subscribers[0]["Protocol"] != "email" or subscribers[0]["Endpoint"] != recipient:
    raise RuntimeError("Completion topic recipients differ from authorized self email")
notifier = runpy.run_path(str(HERE / "configure_completion_email.py"), run_name="bootstrap_module")["NOTIFIER"]
result_link = f"https://s3.console.aws.amazon.com/s3/buckets/{BUCKET}?region={REGION}&prefix={PREFIX}"
email_config = {"recipient": recipient, "topic_arn": topic,
                "configuration_id": hashlib.sha256((NAME + config["release_sha256"]).encode()).hexdigest()}
for status, message in {
    "success": f"Capstone AWS pilot: all 32 scenarios and result uploads verified. Worker {NAME} will stop after this notification. CSV, archive and completion receipt: {result_link}",
    "failure": f"Capstone AWS pilot needs attention: startup, calculation or output upload did not pass completion verification. Worker {NAME} will stop. Logs and receipt: {result_link}",
}.items():
    email_config[status + "_url"] = sns.generate_presigned_url("publish",
        Params={"TopicArn": topic, "Subject": "Capstone AWS 32 scenarios - " + status, "Message": message},
        HttpMethod="GET", ExpiresIn=3600)
transfer = r'''import json,sys,urllib.request
from pathlib import Path
root=Path('/opt/capstone-pilot')
cfg=json.loads((root/'transfer.json').read_text())
mode=sys.argv[1]
if mode=='get':
    urllib.request.urlretrieve(cfg['get_release'], root/'release.tar.gz')
else:
    name,path=sys.argv[2:4]
    payload=Path(path).read_bytes()
    req=urllib.request.Request(cfg['put'][name],data=payload,method='PUT',headers={'Content-Type':'application/octet-stream'})
    with urllib.request.urlopen(req,timeout=90) as response:
        if response.status!=200: raise RuntimeError('S3 upload failed')
    print('Uploaded',name,len(payload),flush=True)
'''
worker = r'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /opt/capstone-pilot/worker.log) 2>&1
cd /opt/capstone-pilot
echo '[worker] started'
python3 - <<'PY'
import datetime,json,subprocess
from pathlib import Path
Path('started.json').write_text(json.dumps({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'detached_service':'capstone-pilot32.service','hard_stop_timer':subprocess.check_output(['systemctl','show','capstone-pilot32-hard-stop.timer','--property=NextElapseUSecMonotonic','--property=ActiveState'],text=True),'physics_timeout_s':1800,'service_timeout_s':2700,'hard_stop_after_s':3600},indent=2))
PY
python3 transfer.py put started.json started.json
echo '[worker] fetching release'
python3 transfer.py get
python3 - <<'PY'
import json,hashlib
from pathlib import Path
cfg=json.loads(Path('transfer.json').read_text())
assert hashlib.sha256(Path('release.tar.gz').read_bytes()).hexdigest()==cfg['release_sha256']
PY
echo '[worker] installing Python and archive tools'
timeout 360 dnf install -y python3.11 python3.11-pip tar gzip
echo '[worker] extracting release'
tar -xzf release.tar.gz
python3.11 -m venv venv
echo '[worker] installing pinned dependencies'
timeout 480 venv/bin/python -m pip install --disable-pip-version-check -r capstone-risk-pilot32-20261115/requirements-pilot.txt
cd capstone-risk-pilot32-20261115
../venv/bin/python -m unittest simulation.test_risk_scenarios
echo '[worker] running 32 scenarios'
bash run_aws_pilot.sh ../venv/bin/python
'''
finish = r'''#!/bin/bash
set +e
cd /opt/capstone-pilot
exec >> /opt/capstone-pilot/worker.log 2>&1
export PILOT_SERVICE_RESULT="${SERVICE_RESULT:-unknown}"
release=capstone-risk-pilot32-20261115
export CSV_UPLOAD_STATUS=missing ARCHIVE_UPLOAD_STATUS=missing
if [ -f "$release/results/scenario_summary.csv" ]; then
    timeout 35 python3 transfer.py put scenario_summary.csv "$release/results/scenario_summary.csv"
    export CSV_UPLOAD_STATUS=$?
fi
if [ -f "$release/aws_pilot32_results.tar.gz" ]; then
    timeout 35 python3 transfer.py put aws_pilot32_results.tar.gz "$release/aws_pilot32_results.tar.gz"
    export ARCHIVE_UPLOAD_STATUS=$?
fi
timeout 35 python3 transfer.py put worker.log worker.log
export LOG_UPLOAD_STATUS=$?
python3 - <<'PY'
import datetime,json,os
from pathlib import Path
root=Path('/opt/capstone-pilot')
receipt=root/'capstone-risk-pilot32-20261115/results/aws_execution_receipt.json'
try:
    r=json.loads(receipt.read_text()) if receipt.exists() else {}
except (ValueError,OSError) as error:
    r={'receipt_error':str(error)}
uploads={name:os.environ[name] for name in ('CSV_UPLOAD_STATUS','ARCHIVE_UPLOAD_STATUS','LOG_UPLOAD_STATUS')}
obj={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'service_result':os.environ['PILOT_SERVICE_RESULT'],'verified_32_cases':r.get('verification_passed',False),'execution_receipt':r,'uploads':uploads,'shutdown':'poweroff immediately after completion receipt upload; independent 60 minute timer remains active'}
(root/'completion.json').write_text(json.dumps(obj,indent=2))
PY
timeout 35 python3 transfer.py put completion.json completion.json
# completion email hook v1
timeout 25 python3 /opt/capstone-pilot/notify_completion.py
timeout 35 python3 transfer.py put completion.json completion.json
systemctl --no-block poweroff
'''
user_data = """#!/bin/bash
set -Eeuo pipefail
BOOT_STAGE=initial
FAILURE_UPLOAD_URL=__FAILURE_UPLOAD_URL__
FAILURE_EMAIL_URL=__FAILURE_EMAIL_URL__
bootstrap_failure() {
    result=$?
    trap - ERR
    set +e
    printf '[bootstrap failure] stage=%s exit=%s date=%s\\n' "$BOOT_STAGE" "$result" "$(date -u +%FT%TZ)"
    { printf 'stage=%s exit=%s\\n' "$BOOT_STAGE" "$result"; tail -n 180 /var/log/capstone-bootstrap.log; } > /tmp/capstone-bootstrap-failure.log
    if [ -n "$FAILURE_EMAIL_URL" ]; then
        curl --silent --show-error --fail --max-time 20 "$FAILURE_EMAIL_URL" > /tmp/capstone-bootstrap-email.xml
        printf 'failure_email_curl_exit=%s\\n' "$?" >> /tmp/capstone-bootstrap-failure.log
    fi
    curl --silent --show-error --fail --max-time 35 --upload-file /tmp/capstone-bootstrap-failure.log "$FAILURE_UPLOAD_URL"
    systemctl --no-block poweroff
    exit "$result"
}
trap bootstrap_failure ERR
exec > >(tee -a /var/log/capstone-bootstrap.log) 2>&1
echo '[bootstrap] initial'
mkdir -p /opt/capstone-pilot
chmod 700 /opt/capstone-pilot
BOOT_STAGE=install-stop-timer
echo '[bootstrap] installing stop timer'
cat > /etc/systemd/system/capstone-pilot32-hard-stop.service <<'UNIT'
[Unit]
Description=Stop bounded capstone EC2 worker
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl --no-block poweroff
UNIT
cat > /etc/systemd/system/capstone-pilot32-hard-stop.timer <<'UNIT'
[Unit]
Description=Independent 60 minute capstone stop limit
[Timer]
OnActiveSec=3600
AccuracySec=1s
Unit=capstone-pilot32-hard-stop.service
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now capstone-pilot32-hard-stop.timer
BOOT_STAGE=ensure-python
echo '[bootstrap] checking Python'
if ! command -v python3 >/dev/null 2>&1; then
    timeout 60 dnf install -y python3
fi
"""
# Keep the six scoped transfer URLs in a private S3 bootstrap. EC2 user data
# contains only one signed download URL and stays below its 16 KiB limit even
# when CloudShell uses a long STS session token.
bootstrap_buffer = io.BytesIO()
with tarfile.open(fileobj=bootstrap_buffer, mode="w:gz") as bootstrap:
    for name, content in {"transfer.json": json.dumps(config), "transfer.py": transfer,
                          "worker.sh": worker, "finish.sh": finish,
                          "completion_email.json": json.dumps(email_config), "notify_completion.py": notifier}.items():
        payload = content.encode()
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        member.mode = 0o600
        bootstrap.addfile(member, io.BytesIO(payload))
bootstrap_payload = bootstrap_buffer.getvalue()
bootstrap_sha = hashlib.sha256(bootstrap_payload).hexdigest()
bootstrap_key = PREFIX + "bootstrap.tar.gz"
s3.put_object(Bucket=BUCKET, Key=bootstrap_key, Body=bootstrap_payload,
              ContentType="application/gzip")
bootstrap_url = s3.generate_presigned_url("get_object",
    Params={"Bucket": BUCKET, "Key": bootstrap_key}, ExpiresIn=3600)
encoded_bootstrap_url = base64.b64encode(bootstrap_url.encode()).decode()
user_data += f"""BOOT_STAGE=download-and-unpack-bootstrap
echo '[bootstrap] fetching and unpacking bootstrap'
python3 - <<'PY'
import base64,hashlib,io,os,tarfile,urllib.request
from pathlib import Path
url=base64.b64decode('{encoded_bootstrap_url}').decode()
with urllib.request.urlopen(url,timeout=90) as response:
    payload=response.read()
if hashlib.sha256(payload).hexdigest()!='{bootstrap_sha}':
    raise RuntimeError('Bootstrap checksum mismatch')
root=Path('/opt/capstone-pilot')
(root/'bootstrap.tar.gz').write_bytes(payload)
expected={{'transfer.json','transfer.py','worker.sh','finish.sh','completion_email.json','notify_completion.py'}}
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
    members=archive.getmembers()
    if {{m.name for m in members}}!=expected or len(members)!=len(expected):
        raise ValueError('Unexpected bootstrap files')
    for member in members:
        if not member.isfile():
            raise ValueError('Bootstrap member must be a regular file')
        target=root/member.name
        target.write_bytes(archive.extractfile(member).read())
        os.chmod(target,0o600)
print('Bootstrap digest and six files verified',flush=True)
PY
"""
user_data += """BOOT_STAGE=install-worker-service
echo '[bootstrap] installing worker service'
chmod 600 /opt/capstone-pilot/transfer.json
cat > /etc/systemd/system/capstone-pilot32.service <<'UNIT'
[Unit]
Description=Capstone 32 scenario pilot with max 8 workers
Wants=network-online.target
After=network-online.target
[Service]
Type=oneshot
WorkingDirectory=/opt/capstone-pilot
ExecStart=/bin/bash /opt/capstone-pilot/worker.sh
ExecStopPost=/bin/bash /opt/capstone-pilot/finish.sh
TimeoutStartSec=2700
TimeoutStopSec=240
KillMode=control-group
Environment=PYBAMM_DISABLE_TELEMETRY=true
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable capstone-pilot32.service
BOOT_STAGE=start-worker-service
echo '[bootstrap] starting worker service'
systemctl --no-block start capstone-pilot32.service
echo '[bootstrap] worker start requested'
"""
user_data = user_data.replace("__FAILURE_UPLOAD_URL__", shlex.quote(config["put"]["bootstrap_failure.log"]))
early_failure_email = len(user_data.replace("__FAILURE_EMAIL_URL__", shlex.quote(email_config["failure_url"])).encode()) <= 16384
user_data = user_data.replace("__FAILURE_EMAIL_URL__", shlex.quote(email_config["failure_url"] if early_failure_email else ""))
if len(user_data.encode()) > 16384:
    raise RuntimeError("User data exceeds EC2 limit")
import subprocess
subprocess.run(['bash','-n'], input=user_data, text=True, check=True, capture_output=True)
launched_at = dt.datetime.now(dt.timezone.utc)
response = ec2.run_instances(ImageId="ami-00000000000000000", InstanceType="m7i.2xlarge", MinCount=1, MaxCount=1,
    ClientToken="capstone-risk-pilot32-20261115-mesh8-v2", IamInstanceProfile={"Name": "capstone-ssm-instance"},
    NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": "subnet-0b92671ac264cbba7", "Groups": ["sg-024918075c115c588"], "AssociatePublicIpAddress": True}],
    BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 16, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
    MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"}, InstanceInitiatedShutdownBehavior="stop",
    UserData=user_data,
    TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}, {"Key": "Purpose", "Value": "bounded-risk-pilot32"}]},
                       {"ResourceType": "volume", "Tags": [{"Key": "Name", "Value": NAME}]}])
instance = response["Instances"][0]
receipt = {"instance_id": instance["InstanceId"], "instance_type": instance["InstanceType"], "region": REGION,
           "launch_utc": launched_at.isoformat(), "expected_finish_utc": (launched_at + dt.timedelta(minutes=12)).isoformat(),
           "hard_stop": "60 minutes after boot timer starts; complete/failure also powers off", "s3": f"s3://{BUCKET}/{PREFIX}",
           "release_sha256": config["release_sha256"], "numerical_gate_passed": True, "auto_stop": True,
           "bootstrap_sha256": bootstrap_sha, "user_data_bytes": len(user_data.encode()),
           "completion_email_recipient": recipient, "bootstrap_failure_email_enabled": early_failure_email,
           "max_workers": 8, "scenario_count": 32, "user_data_sha256": hashlib.sha256(user_data.encode()).hexdigest()}
(HERE / "pilot32_launch_receipt.json").write_text(json.dumps(receipt, indent=2))
s3.put_object(Bucket=BUCKET, Key=PREFIX + "launch_receipt.json", Body=json.dumps(receipt, indent=2).encode())
print(json.dumps(receipt, indent=2))
