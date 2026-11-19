"""Install completion email on the pilot through CloudShell and existing SSM.

Creates scoped, one-hour SNS Publish URLs. Installation itself sends no email.
Run only after the corrected bootstrap has installed /opt/capstone-pilot/finish.sh.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import time
import urllib.parse
from pathlib import Path

import boto3
from botocore.config import Config

REGION = "us-east-1"
BUCKET = "battery-capstone-backup-000000000000-us-east-1"
TOPIC = "arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102"
RECIPIENT = "user@example.com"

NOTIFIER = r'''"""Send the configured result email once, then persist its SNS receipt."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path('/opt/capstone-pilot')

def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)

def main():
    config=json.loads((ROOT/'completion_email.json').read_text())
    completion_path=ROOT/'completion.json'
    completion=json.loads(completion_path.read_text())
    delivery_path=ROOT/'email_delivery.json'
    delivery=json.loads(delivery_path.read_text()) if delivery_path.exists() else {}
    if delivery.get('configuration_id')==config['configuration_id'] and delivery.get('status')=='sent':
        completion['email_notification']=delivery
        save(completion_path,completion)
        print('Completion email already sent',delivery['sns_message_id'],flush=True)
        return
    uploads=completion.get('uploads',{})
    success=(completion.get('verified_32_cases') is True and
             all(str(uploads.get(name))=='0' for name in
                 ('CSV_UPLOAD_STATUS','ARCHIVE_UPLOAD_STATUS','LOG_UPLOAD_STATUS')))
    delivery={'configuration_id':config['configuration_id'],
              'destination':config['recipient'],'success_reported':success,
              'attempted_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        request=urllib.request.Request(config['success_url' if success else 'failure_url'],method='GET')
        with urllib.request.urlopen(request,timeout=20) as response:
            document=ET.fromstring(response.read())
        identifier=document.findtext('.//{*}MessageId')
        if not identifier:
            raise ValueError('SNS response did not contain a MessageId')
        delivery.update(status='sent',sns_message_id=identifier)
        print('Completion email published',identifier,flush=True)
    except Exception as error:
        # Signed URLs never enter logs, even when an HTTP request fails.
        delivery.update(status='failed',error_type=type(error).__name__,http_status=getattr(error,'code',None))
        print('Completion email publish failed',delivery['error_type'],delivery['http_status'],flush=True)
    save(delivery_path,delivery)
    completion['email_notification']=delivery
    save(completion_path,completion)

if __name__=='__main__':
    with (ROOT/'email_delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default="i-00000000000000000")
    parser.add_argument("--prefix", default="risk-pilot/20261115/mesh8-v1/")
    args = parser.parse_args()
    sns = boto3.client("sns", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))
    ssm = boto3.client("ssm", region_name=REGION)
    active = []
    for page in sns.get_paginator("list_subscriptions_by_topic").paginate(TopicArn=TOPIC):
        active.extend(item for item in page["Subscriptions"] if item["SubscriptionArn"] != "PendingConfirmation")
    if len(active) != 1 or active[0]["Protocol"] != "email" or active[0]["Endpoint"] != RECIPIENT:
        raise RuntimeError("Topic recipients differ from the authorized completion-email recipient")
    instances = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [args.instance_id]}])
    if not any(item["PingStatus"] == "Online" for item in instances["InstanceInformationList"]):
        raise RuntimeError("Pilot SSM is not online yet")
    prefix = args.prefix.rstrip("/") + "/"
    result_link = (f"https://s3.console.aws.amazon.com/s3/buckets/{BUCKET}?region={REGION}"
                   f"&prefix={urllib.parse.quote(prefix, safe='')}&showversions=false")
    tail = f"\n\nEC2 {args.instance_id}\n결과 및 실행 기록\n{result_link}\n\n이 알림 이후 서버 자동 중지가 이어집니다."
    messages = {
        "success": ("캡스톤 AWS 32개 시나리오 완료", "32개 시나리오 계산과 CSV·압축 결과·로그의 S3 업로드 검증이 완료됐습니다." + tail),
        "failure": ("캡스톤 AWS 시나리오 실행 확인 필요", "AWS 시나리오 계산 또는 결과 업로드가 완료 검증을 통과하지 못했습니다. completion.json과 worker.log에서 원인을 확인해 주세요." + tail),
    }
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    configuration_id = hashlib.sha256((args.instance_id + prefix + generated).encode()).hexdigest()
    config = {"recipient": RECIPIENT, "topic_arn": TOPIC, "configuration_id": configuration_id,
              "generated_utc": generated, "url_expiry_seconds": 3600}
    for status, (subject, message) in messages.items():
        config[status + "_url"] = sns.generate_presigned_url(
            "publish", Params={"TopicArn": TOPIC, "Subject": subject, "Message": message},
            HttpMethod="GET", ExpiresIn=3600)
    payload = json.dumps(config, ensure_ascii=False).encode()
    key = prefix + "completion_email.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=payload, ContentType="application/json")
    get_url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600)
    encoded_url = base64.b64encode(get_url.encode()).decode()
    encoded_notifier = base64.b64encode(NOTIFIER.encode()).decode()
    expected_sha = hashlib.sha256(payload).hexdigest()
    # No shell tracing, command echo, signed URL or config content is printed.
    patch = f"""set -eu
python3 - <<'PY'
import base64,hashlib,json,os,urllib.request
from pathlib import Path
root=Path('/opt/capstone-pilot')
finish=root/'finish.sh'
text=finish.read_text()
marker='# completion email hook v1'
needle='systemctl --no-block poweroff'
if marker not in text and text.count(needle)!=1:
    raise RuntimeError('Expected one final shutdown command in finish.sh')
with urllib.request.urlopen(base64.b64decode('{encoded_url}').decode(),timeout=30) as response:
    payload=response.read()
if hashlib.sha256(payload).hexdigest()!='{expected_sha}':
    raise RuntimeError('Email configuration digest differs')
config_path=root/'completion_email.json'
config_path.write_bytes(payload)
os.chmod(config_path,0o600)
notifier=root/'notify_completion.py'
notifier.write_bytes(base64.b64decode('{encoded_notifier}'))
os.chmod(notifier,0o600)
if marker not in text:
    text=text.replace('timeout 50 python3 transfer.py','timeout 35 python3 transfer.py')
    hook='''# completion email hook v1
timeout 25 python3 /opt/capstone-pilot/notify_completion.py
timeout 35 python3 transfer.py put completion.json completion.json
systemctl --no-block poweroff'''
    temporary=finish.with_suffix('.sh.tmp')
    temporary.write_text(text.replace(needle,hook))
    os.chmod(temporary,0o600)
    temporary.replace(finish)
print(json.dumps({{'completion_email_installed':True,'recipient':'{RECIPIENT}',
                  'notification_sends_on_finish':True,'finish_upload_and_email_budget_seconds':200,
                  'configuration_sha256':'{expected_sha}'}}))
PY
"""
    command = ssm.send_command(InstanceIds=[args.instance_id], DocumentName="AWS-RunShellScript",
        Parameters={"commands": [patch], "executionTimeout": ["120"]}, TimeoutSeconds=120,
        Comment="Install authorized self completion email for capstone 32-case pilot")
    command_id = command["Command"]["CommandId"]
    receipt = {"command_id": command_id, "instance_id": args.instance_id, "recipient": RECIPIENT,
               "configuration_sha256": expected_sha, "notification_sent_by_installation": False}
    for _ in range(60):
        time.sleep(2)
        try:
            invocation = ssm.get_command_invocation(CommandId=command_id, InstanceId=args.instance_id)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if invocation["Status"] in ("Pending", "InProgress", "Delayed"):
            continue
        receipt["status"] = invocation["Status"]
        receipt["stdout"] = invocation.get("StandardOutputContent", "")
        # The installed script deliberately emits no credentials on failure.
        receipt["stderr"] = invocation.get("StandardErrorContent", "")
        Path("completion_email_install_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        if invocation["Status"] != "Success":
            raise SystemExit(1)
        return
    receipt["status"] = "delivery_pending"
    Path("completion_email_install_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
