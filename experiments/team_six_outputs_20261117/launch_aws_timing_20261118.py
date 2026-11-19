"""Run from the user's authenticated AWS CloudShell, with shutdown and S3 receipts."""
import argparse
import hashlib
from pathlib import Path
import shlex
import time

def launch(archive, instance, bucket, prefix, region, python_path):
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    machine = ec2.describe_instances(InstanceIds=[instance])["Reservations"][0]["Instances"][0]
    if machine["State"]["Name"] != "stopped":
        raise RuntimeError("Expected the existing benchmark instance in stopped state")
    behavior = ec2.describe_instance_attribute(InstanceId=instance, Attribute="instanceInitiatedShutdownBehavior")
    if behavior["InstanceInitiatedShutdownBehavior"]["Value"] != "stop":
        raise RuntimeError("Expected instance initiated shutdown to stop the instance")
    key = prefix + "input.zip"
    s3.upload_file(str(archive), bucket, key)
    with archive.open("rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=840)
    result_url = s3.generate_presigned_url("put_object", Params={
        "Bucket": bucket, "Key": prefix + "results.tar.gz"}, ExpiresIn=840)
    script = """set -eu
umask 077
shutdown -h +8
trap 'shutdown -h +1 || true' EXIT
work_dir=$(mktemp -d /opt/bms-coupled-timing-20261118.XXXXXX)
cd "$work_dir"
curl -fsSL --retry 2 DOWNLOAD -o input.zip
printf '%s  input.zip\\n' ZIP_HASH | sha256sum -c -
PY=PYTHON_PATH
test -x "$PY"
"$PY" -m zipfile -e input.zip .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exit_status=0
/usr/bin/timeout 350 "$PY" -X utf8 experiments/team_six_outputs_20261117/aws_measure_20261118.py results --input bms_pipeline/evidence/02_머신러닝_후_디지털트윈_입력.csv > run.log 2>&1 || exit_status=$?
printf '%s\\n' "$exit_status" > exit_code.txt
tar -czf results.tar.gz docs/imperial_lg_m50t_fields.md run.log exit_code.txt results
curl -fsS --retry 2 --max-time 30 -X PUT --upload-file results.tar.gz RESULT
sha256sum results.tar.gz
tail -3 run.log
exit "$exit_status"
""".replace("DOWNLOAD", shlex.quote(url)).replace("ZIP_HASH", shlex.quote(digest)).replace("RESULT", shlex.quote(result_url)).replace("PYTHON_PATH", shlex.quote(python_path))

    ec2.start_instances(InstanceIds=[instance])
    try:
        for _ in range(35):
            try:
                reply = ssm.send_command(
                    InstanceIds=[instance], DocumentName="AWS-RunShellScript",
                    Parameters={"commands": [script], "executionTimeout": ["420"]},
                    TimeoutSeconds=90, Comment="Capstone selected cell coupled degradation timing 20261118")
                command_id = reply["Command"]["CommandId"]
                print("COMMAND_ID", command_id, flush=True)
                print("RESULT", f"s3://{bucket}/{prefix}results.tar.gz", flush=True)
                break
            except ssm.exceptions.InvalidInstanceId:
                time.sleep(3)
        else:
            raise RuntimeError("SSM agent failed to register after startup")
    except BaseException:
        ec2.stop_instances(InstanceIds=[instance])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="bms-pipeline/coupled-timing/")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--python", default="/opt/capstone-bms-20261116-v1/venv/bin/python")
    args = parser.parse_args()
    launch(args.archive.resolve(), args.instance_id, args.bucket,
           args.prefix.rstrip("/") + "/", args.region, args.python)
