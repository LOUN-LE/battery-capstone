"""Generate a reviewed, single-instance batch policy OFFLINE. Never calls AWS.

Fill the ACTUAL new instance ID only after authorized launch. The caller must
separately obtain authorization for the IAM permission expansion before applying.
Existing role policies are not changed or implicitly restricted by this document.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re

ACCOUNT = "000000000000"
REGION = "us-east-1"
BUCKET = "battery-capstone-backup-000000000000-us-east-1"
TOPIC = "arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102"


def build_policy(instance_id: str, prefix: str, launched_at: dt.datetime,
                 *, multipart_cleanup: bool = False) -> dict:
    if not re.fullmatch(r"i-[0-9a-f]{17}", instance_id):
        raise ValueError("Require the actual single EC2 instance ID, no wildcard")
    if (not prefix or prefix.startswith("/") or not prefix.endswith("/")
            or len(prefix.strip("/").split("/")) < 3
            or any(part in {"", ".", ".."} for part in prefix.strip("/").split("/"))
            or not re.fullmatch(r"[A-Za-z0-9_./-]+/", prefix)):
        raise ValueError("Require a concrete multi-segment batch prefix ending in/; wildcards forbidden")
    if launched_at.tzinfo is None or launched_at.utcoffset() is None:
        raise ValueError("Launch time must include UTC offset")
    launch = launched_at.astimezone(dt.timezone.utc)
    conditions = {
        "ArnEquals": {"ec2:SourceInstanceARN": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/{instance_id}"},
        "DateGreaterThanEquals": {"aws:CurrentTime": (launch-dt.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")},
        "DateLessThan": {"aws:CurrentTime": (launch+dt.timedelta(hours=5)).isoformat().replace("+00:00", "Z")},
        "Bool": {"aws:SecureTransport": "true"},
    }
    s3_actions = ["s3:GetObject", "s3:PutObject"]
    if multipart_cleanup:
        # Optional cleanup of the worker's incomplete multipart uploads only.
        # This does NOT grant DeleteObject; omit if using single-part uploads.
        s3_actions.append("s3:AbortMultipartUpload")
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "OneInstanceOneBatchObjectsUntilExpiry", "Effect": "Allow",
         "Action": s3_actions, "Resource": f"arn:aws:s3:::{BUCKET}/{prefix}*", "Condition": conditions},
        {"Sid": "OneInstanceAuthorizedCompletionTopicUntilExpiry", "Effect": "Allow",
         "Action": "sns:Publish", "Resource": TOPIC, "Condition": conditions},
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--launch-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multipart-cleanup", action="store_true")
    args = parser.parse_args()
    policy = build_policy(args.instance_id, args.prefix, dt.datetime.fromisoformat(args.launch_utc.replace("Z", "+00:00")),
                          multipart_cleanup=args.multipart_cleanup)
    args.output.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps({"generated_only_not_applied": True, "instance_id": args.instance_id,
        "batch_prefix": args.prefix, "expires_utc": policy["Statement"][0]["Condition"]["DateLessThan"]["aws:CurrentTime"],
        "output": str(args.output), "actions": [s["Action"] for s in policy["Statement"]]}))


if __name__ == "__main__":
    main()
