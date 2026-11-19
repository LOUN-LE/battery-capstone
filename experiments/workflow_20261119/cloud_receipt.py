"""Record EC2 execution identity and verify the completed 32-case pilot."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metadata():
    base = "http://169.254.169.254/latest/"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    with opener.open(req, timeout=4) as response:
        token = response.read().decode()
    req = urllib.request.Request(base + "dynamic/instance-identity/document",
                                 headers={"X-aws-ec2-metadata-token": token})
    with opener.open(req, timeout=4) as response:
        identity = json.load(response)
    return {k: identity[k] for k in ("instanceId", "instanceType", "region", "availabilityZone", "architecture")}

def main():
    OUT.mkdir(exist_ok=True)
    mode = sys.argv[1]
    if mode == "before":
        expected = json.loads((ROOT / "release_manifest.json").read_text())
        for name, value in expected["files"].items():
            if digest(ROOT / name) != value:
                raise ValueError("Release digest mismatch: " + name)
        receipt = {"start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "ec2": metadata(),
                   "hostname": platform.node(), "platform": platform.platform(),
                   "python": sys.version, "cpu_count": os.cpu_count(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("pybamm", "pybammsolvers", "numpy", "scipy", "casadi")},
                   "max_workers": 8, "release_manifest_sha256": digest(ROOT / "release_manifest.json")}
        if receipt["cpu_count"] < 8:
            raise ValueError("Selected EC2 instance has fewer than 8 CPUs")
        (OUT / "aws_execution_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2), flush=True)
    elif mode == "after":
        receipt_path = OUT / "aws_execution_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        report = json.loads((OUT / "report.json").read_text())
        manifest = json.loads((OUT / "manifest.json").read_text())
        with (OUT / "scenario_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 32 or report["scenario_count"] != 32 or report["workers"] != 8:
            raise ValueError("Expected 32 scenarios and 8 workers")
        if report["status_counts"]["failed"]:
            raise ValueError("At least one scenario failed")
        if len({r["scenario_id"] for r in rows}) != 32:
            raise ValueError("Nonunique scenario IDs")
        from simulation.risk_scenarios import pilot_scenarios
        from dataclasses import asdict
        if manifest["scenarios"] != [asdict(s) for s in pilot_scenarios()]:
            raise ValueError("Scenario design does not match reviewed pilot")
        trace_rows = 0
        hashes = {}
        for row in rows:
            directory = OUT / row["scenario_id"]
            result = json.loads((directory / "result.json").read_text())
            trace = directory / "trace.csv"
            if result["trace_sha256"] != digest(trace):
                raise ValueError("Trace digest mismatch")
            with trace.open(encoding="utf-8-sig", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            if count != int(row["sample_count"]):
                raise ValueError("Trace row count mismatch")
            if row["thermal_runaway_label"] or row["fire_label"] or row["physical_risk_validated"] != "False":
                raise ValueError("Unexpected physical risk claim")
            trace_rows += count
            hashes[trace.relative_to(OUT).as_posix()] = digest(trace)
        receipt.update(end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       report=report, trace_rows=trace_rows, trace_sha256=hashes,
                       summary_sha256=digest(OUT / "scenario_summary.csv"), verification_passed=True)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({k: receipt[k] for k in ("start_utc", "end_utc", "trace_rows", "verification_passed")}), flush=True)
    else:
        raise ValueError("Expected before or after")

if __name__ == "__main__":
    main()
