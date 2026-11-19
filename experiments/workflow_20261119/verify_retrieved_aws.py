"""Verify authentic retrieved pilot artifacts; optionally extract and combine CSVs.

python work/verify_retrieved_aws.py --archive FILE --expected-sha256 HASH --extract-to work/aws_retrieved --output outputs/aws32
python work/verify_retrieved_aws.py --input work/aws_retrieved --output outputs/aws32
Use --combine-traces to additionally produce a single time-series CSV.
"""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

TASK = Path(__file__).resolve().parents[1]
RUNNER_SHA = "70d423dc4798c072c5d8066e7c91d8dd28b4bec57ecf6a0d45f89ff0172f5682"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + value)))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_scenarios():
    result = []
    for index, (temperature, current, soc, cooling) in enumerate(itertools.product(
            (0., 10., 25., 40.), (.5, 2.5), (.2, .6), (5., 10.)), 1):
        result.append({"scenario_id": f"case_{index:03d}", "initial_soc": soc,
            "initial_temperature_C": temperature, "ambient_temperature_C": temperature,
            "charge_current_A": current, "heat_transfer_W_m2K": cooling,
            "charge_horizon_s": 1800., "rest_s": 300., "voltage_limit_V": 4.2,
            "temperature_stop_C": 60., "sample_period_s": 1., "plating_threshold_A_m2": .001,
            "plating_persistence_s": 5., "mesh_scale": 8, "rtol": 1e-7, "atol": 1e-9})
    return result


def extract_verified(archive, expected_hash, destination):
    require(expected_hash and digest(archive) == expected_hash.lower(), "Archive SHA256 differs from remote evidence")
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in member.name
                    and ":" not in member.name, "Unsafe archive path")
            require(member.name not in seen, "Duplicate archive member")
            seen.add(member.name)
            require(member.isdir() or member.isfile(), "Archive links or special files are unsupported")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def numeric(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite numeric CSV value")
    return result


def verify(root, expected_release):
    results = root / "results" if (root / "results").is_dir() else root
    bundle = results.parent
    receipt = read_json(results / "aws_execution_receipt.json")
    manifest = read_json(results / "manifest.json")
    report = read_json(results / "report.json")
    release = read_json(bundle / "release_manifest.json")
    require(release == read_json(expected_release), "Retrieved release differs from reviewed release")
    require(digest(bundle / "release_manifest.json") == receipt["release_manifest_sha256"], "Release receipt hash mismatch")
    require(release["files"]["simulation/risk_scenarios.py"] == RUNNER_SHA, "Unexpected runner version")
    for path in ("cloud_receipt.py", "requirements-pilot.txt"):
        require(digest(bundle / path) == release["files"][path], "Bundled source hash mismatch " + path)
    require(receipt.get("verification_passed") is True, "AWS completion verification missing")
    require(receipt["max_workers"] == 8 and report["workers"] == 8 and report["scenario_count"] == 32, "Worker/count mismatch")
    require(receipt["ec2"]["instanceId"].startswith("i-") and receipt["cpu_count"] >= 8, "Invalid EC2 execution identity")
    start, end = (dt.datetime.fromisoformat(receipt[name]) for name in ("start_utc", "end_utc"))
    require(start.tzinfo is not None and end > start, "Invalid execution interval")
    expected = expected_scenarios()
    require(manifest["scenarios"] == expected, "Unexpected scenario design")
    provenance = manifest["provenance"]
    require(report["source"] == provenance and receipt["report"] == report, "Provenance/report mismatch")
    for path, sha256 in provenance["sources"].items():
        require(release["files"].get(path) == sha256, "Case source differs from release")
    require(provenance["sources"]["simulation/risk_scenarios.py"] == RUNNER_SHA, "Wrong case source")
    packages = {"pybamm": provenance["pybamm_version"], "numpy": provenance["numpy_version"], **provenance["dependencies"]}
    require(packages == receipt["packages"], "Installed packages differ from case provenance")
    summary_path = results / "scenario_summary.csv"
    summary_hash = digest(summary_path)
    require(summary_hash == report["summary_sha256"] == receipt["summary_sha256"], "Summary hash mismatch")
    rows = read_csv(summary_path)
    require(len(rows) == 32 and {r["scenario_id"] for r in rows} == {s["scenario_id"] for s in expected}, "Expected 32 unique rows")
    by_id = {s["scenario_id"]: s for s in expected}
    require(len(list(results.glob("case_*/result.json"))) == 32, "Unexpected case file count")
    trace_files, checks, status_counts, total = [], [], {}, 0
    for row in rows:
        case = row["scenario_id"]
        scenario = by_id[case]
        record = read_json(results / case / "result.json")
        summary = record["summary"]
        require(record["provenance"] == provenance, "Per-case provenance mismatch")
        identity = hashlib.sha256(json.dumps({"scenario": scenario, "provenance": provenance}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(identity == record["identity_sha256"] == summary["identity_sha256"], "Case identity hash mismatch")
        for key, value in summary.items():
            require(row.get(key) == ("" if value is None else str(value)), f"CSV/JSON mismatch {case}/{key}")
            if isinstance(value, (float, int)):
                require(math.isfinite(value), "Nonfinite summary")
        for key, value in scenario.items():
            require(summary[key] == value, "Scenario input mismatch")
        status = summary["status"]
        require(status in ("completed", "censored", "failed"), "Unknown status")
        status_counts[status] = status_counts.get(status, 0) + 1
        require(summary["fire_label"] is None and summary["thermal_runaway_label"] is None
                and summary["physical_risk_validated"] is False, "Invalid physical safety label")
        if status != "completed":
            require(summary["plating_event"] is not False, "Incomplete scenario labeled event-free")
        if status == "failed":
            require(summary["plating_event"] is None, "Failed case has event label")
            continue
        trace_path = results / case / "trace.csv"
        trace_hash = digest(trace_path)
        relative = f"{case}/trace.csv"
        require(trace_hash == record["trace_sha256"] == receipt["trace_sha256"][relative], "Trace hash mismatch")
        trace = read_csv(trace_path)
        require(len(trace) == summary["sample_count"] and len(trace) >= 2, "Trace count mismatch")
        previous = None
        onset_start = onset = None
        temperatures = []
        for sample in trace:
            require(sample["scenario_id"] == case and sample["phase"] in ("charge", "rest"), "Trace identity/phase mismatch")
            values = {key: numeric(value) for key, value in sample.items() if key not in ("scenario_id", "phase")}
            stamp = values["time_s"]
            require(previous is None or stamp > previous, "Nonmonotonic trace clock")
            previous = stamp
            temperatures.append(values["temperature_C"])
            target_current = -scenario["charge_current_A"] if sample["phase"] == "charge" else 0.
            require(abs(values["current_A"] - target_current) < 1e-7, "Current protocol mismatch")
            if onset is None:
                if values["plating_net_current_A_m2"] < -scenario["plating_threshold_A_m2"]:
                    if onset_start is None:
                        onset_start = stamp
                    if stamp - onset_start >= scenario["plating_persistence_s"]:
                        onset = onset_start
                else:
                    onset_start = None
        require(abs(temperatures[0] - scenario["initial_temperature_C"]) < 1e-6, "Initial temperature mismatch")
        require(abs(max(temperatures) - summary["maximum_temperature_C"]) < 1e-9, "Temperature summary mismatch")
        require(onset == summary["plating_onset_s"], "Event onset mismatch")
        expected_label = True if onset is not None else (False if status == "completed" else None)
        require(summary["plating_event"] is expected_label, "Event label mismatch")
        for quantity in ("metallic_lithium_mol_m3", "dead_lithium_mol_m3", "plating_inventory_Ah", "sei_loss_Ah"):
            values = [numeric(sample[quantity]) for sample in trace]
            derived = {"start": values[0], "peak": max(values), "end": values[-1],
                       "net_change": values[-1] - values[0], "peak_increase": max(values) - values[0]}
            for suffix, value in derived.items():
                require(math.isclose(value, summary[f"{quantity}_{suffix}"], abs_tol=1e-10, rel_tol=1e-10), "Inventory summary mismatch")
        total += len(trace)
        trace_files.append(trace_path)
        checks.append({"scenario_id": case, "status": status, "trace_rows": len(trace), "sha256": trace_hash})
    expected_counts = {status: status_counts.get(status, 0) for status in ("completed", "censored", "failed")}
    require(expected_counts == report["status_counts"] and expected_counts["failed"] == 0, "Unexpected failures/status count")
    require(total == receipt["trace_rows"] and len(receipt["trace_sha256"]) == len(trace_files), "Receipt trace total mismatch")
    return results, trace_files, {"passed": True, "scope": "retrieval_integrity_and_model_output_contract",
        "instance": receipt["ec2"], "start_utc": receipt["start_utc"], "end_utc": receipt["end_utc"],
        "cloud_elapsed_seconds": (end - start).total_seconds(), "simulation_wall_seconds": report["wall_seconds"],
        "cases": checks, "status_counts": expected_counts, "trace_rows": total,
        "summary_sha256": summary_hash, "runner_sha256": RUNNER_SHA,
        "physical_risk_validated": False, "thermal_runaway_validated": False, "fire_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--archive", type=Path)
    selection.add_argument("--input", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--expected-release", type=Path, default=TASK / "work/aws_pilot32_release/release_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combine-traces", action="store_true")
    args = parser.parse_args()
    if args.archive:
        require(args.extract_to is not None, "--extract-to required for archive")
        extract_verified(args.archive, args.expected_sha256, args.extract_to)
    root = args.extract_to if args.archive else args.input
    results, traces, verification = verify(root, args.expected_release)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(results / "scenario_summary.csv", args.output / "scenario_summary.csv")
    with zipfile.ZipFile(args.output / "all_traces.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for trace in traces:
            archive.write(trace, trace.relative_to(results).as_posix())
            record = trace.parent / "result.json"
            archive.write(record, record.relative_to(results).as_posix())
        for name in ("manifest.json", "report.json", "aws_execution_receipt.json"):
            archive.write(results / name, name)
    if args.combine_traces:
        with (args.output / "all_traces.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = None
            for trace in traces:
                rows = read_csv(trace)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                writer.writerows(rows)
    verification["outputs"] = {name: digest(args.output / name) for name in ("scenario_summary.csv", "all_traces.zip")}
    (args.output / "retrieval_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in verification.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
