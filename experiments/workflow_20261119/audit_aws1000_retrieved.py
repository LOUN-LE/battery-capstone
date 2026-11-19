"""Read-only independent CSV and optional full-trace audit; never rerun physics.

Usage: python work/audit_aws1000_retrieved.py --input RESULTS_DIR --output AUDIT.json
       [--trace-root RESULTS_WITH_CASE_DIRECTORIES] [--completion RECEIPT.json]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
APPROVED = WORKSPACE / "work/risk1000_release"
FEATURES = ["ambient_temperature_C", "charge_current_A", "initial_soc", "heat_transfer_W_m2K"]
INDICATORS = ["none", "plating", "elevated_temperature", "both"]
PRIORITIES = ["low", "medium", "high"]
GENERATOR_SHA = "7353b1d91a81c1ce807d8d8565178e0b06b4250c39c8e68bd45974612b291102"
PHYSICS_SHA = "70d423dc4798c072c5d8066e7c91d8dd28b4bec57ecf6a0d45f89ff0172f5682"
POLICY_SHA = "6da69059bda985624329a2a572631e2213e5728b9eb72b6a4c94b48b126f4bde"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        assert names and len(set(names)) == len(names), f"Duplicate or missing columns: {path}"
        rows = list(reader)
    assert all(None not in row for row in rows), f"Uneven CSV record length: {path}"
    return rows


def finite(value):
    x = float(value)
    assert math.isfinite(x), f"Nonfinite number: {value}"
    return x


def boolean(value):
    assert value in {"True", "False"}, f"Not explicit boolean: {value!r}"
    return value == "True"


def near(a, b, atol=1e-8, rtol=1e-9):
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def expected_split(design):
    group = {row["scenario_id"]: json.dumps([row["ambient_temperature_C"], row["charge_current_A"],
                                           row["initial_soc"]], separators=(",", ":")) for row in design}
    order = sorted(set(group.values()), key=lambda value:
                   object_hash({"seed": "risk1000-fixed-split-v1", "group": value}))
    assert len(order) == 125
    assignments = {value: "train" if index < 87 else "validation" if index < 106 else "test"
                   for index, value in enumerate(order)}
    return {sid: (assignments[value], value) for sid, value in group.items()}


def label_from_summary(row):
    if row["status"] != "completed":
        return None, None, None, None
    plating = boolean(row["plating_event"])
    elevated = finite(row["maximum_temperature_C"]) >= 40.0
    count = int(plating) + int(elevated)
    kind = "both" if count == 2 else "plating" if plating else "elevated_temperature" if elevated else "none"
    return kind, PRIORITIES[count], plating, elevated


def count_table(rows, key, categories):
    counter = Counter(row[key] for row in rows)
    return {category: counter[category] for category in categories}


def entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(n / total * math.log2(n / total) for n in counts.values()) if total else None


def numeric_summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    def percentile(fraction):
        index = (len(values) - 1) * fraction
        left = int(index)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)
    return {"n": len(values), "min": values[0], "p05": percentile(.05), "median": percentile(.5),
            "p95": percentile(.95), "max": values[-1]}


def average_ranks(values):
    positions = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[positions[end]] == values[positions[start]]:
            end += 1
        rank = .5 * (start + end - 1)
        for i in range(start, end):
            result[positions[i]] = rank
        start = end
    return result


def rank_correlation(x, y):
    x, y = average_ranks(x), average_ranks(y)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x)**2 for a in x) * sum((b - mean_y)**2 for b in y))
    return numerator / denominator if denominator else None


def simple_baselines(rows, target):
    train = [row for row in rows if row["dataset_split"] == "train"]
    majority = sorted(Counter(row[target] for row in train).items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    by_temperature = defaultdict(Counter)
    for row in train:
        by_temperature[finite(row[FEATURES[0]])][row[target]] += 1
    mapping = {temperature: sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
               for temperature, counts in by_temperature.items()}
    result = {"train_majority_label": majority, "temperature_majority_mapping_fit_train_only": mapping}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["dataset_split"] == split]
        result[split] = {"n": len(selected),
                         "constant_train_majority_accuracy": sum(row[target] == majority for row in selected) / len(selected),
                         "temperature_only_train_majority_accuracy": sum(
                             row[target] == mapping.get(finite(row[FEATURES[0]]), majority) for row in selected) / len(selected)}
    return result


def audit_trace(path, scenario, summary, provenance):
    sid = scenario["scenario_id"]
    result = load_json(path / "result.json")
    assert result["provenance"] == provenance, sid + " provenance"
    assert result["identity_sha256"] == object_hash({"scenario": scenario, "provenance": provenance}), sid + " identity"
    detail = result["summary"]
    for key, value in scenario.items():
        assert detail[key] == value, (sid, key)
    assert detail["status"] == summary["status"] == "completed", sid
    assert digest(path / "trace.csv") == result["trace_sha256"], sid + " trace digest"
    assert detail["fire_label"] is None and detail["thermal_runaway_label"] is None
    assert detail["physical_risk_validated"] is False
    quantity_names = ["metallic_lithium_mol_m3", "dead_lithium_mol_m3", "plating_inventory_Ah", "sei_loss_Ah"]
    quantities = {}
    first, previous, onset, active_start = None, None, None, None
    phases = []
    max_temperature = -math.inf
    min_temperature = math.inf
    phase_balance = {}
    row_count = 0
    with (path / "trace.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            assert row["scenario_id"] == sid and row["phase"] in {"charge", "rest"}
            numeric = {key: finite(value) for key, value in row.items() if key not in {"scenario_id", "phase"}}
            row_count += 1
            stamp = numeric["time_s"]
            if first is None:
                first = numeric
            if previous is not None:
                assert stamp >= previous["time_s"], sid + " reversed time"
            phase = row["phase"]
            if not phases or phases[-1] != phase:
                phases.append(phase)
                phase_balance[phase] = {"start_capacity": numeric["discharge_capacity_Ah"], "integral": 0.}
            elif previous is not None:
                phase_balance[phase]["integral"] += .5 * (numeric["current_A"] + previous["current_A"]) * (stamp - previous["time_s"]) / 3600
            phase_balance[phase]["end_capacity"] = numeric["discharge_capacity_Ah"]
            assert near(numeric["current_A"], -scenario["charge_current_A"] if phase == "charge" else 0., atol=1e-8), sid + " current"
            max_temperature = max(max_temperature, numeric["temperature_C"])
            min_temperature = min(min_temperature, numeric["temperature_C"])
            if numeric["plating_net_current_A_m2"] < -scenario["plating_threshold_A_m2"]:
                if active_start is None:
                    active_start = stamp
                if onset is None and stamp - active_start >= scenario["plating_persistence_s"]:
                    onset = active_start
            else:
                active_start = None
            for quantity in quantity_names:
                value = numeric[quantity]
                if quantity not in quantities:
                    quantities[quantity] = {"start": value, "peak": value, "end": value}
                quantities[quantity]["peak"] = max(quantities[quantity]["peak"], value)
                quantities[quantity]["end"] = value
            previous = numeric
    assert phases == ["charge", "rest"], (sid, phases)
    assert row_count == int(summary["sample_count"]) == detail["sample_count"], sid + " rows"
    assert near(first["temperature_C"], scenario["initial_temperature_C"], atol=1e-6), sid + " initial temperature"
    assert near(first["time_s"], 0), sid + " initial clock"
    assert near(max_temperature, summary["maximum_temperature_C"]) and near(max_temperature, detail["maximum_temperature_C"]), sid + " Tmax"
    assert (onset is not None) == detail["plating_event"] == boolean(summary["plating_event"]), sid + " plating"
    if onset is None:
        assert detail["plating_onset_s"] is None and summary["plating_onset_s"] == ""
    else:
        assert near(onset, detail["plating_onset_s"]) and near(onset, summary["plating_onset_s"]), sid + " onset"
    assert near(previous["time_s"], detail["total_duration_s"])
    assert near(detail["total_duration_s"] - detail["charge_end_s"], scenario["rest_s"], atol=1e-6)
    maximum_balance_error = max(abs(p["end_capacity"] - p["start_capacity"] - p["integral"]) for p in phase_balance.values())
    assert maximum_balance_error <= 1e-6, sid + " charge balance"
    for quantity, values in quantities.items():
        values.update(net_change=values["end"] - values["start"], peak_increase=values["peak"] - values["start"])
        for name, value in values.items():
            key = quantity + "_" + name
            assert near(value, detail[key]) and near(value, summary[key]), (sid, key)
    return {"rows": row_count, "maximum_charge_balance_error_Ah": maximum_balance_error,
            "minimum_temperature_C": min_temperature, "maximum_temperature_C": max_temperature}


def audit(input_dir, output, trace_root=None, completion_path=None):
    source = Path(input_dir)
    design = load_json(APPROVED / "reviewed_design/design_scenarios.json")
    policy = load_json(APPROVED / "reviewed_design/design_plan.json")
    release = load_json(APPROVED / "release_manifest.json")
    assert release["files"]["simulation/risk_dataset1000.py"] == GENERATOR_SHA
    assert release["files"]["simulation/risk_scenarios.py"] == PHYSICS_SHA
    assert digest(APPROVED / "reviewed_design/design_scenarios.json") == release["files"]["reviewed_design/design_scenarios.json"]
    assert object_hash(policy) == POLICY_SHA and load_json(source / "design_plan.json") == policy
    requested = {row["scenario_id"]: row for row in design}
    assert len(requested) == 1000
    split = expected_split(design)
    filenames = ["training_dataset.csv", "features.csv", "labels.csv", "scenario_summary.csv"]
    tables = {name: load_csv(source / name) for name in filenames}
    hashes_before = {name: digest(source / name) for name in filenames + ["dataset_report.json", "cloud_verification.json", "design_plan.json"]}
    maps = {}
    for name, rows in tables.items():
        assert len(rows) == 1000 and len({row["scenario_id"] for row in rows}) == 1000, name + " count/duplicates"
        assert {row["scenario_id"] for row in rows} == set(requested), name + " identity set"
        maps[name] = {row["scenario_id"]: row for row in rows}
    expected_training_columns = {"scenario_id", "dataset_split", *FEATURES, "indicator_type", "screening_priority", "training_eligible", "status"}
    assert set(tables["training_dataset.csv"][0]) == expected_training_columns, "Unexpected training input/output columns"
    assert set(tables["features.csv"][0]) == {"scenario_id", "dataset_split", "protocol_group", *FEATURES}
    report = load_json(source / "dataset_report.json")
    cloud = load_json(source / "cloud_verification.json")
    for name, field in [("training_dataset.csv", "training_dataset_sha256"), ("features.csv", "features_sha256"),
                        ("labels.csv", "labels_sha256"), ("scenario_summary.csv", "summary_sha256")]:
        assert hashes_before[name] == report[field], name + " report digest"
    assert cloud["verification_passed"] is True and cloud["final_csv_sha256"] == hashes_before["training_dataset.csv"]
    assert cloud["source_manifest_sha256"] == digest(APPROVED / "release_manifest.json")
    provenance = report["source"]
    assert provenance["dataset_generator"] == {"schema": "bounded-screening-dataset1000-v1", "module_sha256": GENERATOR_SHA,
                                              "policy_sha256": POLICY_SHA, "scenario_definitions_sha256": object_hash(design)}
    for path, expected in provenance["sources"].items():
        assert release["files"][path] == expected, path + " provenance"
    assert provenance["dependencies"] == {"pybammsolvers": "0.9.1", "casadi": "3.7.2", "scipy": "1.17.1"}
    assert provenance["pybamm_version"] == "26.8.0.0" and provenance["numpy_version"] == "2.4.6"
    assert report["physical_risk_validated"] is False and report["fire_prediction_validated"] is False
    assert report["thermal_runaway_prediction_validated"] is False
    rows = tables["training_dataset.csv"]
    verified_traces = []
    for sid, condition in requested.items():
        summary = maps["scenario_summary.csv"][sid]
        training = maps["training_dataset.csv"][sid]
        feature = maps["features.csv"][sid]
        label = maps["labels.csv"][sid]
        for key, value in condition.items():
            if key != "scenario_id":
                assert near(finite(summary[key]), value, atol=1e-12, rtol=0), (sid, key)
        for key in FEATURES:
            assert finite(training[key]) == finite(feature[key]) == condition[key], (sid, key)
        for row in (training, feature, summary):
            assert row["dataset_split"] == split[sid][0], sid + " split"
        assert feature["protocol_group"] == summary["protocol_group"] == split[sid][1]
        assert training["status"] == label["status"] == summary["status"]
        expected_type, expected_priority, plating, elevated = label_from_summary(summary)
        for row in (training, label, summary):
            assert boolean(row["training_eligible"]) == (summary["status"] == "completed")
            assert row["indicator_type"] == (expected_type or "")
            assert row["screening_priority"] == (expected_priority or "")
        for row in (label, summary):
            assert row["thermal_runaway_label"] == "" and row["fire_label"] == ""
            assert boolean(row["physical_risk_validated"]) is False
            assert row["classification_basis"] == "SPMeT_simulation_policy_proxy"
        if summary["status"] == "completed":
            assert boolean(label["plating_flag"]) == plating and boolean(label["elevated_temperature_flag"]) == elevated
            assert int(label["screening_flag_count"]) == int(plating) + int(elevated)
            if plating:
                assert finite(summary["plating_onset_s"]) >= 0
            else:
                assert summary["plating_onset_s"] == ""
            assert int(summary["sample_count"]) >= 2
            assert finite(summary["charge_balance_error_Ah"]) <= 1e-6
            assert summary["final_termination"] == "final time"
            assert near(finite(summary["total_duration_s"]) - finite(summary["charge_end_s"]), condition["rest_s"], atol=1e-6)
            if trace_root:
                verified_traces.append(audit_trace(Path(trace_root) / sid, condition, summary, provenance))
                if len(verified_traces) % 100 == 0:
                    print(f"Verified {len(verified_traces)}/1000 raw traces", flush=True)
    conditions = [tuple(finite(row[key]) for key in FEATURES) for row in rows]
    assert len(set(conditions)) == 1000
    status_counts = count_table(rows, "status", ["completed", "censored", "failed", "pending"])
    assert status_counts == report["status_counts"] == cloud["status_counts"]
    assert status_counts == {"completed": 1000, "censored": 0, "failed": 0, "pending": 0}
    assert report["all_1000_completed"] is True and report["supervised_target_rows"] == 1000
    assert report["scenario_count"] == 1000 and report["unique_physical_conditions"] == 1000
    assert report["workers"] == 8 and finite(report["wall_seconds"]) > 0
    by_split = {name: [row for row in rows if row["dataset_split"] == name] for name in ["train", "validation", "test"]}
    assert {name: len(selected) for name, selected in by_split.items()} == {"train": 696, "validation": 152, "test": 152}
    groups = defaultdict(set)
    for sid in requested:
        groups[split[sid][1]].add(split[sid][0])
    assert len(groups) == 125 and all(len(value) == 1 for value in groups.values())
    indicators = count_table(rows, "indicator_type", INDICATORS)
    priorities = count_table(rows, "screening_priority", PRIORITIES)
    assert indicators == report["indicator_counts"] and priorities == report["screening_priority_counts"]
    flagged_count = 1000 - indicators["none"]
    min_calls_95_recall = math.ceil(.95 * flagged_count)
    temperature_distribution = {}
    cooling_sensitive_groups = 0
    for temperature in sorted({finite(row[FEATURES[0]]) for row in rows}):
        selected = [row for row in rows if finite(row[FEATURES[0]]) == temperature]
        temperature_distribution[str(temperature)] = {"n": len(selected), "indicator_type": count_table(selected, "indicator_type", INDICATORS),
                                                     "screening_priority": count_table(selected, "screening_priority", PRIORITIES)}
    for group in groups:
        selected = [maps["training_dataset.csv"][sid] for sid in requested if split[sid][1] == group]
        assert len(selected) == 8
        cooling_sensitive_groups += len({row["indicator_type"] for row in selected}) > 1
    summaries = tables["scenario_summary.csv"]
    positives = [row for row in summaries if boolean(row["plating_event"])]
    thermal_rows = [row for row in summaries if finite(row["maximum_temperature_C"]) >= 40.]
    onset_zero = sum(finite(row["plating_onset_s"]) == 0 for row in positives)
    thermal_initial = sum(finite(row["initial_temperature_C"]) >= 40. for row in thermal_rows)
    degradation_columns = ["sei_loss_Ah_net_change", "dead_lithium_mol_m3_net_change",
                           "plating_inventory_Ah_net_change", "metallic_lithium_mol_m3_peak_increase"]
    degradation = {}
    priority_numbers = [PRIORITIES.index(row["screening_priority"]) for row in summaries]
    for column in degradation_columns:
        values = [finite(row[column]) for row in summaries]
        degradation[column] = {
            "all": numeric_summary(values),
            "by_priority": {level: numeric_summary([finite(row[column]) for row in summaries if row["screening_priority"] == level]) for level in PRIORITIES},
            "spearman_correlation_with_flag_count_priority": rank_correlation(priority_numbers, values),
        }
    trace_count = sum(int(row["sample_count"]) for row in summaries)
    assert trace_count == cloud["trace_rows"]
    if trace_root:
        assert len(verified_traces) == 1000 and sum(row["rows"] for row in verified_traces) == trace_count
    completion_verified = False
    if completion_path:
        completion = load_json(completion_path)
        assert completion["success"] is True and completion["verified_1000_cases"] is True
        assert completion["uploads"]["training_dataset.csv"]["sha256"] == hashes_before["training_dataset.csv"]
        completion_verified = True
    assert hashes_before == {name: digest(source / name) for name in hashes_before}, "Input files changed during read-only audit"
    result = {
        "audit_passed": True, "audit_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent CSV/design/provenance/label/split audit; no physics rerun",
        "input_directory": str(source.resolve()), "source_file_sha256": hashes_before,
        "completion_receipt_verified": completion_verified,
        "raw_traces_independently_recomputed": len(verified_traces), "raw_trace_rows_if_checked": trace_count if trace_root else None,
        "raw_trace_scope_note": "All raw traces rechecked" if trace_root else "Raw traces not locally supplied; only cloud verifier reports trace recomputation",
        "requested_unique_conditions": 1000, "status_counts": status_counts, "training_eligible": 1000,
        "indicator_counts": indicators, "screening_priority_counts": priorities,
        "split_counts": {name: len(selected) for name, selected in by_split.items()}, "unique_protocol_groups": 125,
        "group_overlap_count": 0,
        "split_class_counts": {name: {"indicator_type": count_table(selected, "indicator_type", INDICATORS),
                                      "screening_priority": count_table(selected, "screening_priority", PRIORITIES)} for name, selected in by_split.items()},
        "missing_indicator_classes": [name for name, count in indicators.items() if count == 0],
        "missing_priority_classes": [name for name, count in priorities.items() if count == 0],
        "temperature_distribution": temperature_distribution,
        "plating_positive_count": len(positives), "plating_onset_zero_count": onset_zero,
        "elevated_temperature_positive_count": len(thermal_rows), "elevated_at_initial_temperature_count": thermal_initial,
        "cooling_groups_with_indicator_variation": cooling_sensitive_groups,
        "recorded_trace_rows": trace_count,
        "maximum_temperature_C_range": [min(finite(row["maximum_temperature_C"]) for row in summaries), max(finite(row["maximum_temperature_C"]) for row in summaries)],
        "label_entropy_bits": {key: entropy(row[key] for row in rows) for key in ["indicator_type", "screening_priority"]},
        "simple_diagnostics_not_model_selection": {key: simple_baselines(rows, key) for key in ["indicator_type", "screening_priority"]},
        "short_protocol_degradation_proxy_outputs": degradation,
        "conditional_dt_call_count_bound": {
            "positive_definition": "at least one existing proxy flag, indicator_type != none",
            "positive_count": flagged_count, "negative_count": indicators["none"],
            "required_proxy_positive_recall": .95,
            "minimum_dt_calls_if_each_recalled_positive_requires_its_own_dt_confirmation": min_calls_95_recall,
            "maximum_avoided_calls_vs_1000_call_every_scenario_baseline": 1000 - min_calls_95_recall,
            "maximum_avoided_call_fraction": (1000 - min_calls_95_recall) / 1000,
            "assumptions": "One distinct DT call per confirmed positive. Ideal selection with zero false-positive calls. A necessary count bound, not an achieved runtime/cost result or comparison against a measured ML-gated baseline.",
        },
        "ml_suitability": {
            "usable_for_bounded_simulator_surrogate_pipeline": True,
            "validated_as_lifetime_or_degradation_severity_ordering": False,
            "usable_as_remaining_useful_life_ground_truth": False,
            "usable_as_empirical_fire_or_physical_safety_labels": False,
            "independent_real_cell_experiments": 0,
            "paired_same_BMS_input_ML_DT_vs_DL_DT_experiment_completed": False,
            "allowed_input_columns": FEATURES,
            "same_input_comparison_requirement": "Use identical predeclared BMS observation windows/features, cases, target definition, splits, DT reference and budgets for both pipelines. Current four scenario settings are not a paired real-BMS experiment; full future DT outputs cannot be gating inputs.",
            "interpretation": "Deterministic same-model interpolation data. Flag-count priority is not a validated lifetime ranking. Short-protocol SEI/dead-lithium/plating outputs may support bounded surrogate regression, but are not cycle life or RUL. Missing classes, initial-state onset flags and temperature dominance must be assessed.",
        },
    }
    if verified_traces:
        result["maximum_recomputed_charge_balance_error_Ah"] = max(row["maximum_charge_balance_error_Ah"] for row in verified_traces)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["audit_passed", "status_counts", "indicator_counts", "screening_priority_counts", "split_counts", "plating_onset_zero_count", "elevated_at_initial_temperature_count", "raw_traces_independently_recomputed"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--completion", type=Path)
    args = parser.parse_args()
    audit(args.input, args.output, args.trace_root, args.completion)
