import csv
from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation import risk_dataset1000 as dataset
from simulation.rebuilt_lifetime import sha, write_json


class RiskDataset1000Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def summary(self, plating=False, temperature=35., status="completed"):
        return {"status": status, "plating_event": plating, "maximum_temperature_C": temperature,
                "physical_risk_validated": False, "thermal_runaway_label": None, "fire_label": None}

    def prepared(self):
        design, assignments = dataset.write_design(self.output)
        source = dataset.execution_provenance(design)
        manifest = {"schema": dataset.SCHEMA, "provenance": source,
                    "scenarios": [asdict(s) for s in design], "policy": dataset.POLICY,
                    "features_sha256": sha(self.output / "features.csv")}
        write_json(self.output / "manifest.json", manifest)
        return design, assignments, source

    def record(self, scenario, source, status="completed", plating=False, temperature=35.):
        directory = self.output / scenario.scenario_id
        directory.mkdir()
        summary = {**asdict(scenario), **self.summary(plating, temperature, status)}
        result = {"summary": summary, "provenance": source,
                  "identity_sha256": dataset.physics.content_hash({"scenario": asdict(scenario), "provenance": source})}
        if status != "failed":
            trace = directory / "trace.csv"
            trace.write_text("synthetic_software_fixture\n1\n", encoding="utf-8")
            result["trace_sha256"] = sha(trace)
        write_json(directory / "result.json", result)
        return directory

    def read_csv(self, filename):
        with (self.output / filename).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_exact_1000_unique_fresh_conditions_inside_previous_bounds(self):
        design = dataset.scenarios()
        self.assertEqual(len(design), 1000)
        self.assertEqual(len({s.scenario_id for s in design}), 1000)
        self.assertEqual(len({dataset.physical_key(s) for s in design}), 1000)
        self.assertEqual([asdict(s) for s in design], [asdict(s) for s in dataset.scenarios()])
        for s in design:
            self.assertTrue(0 <= s.ambient_temperature_C <= 40)
            self.assertEqual(s.initial_temperature_C, s.ambient_temperature_C)
            self.assertTrue(.5 <= s.charge_current_A <= 2.5)
            self.assertTrue(.2 <= s.initial_soc <= .6)
            self.assertTrue(5 <= s.heat_transfer_W_m2K <= 10)
            self.assertEqual((s.mesh_scale, s.charge_horizon_s, s.rest_s), (8, 1800., 300.))

    def test_split_groups_keep_all_eight_cooling_replicates_together(self):
        design = dataset.scenarios()
        split = dataset.split_assignments(design)
        self.assertEqual(dict(dataset.Counter(a["dataset_split"] for a in split.values())),
                         {"train": 696, "validation": 152, "test": 152})
        for group in {dataset.group_key(s) for s in design}:
            members = [s for s in design if dataset.group_key(s) == group]
            self.assertEqual(len(members), 8)
            self.assertEqual(len({split[s.scenario_id]["dataset_split"] for s in members}), 1)

    def test_four_types_and_count_based_policy_not_physical_labels(self):
        for plating, temperature, indicator, priority in [
            (False, 39.999999, "none", "low"), (True, 39., "plating", "medium"),
            (False, 40., "elevated_temperature", "medium"), (True, 40., "both", "high")]:
            with self.subTest(indicator=indicator):
                label = dataset.classify(self.summary(plating, temperature))
                self.assertEqual((label["indicator_type"], label["screening_priority"]), (indicator, priority))
                self.assertTrue(label["training_eligible"])
                self.assertFalse(label["physical_risk_validated"])
                self.assertIsNone(label["fire_label"])
                self.assertIsNone(label["thermal_runaway_label"])

    def test_incomplete_even_with_observed_flags_never_gets_low_or_other_target(self):
        for status in ("censored", "failed", "pending"):
            for plating in (True, False, None):
                label = dataset.classify(self.summary(plating, 70., status))
                self.assertFalse(label["training_eligible"])
                self.assertIsNone(label["screening_priority"])
                self.assertIsNone(label["indicator_type"])

    def test_completed_missing_nonfinite_or_manufactured_physical_label_rejected(self):
        invalid = [self.summary(None), self.summary("False"), self.summary(temperature=float("nan")),
                   {**self.summary(), "fire_label": False}, {**self.summary(), "physical_risk_validated": True}]
        for row in invalid:
            with self.assertRaises(ValueError):
                dataset.classify(row)

    def test_partial_finalize_retains_1000_rows_with_only_verified_completed_target(self):
        design, _, source = self.prepared()
        self.record(design[0], source, plating=True)
        self.record(design[1], source, status="censored", temperature=60.)
        self.record(design[2], source, status="failed")
        locked = self.record(design[3], source)
        (locked / "running.lock").write_text("synthetic interrupted process fixture", encoding="utf-8")
        report = dataset.run(self.output, finalize_only=True)
        self.assertFalse(report["all_1000_completed"])
        self.assertFalse(report["execution_complete"])
        self.assertEqual(report["status_counts"], {"completed": 1, "censored": 1, "failed": 1, "pending": 997})
        self.assertEqual(report["supervised_target_rows"], 1)
        self.assertEqual(report["locked_cases_excluded"], 1)
        rows = self.read_csv("training_dataset.csv")
        self.assertEqual(len(rows), 1000)
        self.assertEqual(rows[0]["screening_priority"], "medium")
        self.assertTrue(all(row["screening_priority"] == "" for row in rows[1:]))
        self.assertEqual(set(rows[0]), {"scenario_id", "dataset_split", *dataset.FEATURES,
                                       "indicator_type", "screening_priority", "training_eligible", "status"})
        self.assertEqual(sha(self.output / "training_dataset.csv"), report["training_dataset_sha256"])

    def test_input_table_contains_only_prescribed_scenario_features(self):
        dataset.write_design(self.output)
        rows = self.read_csv("features.csv")
        self.assertEqual(len(rows), 1000)
        self.assertEqual(set(rows[0]), {"scenario_id", "dataset_split", "protocol_group", *dataset.FEATURES})
        self.assertFalse(set(rows[0]) & {"maximum_temperature_C", "plating_event", "fire_label", "indicator_type"})

    def test_tampered_trace_or_identity_never_accepted(self):
        design, _, source = self.prepared()
        directory = self.record(design[0], source)
        (directory / "trace.csv").write_text("changed fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            dataset.collect_results(design, self.output, source)

    def test_changed_policy_and_design_refuse_same_output(self):
        self.prepared()
        with patch.dict(dataset.POLICY, temperature_threshold_C=41.):
            with self.assertRaisesRegex(ValueError, "differs"):
                dataset.run(self.output, finalize_only=True)

    def test_no_execution_manifest_cannot_claim_finalized_dataset(self):
        with self.assertRaisesRegex(ValueError, "No execution manifest"):
            dataset.run(self.output, finalize_only=True)

    def test_worker_cap_cannot_be_exceeded_or_noninteger(self):
        for workers in (0, 9, True, 2.0):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                dataset.run(self.output, workers=workers)


if __name__ == "__main__":
    unittest.main()
