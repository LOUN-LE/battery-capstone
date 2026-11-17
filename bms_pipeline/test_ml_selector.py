"""Contract tests with artificial fixtures, not experimental validation data."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("ml_selector_under_test", Path(__file__).with_name("ml_selector.py"))
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def fixtures():
    rows = []
    for letter, origin in (("C", 17111.306812743416), ("D", 17282.27682086406)):
        for second in range(600):
            rows.append({"cell_id": f"imperial_expt2_2_cell_{letter}",
                         "session_id": f"imperial_expt2_2_cell_{letter}_initial_low_rate",
                         "time_s": origin + second * 1.0000000475,
                         "voltage_V": 4.1 - 0.00012 * second + 0.0002 * np.sin(second / 20),
                         "current_A": 0.5 + 0.001 * np.sin(second / 30),
                         "surface_temperature_C": 25 + 0.0003 * second,
                         "source_row": second + 2, "phase": "test_fixture_not_experimental"})
    return rows


class SelectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixtures()
        cls.model, cls.report, cls.decisions = selector.evaluate_rows(cls.rows)

    def test_future_d_values_do_not_change_training_or_thresholds(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row["cell_id"].endswith("_D"):
                row["voltage_V"] += 0.2
                row["current_A"] *= 1.5
                row["surface_temperature_C"] += 2
        model, report, _ = selector.evaluate_rows(changed)
        self.assertEqual(model, self.model)
        self.assertEqual(report["model_sha256"], self.report["model_sha256"])

    def test_input_row_order_does_not_change_results(self):
        changed = copy.deepcopy(self.rows)
        random.Random(440).shuffle(changed)
        model, report, decisions = selector.evaluate_rows(changed)
        self.assertEqual(model, self.model)
        self.assertEqual(report, self.report)
        self.assertEqual(decisions, self.decisions)

    def test_irrelevant_metadata_cannot_change_prediction(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row["source_row"] = -999
            row["phase"] = "future_failure"
            row["risk_label"] = "high"
            row["SOH"] = 0.5
        self.assertEqual(selector.evaluate_rows(changed), (self.model, self.report, self.decisions))

    def test_future_sensor_values_excluded_from_issued_features(self):
        original_session = selector.canonicalize(self.rows)["D"]
        before = selector.make_samples(original_session)
        sample = before[20]
        modified = copy.deepcopy(original_session)
        later = modified["array"][:, 0] > sample["issue_time_s"]
        modified["array"][later, 1:] += 0.123
        after = selector.make_samples(modified)
        updated = next(s for s in after if s["issue_time_s"] == sample["issue_time_s"])
        self.assertEqual(sample["features"], updated["features"])
        self.assertNotEqual(sample["observed_voltage_V"], updated["observed_voltage_V"])
        self.assertEqual(selector.predict_frozen(self.model, [sample["features"]])[0].tolist(),
                         selector.predict_frozen(self.model, [updated["features"]])[0].tolist())

    def test_train_calibration_windows_are_purged(self):
        split = self.report["split"]
        self.assertLess(split["training_last_target_relative_s"],
                        split["calibration_first_history_relative_s"])
        self.assertLessEqual(split["training_last_target_relative_s"], 360)
        self.assertTrue(all(r["issue_relative_s"] >= 370 for r in self.decisions if r["cell_id"].endswith("_C")))

    def test_decisions_carry_original_times_and_observation_cutoff(self):
        for row in self.decisions:
            self.assertLessEqual(row["feature_history_start_time_s"], row["issue_time_s"])
            self.assertLess(row["issue_time_s"], row["target_time_s"])
            self.assertEqual(row["decision_cutoff_s"], row["target_time_s"])
            self.assertGreater(row["issue_time_s"], 17000)
            self.assertGreaterEqual(row["actual_horizon_s"], 10)
            self.assertLessEqual(row["actual_horizon_s"], 11.25)

    def test_frozen_json_predictor_survives_roundtrip(self):
        restored = json.loads(json.dumps(self.model, allow_nan=False))
        features = [[4.08, 0.5, 25.1, -0.00012, 0.0]]
        for left, right in zip(selector.predict_frozen(self.model, features),
                               selector.predict_frozen(restored, features)):
            np.testing.assert_array_equal(left, right)

    def test_missing_temperature_and_duplicate_timestamp_rejected(self):
        invalid = copy.deepcopy(self.rows)
        invalid[20]["surface_temperature_C"] = ""
        with self.assertRaises(ValueError):
            selector.evaluate_rows(invalid)
        invalid = copy.deepcopy(self.rows)
        invalid[20]["time_s"] = invalid[19]["time_s"]
        with self.assertRaises(ValueError):
            selector.evaluate_rows(invalid)

    def test_no_claimed_risk_labels_or_forced_control(self):
        self.assertIsNone(self.report["risk_label"])
        self.assertFalse(self.report["risk_classification_validated"])
        self.assertFalse(self.report["lifetime_prediction_validated"])
        model = copy.deepcopy(self.model)
        model["adopted"] = False
        samples = selector.make_samples(selector.canonicalize(self.rows)["D"])
        decisions, _ = selector.decide(model, samples[:10], "test")
        self.assertTrue(all(not r["selected_by_model"] for r in decisions))
        self.assertTrue(all(r["selection_reason"] == "model_not_adopted" for r in decisions))
        self.assertFalse(any(r["selection_reason"] == "integration_control" for r in decisions))

    def test_csv_entrypoint_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            output = Path(directory) / "out"
            receipt = selector.run_selector(path, output)
            self.assertEqual(receipt["model_sha256"], self.report["model_sha256"])
            with (output / "selection_decisions.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(self.decisions))
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(set(report["artifacts_sha256"]), {"frozen_model.json", "selection_decisions.csv"})
            with self.assertRaises(FileExistsError):
                selector.run_selector(path, output)


if __name__ == "__main__":
    unittest.main()
