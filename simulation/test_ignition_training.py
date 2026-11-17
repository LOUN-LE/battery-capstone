import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from simulation import ignition_training as training


FIELDS = ["experiment_id", "soc_percent", "heating_policy", "reported_intervention_delay_s",
          "ignition_observed", "label_evidence", "maximum_temperature_C", "ignition_time_s"]


class IgnitionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def row(self, identifier="exp1", soc=50, policy="interrupted", delay=10,
            label="1", evidence="synthetic software fixture"):
        return dict(zip(FIELDS, [identifier, str(soc), policy, str(delay), label, evidence,
                                 "DO_NOT_PARSE_OUTCOME_FEATURE", "DO_NOT_PARSE_EVENT_TIME"]))

    def write(self, rows, name="fixture.csv"):
        path = self.root / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_unknown_outcome_excluded_and_retained_without_negative_label(self):
        unknown = self.row("unknown", soc="", policy="unknown", delay="", label="", evidence="")
        included, excluded, x, y, groups = training.read_examples(self.write([self.row(), unknown]))
        self.assertEqual(len(included), 1)
        self.assertEqual(excluded, [unknown])
        np.testing.assert_array_equal(y, [1])
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(len(groups), 1)

    def test_only_explicit_scenario_columns_become_features(self):
        rows = [self.row(), self.row("continuous", policy="continuous", delay="", label="0")]
        _, _, x, y, groups = training.read_examples(self.write(rows))
        self.assertEqual(training.FEATURES, ["soc_percent", "continuous_heating", "reported_intervention_delay_s"])
        np.testing.assert_array_equal(x, [[50., 0., 10.], [50., 1., 0.]])
        np.testing.assert_array_equal(y, [1, 0])
        # Structural zero for continuous heating is distinct from an immediate stop.
        _, _, stopped, _, stopped_group = training.read_examples(self.write([self.row("immediate", delay=0)], "stop.csv"))
        self.assertEqual(float(stopped[0, 1]), 0.)
        self.assertNotEqual(groups[1], stopped_group[0])

    def test_invalid_observed_labels_or_controls_fail_closed(self):
        invalid = [self.row(label="True"), self.row(label="0", evidence=""),
                   self.row(soc="nan"), self.row(soc=35),
                   self.row(policy="continuous", delay=0), self.row(delay=-1),
                   self.row(delay="nan"), self.row(policy="unknown")]
        for index, row in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                training.read_examples(self.write([row]))

    def test_duplicate_experiment_even_if_unknown_cannot_be_counted_twice(self):
        with self.assertRaises(ValueError):
            training.read_examples(self.write([self.row(), self.row(label="")]))

    def test_same_protocol_repeats_never_cross_group_split(self):
        rows = []
        for group in range(12):
            for outcome in (0, 1):
                rows.append(self.row(f"group{group}_repeat{outcome}", delay=group, label=str(outcome)))
        _, _, x, y, groups = training.read_examples(self.write(rows))
        train, test = training.split_examples(x, y, groups)
        self.assertEqual(set(train) | set(test), set(range(len(rows))))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(groups[train]) & set(groups[test]))
        for group in set(groups):
            indices = set(np.flatnonzero(groups == group))
            self.assertTrue(indices <= set(train) or indices <= set(test))

    def test_unfavourable_declared_first_fold_is_not_replaced(self):
        x = np.zeros((6, 3))
        y = np.array([0, 0, 0, 1, 1, 1])
        groups = np.arange(6)
        with patch.object(training, "StratifiedGroupKFold") as splitter:
            splitter.return_value.split.return_value = iter([
                (np.array([0, 1, 2]), np.array([3, 4, 5])),
                (np.array([0, 3, 4]), np.array([1, 2, 5])),
            ])
            with self.assertRaisesRegex(ValueError, "do not search folds"):
                training.split_examples(x, y, groups)

    def test_metrics_report_denominators_and_false_negative(self):
        result = training.score(np.array([0, 0, 1, 1]), np.array([.1, .7, .4, .9]))
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["positives"], 2)
        self.assertEqual(result["confusion_matrix_true_rows_0_1_predicted_columns_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["brier"], (.01 + .49 + .36 + .01) / 4)

    def test_partial_run_directory_is_not_overwritten(self):
        source = self.write([self.row()])
        output = self.root / "partial"
        output.mkdir()
        original = '{"frozen": true}\n'
        (output / "experiment_plan.json").write_text(original, encoding="utf-8")
        with self.assertRaises(FileExistsError):
            training.run(source, output)
        self.assertEqual((output / "experiment_plan.json").read_text(encoding="utf-8"), original)

    def test_changed_input_during_run_prevents_success_report(self):
        rows = [self.row(f"sample{i}", delay=i, label=str(i % 2)) for i in range(12)]
        source = self.write(rows)
        original_network = training.network
        calls = 0

        def changing_source_network():
            nonlocal calls
            calls += 1
            if calls == 1:
                source.write_bytes(source.read_bytes() + b"\n")
            return original_network()

        with patch.object(training, "split_examples", return_value=(np.arange(8), np.arange(8, 12))):
            with patch.object(training, "network", side_effect=changing_source_network):
                with self.assertRaisesRegex(RuntimeError, "changed during the run"):
                    training.run(source, self.root / "changed")
        self.assertFalse((self.root / "changed/report.json").exists())
        self.assertNotEqual((self.root / "changed/training_input_snapshot.csv").read_bytes(), source.read_bytes())

    def adapter_fixture(self):
        rows = [{"experiment_id": str(i), "dataset_id": "NIST_mds2-4027", "cell_format": "21700",
                 "cathode": "NMC", "soc_pct": "50", "heating_mode": "continuous",
                 "intervention_after_svb_s": "", "ignition_observed": "true",
                 "outcome_source": "synthetic fixture paper", "outcome_locator": "fixture figure",
                 "outcome_annotation_method": "synthetic software fixture", "outcome_note": ""}
                for i in range(49, 83)]
        path = self.root / "adapter_fixture.csv"
        return rows, path

    def test_adapter_keeps_unknown_label_and_source_chemistry_separate(self):
        rows, path = self.adapter_fixture()
        rows[-1]["ignition_observed"] = ""
        training.write_csv(path, rows, list(rows[0]))
        adapted = training.adapt_nist(path, self.root / "adapted.csv")
        included, excluded, _, y, _ = training.read_examples(adapted)
        self.assertEqual(len(included), 33)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["ignition_observed"], "")
        self.assertTrue(np.all(y == 1))
        rows[0]["cathode"] = "NCA"
        training.write_csv(path, rows, list(rows[0]))
        with self.assertRaises(ValueError):
            training.adapt_nist(path, self.root / "rejected.csv")

    def test_adapter_does_not_create_nonempty_evidence_from_empty_fields(self):
        rows, path = self.adapter_fixture()
        for key in ("outcome_source", "outcome_locator", "outcome_annotation_method"):
            rows[0][key] = ""
        training.write_csv(path, rows, list(rows[0]))
        with self.assertRaises(ValueError):
            training.adapt_nist(path, self.root / "missing_evidence.csv")

    def test_train_only_scaling_checkpoint_restore_and_test_label_isolation(self):
        # Independent test feature range makes accidental full-data scaling visible.
        rows = [self.row(f"train{i}", soc=25 if i < 6 else 50, delay=i, label=str(i % 2))
                for i in range(12)]
        rows += [self.row(f"test{i}", soc=100, delay=40+i, label=str(i % 2)) for i in range(4)]
        rows += [self.row("unknown", label="", evidence="")]
        source = self.write(rows)
        train, test = np.arange(12), np.arange(12, 16)
        with patch.object(training, "split_examples", return_value=(train, test)):
            first = training.run(source, self.root / "first")
        loaded = torch.load(self.root / "first/ignition_mlp.pt", map_location="cpu", weights_only=True)
        _, _, x, _, _ = training.read_examples(source)
        np.testing.assert_allclose(loaded["scaler_mean"], x[train].mean(axis=0), rtol=1e-7)
        self.assertNotEqual(loaded["scaler_mean"][0], float(x.mean(axis=0)[0]))
        self.assertEqual(set(loaded["train_experiments"]), {f"train{i}" for i in range(12)})
        self.assertEqual(set(loaded["test_experiments"]), {f"test{i}" for i in range(4)})
        self.assertEqual(first["excluded_unknown_experiments"], 1)
        self.assertFalse(first["deployment_validated"])
        self.assertLessEqual(first["integrity"]["checkpoint_max_probability_difference"], 1e-6)
        with (self.root / "first/excluded_experiments.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle))[0]["ignition_observed"], "")
        with self.assertRaises(FileExistsError):
            training.run(source, self.root / "first")
        for i in test:
            rows[int(i)]["ignition_observed"] = str(1 - int(rows[int(i)]["ignition_observed"]))
        changed_test_labels = self.write(rows, "changed_test_labels.csv")
        with patch.object(training, "split_examples", return_value=(train, test)):
            training.run(changed_test_labels, self.root / "second")
        second = torch.load(self.root / "second/ignition_mlp.pt", map_location="cpu", weights_only=True)
        for key in loaded["state_dict"]:
            self.assertTrue(torch.equal(loaded["state_dict"][key], second["state_dict"][key]), key)
        with (self.root / "first/predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            a = list(csv.DictReader(handle))
        with (self.root / "second/predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            b = list(csv.DictReader(handle))
        for first_row, second_row in zip(a, b):
            for name in ("mlp_probability", "logistic_probability", "constant_probability"):
                self.assertEqual(first_row[name], second_row[name], name)


if __name__ == "__main__":
    unittest.main()
