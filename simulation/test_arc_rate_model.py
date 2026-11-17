import csv
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from simulation.arc_rate_model import (
    acceptance, fit_rate, metrics, score_trace, validate_pair, write_comparison,
)
from simulation.thermal_abuse_validation import Trace


class ArcRateModelTests(unittest.TestCase):
    def setUp(self):
        self.temperature = np.linspace(150, 190, 41)
        self.rate = np.exp(2 + 15000 * (1 / 443.15 - 1 / (self.temperature + 273.15)))
        self.model = fit_rate(self.temperature, self.rate, (150, 190))
        self.metadata = {
            "kind": "measured", "rate_source": "provided", "cell_model": "fixture",
            "soc_percent": 100, "protocol": "fixture ARC", "temperature_observable": "surface_center",
            "time_origin": "original_test_clock", "trace_complete": False,
            "replicate": "M1", "csv_sha256": "train_fixture",
        }
        self.rules = {
            "minimum_train_samples": 20, "minimum_evaluation_samples": 10,
            "evaluation_median_absolute_relative_error_max": .25,
            "evaluation_p90_absolute_relative_error_max": .5,
        }

    def trace(self, *, rate=None, temperature=None, time=None, replicate="M1"):
        t = self.temperature if temperature is None else np.asarray(temperature)
        r = self.rate if rate is None else np.asarray(rate)
        clock = np.arange(len(t), dtype=float) if time is None else np.asarray(time)
        metadata = {**self.metadata, "replicate": replicate,
                    "csv_sha256": "train_fixture" if replicate == "M1" else "other_fixture"}
        return Trace(clock, t, r, np.repeat("export", len(t)), metadata)

    def test_kelvin_formula_recovers_known_empirical_coefficients(self):
        self.assertAlmostEqual(self.model.intercept, 2, places=12)
        self.assertAlmostEqual(self.model.slope_K, 15000, places=7)
        np.testing.assert_allclose(self.model.predict(self.temperature), self.rate, rtol=1e-13)

    def test_outside_support_is_missing_without_clipping(self):
        output = self.model.predict(np.array([149.999, 150, 190, 190.001, np.nan]))
        self.assertTrue(np.isnan(output[[0, 3, 4]]).all())
        self.assertTrue(np.isfinite(output[[1, 2]]).all())

    def test_fixed_window_prevents_extreme_outside_points_from_changing_fit(self):
        expanded = fit_rate(np.r_[80, self.temperature, 400], np.r_[1e-20, self.rate, 1e20], (150, 190))
        self.assertEqual(asdict(expanded), asdict(self.model))

    def test_degenerate_nonfinite_nonpositive_training_rejected(self):
        cases = [
            (np.repeat(170., 41), self.rate),
            (self.temperature, np.zeros(41)),
            (self.temperature, np.r_[np.nan, self.rate[1:]]),
            (np.r_[-300, self.temperature[1:]], self.rate),
        ]
        for temperature, rate in cases:
            with self.subTest(temperature=temperature[0], rate=rate[0]), self.assertRaises(ValueError):
                fit_rate(temperature, rate, (150, 190))

    def test_log_and_relative_metrics_have_known_values(self):
        result = metrics(np.array([1, 10, 100]), np.array([2, 20, 200]))
        self.assertAlmostEqual(result["log_RMSE_ln"], np.log(2))
        self.assertAlmostEqual(result["exp_log_RMSE_factor"], 2)
        self.assertEqual(result["median_absolute_relative_error"], 1)
        with self.assertRaises(ValueError):
            metrics(np.array([0]), np.array([1]))

    def test_scoring_does_not_refit_or_align_clock(self):
        before = asdict(self.model)
        shifted_clock = 100000 + np.arange(len(self.rate)) * 7
        heldout = self.trace(rate=self.rate * 2, time=shifted_clock, replicate="M2")
        score, _, _ = score_trace(self.model, heldout)
        self.assertEqual(before, asdict(self.model))
        self.assertAlmostEqual(score["model"]["median_absolute_relative_error"], .5)
        self.assertEqual(score["coverage"]["scored_first_original_clock_s"], 100000)
        self.assertIsNone(score["coverage"]["continuous_time_coverage_fraction"])

    def test_heldout_identity_and_observable_requirements(self):
        with self.assertRaises(ValueError):
            validate_pair(self.trace(), self.trace())
        heldout = self.trace(replicate="M2")
        validate_pair(self.trace(), heldout)
        heldout.metadata["temperature_observable"] = "core"
        with self.assertRaises(ValueError):
            validate_pair(self.trace(), heldout)

    def test_failed_error_gate_kept_and_no_fire_or_runaway_validation(self):
        result, _, _ = score_trace(self.model, self.trace(rate=self.rate * 10, replicate="M2"))
        gate = acceptance(self.model, result, self.rules)
        self.assertFalse(gate["bounded_rate_fit_passed"])
        self.assertFalse(gate["checks"]["median_relative_error"])
        self.assertFalse(gate["fire_prediction_validated"])
        self.assertFalse(gate["thermal_runaway_prediction_validated"])
        good, _, _ = score_trace(self.model, self.trace(replicate="M2"))
        self.assertTrue(acceptance(self.model, good, self.rules)["bounded_rate_fit_passed"])

    def test_comparison_retains_all_raw_rows_and_unavailable_labels(self):
        trace = self.trace(temperature=np.r_[80, self.temperature, 450],
                           rate=np.r_[.02, self.rate, 50000])
        _, prediction, mask = score_trace(self.model, trace)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.csv"
            write_comparison(path, "fixture", "heldout", trace, self.model, prediction, mask)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 43)
        self.assertEqual(float(rows[-1]["rate_C_per_min"]), 50000)
        self.assertEqual(rows[-1]["predicted_rate_C_per_min"], "")
        self.assertEqual(rows[-1]["thermal_runaway_label"], "")
        self.assertEqual(rows[-1]["fire_label"], "")
        self.assertEqual(rows[-1]["exclusion_reason"], "outside_M1_temperature_support")


if __name__ == "__main__":
    unittest.main()
