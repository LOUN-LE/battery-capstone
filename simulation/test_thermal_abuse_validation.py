import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulation.thermal_abuse_validation import (
    Criterion, Trace, compare_traces, first_persistent_crossing, import_ohneseit_arc,
    read_trace, summarize,
)


class ThermalAbuseValidationTests(unittest.TestCase):
    def setUp(self):
        self.criterion = Criterion(50, 1, 2, 2, "Synthetic software regression criterion")
        self.meta = {"kind": "measured", "source_url": "synthetic fixture",
                     "cell_model": "fixture", "protocol": "fixture",
                     "temperature_observable": "surface_center",
                     "time_origin": "test trigger", "trace_complete": True}

    def trace(self, time=(0, 1, 2, 3), temperature=(50, 51, 52, 53), rate=(60, 60, 60, 60)):
        return Trace(np.array(time, dtype=float), np.array(temperature, dtype=float),
                     np.array(rate, dtype=float), np.repeat("0", len(time)), dict(self.meta))

    def prediction(self):
        result = self.trace()
        result.metadata.update(kind="simulation", used_for_fit=False,
                               model_id="fixture", parameters_sha256="frozen fixture")
        return result

    def test_confirmation_requires_elapsed_duration(self):
        result = first_persistent_crossing(self.trace(), self.criterion)
        self.assertEqual(result["onset_s"], 0)
        self.assertEqual(result["confirmed_at_s"], 2)

    def test_gap_and_segment_boundary_break_persistence(self):
        gap = self.trace(time=(0, 1, 100, 101))
        self.assertIsNone(first_persistent_crossing(gap, self.criterion)["onset_s"])
        changed = self.trace()
        changed.segment = np.array(["0", "0", "1", "1"])
        self.assertIsNone(first_persistent_crossing(changed, self.criterion)["onset_s"])

    def test_missing_event_is_censored_for_incomplete_trace(self):
        trace = self.trace(temperature=(20, 21, 22, 23))
        trace.metadata["trace_complete"] = False
        result = summarize(trace, self.criterion)
        self.assertEqual(result["criterion_crossing"]["status"], "censored")
        self.assertIsNone(result["observed_events"]["thermal_runaway"]["observed"])
        self.assertFalse(result["fire_prediction_validated"])

    def test_high_temperature_never_fabricates_fire_or_runaway(self):
        result = summarize(self.trace(temperature=(400, 450, 500, 550)), self.criterion)
        self.assertIsNone(result["observed_events"]["fire"]["observed"])
        self.assertIsNone(result["observed_events"]["thermal_runaway"]["observed"])

    def test_observed_labels_require_measurement_evidence(self):
        trace = self.trace()
        trace.metadata["observed_events"] = {"fire": {"observed": True}}
        with self.assertRaises(ValueError):
            summarize(trace, self.criterion)

    def test_comparison_requires_matching_observable_and_heldout_source(self):
        predicted = self.prediction()
        predicted.metadata["temperature_observable"] = "volume_average"
        with self.assertRaises(ValueError):
            compare_traces(self.trace(), predicted, self.criterion)
        predicted = self.prediction()
        predicted.metadata["used_for_fit"] = True
        with self.assertRaises(ValueError):
            compare_traces(self.trace(), predicted, self.criterion)

    def test_comparison_scores_shift_and_reports_missing_coverage(self):
        predicted = self.prediction()
        predicted.temperature_C += 2
        result = compare_traces(self.trace(), predicted, self.criterion)
        self.assertEqual(result["temperature_RMSE_C"], 2)
        measured = self.trace(time=(-1, 1, 2, 4))
        result = compare_traces(measured, predicted, self.criterion)
        self.assertEqual(result["coverage_fraction"], 0.5)
        self.assertEqual(result["comparison_status"], "partial_coverage")

    def test_no_interpolation_over_prediction_gap(self):
        predicted = self.prediction()
        predicted.time_s = np.array([0, 1, 100, 101])
        result = compare_traces(self.trace(), predicted, self.criterion)
        self.assertEqual(result["covered_samples"], 2)

    def test_csv_units_and_no_rate_across_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata = path / "meta.json"
            metadata.write_text(json.dumps(self.meta))
            source = path / "trace.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["time_s", "temperature_C"], [0, 20], [1, 21], [10, 40]])
            trace = read_trace(source, metadata, 2)
            self.assertTrue(np.isnan(trace.rate_C_per_min[0]))
            self.assertEqual(trace.rate_C_per_min[1], 60)
            self.assertTrue(np.isnan(trace.rate_C_per_min[2]))
            self.assertEqual(len(trace.metadata["csv_sha256"]), 64)

    def test_arc_conversion_preserves_rate_and_converts_only_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            raw = path / "NMC_SOC100_M1.txt"
            payload = ("time / min\ttemperature / °C\ttemperature rate / ( °C/min)\n"
                       "2\t90\t0.02\n3\t91\t0.03\n").encode("cp1252")
            raw.write_bytes(payload)
            output, metadata = path / "trace.csv", path / "meta.json"
            with self.assertRaises(ValueError):
                import_ohneseit_arc(raw, output, metadata, "invalid")
            import_ohneseit_arc(raw, output, metadata, hashlib.md5(payload).hexdigest())
            trace = read_trace(output, metadata, 60)
            np.testing.assert_array_equal(trace.time_s, [120, 180])
            np.testing.assert_allclose(trace.rate_C_per_min, [0.02, 0.03])
            self.assertFalse(trace.metadata["trace_complete"])
            self.assertEqual(trace.metadata["soc_percent"], 100)


if __name__ == "__main__":
    unittest.main()
