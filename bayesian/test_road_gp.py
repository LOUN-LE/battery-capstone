import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import unittest

import numpy as np

from .road_gp import GRID_S, RoadWindow, evaluate, forecast_window, parse_member


def fixture_windows(count=20, vehicle="BEV1"):
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [RoadWindow(f"{vehicle}-{i}", vehicle, origin+timedelta(days=i),
                       origin+timedelta(days=i, seconds=600),
                       np.sin(GRID_S/30)*(i+1)+i, "test", 601)
            for i in range(count)]


def raw_csv(times=None, current=None):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["Timestamp", "RawBattCurrent132"])
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = range(301) if times is None else times
    for i in times:
        writer.writerow([(origin+timedelta(seconds=i)).isoformat(),
                         current(i) if current else (-10 if i % 2 else 20)])
    return buffer.getvalue().encode()


class RoadGPTests(unittest.TestCase):
    member = "dataset/driving sessions/BEV1_2024-01-01_drive1.csv"

    def test_raw_pack_units_and_sign_preserved_on_exact_grid(self):
        window, _ = parse_member(self.member, raw_csv())
        np.testing.assert_array_equal(window.current_a, [20 if t % 2 == 0 else -10 for t in GRID_S])
        self.assertEqual(window.vehicle, "BEV1")
        self.assertEqual(window.source_rows, 301)
        self.assertEqual(window.completed-window.started, timedelta(seconds=300))

    def test_gaps_nonfinite_short_and_duplicate_timestamps_rejected(self):
        cases = [raw_csv(times=[i for i in range(301) if i != 13]),
                 raw_csv(times=range(300)), raw_csv(times=[0, 0, *range(1, 301)]),
                 raw_csv(current=lambda i: "nan" if i == 17 else 10),
                 raw_csv(current=lambda i: "" if i == 17 else 10)]
        for raw in cases:
            with self.subTest(raw=raw[:90]), self.assertRaises(ValueError):
                parse_member(self.member, raw)

    def test_future_current_mutation_cannot_change_earlier_forecast(self):
        windows = fixture_windows()
        target = windows[15]
        original = forecast_window(windows, target)
        changed = [replace(w, current_a=np.full_like(w.current_a, 1000000)) if w.started >= target.started else w
                   for w in windows]
        after = forecast_window(changed, target)
        for key in ("gp", "recent_mean", "all_past_mean", "last_trip", "interval_lower", "interval_upper"):
            np.testing.assert_array_equal(original[key], after[key])
        self.assertEqual(original["training_members"], after["training_members"])

    def test_in_progress_history_and_other_vehicle_are_excluded(self):
        windows = fixture_windows()
        target = windows[15]
        overlapping = replace(windows[14], completed=target.started+timedelta(seconds=1))
        mixed = [overlapping if w.member == overlapping.member else w for w in windows]
        mixed += fixture_windows(vehicle="BEV2")
        prediction = forecast_window(mixed, target)
        self.assertNotIn(overlapping.member, prediction["training_members"])
        self.assertTrue(all(name.startswith("BEV1") for name in prediction["training_members"]))
        self.assertEqual(len(prediction["recent_members"]), 5)

    def test_chronological_last_twenty_percent_and_per_vehicle_metrics(self):
        windows = fixture_windows()+fixture_windows(vehicle="BEV2")
        report = evaluate(windows)
        for vehicle, result in report.items():
            self.assertEqual(result["evaluated_test_sessions"], 4)
            self.assertEqual(result["folds"][0]["member"], vehicle+"-16")
            self.assertEqual(result["grid_points_per_session"], 61)
            self.assertEqual(set(result["metrics"]), {"gp", "recent_mean", "all_past_mean", "last_trip"})
            for fold in result["folds"]:
                self.assertNotIn(fold["member"], fold["training_members"])

    def test_insufficient_duplicate_or_nonfinite_histories_rejected(self):
        windows = fixture_windows()
        with self.assertRaises(ValueError):
            forecast_window(windows[:6], windows[10])
        with self.assertRaises(ValueError):
            forecast_window(windows+[windows[0]], windows[10])
        dirty = [replace(w, current_a=np.full_like(w.current_a, np.nan)) if i == 3 else w for i,w in enumerate(windows)]
        with self.assertRaises(ValueError):
            forecast_window(dirty, windows[10])


if __name__ == "__main__":
    unittest.main()
