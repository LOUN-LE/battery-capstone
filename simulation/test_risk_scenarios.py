"""Regression checks for labels, failure recovery and output identity."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simulation.risk_scenarios import pilot_scenarios, persistent_onset, run_case, run_dataset


class ScenarioContracts(unittest.TestCase):
    def test_persistence_ignores_disconnected_excursions(self):
        self.assertIsNone(persistent_onset(range(9), [-1, -1, 0, -1, -1, 0, -1, -1, 0], .1, 3))
        self.assertEqual(persistent_onset(range(9), [0, 0, -1, -1, -1, -1, 0, 0, 0], .1, 3), 2)

    def test_invalid_inputs_are_rejected(self):
        case = pilot_scenarios()[0]
        for change in ({"initial_soc": float("nan")}, {"rest_s": 0}, {"scenario_id": "../escape"},
                       {"initial_temperature_C": 60}, {"charge_current_A": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(case, **change).validate()

    def test_failed_case_has_no_negative_label_and_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = (asdict(pilot_scenarios()[0]), directory, {"test": 1}, False)
            with patch("simulation.risk_scenarios.simulate", side_effect=RuntimeError("solver failed")) as solve:
                first = run_case(payload)
                second = run_case(payload)
                self.assertEqual(solve.call_count, 1)
            self.assertEqual(first, second)
            self.assertIsNone(first["plating_event"])
            self.assertIsNone(first["fire_label"])
            self.assertEqual(first["status"], "failed")
            self.assertFalse((Path(directory) / first["scenario_id"] / "running.lock").exists())
            with self.assertRaises(ValueError):
                run_case((payload[0], directory, {"test": 2}, False))

    def test_success_reuse_checks_trace_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            case = pilot_scenarios()[0]
            payload = (asdict(case), directory, {"test": 1}, False)
            simulated = ({"scenario_id": case.scenario_id, "status": "completed"}, [{"value": 1}], {})
            with patch("simulation.risk_scenarios.simulate", return_value=simulated) as solve:
                run_case(payload)
                run_case(payload)
                self.assertEqual(solve.call_count, 1)
                (Path(directory) / case.scenario_id / "trace.csv").write_text("corrupt", encoding="utf-8")
                run_case(payload)
                self.assertEqual(solve.call_count, 2)

    def test_concurrent_owner_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            case = pilot_scenarios()[0]
            folder = Path(directory) / case.scenario_id
            folder.mkdir()
            (folder / "running.lock").write_text("active", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_case((asdict(case), directory, {}, False))

    def test_design_ids_and_worker_limit(self):
        cases = pilot_scenarios()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({s.scenario_id for s in cases}), 32)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_dataset(cases, directory, workers=9)
            with self.assertRaises(ValueError):
                run_dataset([cases[0], cases[0]], directory)

    def test_source_mutation_prevents_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            case = pilot_scenarios()[0]
            with patch("simulation.risk_scenarios.provenance", side_effect=[{"source": "old"}, {"source": "new"}]), patch(
                "simulation.risk_scenarios.run_case", return_value={"scenario_id": case.scenario_id, "status": "completed"}
            ), self.assertRaisesRegex(RuntimeError, "changed during"):
                run_dataset([case], directory, workers=1)
            self.assertFalse((Path(directory) / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
