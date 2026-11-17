"""Targeted input/state initialization checks; real replay checks live in reports."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

import twin_runner as twin


class InputAndStateTests(unittest.TestCase):
    def test_transition_row_is_not_a_rest_anchor(self):
        rows = [dict(phase="preparation_step_0_rest", time_s=str(t), current_A="0",
                     voltage_V=str(v), source_row=str(index + 2))
                for index, (t, v) in enumerate(((80, 3.62), (90, 3.6201),
                                               (100, 3.6202), (110, 3.6203), (120.044, 3.67)))]
        anchor, details = twin.rest_anchor(rows)
        self.assertAlmostEqual(anchor, 3.6202)
        self.assertEqual(details["source_times_s"], [90., 100., 110.])

    def test_split_waveform_preserves_piecewise_linear_charge(self):
        table = np.array([[0., .4], [1.7, .8], [3.2, .1], [9., .5]])
        a, b = twin.slice_table(table, 0., 2.1), twin.slice_table(table, 2.1, 9.)
        self.assertAlmostEqual(float(np.trapezoid(table[:, 1], table[:, 0])),
                               float(np.trapezoid(a[:, 1], a[:, 0]) + np.trapezoid(b[:, 1], b[:, 0])), places=12)
        self.assertEqual(a[-1, 1], b[0, 1])
        with self.assertRaises(ValueError):
            twin.slice_table(table, -1., 2.)

    def test_input_rejects_cross_cell_and_nonincreasing_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            first = dict(zip(twin.FIELDS, ["C", "rpt0", "1", "4.1", ".5", "25", "2", "C10"]))
            for change in ({"cell_id": "D", "time_s": "2"}, {"time_s": "1"}):
                second = {**first, "source_row": "3", **change}
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=twin.FIELDS)
                    writer.writeheader()
                    writer.writerows([first, second])
                with self.assertRaises(ValueError):
                    twin.read_input(path)

    def test_cell_specific_voltage_does_not_change_frozen_inventory(self):
        config = twin.read_json(twin.HERE / "configs/frozen_model.json")
        calibration = twin.read_json(twin.HERE / "configs/cell_calibration.json")
        _, c = twin.make_parameters(config, calibration, 3.620251866666667)
        _, d = twin.make_parameters(config, calibration, 3.616711433333333)
        self.assertAlmostEqual(c["particle_lithium_Ah"], d["particle_lithium_Ah"], places=12)
        self.assertLess(abs(d["inventory_residual_Ah"]), 1e-10)
        self.assertNotEqual(c["negative_stoichiometry"], d["negative_stoichiometry"])
        self.assertFalse(c["initial_soc_remapping_used"])
        for name, expected in calibration["rpt0_rest_initialization"]["overrides"].items():
            self.assertAlmostEqual(c["initial_concentrations"][name], expected, places=5)


if __name__ == "__main__":
    unittest.main()
