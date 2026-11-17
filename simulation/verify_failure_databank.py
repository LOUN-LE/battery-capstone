"""Three actual-source integrity tests; run with --source FILE.xlsx."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest

from .failure_databank import build_audit, EXPECTED_SHA256


class DatabankIntegrity(unittest.TestCase):
    source = None

    @classmethod
    def setUpClass(cls):
        cls.before = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.records, cls.audit, *_ = build_audit(cls.source)

    def test_pinned_source_identity_and_tamper_rejection(self):
        self.assertEqual(self.before, EXPECTED_SHA256)
        self.assertEqual(self.audit["all_test_records"], 365)
        self.assertEqual(len(self.records), 25)
        self.assertEqual(self.audit["trigger_counts"], {"Heater (Non-ISC)": 8, "Nail": 17})
        self.assertEqual(len({r["values"]["Test-ID"] for r in self.records}), 25)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.xlsx"
            changed.write_bytes(self.source.read_bytes() + b"revision changed")
            with self.assertRaisesRegex(ValueError, "Source revision changed"):
                build_audit(changed)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.before)

    def test_independent_energy_and_mass_closure(self):
        for record in self.records:
            values = record["values"]
            self.assertAlmostEqual(sum(values[k] for k in
                ("Energy-Fraction-Cell-Body-kJ", "Energy-Fraction-Positive-Ejecta-kJ", "Energy-Fraction-Negative-Ejecta-kJ")),
                values["Corrected-Total-Energy-Yield-kJ"], places=8)
            remainder = values["Pre-Test-Cell-Mass-g"] - sum(values[k] for k in
                ("Post-Test-Mass-Cell-Body-g", "Post-Test-Mass-Positive-Ejecta-Mating-g",
                 "Post-Test-Mass-Positive-Ejecta-Bore-Baffles-g", "Post-Test-Mass-Positive-Copper-Mesh-g",
                 "Post-Test-Mass-Negative-Ejecta-Mating-g", "Post-Test-Mass-Negative-Ejecta-Bore-Baffles-g",
                 "Post-Test-Mass-Negative-Copper-Mesh-g"))
            self.assertAlmostEqual(remainder, values["Post-Test-Mass-Unrecovered-g"], places=8)

    def test_missing_zero_and_unknown_labels_stay_distinct(self):
        records = {r["values"]["Test-ID"]: r for r in self.records}
        no_ejection = [r for r in self.records if r["values"]["Cell-Failure-Mechanism"] == "No Ejection"]
        self.assertEqual(len(no_ejection), 3)
        for r in self.records:
            for key in ("soc_fraction", "age_cycles", "soh_fraction", "thermal_runaway_label", "fire_label"):
                self.assertIsNone(r[key])
            for key, raw in r["raw"].items():
                if raw == "-":
                    self.assertIsNone(r["values"][key])
                if raw == "0":
                    self.assertEqual(r["values"][key], 0)
        self.assertLess(records["SPR2021_ESTA_8B100-01_SOC_RUN203"]["values"]["Post-Test-Mass-Unrecovered-g"], 0)
        self.assertEqual(records["ESRF18_Oct_Run041"]["test_date_iso"], "2018-10-27")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    DatabankIntegrity.source = args.source
    unittest.main(argv=[__file__], verbosity=2)
