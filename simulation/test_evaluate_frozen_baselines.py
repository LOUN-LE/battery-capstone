import json
from pathlib import Path
import tempfile
import unittest

from evaluate_frozen_baselines import load_frozen, metadata, verify_heldout_sources
from fit_empirical_baselines import FrozenBaseline, digest


class FrozenEvaluationTests(unittest.TestCase):
    def test_other_temperature_changes_metadata_not_model_coefficients(self):
        base = {'condition_id': 'Kirkaldy2024_Expt2_25C_SOC70_85', 'experiment_id': 'Kirkaldy2024_Expt2', 'temperature_setpoint_C': 25,
                'SOC_percent': [70, 85], 'source_refs': ['verified source']}
        cold = {**base, 'temperature_setpoint_C': 10, 'condition_id': 'Kirkaldy2024_Expt2_10C_SOC70_85'}
        self.assertNotEqual(metadata(base)['temperature_protocol_id'], metadata(cold)['temperature_protocol_id'])
        self.assertEqual(metadata(base)['axis_definition'], metadata(cold)['axis_definition'])

    def test_missing_or_unreviewed_condition_fails(self):
        with self.assertRaises(ValueError):
            metadata({})
        with self.assertRaises(ValueError):
            metadata({'condition_id': 'test', 'experiment_id': 'Expt2', 'temperature_setpoint_C': 25,
                      'SOC_percent': [0, 100], 'source_refs': ['source']})

    def test_condition_label_cannot_spoof_heldout_temperature_or_experiment(self):
        base = {'condition_id': 'Kirkaldy2024_Expt2_25C_SOC70_85', 'experiment_id': 'Kirkaldy2024_Expt2',
                'temperature_setpoint_C': 25, 'SOC_percent': [70, 85], 'source_refs': ['source']}
        for change in ({'temperature_setpoint_C': 10}, {'experiment_id': 'Kirkaldy2024_Expt5'}):
            with self.assertRaisesRegex(ValueError, 'Experiment and condition identity'):
                metadata({**base, **change})

    def test_changed_predictor_code_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_text(json.dumps({'predictor_dependency': {'file': 'fit_empirical_baselines.py', 'sha256': 'wrong'}}))
            with self.assertRaisesRegex(ValueError, 'predictor code SHA-256'):
                load_frozen(path)

    def test_frozen_schema_does_not_accept_another_training_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_text(json.dumps({'model': {'version': 'empirical_C_only_v1', 'fit_cell': 'D',
                'fit_source_sha256': 'test', 'sqrt_calendar_coefficient_pp': 1, 'linear_efc_coefficient_pp': 1}}))
            with self.assertRaises(ValueError):
                load_frozen(path)

    def test_calibration_source_copy_cannot_be_renamed_as_heldout_D(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fake_cellD.csv'
            path.write_bytes(b'actual calibration C source bytes')
            model = FrozenBaseline('empirical_C_only_v1', 'C', digest(path), 1, 1)
            with self.assertRaisesRegex(ValueError, 'Calibration source bytes'):
                verify_heldout_sources([{'cell_id': 'D', 'summary_csv': str(path)}], model)

    def test_duplicate_raw_aliases_cannot_be_independent_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d = root/'D.csv'
            e = root/'E.csv'
            d.write_bytes(b'one true D raw source')
            e.write_bytes(d.read_bytes())
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps({'archive_url': 'https://zenodo.org/records/10637534/files/Expt%202%2C2%20-%20C-based%20Degradation%202.zip?download=1',
                'files': [{'sha256': digest(d), 'crc32_verified': True,
                           'path': 'Expt 2,2 - C-based Degradation 2/Summary Data/Performance Summary/Expt 2,2 - cell D (25degC) - Processed Data.csv'}]}))
            entries = [{'cell_id': c, 'summary_csv': str(p), 'source_receipt': str(receipt),
                        'condition': {'temperature_setpoint_C': 25}} for c, p in [('D', d), ('E', e)]]
            model = FrozenBaseline('empirical_C_only_v1', 'C', 'different training bytes', 1, 1)
            with self.assertRaisesRegex(ValueError, 'Duplicate raw source'):
                verify_heldout_sources(entries, model)

    def test_declared_temperature_cannot_override_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d = root/'D.csv'
            d.write_bytes(b'one true D raw source')
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps({'archive_url': 'https://zenodo.org/records/10637534/files/Expt%202%2C2%20-%20C-based%20Degradation%202.zip?download=1',
                'files': [{'sha256': digest(d), 'crc32_verified': True,
                           'path': 'Expt 2,2 - C-based Degradation 2/Summary Data/Performance Summary/Expt 2,2 - cell D (25degC) - Processed Data.csv'}]}))
            entry = {'cell_id': 'D', 'summary_csv': str(d), 'source_receipt': str(receipt),
                     'condition': {'temperature_setpoint_C': 40}}
            model = FrozenBaseline('empirical_C_only_v1', 'C', 'different training bytes', 1, 1)
            with self.assertRaisesRegex(ValueError, 'authoritative source member identity'):
                verify_heldout_sources([entry], model)


if __name__ == '__main__':
    unittest.main()
