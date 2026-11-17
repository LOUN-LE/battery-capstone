"""Adversarial native target provenance checks; no measured fit or PDE calls."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from physical_campaign_sources import verify_native_targets
from physical_campaign_inputs import validate_split


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT/'work/external_m50_benchmark'
TEMP_ROOT = ROOT/'work/validation_proposal_20261103'


def ref(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(cell='C'):
    path = BENCHMARK/f'kirkaldy_expt2_cell{cell}_performance_summary.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        native = list(csv.DictReader(handle))
    rows = [{'rpt_id': f"RPT{int(row['Ageing Sets'])}", 'Qref_Ah': float(row['C/10 Capacity [mA h]'])/1000}
            for row in native[:2]]
    campaign = {'data_kind': 'measured', 'cell_id': cell,
        'condition_id': 'Kirkaldy2024_Expt2_25C_SOC70_85',
        'source_identity': {'kind': 'Kirkaldy2024_Expt2_performance_summary',
            'summary': ref(path), 'receipt': ref(BENCHMARK/'kirkaldy_minimal_extract_manifest.json'),
            'temperature_setpoint_C': 25, 'SOC_percent': [70, 85]}}
    return campaign, rows


class NativeCampaignSourceTests(unittest.TestCase):
    def test_actual_native_C_and_D_target_capacities_bind_separately(self):
        identities = []
        for cell in ('C', 'D'):
            campaign, rows = fixture(cell)
            result = verify_native_targets(campaign, ROOT, rows)
            self.assertEqual(result['verified_cell_id'], cell)
            self.assertEqual(result['verified_temperature_setpoint_C'], 25)
            self.assertEqual(result['source_identity_sha256'], campaign['source_identity']['summary']['sha256'])
            identities.append(result['source_member_path'])
        self.assertNotEqual(*identities)

    def test_C_summary_relabelled_as_D_cannot_pass(self):
        campaign, rows = fixture('C')
        campaign['cell_id'] = 'D'
        with self.assertRaisesRegex(ValueError, 'cell/temperature'):
            verify_native_targets(campaign, ROOT, rows)

    def test_changed_target_Ah_or_RPT_identity_is_rejected(self):
        original, targets = fixture('C')
        for altered in ({'Qref_Ah': targets[1]['Qref_Ah']*1000}, {'rpt_id': 'RPT999'}):
            rows = copy.deepcopy(targets); rows[1].update(altered)
            with self.assertRaisesRegex(ValueError, 'Target capacity/RPT'):
                verify_native_targets(original, ROOT, rows)

    def test_modified_summary_relocked_hash_still_does_not_match_official_receipt(self):
        campaign, rows = fixture('C')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            source = Path(campaign['source_identity']['summary']['path'])
            changed = Path(folder)/'modified_summary.csv'
            changed.write_bytes(source.read_bytes()+b'\n')  # numerically same, different underlying raw bytes
            campaign['source_identity']['summary'] = ref(changed)
            with self.assertRaisesRegex(ValueError, 'download record'):
                verify_native_targets(campaign, ROOT, rows)

    def test_source_archive_identity_cannot_be_silently_changed(self):
        campaign, rows = fixture('C')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            original = Path(campaign['source_identity']['receipt']['path'])
            receipt = json.loads(original.read_text(encoding='utf-8'))
            receipt['archive_url'] = 'https://zenodo.org/records/999/files/unreviewed.zip'
            fake = Path(folder)/'changed_receipt.json'; fake.write_text(json.dumps(receipt))
            campaign['source_identity']['receipt'] = ref(fake)
            with self.assertRaisesRegex(ValueError, 'archive identity'):
                verify_native_targets(campaign, ROOT, rows)

    def test_temperature_SOC_condition_or_kind_mismatch_is_rejected(self):
        original, rows = fixture('C')
        for update in ({'temperature_setpoint_C': 10}, {'SOC_percent': [0, 100]}, {'kind': 'generic_csv'}):
            campaign = copy.deepcopy(original)
            campaign['source_identity'].update(update)
            with self.assertRaises(ValueError):
                verify_native_targets(campaign, ROOT, rows)

    def test_shared_outer_download_receipt_does_not_conflate_distinct_C_D_sources(self):
        validated = []
        for cell in ('C', 'D'):
            campaign, rows = fixture(cell)
            identity = verify_native_targets(campaign, ROOT, rows)
            validated.append({'data_kind': 'measured', 'cell_id': cell,
                'targets': {'sha256': hashlib.sha256(json.dumps(rows).encode()).hexdigest()},
                '_native_source_sha256': identity['source_identity_sha256'],
                '_native_source_member_path': identity['source_member_path']})
        self.assertTrue(validate_split(validated[0], [validated[1]]))

    def test_reformatted_target_alias_cannot_reuse_underlying_source_or_member(self):
        campaign, rows = fixture('C')
        identity = verify_native_targets(campaign, ROOT, rows)
        calibration = {'data_kind': 'measured', 'cell_id': 'C', 'targets': {'sha256': '1'*64},
            '_native_source_sha256': identity['source_identity_sha256'],
            '_native_source_member_path': identity['source_member_path']}
        for field in ('_native_source_sha256', '_native_source_member_path'):
            # Isolate the split guard after target CSV bytes and cell label differ.
            heldout = {'data_kind': 'measured', 'cell_id': 'D', 'targets': {'sha256': '2'*64},
                       '_native_source_sha256': '3'*64, '_native_source_member_path': 'different/member'}
            heldout[field] = calibration[field]
            with self.assertRaisesRegex(ValueError, 'underlying native summary or archive member'):
                validate_split(calibration, [heldout])

    def test_first_measurement_must_be_native_RPT0(self):
        campaign, rows = fixture('C')
        with self.assertRaisesRegex(ValueError, 'first native RPT0'):
            verify_native_targets(campaign, ROOT, rows[1:])

    def test_subset_is_explicitly_partial_and_full_native_curve_has_13_RPTs(self):
        campaign, subset = fixture('C')
        partial = verify_native_targets(campaign, ROOT, subset)
        self.assertFalse(partial['full_native_curve_evaluated'])
        self.assertEqual(partial['evaluated_RPT_ids'], ['RPT0', 'RPT1'])
        with Path(campaign['source_identity']['summary']['path']).open(encoding='utf-8-sig', newline='') as handle:
            full = [{'rpt_id': f"RPT{int(row['Ageing Sets'])}", 'Qref_Ah': float(row['C/10 Capacity [mA h]'])/1000}
                    for row in csv.DictReader(handle)]
        complete = verify_native_targets(campaign, ROOT, full)
        self.assertTrue(complete['full_native_curve_evaluated'])
        self.assertEqual(complete['native_RPT_count'], 13)
        self.assertEqual(complete['native_Q0_Ah'], full[0]['Qref_Ah'])

    def test_self_made_receipt_with_correct_official_URL_and_relocked_SHA_is_refused(self):
        campaign, rows = fixture('C')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            actual = Path(campaign['source_identity']['receipt']['path'])
            forged = json.loads(actual.read_text(encoding='utf-8'))
            forged['caller_reviewed'] = True  # URL, original CRC flags and source member still look official.
            path = Path(folder)/'self_made_receipt.json'; path.write_text(json.dumps(forged))
            campaign['source_identity']['receipt'] = ref(path)
            with self.assertRaisesRegex(ValueError, 'Unreviewed official archive identity or receipt bytes'):
                verify_native_targets(campaign, ROOT, rows)


if __name__ == '__main__':
    unittest.main()
