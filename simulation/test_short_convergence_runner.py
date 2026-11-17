"""Exercise staged reporting, provenance rejection, and a real child timeout."""
import copy
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import audit_short_convergence as audit
from test_convergence_contracts import fixture as complete_fixture


def aging_fixture():
    names = {(v['mesh'], v['aging_factor']): v['variant'] for v in audit.VARIANTS}
    rows = complete_fixture()
    for row in rows:
        row['variant'] = names[(row['mesh'], row['aging_factor'])]
        row['measurements'] = row['measurements'][:1]
        row['measurements'][0]['initial_unchanged'] = True
    return rows


def anchor_fixture(directory):
    source, out = directory / 'source', directory / 'out'
    source.mkdir()
    out.mkdir()
    raw = source / 'anchor_rpt.npz'
    raw.write_bytes(b'fixed raw artifact for hash-validation test')
    hashes = {name: 'expected-sha-' + str(index) for index, name in enumerate((
        'simulation/rebuilt_lifetime.py', 'configs/model.json',
        'configs/diagnostics.json', 'configs/cells_pilot9.json'))}
    anchor = {'Q0_anchor_Ah': 5., 'mesh': 8, 'rtol': 1e-8, 'atol': 1e-10,
              'initial_fingerprint': 'fixture-initial-state',
              'reference': {'Qref_Ah': 5., 'main_unchanged': True},
              'raw_sha256': audit.digest(raw)}
    parent = {'source_sha256': hashes.copy(),
              'python': audit.platform.python_version(),
              'platform': audit.platform.platform(),
              'versions': {name: 'fixture-version' for name in
                           ('pybamm', 'pybammsolvers', 'numpy', 'casadi')},
              'anchor': anchor}
    audit.save(source / 'anchor.json', anchor)
    audit.save(source / 'report.json', parent)
    return source, out, hashes


class StagedAgingReport(unittest.TestCase):
    def test_all_twelve_valid_cases_complete_execution_without_precision_pass(self):
        report = audit.assess_aging(aging_fixture(), 1., 5.)
        self.assertTrue(report['execution_complete'])
        self.assertEqual(report['valid_cases'], 12)
        self.assertEqual(len(report['paired_descriptive_comparisons']), 6)
        self.assertFalse(report['numerical_screen_pass'])
        self.assertFalse(report['RPT_convergence_established'])
        self.assertFalse(report['long_term_convergence_established'])
        self.assertTrue(all(pair['descriptive_only'] for pair in
                            report['paired_descriptive_comparisons']))

    def test_failed_case_is_preserved_and_its_pair_is_excluded(self):
        rows = aging_fixture()
        rows[0].update(execution_valid=False, error='sentinel finite-volume failure')
        before = copy.deepcopy(rows)
        report = audit.assess_aging(rows, 1., 5.)
        self.assertEqual(rows, before)
        self.assertFalse(report['execution_complete'])
        self.assertEqual(report['valid_cases'], 11)
        self.assertEqual(len(report['paired_descriptive_comparisons']), 5)
        self.assertEqual(report['invalid_cases'][0]['error'],
                         'sentinel finite-volume failure')
        self.assertNotIn(rows[0]['variant'], [pair['variant'] for pair in
                                            report['paired_descriptive_comparisons']])

    def test_missing_case_cannot_complete_stage(self):
        rows = aging_fixture()[:-1]
        report = audit.assess_aging(rows, 1., 5.)
        self.assertFalse(report['execution_complete'])
        self.assertFalse(report['numerical_screen_pass'])
        self.assertEqual(report['valid_cases'], 11)
        self.assertEqual(len(report['paired_descriptive_comparisons']), 5)
        self.assertTrue(any(row['error'] == 'Case missing' for row in report['invalid_cases']))

    def test_wrong_mesh_metadata_invalidates_case_and_excludes_its_pair(self):
        rows = aging_fixture()
        rows[0]['mesh'] = 999
        report = audit.assess_aging(rows, 1., 5.)
        self.assertFalse(report['execution_complete'])
        self.assertLess(report['valid_cases'], 12)
        self.assertTrue(any(case['variant'] == rows[0]['variant']
                            and case['current_A'] == rows[0]['current_A']
                            for case in report['invalid_cases']))
        self.assertNotIn(rows[0]['variant'], [pair['variant'] for pair in
                                            report['paired_descriptive_comparisons']])

    def test_failed_duplicate_invalidates_both_copies_and_excludes_pair(self):
        rows = aging_fixture()
        duplicate = copy.deepcopy(rows[0])
        duplicate.update(execution_valid=False, error='duplicate worker failure')
        rows.append(duplicate)
        report = audit.assess_aging(rows, 1., 5.)
        self.assertFalse(report['execution_complete'])
        self.assertLessEqual(report['valid_cases'], 11)
        self.assertNotIn(duplicate['variant'], [pair['variant'] for pair in
                                               report['paired_descriptive_comparisons']])

    def test_inconsistent_same_mesh_Q0_excludes_affected_comparison(self):
        rows = aging_fixture()
        measurement = rows[0]['measurements'][0]
        measurement['Q0_Ah'] = 5.01
        # Preserve the SOH ratio so the test isolates a changed initial
        # reference, rather than an already-covered arithmetic inconsistency.
        measurement['Qref_Ah'] = measurement['SOH_percent'] * measurement['Q0_Ah'] / 100
        before = copy.deepcopy(rows)
        report = audit.assess_aging(rows, 1., 5.)
        self.assertEqual(rows, before)
        self.assertFalse(report['execution_complete'])
        self.assertLess(report['valid_cases'], 12)
        self.assertNotIn(rows[0]['variant'], [pair['variant'] for pair in
                                            report['paired_descriptive_comparisons']])


class ChildTimeout(unittest.TestCase):
    def test_real_sleeping_child_is_killed_reaped_and_removed_from_registry(self):
        spawned = []
        popen = audit.subprocess.Popen

        def capture_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            spawned.append(process)
            return process

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                patch.object(audit, 'CHILDREN', set()), \
                patch.object(audit, 'CHILDREN_LOCK', threading.Lock()), \
                patch.object(audit, 'CANCEL', threading.Event()), \
                patch.object(audit.subprocess, 'Popen', side_effect=capture_process):
            receipt = audit.run_child(
                [sys.executable, '-c', 'import time; time.sleep(10)'],
                Path(directory) / 'sleep.log', timeout_s=.2)
            self.assertTrue(receipt['timeout'])
            self.assertTrue(receipt['worker_terminated'])
            self.assertIsNone(receipt['returncode'])
            self.assertEqual(len(spawned), 1)
            self.assertEqual(receipt['pid'], spawned[0].pid)
            self.assertIsNotNone(spawned[0].poll())
            self.assertLess(receipt['elapsed_s'], 5.)
            self.assertEqual(audit.CHILDREN, set())


class AnchorReuseProvenance(unittest.TestCase):
    def test_modified_raw_artifact_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                patch.object(audit.importlib.metadata, 'version', return_value='fixture-version'):
            source, out, hashes = anchor_fixture(Path(directory))
            (source / 'anchor_rpt.npz').write_bytes(b'tampered raw artifact')
            with self.assertRaisesRegex(ValueError, 'raw digest mismatch'):
                audit.reuse_anchor(source, out, hashes)
            self.assertEqual(list(out.iterdir()), [])

    def test_different_engine_hash_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                patch.object(audit.importlib.metadata, 'version', return_value='fixture-version'):
            source, out, hashes = anchor_fixture(Path(directory))
            hashes['simulation/rebuilt_lifetime.py'] = 'different-current-engine'
            with self.assertRaisesRegex(ValueError, 'engine/configuration provenance mismatch'):
                audit.reuse_anchor(source, out, hashes)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
