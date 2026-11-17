"""Exercise the actual report/driver paths with a failed numerical diagnostic."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from test_verification_contracts import valid_pilot_result

CAPSTONE = Path(__file__).resolve().parents[2]
IMPORT_OUT = CAPSTONE/'work/model_review_20261102/report_contract_import'
os.environ['BATTERY_OUTPUT_DIR'] = str(IMPORT_OUT)
import run_pilot9 as P
import verify_criteria as V


def result(cell_id):
    row = valid_pilot_result(cell_id)
    row['cell'].update(negative_particle_radius_m=5.86e-6,
                       sei_solvent_diffusivity_m2s=2.5e-22)
    row['at_5EFC']['Q5_Ah'] = 4.99-cell_id*.001
    return row


class LegacyReportIntegration(unittest.TestCase):
    def test_missing_cell4_strict_evidence_blocks_shared_error_estimate(self):
        with tempfile.TemporaryDirectory(dir=IMPORT_OUT) as directory:
            out = Path(directory)
            for cell_id in (1, 4):
                row = result(cell_id)
                if cell_id == 4:
                    del row['trajectory_strict']['reference_test']
                (out/f'cell_{cell_id}.json').write_text(json.dumps(row))
            with patch.object(P, 'OUT', out), patch.object(P, 'CFG', {'cells': [{'id': 1}, {'id': 4}]}):
                summary = P.summarize()
                self.assertEqual(summary['invalid_or_incomplete_cell_ids'], [1, 4])
                self.assertEqual(summary['distinguishable_pairs']['Q5'], [0, 0])

    def test_actual_driver_records_strict_failures_and_state_mutation(self):
        class State:
            def __init__(self):
                self.y = np.array([1., 2.])
                self.t = np.array([1.])
            @property
            def last_state(self):
                return self
            def copy(self):
                return copy.deepcopy(self)

        for failure in ('none', 'initial_r10', 'strict_reference', 'strict_mutation',
                        'strict_trajectory', 'strict_trajectory_reference'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=IMPORT_OUT) as directory:
                made = []
                class Cell:
                    def __init__(self, *args, **kwargs):
                        self.index = len(made)
                        made.append(self)
                        self.param = {k: 1. for k in ('Negative particle radius [m]',
                            'SEI solvent diffusivity [m2.s-1]', 'Lithium plating kinetic rate constant [m.s-1]',
                            'Nominal cell capacity [A.h]')}
                        self.r10_calls = 0
                        self.reference_calls = 0
                    def initial_state(self):
                        return State()
                    def reference_test(self, sol):
                        self.reference_calls += 1
                        failed = ((failure == 'strict_reference' and self.index == 3) or
                                  (failure == 'strict_trajectory_reference' and self.index == 4
                                   and self.reference_calls == 2))
                        return {'ok': not failed,
                                'Q_trapz_Ah': 5., 'Q_cum_Ah': 5.}
                    def r10_test(self, sol):
                        self.r10_calls += 1
                        if failure == 'strict_mutation' and self.index == 3:
                            sol.y[0] += 1.
                        return {'ok': not (failure == 'initial_r10' and self.index == 0 and self.r10_calls == 1),
                                'R10_ohm': .03}
                def operation(cell, source, cell_id, trajectory, *args, **kwargs):
                    cycle = {'charge_time_s': 300., 'soc_charge_end': .8, 'soc_discharge_end': .2}
                    status = 'max_cycles' if trajectory.startswith('ctrl_') else 'efc_target'
                    if failure == 'strict_trajectory' and trajectory.endswith('_strict'):
                        status = 'wall_limit'
                    return source.copy(), SimpleNamespace(efc=5.), {'status': status, 'cycles': [cycle],
                        'end_state': {k: 1. for k in ('sei_Ah', 'plated_Ah', 'dead_Ah', 'discharge_capacity_Ah', 'soc')},
                        'wall_s': 1.}
                with patch.object(P, 'OUT', Path(directory)), patch.object(P.B, 'Cell', Cell), \
                     patch.object(P.B, 'y_end', side_effect=lambda sol: sol.y), \
                     patch.object(P.B, 'ledger', side_effect=lambda ledger: vars(ledger).copy()), \
                     patch.object(P, 'operation', side_effect=operation):
                    row = P.run_cell(4, {'rtol': 1e-4, 'atol': 1e-6})
                self.assertNotIn('exception', row['failures'], row.get('traceback'))
                if failure == 'none':
                    self.assertEqual(row['failures'], [])
                    self.assertTrue(P.pilot_result_complete(row))
                    continue
                expected = {'initial_r10': 'R10_initial_termination',
                            'strict_reference': 'strict_diagnostic_at_5EFC_termination',
                            'strict_mutation': 'strict_diagnostic_changed_main_or_ledger',
                            'strict_trajectory': 'strict_trajectory:wall_limit',
                            'strict_trajectory_reference': 'strict_trajectory_diagnostic_invalid'}[failure]
                self.assertIn(expected, row['failures'])
                self.assertFalse(P.pilot_result_complete(row))

    def test_verification_failure_returns_nonzero_exit(self):
        with tempfile.TemporaryDirectory(dir=IMPORT_OUT) as directory:
            with patch.object(V, 'OUT', Path(directory)), patch.object(V, 'REPORT', {}), \
                 patch.object(V, 'part_a', side_effect=RuntimeError('sentinel bad reference')):
                self.assertEqual(V.main(), 1)
                self.assertIn('sentinel bad reference', V.REPORT['error'])

    def test_single_cell_failure_returns_nonzero_exit(self):
        with tempfile.TemporaryDirectory(dir=IMPORT_OUT) as directory:
            verify = Path(directory)/'verify.json'
            verify.write_text(json.dumps({'all_pass': True, 'selected_tolerance': {'rtol': 1e-4, 'atol': 1e-6}}))
            with patch.object(P, 'VERIFY', verify), patch.object(P, 'run_cell', return_value={'failures': ['exception']}), \
                 patch.object(sys, 'argv', ['run_pilot9.py', '--cell', '1']):
                with self.assertRaises(SystemExit):
                    P.main()

    def test_failed_cell_is_excluded_and_driver_does_not_write_DONE(self):
        with tempfile.TemporaryDirectory(dir=IMPORT_OUT) as directory:
            out = Path(directory)
            good, bad = result(1), result(2)
            bad['numerical_initial']['mesh_x2']['ok'] = False
            for i, row in ((1, good), (2, bad)):
                (out/f'cell_{i}.json').write_text(json.dumps(row))
            verify = out/'verify.json'
            verify.write_text(json.dumps({'all_pass': True, 'selected_tolerance': {'rtol': 1e-4, 'atol': 1e-6}}))
            (out/'DONE').write_text('stale marker from earlier attempt')
            cfg = {'cells': [{'id': 1}, {'id': 2}]}
            with patch.object(P, 'OUT', out), patch.object(P, 'VERIFY', verify), patch.object(P, 'CFG', cfg):
                summary = P.summarize()
                self.assertFalse(summary['all_pass'])
                self.assertEqual(summary['invalid_or_incomplete_cell_ids'], [2])
                self.assertEqual(summary['distinguishable_pairs']['Q5'], [0, 0])
                self.assertIsNone(summary['rows'][1]['eps_Q_Ah'])
                with patch.object(sys, 'argv', ['run_pilot9.py']), patch.object(P.subprocess, 'run'):
                    with self.assertRaises(SystemExit):
                        P.main()
                self.assertFalse((out/'DONE').exists())
                self.assertEqual(len(list(out.glob('DONE.previous.*'))), 1)


if __name__ == '__main__':
    unittest.main()
