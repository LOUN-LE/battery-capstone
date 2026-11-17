import unittest
import copy
from verification_contracts import reference_variants_valid, pilot_result_complete


def valid_pilot_result(cell_id=1):
    result = {'cell': {'id': cell_id}, 'memory': {}, 'failures': [],
        'Q0': {'Q0_Ah': 5., 'reference_test': {'ok': True}},
        'R10_initial': {'ok': True, 'R10_ohm': .03}, 'initial_state_unchanged': True,
        'numerical_initial': {k: {'ok': True, 'dQ0_Ah': .00001, 'dR10_ohm': .00001,
                                  'main_state_unchanged': True}
                              for k in ('tol_x0.01', 'mesh_x2')},
        'at_5EFC': {'Q5_Ah': 4.99, 'reference_test': {'ok': True},
            'R10': {'R10_ohm': .03, 'ok': True}, 'main_state_unchanged': True,
            'ledger_unchanged': True, 'strict_from_same_state': {
                'Q5_Ah': 4.99001, 'R10_ohm': .03001,
                'dQ5_Ah': .00001, 'dR10_ohm': .00001,
                'reference_test': {'ok': True}, 'R10': {'ok': True},
                'main_state_unchanged': True, 'ledger_unchanged': True}}}
    if cell_id == 4:
        result['trajectory_strict'] = {'status': 'efc_target', 'Q0_Ah': 5.,
            'Q5_Ah': 4.99001, 'R10_ohm': .03001, 'dQ5_Ah': .00001,
            'dR10_ohm': .00001, 'dSOH_pp': .0002,
            'Q0_reference_test': {'ok': True}, 'reference_test': {'ok': True},
            'R10': {'ok': True}, 'initial_state_unchanged': True,
            'main_state_unchanged': True, 'ledger_unchanged': True}
    return result


class ReferenceVerification(unittest.TestCase):
    def test_matching_capacity_cannot_hide_incomplete_reference(self):
        variants = [{'ok': True, 'Q_Ah': 5.}, {'ok': False, 'Q_Ah': 5.}]
        self.assertFalse(reference_variants_valid(variants))

    def test_missing_or_invalid_capacity_is_not_evidence(self):
        for value in (float('nan'), float('inf'), 0., -5., None):
            self.assertFalse(reference_variants_valid([{'ok': True, 'Q_Ah': value}]))
        self.assertFalse(reference_variants_valid([]))

    def test_all_complete_positive_references(self):
        self.assertTrue(reference_variants_valid([{'ok': True, 'Q_Ah': 5.},
                                                   {'ok': True, 'Q_Ah': 4.99999}]))

    def test_failed_or_partial_pilot_cannot_be_called_complete(self):
        self.assertFalse(pilot_result_complete({'memory': {}, 'failures': []}))
        self.assertFalse(pilot_result_complete({'memory': {}, 'failures': ['exception'],
                                               'at_5EFC': {'Q5_Ah': 5.}}))

    def test_pilot_requires_both_initial_numerical_diagnostics(self):
        result = valid_pilot_result()
        self.assertTrue(pilot_result_complete(result))
        result['numerical_initial']['mesh_x2']['ok'] = False
        self.assertFalse(pilot_result_complete(result))

    def test_matching_strict_numbers_cannot_hide_missing_invalid_or_mutating_diagnostics(self):
        for field in ('reference_test', 'R10', 'main_state_unchanged', 'ledger_unchanged'):
            for missing in (False, True):
                result = valid_pilot_result()
                strict = result['at_5EFC']['strict_from_same_state']
                if missing:
                    del strict[field]
                else:
                    strict[field] = {'ok': False} if field in ('reference_test', 'R10') else False
                with self.subTest(field=field, missing=missing):
                    self.assertFalse(pilot_result_complete(result))

    def test_initial_r10_must_be_valid_and_preserved(self):
        result = valid_pilot_result()
        result['R10_initial']['ok'] = False
        self.assertFalse(pilot_result_complete(result))
        result = valid_pilot_result()
        result['initial_state_unchanged'] = False
        self.assertFalse(pilot_result_complete(result))

    def test_cell4_requires_completed_preserved_strict_trajectory(self):
        good = valid_pilot_result(4)
        self.assertTrue(pilot_result_complete(good))
        for key in ('status', 'Q0_reference_test', 'reference_test', 'R10',
                    'initial_state_unchanged', 'main_state_unchanged', 'ledger_unchanged'):
            result = copy.deepcopy(good)
            del result['trajectory_strict'][key]
            with self.subTest(missing=key):
                self.assertFalse(pilot_result_complete(result))
        result = copy.deepcopy(good)
        result['trajectory_strict']['status'] = 'wall_limit'
        self.assertFalse(pilot_result_complete(result))


if __name__ == '__main__':
    unittest.main()
