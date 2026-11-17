"""Regression tests for misleading numerical-comparison success reports."""
import copy
import math
import unittest

from convergence_contracts import DIAGNOSTIC_LEVELS, assess


VARIANTS = [{'variant': f'm{m}_f{f}', 'mesh': m, 'aging_factor': f}
            for m in (4, 8) for f in (1., .1, .01)]


def fixture():
    rows = []
    for variant in VARIANTS:
        for current in (3., 3.3):
            soh = 99.9 + (0.001 if current == 3.3 else 0.)
            rows.append({**variant, 'current_A': current,
                         'target_EFC': 1., 'achieved_EFC': 1.,
                         'throughput_target_Ah': 5., 'discharge_Ah': 5.,
                         'Q0_anchor_Ah': 5., 'execution_valid': True,
                         'measurements': [
                             {'rtol': rt, 'atol': at, 'Q0_Ah': 5.,
                              'Qref_Ah': soh * .05, 'SOH_percent': soh,
                              'main_unchanged': True}
                             for rt, at in DIAGNOSTIC_LEVELS]})
    return rows


def set_soh(measurement, soh):
    measurement['SOH_percent'] = soh
    measurement['Qref_Ah'] = soh * measurement['Q0_Ah'] / 100


class ConvergenceContracts(unittest.TestCase):
    def test_complete_stable_study_passes_only_short_screen(self):
        report = assess(fixture(), VARIANTS, 1.)
        self.assertTrue(report['execution_complete'], report['errors'])
        self.assertTrue(report['numerical_screen_pass'])
        self.assertFalse(report['long_term_convergence_established'])
        self.assertFalse(report['EOL_convergence_established'])
        self.assertFalse(report['real_cell_validation'])

    def test_stable_rank_with_large_aging_spread_still_fails(self):
        rows = fixture()
        for item in rows[1]['measurements']:
            set_soh(item, 99.902)
        report = assess(rows, VARIANTS, 1.)
        self.assertTrue(report['execution_complete'])
        self.assertTrue(report['rank_stable'])
        self.assertFalse(report['numerical_screen_pass'])
        self.assertGreater(report['aging_and_mesh_delta_span_pp'], 9e-4)

    def test_missing_or_duplicate_row_is_incomplete(self):
        rows = fixture()
        for bad in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.subTest(length=len(bad)):
                report = assess(bad, VARIANTS, 1.)
                self.assertFalse(report['execution_complete'])
                self.assertFalse(report['numerical_screen_pass'])

    def test_wrong_throughput_and_different_anchor_cannot_compare(self):
        for field, value in (('discharge_Ah', 4.9999), ('Q0_anchor_Ah', 5.01),
                             ('achieved_EFC', .999), ('target_EFC', 2.)):
            with self.subTest(field=field):
                rows = fixture()
                rows[0][field] = value
                self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])

    def test_nonfinite_data_and_false_execution_fail_closed(self):
        for bad_value in (math.nan, math.inf, None, True, 0., -1.):
            rows = fixture()
            rows[0]['measurements'][0]['Qref_Ah'] = bad_value
            self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])
        rows = fixture()
        rows[0]['execution_valid'] = False
        self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])

    def test_noisy_RPT_with_same_finest_aging_delta_fails(self):
        rows = fixture()
        set_soh(rows[1]['measurements'][0], 99.9013)
        report = assess(rows, VARIANTS, 1.)
        self.assertTrue(report['execution_complete'])
        self.assertEqual(report['aging_and_mesh_delta_span_pp'], 0.)
        self.assertTrue(report['rank_stable'])
        self.assertFalse(report['checks']['RPT_delta_span_within_limit'])
        self.assertFalse(report['numerical_screen_pass'])

    def test_individual_SOH_changes_do_not_hide_in_cancelled_delta(self):
        rows = fixture()
        for row in rows[:2]:
            for measurement in row['measurements']:
                set_soh(measurement, measurement['SOH_percent'] + .001)
        report = assess(rows, VARIANTS, 1.)
        self.assertTrue(report['execution_complete'])
        self.assertLess(report['aging_and_mesh_delta_span_pp'], 1e-12)
        self.assertFalse(report['checks']['individual_SOH_span_within_absolute_limit'])
        self.assertFalse(report['numerical_screen_pass'])

    def test_relative_limit_resolves_tiny_signals_conservatively(self):
        rows = fixture()
        for row in rows:
            if row['current_A'] == 3.3:
                for measurement in row['measurements']:
                    set_soh(measurement, 99.90001)
        for measurement in rows[1]['measurements']:
            set_soh(measurement, 99.900012)
        report = assess(rows, VARIANTS, 1.)
        self.assertTrue(report['rank_stable'])
        self.assertLess(report['aging_and_mesh_delta_span_pp'], 1e-4)
        self.assertFalse(report['numerical_screen_pass'])

    def test_unexpected_measurement_and_mutation_are_incomplete(self):
        for change in ('missing', 'duplicate', 'tolerance', 'mutation', 'bad_event'):
            rows = fixture()
            measurements = rows[0]['measurements']
            if change == 'missing':
                measurements.pop()
            elif change == 'duplicate':
                measurements.append(copy.deepcopy(measurements[0]))
            elif change == 'tolerance':
                measurements[0]['atol'] = 1e-6
            elif change == 'mutation':
                measurements[0]['main_unchanged'] = False
            else:
                measurements[0]['ok'] = False
            with self.subTest(change=change):
                self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])

    def test_inconsistent_ratio_or_Q0_and_missing_mesh_are_incomplete(self):
        rows = fixture()
        rows[0]['measurements'][0]['SOH_percent'] += .01
        self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])
        rows = fixture()
        measurement = rows[0]['measurements'][0]
        measurement['Q0_Ah'] = 5.01
        set_soh(measurement, measurement['SOH_percent'])
        self.assertFalse(assess(rows, VARIANTS, 1.)['execution_complete'])
        self.assertFalse(assess(fixture(), VARIANTS[:3], 1.)['execution_complete'])


if __name__ == '__main__':
    unittest.main()
