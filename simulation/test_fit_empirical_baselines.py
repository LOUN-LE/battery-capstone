import copy
import tempfile
from pathlib import Path
import unittest

from fit_empirical_baselines import fit_calibration, predict, load_summary, observed_eol


def fixture(cell='C'):
    return [{'cell': cell, 'calendar_days_since_RPT0': day,
             'absolute_ageing_EFC': efc, 'SOH_percent': soh}
            for day, efc, soh in [(0, 0, 100), (10, 30, 99), (40, 100, 97)]]


class BaselineTests(unittest.TestCase):
    def test_no_heldout_outcomes_in_prediction_or_fit(self):
        model = fit_calibration(fixture(), 'C', 'frozen-test-hash')
        heldout = fixture('D')
        before = predict(model, heldout)
        for point in heldout:
            point['SOH_percent'] = -500
        self.assertEqual(before, predict(model, heldout))
        self.assertEqual(model, fit_calibration(fixture(), 'C', 'frozen-test-hash'))
        with self.assertRaises(ValueError):
            fit_calibration(heldout, 'C', 'not-a-training-source')

    def test_fixed_intercept_and_nonnegative_coefficients(self):
        model = fit_calibration(fixture(), 'C', 'source')
        self.assertGreaterEqual(model.sqrt_calendar_coefficient_pp, 0)
        self.assertGreaterEqual(model.linear_efc_coefficient_pp, 0)
        self.assertEqual(predict(model, fixture())[0], 100)
        with self.assertRaises(Exception):
            model.fit_cell = 'D'

    def test_censored_eol_never_extrapolates(self):
        result = observed_eol(fixture())
        self.assertEqual(result['status'], 'right_censored')
        self.assertIsNone(result['EFC80'])

    def test_rpt_date_not_reported_ageing_days(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.csv'
            path.write_text('Ageing Sets,Ageing Cycles,RPT start date,Days of degradation,Charge Throughput [A h],C/10 Capacity [mA h]\n'
                            '0,0,2020-01-01,0,0,5000\n1,1,2020-01-11,3,10,4900\n2,2,2020-01-21,8,20,4800\n', encoding='utf-8')
            points = load_summary(path, 'C')
            self.assertEqual(points[1]['calendar_days_since_RPT0'], 10)
            self.assertEqual(points[1]['reported_days_of_degradation'], 3)
            self.assertEqual(points[1]['absolute_ageing_EFC'], 1)
            self.assertAlmostEqual(points[1]['SOH_percent'], 98)

    def test_nonfinite_exposure_rejected(self):
        bad = copy.deepcopy(fixture())
        bad[1]['absolute_ageing_EFC'] = float('nan')
        with self.assertRaises(ValueError):
            fit_calibration(bad, 'C', 'source')


if __name__ == '__main__':
    unittest.main()
