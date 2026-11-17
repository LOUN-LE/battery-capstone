import unittest
from empirical_validation import prepare_summary, eol_bracket, compare_curves, MATCH_FIELDS


def row(rpt, cycles, throughput, capacity):
    return {'Ageing Sets': rpt, 'Ageing Cycles': cycles, 'Charge Throughput [A h]': throughput,
            'C/10 Capacity [mA h]': capacity, 'Days of degradation': rpt * 10}


def metadata():
    return {**{key: key for key in MATCH_FIELDS}, 'axis_column': 'absolute_ageing_EFC',
            'axis_definition': 'charge_plus_discharge_throughput_Ah/(2*RPT0_C10_Ah)'}


class EmpiricalTests(unittest.TestCase):
    def test_fixed_measured_q0_and_absolute_axis(self):
        p = prepare_summary([row(0, 0, 0, 5000), row(1, 517, 10, 4500)])
        self.assertEqual(p[1]['SOH_percent'], 90)
        self.assertEqual(p[1]['absolute_ageing_EFC'], 1)
        self.assertEqual(p[1]['approximate_EFC'], p[1]['absolute_ageing_EFC'])
        self.assertEqual(p[1]['ageing_cycles'], 517)

    def test_no_eol_extrapolation(self):
        p = prepare_summary([row(0, 0, 0, 5000), row(1, 100, 10, 4500)])
        self.assertEqual(eol_bracket(p, 'ageing_cycles')['status'], 'right_censored')
        self.assertIsNone(eol_bracket(p, 'ageing_cycles')['upper'])

    def test_eol_interval_not_exact_cycle(self):
        p = prepare_summary([row(0, 0, 0, 5000), row(1, 100, 10, 3900)])
        self.assertEqual(eol_bracket(p, 'ageing_cycles')['upper'], 100)
        self.assertEqual(eol_bracket(p, 'ageing_cycles')['lower'], 0)

    def test_bad_measured_inputs_rejected(self):
        for bad in (float('nan'), 0, -1):
            with self.assertRaises(ValueError):
                prepare_summary([row(0, 0, 0, bad)])
        with self.assertRaises(ValueError):
            prepare_summary([row(0, 0, 0, 5000), row(1, 1, 0, 4900)])

    def test_comparison_requires_matching_conditions_and_full_coverage(self):
        meta = metadata()
        p = [{'absolute_ageing_EFC': 0, 'SOH_percent': 100},
             {'absolute_ageing_EFC': 1, 'SOH_percent': 90}]
        result = compare_curves(p, p, meta, meta, 'absolute_ageing_EFC')
        self.assertEqual(result['RMSE_pp'], 0)
        self.assertIsNone(result['physical_validation_pass'])
        with self.assertRaises(ValueError):
            compare_curves(p, p, meta, {}, 'absolute_ageing_EFC')
        with self.assertRaises(ValueError):
            compare_curves(p + [{'absolute_ageing_EFC': 2, 'SOH_percent': 85}], p,
                           meta, meta, 'absolute_ageing_EFC')

    def test_nonfinite_prediction_rejected(self):
        meta = metadata()
        p = [{'absolute_ageing_EFC': 0, 'SOH_percent': 100},
             {'absolute_ageing_EFC': 1, 'SOH_percent': float('nan')}]
        with self.assertRaises(ValueError):
            compare_curves(p, p, meta, meta, 'absolute_ageing_EFC')

    def test_axis_column_and_definition_are_separate_and_both_required(self):
        meta = metadata()
        p = prepare_summary([row(0, 0, 0, 5000), row(1, 517, 10, 4500)])
        self.assertEqual(compare_curves(p, p, meta, meta, 'absolute_ageing_EFC')['RMSE_pp'], 0)
        for key in ('axis_column', 'axis_definition'):
            altered = dict(meta); del altered[key]
            with self.assertRaises(ValueError):
                compare_curves(p, p, meta, altered, 'absolute_ageing_EFC')
        for altered in (dict(meta, axis_column='ageing_cycles'),
                        dict(meta, axis_definition='discharge_Ah/fixed_Q0_Ah')):
            with self.assertRaises(ValueError):
                compare_curves(p, p, meta, altered, 'absolute_ageing_EFC')
        with self.assertRaises(ValueError):
            compare_curves(p, p, meta, meta, 'ageing_cycles')

    def test_interpolation_error_metrics_with_real_axis_names(self):
        meta = metadata()
        measured = [{'absolute_ageing_EFC': x, 'SOH_percent': y} for x, y in ((0, 100), (1, 94), (2, 90))]
        predicted = [{'absolute_ageing_EFC': x, 'SOH_percent': y} for x, y in ((0, 100), (2, 90))]
        result = compare_curves(measured, predicted, meta, meta, 'absolute_ageing_EFC')
        self.assertAlmostEqual(result['RMSE_pp'], (1/3)**.5)
        self.assertEqual(result['max_abs_error_pp'], 1)
        self.assertAlmostEqual(result['mean_error_pp'], 1/3)

    def test_eol_invalid_observations_cannot_be_censored(self):
        for points in ([],
                       [{'x': 0, 'SOH_percent': float('nan')}],
                       [{'x': float('inf'), 'SOH_percent': 90}],
                       [{'x': 1, 'SOH_percent': 90}, {'x': 1, 'SOH_percent': 85}],
                       [{'x': 1, 'SOH_percent': 90}, {'x': 0, 'SOH_percent': 85}],
                       [{'x': 0, 'SOH_percent': 100}, {'x': 1, 'SOH_percent': 79},
                        {'x': 2, 'SOH_percent': float('nan')}]):
            with self.assertRaises(ValueError):
                eol_bracket(points, 'x')

    def test_first_observation_below_threshold_is_left_censored(self):
        result = eol_bracket([{'x': 10, 'SOH_percent': 79}], 'x')
        self.assertEqual(result['status'], 'left_censored')
        self.assertIsNone(result['lower'])
        self.assertEqual(result['upper'], 10)


if __name__ == '__main__':
    unittest.main()
