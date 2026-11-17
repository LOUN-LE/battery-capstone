import copy
import unittest
import numpy as np

from dynamic_observation_operator import RawTrace,first_reached_sample,voltage_shapes,c_half_rate_cutoff


def fixture():
    time=np.array([100.,100.3,101.2,109.9,110.2,129.9,130.,131.,140.,160.,700.,1900.])
    n=len(time)
    arrays=dict(raw_source_row=np.arange(n)+200,run_index=np.array([0]*6+[1]*6),time_s=time,
        Ns=np.array([5]*6+[6]*6),I_mA=np.array([-2500.]*6+[0.]*6),V=np.arange(n)*.01+3.,
        mode=np.array([1]*6+[3]*6),surface_temperature_C=np.ones(n)*25.,dq_mAh=np.ones(n),
        Q_minus_Qo_mAh=np.arange(n,dtype=float))
    run=dict(file_id='synthetic',part='long_03',purpose='unit_test',run_index=0,Ns=5,
        row_start=0,row_end_exclusive=6,warning_flags=['synthetic_flag'],role='pulse')
    return arrays,run


class DynamicObservationTests(unittest.TestCase):
    def test_first_at_or_after_uses_actual_raw_sample_without_interpolation(self):
        a,r=fixture();p=first_reached_sample(RawTrace(a),r,1.)
        self.assertEqual(p['raw_source_row'],202)
        self.assertAlmostEqual(p['actual_elapsed_s'],1.2)
        self.assertAlmostEqual(p['overshoot_from_request_s'],.2)
        self.assertEqual(p['V'],a['V'][2])
        self.assertEqual(p['run_warning_flags'],['synthetic_flag'])

    def test_target_at_next_run_is_missing_instead_of_borrowing_following_step(self):
        a,r=fixture();p=first_reached_sample(RawTrace(a),r,30.)
        self.assertEqual(p['sample_status'],'missing')
        self.assertIsNone(p['raw_source_row'])
        self.assertIn('this_run',p['missing_reason'])

    def test_voltage_shape_has_signed_delta_and_actual_reference_elapsed(self):
        a,r=fixture();before={k:v.copy() for k,v in a.items()}
        rows=voltage_shapes(RawTrace(a),r,[1.,10.,30.],'test_shape')
        self.assertAlmostEqual(rows[1]['delta_voltage_mV_from_1s_selected_sample'],20.)
        self.assertAlmostEqual(rows[1]['actual_time_since_reference_s'],9.)
        self.assertFalse(rows[1]['is_resistance'])
        self.assertEqual(rows[2]['shape_status'],'missing')
        self.assertTrue(all(np.array_equal(a[k],v) for k,v in before.items()))

    def test_nonfinite_voltage_is_explicit_and_does_not_get_zero_filled(self):
        a,r=fixture();a['V'][2]=np.nan
        rows=voltage_shapes(RawTrace(a),r,[1.,10.,30.],'test_shape')
        self.assertEqual(rows[0]['sample_status'],'nonfinite_sample')
        self.assertIsNone(rows[0]['V'])
        self.assertIsNone(rows[1]['delta_voltage_mV_from_1s_selected_sample'])
        self.assertEqual(rows[1]['shape_missing_reason'],'reference_sample_unavailable')

    def test_repeated_Ns_does_not_merge_executions(self):
        a,r=fixture();a['Ns'][6:]=5
        r['row_end_exclusive']=12
        with self.assertRaisesRegex(ValueError,'different recorded executions'):
            first_reached_sample(RawTrace(a),r,1.)

    def test_temperature_flag_is_preserved_without_removing_sample(self):
        a,r=fixture();q=np.zeros(len(a['time_s']),dtype=bool);q[2]=True;a['surface_temperature_C'][2]=5.
        p=first_reached_sample(RawTrace(a,q),r,1.)
        self.assertEqual(p['sample_status'],'observed')
        self.assertEqual(p['surface_temperature_C'],5.)
        self.assertIn('temperature_adjacent_to_gt2C_sample_jump_diagnostic_mask',p['point_quality_flags'])

    def test_clock_reversal_and_duplicate_are_not_silently_sorted_or_dropped(self):
        for value in (100.,99.,float('nan')):
            a,r=fixture();a['time_s'][1]=value
            with self.assertRaises(ValueError):RawTrace(a)

    def test_bad_requested_time_refused(self):
        a,r=fixture()
        for tau in (-1.,float('nan'),True):
            with self.assertRaises(ValueError):first_reached_sample(RawTrace(a),r,tau)

    def test_coarse_rest_sampling_does_not_create_a_false_zero_shape(self):
        a,r=fixture();a['time_s'][:6]=[100.,110.1,120.,129.9,129.95,129.99]
        rows=voltage_shapes(RawTrace(a),r,[1.,10.,30.],'coarse_shape')
        self.assertEqual(rows[0]['raw_source_row'],rows[1]['raw_source_row'])
        self.assertEqual(rows[1]['sample_status'],'observed')
        self.assertEqual(rows[1]['shape_status'],'missing')
        self.assertIsNone(rows[1]['delta_voltage_mV_from_1s_selected_sample'])
        self.assertIn('same_recorded_sample',rows[1]['shape_missing_reason'])

    def test_cutoff_not_reached_is_explicit_and_Q_is_absolute_integral_not_SOH(self):
        a,r=fixture();q=c_half_rate_cutoff(RawTrace(a),r,2.5,'discharge')
        self.assertFalse(q['cutoff_crossing_in_recorded_samples'])
        self.assertIsNone(q['first_cutoff_source_row'])
        self.assertAlmostEqual(q['signed_Ah_trapezoid_within_run'],-2.5*29.9/3600)
        self.assertAlmostEqual(q['instrument_dq_Ah_sum'],.006)
        self.assertTrue(q['not_SOHRPT_Q0'])
        self.assertAlmostEqual(q['following_sample_gap_s'],.1)

    def test_no_partial_run_claim_or_nonmonotone_shape_request(self):
        a,r=fixture();bad=copy.deepcopy(r);bad['row_start']=1
        with self.assertRaises(ValueError):first_reached_sample(RawTrace(a),bad,1.)
        with self.assertRaises(ValueError):voltage_shapes(RawTrace(a),r,[1.,30.,10.],'bad')


if __name__=='__main__':unittest.main()
