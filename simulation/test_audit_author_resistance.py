import unittest
import numpy as np
from audit_author_resistance import observed_resistance_rows


class AuthorResistanceTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(20,dtype=float);v=np.full(20,4.);i=np.zeros(20);flags=np.zeros(20,dtype=bool)
        i[12:]=-2000.;v[12:]=3.9;flags[12]=True
        return t,v,i,flags

    def test_nine_baseline_samples_exclude_contaminated_prior_row(self):
        t,v,i,f=self.fixture();v[11]=3.9
        r=observed_resistance_rows(t,v,i,f)[0]
        self.assertEqual((r['baseline_row_start'],r['baseline_row_end_exclusive']),(2,11))
        self.assertEqual(r['baseline_voltage_V'],4)
        self.assertAlmostEqual(r['resistance_Ohm'],.05)

    def test_flag_after_largest_current_step_shifts_one_row_left(self):
        t,v,i,f=self.fixture();i[11]=-1900
        r=observed_resistance_rows(t,v,i,f)[0]
        self.assertTrue(r['row_shifted_left']);self.assertEqual(r['adjusted_row'],11)

    def test_point_one_window_can_select_zero_current_prior_row(self):
        t,v,i,f=self.fixture();t=np.arange(20,dtype=float)*.1
        t[11]=t[12]-.002;t[13]=t[12]+.100000010
        r=observed_resistance_rows(t,v,i,f,timestep=.1)[0]
        self.assertEqual(r['endpoint_row'],11)
        self.assertTrue(r['zero_denominator'])
        self.assertEqual(r['status'],'upstream_nonfinite_resistance')
        self.assertIsNone(r['resistance_Ohm'])

    def test_actual_current_difference_sets_denominator(self):
        t,v,i,f=self.fixture();i[:12]=50.;i[12:]=-1950
        r=observed_resistance_rows(t,v,i,f)[0]
        self.assertEqual(r['denominator_A'],2)

    def test_time_endpoint_is_first_open_window_match(self):
        t,v,i,f=self.fixture()
        r=observed_resistance_rows(t,v,i,f,timestep=2)[0]
        self.assertEqual(r['endpoint_row'],13)

    def test_absent_endpoint_remains_skipped(self):
        t,v,i,f=self.fixture()
        r=observed_resistance_rows(t,v,i,f,timestep=100)[0]
        self.assertEqual(r['status'],'upstream_no_endpoint_skip')

    def test_raw_arrays_are_unchanged(self):
        data=self.fixture();before=[a.copy() for a in data]
        observed_resistance_rows(*data)
        for a,b in zip(data,before):np.testing.assert_array_equal(a,b)

    def test_invalid_shape_refused(self):
        with self.assertRaises(ValueError):observed_resistance_rows([0,1],[4],[0,0],[0,0])


if __name__=='__main__':unittest.main()
