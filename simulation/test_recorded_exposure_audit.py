import unittest
import numpy as np
from recorded_exposure_audit import integrate_window, seconds_us


class IntervalAuditTests(unittest.TestCase):
    def test_stale_rest_current_is_not_integrated(self):
        r=integrate_window([0,1,3601,3602],[1500,1500,0,0],np.array([True,False,False]),0,3602)
        self.assertAlmostEqual(r['observed_charge_Ah'],1.5/3600)
        self.assertEqual(r['valid_duration_s'],1)

    def test_partial_interval_remains_unknown(self):
        r=integrate_window([0,10],[0,2000],np.array([True]),2,8)
        self.assertEqual(r['valid_duration_s'],0)
        self.assertEqual(r['observed_charge_Ah'],0)

    def test_sign_crossing_counts_both_directions(self):
        r=integrate_window([0,10],[-2000,2000],np.array([True]),0,10)
        self.assertAlmostEqual(r['signed_native_current_Ah'],0)
        self.assertAlmostEqual(r['observed_charge_Ah'],5/3600)
        self.assertAlmostEqual(r['observed_discharge_Ah'],5/3600)

    def test_clock_overlap_requires_noneligible_mask(self):
        with self.assertRaises(ValueError):
            integrate_window([0,10,8,12],[1000]*4,np.ones(3,dtype=bool),0,12)
        r=integrate_window([0,10,8,12],[1000]*4,np.array([True,False,False]),0,8)
        self.assertEqual(r['valid_duration_s'],0)

    def test_zero_endpoint_keeps_discharge_sign(self):
        r=integrate_window([0,10],[0,-2000],np.array([True]),0,10)
        self.assertEqual(r['observed_charge_Ah'],0)
        self.assertAlmostEqual(r['observed_discharge_Ah'],10/3600)

    def test_sparse_current_cannot_be_made_eligible(self):
        with self.assertRaises(ValueError):
            integrate_window([0,3600],[1500,0],np.array([True]),0,3600)

    def test_nonfinite_current_must_not_enter_integral(self):
        with self.assertRaises(ValueError):
            integrate_window([0,1],[1500,np.nan],np.array([True]),0,1)

    def test_integer_duration_across_days(self):
        self.assertEqual(seconds_us('2020-01-01T23:59:59.999999','2020-01-03T00:00:00.000001'),86400000002)


if __name__ == '__main__':
    unittest.main()
