import unittest
import numpy as np
from diagnose_low_rate_recharge import compare_axis,raw_charge_Ah


class LowRateRechargeTests(unittest.TestCase):
    def test_absolute_axis_does_not_stretch_endpoints(self):
        score,mask,error=compare_axis([0,1,2],[4,3.5,3],[0,1,2,3],[4,3.5,3,2.5])
        self.assertEqual(score['observed_sample_coverage_fraction'],.75)
        self.assertEqual(score['model_minus_observed_endpoint'],-1)
        self.assertFalse(mask[-1]);self.assertTrue(np.isnan(error[-1]))
        self.assertEqual(score['voltage_RMSE_mV'],0)

    def test_recorded_charge_uses_actual_Ah_no_capacity_normalisation(self):
        q=raw_charge_Ah([100,1900,3700],[500,500,500])
        np.testing.assert_allclose(q,[0,.25,.5])

    def test_nonmonotone_axes_and_current_clock_refused(self):
        with self.assertRaises(ValueError):compare_axis([0,0,2],[4,3.5,3],[0,1],[4,3.5])
        with self.assertRaises(ValueError):raw_charge_Ah([1,0],[500,500])


if __name__=='__main__':unittest.main()
