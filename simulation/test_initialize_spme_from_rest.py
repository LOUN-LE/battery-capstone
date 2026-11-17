import unittest
from initialize_spme_from_rest import compare_phase


class PhaseComparison(unittest.TestCase):
    def test_absolute_seconds_no_time_stretch(self):
        r=compare_phase([0,10],[3.,4.],[0,5,10,15],[3.,3.5,4.,4.1])
        self.assertEqual(r['voltage_RMSE_mV'],0)
        self.assertEqual(r['duration_difference_s'],-5)
        self.assertEqual(r['observed_sample_coverage_fraction'],.75)

    def test_recorded_time_origin_offset_only(self):
        r=compare_phase([100,110],[3.,4.],[1000,1005,1010],[3.,3.5,4.])
        self.assertEqual(r['voltage_RMSE_mV'],0)

    def test_reverse_time_refused(self):
        with self.assertRaises(ValueError):compare_phase([0,10],[3.,4.],[0,-1],[3.,4.])


if __name__=='__main__':unittest.main()
