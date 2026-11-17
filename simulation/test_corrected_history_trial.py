import unittest
import numpy as np
from run_corrected_history_trial import waveform


class ActualInputTests(unittest.TestCase):
    def fixture(self):
        data=dict(t_s=np.array([10.,11.,12.]),I_mA=np.array([1499.,1500.,1501.]),
                  mode=np.ones(3,dtype=int),Ns=np.ones(3,dtype=int),
                  current_drive_eligible_interval=np.ones(3,dtype=bool))
        source=dict(row_start=0,row_end_exclusive=3,start_s=10.,end_s=12.,duration_us=2000000)
        return source,data

    def test_native_charge_sign_and_duration(self):
        s,a=self.fixture();result=waveform(s,a)
        np.testing.assert_array_equal(result[:,0],[0.,1.,2.])
        np.testing.assert_array_equal(result[:,1],[-1.499,-1.5,-1.501])
        self.assertAlmostEqual(np.trapezoid(result[:,1],result[:,0]),-3.)

    def test_stale_mode3_row_cannot_be_interpolated(self):
        s,a=self.fixture();a['mode'][0]=3
        with self.assertRaisesRegex(ValueError,'mode/Ns'):waveform(s,a)

    def test_Ns_boundary_cannot_be_interpolated(self):
        s,a=self.fixture();a['Ns'][-1]=2
        with self.assertRaisesRegex(ValueError,'mode/Ns'):waveform(s,a)

    def test_ineligible_interval_refused(self):
        s,a=self.fixture();a['current_drive_eligible_interval'][0]=False
        with self.assertRaisesRegex(ValueError,'Unreviewed'):waveform(s,a)


if __name__=='__main__':unittest.main()
