import copy, tempfile, unittest
from pathlib import Path
import numpy as np
from run_exposure_scenario_smoke import current_window,rest_metadata,sha


class ExposureRuntimeTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.path=self.root/'raw.npz'
        np.savez(self.path,t_s=np.array([10.,11.,12.]),I_mA=np.array([1000.,1500.,2000.]),
            mode=np.array([1,1,1]),Ns=np.array([7,7,7]),current_drive_eligible_interval=np.array([True,True]))
        self.step=dict(kind='eligible_recorded_current',arrays_file='raw.npz',arrays_sha256=sha(self.path),
            row_start=0,row_end_exclusive=3,start_s=10.,end_s=12.,duration_us=2000000)

    def test_native_charge_sign_and_whole_interval_integral(self):
        table,meta=current_window(self.step,self.root)
        np.testing.assert_array_equal(table,[[0,-1],[1,-1.5],[2,-2]])
        self.assertAlmostEqual(meta['model_current_integral_Ah'],-3/3600)
        self.assertAlmostEqual(meta['model_current_integral_Ah'],-meta['native_current_integral_Ah'])

    def test_mask_boundary_and_hash_tamper_rejected(self):
        np.savez(self.path,t_s=[10.,11.,12.],I_mA=[1000.,1500.,2000.],mode=[1,3,1],Ns=[7,7,7],
            current_drive_eligible_interval=np.array([False,False]))
        with self.assertRaises(ValueError):current_window(self.step,self.root)
        self.step['arrays_sha256']=sha(self.path)
        with self.assertRaises(ValueError):current_window(self.step,self.root)

    def test_mode3_stale_current_is_preserved_but_never_interpolated(self):
        np.savez(self.path,mode=np.array([3,3]),I_mA=np.array([1499.625,0.]))
        s=dict(kind='recorded_rest_controller',recorded_rest_current_A=0.,arrays_file='raw.npz',
            arrays_sha256=sha(self.path),row_start=0,row_end_exclusive=2)
        r=rest_metadata(s,self.root)
        self.assertEqual(r['raw_stale_current_mA'],1499.625)
        self.assertFalse(r['raw_current_interpolation_permitted'])

    def test_gap_cannot_silently_inherit_temperature_or_nonzero_current(self):
        s=dict(kind='assumed_exposure',assumed_operation='rest',assumed_current_A=0,assumed_temperature_C=25,assumption_id='A1')
        self.assertEqual(rest_metadata(s,self.root)['operation_source'],'explicit_assumption')
        s['assumed_temperature_C']=40
        with self.assertRaises(ValueError):rest_metadata(s,self.root)
        s['assumed_temperature_C']=25;s['assumed_current_A']=1
        with self.assertRaises(ValueError):rest_metadata(s,self.root)

    def test_duration_change_is_not_hidden_as_rounding(self):
        self.step['duration_us']+=100
        with self.assertRaises(ValueError):current_window(self.step,self.root)


if __name__=='__main__':unittest.main()
