import unittest
from unittest.mock import patch
import numpy as np
from bounded_current_waveform import compress, verified_setpoint


class WaveformTests(unittest.TestCase):
    def test_edges_signed_Ah_and_continuous_error(self):
        t=np.r_[np.arange(200.),199.002]
        current=-1.5+.0006*np.sin(t*2.7);current[0]=-.8;current[-1]=-1.3
        source=np.column_stack([t,current]);saved=source.copy()
        reduced,meta=compress(source,-1.5)
        self.assertTrue(meta['compressed']);self.assertLess(len(reduced),len(source)//2)
        np.testing.assert_array_equal(source,saved)
        np.testing.assert_array_equal(reduced[[0,1,-2,-1]],source[[0,1,-2,-1]])
        self.assertAlmostEqual(np.trapezoid(reduced[:,1],reduced[:,0]),np.trapezoid(current,t),places=11)
        self.assertAlmostEqual(meta['compressed_absolute_Ah'],meta['raw_absolute_Ah'],places=14)
        dense=np.linspace(0,t[-1],30000)
        self.assertLessEqual(np.max(abs(np.interp(dense,reduced[:,0],reduced[:,1])-np.interp(dense,t,current))),.0015)

    def test_sign_duration_and_nonuniform_timestamps_preserved(self):
        t=np.cumsum(np.linspace(.7,1.3,300));current=5+.001*np.cos(t)
        reduced,meta=compress(np.column_stack([t,current]),5.)
        self.assertTrue(meta['compressed']);self.assertGreater(meta['raw_signed_Ah'],0)
        self.assertAlmostEqual(meta['compressed_absolute_Ah'],meta['raw_absolute_Ah'],places=14)
        self.assertEqual(reduced[-1,0]-reduced[0,0],t[-1]-t[0])
        self.assertTrue(set(reduced[:,0]).issubset(set(t)))

    def test_unverified_or_pulsed_input_is_not_compressed(self):
        t=np.arange(200.);current=np.ones(200)*1.5
        raw=np.column_stack([t,current]);reduced,meta=compress(raw,None)
        self.assertFalse(meta['compressed']);np.testing.assert_array_equal(reduced,raw)
        raw[100,1]=5.;reduced,meta=compress(raw,1.5)
        self.assertFalse(meta['compressed']);np.testing.assert_array_equal(reduced,raw)

    def test_reject_bad_clock_and_nonfinite(self):
        for bad in ([[0,1],[0,1]],[[0,1],[1,float('nan')]]):
            with self.assertRaises(ValueError):compress(bad,1.)

    def test_controller_scope_fails_closed(self):
        arrays=dict(mode=np.ones(20),Ns=np.zeros(20),current_drive_eligible_interval=np.ones(20,dtype=bool));s=dict(kind='eligible_recorded_current',row_start=0,row_end_exclusive=20,raw_source_sha256='x')
        r=dict(source_kind='rpt',kind='recorded_mode_block',recorded_mode=1,native_Ns=0)
        self.assertIsNone(verified_setpoint(r,s,arrays,{}))
        r['source_kind']='ageing';r['recorded_mode']=3
        self.assertIsNone(verified_setpoint(r,s,arrays,{}))
        r['recorded_mode']=1;arrays['mode'][0]=3
        self.assertIsNone(verified_setpoint(r,s,arrays,{}))
        r['recorded_mode']=2;arrays['mode'][:]=2
        self.assertIsNone(verified_setpoint(r,s,arrays,{}))

    def test_disabled_interval_or_mixed_Ns_refused(self):
        arrays=dict(mode=np.ones(20),Ns=np.zeros(20),current_drive_eligible_interval=np.ones(20,dtype=bool))
        s=dict(kind='eligible_recorded_current',row_start=0,row_end_exclusive=20,raw_source_sha256='x')
        r=dict(source_kind='ageing',kind='recorded_mode_block',recorded_mode=1,native_Ns=0)
        p={'x':dict(technique='MB',raw_program_rows=[{}],sequence_count=1)}
        with patch('bounded_current_waveform.decode_mb',return_value=dict(operation='current',value=-1.5)):
            self.assertEqual(verified_setpoint(r,s,arrays,p),-1.5)
            arrays['current_drive_eligible_interval'][5]=False
            self.assertIsNone(verified_setpoint(r,s,arrays,p))
            arrays['current_drive_eligible_interval'][:]=True;arrays['Ns'][5]=1
            self.assertIsNone(verified_setpoint(r,s,arrays,p))

    def test_edge_sign_reversal_kept_raw_and_absolute_Ah_exact(self):
        raw=np.column_stack([np.arange(200.),np.full(200,-1.5)]);raw[0,1]=1.5
        result,meta=compress(raw,-1.5)
        np.testing.assert_array_equal(raw,result);self.assertFalse(meta['compressed'])
        self.assertAlmostEqual(meta['raw_absolute_Ah'],(198*1.5+.75)/3600,places=14)

    def test_CP_units_direction_and_relative_current_rejection(self):
        arrays=dict(mode=np.ones(20),Ns=np.zeros(20),current_drive_eligible_interval=np.ones(20,dtype=bool))
        s=dict(kind='eligible_recorded_current',row_start=0,row_end_exclusive=20,raw_source_sha256='x')
        r=dict(source_kind='ageing',kind='recorded_mode_block',recorded_mode=1,native_Ns=0)
        row=dict(Is=-5,Is_unit=0,Is_vs=2);programs={'x':dict(technique='CP',raw_program_rows=[row])}
        self.assertEqual(verified_setpoint(r,s,arrays,programs),5.)
        row.update(Is=1500,Is_unit=1);self.assertEqual(verified_setpoint(r,s,arrays,programs),-1.5)
        row['Is_vs']=1;self.assertIsNone(verified_setpoint(r,s,arrays,programs))
        row.update(Is_vs=2,Is_unit=7);self.assertIsNone(verified_setpoint(r,s,arrays,programs))


if __name__=='__main__':unittest.main()
