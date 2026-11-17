import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from recorded_exposure_scenarios import compile_scenario,eligible_windows,validate_policies


def rest_policy(temp=25.):
    return dict(policy_id='explicit-rest',operation='rest',temperature_C=temp,rationale='Declared diagnostic assumption, not measured operation')


class ScenarioCompilerTests(unittest.TestCase):
    def make_input(self,root,*,mask=None):
        raw=root/'raw.npz'
        np.savez(raw,t_s=np.array([0.,1.,2.,3.,4.,5.]),I_mA=np.array([1500.,1500.,0.,0.,0.,0.]),
            mode=np.array([1,1,3,3,3,3]),Ns=np.array([1,1,2,2,2,2]),
            current_drive_eligible_interval=np.array([True,False,False,False,False]) if mask is None else mask)
        source=dict(source_id='raw0',arrays_file='raw.npz',arrays_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            native_sha256='a'*64,rows=6,timestamp_naive='2020-01-01T00:00:00')
        def step(kind,operation,start,end,a,b):
            return dict(kind=kind,operation=operation,source_id='raw0',arrays_file='raw.npz',raw_source_sha256='a'*64,
                source_ref='F0#r0',row_start=a,row_end_exclusive=b,start_s=start,end_s=end,
                absolute_start_naive=f'2020-01-01T00:00:0{start}',absolute_end_naive=f'2020-01-01T00:00:0{end}',
                duration_us=(end-start)*1000000,rest_current_A=0 if operation=='rest' else None,
                raw_current_interpolation_permitted=operation=='recorded_current_drive')
        steps=[step('recorded_mode_block','recorded_current_drive',0,1,0,2),
               step('Ns_transition_interval',None,1,2,1,3),
               step('recorded_mode_block','rest',2,5,2,6),
               step('unobserved_gap',None,5,6,5,6)]
        payload=dict(schema_version=2,data_kind='recorded_exposure_reconstruction',pde_replay_admissible=False,
            physical_validation_complete=False,exposure_integrity=dict(raw_mode_boundaries_preserved=True,
                internal_intervals_accounted=True,duration_reconciliation_passed=True,source_rows_preserved=True,
                assumed_intervals_disclosed=True),cell=dict(cell_id='C',steps=steps,source_files=[source]))
        path=root/'cell_C.json';path.write_text(json.dumps(payload),encoding='utf-8')
        return path,payload

    def test_missing_policy_refuses_instead_of_inserting_rest(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=self.make_input(Path(d))
            with self.assertRaisesRegex(ValueError,'Missing explicit unresolved policy'):
                compile_scenario(path,{},scenario_id='none')

    def test_explicit_assumptions_preserve_recorded_exposure_and_clock(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=self.make_input(Path(d))
            result=compile_scenario(path,{k:rest_policy() for k in ('Ns_transition_interval','unobserved_gap')},scenario_id='rest25')
            self.assertEqual(result['time_ledger_us'],dict(raw_total_duration_us=6000000,
                eligible_recorded_current_us=1000000,assumed_unresolved_us=2000000,recorded_rest_controller_us=3000000))
            self.assertEqual(len(result['assumption_inventory']),2)
            self.assertEqual(result['steps'][0]['row_start'],0)
            self.assertEqual(result['steps'][0]['row_end_exclusive'],2)
            self.assertAlmostEqual(result['steps'][0]['signed_native_current_Ah'],1.5/3600)
            self.assertFalse(result['pde_replay_admissible'])
            self.assertFalse(result['conditional_ageing_input_compatible'])
            self.assertFalse(result['steps'][0]['capacity_target_reinitialised'])
            self.assertNotIn('assumed_temperature_C',result['steps'][2])

    def test_noneligible_current_hole_requires_separate_policy(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=self.make_input(Path(d),mask=np.zeros(5,dtype=bool))
            p={k:rest_policy() for k in ('Ns_transition_interval','unobserved_gap')}
            with self.assertRaisesRegex(ValueError,'noneligible_current_sample_interval'):
                compile_scenario(path,p,scenario_id='hole')
            p['noneligible_current_sample_interval']=rest_policy(10)
            result=compile_scenario(path,p,scenario_id='hole')
            self.assertEqual(result['assumed_unresolved_by_kind_us']['noneligible_current_sample_interval'],1000000)

    def test_partial_interval_is_unresolved_not_interpolated(self):
        result=eligible_windows([0,10],[0,2000],np.array([True]),row_start=0,row_end_exclusive=2,start_s=2,end_s=8)
        self.assertEqual(result,[])

    def test_pulse_waveform_keeps_original_rows_and_signed_integral(self):
        result=eligible_windows([0,1,2],[1500,-1500,1500],np.ones(2,dtype=bool),row_start=0,row_end_exclusive=3,start_s=0,end_s=2)
        self.assertEqual(result[0]['row_end_exclusive'],3)
        self.assertEqual(result[0]['eligible_interval_count'],2)
        self.assertEqual(result[0]['signed_native_current_Ah'],0)

    def test_scenario_temperature_changes_no_time_or_raw_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=self.make_input(Path(d));results=[]
            for temp in (10,25,40):
                results.append(compile_scenario(path,{k:rest_policy(temp) for k in ('Ns_transition_interval','unobserved_gap')},scenario_id=str(temp)))
            self.assertTrue(all(r['time_ledger_us']==results[0]['time_ledger_us'] for r in results))
            self.assertEqual(len({r['raw_reconstruction_sha256'] for r in results}),1)
            self.assertEqual(len({r['policies_sha256'] for r in results}),3)

    def test_source_hash_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,_=self.make_input(root);(root/'raw.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Raw array hash changed'):
                compile_scenario(path,{},scenario_id='tamper')

    def test_mask_crossing_rest_boundary_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path,_=self.make_input(Path(d),mask=np.array([True,True,False,False,False]))
            with self.assertRaisesRegex(ValueError,'mask crosses'):
                compile_scenario(path,{},scenario_id='badmask')

    def test_legacy_schema_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'old.json';path.write_text('{"schema_version":1}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'raw reconstruction schema2'):
                compile_scenario(path,{},scenario_id='legacy')

    def test_unsafe_or_incomplete_policy_refused(self):
        with self.assertRaises(ValueError):validate_policies({'gap':dict(operation='rest',temperature_C=25)})
        with self.assertRaises(ValueError):validate_policies({'gap':rest_policy(float('nan'))})
        with self.assertRaises(ValueError):validate_policies({'gap':dict(rest_policy(),current_A=1)})
        with self.assertRaises(ValueError):validate_policies({'gap':dict(rest_policy(),capacity_target_mAh=730)})

    def test_dense_mask_cannot_cover_sparse_interval(self):
        with self.assertRaises(ValueError):
            eligible_windows([0,3600],[1500,0],np.array([True]),row_start=0,row_end_exclusive=2,start_s=0,end_s=3600)

    def test_explicit_nonzero_current_policy_is_disclosed(self):
        policy=dict(policy_id='load',rationale='Sensitivity only',operation='constant_current',current_A=-.1,temperature_C=40)
        self.assertEqual(validate_policies({'gap':policy})['gap']['current_A'],-.1)


if __name__=='__main__':unittest.main()
