import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from diagnose_cutoff_transport import CASES,scale_function,tail_median,validate_fine_reference,run_suite

class CutoffTransportTests(unittest.TestCase):
    def test_exactly_seven_one_at_a_time_cases(self):
        self.assertEqual(len(CASES),7)
        self.assertTrue(all(sum(f!=1 for f in c[1:])<=1 for c in CASES))
        self.assertEqual(CASES[0][1:],(1.,1.,1.))
    def test_baseline_function_identity_is_preserved(self):
        def f(x,y):return x+y
        self.assertIs(scale_function(f,1),f)
    def test_only_function_output_is_scaled(self):
        def f(x,y):return x*y
        self.assertEqual(scale_function(f,.1)(2,5),1)
        self.assertEqual(f(2,5),10)
    def test_constant_parameter_can_be_scaled(self):self.assertEqual(scale_function(5,10),50)
    def test_nonpositive_factor_refused(self):
        with self.assertRaises(ValueError):scale_function(5,0)
    def test_tail_median_preserves_boundary_sensitivity(self):
        t=np.arange(0,301,10);v=np.ones(len(t));v[-2:]=5
        all_rows=tail_median(t,v);inner=tail_median(t,v,exclude_last_rows=2)
        self.assertEqual(all_rows['sample_count'],13)
        self.assertEqual(inner['sample_count'],11)
        self.assertEqual(inner['median'],1)
    def test_invalid_tail_pair_refused(self):
        with self.assertRaises(ValueError):tail_median([0,1],[2])
    def test_metadata_hash_change_requires_identical_physical_inputs(self):
        init=dict(material_parameters_sha256='m',source_C={'sha256':'raw'},initial_state_updates={'n':1,'p':2},
            effective_initial_soc=.369,anchor_policy_id='policy',anchor_policy={'selected_indices':[9,10,11]})
        fine=dict(material_sha256='m',source_C_sha256='raw',initial_state_updates_applied={'n':1,'p':2},
            effective_initial_soc_report_only=.369,anchor_policy_id='policy',selected_anchor_rows=[9,10,11],initialization_sha256='old')
        result=validate_fine_reference(fine,init,'new')
        self.assertTrue(result['physical_inputs_exactly_match'])
        self.assertFalse(result['whole_initialization_file_sha256_matches'])
        init['initial_state_updates']['p']=3
        with self.assertRaises(ValueError):validate_fine_reference(fine,init,'new')
    def test_failed_children_do_not_return_completed_suite(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('diagnose_cutoff_transport.subprocess.run',return_value=SimpleNamespace(returncode=2,stdout='failed')):
                result=run_suite('.', '.', '.',Path(d)/'suite',360)
        self.assertEqual(result['status'],'engineering_screen_incomplete')
        self.assertEqual(result['successful_cases'],0)
        self.assertEqual(result['actual_solve_calls'],0)

if __name__=='__main__':unittest.main()
