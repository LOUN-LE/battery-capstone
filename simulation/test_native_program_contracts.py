import unittest
from native_program_contracts import current_A, decode_mb


class NativePrograms(unittest.TestCase):
    def rest(self):
        return dict(ctrl_type=4, lim_nb=1, lim1_type=0, lim1_comp=0,
                    lim1_value=2., lim1_value_unit=1, lim1_action=0)

    def test_actual_two_hour_rest_unit(self):
        d=decode_mb(self.rest())
        self.assertEqual(d['limits'][0]['value'],7200)
        self.assertFalse(d['measured_history_admissible'])

    def test_current_units(self):
        self.assertEqual(current_A(500,1),.5)
        self.assertEqual(current_A(1.5,0),1.5)
        with self.assertRaises(ValueError): current_A(5,3)

    def test_branch_not_silently_next_step(self):
        r=self.rest();r['lim1_action']=2
        with self.assertRaises(ValueError): decode_mb(r)

    def test_unknown_limit_refused(self):
        r=self.rest();r['lim1_type']=87
        with self.assertRaises(ValueError): decode_mb(r)

    def test_nonfinite_duration_refused(self):
        r=self.rest();r['lim1_value']=float('nan')
        with self.assertRaises(ValueError): decode_mb(r)

    def test_branch_metadata_does_not_admit_execution(self):
        r=self.rest();r.update(lim1_action=1,lim1_seq=3)
        d=decode_mb(r,preserve_branch_metadata=True,sequence_count=4)
        self.assertTrue(d['limits'][0]['branch_execution_review_required'])
        self.assertFalse(d['measured_history_admissible'])
        with self.assertRaises(ValueError): decode_mb(r,preserve_branch_metadata=True,sequence_count=3)

    def test_charge_reference_never_misread_as_mAh(self):
        r=dict(ctrl_type=0,apply_I_C=0,ctrl1_val=5.,ctrl1_val_unit=0,ctrl1_val_vs=2,
               lim_nb=1,lim1_type=5,lim1_comp=0,lim1_value=.15,lim1_value_unit=1,
               lim1_Q=2,lim1_action=0)
        r.update({'apply_I/C':0,'charge/discharge':1})
        with self.assertRaisesRegex(ValueError,'charge-reference'):decode_mb(r)
        d=decode_mb(r,preserve_charge_metadata=True)
        self.assertEqual(d['limits'][0]['quantity'],'unresolved_native_charge_reference')
        self.assertEqual(d['limits'][0]['value'],.15)
