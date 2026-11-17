import unittest
from validate_c_gitt import validate_gitt_route,allow_empty,model_point,observation_mode,pulse_charge_status
from native_program_contracts import decode_mb

class GITTContractTests(unittest.TestCase):
    def test_actual_native_rest_enum_and_hour_unit_fixture(self):
        row=dict(ctrl_type=4,lim_nb=1,lim1_type=0,lim1_comp=0,lim1_value_unit=1,lim1_value=1.,lim1_action=0)
        d=decode_mb(row)
        self.assertEqual(d['operation'],'rest');self.assertEqual(d['limits'][0]['value'],3600)
    def test_native_Q0_metadata_remains_unresolved_in_controller_fixture(self):
        row=dict(ctrl_type=0,apply_I_C=0,ctrl1_val=2.5,ctrl1_val_unit=0,ctrl1_val_vs=2,lim_nb=2)
        row.update({'apply_I/C':0,'charge/discharge':1,'lim1_type':5,'lim1_comp':0,'lim1_value_unit':1,
            'lim1_value':200.,'lim1_action':0,'lim1_Q':0,'lim2_type':1,'lim2_comp':1,'lim2_value_unit':0,'lim2_value':2.5,'lim2_action':0})
        d=decode_mb(row,preserve_charge_metadata=True)
        self.assertEqual(d['value'],2.5);self.assertEqual(d['limits'][0]['quantity'],'unresolved_native_charge_reference')
        mode=observation_mode(d,is_capacity=False,observed_duration_s=288.03,pulse_endpoint_policy='recorded-duration')
        self.assertFalse(mode['Q_reference_admitted']);self.assertEqual(mode['termination'],'2.5 V')
    def test_observed25_triplets_without_executed_goto(self):
        validate_gitt_route([0,1,2,3]+[4,5,6]*25+[8])
        with self.assertRaises(ValueError):validate_gitt_route([0,1,2,3]+[4,5,6,7]*25+[8])
    def test_Q_reference_never_promoted_to_absolute_capacity(self):
        d=dict(operation='current',value=2.5,limits=[dict(quantity='unresolved_native_charge_reference',value=200,native_Q_code=0)])
        with self.assertRaises(ValueError):observation_mode(d,is_capacity=False,observed_duration_s=288.)
        mode=observation_mode(d,is_capacity=False,observed_duration_s=288.03,pulse_endpoint_policy='recorded-duration')
        self.assertEqual(mode['duration'],288.03);self.assertFalse(mode['Q_reference_admitted'])
        self.assertNotIn('capacity_target_Ah',mode)
    def test_capacity_cannot_use_observed_duration_or_empty_fallback(self):
        self.assertFalse(allow_empty(dict(is_capacity=True,conditional_pulse=True,allow_initial_event=True)))
        self.assertTrue(allow_empty(dict(is_capacity=False,conditional_pulse=True)))
        d=dict(operation='current',value=2.5,limits=[dict(quantity='unresolved_native_charge_reference')])
        with self.assertRaises(ValueError):observation_mode(d,is_capacity=True,observed_duration_s=288,pulse_endpoint_policy='recorded-duration')
    def test_paper_constant_current_Q_route_is_independent_of_observed_duration(self):
        d=dict(operation='current',value=2.5,limits=[dict(quantity='unresolved_native_charge_reference')])
        mode=observation_mode(d,is_capacity=False,observed_duration_s=43.17,pulse_endpoint_policy='paper-200mAh')
        self.assertEqual(mode['duration'],288)
        self.assertAlmostEqual(2.5*mode['duration']/3600,.2)
        self.assertLess(2.5*144/3600,.2)
        self.assertFalse(mode['pulse_duration_from_observation']);self.assertFalse(mode['Q_reference_admitted'])
    def test_single_recorded_point_is_not_automatically_skippable(self):
        self.assertFalse(allow_empty(dict(is_capacity=False,observed_duration_s=0)))
    def test_native_rest_unit_converted_by_decoder_before_control(self):
        d=dict(operation='rest',limits=[dict(quantity='time_s',value=3600.)])
        self.assertEqual(observation_mode(d,is_capacity=False,observed_duration_s=3599.9)['duration'],3600)
    def test_model_only_interpolation_no_endpoint_extrapolation(self):
        self.assertEqual(model_point([0,1,2],[4,3.9,3.8],1.25),3.875)
        self.assertIsNone(model_point([0,1,2],[4,3.9,3.8],2.00001))
    def test_low_rate_recharge_watchdog_exceeds_normal33800s(self):
        d=dict(operation='current',value=-.5,limits=[dict(quantity='voltage_V',value=4.2)])
        self.assertEqual(observation_mode(d,is_capacity=False,observed_duration_s=34000)['duration'],50000)
    def test_exact_charge_with_voltage_event_is_not_early_charge(self):
        self.assertEqual(pulse_charge_status(.2,.2),dict(paper_charge_target_met=True,voltage_early_cutoff_before_paper_charge=False))
        self.assertTrue(pulse_charge_status(.1,.2)['voltage_early_cutoff_before_paper_charge'])
    def test_nonmonotone_model_refused(self):
        with self.assertRaises(ValueError):model_point([0,1,1],[4,3.9,3.8],.5)

if __name__=='__main__':unittest.main()
