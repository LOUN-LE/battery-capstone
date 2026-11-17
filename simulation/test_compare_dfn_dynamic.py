import copy
import unittest

from compare_dfn_dynamic import native_spec, paired_axis_score, select_half_rate_prefix, select_gitt_route, verify_dynamic_control


class DynamicComparisonTests(unittest.TestCase):
    def protocol(self):
        specs = [dict(file='F045', Ns=n) for n in range(10)] + [dict(file='gap', Ns=None)]
        specs += [dict(file='F046', Ns=n) for n in range(6)]
        specs[-1].update(operation='current',value=2.5,is_capacity=True,termination='2.5 V')
        return specs

    def test_full_conditioning_is_required_and_later_exposure_excluded(self):
        specs = self.protocol()
        self.assertEqual(select_half_rate_prefix(specs+[dict(file='F046',Ns=6)]),specs)
        with self.assertRaisesRegex(ValueError,'conditioning'):select_half_rate_prefix(specs[1:])

    def test_observed_duration_must_not_replace_capacity_event(self):
        for key,value in [('conditional_pulse',True),('value',.5),('termination','3 V'),('is_capacity',False)]:
            specs=copy.deepcopy(self.protocol());specs[-1][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Independent'):select_half_rate_prefix(specs)

    def test_axes_are_not_stretched_and_uncovered_tail_is_counted(self):
        score=paired_axis_score([0,1,2],[4,3,2.5],[0,1,2,3],[4,3,2.5,2.4])
        self.assertEqual(score['voltage_RMSE_mV'],0.)
        self.assertEqual(score['uncovered_rows'],1)
        self.assertFalse(score['axes_stretched'])
        with self.assertRaises(ValueError):paired_axis_score([0,0],[4,3],[0,1],[4,3])

    def test_rest_and_independent_event_watchdog_are_distinct(self):
        rest=native_spec(dict(operation='rest',duration=120.))
        event=native_spec(dict(operation='current',value=2.5,duration=50000.,termination='2.5 V'))
        self.assertTrue(rest['timed_endpoint']);self.assertFalse(event['timed_endpoint'])
        self.assertEqual(event['termination'],['2.5 V'])

    def test_full_precision_native_event_label_keeps_exact_target(self):
        target=4.199999809265137
        spec=dict(operation='current',value=-1.5000001192092896,duration=50000.,termination=str(target)+' V')
        contract={'solver_diagnostic_guards':{'applied_voltage_max_error_V':1e-6,'native_event_terminal_residual_V_or_A':1e-6}}
        label='event: Voltage > '+str(target)+' [V] [experiment]'
        evidence=verify_dynamic_control(spec,[-1.5,-1.5],[4.,target],100.,label,contract,{})
        self.assertEqual(evidence['native_terminal_residual'],0.)
        self.assertEqual(evidence['native_event_label_original'],label)
        for invalid in (label.replace('>','<'),label.replace(str(target),'4.2')):
            with self.assertRaisesRegex(ValueError,'exact decoded'):
                verify_dynamic_control(spec,[-1.5,-1.5],[4.,target],100.,invalid,contract,{})
        with self.assertRaisesRegex(ValueError,'watchdog'):
            verify_dynamic_control(spec,[-1.5,-1.5],[4.,target],50000.,'final time',contract,{})

    def test_full_GITT_requires_complete_route_and_conditional_paper_endpoints(self):
        specs=self.protocol()+[dict(file='F046',Ns=n) for n in range(6,10)]+[dict(file='gap',Ns=None)]
        specs += [dict(file='F047',Ns=n) for n in [0,1,2,3]+[4,5,6]*25+[8]]
        for spec in specs:
            if spec['file']=='F047' and spec['Ns']==5:
                spec.update(conditional_pulse=True,paper_charge_target_Ah=.2,duration=288.,value=2.5,
                    termination='2.5 V',pulse_duration_from_observation=False)
        prepared=dict(specifications=specs,pulse_endpoint_policy='paper-200mAh',native_Q_reference_admitted=False)
        self.assertEqual(len(select_gitt_route(prepared)),102)
        altered=copy.deepcopy(prepared)
        next(s for s in altered['specifications'] if s['file']=='F047' and s['Ns']==5)['duration']=300.
        with self.assertRaises(ValueError):select_gitt_route(altered)
        altered=copy.deepcopy(prepared);altered['native_Q_reference_admitted']=True
        with self.assertRaises(ValueError):select_gitt_route(altered)

    def test_paper_pulse_final_time_is_a_declared_endpoint_not_watchdog(self):
        spec=dict(operation='current',value=2.5,duration=288.,termination='2.5 V',conditional_pulse=True)
        contract={'solver_diagnostic_guards':{'applied_voltage_max_error_V':1e-6,'native_event_terminal_residual_V_or_A':1e-6}}
        result=verify_dynamic_control(spec,[2.5,2.5],[3.5,3.4],288.,'final time',contract,{})
        self.assertEqual(result['native_endpoint_verified'],'declared timed endpoint')
        with self.assertRaisesRegex(ValueError,'watchdog'):
            verify_dynamic_control(spec,[2.5,2.5],[3.5,3.4],280.,'final time',contract,{})


if __name__=='__main__':unittest.main()
