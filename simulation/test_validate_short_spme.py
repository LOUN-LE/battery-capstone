import copy
import json
from pathlib import Path
import tempfile
import unittest
from validate_short_spme import make_protocol,replay_terminal_diagnostic_group,summarize,read_json,completed_screen_case


def fixture():
    parameters=dict(C_only_fit=True,parameter_updates={'Negative electrode active material volume fraction':.66,
        'Positive electrode active material volume fraction':.58,'Initial concentration in negative electrode [mol.m-3]':32450.})
    initialization=dict(effective_initial_soc=.393)
    model=dict(parameter_set='OKane2022',options={'thermal':'isothermal'})
    case=dict(name='fixture',x_points=10,r_points=20,rtol=1e-7)
    return make_protocol(parameters,initialization,model,case)


class ShortSPMeTests(unittest.TestCase):
    def test_worker_exit_failure_cannot_reuse_completed_report(self):
        value=dict(status='short_screen_completed',process_exit_code=0,source_hashes_unchanged=True,
                   copied_RPT_main_state_unchanged=True,Q0_Ah=5.,Qref_Ah=4.9,SOH_percent=98.)
        self.assertTrue(completed_screen_case(value))
        value['process_exit_code']=1
        self.assertFalse(completed_screen_case(value))

    def test_completed_report_must_have_reproducible_soh(self):
        value=dict(status='short_screen_completed',process_exit_code=0,source_hashes_unchanged=True,
                   copied_RPT_main_state_unchanged=True,Q0_Ah=5.,Qref_Ah=4.9,SOH_percent=99.)
        self.assertFalse(completed_screen_case(value))

    def test_native_preparation_and_fixed730mAh(self):
        p=fixture();by={s['phase']:s for s in p['steps']}
        self.assertEqual(by['preparation_initial_rest']['duration_s'],3600)
        self.assertEqual(by['preparation_postCV_rest']['duration_s'],7200)
        self.assertEqual(by['preparation_730mAh_discharge']['value'],1.5)
        for s in p['steps']:
            if s['phase'].startswith('loop') or '730mAh' in s['phase']:
                self.assertAlmostEqual(abs(s['value'])*s['duration_s']/3600,.73)

    def test_diagnostic_group_carries_copy_state_between_all_five_steps(self):
        p=fixture();seen=[]
        def advance(state,s):
            if state is None:state={'n':0}
            seen.append((s['phase'],state['n']));state['n']+=1
            return state,dict(phase=s['phase'],exposure=state['n'])
        main,records,audit=replay_terminal_diagnostic_group(p,None,advance,fingerprint=lambda s:s['n'])
        self.assertEqual(main['n'],15)
        self.assertEqual([n for _,n in seen[-5:]],[15,16,17,18,19])
        self.assertTrue(audit['copied_RPT_main_state_unchanged'])
        self.assertTrue(all(not r['propagated'] and r['branch_propagated'] for r in records[-5:]))

    def test_nonterminal_diagnostic_groups_refused(self):
        p=fixture();p['steps'].append(copy.deepcopy(p['steps'][0]))
        with self.assertRaises(ValueError):replay_terminal_diagnostic_group(p,None,lambda s,p:(s,{}))

    def test_actual_rpt_exposure_is_not_discarded(self):
        p=fixture();self.assertTrue(all(s['role']=='rpt' for s in p['steps'][:5]))
        self.assertTrue(all(s['role']=='diagnostic' for s in p['steps'][-5:]))

    def test_bad_effective_soc_refused(self):
        for value in (float('nan'),True,-.1,1.1):
            with self.assertRaises(ValueError):make_protocol(dict(parameter_updates={},C_only_fit=True),dict(effective_initial_soc=value),{}, {})

    def test_preparation_does_not_enter_cycling_only_EFC(self):
        rows=[dict(phase='initial_actual_RPT_C10',charge_Ah=0.,discharge_Ah=5.),
              dict(phase='preparation_CC',charge_Ah=2.,discharge_Ah=0.),
              dict(phase='loop1_discharge',charge_Ah=0.,discharge_Ah=.73),
              dict(phase='loop1_charge',charge_Ah=.73,discharge_Ah=0.),
              dict(phase='copied_postage_RPT_C10',charge_Ah=0.,discharge_Ah=4.9)]
        r=summarize(rows)
        self.assertAlmostEqual(r['cycling_only_absolute_throughput_EFC_model_fixed_Q0'],.146)
        self.assertAlmostEqual(r['main_absolute_throughput_EFC_including_preparation_model_fixed_Q0'],.346)
        self.assertFalse(r['physical_validation_complete'])

    def test_korean_paths_and_utf8_configuration_are_not_cp949_decoded(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'초기상태.json';p.write_text(json.dumps({'source':'예시/초기보정'},ensure_ascii=False),encoding='utf-8')
            self.assertEqual(read_json(p)['source'],'예시/초기보정')


if __name__=='__main__':unittest.main()
