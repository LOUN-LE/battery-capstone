"""Gate contract tests; the positive fixture is NOT measured physical evidence."""
import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from empirical_validation import MATCH_FIELDS
from lifetime_restart_gate import assess, sha


class RestartGateTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.manifest={'schema_version':1,'admission_route':'exact_measured_history_replay'}
        def save(name,data):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(data),encoding='utf-8');return {'path':name,'sha256':sha(p)}
        self.save=save
        self.manifest['proposal']=save('proposal.json',{'fixture_only':True})
        self.criteria=json.loads((Path(__file__).parents[1]/'configs/empirical_acceptance.json').read_text())
        self.criteria['source']['sha256']=self.manifest['proposal']['sha256']
        self.manifest['criteria']=save('criteria.json',self.criteria)
        self.manifest['candidate']=save('candidate.json',{'fixture':'C-only'})
        self.sources={}
        for name in ('lifetime_restart_gate.py','physical_campaign_inputs.py','score_empirical_validation.py'):
            p=self.root/'repo/simulation'/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(Path(__file__).with_name(name).read_bytes());self.sources[str(p.relative_to(self.root))]=sha(p)
        self.manifest['code_sources']=save('code_sources.json',self.sources)
        self.binding={'candidate_sha256':self.manifest['candidate']['sha256'],'code_sources':self.sources}
        self.software=dict(all_pass=True,source_hashes_stable=True,frozen_six_unchanged=True,
            tested_code_sources=self.sources,total_tests=1,suites=[dict(exit_code=0,test_count=1)])
        self.manifest['software']=save('software.json',self.software)
        metadata={k:k for k in MATCH_FIELDS};metadata.update(axis_column='EFC',axis_unit='EFC')
        self.items=[];self.cases={};self.runs={};self.campaigns={}
        for cell,condition in (('D','25C'),('A','10C')):
            campaign=dict(data_kind='measured',cell_id=cell,condition_id=condition)
            self.campaigns[cell]=campaign;cref=save(cell+'/campaign.json',campaign)
            series=[dict(rpt_id='0',EFC=0,Qref_Ah=5.),dict(rpt_id='1',EFC=100,Qref_Ah=4.5)]
            run=dict(self.binding,evaluation_axis_policy='native_ageing_throughput_at_RPT',data_kind='measured',status='measured_campaign_complete',campaign_sha256=cref['sha256'],measured=series,predicted=copy.deepcopy(series))
            self.runs[cell]=run
            case=dict(cell_id=cell,experiment_id='fixture_only',condition_id=condition,axis_column='EFC',
                measured_metadata=metadata,predicted_metadata=metadata.copy(),
                measured=[dict(EFC=0.,SOH_percent=100.),dict(EFC=100.,SOH_percent=90.)],
                predicted=[dict(EFC=0.,SOH_percent=100.),dict(EFC=100.,SOH_percent=90.)],
                model=dict(id='fixture',kind='physical',artifact_path='candidate.json',artifact_sha256=self.manifest['candidate']['sha256'],
                    fit_cell_ids=['C'],fit_condition_ids=['25C'],frozen_before_evaluation=True,retuned_on_validation=False,heldout_initialization='RPT0_Q0_only'))
            if cell=='A':case['heldout_condition_axis']='temperature'
            self.cases[cell]=case
            self.items.append(dict(campaign=cref,run=save(cell+'/run.json',run),case=save(cell+'/case.json',case)))
        self.manifest['validation']=save('validation.json',self.items)
        declaration=dict(self.binding,SOH_difference_limit_pp=.0001,state_comparison_atol=1e-10)
        dref=save('declaration.json',declaration)
        self.numerics=dict(self.binding,declaration=dref,scope='full_validation_exposure',completed_EFC=100,
            pairs=[dict(coarse_config={'mesh':4},fine_config={'mesh':8},EFC=[0,100],coarse_SOH_percent=[100,90],fine_SOH_percent=[100,90.00001])])
        self.manifest['numerics']=save('numerics.json',self.numerics)
        payload=save('checkpoint_payload.json',{'fixture_state':[1,2]})
        self.checkpoint=dict(self.binding,declaration=dref,checkpoint_payload=payload,scope='current_candidate_actual_stop_resume',fresh_initialization=True,
            saved_vs_loaded=dict(left=[1.,2.],right=[1.,2.]),uninterrupted_vs_resumed=dict(left=[1.,2.],right=[1.,2.]),
            Q0_before_Ah=5.,Q0_after_Ah=5.,virtual_RPT_main_state_before=[1.,2.],virtual_RPT_main_state_after=[1.,2.])
        self.manifest['checkpoint']=save('checkpoint.json',self.checkpoint)

    def evaluate(self,patched_admission=True):
        def admitted(c,root):
            return dict(c,_targets_rows=[dict(rpt_id='0',Qref_Ah=5.),dict(rpt_id='1',Qref_Ah=4.5)])
        if patched_admission:
            with patch('lifetime_restart_gate.validate_campaign',side_effect=admitted), patch('lifetime_restart_gate.native_efc_coordinates',return_value={'0':0.,'1':100.}):return assess(self.manifest,self.root)
        return assess(self.manifest,self.root)

    def replace_run(self,cell):
        index=0 if cell=='D' else 1
        self.items[index]['run']=self.save(cell+'/run.json',self.runs[cell])
        self.items[index]['case']=self.save(cell+'/case.json',self.cases[cell])
        self.manifest['validation']=self.save('validation.json',self.items)

    def test_positive_gate_contract_fixture_never_claims_observed80(self):
        r=self.evaluate();self.assertTrue(r['lifetime_restart_admitted'],r['reasons'])
        self.assertFalse(r['SOH80_validation_complete']);self.assertFalse(r['AWS_started'])

    def test_unadmitted_mock_campaign_fails_real_source_validator(self):
        self.assertFalse(self.evaluate(False)['lifetime_restart_admitted'])

    def test_hash_tamper_and_empty_evidence_fail_closed(self):
        (self.root/'candidate.json').write_text('{}')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])
        self.assertFalse(assess({'schema_version':1},self.root)['lifetime_restart_admitted'])

    def test_conditional_history_cannot_pass_even_with_perfect_scores(self):
        c=self.campaigns['D'];c['data_kind']='diagnostic_exposure_scenario'
        self.items[0]['campaign']=self.save('D/campaign.json',c)
        self.manifest['validation']=self.save('validation.json',self.items)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_short_screen_cannot_cover_100EFC_validation(self):
        self.numerics['scope']='short_screen'
        self.manifest['numerics']=self.save('numerics.json',self.numerics)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_declared_numerical_pass_flag_does_not_hide_actual_difference(self):
        self.numerics['passed']=True;self.numerics['pairs'][0]['fine_SOH_percent'][1]=91
        self.manifest['numerics']=self.save('numerics.json',self.numerics)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_criteria_relaxation_refused(self):
        self.criteria['basic']['SOH_RMSE_pp_max']=10
        self.manifest['criteria']=self.save('criteria.json',self.criteria)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_perfect_summary_cannot_hide_changed_native_capacity(self):
        self.runs['D']['measured'][1]['Qref_Ah']=4.9;self.replace_run('D')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_native_EFC_warp_rejected_even_when_case_and_run_agree(self):
        self.runs['D']['measured'][1]['EFC']=200.
        self.runs['D']['predicted'][1]['EFC']=200.
        self.cases['D']['measured'][1]['EFC']=200.
        self.cases['D']['predicted'][1]['EFC']=200.
        self.replace_run('D')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_prediction_RPT_axis_cannot_be_shifted_to_improve_score(self):
        self.runs['D']['predicted'][1]['EFC']=200.
        self.cases['D']['predicted'][1]['EFC']=200.
        self.replace_run('D')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_one_bad_cell_blocks_whole_gate(self):
        self.runs['A']['predicted'][1]['Qref_Ah']=4.7
        self.cases['A']['predicted'][1]['SOH_percent']=94.;self.replace_run('A')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_old_candidate_or_changed_Q0_checkpoint_cannot_pass(self):
        self.checkpoint['candidate_sha256']='0'*64
        self.manifest['checkpoint']=self.save('checkpoint.json',self.checkpoint)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])
        self.checkpoint['candidate_sha256']=self.binding['candidate_sha256'];self.checkpoint['Q0_after_Ah']=4.9
        self.manifest['checkpoint']=self.save('checkpoint.json',self.checkpoint)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_missing_stage_and_heldout_fit_leakage_rejected(self):
        self.cases['A']['model']['fit_cell_ids']=['C','A'];self.replace_run('A')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])

    def test_current_code_change_and_changed_virtual_main_state_rejected(self):
        (self.root/'repo/simulation/score_empirical_validation.py').write_text('changed')
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])
        self.checkpoint['virtual_RPT_main_state_after']=[1.,3.]
        self.manifest['checkpoint']=self.save('checkpoint.json',self.checkpoint)
        self.assertFalse(self.evaluate()['lifetime_restart_admitted'])


if __name__=='__main__':unittest.main()
