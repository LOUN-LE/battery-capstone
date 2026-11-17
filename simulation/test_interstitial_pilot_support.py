"""No-PDE tests for pilot admission, provenance and safe checkpoint boundaries."""
import copy,json,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from interstitial_pilot_support import (validate_contract,rest_batches,voltage_extrema,validate_charge_residuals,
    validate_manifest,commit_checkpoint,candidate_key,native_control_evidence,transition_state_evidence,record_worker_termination)
from runtime_physics_identity import digest
from run_exposure_scenario_smoke import read,write,sha

ROOT=Path(__file__).resolve().parents[2]
W=ROOT/'work/validation_proposal_20261103/latest_spme_review_20261103'
CONTRACT=W/'interstitial_history_pilot_20261105/contract.json'
TEST_TMP=ROOT/'work/interstitial_checkpoint_test_tmp'
TEST_TMP.mkdir(exist_ok=True)

class Values:
    def __init__(self,x):self.entries=np.asarray(x)
class FakeState:
    def __init__(self,current=0.):self.current=current;self.all_ys=[np.array([[1.],[2.]])];self.t=np.array([5.])
    def __getitem__(self,key):return Values([self.current if key=='Current [A]' else 0.])

class PilotTests(unittest.TestCase):
    def test_execution_requires_exact_reviewed_contract(self):
        validate_contract(CONTRACT,1.,executing=False)
        with self.assertRaisesRegex(ValueError,'reviewed'):validate_contract(CONTRACT,1.,executing=True)
        validate_contract(CONTRACT,1.,sha(CONTRACT),True)

    def test_engineering_range_nonfinite_and_bool_rejected(self):
        for value in (0.,.001,10001.,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):validate_contract(CONTRACT,value)
        for value in (.01,1.,10000.):validate_contract(CONTRACT,value)

    def test_rest_batch_boundary_is_explicit(self):
        plan=[{'kind':'eligible_recorded_current'},{'kind':'recorded_rest_controller'},
              {'kind':'eligible_recorded_current'},{'kind':'native_RPT_control','spec':{'operation':'rest'}},
              {'kind':'native_RPT_control','spec':{'operation':'current'}}]
        self.assertEqual(list(rest_batches(plan,max_steps=3)),[(0,2),(2,5)])
        with self.assertRaisesRegex(ValueError,'safe rest'):list(rest_batches(plan,max_steps=1))

    def test_voltage_overruns_recorded_without_clipping(self):
        v=voltage_extrema([2.48,3.4,4.23]);self.assertAlmostEqual(v['below_nominal_V'],.02);self.assertAlmostEqual(v['above_nominal_V'],.03)
        self.assertTrue(v['nominal_limit_violation']);self.assertFalse(voltage_extrema([2.5,4.2])['nominal_limit_violation'])

    def test_solver_residual_guard_is_not_silently_relaxed(self):
        c=read(CONTRACT);validate_charge_residuals(8.17e-6,-8.17e-6,c)
        with self.assertRaisesRegex(ValueError,'residual'):validate_charge_residuals(1.01e-5,0,c)

    def fixture(self,tmp):
        root=Path(tmp);out=root/'work/validation_proposal_20261103/latest_spme_review_20261103/chunk';out.mkdir(parents=True)
        content=dict(actual_parameter_values={'Dint':1e-20},runtime_versions={'python':'test'},source_sha256={})
        identity=dict(schema_version=1,sha256=digest(content),**content)
        report=dict(candidate_key=candidate_key('a'*64,1.,identity),contract_sha256='a'*64,runtime_identity_sha256=identity['sha256'],
            completed_steps=1,source_sha256={})
        plan=[{'kind':'recorded_rest_controller'},{'kind':'native_RPT_control','measurement':{'rpt':0}}]
        journal=[{'step':0}]
        return root,out,identity,report,plan,journal

    def test_atomic_checkpoint_ignores_uncommitted_staging(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root,out,identity,report,plan,journal=self.fixture(tmp)
            receipt=commit_checkpoint(out,FakeState(),report,plan,journal,[],identity)
            cp=Path(receipt['directory']);self.assertTrue(cp.is_dir());self.assertFalse((cp/'manifest.partial.json').exists())
            write(out/'run_receipt.json',{'status':'worker_terminated_at_wall_budget'})
            write(out/'report.json',{'status':'terminated_at_declared_wall_budget'})
            (cp.parent/'step_000002').mkdir();(cp.parent/'step_000002'/'state.pkl').write_bytes(b'incomplete')
            validate_manifest(cp,receipt['manifest_sha256'],report,plan,identity,root)
            self.assertEqual(read(out/'latest_checkpoint.json')['directory'],str(cp))

    def test_different_candidate_or_corrupted_component_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root,out,identity,report,plan,journal=self.fixture(tmp)
            receipt=commit_checkpoint(out,FakeState(),report,plan,journal,[],identity);cp=Path(receipt['directory'])
            write(out/'run_receipt.json',{'status':'worker_finished'});write(out/'report.json',{'status':'failed'})
            other=dict(report,candidate_key=candidate_key('a'*64,2.,identity))
            with self.assertRaisesRegex(ValueError,'different candidate'):validate_manifest(cp,receipt['manifest_sha256'],other,plan,identity,root)
            (cp/'terminal.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'component hash'):validate_manifest(cp,receipt['manifest_sha256'],report,plan,identity,root)

    def test_nonrest_or_incomplete_prefix_cannot_commit(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root,out,identity,report,plan,journal=self.fixture(tmp)
            with self.assertRaisesRegex(ValueError,'zero-current'):commit_checkpoint(out,FakeState(.5),report,plan,journal,[],identity)
            with self.assertRaisesRegex(ValueError,'contiguous'):commit_checkpoint(out,FakeState(),report,plan,[],[],identity)
            plan[0]={'kind':'eligible_recorded_current'}
            with self.assertRaisesRegex(ValueError,'explicit rest'):commit_checkpoint(out,FakeState(),report,plan,journal,[],identity)

    def test_finished_candidate_cannot_replay_old_prefix(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root,out,identity,report,plan,journal=self.fixture(tmp)
            receipt=commit_checkpoint(out,FakeState(),report,plan,journal,[],identity);cp=Path(receipt['directory'])
            write(out/'run_receipt.json',{'status':'worker_finished'});write(out/'report.json',{'status':'interstitial_pilot_complete'})
            with self.assertRaisesRegex(ValueError,'Completed candidate'):validate_manifest(cp,receipt['manifest_sha256'],report,plan,identity,root)

    def test_torn_live_report_cannot_destroy_committed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root,out,identity,report,plan,journal=self.fixture(tmp)
            checkpoint=commit_checkpoint(out,FakeState(),report,plan,journal,[],identity);cp=Path(checkpoint['directory'])
            (out/'report.json').write_text('{"status":',encoding='utf-8')
            receipt=record_worker_termination(out,{'status':'worker_terminated_at_wall_budget','max_wall_s':600})
            self.assertIn('live_report_annotation_error',receipt)
            self.assertEqual(read(out/'run_receipt.json')['status'],'worker_terminated_at_wall_budget')
            validate_manifest(cp,checkpoint['manifest_sha256'],report,plan,identity,root)
            record_worker_termination(out,{'status':'worker_finished','exit_code':0})
            with self.assertRaisesRegex(ValueError,'Completed candidate'):validate_manifest(cp,checkpoint['manifest_sha256'],report,plan,identity,root)

    def test_cv_control_and_native_event_are_verified(self):
        c=read(CONTRACT);spec=dict(operation='voltage',value=4.2,termination=['0.05 A'],duration_s=60000,timed_endpoint=False)
        event='event: abs(Current [A]) < 0.05 [A] [experiment]'
        result=native_control_evidence(spec,[-1.,-.05],[4.2,4.2],100.,event,c)
        self.assertEqual(result['applied_voltage_max_error_V'],0.)
        with self.assertRaisesRegex(ValueError,'applied voltage'):native_control_evidence(spec,[-1.,-.05],[4.1,4.2],100.,event,c)
        with self.assertRaisesRegex(ValueError,'terminal'):native_control_evidence(spec,[-1.,-.1],[4.2,4.2],100.,event,c)
        with self.assertRaisesRegex(ValueError,'watchdog'):native_control_evidence(spec,[-1.,-.05],[4.2,4.2],60000.,'final time',c)
        with self.assertRaisesRegex(ValueError,'Unexpected'):native_control_evidence(spec,[-1.,-.05],[4.2,4.2],100.,'event: Minimum voltage [V]',c)

    def test_controller_algebraic_current_may_change_but_differential_state_cannot(self):
        def model(n):return SimpleNamespace(rhs={'physical':None},y_slices={'physical':[slice(0,2)]},len_rhs=2,len_alg=n)
        previous=SimpleNamespace(all_models=[model(3)],all_ys=[np.array([[1.],[2.],[8.],[9.],[10.]])])
        current=SimpleNamespace(all_models=[model(4)],all_ys=[np.array([[1.],[2.],[8.],[9.],[10.],[-.5]])])
        proof=transition_state_evidence(previous,current)
        self.assertTrue(proof['differential_state_continuity_verified']);self.assertEqual(proof['next_algebraic_state_count'],4)
        current.all_ys[0][0,0]=1.01
        with self.assertRaisesRegex(ValueError,'reset'):transition_state_evidence(previous,current)

if __name__=='__main__':unittest.main()
