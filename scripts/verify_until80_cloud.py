"""Offline receipt/control tests. Never imports physics, loads pickle, or uses AWS."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('until80_cloud', HERE/'until80_cloud.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.source = self.home/'original'; self.source.mkdir()
        self.cp = self.source/'checkpoint.pkl'; self.cp.write_bytes(b'opaque trusted checkpoint fixture; never unpickled')
        C.save(self.source/'summary.json', {'elapsed_wall_s':58433.699568415})
        C.save(self.source/'FAILED.json', {'elapsed_wall_s':58455.26863223})
        self.receipt = self.make_receipt()
        self.receipt_path = self.home/'admission.json'; C.save(self.receipt_path, self.receipt)
        self.args = SimpleNamespace(admission_receipt=str(self.receipt_path), source_cp=str(self.cp),
            numfactor=.1, max_cumulative_wall_s=C.MAX_WALL_S, worker_wall_s=72000,
            max_efc=1000000, root=str(self.home/'new-cloud-run'), repo=str(self.home/'repo'),
            python=str(self.home/'venv'/'bin'/'python'))

    def tearDown(self):
        self.tmp.cleanup()

    def ref(self, name, data):
        path=self.home/name; C.save(path,data)
        return {'path':str(path),'sha256':C.digest(path)}

    def make_receipt(self):
        replay={'original_failure_reproduced':True,'original_files_unchanged':True,
                'worker_exit_code':0,'threshold_unchanged':1e-10}
        rep={'source_state_and_ledger_unchanged':True,'reference':{'ok':True},
             'existing_integral_check_pass':True,'direct_integral_relative_difference':1e-13}
        aging={'status':'complete','required_readouts_available':True,
               'original_source_files_unchanged':True,'chemistry_verified':False,
               'states':{n:{'source_state_and_ledger_unchanged':True,
                          'reference_repeats':[copy.deepcopy(rep),copy.deepcopy(rep)],
                          'repeat_comparison':{'max_minus_min_Ah':0.0}}
                         for n in ['initial','legacy100','final']}}
        precision={'original_files_unchanged':True,'worker_exit_code':0,
                   'worker_status':'not_reproduced_within_cycle_bound','tolerance_factor':.1}
        result={'completed_cycles':3,'source_solution_unchanged_in_memory':True,
                'solver_precision_control':{'connection_gate':1e-10,'physics_changed':False,
                    'connection_gate_changed':False,'factor':.1},
                'event_time_correspondence':[{'same_baseline_cycle':True}]}
        def mesh(scale, soh):
            return {'status':'efc_target','identity':{'cell':4,'current_A':3.2,'mesh_scale':scale},
                    'ledger':{'efc':100.0},'diagnostics':[{'efc':100.0,'SOH_percent':soh}]}
        return {'schema':'capstone-until80-admission-v1','source_checkpoint_sha256':C.digest(self.cp),
                'selected_numerical_factor':.1,
                'replay_reports':[self.ref('replay_a.json',replay),self.ref('replay_b.json',replay)],
                'aging_report':self.ref('aging.json',aging),
                'precision_report':self.ref('precision.json',precision),
                'precision_worker_result':self.ref('precision_worker.json',result),
                'mesh4_summary':self.ref('mesh4.json',mesh(4,99.791)),
                'mesh8_summary':self.ref('mesh8.json',mesh(8,99.790)),
                'cost_admission':{'budget_KRW':600000,'reserve_KRW':150000,
                    'planning_USDKRW':2000,'budget_buffer':1.10,'prior_spend_upper_USD':45.45,
                    'observed_billed_USD':8.,'unbilled_estimate_USD':2.,'rate_USD_h':.4032,
                    'reviewed_utc':C.stamp()}}

    def update_receipt(self):
        C.save(self.receipt_path,self.receipt)

    def change_evidence(self, key, mutate):
        ref=self.receipt[key]; data=C.read(ref['path']); mutate(data)
        C.save(ref['path'],data); ref['sha256']=C.digest(ref['path']); self.update_receipt()

    def test_valid_receipt_and_conservative_budget(self):
        admission=C.validate_admission(self.args)
        self.assertAlmostEqual(admission['budget']['planning_total_upper_KRW'],548035.44,places=5)
        self.assertEqual(admission['source_cumulative_wall_s'],58455.26863223)
        self.assertEqual(admission['maximum_new_wall_s'],C.MAX_WALL_S-58455.26863223)
        self.assertFalse(admission['chemistry_validated'])
        self.assertFalse(admission['budget']['FX_is_live_quote'])
        self.assertFalse(admission['budget']['is_AWS_account_hard_spend_cap'])

    def test_failed_precision_blocks_production(self):
        self.change_evidence('precision_report',lambda d:d.update(worker_status='original_connection_failure_reproduced'))
        with self.assertRaisesRegex(ValueError,'precision control did not pass'):
            C.validate_admission(self.args)

    def test_insufficient_cycles_blocks_production(self):
        self.change_evidence('precision_worker_result',lambda d:d.update(completed_cycles=2))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_changed_gate_blocks_production(self):
        self.change_evidence('precision_worker_result',lambda d:d['solver_precision_control'].update(connection_gate=1e-8))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_missing_matching_transition_blocks_production(self):
        self.change_evidence('precision_worker_result',lambda d:d.update(event_time_correspondence=[]))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_mutated_rpt_source_blocks_production(self):
        self.change_evidence('aging_report',lambda d:d['states']['final'].update(source_state_and_ledger_unchanged=False))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_missing_readouts_blocks_production(self):
        self.change_evidence('aging_report',lambda d:d.update(required_readouts_available=False))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_mesh_precheck_boundary(self):
        self.change_evidence('mesh8_summary',lambda d:d['diagnostics'][0].update(SOH_percent=99.5))
        with self.assertRaisesRegex(ValueError,'0.1 percent-point'):
            C.validate_admission(self.args)

    def test_incomplete_mesh_cannot_admit(self):
        self.change_evidence('mesh8_summary',lambda d:d.update(status='max_cycles'))
        with self.assertRaises(ValueError): C.validate_admission(self.args)

    def test_evidence_tamper_rejected(self):
        C.save(self.receipt['precision_report']['path'],{'status':'fabricated'})
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            C.validate_admission(self.args)

    def test_cost_over_prior_allowance_rejected(self):
        self.receipt['cost_admission'].update(observed_billed_USD=35,unbilled_estimate_USD=11)
        self.update_receipt()
        with self.assertRaisesRegex(ValueError,'exceeds prior allowance'):
            C.validate_admission(self.args)

    def test_cost_review_stale_rejected(self):
        self.receipt['cost_admission']['reviewed_utc']='2020-01-01T00:00:00+00:00';self.update_receipt()
        with self.assertRaisesRegex(ValueError,'24 hours'):
            C.validate_admission(self.args)

    def test_unsupported_source_rejected(self):
        self.cp.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError,'checkpoint does not match'):
            C.validate_admission(self.args)

    def test_plan_constructs_no_runtime_output(self):
        original={p:C.digest(p) for p in self.source.iterdir()}
        with mock.patch.object(C.subprocess,'run',side_effect=AssertionError('No system commands in plan')):
            admission=C.validate_admission(self.args);config=C.build_config(self.args,admission)
            units=C.unit_files(config,Path('until80_cloud.py'))
            command=C.worker_command(config,False)
        self.assertFalse(Path(self.args.root).exists())
        self.assertEqual(original,{p:C.digest(p) for p in self.source.iterdir()})
        self.assertIn('venv',command[0]);self.assertIn('--migrate-checkpoint',command)
        self.assertIn('--deadline-utc',command)
        self.assertTrue(any('Persistent=true' in t for t in units.values()))
        self.assertTrue(any('Restart=no' in t for t in units.values()))
        self.assertTrue(any('--boot-guard' in t for t in units.values()))
        self.assertFalse(config['S3_automatic_upload'])
        self.assertFalse(config['SOH80_arrival_estimate_provided'])

    def test_only_healthy_rollover_can_restart(self):
        report={'status':'rollover_checkpointed','terminal':False,'checkpoint_commit_id':10,
                'history_commit_id':'a'*64,'last_valid_EFC':2000.,'cumulative_wall_s':80000.,
                'last_SOH_measurement_EFC':2000.,'last_SOH_percent':98.}
        C.validate_rollover(75,report)
        for code in [0,1,124]:
            with self.assertRaises(ValueError): C.validate_rollover(code,report)
        old=copy.deepcopy(report);old['checkpoint_commit_id']=10
        with self.assertRaises(ValueError): C.validate_rollover(75,report,old)
        report['terminal']=True
        with self.assertRaises(ValueError): C.validate_rollover(75,report)

    def test_old_outcome_does_not_override_latest_commit(self):
        root=self.home/'trajectory';root.mkdir()
        C.save(root/'summary.json',{'status':'running','last_valid_EFC':2500})
        C.save(root/'outcome.json',{'status':'rollover_checkpointed','last_valid_EFC':2000})
        policy=SimpleNamespace(validate_checkpoint_report=mock.Mock(return_value=True))
        report=C.latest_committed_report(root,policy)
        self.assertEqual(report['last_valid_EFC'],2500)
        policy.validate_checkpoint_report.assert_called_once()

    def guard_fixture(self):
        root=self.home/'guard';root.mkdir()
        config={'root':str(root),'unit':'test-until80','deadline_utc':(C.now()-C.dt.timedelta(seconds=1)).isoformat()}
        C.save(root/'cloud_state.json',{'status':'running','terminal':False,
               'server_shutdown_lease':True,'boot_id':'old-boot'})
        return config,root

    def test_terminal_recovery_boot_does_not_shutdown(self):
        config,root=self.guard_fixture()
        C.save(root/'cloud_state.json',{'status':'eol_observed','terminal':True,'server_shutdown_lease':True})
        with mock.patch.object(C,'linux_root'),mock.patch.object(C,'halt') as halt:
            result=C.guard_action('boot-guard',config)
        self.assertEqual(result['status'],'already_terminal_recovery_allowed');halt.assert_not_called()

    def test_deadline_shutdown_independent_of_systemctl_failure(self):
        config,root=self.guard_fixture()
        with mock.patch.object(C,'linux_root'),mock.patch.object(C,'halt') as halt,\
             mock.patch.object(C.subprocess,'run',side_effect=subprocess.TimeoutExpired('systemctl',15)):
            C.guard_action('deadline-fire',config)
        halt.assert_called_once();self.assertEqual(C.read(root/'cloud_state.json')['status'],'global_time_censored')

    def test_unclean_boot_marks_failed_then_halts(self):
        config,root=self.guard_fixture();original=Path.read_text
        def reader(path,*args,**kwargs):
            return 'new-boot' if path.name=='boot_id' else original(path,*args,**kwargs)
        with mock.patch.object(C,'linux_root'),mock.patch.object(C,'halt') as halt,\
             mock.patch.object(Path,'read_text',reader):
            C.guard_action('boot-guard',config)
        halt.assert_called_once();self.assertEqual(C.read(root/'cloud_state.json')['status'],'unclean_boot_failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json-out');args=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'physics_or_pickle_executed':False,'AWS_or_systemd_or_shutdown_executed':False}
    if args.json_out:
        path=Path(args.json_out);path.parent.mkdir(parents=True,exist_ok=True);C.save(path,report)
    print(json.dumps(report));return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
