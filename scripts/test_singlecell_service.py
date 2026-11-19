"""Offline contract tests: no AWS, no systemd mutations, no battery pickle loads."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import singlecell_service as S


class ServiceTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).resolve().parents[2]/'work/repair_20261102/service_test_fixtures'
        fixtures.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=fixtures)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root/'repo'; self.repo.mkdir()
        (self.repo/'runner.py').write_text('print(1)')
        self.source = self.root/'original'/'trajectory'; self.source.mkdir(parents=True)
        (self.source/'checkpoint.pkl').write_bytes(b'not-a-pickle-must-never-load')
        S.save(self.source/'summary.json', {'elapsed_wall_s':58433.7})
        S.save(self.source/'FAILED.json', {'elapsed_wall_s':58455.2})
        self.archive = self.root/'archive.tar.gz'; self.archive.write_bytes(b'archive')
        receipt = {'passed':True, 'mesh_scale':4,
            'source_checkpoint_sha256':S.digest(self.source/'checkpoint.pkl'),
            'reached_EFC':1620.6, 'failed_transition_passed':True,
            'post_rpt_transition_passed':True, 'source_unchanged':True}
        self.receipt = self.root/'validation.json'; S.save(self.receipt,receipt)
        self.admission = self.root/'admission.json'
        self.data = {'schema':'singlecell-admission-v1','instance_id':'i-00000000000000000',
            'checkpoint_sha256':S.digest(self.source/'checkpoint.pkl'),
            'archive_sha256':S.digest(self.archive), 'source_hashes':{'runner.py':S.digest(self.repo/'runner.py')},
            'validation_receipt':{'path':'validation.json','sha256':S.digest(self.receipt)},
            'cost':{'rate_USD_h':.4032, 'prior_spend_upper_USD':20,
                'planning_USDKRW':2000, 'budget_buffer':1.1,'reserve_KRW':150000,'budget_KRW':600000,
                'rate_verified_utc':'2026-11-02T05:00:00Z','prior_spend_reviewed_utc':'2026-11-02T05:00:00Z'}}
        S.save(self.admission,self.data)
        self.args=SimpleNamespace(deadline_utc='2026-11-02T14:59:00Z', shutdown_reserve_s=600,
            email_interval_s=3600, region='us-east-1',topic_arn='arn:aws:sns:us-east-1:000000000000:capstone-test',
            s3_prefix='s3://battery-capstone-backup-000000000000-us-east-1/repair/20261102/test',
            repo_root=str(self.repo),source_cp=str(self.source/'checkpoint.pkl'),source_archive=str(self.archive),
            output_dir=str(self.root/'new'),admission=str(self.admission),python='/venv/bin/python')
        self.now=dt.datetime(2026,11,2,5,30,tzinfo=dt.timezone.utc)

    def validate(self):
        return S.validate(self.args,self.now)

    def test_one_worker_bounded_by_absolute_soft_deadline(self):
        plan=self.validate()
        self.assertEqual(plan['worker_soft_stop_utc'],'2026-11-02T14:49:00+00:00')
        self.assertEqual(plan['source_elapsed_wall_s'],58455.2)
        self.assertAlmostEqual(plan['worker_max_cumulative_s']-58455.2,33540)
        cmd=S.worker_command(self.args,plan)
        self.assertIn('--migrate-checkpoint',cmd)
        self.assertNotIn('--test-scope',cmd)
        self.assertEqual(cmd[cmd.index('--mesh-scale')+1],'4')

    def test_modified_source_and_forged_validation_refused(self):
        (self.repo/'runner.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source hash changed'):
            self.validate()
        (self.repo/'runner.py').write_text('print(1)')
        receipt=S.read(self.receipt);receipt['post_rpt_transition_passed']=False
        S.save(self.receipt,receipt)
        self.data['validation_receipt']['sha256']=S.digest(self.receipt);S.save(self.admission,self.data)
        with self.assertRaisesRegex(ValueError,'validation is not admitted'):
            self.validate()

    def test_budget_and_stale_cost_refused(self):
        self.data['cost']['prior_spend_upper_USD']=300;S.save(self.admission,self.data)
        with self.assertRaisesRegex(ValueError,'aggregate cost'):
            self.validate()
        self.data['cost']['prior_spend_upper_USD']=20
        self.data['cost']['rate_verified_utc']='2026-10-30T05:00:00Z';S.save(self.admission,self.data)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.validate()

    def test_no_deadline_extension_or_short_backup_reserve(self):
        self.args.deadline_utc='2026-11-03T14:59:00Z'
        with self.assertRaisesRegex(ValueError,'approved today'):
            self.validate()
        self.args.deadline_utc='2026-11-02T14:59:00Z';self.args.shutdown_reserve_s=299
        with self.assertRaisesRegex(ValueError,'shutdown reserve'):
            self.validate()

    def test_independent_shutdown_never_needs_python(self):
        plan=self.validate();units=S.systemd_units(self.args,plan,['--deadline-utc',self.args.deadline_utc])
        stop=units[plan['unit']+'-hardstop.service']
        timer=units[plan['unit']+'-hardstop.timer']
        self.assertIn('ExecStart=/usr/sbin/shutdown -h now',stop)
        self.assertNotIn('python',stop)
        self.assertIn('OnCalendar=2026-11-02 14:59:00 UTC',timer)
        self.assertIn('Persistent=true',timer)
        self.assertNotIn('[Install]',units[plan['unit']+'.service'])

    def test_existing_output_or_source_overlap_refused(self):
        self.args.output_dir=str(self.source/'extension')
        with self.assertRaisesRegex(ValueError,'disjoint'):
            self.validate()
        self.args.output_dir=str(self.root/'new');Path(self.args.output_dir).mkdir()
        (Path(self.args.output_dir)/'existing').write_text('keep')
        with self.assertRaisesRegex(ValueError,'must be empty'):
            self.validate()


if __name__=='__main__':
    unittest.main(verbosity=2)
