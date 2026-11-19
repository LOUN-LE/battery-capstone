"""Offline network mocks for cloud delivery; never creates AWS resources."""
import base64
import contextlib
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(r'<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo')
sys.path.insert(0, str(REPO))
import aws1000_ops as ops
from simulation import risk_dataset1000 as dataset


class FakeS3:
    def __init__(self, corrupt=False, failing_name=None):
        self.objects = {}
        self.corrupt = corrupt
        self.failing_name = failing_name

    def put_object(self, **kwargs):
        if self.failing_name and kwargs['Key'].endswith(self.failing_name):
            raise RuntimeError('simulated upload outage')
        expected = base64.b64encode(hashlib.sha256(kwargs['Body']).digest()).decode()
        assert kwargs['ChecksumSHA256'] == expected
        assert kwargs['Metadata']['sha256'] == hashlib.sha256(kwargs['Body']).hexdigest()
        self.objects[kwargs['Key']] = kwargs['Body']
        return {'ResponseMetadata': {'HTTPStatusCode': 200},
                'ChecksumSHA256': 'wrong' if self.corrupt else expected, 'ETag': 'mock-etag'}


class FakeSNS:
    def __init__(self): self.messages=[]
    def publish(self, **kwargs):
        self.messages.append(kwargs)
        return {'MessageId':'offline-mock-message'}


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root/'release'
        self.out = self.base/'results'
        self.out.mkdir(parents=True)
        self.patch_base = patch.multiple(ops, BASE=self.base, OUT=self.out, CONFIG=self.root/'runtime_config.json')
        self.patch_base.start()
        self.cfg = {'bucket':'offline-bucket','prefix':'unit/batch/', 'region':'us-east-1',
                    'topic_arn':'arn:aws:sns:us-east-1:000000000000:offline', 'recipient':'authorized-self'}
        self.s3 = FakeS3()
        self.sns = FakeSNS()
        (self.base/'release_manifest.json').write_text('{"files":{}}')
        (self.base/'requirements.txt').write_text('offline')
        (self.root/'worker.log').write_text('offline worker log')

    def tearDown(self):
        self.patch_base.stop()
        self.temporary.cleanup()

    def put_json(self, path, obj): ops.save(path, obj)

    def test_upload_validates_returned_checksum(self):
        record = ops.upload(self.s3, self.cfg, 'data.bin', b'actualbytes')
        self.assertEqual(record['sha256'], hashlib.sha256(b'actualbytes').hexdigest())
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            ops.upload(FakeS3(corrupt=True), self.cfg, 'data.bin', b'actualbytes')

    def test_checkpoint_excludes_running_cases_and_retains_failed_diagnostics(self):
        for i, locked in [(1,True),(2,False)]:
            case=self.out/f'risk1000_{i:04d}'
            case.mkdir()
            self.put_json(case/'result.json', {'summary':{'status':'failed','error':'test failure'}})
            if locked: (case/'running.lock').write_text('owned')
        with contextlib.redirect_stdout(io.StringIO()):
            count=ops.checkpoint(self.cfg,self.s3)
        self.assertEqual(count,1)
        self.assertIn('unit/batch/cases/risk1000_0002/result.json',self.s3.objects)
        self.assertFalse(any('risk1000_0001' in key for key in self.s3.objects))

    def test_checkpoint_rejects_mismatched_final_trace(self):
        case=self.out/'risk1000_0001';case.mkdir()
        (case/'trace.csv').write_text('changed')
        self.put_json(case/'result.json',{'summary':{'status':'completed'},'trace_sha256':'not_matching'})
        with self.assertRaisesRegex(ValueError,'checksum'):
            ops.checkpoint(self.cfg,self.s3)

    def finish_fixture(self, completed, returncode=0, upload_failure=None):
        for name in ['training_dataset.csv','scenario_summary.csv']:
            (self.out/name).write_text('offline fixture\n')
        self.put_json(self.out/'dataset_report.json',{'all_1000_completed':completed})
        verification={'verification_passed':completed,'expected_count':1000,
                      'status_counts':{'completed':1000 if completed else 999,'pending':0 if completed else 1}}
        def verify():
            self.put_json(self.out/'cloud_verification.json',verification)
            return verification
        self.s3.failing_name=upload_failure
        with patch.object(ops,'clients',return_value=(self.cfg,self.s3,self.sns)), \
             patch.object(ops,'verify',side_effect=verify), \
             patch.object(ops,'checkpoint',return_value=1000 if completed else 999), \
             patch.object(ops.subprocess,'run',return_value=subprocess.CompletedProcess([],returncode,'partial output','')) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            ops.finish()
            call_count=run.call_count
        return json.loads((self.root/'completion.json').read_text()),call_count

    def test_partial_finalizer_exit1_uploads_archive_but_never_success_email(self):
        result,calls=self.finish_fixture(False,returncode=1)
        self.assertEqual(calls,1)
        self.assertFalse(result['success'])
        self.assertFalse(result['verified_1000_cases'])
        self.assertEqual(result['errors'],[])
        self.assertIn('unit/batch/risk1000_full_results.tar.gz',self.s3.objects)
        self.assertIn('NEEDS ATTENTION',self.sns.messages[0]['Subject'])

    def test_broken_finalizer_still_packages_partial_files(self):
        result,_=self.finish_fixture(False,returncode=2)
        self.assertFalse(result['success'])
        self.assertTrue(any(e['phase']=='finalize_and_verify' for e in result['errors']))
        self.assertIn('unit/batch/risk1000_full_results.tar.gz',self.s3.objects)

    def test_completed_report_preserves_original_run_and_success_requires_uploads(self):
        result,calls=self.finish_fixture(True)
        self.assertEqual(calls,0)
        self.assertTrue(result['success'])
        self.assertIn('SUCCESS',self.sns.messages[0]['Subject'])

    def test_failed_required_archive_upload_prevents_success(self):
        result,_=self.finish_fixture(True,upload_failure='risk1000_full_results.tar.gz')
        self.assertFalse(result['success'])
        self.assertIn('NEEDS ATTENTION',self.sns.messages[0]['Subject'])

    def write_pending_tables(self):
        design=dataset.scenarios()
        splits=dataset.split_assignments(design)
        summaries=[];training=[]
        for scenario in design:
            row=asdict(scenario)
            summaries.append({**row,'status':'pending'})
            training.append({'scenario_id':scenario.scenario_id,
                'dataset_split':splits[scenario.scenario_id]['dataset_split'],
                **{key:row[key] for key in dataset.FEATURES},
                'status':'pending','training_eligible':'false','indicator_type':'','screening_priority':''})
        self.write_csv(self.out/'scenario_summary.csv',summaries)
        self.write_csv(self.out/'training_dataset.csv',training)
        return summaries,training

    def write_csv(self,path,rows):
        with path.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_1000_pending_rows_are_not_1000_completed(self):
        self.write_pending_tables()
        with patch.object(ops,'release_check',return_value='offlinehash'):
            result=ops.verify()
        self.assertFalse(result['verification_passed'])
        self.assertEqual(result['status_counts']['pending'],1000)
        self.assertEqual(result['training_eligible'],0)

    def test_999_rows_rejected_even_with_valid_ids(self):
        _,training=self.write_pending_tables()
        self.write_csv(self.out/'training_dataset.csv',training[:-1])
        with self.assertRaisesRegex(ValueError,'identities'):
            ops.verify()

    def test_nonfinite_input_does_not_bypass_numeric_comparison(self):
        summaries,_=self.write_pending_tables()
        summaries[0]['initial_soc']='nan'
        self.write_csv(self.out/'scenario_summary.csv',summaries)
        with self.assertRaisesRegex(ValueError,'condition mismatch'):
            ops.verify()

    def test_incomplete_case_cannot_acquire_training_target(self):
        _,training=self.write_pending_tables()
        training[0].update(training_eligible='true',screening_priority='low',indicator_type='none')
        self.write_csv(self.out/'training_dataset.csv',training)
        with self.assertRaisesRegex(ValueError,'Incomplete case'):
            ops.verify()


if __name__=='__main__': unittest.main()
