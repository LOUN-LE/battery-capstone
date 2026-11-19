import datetime as dt
import unittest
from generate_aws1000_role_policy import build_policy, BUCKET, TOPIC


class BatchRolePolicyTests(unittest.TestCase):
    launch = dt.datetime(2026,11,15,3,0,tzinfo=dt.timezone.utc)
    instance = 'i-00000000000000000'
    prefix = 'risk-training/20261115/mesh8-1000-v1/'

    def test_only_batch_objects_and_authorized_topic(self):
        doc = build_policy(self.instance, self.prefix, self.launch)
        self.assertEqual(doc['Statement'][0]['Action'], ['s3:GetObject','s3:PutObject'])
        self.assertEqual(doc['Statement'][0]['Resource'], f'arn:aws:s3:::{BUCKET}/{self.prefix}*')
        self.assertEqual(doc['Statement'][1]['Action'], 'sns:Publish')
        self.assertEqual(doc['Statement'][1]['Resource'], TOPIC)

    def test_all_grants_require_same_instance_tls_and_expiry(self):
        doc = build_policy(self.instance, self.prefix, self.launch)
        for statement in doc['Statement']:
            condition = statement['Condition']
            self.assertTrue(condition['ArnEquals']['ec2:SourceInstanceARN'].endswith('/'+self.instance))
            self.assertEqual(condition['Bool'], {'aws:SecureTransport':'true'})
            self.assertEqual(condition['DateLessThan']['aws:CurrentTime'], '2026-11-15T08:00:00Z')
            self.assertFalse(any(k.endswith('IfExists') for k in condition))

    def test_reject_wildcards_root_prefix_and_naive_clock(self):
        for prefix in ['', '*', '/', 'risk-training/*/', 'risk-training/../x/', 'risk-training/']:
            with self.assertRaises(ValueError):
                build_policy(self.instance, prefix, self.launch)
        with self.assertRaises(ValueError):
            build_policy('i-*', self.prefix, self.launch)
        with self.assertRaises(ValueError):
            build_policy(self.instance, self.prefix, self.launch.replace(tzinfo=None))

    def test_multipart_cleanup_is_explicit_and_never_object_deletion(self):
        doc = build_policy(self.instance, self.prefix, self.launch, multipart_cleanup=True)
        self.assertEqual(doc['Statement'][0]['Action'], ['s3:GetObject','s3:PutObject','s3:AbortMultipartUpload'])


if __name__=='__main__': unittest.main()
