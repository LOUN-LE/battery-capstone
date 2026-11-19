"""Offline tests only. The AWS example strings below are public test credentials."""
import contextlib
import datetime as dt
import hashlib
import hmac
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import aws1000_emergency as emergency


class Response(io.BytesIO):
    status = 200


class EmergencyTests(unittest.TestCase):
    cfg = {'region': 'us-east-1', 'bucket': 'test-bucket', 'prefix': 'batch/test/',
           'topic_arn': 'arn:aws:sns:us-east-1:123456789012:test', 'recipient': 'self@example.com'}

    def test_published_aws_signing_vector(self):
        # boto/botocore tests/unit/auth/aws4_testsuite/get-vanilla/get-vanilla.sts and .authz
        # https://github.com/boto/botocore/tree/develop/tests/unit/auth/aws4_testsuite/get-vanilla
        key = emergency.signing_key('wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY', '20150830', 'us-east-1', 'service')
        to_sign = 'AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/service/aws4_request\nbb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63'
        self.assertEqual(hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest(),
                         '5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31')

    def test_http_imdsv2_refresh_signed_put_and_sns(self):
        requests, generation = [], [0]
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                if request.full_url.endswith('api/token'):
                    generation[0] += 1
                    return Response(b'imds-token')
                if request.full_url.endswith('iam/security-credentials/'):
                    return Response(b'role-name')
                if request.full_url.endswith('iam/security-credentials/role-name'):
                    return Response(json.dumps({'Code': 'Success', 'AccessKeyId': 'EXAMPLE', 'SecretAccessKey': 'secret',
                        'Token': 'session-' + str(generation[0]), 'Expiration': '2099-01-01T00:00:00Z'}).encode())
                return Response(b'')
        with patch.object(emergency.urllib.request, 'build_opener', return_value=Opener()):
            emergency.put(self.cfg, 'space name.txt', b'payload')
            emergency.aws_request(self.cfg, 'sns', 'POST', '/', b'Action=Publish', 'application/x-www-form-urlencoded')
        self.assertEqual(generation[0], 2)
        aws = [r for r in requests if r.full_url.startswith('https:')]
        self.assertEqual(aws[0].full_url, 'https://test-bucket.s3.us-east-1.amazonaws.com/batch/test/space%20name.txt')
        for i, request in enumerate(aws, 1):
            headers = {k.lower(): v for k, v in request.header_items()}
            self.assertEqual(headers['x-amz-security-token'], 'session-' + str(i))
            self.assertIn('x-amz-security-token', headers['authorization'])
            self.assertEqual(headers['x-amz-content-sha256'], hashlib.sha256(request.data).hexdigest())
        self.assertEqual(aws[0].method, 'PUT')
        self.assertEqual(aws[1].method, 'POST')

    def test_failure_receipt_mail_once_and_redaction(self):
        calls = []
        def fake(cfg, service, method, path, body, content_type):
            calls.append((service, path, body))
            return b'<PublishResponse><PublishResult><MessageId>message-1</MessageId></PublishResult></PublishResponse>' if service == 'sns' else b''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emergency.save(root / 'runtime_config.json', self.cfg)
            (root / 'worker.log').write_text('failed https://example.com/?X-Amz-Signature=hidden\nSecretAccessKey=hidden-secret\nToken=hidden-session', encoding='utf-8')
            with patch.object(emergency, 'aws_request', side_effect=fake), contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(emergency.run(root), 0)
                original = (root / 'completion.json').read_bytes()
                self.assertEqual(emergency.run(root), 0)
            self.assertEqual(original, (root / 'completion.json').read_bytes())
            self.assertEqual(sum(s == 'sns' for s, _, _ in calls), 1)
            completion = emergency.load(root / 'completion.json')
            self.assertFalse(completion['success'])
            self.assertEqual(completion['email_notification']['sns_message_id'], 'message-1')
            self.assertTrue(all(path.startswith('/batch/test/') for service, path, _ in calls if service == 's3'))
            log = next(body for _, path, body in calls if path.endswith('bootstrap_failure.log'))
            for secret in (b'hidden-secret', b'hidden-session', b'X-Amz-Signature'):
                self.assertNotIn(secret, log)
                self.assertNotIn(secret.decode(), stdout.getvalue())

    def test_valid_success_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emergency.save(root / 'runtime_config.json', self.cfg)
            emergency.save(root / 'completion.json', {'requested_scenarios': 1000, 'verified_1000_cases': True,
                'success': True, 'finished_utc': '2026-11-15T00:00:00Z'})
            before = (root / 'completion.json').read_bytes()
            with patch.object(emergency, 'aws_request') as request, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(emergency.run(root), 0)
                request.assert_not_called()
            self.assertEqual(before, (root / 'completion.json').read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
