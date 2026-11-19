"""Standard-library-only failure receipt/email. Caller always powers off separately."""
import datetime as dt
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path('/opt/capstone-risk1000')
IMDS = 'http://169.254.169.254/latest/'


def now():
    return dt.datetime.now(dt.timezone.utc)


def load(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def credentials(opener):
    request = urllib.request.Request(IMDS + 'api/token', method='PUT',
        headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
    with opener.open(request, timeout=3) as response:
        token = response.read().decode()
    headers = {'X-aws-ec2-metadata-token': token}
    with opener.open(urllib.request.Request(IMDS + 'meta-data/iam/security-credentials/', headers=headers), timeout=3) as response:
        roles = response.read().decode().strip().splitlines()
    if len(roles) != 1 or not re.fullmatch(r'[A-Za-z0-9+=,.@_-]+', roles[0]):
        raise ValueError('Unexpected instance role list')
    endpoint = IMDS + 'meta-data/iam/security-credentials/' + roles[0]
    with opener.open(urllib.request.Request(endpoint, headers=headers), timeout=3) as response:
        result = json.load(response)
    if result.get('Code') != 'Success' or any(not result.get(k) for k in ('AccessKeyId', 'SecretAccessKey', 'Token')):
        raise ValueError('Instance credentials unavailable')
    if dt.datetime.fromisoformat(result['Expiration'].replace('Z', '+00:00')) <= now():
        raise ValueError('Expired instance credentials')
    return result


def signing_key(secret, date, region, service):
    key = ('AWS4' + secret).encode()
    for value in (date, region, service, 'aws4_request'):
        key = hmac.new(key, value.encode(), hashlib.sha256).digest()
    return key


def signed_headers(method, host, path, body, region, service, creds, stamp, content_type):
    day, timestamp = stamp.strftime('%Y%m%d'), stamp.strftime('%Y%m%dT%H%M%SZ')
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {'host': host, 'content-type': content_type, 'x-amz-date': timestamp,
               'x-amz-content-sha256': payload_hash, 'x-amz-security-token': creds['Token']}
    names = ';'.join(sorted(headers))
    canonical = '\n'.join((method, path, '', ''.join(k + ':' + ' '.join(headers[k].split()) + '\n' for k in sorted(headers)), names, payload_hash))
    scope = '/'.join((day, region, service, 'aws4_request'))
    to_sign = '\n'.join(('AWS4-HMAC-SHA256', timestamp, scope, hashlib.sha256(canonical.encode()).hexdigest()))
    signature = hmac.new(signing_key(creds['SecretAccessKey'], day, region, service), to_sign.encode(), hashlib.sha256).hexdigest()
    headers['Authorization'] = f"AWS4-HMAC-SHA256 Credential={creds['AccessKeyId']}/{scope}, SignedHeaders={names}, Signature={signature}"
    return headers


def aws_request(cfg, service, method, path, body, content_type):
    # Bypass proxies for IMDS and AWS; credentials are freshly fetched each call.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    creds = credentials(opener)
    host = (cfg['bucket'] + '.s3.' if service == 's3' else 'sns.') + cfg['region'] + '.amazonaws.com'
    headers = signed_headers(method, host, path, body, cfg['region'], service, creds, now(), content_type)
    request = urllib.request.Request('https://' + host + path, data=body, headers=headers, method=method)
    with opener.open(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError('AWS request not acknowledged')
        return response.read()


def put(cfg, name, body):
    key = cfg['prefix'] + name
    path = '/' + urllib.parse.quote(key, safe='/-_.~')
    return aws_request(cfg, 's3', 'PUT', path, body, 'application/octet-stream')


def safe_error(error):
    return type(error).__name__ + (':' + str(error.code) if isinstance(error, urllib.error.HTTPError) else '')


def log_tail(root):
    parts = []
    for path in (root / 'worker.log', root / 'bootstrap.log', Path('/var/log/capstone-risk1000-bootstrap.log'), Path('/var/log/cloud-init-output.log')):
        try:
            with path.open('rb') as file:
                file.seek(0, 2)
                file.seek(max(0, file.tell() - 24000))
                value = file.read().decode('utf-8', errors='replace')
            value = re.sub(r'https?://\S+', '[URL REDACTED]', value)
            value = re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b', '[ACCESS KEY REDACTED]', value)
            value = re.sub(r'(?im)^.*(?:secretaccesskey|aws_secret_access_key|x-amz-security-token|aws_session_token|sessiontoken|\btoken\b|authorization).*$','[CREDENTIAL LINE REDACTED]', value)
            parts.append(path.name + '\n' + value)
        except OSError:
            continue
    return ('Emergency failure backup ' + now().isoformat() + '\n\n' + '\n\n'.join(parts)).encode()


def run(root=ROOT):
    cfg = load(root / 'runtime_config.json')
    if (not re.fullmatch(r'[a-z]{2}-[a-z]+-\d', cfg.get('region', ''))
            or not re.fullmatch(r'[a-z0-9][a-z0-9.-]+[a-z0-9]', cfg.get('bucket', ''))
            or not cfg.get('prefix', '').endswith('/') or cfg['prefix'].startswith('/')
            or '..' in cfg['prefix'].split('/') or not cfg.get('topic_arn') or not cfg.get('recipient')):
        raise ValueError('Incomplete scoped runtime configuration')
    completion_path = root / 'completion.json'
    completion = load(completion_path)
    valid = (completion.get('requested_scenarios') == 1000 and isinstance(completion.get('verified_1000_cases'), bool)
             and isinstance(completion.get('success'), bool) and bool(completion.get('finished_utc')))
    if valid and completion['success'] and completion['verified_1000_cases']:
        print('Emergency path: existing success preserved', flush=True)
        return 0
    generated = not valid
    if generated:
        completion = {'finished_utc': now().isoformat(), 'requested_scenarios': 1000, 'verified_1000_cases': False,
            'success': False, 'service_result': os.environ.get('SERVICE_RESULT', 'bootstrap_or_runtime_failure'),
            'emergency_fallback': True, 'shutdown': 'Caller requests poweroff independently', 'errors': []}
        save(completion_path, completion)
    errors = []
    try:
        put(cfg, 'bootstrap_failure.log', log_tail(root))
    except Exception as error:
        errors.append({'phase': 'failure_log_upload', 'type': safe_error(error)})
    delivery = load(root / 'email_delivery.json')
    if not delivery.get('sns_message_id'):
        link = f"https://s3.console.aws.amazon.com/s3/buckets/{cfg['bucket']}?region={cfg['region']}&prefix={urllib.parse.quote(cfg['prefix'], safe='/')}"
        message = 'Capstone1000 batch NEEDS ATTENTION.\nThe emergency fallback ran;1000 completed cases are not verified.\nResults/logs: ' + link + '\nThe server requests shutdown independently.'
        body = urllib.parse.urlencode({'Action': 'Publish', 'Version': '2010-03-31', 'TopicArn': cfg['topic_arn'],
            'Subject': 'Capstone1000 scenarios - NEEDS ATTENTION', 'Message': message}).encode()
        try:
            response = aws_request(cfg, 'sns', 'POST', '/', body, 'application/x-www-form-urlencoded')
            message_id = next((e.text for e in ET.fromstring(response).iter() if e.tag.split('}')[-1] == 'MessageId'), None)
            if not message_id:
                raise ValueError('SNS message receipt missing')
            delivery = {'sns_message_id': message_id, 'sent_utc': now().isoformat(), 'recipient': cfg['recipient'], 'success_reported': False}
            save(root / 'email_delivery.json', delivery)
        except Exception as error:
            errors.append({'phase': 'failure_email', 'type': safe_error(error)})
    if generated:
        completion.update(email_notification=delivery, errors=errors)
        save(completion_path, completion)
        try:
            put(cfg, 'completion.json', completion_path.read_bytes())
        except Exception as error:
            errors.append({'phase': 'failure_receipt_upload', 'type': safe_error(error)})
            save(completion_path, completion)
    print(json.dumps({'event': 'emergency_fallback', 'sns_receipt_present': bool(delivery.get('sns_message_id')), 'errors': errors}), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    try:
        with (ROOT / 'emergency.lock').open('w') as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            raise SystemExit(run())
    except Exception as error:
        print('Emergency fallback error: ' + safe_error(error), flush=True)
        raise SystemExit(1)
