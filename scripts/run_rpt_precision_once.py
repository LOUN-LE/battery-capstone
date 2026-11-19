"""Finite Linux RPT diagnostic job: backup, mail authorized recipient, stop EC2."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/var/lib/battery-capstone/repair_20261102/rpt_precision_20261102')
REPO = Path('/var/lib/battery-capstone/repair_20261102/release1/repo')
PYTHON = '/var/lib/battery-capstone/venv/bin/python'
TOPIC = 'arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102'
DEST = 's3://battery-capstone-backup-000000000000-us-east-1/repair/20261102/rpt_precision/'


def main():
    began = time.time()
    receipt = {'scope': 'RPT diagnostic only; no aging trajectory advancement', 'started_epoch': began}
    try:
        # Fail closed if the old lifetime service is still running.
        old = subprocess.run(['systemctl', 'is-active', 'capstone-singlecell-2d5c17a6f8.service'],
                             capture_output=True, text=True)
        if old.stdout.strip() in ('active', 'activating', 'deactivating'):
            raise RuntimeError('Lifetime worker is not stopped')
        with (BASE/'worker.log').open('wb') as log:
            proc = subprocess.Popen([PYTHON, '-B', str(BASE/'audit_rpt_precision.py'),
                '--input-root', '/var/lib/battery-capstone/results/main_100efc',
                '--output', str(BASE/'out')], cwd=REPO/'simulation',
                env={**__import__('os').environ, 'PYTHONPATH': str(REPO/'simulation'),
                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                stdout=log, stderr=subprocess.STDOUT)
            try:
                receipt['exit_code'] = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                receipt.update(exit_code=124, error='900-second diagnostic watchdog')
    except Exception as exc:
        receipt['error'] = repr(exc)
    finally:
        receipt['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt['elapsed_s'] = time.time()-began
        (BASE/'execution_receipt.json').write_text(json.dumps(receipt, indent=2))
        try:
            subprocess.run(['aws', '--region', 'us-east-1', 's3', 'sync', str(BASE), DEST,
                            '--exclude', '*.tmp', '--only-show-errors'], check=True, timeout=120)
            receipt['backup_completed'] = True
            report_path = BASE/'out/report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            message = {**receipt, 'status': report.get('status'), 'comparison': report.get('comparison'),
                       'sources_preserved': report.get('source_hashes_unchanged'), 'S3': DEST}
            subprocess.run(['aws', '--region', 'us-east-1', 'sns', 'publish', '--topic-arn', TOPIC,
                '--subject', 'Battery: RPT precision audit finished', '--message', json.dumps(message)],
                check=True, timeout=30)
            (BASE/'execution_receipt.json').write_text(json.dumps(receipt, indent=2))
            subprocess.run(['aws', '--region', 'us-east-1', 's3', 'cp',
                str(BASE/'execution_receipt.json'), DEST+'execution_receipt.json', '--only-show-errors'],
                check=True, timeout=30)
        finally:
            subprocess.run(['/usr/sbin/shutdown', '-h', '+1'], check=True, timeout=10)


if __name__ == '__main__':
    main()
