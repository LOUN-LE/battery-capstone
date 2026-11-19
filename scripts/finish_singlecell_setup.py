"""Finish already authorized setup from actual validation receipts, then exit.

This is a finite setup process, not an LLM or a monitoring chat. It starts no
physics worker until the original Linux validation and provenance checks pass.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/var/lib/battery-capstone/repair_20261102')
REPO=BASE/'release1/repo'
PYTHON='/var/lib/battery-capstone/venv/bin/python'
TOPIC='arn:aws:sns:us-east-1:000000000000:capstone-cell4-progress-20261102'
BUCKET='s3://battery-capstone-backup-000000000000-us-east-1/repair/20261102/'
SOURCE=Path('/var/lib/battery-capstone/results/lifetime_cell4_3p2A_20261028_v2/trajectory/checkpoint.pkl')
ARCHIVE=Path('/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def run(args,timeout=120):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def mail(event,data):
    run(['aws','--region','us-east-1','sns','publish','--topic-arn',TOPIC,
         '--subject','Battery cell4: '+event,'--message',json.dumps(data),'--no-cli-pager'])

def main():
    try:
        expected=read(BASE/'expected_release_sources.json')
        for rel,sha in expected.items():
            if digest(REPO/rel)!=sha:raise RuntimeError('Release source changed: '+rel)
        end=time.monotonic()+1200
        receipt=BASE/'a2_validation/admission_receipt.json'
        while not receipt.exists():
            if time.monotonic()>=end:raise RuntimeError('Actual validation did not finish within setup bound')
            state=run(['systemctl','show','capstone-a2-validation-20261102','--property=ActiveState','--value']).stdout.strip()
            if state not in ('active','activating') and not receipt.exists():raise RuntimeError('Validation stopped without a receipt: '+state)
            time.sleep(5)
        result=read(receipt)
        if result.get('passed') is not True:raise RuntimeError('Actual validation failed: '+json.dumps(result))
        cause=read(BASE/'a1_forensics/cause_summary.json')
        hook=read(BASE/'a1_forensics/failing_join.json')['recent_solver_step_hooks'][-1]
        mapping=max(float(x['max_rel_diff']) for x in hook['mapping_change'].values())
        initialized=max(float(x['max_rel_diff']) for x in hook['ic_change'].values())
        returned=max(float(x['max_rel_diff']) for x in hook['first_output_change'].values())
        if not (all(cause.get(k) is True for k in ('reproduced','complete','source_unchanged','state_layout_equal'))
                and mapping<=1e-10 and initialized<=1e-10 and returned>1e-10 and cause['jump']>1e-10):
            raise RuntimeError('Forensic cause has not established valid initial transfer and failing first output')
        save(BASE/'numerical_cause_receipt.json',{'mapping_max_relative':mapping,'initialization_max_relative':initialized,
            'first_returned_output_max_relative':returned,'legacy_physical_output_jump':cause['jump'],
            'physical_transfer_limit_unchanged':1e-10,'initial_mapping_roundoff_is_not_bitwise_zero':True,
            'note':'The versioned check observes actual integrator initial physical state; raw output stays unchanged'})
        admission={'schema':'singlecell-admission-v1','instance_id':'i-00000000000000000',
            'checkpoint_sha256':digest(SOURCE),'archive_sha256':digest(ARCHIVE),
            'source_hashes':expected,'validation_receipt':{'path':str(receipt),'sha256':digest(receipt)},
            'cost':{'rate_USD_h':0.4032,'prior_spend_upper_USD':20.,'planning_USDKRW':2000.,
                'budget_buffer':1.10,'reserve_KRW':150000,'budget_KRW':600000,
                'rate_verified_utc':'2026-11-02T03:36:40Z','prior_spend_reviewed_utc':'2026-11-02T03:36:40Z',
                'observed_usage_before_credits_USD':8.2290925757,
                'prior_upper_note':'20USD includes observed usage before credits plus conservative setup/unbilled/storage allowance; reserve is additional'}}
        path=BASE/'singlecell_admission.json';save(path,admission)
        # Preserve the source version and actual admission evidence before launch.
        for source,suffix in ((BASE/'a1_forensics','a1_forensics'),(BASE/'a2_validation','a2_validation'),(REPO,'final_source')):
            run(['aws','--region','us-east-1','s3','sync',str(source),BUCKET+suffix+'/',
                 '--exclude','__pycache__/*','--exclude','*.tmp','--only-show-errors'],timeout=240)
        run(['aws','--region','us-east-1','s3','cp',str(path),BUCKET+'singlecell_admission.json','--only-show-errors'])
        out=BASE/'cell4_until80_v2'
        args=[PYTHON,'-B',str(REPO/'scripts/singlecell_service.py'),'--repo-root',str(REPO),
            '--python',PYTHON,'--source-cp',str(SOURCE),'--source-archive',str(ARCHIVE),
            '--output-dir',str(out),'--admission',str(path),'--deadline-utc','2026-11-02T14:59:00Z',
            '--topic-arn',TOPIC,'--s3-prefix',BUCKET+'singlecell/','--email-interval-s','3600','--install']
        installed=run(args,timeout=120)
        save(BASE/'singlecell_install_receipt.json',json.loads(installed.stdout))
        # The temporary setup stop is cancelled only after the durable production
        # hard-stop timer and detached service have been verified by the installer.
        run(['systemctl','stop','capstone-setup-stop-20261102.timer'])
        deadline=time.monotonic()+120
        while not (out/'summary.json').exists():
            if time.monotonic()>=deadline:raise RuntimeError('Detached worker did not create a committed initial checkpoint')
            time.sleep(2)
        st=read(out/'summary.json')
        data={'setup_status':'detached_worker_started','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'worker_summary':st,'hard_stop_utc':'2026-11-02T14:59:00Z','soft_stop_utc':'2026-11-02T14:49:00Z',
            'SOH80_arrival_estimate':'not established; deadline is a resource limit',
            'validation_receipt_sha256':digest(receipt)}
        save(BASE/'setup_complete.json',data)
        run(['aws','--region','us-east-1','s3','cp',str(BASE/'setup_complete.json'),BUCKET+'setup_complete.json','--only-show-errors'])
        print(json.dumps(data),flush=True)
    except BaseException as exc:
        data={'setup_status':'failed','error':repr(exc),'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        save(BASE/'setup_failure.json',data)
        try:mail('setup_failed',data)
        finally:run(['/usr/sbin/shutdown','-h','+1'])
        raise

if __name__=='__main__':main()
