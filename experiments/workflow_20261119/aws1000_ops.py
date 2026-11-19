"""EC2 role-authenticated checkpointing, verification and final delivery.

No console session or presigned URL is used after bootstrap. The instance role
grant is restricted to this instance, this batch prefix, one SNS topic and time.
"""
from __future__ import annotations
import concurrent.futures
import base64
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
CONFIG = BASE.parent / 'runtime_config.json'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default

def metadata():
    url='http://169.254.169.254/latest/'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request(url+'api/token',method='PUT',headers={'X-aws-ec2-metadata-token-ttl-seconds':'60'})
    with opener.open(req,timeout=5) as r:
        token=r.read().decode()
    req=urllib.request.Request(url+'dynamic/instance-identity/document',headers={'X-aws-ec2-metadata-token':token})
    with opener.open(req,timeout=5) as r:
        doc=json.load(r)
    return {k:doc[k] for k in ('instanceId','instanceType','region','availabilityZone','architecture')}

def clients():
    import boto3
    from botocore.config import Config
    cfg=load(CONFIG)
    options=Config(connect_timeout=10,read_timeout=90,retries={'mode':'standard','max_attempts':4})
    return cfg,boto3.client('s3',region_name=cfg['region'],config=options),boto3.client('sns',region_name=cfg['region'],config=options)

def upload(s3,cfg,name,data):
    # Single PutObject: the scoped policy needs no multipart/list/delete actions.
    if isinstance(data,Path):
        data=data.read_bytes()
    if isinstance(data,str):
        data=data.encode('utf-8')
    if len(data)>=4*1024**3:
        raise ValueError('Single-part artifact unexpectedly exceeds bounded size')
    checksum=base64.b64encode(hashlib.sha256(data).digest()).decode()
    result=s3.put_object(Bucket=cfg['bucket'],Key=cfg['prefix']+name,Body=data,
                         ChecksumSHA256=checksum,Metadata={'sha256':hashlib.sha256(data).hexdigest()})
    if result['ResponseMetadata']['HTTPStatusCode']!=200:
        raise RuntimeError('Upload not acknowledged')
    if result.get('ChecksumSHA256')!=checksum:
        raise RuntimeError('S3 checksum acknowledgment differs')
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'etag':result.get('ETag'),'s3_checksum_sha256':result['ChecksumSHA256']}

def release_check():
    manifest=load(BASE/'release_manifest.json')
    for name,expected in manifest['files'].items():
        if digest(BASE/name)!=expected:
            raise ValueError('Release checksum mismatch: '+name)
    return digest(BASE/'release_manifest.json')

def before():
    cfg,s3,_=clients()
    OUT.mkdir(exist_ok=True)
    stamp=now()
    receipt={'start_utc':stamp,'ec2':metadata(),'cpu_count':os.cpu_count(),'python':sys.version,
             'packages':{n:importlib.metadata.version(n) for n in ('pybamm','pybammsolvers','numpy','scipy','casadi','boto3')},
             'release_manifest_sha256':release_check(),'max_workers':8,'requested_scenarios':1000,
             'physics_timeout_s':9000,'service_timeout_s':10800,'force_stop_after_boot_s':18000,
             'detached_service':'capstone-risk1000.service','checkpoint_interval_s':300,
             'hard_stop_timer':subprocess.check_output(['systemctl','show','capstone-risk1000-hard-stop.timer','--property=ActiveState','--property=NextElapseUSecMonotonic'],text=True),
             'credential_mode':'EC2 instance role; automatically refreshed by SDK',
             'policy_expiry_utc':cfg['policy_expiry_utc']}
    if receipt['cpu_count']<8:
        raise ValueError('Expected at least8vCPU')
    save(OUT/'aws_execution_receipt.json',receipt)
    # IAM propagation may lag launch; keep retrying only within a bounded window.
    deadline=time.monotonic()+300
    while True:
        try:
            upload(s3,cfg,'started.json',json.dumps(receipt,indent=2))
            if s3.get_object(Bucket=cfg['bucket'],Key=cfg['prefix']+'started.json')['Body'].read()!=json.dumps(receipt,indent=2).encode():
                raise ValueError('Role read/write preflight mismatch')
            break
        except Exception as e:
            if time.monotonic()>deadline:
                raise
            print('Waiting for scoped role policy:',type(e).__name__,flush=True)
            time.sleep(10)
    print(json.dumps({'event':'role_auth_read_write_verified','instance':receipt['ec2']['instanceId'],'time':now()}),flush=True)

def checkpoint(cfg,s3):
    stored=load(BASE.parent/'uploaded_cases.json',{})
    ready=[]
    for result_path in sorted(OUT.glob('risk1000_*/result.json')):
        case=result_path.parent
        if (case/'running.lock').exists():
            continue
        result=load(result_path)
        trace=case/'trace.csv'
        checksum=digest(result_path)
        if stored.get(case.name)==checksum:
            continue
        if result['summary']['status']!='failed':
            if not trace.exists() or digest(trace)!=result.get('trace_sha256'):
                raise ValueError('Finalized case trace checksum mismatch: '+case.name)
        ready.append((case,result_path,trace,checksum,result['summary']['status']))
    def send(item):
        case,result_path,trace,checksum,status=item
        if status!='failed':
            upload(s3,cfg,'cases/'+case.name+'/trace.csv',trace)
        upload(s3,cfg,'cases/'+case.name+'/result.json',result_path)
        return case.name,checksum
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,checksum in pool.map(send,ready):
            stored[name]=checksum
            save(BASE.parent/'uploaded_cases.json',stored)
    for name in ('progress.json','partial_summary.csv','manifest.json','dataset_report.json','report.json'):
        path=OUT/name
        if path.exists():
            upload(s3,cfg,'checkpoint/'+name,path)
    state={'timestamp_utc':now(),'uploaded_finalized_cases':len(stored),'progress':load(OUT/'progress.json',{})}
    save(BASE.parent/'checkpoint_status.json',state)
    upload(s3,cfg,'checkpoint_status.json',json.dumps(state,indent=2))
    print(json.dumps({'event':'checkpoint','uploaded_cases':len(stored),'time':now()}),flush=True)
    return len(stored)

def monitor():
    cfg,s3,_=clients()
    while True:
        time.sleep(300)
        try:
            checkpoint(cfg,s3)
        except Exception as e:
            print(json.dumps({'event':'checkpoint_retry_next_cycle','error_type':type(e).__name__,'time':now()}),flush=True)

def verify():
    from dataclasses import asdict
    from simulation.risk_dataset1000 import scenarios, execution_provenance, classify, FEATURES, split_assignments
    from simulation.risk_scenarios import content_hash, persistent_onset
    design=scenarios()
    expected={s.scenario_id:asdict(s) for s in design}
    source=execution_provenance(design)
    splits=split_assignments(design)
    with (OUT/'training_dataset.csv').open(encoding='utf-8-sig',newline='') as h:
        training=list(csv.DictReader(h))
    if len(training)!=1000 or {r['scenario_id'] for r in training}!=set(expected):
        raise ValueError('Training table lost scenario identities')
    training_map={r['scenario_id']:r for r in training}
    with (OUT/'scenario_summary.csv').open(encoding='utf-8-sig',newline='') as h:
        rows=list(csv.DictReader(h))
    if len(rows)!=1000 or {r['scenario_id'] for r in rows}!=set(expected):
        raise ValueError('Expected exactly1000 unique intended scenario rows')
    counts={k:sum(r['status']==k for r in rows) for k in ('completed','censored','failed','pending')}
    traces=0
    for r in rows:
        for k,value in expected[r['scenario_id']].items():
            if k!='scenario_id' and (not math.isfinite(float(r[k])) or abs(float(r[k])-value)>1e-9):
                raise ValueError('Scenario condition mismatch')
        tr=training_map[r['scenario_id']]
        if tr['dataset_split']!=splits[r['scenario_id']]['dataset_split'] or tr['status']!=r['status']:
            raise ValueError('Training split/status mismatch')
        for key in FEATURES:
            if not math.isfinite(float(tr[key])) or abs(float(tr[key])-expected[r['scenario_id']][key])>1e-9:
                raise ValueError('Training feature mismatch')
        if r['status']!='completed':
            if tr['training_eligible'].lower()!='false' or tr['screening_priority'] or tr['indicator_type']:
                raise ValueError('Incomplete case incorrectly has a training target')
            continue
        case=OUT/r['scenario_id']
        result=load(case/'result.json')
        identity=content_hash({'scenario':expected[r['scenario_id']],'provenance':source})
        if result['identity_sha256']!=identity or result['provenance']!=source or result['summary']['status']!='completed':
            raise ValueError('Case identity/provenance/status mismatch')
        if digest(case/'trace.csv')!=result['trace_sha256']:
            raise ValueError('Final case hash mismatch')
        with (case/'trace.csv').open(encoding='utf-8-sig',newline='') as h:
            values=list(csv.DictReader(h))
        if len(values)!=int(r['sample_count']) or any(v['scenario_id']!=r['scenario_id'] for v in values):
            raise ValueError('Trace sample count/identity mismatch')
        times=[float(v['time_s']) for v in values]
        temps=[float(v['temperature_C']) for v in values]
        currents=[float(v['plating_net_current_A_m2']) for v in values]
        if not all(math.isfinite(v) for v in times+temps+currents) or any(b<a for a,b in zip(times,times[1:])):
            raise ValueError('Nonfinite or reversed trace')
        onset=persistent_onset(times,currents,expected[r['scenario_id']]['plating_threshold_A_m2'],expected[r['scenario_id']]['plating_persistence_s'])
        if (onset is not None)!=result['summary']['plating_event'] or abs(max(temps)-result['summary']['maximum_temperature_C'])>1e-8:
            raise ValueError('Recomputed output metrics disagree')
        label=classify(result['summary'])
        if tr['training_eligible'].lower()!='true' or tr['indicator_type']!=label['indicator_type'] or tr['screening_priority']!=label['screening_priority']:
            raise ValueError('Training target mismatch')
        if r.get('fire_label') or r.get('thermal_runaway_label') or r.get('physical_risk_validated','').lower()!='false':
            raise ValueError('Unexpected actual fire validation claim')
        traces+=int(r['sample_count'])
    if 'maximum_temperature_C' in training[0] or 'plating_event' in training[0]:
        raise ValueError('Outcome measurements must not be supplied as model input features')
    good=sum(str(r.get('training_eligible','')).lower()=='true' for r in training)
    passed=counts['completed']==1000 and good==1000
    report={'verification_passed':passed,'expected_count':1000,'status_counts':counts,'training_eligible':good,
            'trace_rows':traces,'source_manifest_sha256':release_check(),
            'final_csv_sha256':digest(OUT/'training_dataset.csv'),'finished_verification_utc':now()}
    save(OUT/'cloud_verification.json',report)
    return report

def package():
    summary='''CAPSTONE 1000 SCENARIOS\nThese are1000 unique SPMeT simulations, not1000 measured fires.\nUse training_dataset.csv for inputs and proxy-screening labels; split is included.\nlow/medium/high is an engineering screening policy, not validated physical safety.\nMeasured NIST ignition experiments are separate and not duplicated into this table.\nRead dataset_report.json and design_plan.json for definitions and eligibility.\n'''
    (OUT/'README.txt').write_text(summary,encoding='utf-8')
    share=BASE/'risk1000_csv_for_sharing.zip'
    allowed=('training_dataset.csv','features.csv','labels.csv','scenario_summary.csv','dataset_report.json',
             'cloud_verification.json','README.txt','design_plan.json')
    with zipfile.ZipFile(share,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in allowed:
            p=OUT/name
            if p.exists():
                archive.write(p,name)
    archive_path=BASE/'risk1000_full_results.tar.gz'
    with tarfile.open(archive_path,'w:gz',compresslevel=4) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.endswith(('.tmp','.lock')):
                archive.add(p,arcname='results/'+p.relative_to(OUT).as_posix(),recursive=False)
        for name in ('release_manifest.json','requirements.txt'):
            archive.add(BASE/name,arcname=name)
        if (BASE/'finalizer.log').exists():
            archive.add(BASE/'finalizer.log',arcname='finalizer.log')
    return share,archive_path

def finish():
    cfg,s3,sns=clients()
    completion={'finished_utc':now(),'service_result':os.environ.get('SERVICE_RESULT','unknown'),
                'requested_scenarios':1000,'verified_1000_cases':False,'uploads':{},'errors':[],
                'shutdown':'poweroff after finalization; independent5hourforce-stop remains armed'}
    try:
        command=[sys.executable,'-m','simulation.risk_dataset1000','--output',str(OUT),'--workers','8','--finalize-only']
        prior=load(OUT/'dataset_report.json',{})
        if prior.get('all_1000_completed'):
            completion['finalizer_exit_code']=0
            completion['finalization_mode']='verify_existing_complete_outputs'
        else:
            p=subprocess.run(command,cwd=BASE,timeout=240,capture_output=True,text=True)
            completion['finalizer_exit_code']=p.returncode
            (BASE/'finalizer.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
            if p.returncode not in (0,1) or not (OUT/'dataset_report.json').exists():
                raise RuntimeError('Dataset finalizer failed; raw cases still preserved')
        verification=verify()
        completion['verification']=verification
        completion['verified_1000_cases']=verification['verification_passed']
    except Exception as e:
        completion['errors'].append({'phase':'finalize_and_verify','type':type(e).__name__,'detail':str(e)[:400]})
    try:
        completion['checkpointed_cases']=checkpoint(cfg,s3)
    except Exception as e:
        completion['errors'].append({'phase':'final_checkpoint','type':type(e).__name__})
    try:
        share,archive=package()
        items=[('risk1000_csv_for_sharing.zip',share),('risk1000_full_results.tar.gz',archive)]
        for name in ('training_dataset.csv','scenario_summary.csv','dataset_report.json','cloud_verification.json'):
            if (OUT/name).exists():
                items.append((name,OUT/name))
        if (BASE.parent/'worker.log').exists():
            items.append(('worker.log',BASE.parent/'worker.log'))
        for name,path in items:
            try:
                completion['uploads'][name]=upload(s3,cfg,name,path)
            except Exception as e:
                completion['errors'].append({'phase':'upload','file':name,'type':type(e).__name__})
    except Exception as e:
        completion['errors'].append({'phase':'package','type':type(e).__name__})
    required={'risk1000_csv_for_sharing.zip','risk1000_full_results.tar.gz','training_dataset.csv','cloud_verification.json','worker.log'}
    completion['success']=completion['verified_1000_cases'] and required.issubset(completion['uploads']) and not completion['errors']
    completion['finished_utc']=now()
    save(BASE.parent/'completion.json',completion)
    try:
        upload(s3,cfg,'completion.json',BASE.parent/'completion.json')
    except Exception as e:
        completion['success']=False
        completion['errors'].append({'phase':'completion_upload','type':type(e).__name__})
    try:
        delivery=load(BASE.parent/'email_delivery.json',{})
        if not delivery.get('sns_message_id'):
            status='SUCCESS' if completion['success'] else 'NEEDS ATTENTION'
            link=f"https://s3.console.aws.amazon.com/s3/buckets/{cfg['bucket']}?region={cfg['region']}&prefix={cfg['prefix']}"
            counts=completion.get('verification',{}).get('status_counts',{})
            message=f"Capstone1000 scenario batch: {status}\nCounts: {json.dumps(counts)}\nCSV: training_dataset.csv\nShareableZIP: risk1000_csv_for_sharing.zip\nResults: {link}\nThese are simulation proxy-risk labels, not observed fire labels.\nThe server now requests shutdown."
            response=sns.publish(TopicArn=cfg['topic_arn'],Subject='Capstone1000 scenarios - '+status,Message=message)
            delivery={'sns_message_id':response['MessageId'],'recipient':cfg['recipient'],'sent_utc':now(),'success_reported':completion['success']}
            save(BASE.parent/'email_delivery.json',delivery)
        completion['email_notification']=delivery
    except Exception as e:
        completion['email_notification']={'status':'failed','error_type':type(e).__name__}
    save(BASE.parent/'completion.json',completion)
    try:
        upload(s3,cfg,'completion.json',BASE.parent/'completion.json')
    except Exception as e:
        print('Final receipt retained locally:',type(e).__name__,flush=True)
    print(json.dumps(completion,ensure_ascii=False),flush=True)

if __name__=='__main__':
    {'before':before,'monitor':monitor,'finish':finish,'verify':verify}[sys.argv[1]]()
