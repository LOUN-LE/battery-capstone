"""Bounded, independent Linux audit supervisor. --plan never runs jobs or shuts down.

Sources are read only. Backup credentials are read from a private file outside
run-root and are never recorded in state, console, or the result archive.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse

GIB = 1024**3
STOP = {'signal': None}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def contains(parent, path):
    return path == parent or parent in path.parents


def build_jobs(args):
    # Do not resolve the venv launcher symlink into the base interpreter.
    python = str(Path(args.python).absolute())
    repo = Path(args.repo_root).resolve()
    root = Path(args.run_root).resolve()
    legacy = str(Path(args.legacy_dir).resolve())
    final = Path(args.final_dir).resolve()
    replay = [python, str(repo/'simulation/audit_connection_replay.py'),
              '--checkpoint', str(final/'checkpoint.pkl'),
              '--source-archive', str(Path(args.source_archive).resolve()),
              '--max-wall-s', '600', '--max-cycles', '3']
    return [
        {'name':'replay_a', 'outer_wall_s':660, 'command':[*replay, '--output-dir', str(root/'replay_a')]},
        {'name':'replay_b', 'outer_wall_s':660, 'command':[*replay, '--output-dir', str(root/'replay_b')]},
        {'name':'aging_rpt', 'outer_wall_s':1900,
         'command':[python, str(repo/'simulation/audit_aging_rpt.py'),
                    '--legacy-dir', legacy, '--final-dir', str(final),
                    '--output-dir', str(root/'aging_rpt'), '--cell','4','--current','3.2',
                    '--mesh-scale','4','--repeats','2','--state-timeout-s','600']},
        {'name':'mesh8_100', 'outer_wall_s':10860,
         'command':[python, str(repo/'simulation/run_100efc.py'), '--cell','4',
                    '--current','3.2','--mesh-scale','8','--efc','100',
                    '--max-wall-s','10800','--output-dir',str(root/'mesh8_100')]},
        {'name':'mesh_compare','outer_wall_s':60,
         'command':[python,str(repo/'simulation/audit_mesh_compare.py'),
                    '--mesh4-dir',legacy,'--mesh8-dir',str(root/'mesh8_100'),
                    '--output',str(root/'mesh_comparison.json')]},
    ]


def resources(root):
    free = shutil.disk_usage(root).free
    available = None
    try:
        m = {line.split(':',1)[0]:int(line.split()[1])
             for line in Path('/proc/meminfo').read_text().splitlines()}
        available = m['MemAvailable']/m['MemTotal']
    except (OSError, ValueError, KeyError):
        pass
    reason = 'disk_guard' if free < 2*GIB else 'memory_guard' if available is not None and available < .08 else None
    return {'reason':reason, 'free_GiB':round(free/GIB,4),
            'available_ram_fraction':available}


def group_members(pgid):
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')',1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def terminate_group(proc, grace_s=60):
    result = {'term_sent':False,'kill_sent':False,'group_empty':False}
    if group_members(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM); result['term_sent']=True
        except ProcessLookupError:
            pass
        end = time.monotonic()+max(0,grace_s)
        while group_members(proc.pid) and time.monotonic()<end:
            proc.poll(); time.sleep(.2)
        if group_members(proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL); result['kill_sent']=True
            except ProcessLookupError:
                pass
            end = time.monotonic()+10
            while group_members(proc.pid) and time.monotonic()<end:
                proc.poll(); time.sleep(.1)
    result['group_empty'] = not group_members(proc.pid)
    if proc.poll() is not None:
        proc.wait(timeout=1)
    return result


def run_process(command, log_path, cwd, wall_s, global_deadline, root,
                on_progress=None, env=None, honor_stop=True, sensitive=False):
    began = time.monotonic()
    # Reserve the group cleanup grace inside the global computation deadline.
    deadline = min(began+wall_s, global_deadline-60)
    proc = None
    record = {'started_utc':utc(), 'timeout_s':wall_s,
              'status':'starting', 'returncode':None}
    try:
        with Path(log_path).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status='running', pid=proc.pid)
            next_progress = 0
            while proc.poll() is None:
                now = time.monotonic(); resource = resources(root)
                reason = ('signal_stop' if honor_stop and STOP['signal'] is not None else resource['reason']
                          or ('wall_limit' if now>=deadline else None))
                if now>=next_progress and on_progress:
                    on_progress({'elapsed_s':round(now-began,2),'pid':proc.pid,
                                 'resources':resource})
                    next_progress = now+60
                if reason:
                    record['status']=reason
                    record['termination']=terminate_group(proc, min(60,max(0,global_deadline-time.monotonic())))
                    break
                time.sleep(1)
            # A parent that exits early must not leave its worker group running.
            if group_members(proc.pid):
                record['termination']=terminate_group(proc, min(60,max(0,global_deadline-time.monotonic())))
                if record['status']=='running':
                    record['status']='parent_exited_with_live_children'
            record['returncode']=proc.wait(timeout=15)
            if record['status']=='running':
                record['status']='exited_zero' if record['returncode']==0 else 'exited_nonzero'
    except BaseException as exc:
        record.update(status='supervisor_job_exception',error_type=type(exc).__name__,
                      error='Private transfer process failed' if sensitive else str(exc)[:600])
    finally:
        if proc is not None and group_members(proc.pid):
            record['termination']=terminate_group(proc, min(60,max(0,global_deadline-time.monotonic())))
        record.update(finished_utc=utc(),wall_s=time.monotonic()-began)
    return record


def source_manifest(args):
    result = {}
    for label, folder in [('legacy',Path(args.legacy_dir)),('final',Path(args.final_dir))]:
        for name in ['checkpoint.pkl','summary.json','diagnostics.json','outcome.json','FAILED.json']:
            path = folder/name
            if path.is_file(): result[label+'/'+name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    return result


def job_result(name, root):
    if name=='mesh_compare':
        data=read(root/'mesh_comparison.json')
        return {k:data.get(k) for k in ['max_SOH_abs_diff_pp',
            'max_paired_charge_time_diff_s','EOL_convergence_established',
            'long_term_convergence_established']}
    if name.startswith('replay_'):
        data=read(root/name/'audit_summary.json')
        return {k:data.get(k) for k in ['worker_status','original_failure_reproduced',
                                       'original_files_unchanged','worker_exit_code']}
    if name=='aging_rpt':
        data=read(root/name/'audit_report.json')
        return {k:data.get(k) for k in ['status','original_source_files_unchanged',
                                       'validation_status','chemistry_verified']}
    data=read(root/name/'summary.json')
    failure=read(root/name/'FAILED.json')
    efc=(data.get('ledger') or {}).get('efc')
    last=(data.get('diagnostics') or [{}])[-1]
    return {'status':data.get('status'),'EFC':efc,'last_RPT_EFC':last.get('efc'),
            'last_SOH_percent':last.get('SOH_percent'), 'failure_file_present':bool(failure),
            'target_complete':bool((root/name/'DONE').is_file() and not failure
               and data.get('status')=='efc_target' and isinstance(efc,(int,float)) and efc>=100-1e-8),
            'from_initial_state':True,'cross_mesh_checkpoint_transfer':False}


def backup_url(args, root):
    if not args.backup_url_file:return None
    path=Path(args.backup_url_file).resolve(strict=True)
    if contains(root,path) or stat.S_IMODE(path.stat().st_mode)!=0o600:
        raise ValueError('Backup URL file must be mode 0600 and outside run-root')
    value=path.read_text(encoding='utf-8').strip()
    parsed=urlparse(value)
    if parsed.scheme!='https' or parsed.username or not parsed.hostname or not parsed.hostname.endswith('.amazonaws.com'):
        raise ValueError('Expected private HTTPS AWS S3 upload URL')
    return value


def compact(state):
    return {'stage':'audit_final','status':state.get('status'),
        'started_utc':state.get('started_utc'),'finished_utc':state.get('finished_utc'),
        'wall_s':state.get('wall_s'),'jobs':{n:{'status':r.get('status'),
            'returncode':r.get('returncode'),'result':r.get('result')} for n,r in state.get('jobs',{}).items()},
        'original_sources_unchanged':state.get('original_sources_unchanged'),
        'backup':state.get('backup'),'shutdown_requested':True,'run_root':state.get('run_root')}


def emit(value):
    line=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    print(line,flush=True)
    try:
        with open('/dev/console','a',encoding='utf-8') as console:console.write(line+'\n')
    except OSError:pass


def run(args):
    root=Path(args.run_root).resolve();repo=Path(args.repo_root).resolve()
    jobs=build_jobs(args)
    if args.plan:
        print(json.dumps({'mode':'plan','no_jobs_started':True,'no_shutdown':True,
           'jobs':jobs,'total_wall_limit_s':args.max_wall_s,
           'independent_hard_stop_s':args.max_wall_s+900,'backup_reserve_s':300,
           'backup_url_in_results':False,'cross_mesh_checkpoint_transfer':False},indent=2))
        return 0
    if os.name!='posix' or not hasattr(os,'geteuid') or os.geteuid()!=0:
        raise RuntimeError('Actual audit suite requires Linux root; use --plan elsewhere')
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty audit run-root')
    for source in [repo,Path(args.legacy_dir).resolve(),Path(args.final_dir).resolve()]:
        if contains(source,root) or contains(root,source):raise ValueError('Audit root must be disjoint from source trees')
    root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+args.max_wall_s
    state={'status':'starting','started_utc':utc(),'run_root':str(root),
           'max_wall_s':args.max_wall_s,'hard_stop_s':args.max_wall_s+900,
           'jobs':{},'original_sources_unchanged':None,
           'chemistry_validated':False,'grid_scope':'independent initial-to-100 EFC x8 only',
           'backup':{'status':'not_attempted'},'chat_required':False}
    active=None
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda number,_frame:STOP.update(signal=number))
    try:
        subprocess.run(['systemd-run','--unit=capstone-audit-hard-stop',
            '--on-active='+str(args.max_wall_s+900)+'s','--timer-property=AccuracySec=1s',
            '/usr/sbin/shutdown','-h','now'],check=True,timeout=30)
        save(root/'audit_status.json',state)
        url=backup_url(args,root)
        for job in jobs:
            if not Path(job['command'][1]).is_file():raise FileNotFoundError('Missing reviewed audit entrypoint')
        if not Path(args.python).is_file():raise FileNotFoundError('Selected Python executable missing')
        before=source_manifest(args);save(root/'source_manifest_before.json',before)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for job in jobs:
            if STOP['signal'] is not None or time.monotonic()>=deadline-360:
                state['status']='signal_stopped' if STOP['signal'] else 'suite_wall_limited';break
            resource=resources(root)
            if resource['reason']:state['status']=resource['reason'];break
            name=job['name']
            if name=='mesh_compare' and not state['jobs'].get('mesh8_100',{}).get('result',{}).get('target_complete'):
                state['jobs'][name]={'status':'skipped_incomplete_mesh8','returncode':None}
                save(root/'audit_status.json',state);continue
            if name=='mesh8_100':
                core={'stage':'audit_core_complete','status':'core_jobs_finished',
                    'jobs':{n:{'status':r.get('status'),'returncode':r.get('returncode'),
                               'result':r.get('result')} for n,r in state['jobs'].items()},
                    'paths':{n:str(root/n) for n in state['jobs']},
                    'validation_complete':False,'utc':utc()}
                save(root/'audit_core_summary.json',core);emit(core)
            state.update(status='running',current_job=name)
            state['jobs'][name]={'status':'starting','started_utc':utc()}
            save(root/'audit_status.json',state);emit({'stage':'audit_job_started','job':name,'utc':utc()})
            def progress(value):
                state['jobs'][name].update(status='running',progress=value)
                save(root/'audit_status.json',state)
            record=run_process(job['command'],root/(name+'.log'),repo,
                job['outer_wall_s'],deadline-300,root,on_progress=progress,env=env)
            record['result']=job_result(name,root);state['jobs'][name]=record
            save(root/'audit_status.json',state)
            emit({'stage':'audit_job_finished','job':name,'status':record['status'],
                  'returncode':record['returncode'],'result':record['result']})
            if record['status'] in ('disk_guard','memory_guard','signal_stop'):
                state['status']=record['status'];break
        else:
            state['status']='completed' if all(r.get('status')=='exited_zero' for r in state['jobs'].values()) else 'completed_with_job_failures'
        after=source_manifest(args);save(root/'source_manifest_after.json',after)
        state['original_sources_unchanged']=before==after
        if before!=after:state['status']='original_source_modified'
    except BaseException as exc:
        state.update(status='suite_failed',error_type=type(exc).__name__,error=str(exc)[:600],traceback=traceback.format_exc()[-2500:])
        # All subprocess error strings here omit private upload URLs.
    finally:
        state.update(finished_utc=utc(),wall_s=time.monotonic()-started,current_job=None)
        try:
            save(root/'audit_status.json',state)
            # Sibling archive excludes URL file and cannot recursively include itself.
            backups=root.parent/'audit_backups';backups.mkdir(exist_ok=True)
            archive=backups/(root.name+'_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.tar.gz')
            if contains(root,archive):raise RuntimeError('Archive must be outside audit root')
            if time.monotonic()<deadline-30 and shutil.disk_usage(backups).free>2*GIB:
                tar_record=run_process(['tar','-I','gzip -1','-cf',str(archive),'-C',str(root.parent),root.name],
                    backups/(root.name+'_tar.log'),root.parent,min(180,max(1,deadline-time.monotonic()-60)),deadline,root,honor_stop=False)
                if tar_record['returncode']==0 and tar_record['status']=='exited_zero':
                    state['backup']={'status':'archived_on_EBS','archive':str(archive),
                        'bytes':archive.stat().st_size,'sha256':digest(archive)}
                    if 'url' in locals() and url and time.monotonic()<deadline-15:
                        # Never record this command or CalledProcessError: it contains a signed URL.
                        upload=run_process(['curl','--fail','--silent','--show-error','--retry','1',
                            '--upload-file',str(archive),'-H','x-amz-server-side-encryption: AES256',url],
                            backups/(root.name+'_upload.log'),root.parent,max(1,deadline-time.monotonic()-15),deadline,root,honor_stop=False,sensitive=True)
                        state['backup'].update(status='uploaded' if upload['status']=='exited_zero' else 'upload_failed',
                                               upload_returncode=upload['returncode'])
                else:state['backup']={'status':'archive_failed','returncode':tar_record['returncode']}
            else:state['backup']={'status':'archive_skipped_time_or_disk_guard'}
        except BaseException as exc:
            state['backup']={'status':'backup_exception','error_type':type(exc).__name__}
        finally:
            state.update(finished_utc=utc(),wall_s=time.monotonic()-started,shutdown_requested=True)
            try:
                save(root/'audit_status.json',state);emit(compact(state))
            finally:
                try:subprocess.run(['sync'],check=False,timeout=30)
                finally:subprocess.run(['/usr/sbin/shutdown','-h','now'],check=False,timeout=30)
    return 0 if state.get('status')=='completed' and state.get('original_sources_unchanged') else 1


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--legacy-dir',required=True);p.add_argument('--final-dir',required=True)
    p.add_argument('--python',required=True);p.add_argument('--backup-url-file')
    p.add_argument('--source-archive',default='/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')
    p.add_argument('--max-wall-s',type=int,default=14400);p.add_argument('--plan',action='store_true')
    args=p.parse_args()
    if not 600<=args.max_wall_s<=14400:p.error('Suite wall limit must be 600..14400 seconds')
    return args


if __name__=='__main__':
    raise SystemExit(run(parse()))
