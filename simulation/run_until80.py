"""Bounded until-SOH80 continuation using unchanged physical/controller modules.

No ML, parameter fitting or relaxed connection gate. Every resume requires a
healthy committed rollover. Full history stays in immutable segment/RPT files;
checkpoint contains one physical state and a journal tip. Cloud orchestration
and the persistent absolute server-stop timer are separate modules.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
from types import SimpleNamespace
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
sys.dont_write_bytecode = True
import until80_policy as P


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--migrate-checkpoint')
    source.add_argument('--resume',action='store_true')
    source.add_argument('--fresh-test-only',action='store_true')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--cell',type=int,default=4,choices=range(1,10))
    p.add_argument('--current',type=float,default=3.2,choices=(3.,3.1,3.2,3.3))
    p.add_argument('--mesh-scale',type=int,default=4,choices=(1,2,4,8))
    p.add_argument('--tolerance-factor',type=float,default=1.,choices=(1.,.1,.01))
    p.add_argument('--numerical-control-report')
    p.add_argument('--source-archive',default='/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')
    p.add_argument('--max-efc',type=float,default=P.MAX_EFC)
    p.add_argument('--max-wall-s',type=float,default=P.MAX_WALL_S)
    p.add_argument('--worker-wall-s',type=float,default=P.MAX_WORKER_S)
    p.add_argument('--deadline-utc')
    p.add_argument('--rpt-every-efc',type=float,default=10.)
    p.add_argument('--max-memory-gib',type=float,default=24.)
    p.add_argument('--min-free-memory-gib',type=float,default=2.)
    p.add_argument('--min-free-disk-gib',type=float,default=2.)
    p.add_argument('--smoke-segments',type=int,default=0,help='Expected healthy rollover after N segments')
    p.add_argument('--test-scope',action='store_true',help='Local mesh1 <=2EFC/600s verification only')
    p.add_argument('--test-failure-after-pieces',type=int,default=0,help=argparse.SUPPRESS)
    p.add_argument('--test-rollover-after-pieces',type=int,default=0,help=argparse.SUPPRESS)
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    return p


def worker(args):
    import casadi
    import pickle
    import numpy as np
    import pybamm
    import battery as B
    import run_100efc as R
    import run_lifetime as L
    import lifetime_operation as O
    import lifetime_policy as LEGACY
    import connection_initial_state as C
    import shutil
    began = time.perf_counter()
    out = Path(args.output_dir).resolve()
    source_manifest = {}
    state = None
    obs = None
    initialized = False
    pending_rpts = []
    accepted=0
    stop = {'signal':None}
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda number,_frame:stop.update(signal=number))

    def source_hashes():
        return {str(p):P.digest(p) for p in source_manifest}

    def compact_solution(main):
        # Solution.last_state retains its last sub-solution. Construct a one-time
        # Solution with exactly the same y/model/input and no prior time history.
        last = main.last_state
        if last.has_sensitivities():
            raise RuntimeError('Sensitivity-bearing checkpoints require a separate validated adapter')
        tiny = pybamm.Solution([np.array(last.t,copy=True)],
            [np.asarray(last.all_ys[-1]).copy()],last.all_models[-1:],
            [copy.deepcopy(last.all_inputs[-1])],last.t_event,last.y_event,last.termination,
            all_yps=None if last.all_yps is None else [np.asarray(last.all_yps[-1]).copy()],
            options=last.user_options)
        if not np.array_equal(B.y_end(last),B.y_end(tiny)):
            raise RuntimeError('Storage compaction changed physical state')
        return tiny

    def save_checkpoint():
        temp = out/'checkpoint.pkl.tmp'
        with casadi.global_pickle_context():
            with temp.open('wb') as handle:
                pickle.dump(state,handle,protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush();os.fsync(handle.fileno())
        temp.replace(out/'checkpoint.pkl')
        if os.name == 'posix':
            fd=os.open(out,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)

    def identity(source_sha):
        control=P.validate_control(args.numerical_control_report,args.tolerance_factor,source_sha)
        paths=[B.ROOT/rel for rel in LEGACY.LEGACY_SHA256]
        paths += [Path(L.__file__),Path(O.__file__),Path(LEGACY.__file__),
                  B.ROOT/'simulation/lifetime_supervision.py',Path(__file__),Path(P.__file__),Path(C.__file__)]
        return {'format':P.FORMAT,'cell':args.cell,'current_A':args.current,
            'mesh_scale':args.mesh_scale,'eol_SOH_percent':80.,
            'numerical_spec':{'factor':args.tolerance_factor,'rtol':B.DEFAULT_RTOL*args.tolerance_factor,
                              'atol':B.DEFAULT_ATOL*args.tolerance_factor,'connection_limit':1e-10,
                              'connection_definition':C.DEFINITION,'raw_first_output_preserved':True},
            'numerical_control':control,'max_EFC':args.max_efc,'max_wall_s':args.max_wall_s,
            'deadline_utc':args.deadline_utc,'rpt_every_EFC':args.rpt_every_efc,'output_period_s':1.,
            'resource_limits':{'max_memory_GiB':args.max_memory_gib,
                'minimum_free_memory_GiB':args.min_free_memory_gib,'minimum_free_disk_GiB':args.min_free_disk_gib},
            'test_scope':args.test_scope,'python':platform.python_version(),'versions':L.versions(),
            'source_sha256':{p.relative_to(B.ROOT).as_posix():P.digest(p) for p in paths}}

    def report(checkpoint_state=None,status=None):
        st=checkpoint_state or state
        latest=st.get('latest_rpt') or {}
        return {'format':P.FORMAT,'status':status or st['status'],'terminal':st['terminal'],
            'identity_sha256':P.identity_sha(st['identity']),
            'source_checkpoint_sha256':st['source_checkpoint_sha256'],
            'current_checkpoint_sha256':P.digest(out/'checkpoint.pkl'),
            'checkpoint_bytes':(out/'checkpoint.pkl').stat().st_size,
            'checkpoint_commit_id':st['commit_index'],'history_commit_id':st['journal_head_sha256'],
            'last_valid_EFC':st['ledger']['efc'],'last_SOH_percent':latest.get('SOH_percent'),
            'last_SOH_measurement_EFC':latest.get('efc'),'eol':st.get('eol'),
            'right_censored':False if st.get('eol') and st['status']=='eol_observed' else None if (status or st['status'])=='failed' else True,
            'cumulative_wall_s':st['cumulative_wall_s'],'source_elapsed_wall_s':st['source_elapsed_wall_s'],
            'branch_elapsed_wall_s':st['branch_elapsed_wall_s'],'worker_elapsed_wall_s':time.perf_counter()-began,
            'row_count':st['row_count'],'rpt_count':st['rpt_count'],
            'max_EFC':args.max_efc,'max_wall_s':args.max_wall_s,
            'worker_wall_limit_s':args.worker_wall_s,'deadline_utc':args.deadline_utc,
            'numerical_spec':st['identity']['numerical_spec'],
            'limitation':'SOH80 is an RPT interval observation, not validated real LG M50 lifetime'}

    def elapsed():
        return state['source_elapsed_wall_s']+prior_branch+time.perf_counter()-began

    def guard():
        if stop['signal'] is not None:return 'interrupted_censored'
        if elapsed() >= args.max_wall_s:return 'global_time_censored'
        if args.deadline_utc and dt.datetime.now(dt.timezone.utc)>=P.deadline(args.deadline_utc):return 'deadline_censored'
        if B.memory_mb().get('rss_mb',0)>args.max_memory_gib*1024:return 'resource_limit_censored:process_memory'
        if shutil.disk_usage(out).free<args.min_free_disk_gib*2**30:return 'resource_limit_censored:disk'
        if sys.platform.startswith('linux'):
            available=next((float(line.split()[1])/1024**2 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),None)
            if available is not None and available<args.min_free_memory_gib:return 'resource_limit_censored:system_memory'
        if args.test_rollover_after_pieces and accepted>=args.test_rollover_after_pieces:return 'worker_rollover_requested'
        if time.perf_counter()-began>=args.worker_wall_s:return 'worker_rollover_requested'
        return None

    def artifact(path,**extra):
        return {'path':path.relative_to(out).as_posix(),'sha256':P.digest(path),'bytes':path.stat().st_size,**extra}

    def commit(status,checks=None,cycle=None):
        nonlocal obs
        index=state['commit_index']+1
        parquet=None
        if obs is not None:
            obs.close()
            if obs.rows:parquet=artifact(obs.path,rows=obs.rows)
        rpt_artifacts=[]
        for point in pending_rpts:
            state['rpt_count']+=1
            path=out/'rpt'/f"rpt_{state['rpt_count']:08d}.json"
            P.save(path,point);rpt_artifacts.append(artifact(path,index=state['rpt_count']))
        branch_now=prior_branch+time.perf_counter()-began
        cumulative_now=state['source_elapsed_wall_s']+branch_now
        entry={'index':index,'previous_journal_sha256':state['journal_head_sha256'],
               'source_checkpoint_sha256':state['source_checkpoint_sha256'],
               'identity_sha256':P.identity_sha(state['identity']),'status':status,
               'ledger':B.ledger(rs),'cursor':copy.deepcopy(cursor),
               'checks':checks,'cycle':cycle,'rpts':rpt_artifacts,'parquet':parquet,
               'cumulative_wall_s':cumulative_now}
        path=out/'segments'/f'segment_{index:08d}.json';P.save(path,entry)
        state.update(commit_index=index,journal_head_sha256=P.digest(path),status=status,
            terminal=status not in ('running','ready',P.ROLLOVER),ledger=B.ledger(rs),
            cursor=copy.deepcopy(cursor),solution=compact_solution(sol),
            cumulative_wall_s=cumulative_now,branch_elapsed_wall_s=branch_now,
            row_count=state['row_count']+(obs.rows if obs else 0))
        save_checkpoint()
        P.save(out/'summary.json',report())
        pending_rpts.clear();obs=None

    def diagnose(main,ledger,target):
        before=B.y_end(main);book=copy.deepcopy(B.ledger(ledger))
        reference=cell.reference_test(main.last_state.copy());q=reference['Q_cum_Ah']
        if not reference['ok'] or not np.isfinite(q) or q<=0 or abs(reference['Q_trapz_Ah']-q)/q>1e-4:
            raise RuntimeError('Invalid reference capacity')
        if not np.array_equal(before,B.y_end(main)) or book!=B.ledger(ledger):
            raise RuntimeError('Reference test changed operational state')
        if ledger.q0_Ah is None:ledger.q0_Ah=q
        point={'efc':ledger.efc,'target_efc':target,'Qref_Ah':q,
               'SOH_percent':100*q/ledger.q0_Ah,'reference':reference,
               'state_max_abs_change':0.,'ledger_unchanged':True,'origin':'until80_RPT'}
        raw=cell.last_reference_observations
        if not raw or len(raw)!=4:raise RuntimeError('Raw reference observations missing')
        raw_path=out/'rpt'/f"rpt_{state['rpt_count']+len(pending_rpts)+1:08d}_raw.npz"
        with raw_path.open('wb') as handle:
            np.savez_compressed(handle,**{f'{phase}_{name}':values for phase,part in
                zip(('cc_charge','cv_hold','rest','discharge'),raw) for name,values in part.items()})
            handle.flush();os.fsync(handle.fileno())
        point['raw_observations']=artifact(raw_path,period='inherited reference Experiment output grid')
        cell.last_reference_observations=None
        previous=state.get('latest_rpt');state['latest_rpt']=point
        state['eol']=P.eol_bracket(previous,point)
        pending_rpts.append(point)
        return 'eol_observed' if state['eol'] else guard()

    try:
        for rel,expected in LEGACY.LEGACY_SHA256.items():
            if P.digest(B.ROOT/rel)!=expected:raise ValueError('Frozen physical source mismatch: '+rel)
        if args.resume:
            if not out.is_dir():raise ValueError('Resume branch is missing')
            current_report=P.read(out/'summary.json');P.validate_checkpoint_report(out,current_report)
            if (out/'FAILED.json').exists():raise ValueError('Failure is fatal; do not automatically retry this branch')
            state=R.load_checkpoint(out/'checkpoint.pkl')
            if state.get('format')!=P.FORMAT or state.get('status')!=P.ROLLOVER or state.get('terminal'):
                raise ValueError('Only healthy checkpointed rollover may resume')
            if state['identity']!=identity(state['source_checkpoint_sha256']):
                raise ValueError('Continuation identity/config/numerical-control mismatch')
            if state['commit_index']!=current_report['checkpoint_commit_id'] or state['journal_head_sha256']!=current_report['history_commit_id']:
                raise ValueError('Checkpoint and result commit tips differ')
            P.save(out/'resume_state_receipt.json',{'checkpoint_sha256':P.digest(out/'checkpoint.pkl'),
                'physical_y_sha256':P.hashlib.sha256(np.ascontiguousarray(B.y_end(state['solution'])).tobytes()).hexdigest(),
                'ledger':copy.deepcopy(state['ledger']),'cursor':copy.deepcopy(state['cursor']),
                'commit_index':state['commit_index'],'journal_head_sha256':state['journal_head_sha256'],
                'identity_verified':True,'before_first_advance':True})
            quarantined=P.quarantine_uncommitted(out,state)
            if quarantined:P.save(out/'last_quarantine.json',quarantined)
        else:
            if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty continuation branch')
            if args.migrate_checkpoint:
                source=Path(args.migrate_checkpoint).resolve(strict=True)
                source_root=source.parent.parent
                if source_root==out or source_root in out.parents or out in source_root.parents:
                    raise ValueError('Source and continuation branch must be disjoint')
                source_sha=P.digest(source)
                if not args.test_scope:
                    if not sys.platform.startswith('linux'):raise ValueError('AWS migration requires the identical Linux environment')
                    if source_sha!=P.SOURCE_CHECKPOINT_SHA256 or P.digest(args.source_archive)!=P.SOURCE_ARCHIVE_SHA256:
                        raise ValueError('Trusted frozen1614 checkpoint/archive SHA mismatch')
                source_manifest={p:P.digest(p) for p in [source,*[source.parent/n for n in ('summary.json','diagnostics.json','outcome.json','FAILED.json') if (source.parent/n).is_file()]]}
                old=R.load_checkpoint(source)
                sig=old['identity'];baseline=SimpleNamespace(cell=args.cell,current=args.current,mesh_scale=args.mesh_scale,
                    efc=sig.get('target_EFC'),eol_soh=sig.get('eol_SOH_percent'),rpt_every_efc=sig.get('rpt_every_after_100_EFC'))
                expected_legacy=L.identity(baseline)
                # The single reviewed legacy->v2 difference is this controller's
                # continuity observation. Every other source/config/dependency
                # remains checked against the trusted parent checkpoint identity.
                expected_legacy['source_sha256']['simulation/lifetime_operation.py']=P.LEGACY_OPERATION_SHA256
                if sig!=expected_legacy:raise ValueError('Original checkpoint identity/source/dependencies differ')
                if old.get('terminal') or old['ledger']['efc']>=args.max_efc:
                    raise ValueError('Migration must extend a nonterminal physical state')
                prior=float(old['elapsed_wall_s'])
                if (source.parent/'FAILED.json').is_file():prior=max(prior,float(P.read(source.parent/'FAILED.json').get('elapsed_wall_s',prior)))
                last=copy.deepcopy((old.get('diagnostics') or [])[-1])
                if last['SOH_percent']<80:raise ValueError('Source already contains observed EOL')
                new_identity=identity(source_sha)
                lineage={'source_checkpoint':str(source),'source_checkpoint_sha256':source_sha,
                    'source_archive_sha256':None if args.test_scope else P.SOURCE_ARCHIVE_SHA256,
                    'legacy_identity':copy.deepcopy(sig),'source_files_sha256':{str(p):v for p,v in source_manifest.items()},
                    'source_elapsed_wall_s':prior,'source_EFC':old['ledger']['efc'],
                    'same_physics':True,'numerical_branch':True,
                    'continuity_migration':{'from':'previous-end vs first returned output',
                        'to':C.DEFINITION,'threshold_before':1e-10,'threshold_after':1e-10,
                        'legacy_operation_sha256':P.LEGACY_OPERATION_SHA256,
                        'current_operation_sha256':P.digest(Path(O.__file__)),
                        'raw_first_output_preserved':True},
                    'test_scope':args.test_scope,'note':'Parent failure and raw history remain unchanged; this is a new branch'}
                state={'format':P.FORMAT,'identity':new_identity,'source_checkpoint_sha256':source_sha,
                    'source_elapsed_wall_s':prior,'branch_elapsed_wall_s':0.,'cumulative_wall_s':prior,
                    'solution':compact_solution(old['solution']),'ledger':copy.deepcopy(old['ledger']),
                    'cursor':copy.deepcopy(old['cursor']),'latest_rpt':last,'eol':None}
                migration_receipt={'source_checkpoint_sha256':source_sha,
                    'source_y_sha256':P.hashlib.sha256(np.ascontiguousarray(B.y_end(old['solution'])).tobytes()).hexdigest(),
                    'migrated_y_sha256':P.hashlib.sha256(np.ascontiguousarray(B.y_end(state['solution'])).tobytes()).hexdigest(),
                    'physical_y_bitwise_equal':bool(np.array_equal(B.y_end(old['solution']),B.y_end(state['solution']))),
                    'ledger_unchanged':state['ledger']==old['ledger'],
                    'cursor_unchanged':state['cursor']==old['cursor'],'source_identity_verified':True}
                last['origin']='copied_parent_RPT';pending_rpts.append(last)
                del old;gc.collect()
            else:
                if not args.test_scope or not args.fresh_test_only:raise ValueError('Fresh physical initialization is verification-only')
                state={'format':P.FORMAT,'identity':identity(None),'source_checkpoint_sha256':None,
                    'source_elapsed_wall_s':0.,'branch_elapsed_wall_s':0.,'cumulative_wall_s':0.,
                    'latest_rpt':None,'eol':None}
                lineage={'test_scope':True,'source':'fresh mesh1 verification only'}
            out.mkdir(parents=True,exist_ok=True);(out/'segments').mkdir();(out/'rpt').mkdir()
            P.save(out/'lineage.json',lineage)
            if args.migrate_checkpoint:P.save(out/'migration_state_receipt.json',migration_receipt)
            state.update(commit_index=-1,journal_head_sha256=None,rpt_count=0,row_count=0,status='ready',terminal=False)
        initialized=True
        prior_branch=float(state['branch_elapsed_wall_s'])
        template=L.make_cell(args)
        cell=C.InitialStateCell(template.overrides,rtol=template.rtol*args.tolerance_factor,
            atol=template.atol*args.tolerance_factor,mesh_scale=args.mesh_scale)
        del template
        if 'solution' in state:
            sol=state['solution'];rs=B.RunState(**state['ledger']);cursor=copy.deepcopy(state['cursor'])
        else:
            sol=cell.initial_state();rs=B.RunState(args.cell,'cell4_3.2A_until80_test',t_abs_s=float(sol.t[-1]))
            cursor=O.new_cursor(rs)
            reason=diagnose(sol,rs,0.)
            if reason and reason not in ('worker_rollover_requested',):raise RuntimeError('Initial verification could not establish Q0: '+reason)
        if state['commit_index']<0:commit('ready')
        initial_y=B.y_end(sol)
        accepted=0;segments=0
        def on_piece(main,step,kind,bounds,ledger):
            nonlocal accepted
            obs.add(main,step,kind,bounds,ledger);accepted+=1
            if args.test_failure_after_pieces and accepted>=args.test_failure_after_pieces:
                raise RuntimeError('test_sentinel_failure')
        while True:
            reason=guard()
            if reason:
                status=P.ROLLOVER if reason=='worker_rollover_requested' else reason
                commit(status);break
            if rs.efc>=args.max_efc-1e-8:
                status='max_efc_censored';commit(status);break
            index=state['commit_index']+1
            obs=R.Observations(out/'segments'/f'segment_{index:08d}.parquet')
            targets=P.upcoming_diagnostics(rs.efc,args.max_efc,args.rpt_every_efc)
            sol,cursor,reason,checks,cycle=O.run_segment(cell,sol,rs,cursor,args.current,args.max_efc,
                targets,diagnose,on_piece,guard)
            if reason=='cycle_complete':
                segments+=1
                status=P.ROLLOVER if args.smoke_segments and segments>=args.smoke_segments else 'running'
            else:status=P.ROLLOVER if reason=='worker_rollover_requested' else reason
            commit(status,checks,cycle)
            if status!='running':break
        if source_manifest and source_hashes()!= {str(p):v for p,v in source_manifest.items()}:
            raise RuntimeError('Migration source changed during continuation')
        result=report();P.save(out/'outcome.json',result)
        if state['terminal']:(out/'DONE').write_text(status+'\n',encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 75 if status==P.ROLLOVER else 0 if status in ('eol_observed','max_efc_censored') else 124
    except BaseException:
        error=traceback.format_exc()
        if not initialized:
            print(json.dumps({'status':'migration_or_resume_refused','error':error},ensure_ascii=False),flush=True)
            return 1
        if obs:obs.close(commit=False)
        if (out/'checkpoint.pkl').is_file():
            safe=R.load_checkpoint(out/'checkpoint.pkl')
            quarantine=P.quarantine_uncommitted(out,safe)
            result=report(safe,status='failed');result['terminal']=True;result['right_censored']=None
        else:
            quarantine=[];result={'format':P.FORMAT,'status':'failed','terminal':True,'last_valid_EFC':None}
        result.update(error=error,quarantined_uncommitted=quarantine,
            worker_elapsed_wall_s=time.perf_counter()-began,
            source_files_unchanged=(source_hashes()=={str(p):v for p,v in source_manifest.items()}) if source_manifest else None)
        P.save(out/'FAILED.json',result);P.save(out/'outcome.json',result)
        print(json.dumps({k:result.get(k) for k in ('status','last_valid_EFC','error')},ensure_ascii=False),flush=True)
        return 1


def main():
    args=parser().parse_args()
    try:P.validate_limits(args)
    except (ValueError,TypeError) as exc:raise SystemExit(str(exc))
    if args._worker:raise SystemExit(worker(args))
    out=Path(args.output_dir).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    prior=0.
    source=out if args.resume else Path(args.migrate_checkpoint).resolve().parent if args.migrate_checkpoint else None
    if source:
        for name in ('summary.json','outcome.json','FAILED.json'):
            path=source/name
            if path.is_file():
                data=P.read(path);prior=max(prior,float(data.get('cumulative_wall_s',data.get('elapsed_wall_s',0.))))
    import lifetime_supervision as S
    def forced(reason):
        if out.is_dir():
            safe=P.read(out/'summary.json') if (out/'summary.json').is_file() else {}
            result={**safe,'status':'failed','terminal':True,'right_censored':None,
                'error':'Worker exceeded bounded shutdown wait: '+reason,
                'last_checkpoint_authoritative':True}
            P.save(out/'FAILED.json',result);P.save(out/'outcome.json',result)
    budget=min(args.worker_wall_s,max(30.,args.max_wall_s-prior))
    code=S.supervise([sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--_worker'],
        out.parent/(out.name+'.runner.lock'),budget,on_forced_stop=forced)
    raise SystemExit(code)


if __name__=='__main__':main()
