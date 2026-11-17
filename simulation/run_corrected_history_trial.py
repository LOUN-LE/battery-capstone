"""Bounded chronological conditional trial using reconstructed modes and raw current.

No legacy input bypass. Native RPT controls predict endpoints; ageing waveforms,
recorded mode3 rests and separately labelled unknown-gap assumptions are kept.
"""
from __future__ import annotations
import argparse,gc,json,os,pickle,subprocess,sys,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from run_exposure_scenario_smoke import read,write,sha,validate_input
from native_program_contracts import decode_mb
from conditional_history_inputs import native_step
from bounded_current_waveform import compress,verified_setpoint,POLICY
from corrected_history_checkpoint import load_resume,state_layout,verify_rest_equations
from runtime_physics_identity import capture


class Arrays:
    def __init__(self,directory):self.directory=directory;self.cache=OrderedDict();self.guards={}
    def get(self,source):
        path=(self.directory/source['arrays_file']).resolve();path.relative_to(self.directory)
        if str(path) not in self.guards:
            if sha(path)!=source['arrays_sha256']:raise ValueError('Raw NPZ changed')
            self.guards[str(path)]=source['arrays_sha256']
        if str(path) not in self.cache:
            with np.load(path,allow_pickle=False) as z:self.cache[str(path)]={k:z[k] for k in ('t_s','I_mA','mode','Ns','current_drive_eligible_interval')}
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(str(path));return self.cache[str(path)]


def waveform(s,a):
    lo,hi=s['row_start'],s['row_end_exclusive'];t=a['t_s'][lo:hi];I=a['I_mA'][lo:hi];m=a['current_drive_eligible_interval'][lo:hi-1]
    if len(t)<2 or m.dtype!=np.bool_ or not np.all(m) or not np.all(np.diff(t)>0):raise ValueError('Unreviewed raw drive interval')
    modes=a['mode'][lo:hi];ns=a['Ns'][lo:hi]
    if len(np.unique(modes))!=1 or modes[0] not in (1,2) or len(np.unique(ns))!=1:
        raise ValueError('Raw drive crosses native mode/Ns boundary')
    if not np.isfinite(t).all() or not np.isfinite(I).all():raise ValueError('Nonfinite drive')
    if t[0]!=s['start_s'] or t[-1]!=s['end_s'] or abs(t[-1]-t[0]-s['duration_us']/1e6)>1.5e-6:raise ValueError('Current clock/window mismatch')
    return np.column_stack([t-t[0],-I/1000])


def compile_route(root,scenario,stop_after_rpt):
    s,raw_path=validate_input(scenario);raw=read(raw_path);cell=raw['cell'];arrays=Arrays(raw_path.parent)
    program_path=root/'work/validation_proposal_20261103/metadata_control_probe_v2/decoded_programs_raw.json'
    programs={p['whole_file_sha256']:p for p in read(program_path)};sources={x['source_id']:x for x in cell['source_files']}
    measurements={(a['MPR_SHA256'],a['row_start']):a for a in cell['measurements']}
    grouped={};ref_duration={}
    for step in s['steps']:ref_duration[step.get('source_ref')]=ref_duration.get(step.get('source_ref'),0)+step['duration_us']
    for index,step in enumerate(s['steps']):
        rawstep=cell['steps'][step['raw_step_index']]
        if rawstep.get('source_kind')=='rpt' and rawstep.get('kind')=='recorded_mode_block':
            grouped.setdefault(step['source_ref'],[]).append((index,step,rawstep))
    plan=[];seen=set();zero=[];found=[]
    for i,step in enumerate(s['steps']):
        rawstep=cell['steps'][step['raw_step_index']];ref=step.get('source_ref')
        if ref in grouped:
            if ref in seen:continue
            seen.add(ref);g=grouped[ref]
            if len(g)!=1:raise ValueError('RPT split/mixed run requires explicit control review: '+ref)
            _,source_step,r=g[0]
            if abs(ref_duration[ref]-r['duration_us'])>1:raise ValueError('RPT envelope would omit reconstructed internal exposure: '+ref)
            source=sources[r['source_id']];a=arrays.get(source)
            lo,hi=r['row_start'],r['row_end_exclusive'];modes=np.unique(a['mode'][lo:hi]);ns=np.unique(a['Ns'][lo:hi])
            if len(modes)!=1 or len(ns)!=1:raise ValueError('Mixed RPT source cannot collapse')
            program=programs[source['native_sha256']]
            if program['technique']!='MB':raise ValueError('RPT MB programme required')
            decoded=decode_mb(program['raw_program_rows'][int(ns[0])],preserve_branch_metadata=True,sequence_count=program['sequence_count'],preserve_charge_metadata=True)
            if modes[0]!={'current':1,'voltage':2,'rest':3}[decoded['operation']]:raise ValueError('RPT raw/control mode mismatch')
            spec=native_step(decoded,float(r['end_s']-r['start_s']));anchor=measurements.get((source['native_sha256'],lo))
            if anchor is not None:
                if spec['operation']!='current' or abs(spec['value']-.5)>1e-9 or spec['termination']!=['2.5 V'] or spec['timed_endpoint']:
                    raise ValueError('RPT capacity must independently terminate at2.5V')
                found.append(anchor['rpt'])
            item=dict(kind='native_RPT_control',source_ref=ref,scenario_index=i,spec=spec,measurement=anchor,
                source_step=source_step,observed_duration_s=float(r['end_s']-r['start_s']),native_source=source,
                endpoint_scope='conditional observed-duration only where native charge reference unresolved; capacity reference remains independent')
            plan.append(item)
            if anchor is not None and anchor['rpt']==stop_after_rpt:break
        else:
            if step['duration_us']==0:zero.append(dict(scenario_index=i,step=step));continue
            if step['kind'] not in ('eligible_recorded_current','recorded_rest_controller','assumed_exposure'):raise ValueError('Unsupported corrected scenario kind')
            plan.append(dict(kind=step['kind'],source_ref=ref,scenario_index=i,source_step=step))
    if found!=list(range(stop_after_rpt+1)):raise ValueError('Requested chronological full RPT targets not found')
    return s,raw_path,plan,zero,arrays,program_path


def worker(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import pybamm
    from composite_candidate_parameters import build_frozen_candidate
    root=args.root.resolve();out=args.output.resolve();began=time.monotonic();measure=[];journal=[];state=None;guards={};last=None;resume_at=0;compression=[];timings=[]
    report=dict(status='started',data_kind='corrected_conditional_history_trial',physical_validation_complete=False,measured_history_admissible=False,
        AWS_started=False,coefficient_fitting_performed=False,stop_after_rpt=args.stop_after_rpt,completed_steps=0)
    try:
        scenario,raw,plan,zero,arrays,program_path=compile_route(root,args.scenario,args.stop_after_rpt)
        rawsteps=read(raw)['cell']['steps'];programs={x['whole_file_sha256']:x for x in read(program_path)}
        p,options,meta=build_frozen_candidate(args.candidate,args.anchor)
        options['SEI']=('solvent-diffusion limited','none');options['SEI porosity change']='false'
        if any(p[prefix+'SEI solvent diffusivity [m2.s-1]']!=2.5e-22 for prefix in ('Primary: ','Secondary: ')):
            raise ValueError('This first trial must retain declared example phase SEI coefficients')
        sourcepaths=[Path(__file__),Path(__file__).with_name('bounded_current_waveform.py'),Path(__file__).with_name('corrected_history_checkpoint.py'),Path(__file__).with_name('runtime_physics_identity.py'),Path(__file__).with_name('run_exposure_scenario_smoke.py'),Path(__file__).with_name('composite_candidate_parameters.py'),Path(__file__).with_name('native_program_contracts.py'),Path(__file__).with_name('conditional_history_inputs.py'),args.scenario,raw,program_path,args.candidate,args.anchor]
        identity=capture(p,[Path(__file__).with_name(n) for n in ('composite_candidate_parameters.py','joint_spme_bol_fit.py','initialization_contract.py')])
        write(out/'runtime_identity.json',identity)
        guards={str(p.resolve()):sha(p) for p in sourcepaths};guards.update(meta['source_sha256']);guards.update(arrays.guards);guards.update(identity['source_sha256'])
        report.update(candidate_metadata=meta,options=options,planned_steps=len(plan),zero_span_annotations=len(zero),source_sha256=guards,
            temperature_C=25,all_main_RPT_exposures_propagated=True,initialization_count=1,
            assumptions=['raw eligible ageing current follows measured duration; no730mAh reset within mixedNs',
             'recorded mode3 holds0A including stale entry-current annotations',
             'unknown gaps follow explicit scenario0A25C assumption',
             'RPT nativeV/I endpoints independently predicted; unresolved noncapacityQ references use observed duration',
             'observed route retained; model clock drift not compensated; no future-SOH fitting'],
            mesh=dict(x_n=10,x_s=10,x_p=10,r_n=20,r_n_prim=20,r_n_sec=20,r_p=20),rtol=1e-7,atol=1e-9,period_s=10,
            SEI_parameter_status='example coefficient; not calibrated; one diagnostic trial')
        report['current_approximation_policy']=POLICY if args.compress_cc else None
        report['runtime_identity_sha256']=identity['sha256']
        if args.resume_directory:
            state,journal,measure,last,resume_at,proof=load_resume(args.resume_directory,args.checkpoint_sha256,root,report,plan,identity)
            proof.update(verify_rest_equations(state,p,options,report['mesh']))
            report.update(resume_proof=proof,completed_steps=resume_at,measurements=measure,initialization_count=0)
            write(out/'resume_receipt.json',proof);write(out/'measurements.json',measure)
        write(out/'declared_experiment.json',report);write(out/'route.json',dict(plan=plan,zero_span_annotations=zero));write(out/'report.json',report)
        snap=out/'source_snapshots';snap.mkdir()
        for i,(name,digest) in enumerate(guards.items()):
            path=Path(name)
            if path.suffix in ('.py','.json') and path.stat().st_size<2_000_000:(snap/f'{i:02d}_{path.name}').write_bytes(path.read_bytes())
        for begin in range(resume_at,len(plan),args.batch_size):
            if time.monotonic()-began>args.max_wall_s-10:raise TimeoutError('Declared runtime budget reached at batch boundary')
            items=plan[begin:begin+args.batch_size];controls=[];tables={};batch_started=time.monotonic()
            for item in items:
                if item['kind']=='native_RPT_control':
                    spec=item['spec'];make=getattr(pybamm.step,spec['operation'])
                    kw=dict(duration=spec['duration_s'],termination=spec['termination'],skip_ok=item['measurement'] is None and bool(spec['termination']),period=max(.01,min(10,spec['duration_s'])))
                    step=make(**kw) if spec['operation']=='rest' else make(spec['value'],**kw)
                else:
                    s=item['source_step']
                    if item['kind']=='eligible_recorded_current':
                        ar=arrays.get(s);table=waveform(s,ar)
                        if args.compress_cc:
                            cc=verified_setpoint(rawsteps[s['raw_step_index']],s,ar,programs)
                            table,cm=compress(table,cc);compression.append(dict(step=begin+len(controls),source_ref=item['source_ref'],**cm))
                        tables[len(controls)]=table
                        np.savez_compressed(out/f'input_current_{begin+len(controls):06d}.npz',model_t_s=table[:,0],model_I_A=table[:,1])
                        step=pybamm.step.current(table,duration=float(table[-1,0]),skip_ok=False,period=max(.01,min(10,float(table[-1,0]))))
                    else:
                        if item['kind']=='assumed_exposure' and (s.get('assumed_operation')!='rest' or s.get('assumed_current_A')!=0 or s.get('assumed_temperature_C')!=25):raise ValueError('Unexpected assumed exposure policy')
                        if item['kind']=='recorded_rest_controller':
                            a=arrays.get(s)
                            if not np.all(a['mode'][s['row_start']:s['row_end_exclusive']]==3):raise ValueError('Mode3 interval changed')
                        step=pybamm.step.rest(s['duration_us']/1e6,skip_ok=False,period=max(.01,min(10,s['duration_us']/1e6)))
                controls.append(step)
            sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=p.copy(),var_pts=report['mesh'],
                experiment=pybamm.Experiment([tuple(controls)],temperature=298.15),solver=pybamm.IDAKLUSolver(rtol=1e-7,atol=1e-9))
            prebuild=time.monotonic();print('batch',begin,'build start',flush=True)
            sim.build_for_experiment();built=time.monotonic();print('batch',begin,'built',built-prebuild,flush=True)
            if begin==resume_at and state is not None:
                first=next(iter(sim.steps_to_built_models.values()))
                if state_layout(state.all_models[-1])!=state_layout(first):raise ValueError('Resumed state layout differs from freshly built model')
            sol=sim.solve(calc_esoh=False,**({'starting_solution':state} if state is not None else {}));solved=time.monotonic();pieces=sol.cycles[-1].steps
            timings.append(dict(begin=begin,end=begin+len(items),prepare_s=prebuild-batch_started,build_s=built-prebuild,solve_s=solved-built))
            write(out/'batch_timings.json',timings);write(out/'compression_audit.json',compression)
            for j,piece in enumerate(pieces):
                idx=begin+j;item=items[j];rec=dict(step=idx,source_ref=item['source_ref'],kind=item['kind'],scenario_index=item['scenario_index'])
                if isinstance(piece,pybamm.EmptySolution):
                    if item['kind']!='native_RPT_control' or item['measurement'] is not None:raise ValueError('Unreviewed empty actual exposure')
                    rec.update(status='model_initial_event',actual_exposure_mismatch=True,observed_duration_s=item['observed_duration_s']);journal.append(rec);continue
                t,I,V,q=[np.asarray(piece[v].entries,dtype=float) for v in ('Time [s]','Current [A]','Voltage [V]','Discharge capacity [A.h]')]
                if last is not None and (abs(t[0]-last[0])>1e-5 or abs(q[0]-last[1])>1e-8):raise ValueError('Batch/main clock or charge discontinuity')
                last=(t[-1],q[-1]);duration=float(t[-1]-t[0]);rec.update(status='simulated',model_start_s=float(t[0]),model_end_s=float(t[-1]),duration_s=duration,termination=str(piece.termination))
                if item['kind']=='native_RPT_control':
                    spec=item['spec'];rec['model_minus_observed_duration_s']=duration-item['observed_duration_s']
                    if not spec['timed_endpoint'] and str(piece.termination)=='final time':raise ValueError('RPT event hit watchdog')
                    anchor=item['measurement']
                    if anchor is not None:
                        if str(piece.termination)!='event: Voltage < 2.5 [V] [experiment]':raise ValueError('Capacity independent cutoff failed')
                        cap=float(q[-1]-q[0]);sei=float(np.asarray(piece['Loss of capacity to negative SEI [A.h]'].entries).reshape(-1)[-1]);lli=float(np.asarray(piece['Loss of lithium inventory [%]'].entries).reshape(-1)[-1])
                        if not np.isfinite([cap,sei,lli]).all() or cap<=0:raise ValueError('Invalid RPT/mechanism result')
                        measure.append(dict(rpt_id='RPT'+str(anchor['rpt']),Qref_Ah=cap,target_Qref_Ah=anchor['performance_summary_C10_capacity_Ah'],SEI_loss_Ah=sei,LLI_percent=lli))
                        write(out/'measurements.json',measure)
                else:
                    expected=item['source_step']['duration_us']/1e6
                    if abs(duration-expected)>2e-6:raise ValueError('Observed raw exposure truncated by model: '+item['source_ref'])
                    if item['kind'] in ('recorded_rest_controller','assumed_exposure') and not np.all(I==0):raise ValueError('Rest current nonzero')
                    if item['kind']=='eligible_recorded_current':
                        table=tables[j];expected_I=np.interp(t-t[0],table[:,0],table[:,1]);applied_error=float(max(abs(I-expected_I)))
                        if applied_error>1e-8:raise ValueError('Applied current differs from declared waveform/sign')
                        expected_Ah=-item['source_step']['signed_native_current_Ah']
                        throughput=np.asarray(piece['Throughput capacity [A.h]'].entries,dtype=float)
                        from bounded_current_waveform import absolute_area
                        expected_abs_Ah=absolute_area(table[:,0],table[:,1])/3600
                        rec.update(raw_signed_Ah=expected_Ah,model_signed_Ah=float(q[-1]-q[0]),model_minus_raw_signed_Ah=float(q[-1]-q[0]-expected_Ah),
                                   applied_current_max_error_A=applied_error,declared_absolute_Ah=expected_abs_Ah,
                                   model_minus_declared_absolute_Ah=float(throughput[-1]-throughput[0]-expected_abs_Ah))
                if item.get('measurement') is not None or j==len(pieces)-1:
                    np.savez_compressed(out/f'step_{idx:06d}.npz',t_s=t,I_A=I,V=V,Q_Ah=q,terminal_y=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1))
                journal.append(rec)
            write(out/'phase_journal.json',journal)
            if len(pieces)!=len(items):raise ValueError('Incomplete batch: early model event')
            state=sol.last_state.copy()
            temporary=out/'last_state.partial.pkl'
            with temporary.open('wb') as f:pickle.dump(state,f)
            os.replace(temporary,out/'last_state.pkl')
            np.savez_compressed(out/'terminal_state.npz',t_s=np.asarray(state.t,dtype=float),
                Q_Ah=np.asarray(state['Discharge capacity [A.h]'].entries,dtype=float),
                terminal_y=np.asarray(state.all_ys[-1],dtype=float).reshape(-1))
            guards.update(arrays.guards)
            report.update(status='running',completed_steps=begin+len(items),measurements=measure,wall_s=time.monotonic()-began,
                          source_sha256=guards,checkpoint_sha256=sha(out/'last_state.pkl'))
            report['terminal_state_artifact']='terminal_state.npz'
            write(out/'progress.json',report);write(out/'report.json',report);print('completed',begin+len(items),'/',len(plan),'RPT',len(measure),flush=True)
            del sim,sol,pieces,controls;gc.collect()
        if [x['rpt_id'] for x in measure]!=['RPT'+str(i) for i in range(args.stop_after_rpt+1)]:raise ValueError('Missing actual RPT series')
        predicted=[100*x['Qref_Ah']/measure[0]['Qref_Ah'] for x in measure];observed=[100*x['target_Qref_Ah']/measure[0]['target_Qref_Ah'] for x in measure]
        errors=np.array(predicted)-observed
        report.update(status='conditional_trial_complete',measurements=measure,predicted_SOH_percent=predicted,observed_SOH_percent=observed,
            errors_pp=errors.tolist(),conditional_RMSE_pp=float(np.sqrt(np.mean(errors**2))),conditional_max_abs_pp=float(max(abs(errors))),
            scored_scope='bounded C diagnostic only; no independent validation or full-curve acceptance')
    except Exception as exc:
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if (out/'progress.json').exists():report=read(out/'progress.json')
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),active_attempt_journal_steps=len(journal))
    guards.update(arrays.guards if 'arrays' in locals() else {});report.update(source_sha256=guards,source_hashes_unchanged=bool(guards) and all(sha(p)==h for p,h in guards.items()),wall_s=time.monotonic()-began)
    if report['source_hashes_unchanged'] is False:report['status']='failed_source_integrity'
    write(out/'report.json',report);print(json.dumps({k:report.get(k) for k in ('status','completed_steps','measurements','conditional_RMSE_pp','wall_s','error')},ensure_ascii=False));return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--scenario',type=Path,required=True)
    a.add_argument('--candidate',type=Path,required=True);a.add_argument('--anchor',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--stop-after-rpt',type=int,default=1);a.add_argument('--batch-size',type=int,default=64);a.add_argument('--max-wall-s',type=int,default=600);a.add_argument('--worker',action='store_true')
    a.add_argument('--compress-cc',action='store_true');a.add_argument('--resume-directory',type=Path);a.add_argument('--checkpoint-sha256');v=a.parse_args()
    if bool(v.resume_directory)!=bool(v.checkpoint_sha256):a.error('Resume directory and verified checkpoint digest must be supplied together')
    if v.worker:r=worker(v);raise SystemExit(0 if r['status']=='conditional_trial_complete' else 2)
    v.output.mkdir(parents=True,exist_ok=False)
    try:
        with (v.output/'worker.log').open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],stdout=log,stderr=subprocess.STDOUT,timeout=v.max_wall_s,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        receipt=dict(status='worker_finished',exit_code=p.returncode)
    except subprocess.TimeoutExpired:
        receipt=dict(status='worker_terminated_at_wall_budget',max_wall_s=v.max_wall_s)
        report_path=v.output/'report.json'
        if report_path.exists():
            partial=read(report_path);partial.update(status='terminated_at_declared_wall_budget',RPT1_reached=False,
                full_curve_score_available=False,partial_active_batch_not_claimed_complete=True)
            write(report_path,partial)
    write(v.output/'run_receipt.json',receipt);print(json.dumps(receipt));raise SystemExit(0 if receipt.get('exit_code')==0 else 2)
