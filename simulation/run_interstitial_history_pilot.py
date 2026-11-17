"""Isolated C-only chronological interstitial-SEI pilot, gated before execution.

No fitting loop, prior solvent checkpoint, or independent-condition selection.
"""
from __future__ import annotations
import argparse,gc,json,os,subprocess,sys,time
from pathlib import Path
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_thread_variable]='1'
os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
import numpy as np
from run_exposure_scenario_smoke import read,write,sha
from run_corrected_history_trial import compile_route,waveform
from bounded_current_waveform import compress,verified_setpoint,absolute_area,POLICY
from interstitial_pilot_support import (validate_contract,prepare_parameters,rest_batches,voltage_extrema,raw_voltage,
    validate_charge_residuals,bind_rpt_artifact,candidate_key,commit_checkpoint,load_checkpoint,native_control_evidence,
    transition_state_evidence,record_worker_termination)

def prepare(args,executing=False):
    root=args.root.resolve();out=args.output.resolve()
    out.relative_to((root/'work/validation_proposal_20261103/latest_spme_review_20261103').resolve())
    contract=validate_contract(args.contract,args.dint_multiplier,args.execute_reviewed_contract,executing)
    if not 30<=args.max_wall_s<=contract['runtime']['chunk_wall_limit_s']:raise ValueError('Chunk wall budget must be30..600s')
    scenario,raw,plan,zero,arrays,program_path=compile_route(root,args.scenario,1)
    if scenario['cell_id']!='C':raise ValueError('Only C history can enter this pilot')
    batches=list(rest_batches(plan,0,contract['runtime']['max_batch_steps']))
    p,options,meta,identity=prepare_parameters(args.candidate,args.anchor,contract,args.dint_multiplier)
    sources=[Path(__file__),args.contract,args.scenario,raw,program_path,args.candidate,args.anchor]
    sources.extend(Path(__file__).with_name(n) for n in ('interstitial_pilot_support.py','run_corrected_history_trial.py',
        'corrected_history_checkpoint.py','runtime_physics_identity.py','bounded_current_waveform.py','run_exposure_scenario_smoke.py',
        'native_program_contracts.py','conditional_history_inputs.py'))
    guards={str(v.resolve()):sha(v) for v in sources};guards.update(meta['source_sha256']);guards.update(identity['source_sha256']);guards.update(arrays.guards)
    report=dict(status='prepared',role=contract['role'],candidate_key=candidate_key(sha(args.contract),args.dint_multiplier,identity),
        contract_sha256=sha(args.contract),Dint_multiplier=args.dint_multiplier,Dint_both_phases_m2_s=1e-20*args.dint_multiplier,
        SEI_activation_energy_J_mol=38000.,candidate_metadata=meta,runtime_identity_sha256=identity['sha256'],
        options=options,mesh=contract['mesh'],rtol=contract['rtol'],atol=contract['atol'],period_s=contract['period_s'],
        planned_steps=len(plan),stop_after_rpt=1,completed_steps=0,planned_batches=batches,zero_span_annotations=len(zero),
        temperature_C=25.,source_sha256=guards,coefficient_fitting_performed=False,initialization_count=1,
        physical_validation_complete=False,measured_history_admissible=False,lifetime_restart_admitted=False,
        charge_optimization_or_120_labels_admitted=False,AWS_started=False,
        current_approximation_policy=POLICY,unknown_gap_policy='Declared0A25C assumption, not recovered history',
        SEI_parameter_status='new interstitial mechanism; installed Dint reference times common engineering multiplier; no Dsol multiplier transfer')
    return contract,scenario,raw,plan,zero,arrays,program_path,p,options,identity,report

def prepare_only(args):
    import pybamm
    began=time.monotonic()
    c,s,raw,plan,zero,arrays,program,p,options,identity,report=prepare(args,False)
    rawsteps=read(raw)['cell']['steps'];evidence=[]
    for idx,item in enumerate(plan):evidence.append(dict(step=idx,source_ref=item['source_ref'],kind=item['kind'],raw_voltage=raw_voltage(item,rawsteps,arrays)))
    # Discretisation only; this checks actual parameter names/phase support without solving.
    sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=p.copy(),var_pts=c['mesh'],
        experiment=pybamm.Experiment([pybamm.step.rest(1)],temperature=298.15),solver=pybamm.IDAKLUSolver(rtol=c['rtol'],atol=c['atol']))
    sim.build_for_experiment()
    report['source_sha256'].update(arrays.guards)
    report.update(status='prepared_build_only_no_PDE',model_build_passed=True,PDE_calls=0,
        all_source_hashes_unchanged=all(sha(path)==h for path,h in report['source_sha256'].items()),
        raw_voltage_segments=len(evidence),raw_nominal_limit_violating_segments=sum(v['raw_voltage'].get('nominal_limit_violation',False) for v in evidence),
        unknown_voltage_segments=sum(not v['raw_voltage']['available'] for v in evidence),
        actual_short_pilot_runtime_verified=False,actual_checkpoint_resume_with_interstitial_verified=False,
        execution_requires_root_contract_review=True,wall_s=time.monotonic()-began)
    out=args.output.resolve();write(out/'preparation_receipt.json',report);write(out/'runtime_identity.json',identity)
    write(out/'route.json',dict(plan=plan,zero_span_annotations=zero));write(out/'raw_voltage_inventory.json',evidence)
    print(json.dumps({k:report[k] for k in ('status','planned_steps','model_build_passed','PDE_calls','all_source_hashes_unchanged','raw_voltage_segments','raw_nominal_limit_violating_segments','unknown_voltage_segments')}))
    return report

def worker(args):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import pybamm
    began=time.monotonic();out=args.output.resolve();state=None;last=None;journal=[];measure=[];compression=[];timings=[];resume_at=0;baseline_Li=None
    report=dict(status='started',physical_validation_complete=False,lifetime_restart_admitted=False,AWS_started=False)
    guards={}
    try:
        contract,scenario,raw,plan,zero,arrays,program_path,p,options,identity,report=prepare(args,True)
        guards=report['source_sha256'];rawsteps=read(raw)['cell']['steps'];programs={v['whole_file_sha256']:v for v in read(program_path)}
        if args.resume_checkpoint:
            state,journal,measure,last,resume_at,proof=load_checkpoint(args.resume_checkpoint,args.manifest_sha256,report,plan,identity,args.root.resolve(),p,options,contract['mesh'])
            report.update(resume_proof=proof,completed_steps=resume_at,initialization_count=0)
            baseline_Li=read(Path(args.resume_checkpoint)/'report.json')['initial_total_Li_plus_SEI_Ah']
            write(out/'resume_receipt.json',proof)
        write(out/'runtime_identity.json',identity);write(out/'declared_experiment.json',report);write(out/'route.json',dict(plan=plan,zero_span_annotations=zero));write(out/'report.json',report)
        snap=out/'source_snapshots';snap.mkdir()
        for i,(name,h) in enumerate(guards.items()):
            path=Path(name)
            if path.suffix in ('.py','.json') and path.stat().st_size<2_000_000:(snap/f'{i:02d}_{path.name}').write_bytes(path.read_bytes())
        for begin,end in rest_batches(plan,resume_at,contract['runtime']['max_batch_steps']):
            if time.monotonic()-began>args.max_wall_s-30:raise TimeoutError('Checked chunk budget reached at committed batch boundary')
            items=plan[begin:end];controls=[];tables={};batch_start=time.monotonic();rawV={}
            for offset,item in enumerate(items):
                rawV[offset]=raw_voltage(item,rawsteps,arrays)
                if item['kind']=='native_RPT_control':
                    spec=item['spec'];make=getattr(pybamm.step,spec['operation'])
                    kw=dict(duration=spec['duration_s'],termination=spec['termination'],skip_ok=item['measurement'] is None and bool(spec['termination']),period=max(.01,min(10,spec['duration_s'])))
                    step=make(**kw) if spec['operation']=='rest' else make(spec['value'],**kw)
                elif item['kind']=='eligible_recorded_current':
                    source=item['source_step'];a=arrays.get(source);table=waveform(source,a)
                    cc=verified_setpoint(rawsteps[source['raw_step_index']],source,a,programs)
                    table,audit=compress(table,cc);compression.append(dict(step=begin+offset,source_ref=item['source_ref'],**audit));tables[offset]=table
                    np.savez_compressed(out/f'input_current_{begin+offset:06d}.npz',model_t_s=table[:,0],model_I_A=table[:,1])
                    step=pybamm.step.current(table,duration=float(table[-1,0]),skip_ok=False,period=max(.01,min(10,float(table[-1,0]))))
                else:
                    source=item['source_step']
                    if item['kind']=='assumed_exposure' and (source.get('assumed_operation')!='rest' or source.get('assumed_current_A')!=0 or source.get('assumed_temperature_C')!=25):raise ValueError('Unexpected unknown-gap assumption')
                    if item['kind']=='recorded_rest_controller' and not np.all(arrays.get(source)['mode'][source['row_start']:source['row_end_exclusive']]==3):raise ValueError('Rest controller mode changed')
                    step=pybamm.step.rest(source['duration_us']/1e6,skip_ok=False,period=max(.01,min(10,source['duration_us']/1e6)))
                controls.append(step)
            sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=p.copy(),var_pts=contract['mesh'],
                experiment=pybamm.Experiment([tuple(controls)],temperature=298.15),solver=pybamm.IDAKLUSolver(rtol=contract['rtol'],atol=contract['atol']))
            prebuild=time.monotonic();sim.build_for_experiment();built=time.monotonic()
            sol=sim.solve(calc_esoh=False,**({'starting_solution':state} if state is not None else {}));solved=time.monotonic();pieces=sol.cycles[-1].steps
            if state is not None:
                first=next(piece for piece in pieces if not isinstance(piece,pybamm.EmptySolution))
                transition=transition_state_evidence(state,first)
                write(out/f'transition_{begin:06d}.json',transition)
            timings.append(dict(begin=begin,end=end,prepare_s=prebuild-batch_start,build_s=built-prebuild,solve_s=solved-built))
            write(out/'batch_timings.json',timings);write(out/'compression_audit.json',compression)
            for offset,piece in enumerate(pieces):
                idx=begin+offset;item=items[offset];rec=dict(step=idx,source_ref=item['source_ref'],kind=item['kind'],scenario_index=item['scenario_index'],raw_voltage=rawV[offset])
                if isinstance(piece,pybamm.EmptySolution):
                    if item['kind']!='native_RPT_control' or item['measurement'] is not None:raise ValueError('Unreviewed empty exposure')
                    rec.update(status='model_initial_event',actual_exposure_mismatch=True,observed_duration_s=item['observed_duration_s'],model_voltage_available=False)
                    journal.append(rec);continue
                t,I,V,q,throughput,Li,sei,sp,ss=[np.asarray(piece[v].entries,dtype=float) for v in ('Time [s]','Current [A]','Voltage [V]',
                    'Discharge capacity [A.h]','Throughput capacity [A.h]','Total lithium in particles [mol]',
                    'Loss of capacity to negative SEI [A.h]','Loss of capacity to negative primary SEI [A.h]','Loss of capacity to negative secondary SEI [A.h]')]
                if not all(np.isfinite(a).all() for a in (t,I,V,q,throughput,Li,sei,sp,ss)):raise ValueError('Nonfinite model trajectory')
                if last is not None and (abs(t[0]-last[0])>1e-5 or abs(q[0]-last[1])>1e-8):raise ValueError('Main clock or accumulated charge discontinuity')
                last=(float(t[-1]),float(q[-1]));duration=float(t[-1]-t[0]);signed=float(q[-1]-q[0]);absolute=float(throughput[-1]-throughput[0])
                conserved=Li*96485.33212331001/3600+sei
                if baseline_Li is None:baseline_Li=float(conserved[0])
                Li_error=float(np.max(abs(conserved-baseline_Li)))
                rec.update(status='simulated',model_start_s=float(t[0]),model_end_s=float(t[-1]),duration_s=duration,termination=str(piece.termination),
                    model_voltage_available=True,model_voltage=voltage_extrema(V),model_signed_Ah=signed,model_absolute_Ah=absolute,
                    model_minus_output_sample_integral_Ah=signed-float(np.trapezoid(I,t)/3600),
                    output_sample_integral_scope='Output quadrature only; not a solver-current residual for dynamic nativeCV',
                    Li_plus_SEI_conservation_error_Ah=Li_error,phase_SEI_aggregate_error_Ah=float(np.max(abs(sei-sp-ss))))
                # Keep evidence from the failing piece in the active journal as well.
                journal.append(rec)
                artifact=out/f'step_{idx:06d}.npz'
                np.savez_compressed(artifact,t_s=t,I_A=I,V=V,Q_Ah=q,throughput_Ah=throughput,Li_mol=Li,SEI_Ah=sei,SEI_primary_Ah=sp,SEI_secondary_Ah=ss,terminal_y=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1))
                if Li_error>contract['solver_diagnostic_guards']['Li_plus_SEI_conservation_max_error_Ah'] or rec['phase_SEI_aggregate_error_Ah']>1e-10:raise ValueError('SEI aggregate/lithium conservation diagnostic guard failed')
                if item['kind']=='native_RPT_control':
                    spec=item['spec'];rec['model_minus_observed_duration_s']=duration-item['observed_duration_s']
                    native_control_evidence(spec,I,V,duration,str(piece.termination),contract,evidence=rec)
                    if spec['operation'] in ('current','rest'):
                        expected_current=spec['value'] if spec['operation']=='current' else 0.
                        applied=float(np.max(abs(I-expected_current)));sr=signed-expected_current*duration/3600;ar=absolute-abs(expected_current)*duration/3600
                        rec.update(applied_current_max_error_A=applied,solver_signed_Ah_residual=sr,solver_absolute_Ah_residual=ar)
                        if applied>1e-8:raise ValueError('Native current control mismatch')
                        validate_charge_residuals(sr,ar,contract)
                    if item['measurement'] is not None:
                        anchor=item['measurement']
                        if str(piece.termination)!='event: Voltage < 2.5 [V] [experiment]':raise ValueError('Independent RPT capacity cutoff absent')
                        m=dict(rpt_id='RPT'+str(anchor['rpt']),Qref_Ah=signed,target_Qref_Ah=anchor['performance_summary_C10_capacity_Ah'],SEI_loss_Ah=float(sei[-1]))
                        bind_rpt_artifact(artifact,m);measure.append(m)
                else:
                    if abs(duration-item['source_step']['duration_us']/1e6)>2e-6:raise ValueError('Observed exposure truncated/expanded')
                    if item['kind']=='eligible_recorded_current':
                        table=tables[offset];expected=np.interp(t-t[0],table[:,0],table[:,1]);applied=float(np.max(abs(I-expected)))
                        declared_signed=float(np.trapezoid(table[:,1],table[:,0])/3600);declared_abs=absolute_area(table[:,0],table[:,1])/3600
                        sr=signed-declared_signed;ar=absolute-declared_abs
                        rec.update(raw_signed_Ah=-item['source_step']['signed_native_current_Ah'],declared_signed_Ah=declared_signed,declared_absolute_Ah=declared_abs,
                            applied_current_max_error_A=applied,solver_signed_Ah_residual=sr,solver_absolute_Ah_residual=ar)
                        if applied>1e-8:raise ValueError('Actual waveform differs from declared input')
                        validate_charge_residuals(sr,ar,contract)
                    else:
                        if not np.all(I==0):raise ValueError('Rest has nonzero current')
                        rec.update(solver_signed_Ah_residual=signed,solver_absolute_Ah_residual=absolute);validate_charge_residuals(signed,absolute,contract)
            if len(pieces)!=len(items):raise ValueError('Incomplete batch after early model event')
            state=sol.last_state.copy();guards.update(arrays.guards)
            report.update(status='running',completed_steps=end,measurements=measure,initial_total_Li_plus_SEI_Ah=baseline_Li,
                source_sha256=guards,wall_s=time.monotonic()-began,model_nominal_voltage_violating_segments=sum(r.get('model_voltage',{}).get('nominal_limit_violation',False) for r in journal))
            if end<len(plan):
                receipt=commit_checkpoint(out,state,report,plan,journal,measure,identity);report['latest_checkpoint']=receipt
            write(out/'phase_journal.json',journal);write(out/'measurements.json',measure);write(out/'report.json',report)
            print('committed',end,'/',len(plan),'RPT',len(measure),flush=True)
            del sim,sol,pieces,controls;gc.collect()
        if [m['rpt_id'] for m in measure]!=['RPT0','RPT1']:raise ValueError('Complete C RPT0/1 sequence required')
        predicted=[100*m['Qref_Ah']/measure[0]['Qref_Ah'] for m in measure];observed=[100*m['target_Qref_Ah']/measure[0]['target_Qref_Ah'] for m in measure]
        errors=np.asarray(predicted)-observed
        report.update(status='interstitial_pilot_complete',predicted_SOH_percent=predicted,observed_SOH_percent=observed,errors_pp=errors.tolist(),
            conditional_RMSE_pp=float(np.sqrt(np.mean(errors**2))),conditional_max_abs_pp=float(np.max(abs(errors))),fixed_Q0_Ah=measure[0]['Qref_Ah'],
            scored_scope='C RPT0/1 conditional pilot only; not full calibration or independent-condition validation')
    except Exception as exc:
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write(out/'active_batch_partial_journal.json',journal)
        report.update(status='checked_budget_boundary' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc),partial_active_batch_not_committed=True)
    guards.update(arrays.guards if 'arrays' in locals() else {})
    report.update(source_sha256=guards,source_hashes_unchanged=bool(guards) and all(sha(path)==h for path,h in guards.items()),wall_s=time.monotonic()-began)
    if not report['source_hashes_unchanged']:report['status']='failed_source_integrity'
    write(out/'report.json',report);print(json.dumps({k:report.get(k) for k in ('status','completed_steps','conditional_RMSE_pp','wall_s','error')}),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser()
    for name in ('root','scenario','candidate','anchor','contract','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dint-multiplier',type=float,required=True);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--execute-reviewed-contract');parser.add_argument('--resume-checkpoint',type=Path);parser.add_argument('--manifest-sha256')
    parser.add_argument('--max-wall-s',type=int,default=600);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if bool(args.resume_checkpoint)!=bool(args.manifest_sha256):parser.error('Resume checkpoint and explicit trusted manifest SHA must be supplied together')
    if args.prepare_only and args.resume_checkpoint:parser.error('Preparation cannot resume a checkpoint')
    if args.worker:
        r=worker(args);return 0 if r['status']=='interstitial_pilot_complete' else 2
    validate_contract(args.contract,args.dint_multiplier,args.execute_reviewed_contract,not args.prepare_only)
    args.output.resolve().relative_to((args.root.resolve()/'work/validation_proposal_20261103/latest_spme_review_20261103').resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    if args.prepare_only:prepare_only(args);return 0
    try:
        with (args.output/'worker.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],stdout=log,stderr=subprocess.STDOUT,
                timeout=args.max_wall_s,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        receipt=dict(status='worker_finished',exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        receipt=dict(status='worker_terminated_at_wall_budget',max_wall_s=args.max_wall_s)
    receipt=record_worker_termination(args.output,receipt)
    print(json.dumps(receipt));return 0 if receipt.get('exit_code')==0 else 2

if __name__=='__main__':raise SystemExit(main())
