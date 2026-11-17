"""Bounded PDE smoke for corrected exposure scenarios. Never physical admission.

Runs RPT0 with independent capacity endpoints, and separately a fresh-state
F025 mixed-mode diagnostic. The latter is not the actual aged F025 cell state.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def validate_input(path):
    path=Path(path).resolve();s=read(path)
    if s.get('data_kind')!='diagnostic_exposure_scenario' or s.get('physical_validation_complete') is not False or s.get('measured_history_admissible') is not False:
        raise ValueError('Explicit non-admitted diagnostic scenario required')
    raw=Path(s['raw_reconstruction_path']).resolve()
    if sha(raw)!=s['raw_reconstruction_sha256']:raise ValueError('Raw reconstruction changed')
    r=read(raw)
    if r.get('data_kind')!='recorded_exposure_reconstruction' or r['cell']['cell_id']!=s['cell_id']:
        raise ValueError('Scenario source identity mismatch')
    if s.get('all_time_accounted_once') is not True or s.get('raw_current_waveform_preserved') is not True:
        raise ValueError('Scenario clock/current audit missing')
    return s,raw


def current_window(step,raw_directory):
    if step['kind']!='eligible_recorded_current':raise ValueError('Only eligible raw current can be interpolated')
    path=(raw_directory/step['arrays_file']).resolve();path.relative_to(raw_directory)
    if sha(path)!=step['arrays_sha256']:raise ValueError('Raw source array changed')
    with np.load(path,allow_pickle=False) as a:
        lo,hi=step['row_start'],step['row_end_exclusive']
        t=np.array(a['t_s'][lo:hi]);I=np.array(a['I_mA'][lo:hi]);mask=a['current_drive_eligible_interval'][lo:hi-1]
        mode=np.array(a['mode'][lo:hi]);ns=np.array(a['Ns'][lo:hi])
    if len(t)<2 or not np.all(mask) or not np.all(np.diff(t)>0) or not np.all(np.isfinite([t,I])):
        raise ValueError('Noneligible/nonmonotone/nonfinite current interval')
    if len(np.unique(mode))!=1 or mode[0] not in (1,2) or len(np.unique(ns))!=1:
        raise ValueError('Raw current drive crosses native mode/Ns boundary')
    if t[0]!=step['start_s'] or t[-1]!=step['end_s']:raise ValueError('Current source window mismatch')
    duration=float(t[-1]-t[0]);delta=duration-step['duration_us']/1e6
    if abs(delta)>1.5e-6:raise ValueError('Current duration differs beyond canonical microsecond rounding')
    table=np.column_stack([t-t[0],-I/1000])
    return table,dict(source_path=str(path),source_sha256=sha(path),duration_s=duration,canonical_duration_us=step['duration_us'],
        model_minus_canonical_duration_s=delta,model_current_integral_Ah=float(np.trapezoid(table[:,1],table[:,0])/3600),
        native_current_integral_Ah=float(np.trapezoid(I,t)/3600000),raw_rows_preserved=len(t))


def rest_metadata(step,raw_directory):
    if step['kind']=='assumed_exposure':
        if step.get('assumed_operation')!='rest' or step.get('assumed_current_A')!=0 or step.get('assumed_temperature_C')!=25:
            raise ValueError('This bounded smoke requires explicitly declared zero-current25C gap policy')
        return dict(operation_source='explicit_assumption',assumption_id=step['assumption_id'],raw_stale_current_mA=None)
    if step['kind']!='recorded_rest_controller' or step.get('recorded_rest_current_A')!=0:
        raise ValueError('Source-supported mode3 rest required')
    path=(raw_directory/step['arrays_file']).resolve();path.relative_to(raw_directory)
    if sha(path)!=step['arrays_sha256']:raise ValueError('Rest source changed')
    with np.load(path,allow_pickle=False) as a:
        lo,hi=step['row_start'],step['row_end_exclusive'];m=a['mode'][lo:hi];I=a['I_mA'][lo:hi]
        if not len(m) or not np.all(m==3):raise ValueError('Recorded rest crosses a mode boundary')
        conflict=float(I[np.argmax(abs(I))]) if len(I) else None
    return dict(operation_source='recorded_mode3_controller',raw_stale_current_mA=conflict,
        raw_current_interpolation_permitted=False,source_path=str(path),source_sha256=sha(path))


def prepare(root,scenario_path,contract_path,case):
    s,raw=validate_input(scenario_path);contract=read(contract_path)
    for p,h in contract['source_sha256'].items():
        if sha(p)!=h:raise ValueError('RPT native contract/source changed')
    steps=s['steps'];plan=[]
    if case=='F025_mixed_mode_fresh':
        selected=[(i,p) for i,p in enumerate(steps) if p.get('source_ref')=='F025#r00103']
        if not selected or not any(p['kind']=='eligible_recorded_current' for i,p in selected) or not any(p['kind']=='recorded_rest_controller' for i,p in selected):
            raise ValueError('Mixed source must include both current and recorded rest')
        for i,p in selected:plan.append(dict(kind='scenario',scenario_index=i,source=p))
        # Source-ref-only selection must never silently omit intervening exposures.
        indexes=[i for i,p in selected]
        if indexes!=list(range(indexes[0],indexes[-1]+1)):
            raise ValueError('Selected mixed source route is not contiguous')
    elif case=='RPT0_following_rest':
        specs=[p for p in contract['specifications'] if p['file']=='F045']
        if [p['Ns'] for p in specs]!=list(range(10)):raise ValueError('F045 complete actual RPT route required')
        plan.extend(dict(kind='native_RPT_control',spec=p) for p in specs)
        first=next(i for i,p in enumerate(steps) if p.get('source_ref')=='F046#r00000')
        prior=steps[first-1]
        if prior['kind']!='assumed_exposure' or prior.get('unresolved_kind')!='unobserved_gap':
            raise ValueError('Explicit inter-file gap must precede F046')
        plan.extend([dict(kind='scenario',scenario_index=first-1,source=prior),dict(kind='scenario',scenario_index=first,source=steps[first])])
    else:raise ValueError('Named bounded smoke required')
    return s,raw,plan


def run(root,scenario_path,contract_path,output,case):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import pybamm
    from joint_spme_bol_fit import derive_updates,homogeneous_state,select_rest_anchor
    from initialize_spme_from_rest import load_native
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    record=dict(status='started',case=case,physical_validation_complete=False,measured_history_admissible=False,
        lifetime_restart_admitted=False,ground_truth_export_admitted=False,model_solve_calls=0,temperature_C=25,
        model_time_and_source_time_separate=True,AWS_started=False)
    guards={};journal=[]
    try:
        s,raw,plan=prepare(root,scenario_path,contract_path,case)
        base=root/'work/validation_proposal_20261103';mfile=base/'physical_BOL_fit_v3/parameters_frozen.json';configfile=mfile.with_name('declared_experiment.json')
        material,config=read(mfile),read(configfile)
        source_files=[Path(__file__),Path(__file__).with_name('joint_spme_bol_fit.py'),Path(__file__).with_name('initialize_spme_from_rest.py'),Path(scenario_path),Path(contract_path),raw,mfile,configfile]
        guards={str(p.resolve()):sha(p) for p in source_files}
        p=pybamm.ParameterValues('OKane2022');updates,inventory=derive_updates(p,material['s_Q'],material['s_Q'],material['delta_Q_Li_Ah']);p.update(updates)
        native,source=load_native(root,'C');anchor,anchor_meta=select_rest_anchor(native[0]);state,state_meta=homogeneous_state(p,inventory,anchor);p.update(state)
        guards[source['path']]=sha(source['path']);p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
        controls=[];control_meta=[];current_tables={};zero_span_annotations=[]
        for plan_index,item in enumerate(plan):
            if item['kind']=='scenario' and item['source']['duration_us']==0:
                zero_span_annotations.append(item);continue
            index=len(controls)
            if item['kind']=='native_RPT_control':
                spec=item['spec'];args=dict(duration=spec['duration'],skip_ok=False)
                if 'termination' in spec:args['termination']=spec['termination']
                op=getattr(pybamm.step,spec['operation']);step=op(**args) if spec['operation']=='rest' else op(spec['value'],**args)
                meta=dict(kind=item['kind'],source_ref=f"F045#r{spec['run_index']:05d}",spec=spec)
            else:
                source_step=item['source'];kind=source_step['kind']
                if kind=='eligible_recorded_current':
                    table,meta=current_window(source_step,raw.parent);current_tables[index]=table
                    step=pybamm.step.current(table,duration=meta['duration_s'],skip_ok=False)
                    np.savez_compressed(out/f'input_current_{index:03d}.npz',model_t_s=table[:,0],model_I_A=table[:,1])
                else:
                    meta=rest_metadata(source_step,raw.parent);step=pybamm.step.rest(source_step['duration_us']/1e6,skip_ok=False)
                if 'source_path' in meta:guards[meta['source_path']]=meta['source_sha256']
                meta.update(kind=kind,scenario_index=item['scenario_index'],source_ref=source_step['source_ref'],canonical_duration_us=source_step['duration_us'],source=source_step)
            controls.append(step);control_meta.append(meta)
        record.update(initial_state=state_meta,initial_concentrations=state,material_sha256=sha(mfile),
            scenario_sha256=sha(scenario_path),raw_reconstruction_sha256=sha(raw),source_sha256=guards,planned_steps=len(controls),
            control_metadata=control_meta,zero_span_annotations=zero_span_annotations,raw_zero_spans_not_claimed_as_missing_exposure=True,initialization_scope='fresh retained-v3 C anchor; F025 case is not actual aged-state reconstruction',
            source_clock_compensation_applied=False,actual_RPT_state_propagated=True,virtual_RPT_executed=False,
            mesh_points=config['fit']['mesh_points'],rtol=1e-7,atol=1e-9,output_period_s=10)
        write(out/'declared_experiment.json',record);write(out/'report.json',record)
        snapshots=out/'source_snapshots';snapshots.mkdir()
        for i,(name,h) in enumerate(guards.items()):
            path=Path(name)
            if path.suffix in ('.py','.json') and path.stat().st_size<2_000_000:(snapshots/f'{i:02d}_{path.name}').write_bytes(path.read_bytes())
        model=pybamm.lithium_ion.SPMe(options=config['options'])
        sim=pybamm.Simulation(model,parameter_values=p,experiment=pybamm.Experiment([tuple(controls)],period=10,temperature=298.15),
            var_pts=config['fit']['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=1e-7,atol=1e-9))
        record['model_solve_calls']=1;write(out/'report.json',record);solution=sim.solve(calc_esoh=False);pieces=solution.cycles[-1].steps
        previous=None
        for i,piece in enumerate(pieces):
            t,V,I,q=[np.asarray(piece[v].entries,dtype=float) for v in ('Time [s]','Voltage [V]','Current [A]','Discharge capacity [A.h]')]
            arrays=dict(t_s=t,V=V,I_A=I,Q_Ah=q,terminal_y=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1));np.savez_compressed(out/f'phase_{i:03d}.npz',**arrays)
            meta=control_meta[i];phase=dict(step=i,source_ref=meta['source_ref'],kind=meta['kind'],duration_s=float(t[-1]-t[0]),
                model_start_s=float(t[0]),model_end_s=float(t[-1]),termination=str(piece.termination),transferred_Ah=float(q[-1]-q[0]))
            if previous and (abs(t[0]-previous[0])>1e-5 or abs(q[0]-previous[1])>1e-9):raise ValueError('Chronological main-state clock/capacity discontinuity')
            previous=(t[-1],q[-1])
            if meta['kind']=='native_RPT_control':
                spec=meta['spec'];phase['observed_duration_s']=spec['observed_duration_s'];phase['model_minus_observed_duration_s']=phase['duration_s']-spec['observed_duration_s']
                if spec.get('is_capacity'):
                    if str(piece.termination)!='event: Voltage < 2.5 [V] [experiment]':raise ValueError('RPT independent2.5V capacity event absent')
                    record['RPT0_Qref_Ah']=float(q[-1]-q[0]);record['RPT0_capacity_source']='independent model cutoff, never observed timed capacity'
                elif 'termination' in spec and str(piece.termination)=='final time':raise ValueError('RPT controller reached watchdog')
            else:
                phase['canonical_duration_us']=meta['canonical_duration_us'];phase['model_minus_canonical_s']=phase['duration_s']-meta['canonical_duration_us']/1e6
                if abs(phase['model_minus_canonical_s'])>2e-6:raise ValueError('Scenario exposure duration was truncated or expanded')
                if meta['kind']=='eligible_recorded_current':
                    table=current_tables[i];expected=np.interp(t-t[0],table[:,0],table[:,1]);phase['current_waveform_max_abs_error_A']=float(np.max(abs(I-expected)))
                    if phase['current_waveform_max_abs_error_A']>1e-8:raise ValueError('Native current waveform/sign changed')
                elif not np.all(I==0):raise ValueError('Mode3 or assumed rest must apply zero current')
            journal.append(phase);write(out/'phase_journal.json',journal)
        if len(pieces)!=len(controls):raise ValueError('Model stopped before complete bounded route')
        record.update(status='conditional_scenario_smoke_complete',completed_steps=len(pieces),phase_results=journal,
            canonical_scenario_duration_us=sum(m.get('canonical_duration_us',0) for m in control_meta),
            model_terminal_state_file=f'phase_{len(pieces)-1:03d}.npz')
    except Exception as exc:
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');record.update(status='failed',error_type=type(exc).__name__,error=str(exc),completed_steps=len(journal))
    record['source_hashes_unchanged']=bool(guards) and all(sha(p)==h for p,h in guards.items());record['wall_s']=time.monotonic()-began
    if not record['source_hashes_unchanged']:record['status']='failed_source_integrity'
    write(out/'report.json',record);return record


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--scenario',type=Path,required=True)
    a.add_argument('--rpt-contract',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--case',choices=('RPT0_following_rest','F025_mixed_mode_fresh'),required=True);v=a.parse_args()
    r=run(v.root,v.scenario,v.rpt_contract,v.output,v.case)
    print(json.dumps({k:r.get(k) for k in ('status','case','completed_steps','RPT0_Qref_Ah','wall_s','error')},ensure_ascii=False))
    raise SystemExit(0 if r['status']=='conditional_scenario_smoke_complete' else 2)
