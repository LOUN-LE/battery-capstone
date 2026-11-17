"""Bounded conditional C-only SEI fit, followed by frozen-parameter D evaluation.

This deliberately does not relax the measured campaign admission gate. Missing
exposure and branch routing remain assumptions; results cannot certify real SOH.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
from conditional_calibration_review import (CalibrationNotAccepted, bind_targets, bound_review,
    pinned, review_result, score_curve, validate_criteria)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def normalized_errors(predicted,targets):
    if (len(predicted)!=len(targets) or len(predicted)<2 or
        any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
            for v in list(predicted)+list(targets))):
        raise ValueError('Complete positive capacity series required')
    return [100*(p/predicted[0]-t/targets[0]) for p,t in zip(predicted,targets)]


def controller_boundary_empty_allowed(spec, *, is_capacity_measurement):
    """A native V/I controller may terminate immediately at its boundary.

    This is a model-predicted exposure mismatch, not proof that the recorded
    pulse was empty. Full reference discharge must never be skipped.
    """
    return (not is_capacity_measurement and spec['operation'] in ('current','voltage')
            and bool(spec['termination']) and any(l['quantity'] in ('voltage_V','absolute_current_A')
                                                 for l in spec['native_limits']))


def validate_replay_input_integrity(payload):
    if payload.get('data_kind')!='conditional_observed_program_route' or payload.get('measured_history_admissible') is not False:
        raise ValueError('Explicit conditional input required')
    # v1 collapsed observed mode changes within a repeated Ns into one native
    # controller, losing multi-hour recorded rest. Historical results remain
    # readable for audit, but the old input must never launch another PDE fit.
    if payload.get('schema_version')!=2:
        raise ValueError('Legacy Ns-aggregated replay retired: raw mode boundaries and internal time intervals must be reconstructed')
    integrity=payload.get('exposure_integrity',{})
    if any(integrity.get(k) is not True for k in ('raw_mode_boundaries_preserved','internal_intervals_accounted','assumed_intervals_disclosed')):
        raise ValueError('Raw exposure integrity unresolved; no conditional PDE replay allowed')


def candidate(inputs,output,diffusivity,batch_size=500,stop_after=None):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    payload=json.loads(Path(inputs).read_text(encoding='utf-8-sig'))
    validate_replay_input_integrity(payload)
    import numpy as np
    import pybamm
    from measured_protocol_replay import normalized_model_options
    cell=payload['cell'];p=payload['reference'];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    parameters=pybamm.ParameterValues('OKane2022');parameters.update(p['parameter_overrides'])
    parameters.update({'SEI solvent diffusivity [m2.s-1]':diffusivity,
                       'Initial temperature [K]':p['initial_temperature_K']})
    steps=cell['steps'];measure={m['step_index']:m for m in cell['measurements']}
    if stop_after is not None:steps=steps[:stop_after]
    records=[];state=None;started=time.monotonic();empty_boundaries=[];endpoint_time_mismatches=[]
    class Guard(pybamm.callbacks.Callback):
        def __init__(self):self.failure=None
        def on_experiment_error(self,logs):self.failure=str(logs.get('error'))
        def on_experiment_infeasible_event(self,logs):self.failure=str(logs.get('termination'))
        def on_experiment_infeasible_time(self,logs):self.failure='event-only endpoint reached watchdog'
    for begin in range(0,len(steps),batch_size):
        specs=steps[begin:begin+batch_size];native=[]
        for s in specs:
            make=getattr(pybamm.step,s['operation'])
            kwargs=dict(duration=s['duration_s'],termination=s['termination'],temperature=s['temperature_K'],
                        period=max(1,min(60,s['duration_s'])),skip_ok=False)
            native.append(make(**kwargs) if s['operation']=='rest' else make(s['value'],**kwargs))
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=normalized_model_options(p['options'])),
            parameter_values=parameters.copy(),var_pts=p['mesh_points'],experiment=pybamm.Experiment([tuple(native)]),
            solver=pybamm.IDAKLUSolver(rtol=p['rtol'],atol=p['atol']))
        guard=Guard()
        kwargs={'initial_soc':p['initial_soc']} if state is None else {'starting_solution':state}
        sol=sim.solve(calc_esoh=False,callbacks=guard,**kwargs)
        # Save returned output BEFORE interpreting success/failure. Prior batches
        # survive a later solver exception or process timeout.
        pieces=sol.cycles[-1].steps
        for j,piece in enumerate(pieces):
            index=begin+j
            if isinstance(piece,pybamm.EmptySolution):
                current=specs[j]
                zero_observed=(current.get('recorded_start') is not None and
                               current.get('recorded_start')==current.get('recorded_end') and bool(current['termination']))
                allowed=controller_boundary_empty_allowed(current,is_capacity_measurement=index in measure)
                if not allowed:raise ValueError('Unreviewed initial-boundary skip at '+current['source_ref'])
                empty_boundaries.append(dict(step_index=index,source_ref=current['source_ref'],
                    reason='zero recorded sample span and model endpoint already exceeded' if zero_observed else
                           'MODEL EXPOSURE MISMATCH: native V/I endpoint already exceeded; fixed observed route retained',
                    recorded_start=current.get('recorded_start'),recorded_end=current.get('recorded_end'),
                    operation=current['operation'],setpoint=current.get('value'),
                    model_duration_s=0,observed_empty=zero_observed,
                    physical_history_reproduction_pass=False))
                continue
            names=('Time [s]','Current [A]','Voltage [V]','Discharge capacity [A.h]')
            raw={n:np.asarray(piece[n].entries,dtype=float) for n in names}
            current=specs[j]
            if current.get('recorded_start') is not None:
                observed_duration=(datetime.fromisoformat(current['recorded_end'])-datetime.fromisoformat(current['recorded_start'])).total_seconds()
                model_duration=float(raw[names[0]][-1]-raw[names[0]][0])
                if abs(model_duration-observed_duration)>1:
                    endpoint_time_mismatches.append(dict(step_index=index,source_ref=current['source_ref'],
                        role=current['role'],observed_duration_s=observed_duration,model_duration_s=model_duration))
            if index in measure or j==len(pieces)-1:
                raw['_terminal_state_y']=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1)
                np.savez_compressed(out/f'step_{index:06d}.npz',**raw)
            if not specs[j]['timed_endpoint'] and piece.termination=='final time':
                raise ValueError('Event-only step reached watchdog: '+specs[j]['source_ref'])
            if index in measure:
                expected='event: Voltage < 2.5 [V] [experiment]'
                if piece.termination!=expected:raise ValueError('RPT did not reach independent 2.5V endpoint')
                t,I,Q=raw[names[0]],raw[names[1]],raw[names[3]]
                q=float(Q[-1]-Q[0]);integral=float(np.trapezoid(I,t)/3600)
                if q<=0 or abs(q-integral)>1e-7+1e-4*q:raise ValueError('RPT integral check failed')
                # Observability for the next controlled model comparison. These
                # quantities are NOT automatically equal to measured capacity fade.
                variables={'SEI_loss_Ah':'Loss of capacity to negative SEI [A.h]',
                    'plating_loss_Ah':'Loss of capacity to negative lithium plating [A.h]',
                    'LLI_percent':'Loss of lithium inventory [%]',
                    'LAM_negative_percent':'Loss of active material in negative electrode [%]',
                    'LAM_positive_percent':'Loss of active material in positive electrode [%]',
                    'temperature_K':'X-averaged cell temperature [K]'}
                mechanisms={name:float(np.asarray(piece[var].entries).reshape(-1)[-1]) for name,var in variables.items()}
                if not all(math.isfinite(v) for v in mechanisms.values()):raise ValueError('Nonfinite mechanism diagnostics')
                records.append(dict(rpt_id=measure[index]['rpt_id'],Qref_Ah=q,
                                    target_Qref_Ah=measure[index]['target_Qref_Ah'],termination=str(piece.termination),
                                    mechanism_diagnostics=mechanisms,model_time_start_s=float(t[0]),model_time_end_s=float(t[-1])))
        state=sol.last_state.copy()
        y=np.asarray(state.all_ys[-1],dtype=float).reshape(-1)
        if not np.all(np.isfinite(y)):raise ValueError('Nonfinite model state')
        write(out/'progress.json',dict(status='conditional_candidate_running',cell_id=cell['cell_id'],
            completed_steps=begin+len(pieces),total_steps=len(steps),measurements=records,empty_boundaries=empty_boundaries,
            endpoint_time_mismatch_count=len(endpoint_time_mismatches),
            elapsed_s=time.monotonic()-started,physical_validation_complete=False))
        if guard.failure or len(pieces)!=len(specs):raise ValueError('Incomplete native route: '+str(guard.failure))
        # Persist only a trusted local state from this exact code/mesh, never
        # import arbitrary checkpoints and never map a state across meshes.
        with (out/'last_state.pkl').open('wb') as checkpoint:pickle.dump(state,checkpoint)
    result=dict(status='conditional_candidate_completed',cell_id=cell['cell_id'],
                inputs_sha256=sha(inputs),diffusivity_m2_s=diffusivity,measurements=records,
                elapsed_s=time.monotonic()-started,completed_steps=len(steps),empty_boundaries=empty_boundaries,
                model_predicted_empty_but_recorded_nonempty_count=sum(not e['observed_empty'] for e in empty_boundaries),
                endpoint_time_mismatches=endpoint_time_mismatches,
                physical_validation_complete=False,measured_history_admissible=False)
    if stop_after is None:
        if [m['rpt_id'] for m in records]!=[m['rpt_id'] for m in cell['measurements']]:raise ValueError('Missing or reordered RPT')
        errors=normalized_errors([m['Qref_Ah'] for m in records],[m['target_Qref_Ah'] for m in records])
        result.update(errors_pp=errors,RMSE_pp=float(np.sqrt(np.mean(np.square(errors)))),max_abs_error_pp=max(map(abs,errors)))
    write(out/'result.json',result)
    return result


def run(manifest_path,output):
    manifest_path=Path(manifest_path).resolve();m=json.loads(manifest_path.read_text());base=manifest_path.parent
    # Require the user's pinned criteria before spending any PDE budget.
    criteria_path=pinned(m.get('criteria'),base)
    criteria=validate_criteria(json.loads(criteria_path.read_text(encoding='utf-8-sig')))
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if m['data_kind']!='conditional_observed_program_route':raise ValueError('Conditional scope required')
    c=base/m['calibration']['path']
    if sha(c)!=m['calibration']['sha256']:raise ValueError('C inputs changed')
    import numpy as np
    from scipy.optimize import least_squares
    count=0;cache={};evaluation_history=[]
    def evaluate(path,coefficient,label):
        nonlocal count
        remaining=m['total_wall_s']-(time.monotonic()-started)
        if remaining<30 or count>=m['max_model_calls']:raise RuntimeError('Declared calibration budget exhausted')
        count+=1;dest=out/f'{count:03d}_{label}'
        cmd=[sys.executable,__file__,'--candidate','--inputs',str(path),'--output',str(dest),'--diffusivity',str(coefficient)]
        write(out/'status.json',dict(status='conditional_fit_running',call=count,phase=label,coefficient=coefficient,
            elapsed_s=time.monotonic()-started,physical_validation_complete=False))
        try:r=subprocess.run(cmd,timeout=min(remaining,m['per_candidate_wall_s']),capture_output=True,text=True,encoding='utf-8',errors='replace')
        except subprocess.TimeoutExpired as e:
            write(out/f'{count:03d}_timeout.json',dict(error='candidate timeout',phase=label));raise RuntimeError('Candidate timed out; prior batch artifacts retained') from e
        (out/f'{count:03d}_stdout.log').write_text(r.stdout,encoding='utf-8')
        (out/f'{count:03d}_stderr.log').write_text(r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError('Candidate failed; inspect '+str(dest))
        result=json.loads((dest/'result.json').read_text())
        binding=m['calibration'] if label=='C_fit' else m['heldout']
        bind_targets(result,json.loads(path.read_text(encoding='utf-8-sig')),binding['sha256'])
        score=score_curve(result,criteria)
        evaluation_history.append(dict(call=count,phase=label,coefficient=coefficient,
            RMSE_pp=score['RMSE_pp'],max_abs_error_pp=score['max_abs_error_pp'],
            result_sha256=sha(dest/'result.json')))
        write(out/'evaluation_history.json',evaluation_history)
        return result
    nominal=m['nominal_diffusivity_m2_s'];bounds=m['log10_multiplier_bounds']
    def objective(z):
        key=float(z[0])
        if key not in cache:cache[key]=evaluate(c,nominal*10**key,'C_fit')
        return np.array(cache[key]['errors_pp'])[1:]
    def jacobian(z):
        # scipy's relative diff_step degenerates near log-multiplier zero.
        # Use an absolute perturbation in the dimensionless log coordinate.
        x=float(z[0]);h=.05 if x+.05<=bounds[1] else -.05
        return ((objective([x+h])-objective([x]))/h).reshape(-1,1)
    opt=least_squares(objective,[0.],bounds=([bounds[0]],[bounds[1]]),jac=jacobian,
                      max_nfev=m['optimizer_max_nfev'],ftol=1e-4,xtol=1e-3,gtol=1e-4)
    if not opt.success:raise RuntimeError('Optimizer did not converge; no frozen parameters or D evaluation')
    # Explicitly evaluate the optimizer solution, never freeze its last FD probe.
    objective(opt.x);best=cache[float(opt.x[0])]
    c_score=score_curve(best,criteria)
    selected=nominal*10**float(opt.x[0])
    frozen=dict(status='conditional_candidate_frozen_for_diagnostics',diffusivity_m2_s=selected,
                C_inputs_sha256=sha(c),C_result=best,optimizer_message=str(opt.message),
                optimizer_converged=True,C_curve_metrics_pass=c_score['curve_metrics_pass'],
                parameter_bound_review=bound_review(selected,m),
                criteria_sha256=sha(criteria_path),lifetime_parameter_export_allowed=False,
                assumptions=m['assumptions'],physical_validation_complete=False)
    write(out/'parameters_frozen.json',frozen);freeze_hash=sha(out/'parameters_frozen.json')
    d=base/m['heldout']['path']
    if sha(d)!=m['heldout']['sha256']:raise ValueError('D inputs changed')
    heldout=evaluate(d,frozen['diffusivity_m2_s'],'D_postfreeze')
    if sha(out/'parameters_frozen.json')!=freeze_hash:raise ValueError('Frozen parameters changed during D evaluation')
    report=dict(status='conditional_fit_and_D_evaluation_completed',C=best,D=heldout,
        parameters_frozen_sha256=freeze_hash,D_used_for_refit=False,model_calls=count,
        elapsed_s=time.monotonic()-started,physical_validation_complete=False,
        real_SOH_acceptance='not_evaluated_unknown_exposure_and_branch_assumptions',EFC80='not_evaluated')
    assessment=review_result(report,frozen,m,criteria)
    write(out/'calibration_assessment.json',assessment)
    report.update(status=assessment['status'],calibration_assessment=assessment,
                  lifetime_parameter_export_allowed=False,ground_truth_label_export_allowed=False)
    write(out/'report.json',report)
    if not assessment['conditional_diagnostics_accepted']:
        # Preserve all diagnostics, but don't emit a misleading successful job exit.
        raise CalibrationNotAccepted('Optimizer finished; curve criteria failed or bound review required. See calibration_assessment.json')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--candidate',action='store_true');a.add_argument('--inputs',type=Path)
    a.add_argument('--manifest',type=Path);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--diffusivity',type=float);a.add_argument('--stop-after',type=int)
    args=a.parse_args()
    try:
        if args.candidate:candidate(args.inputs,args.output,args.diffusivity,stop_after=args.stop_after)
        else:run(args.manifest,args.output)
    except CalibrationNotAccepted as e:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'acceptance_failure.json',dict(status='calculation_completed_calibration_not_accepted',
              error=str(e),exit_code=2,physical_validation_complete=False))
        print(str(e));raise SystemExit(2)
    except Exception as e:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'failure.json',dict(status='failed',error_type=type(e).__name__,error=str(e),physical_validation_complete=False))
        raise
