"""Fresh, bounded SPMe/DFN comparison of measured C/2 after identical conditioning.

The preserved native F045->F046 chronology is replayed continuously. Only the
model class changes; no fitting, cross-family state transfer or AWS is used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'

import numpy as np
from run_exposure_scenario_smoke import read, write, sha
from diagnostic_model_family import make_model
from compare_dfn_initial_rpt import check_local_bounds, strictly_increasing_trace, score_observed_window, memory_mb
from interstitial_pilot_support import prepare_parameters, validate_contract, native_control_evidence, validate_charge_residuals


def select_half_rate_prefix(specifications):
    index = next((i for i, spec in enumerate(specifications)
        if spec.get('file') == 'F046' and spec.get('Ns') == 5), None)
    if index is None:
        raise ValueError('Measured F046 C/2 discharge absent')
    result = specifications[:index+1]
    expected = [('F045', n) for n in range(10)] + [('gap', None)] + [('F046', n) for n in range(6)]
    if [(s['file'], s['Ns']) for s in result] != expected:
        raise ValueError('Complete F045 conditioning and F046 preparation are required')
    last = result[-1]
    if (last['operation'] != 'current' or last.get('value') != 2.5 or not last.get('is_capacity')
            or last.get('termination') != '2.5 V' or last.get('conditional_pulse', False)):
        raise ValueError('Independent 2.5A/2.5V C/2 capacity contract required')
    return result


def select_gitt_route(prepared):
    specs = prepared['specifications']
    expected = ([('F045', n) for n in range(10)] + [('gap', None)]
        + [('F046', n) for n in range(10)] + [('gap', None)]
        + [('F047', n) for n in [0,1,2,3]+[4,5,6]*25+[8]])
    if [(s['file'],s['Ns']) for s in specs] != expected or len(specs)!=102:
        raise ValueError('Complete audited102-phase GITT route required')
    if prepared.get('pulse_endpoint_policy')!='paper-200mAh' or prepared.get('native_Q_reference_admitted') is not False:
        raise ValueError('Explicit paper200mAh conditional policy and unresolved native MB endpoint required')
    for spec in (s for s in specs if s['file']=='F047' and s['Ns']==5):
        if (not spec.get('conditional_pulse') or spec.get('paper_charge_target_Ah')!=.2
                or spec.get('duration')!=288. or spec.get('value')!=2.5 or spec.get('termination')!='2.5 V'
                or spec.get('pulse_duration_from_observation') is not False):
            raise ValueError('GITT pulse must retain paper200mAh/288s policy with2.5V earlier cutoff')
    select_half_rate_prefix(specs)
    return specs


def paired_axis_score(model_x, model_v, observed_x, observed_v):
    x, v, ox, ov = (np.asarray(a, dtype=float) for a in (model_x, model_v, observed_x, observed_v))
    if (len(x) < 2 or len(ox) < 2 or x.shape != v.shape or ox.shape != ov.shape
            or not all(np.isfinite(a).all() for a in (x, v, ox, ov))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(ox) <= 0)):
        raise ValueError('Finite strictly increasing independent score axes required')
    mask = (ox >= x[0]) & (ox <= x[-1])
    errors = np.interp(ox[mask], x, v)-ov[mask]
    return dict(observed_rows=len(ox), scored_rows=int(mask.sum()), uncovered_rows=int((~mask).sum()),
        voltage_RMSE_mV=float(np.sqrt(np.mean(errors**2))*1000) if len(errors) else None,
        voltage_max_abs_error_mV=float(np.max(np.abs(errors))*1000) if len(errors) else None,
        model_endpoint=float(x[-1]), observed_endpoint=float(ox[-1]),
        axes_stretched=False, extrapolated=False)


def prepare(args):
    contract = validate_contract(args.contract, args.dint_multiplier)
    prepared = read(args.prepared/'protocol_contract.json')
    specs = select_gitt_route(prepared) if args.include_gitt else select_half_rate_prefix(prepared['specifications'])
    if prepared.get('native_Q_reference_admitted') is not False:
        raise ValueError('Keep unresolved native charge-reference status explicit')
    guards = dict(prepared['source_sha256'])
    for path, expected in guards.items():
        if sha(path) != expected:
            raise ValueError('Preserved native source changed: '+path)
    data = {}
    for file_id in (('F045', 'F046', 'F047') if args.include_gitt else ('F045', 'F046')):
        path = args.prepared/(file_id+'_native.npz')
        if sha(path) != prepared['prepared_native_sha256'][file_id]:
            raise ValueError('Prepared native arrays changed: '+file_id)
        with np.load(path, allow_pickle=False) as saved:
            data[file_id] = {key: saved[key] for key in ('time_s', 'V', 'I_mA', 'Ns', 'mode')}
        guards[str(path)] = sha(path)
    p, options, metadata, identity = prepare_parameters(args.candidate, args.anchor, contract, args.dint_multiplier)
    guards.update(identity['source_sha256'])
    for path in (Path(__file__), args.prepared/'protocol_contract.json', args.contract, args.candidate, args.anchor,
                 Path(__file__).with_name('diagnostic_model_family.py'), Path(__file__).with_name('compare_dfn_initial_rpt.py')):
        guards[str(path.resolve())] = sha(path)
    mesh = dict(x_n=args.x_points, x_s=args.x_points, x_p=args.x_points,
        r_n=args.r_points, r_n_prim=args.r_points, r_n_sec=args.r_points, r_p=args.r_points)
    return contract, prepared, specs, data, p, options, metadata, identity, mesh, guards


def native_spec(spec):
    return dict(operation=spec['operation'], value=spec.get('value'), duration_s=spec['duration'],
        termination=[spec['termination']] if 'termination' in spec else [],
        timed_endpoint='termination' not in spec or bool(spec.get('conditional_pulse')))


def verify_dynamic_control(spec, current, voltage, duration, termination, contract, evidence):
    """Authenticate exact native float targets before adapting label formatting.

    PyBaMM preserves raw float digits in event names; the older shared checker
    formats the same number with :g. Only the label representation is adapted.
    The physical setpoint, endpoint and residual guards retain their exact value.
    """
    original = termination
    match = re.fullmatch(r'event: Voltage ([<>]) ([-+0-9.eE]+) \[V\] \[experiment\]', termination)
    if match:
        target, unit = spec.get('termination','').split()
        expected_sign = '>' if spec.get('value',0)<0 else '<'
        if unit!='V' or match[1]!=expected_sign or float(match[2])!=float(target):
            raise ValueError('Native voltage event label differs from exact decoded target or direction')
        termination = f'event: Voltage {expected_sign} {float(target):g} [V] [experiment]'
    else:
        match = re.fullmatch(r'event: abs\(Current \[A\]\) < ([-+0-9.eE]+) \[A\] \[experiment\]', termination)
        if match:
            target, unit = spec.get('termination','').split()
            if unit!='A' or float(match[1])!=float(target):
                raise ValueError('Native current event label differs from exact decoded target')
            termination = f'event: abs(Current [A]) < {float(target):g} [A] [experiment]'
    native_control_evidence(native_spec(spec), current, voltage, duration, termination, contract, evidence=evidence)
    evidence['native_event_label_original'] = original
    return evidence


def worker(args):
    import pybamm
    out = args.output/args.worker_family
    out.mkdir()
    began = time.monotonic()
    phases, guards = [], {}
    report = dict(status='started', model_family=args.worker_family,
        condition='C/2 and full conditional GITT102phases' if args.include_gitt else 'measured C/2 F046 Ns5',
        fresh_initial_state=True, source_checkpoint_loaded=False, initial_SOC_remap=False,
        parameter_fitting_performed=False, physical_validation_complete=False, lifetime_restart_admitted=False,
        AWS_started=False, observed_data_status='Previously viewed development condition; not new blind validation')
    write(out/'report.json', report)
    try:
        c, prepared, specs, data, p, options, meta, identity, mesh, guards = prepare(args)
        report.update(mesh=mesh, options=options, rtol=args.rtol, atol=args.atol, period_s=args.period_s,
            Dint_multiplier=args.dint_multiplier, candidate_metadata=meta, runtime_identity_sha256=identity['sha256'],
            source_sha256=guards, planned_steps=len(specs), completed_steps=0,
            interfile_assumption=[s for s in specs if s['file']=='gap'],
            scope='Continuous F045 conditioning, F046 independent C/2, optional full F047 paper-conditional GITT;25C isothermal',
            GITT_pulse_endpoint_policy=prepared['pulse_endpoint_policy'] if args.include_gitt else None,
            GITT_native_MB_charge_endpoint_decoded=False,
            GITT_pulse_assumption='paper0.2Ah at2.5A=288s with independent2.5V earlier stop; not empirical MB endpoint decoding' if args.include_gitt else None)
        write(out/'declaration.json', dict(report=report, specifications=specs))
        write(out/'runtime_identity.json', identity)
        steps = []
        for spec in specs:
            kw = dict(duration=spec['duration'], period=min(args.period_s, spec['duration']),
                skip_ok=bool(spec.get('allow_initial_event') or spec.get('conditional_pulse')) and not spec.get('is_capacity', False))
            if 'termination' in spec: kw['termination'] = spec['termination']
            make = getattr(pybamm.step, spec['operation'])
            steps.append(make(**kw) if spec['operation']=='rest' else make(spec['value'], **kw))
        simulation = pybamm.Simulation(make_model(args.worker_family, options), parameter_values=p.copy(), var_pts=mesh,
            experiment=pybamm.Experiment([tuple(steps)], temperature=298.15),
            solver=pybamm.IDAKLUSolver(rtol=args.rtol, atol=args.atol))
        before_build = time.monotonic(); simulation.build_for_experiment(); built = time.monotonic()
        solution = simulation.solve(calc_esoh=False); solved = time.monotonic()
        pieces = solution.cycles[-1].steps
        report.update(build_s=built-before_build, solve_s=solved-built, memory_after_solve=memory_mb())
        if len(pieces) != len(specs): raise ValueError('Incomplete conditioning route')
        baseline_li, previous = None, None
        trace, observed_parts = [], []
        half_rate = None
        for idx, (spec, piece) in enumerate(zip(specs, pieces)):
            phase = dict(index=idx, spec=spec)
            phases.append(phase)
            if isinstance(piece, pybamm.EmptySolution):
                if not (spec.get('allow_initial_event') or spec.get('conditional_pulse')) or spec.get('is_capacity'):
                    raise ValueError('Unapproved skipped exposure')
                phase.update(status='initial_event_skip', actual_exposure_mismatch=True)
                if spec.get('conditional_pulse'):
                    phase.update(paper_charge_target_met=False, paper_charge_deficit_Ah=spec['paper_charge_target_Ah'],
                        observed_exposure_not_fully_covered=True)
                if spec['file']!='gap':
                    raw = data[spec['file']]; a,b=spec['observed_row_start'],spec['observed_row_end_exclusive']
                    observed=dict(t_s=raw['time_s'][a:b]-data['F045']['time_s'][0],V=raw['V'][a:b],
                        I_A=-raw['I_mA'][a:b]/1000,valid=np.isfinite(raw['V'][a:b])&np.isfinite(raw['I_mA'][a:b]))
                    observed_parts.append((phase,observed))
                    np.savez_compressed(out/f'skipped_observed_{idx:03d}.npz',**observed)
                continue
            names = ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]',
                'Throughput capacity [A.h]', 'Total lithium in particles [mol]', 'Loss of capacity to negative SEI [A.h]')
            t, current, voltage, charge, throughput, lithium, sei = [np.asarray(piece[n].entries, dtype=float).reshape(-1) for n in names]
            if not all(np.isfinite(a).all() for a in (t, current, voltage, charge, throughput, lithium, sei)):
                raise ValueError('Nonfinite primary output')
            duration = float(t[-1]-t[0]); signed = float(charge[-1]-charge[0]); absolute = float(throughput[-1]-throughput[0])
            phase.update(status='simulated', model_start_s=float(t[0]), model_end_s=float(t[-1]),
                model_duration_s=duration, termination=str(piece.termination), signed_Ah=signed, absolute_Ah=absolute)
            artifact = dict(t_s=t, I_A=current, V=voltage, Q_Ah=charge, throughput_Ah=throughput, Li_mol=lithium, SEI_Ah=sei)
            local = {}
            for key, variable in (('electrolyte','Electrolyte concentration [mol.m-3]'),
                    ('negative_primary','Negative primary particle concentration [mol.m-3]'),
                    ('negative_secondary','Negative secondary particle concentration [mol.m-3]'),
                    ('positive','Positive particle concentration [mol.m-3]'),
                    ('negative_porosity','Negative electrode porosity'), ('positive_porosity','Positive electrode porosity')):
                field = np.asarray(piece[variable].entries, dtype=float)
                if not np.isfinite(field).all(): raise ValueError('Nonfinite local field: '+key)
                local[key] = dict(min=float(field.min()), max=float(field.max()))
                spatial = field.reshape(-1, len(t)); artifact[key+'_min'] = spatial.min(axis=0); artifact[key+'_max'] = spatial.max(axis=0)
            phase['local_field_extrema'] = local
            phase['local_field_bounds_failures'] = check_local_bounds(local, p)
            conserved = lithium*96485.33212331001/3600+sei
            if baseline_li is None: baseline_li = float(conserved[0])
            phase['Li_plus_SEI_error_Ah'] = float(np.max(np.abs(conserved-baseline_li)))
            if spec['file'] != 'gap':
                raw = data[spec['file']]; a, b = spec['observed_row_start'], spec['observed_row_end_exclusive']
                ot = raw['time_s'][a:b]-data['F045']['time_s'][0]
                artifact.update(observed_global_t_s=ot, observed_V=raw['V'][a:b], observed_I_A=-raw['I_mA'][a:b]/1000)
                phase.update(model_minus_raw_start_s=float(t[0]-ot[0]), model_minus_raw_end_s=float(t[-1]-ot[-1]))
                observed = dict(t_s=ot, V=raw['V'][a:b], I_A=-raw['I_mA'][a:b]/1000,
                    valid=np.isfinite(raw['V'][a:b])&np.isfinite(raw['I_mA'][a:b]))
                observed_parts.append((phase, observed))
                if len(ot)>=2 and len(t)>=2:
                    phase['phase_elapsed_secondary'] = paired_axis_score(t-t[0], voltage, ot-ot[0], observed['V'])
                    phase['phase_voltage_shape_secondary'] = paired_axis_score(t-t[0], voltage-voltage[0],
                        ot-ot[0], observed['V']-observed['V'][0])
            np.savez_compressed(out/f'phase_{idx:03d}.npz', **artifact)
            if phase['local_field_bounds_failures']: raise ValueError('; '.join(phase['local_field_bounds_failures']))
            if previous is not None:
                phase.update(boundary_time_jump_s=abs(float(t[0])-previous[0]), boundary_charge_jump_Ah=abs(float(charge[0])-previous[1]))
                if phase['boundary_time_jump_s']>1e-5 or phase['boundary_charge_jump_Ah']>1e-8:
                    raise ValueError('Main time or charge discontinuity')
            previous = (float(t[-1]), float(charge[-1]))
            verify_dynamic_control(spec, current, voltage, duration, str(piece.termination), c, phase)
            if spec.get('conditional_pulse'):
                target = spec['paper_charge_target_Ah']
                if signed>target+1e-7 or (str(piece.termination)=='final time' and abs(signed-target)>1e-7):
                    raise ValueError('Paper-pulse current/time charge identity failed')
                phase.update(paper_charge_target_Ah=target, paper_charge_target_met=abs(signed-target)<=1e-7,
                    paper_charge_deficit_Ah=max(0.,target-signed),
                    voltage_early_cutoff_before_paper_charge=signed<target-1e-7,
                    observed_exposure_not_fully_covered=duration+1e-6<spec['observed_duration_s'])
            if spec['operation'] in ('current','rest'):
                expected = spec.get('value',0.) if spec['operation']=='current' else 0.
                phase['applied_current_max_error_A'] = float(np.max(np.abs(current-expected)))
                if phase['applied_current_max_error_A']>c['solver_diagnostic_guards']['applied_current_max_error_A']:
                    raise ValueError('Applied current differs from native CC/rest')
                sr, ar = signed-expected*duration/3600, absolute-abs(expected)*duration/3600
                phase.update(signed_charge_residual_Ah=sr, absolute_charge_residual_Ah=ar)
                validate_charge_residuals(sr, ar, c)
            if phase['Li_plus_SEI_error_Ah']>c['solver_diagnostic_guards']['Li_plus_SEI_conservation_max_error_Ah']:
                raise ValueError('Li+SEI conservation failed')
            trace.append((t, voltage, current)); report['completed_steps'] = idx+1
            if spec['file']=='F046' and spec['Ns']==5:
                if str(piece.termination)!='event: Voltage < 2.5 [V] [experiment]':
                    raise ValueError('Independent C/2 cutoff absent')
                half_rate = dict(spec=spec,t=t.copy(),current=current.copy(),voltage=voltage.copy(),charge=charge.copy())
            write(out/'phase_journal.json', phases)
        if half_rate is None: raise ValueError('Actual independent C/2 measurement missing')
        last = half_rate['spec']; raw = data['F046']; a, b = last['observed_row_start'], last['observed_row_end_exclusive']
        ot = raw['time_s'][a:b]-raw['time_s'][a]; ov = raw['V'][a:b]; oi = -raw['I_mA'][a:b]/1000
        oq = np.r_[0., np.cumsum((oi[1:]+oi[:-1])*.5*np.diff(ot)/3600)]
        t, voltage, charge = half_rate['t'],half_rate['voltage'],half_rate['charge']
        mt = t-t[0]; mq = charge-charge[0]
        global_t, global_v, global_i = strictly_increasing_trace(*(np.concatenate([part[k] for part in trace]) for k in range(3)))
        np.savez_compressed(out/'model_trace.npz',t_s=global_t,V=global_v,I_A=global_i)
        for phase, observed_phase in observed_parts:
            phase['absolute_global_clock_score'] = score_observed_window(global_t,global_v,global_i,observed_phase)
        observed = dict(t_s=raw['time_s'][a:b]-data['F045']['time_s'][0], V=ov, I_A=oi,
            valid=np.isfinite(ov)&np.isfinite(oi))
        report.update(status='diagnostic_complete', independent_capacity_event_verified=True,
            C_half_rate_absolute_global_clock=score_observed_window(global_t, global_v, global_i, observed),
            C_half_rate_phase_elapsed_secondary=paired_axis_score(mt, voltage, ot, ov),
            C_half_rate_absolute_discharged_Ah=paired_axis_score(mq, voltage, oq, ov),
            model_C_half_capacity_Ah=float(mq[-1]), observed_C_half_capacity_Ah=float(oq[-1]),
            C_half_capacity_bias_percent=float(100*(mq[-1]/oq[-1]-1)),
            C_half_model_duration_s=float(mt[-1]), C_half_observed_duration_s=float(ot[-1]),
            initial_event_skip_count=sum(phase['status']=='initial_event_skip' for phase in phases),
            GITT_pulses=[phase for phase in phases if phase['spec']['file']=='F047' and phase['spec']['Ns']==5],
            GITT_relaxations=[phase for phase in phases if phase['spec']['file']=='F047' and phase['spec']['Ns']==6],
            all_phases_simulated=all(phase['status']=='simulated' for phase in phases),
            scoring_scope='Primary original global elapsed clock; secondary phase elapsed resets both clocks only, without stretching; Ah starts at actual discharge onset')
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        (out/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    report.update(wall_s=time.monotonic()-began, memory_final=memory_mb(), source_sha256=guards,
        source_hashes_unchanged=bool(guards) and all(sha(path)==value for path,value in guards.items()))
    if guards and not report['source_hashes_unchanged']: report['status']='failed_source_integrity'
    write(out/'phase_journal.json', phases); write(out/'report.json', report)
    return 0 if report['status']=='diagnostic_complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','candidate','anchor','contract','prepared','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--families', nargs='+', choices=('SPMe','DFN'), default=['SPMe','DFN'])
    parser.add_argument('--worker-family', choices=('SPMe','DFN'), help=argparse.SUPPRESS)
    parser.add_argument('--x-points', type=int, default=10); parser.add_argument('--r-points', type=int, default=20)
    parser.add_argument('--rtol', type=float, default=1e-8); parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--period-s', type=float, default=10.); parser.add_argument('--dint-multiplier', type=float, default=1000.)
    parser.add_argument('--family-wall-s', type=float, default=180.); parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--include-gitt',action='store_true',help='Replay full audited102phase route with explicit paper200mAh/288s pulse policy')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve(); args.output.relative_to(args.root)
    if not 30<=args.family_wall_s<=180: parser.error('Per-family wall budget must be30..180s')
    if not 3<=args.x_points<=100 or not 3<=args.r_points<=100: parser.error('Mesh resolution must be3..100')
    if (not all(np.isfinite(v) and v>0 for v in (args.rtol,args.atol,args.period_s)) or args.period_s>10
            or len(set(args.families))!=len(args.families)): parser.error('Positive finite settings, period<=10s and unique families required')
    if args.worker_family: return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepare_only:
        _,_,specs,_,_,options,meta,identity,mesh,guards = prepare(args)
        write(args.output/'preparation.json', dict(status='prepared_no_PDE', specifications=specs, mesh=mesh,
            options=options, candidate_metadata=meta, runtime_identity_sha256=identity['sha256'], source_sha256=guards))
        return 0
    results = []
    for family in args.families:
        with (args.output/(family+'_worker.log')).open('w',encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker-family',family],
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                code = process.wait(timeout=args.family_wall_s)
                status = 'worker_completed' if code==0 else 'worker_failed'
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); code = process.returncode; status = 'terminated_per_family_wall_budget'
        path = args.output/family/'report.json'
        results.append(dict(family=family,status=status,returncode=code,pid=process.pid,report=read(path) if path.exists() else None))
        write(args.output/'comparison.json', dict(results=results,physical_validation_complete=False,AWS_started=False))
    complete = all(result['status']=='worker_completed' for result in results)
    write(args.output/'comparison.json', dict(status='diagnostic_complete' if complete else 'incomplete_or_failed',
        results=results,per_family_wall_s=args.family_wall_s,physical_validation_complete=False,AWS_started=False))
    return 0 if complete else 1


if __name__=='__main__': raise SystemExit(main())
