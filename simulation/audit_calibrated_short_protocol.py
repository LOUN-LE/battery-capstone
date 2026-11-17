"""Frozen calibrated-candidate numerical check on a fixed-Ah Expt2 short protocol.

This is not the full measured chronology or physical SOH validation. Virtual RPT
copies intentionally do not age the main state in this numerical comparison.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def short_steps():
    """Fixed730mAh quantity control, never re-normalized using aged capacity."""
    steps = [dict(phase='preparation_CC', kind='current', value=-1.5, duration=21600., termination=['4.2 V'], expected='event: Voltage > 4.2 [V] [experiment]'),
             dict(phase='preparation_CV', kind='voltage', value=4.2, duration=21600., termination=['0.05 A'], expected='event: abs(Current [A]) < 0.05 [A] [experiment]'),
             dict(phase='preparation_rest1', kind='rest', duration=3600., termination=[], expected='final time'),
             dict(phase='preparation_730mAh_discharge', kind='current', value=5., duration=525.6, termination=['2.5 V'], expected='final time'),
             dict(phase='preparation_rest3', kind='rest', duration=10800., termination=[], expected='final time')]
    for index in range(6):
        steps.extend([dict(phase=f'loop{index+1}_discharge', kind='current', value=5., duration=525.6, termination=['2.5 V'], expected='final time'),
                      dict(phase=f'loop{index+1}_charge', kind='current', value=-1.5, duration=1752., termination=['4.2 V'], expected='final time')])
    return steps


def parameters_ready(folder):
    report_path, params_path = folder/'report.json', folder/'parameters_frozen.json'
    if not report_path.exists() or not params_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    params = json.loads(params_path.read_text(encoding='utf-8-sig'))
    return (report.get('status') == 'bounded_BOL_calibration_and_screen_completed'
            and report.get('optimizer_converged_for_all_starts') is True
            and report.get('verification', {}).get('fine_candidate', {}).get('status') == 'evaluated'
            and params.get('C_only_fit') is True)


def guard(config):
    for path, expected in config['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Source changed during numerical test: '+path)


def worker(config, case, out):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'
    import numpy as np
    import pybamm
    import rebuilt_lifetime as N
    guard(config)
    params = json.loads(Path(config['parameters_path']).read_text(encoding='utf-8-sig'))
    overrides = params['parameter_updates']
    allowed = {'Negative electrode active material volume fraction', 'Positive electrode active material volume fraction',
               'Initial concentration in negative electrode [mol.m-3]'}
    if set(overrides) != allowed:
        raise ValueError('Only frozen BOL parameter updates are allowed; no ageing coefficients')
    initial = N.Engine(mesh=case['mesh'], rtol=1e-8, atol=1e-10, overrides=overrides)
    age = N.Engine(mesh=case['mesh'], rtol=case['rtol'], atol=case['rtol']/100, overrides=overrides)
    rpt = N.Engine(mesh=case['mesh'], rtol=1e-8, atol=1e-10, overrides=overrides)
    main = initial.start()
    init_fingerprint = N.fingerprint(main)
    np.savez_compressed(out/'initial_state.npz', y=np.asarray(main.all_ys[-1]), t=np.asarray(main.all_ts[-1]))
    q0, raw0 = rpt.reference(main, rest_s=config['virtual_RPT_rest_s'])
    np.savez_compressed(out/'Q0_virtual_RPT.npz', **raw0)
    if N.fingerprint(main) != init_fingerprint:
        raise ValueError('Q0 virtual RPT changed main state')
    definitions = short_steps()
    experiment_steps = []
    for spec in definitions:
        kwargs = dict(duration=spec['duration'], termination=spec['termination'], period=1., skip_ok=False)
        factory = getattr(pybamm.step, spec['kind'])
        experiment_steps.append(factory(**kwargs) if spec['kind'] == 'rest' else factory(spec['value'], **kwargs))
    model = age.model()
    points = {key: int(value*case['mesh']) for key, value in model.default_var_pts.items()}
    simulation = pybamm.Simulation(model, parameter_values=age.param.copy(), var_pts=points,
        experiment=pybamm.Experiment(experiment_steps, temperature=298.15),
        solver=pybamm.IDAKLUSolver(rtol=case['rtol'], atol=case['rtol']/100))
    began = time.monotonic()
    solution = simulation.solve(starting_solution=main, calc_esoh=False)
    # The supplied starting solution is carried as an earlier cycle. Restrict to
    # the17 newly prescribed one-step cycles, never count its initial rest again.
    new_cycles = solution.cycles[-len(definitions):]
    pieces = [cycle.steps[-1] for cycle in new_cycles]
    if len(pieces) != len(definitions):
        raise ValueError('Incomplete main ageing protocol')
    charge_Ah = discharge_Ah = 0.
    audits = []
    failures = []
    for index, (spec, piece) in enumerate(zip(definitions, pieces)):
        raw = {name: np.asarray(piece[name].entries, dtype=float) for name in N.RAW_VARIABLES}
        np.savez_compressed(out/f'main_step_{index+1:02d}.npz', **raw)
        t, current = raw['Time [s]'], raw['Current [A]']
        q = raw['Discharge capacity [A.h]']
        dq, integral = float(q[-1]-q[0]), float(np.trapezoid(current,t)/3600)
        positive = float(np.trapezoid(np.maximum(current,0),t)/3600)
        negative = float(np.trapezoid(np.maximum(-current,0),t)/3600)
        discharge_Ah += positive; charge_Ah += negative
        record = {'phase': spec['phase'], 'termination': str(piece.termination),
                  'duration_s': float(t[-1]-t[0]), 'capacity_delta_Ah': dq,
                  'current_integral_Ah': integral, 'discharge_Ah': positive, 'charge_Ah': negative}
        audits.append(record)
        if not all(np.all(np.isfinite(value)) for value in raw.values()) or not np.all(np.diff(t)>0):
            failures.append(spec['phase']+': nonfinite or nonincreasing output')
        if str(piece.termination) != spec['expected']:
            failures.append(spec['phase']+': wrong endpoint '+str(piece.termination))
        if abs(dq-integral) > 1e-7+1e-4*abs(dq):
            failures.append(spec['phase']+': current/capacity integral disagreement')
        if spec['expected'] == 'final time' and abs(record['duration_s']-spec['duration']) > 1e-6:
            failures.append(spec['phase']+': timed step cut short')
        if '730mAh' in spec['phase'] or spec['phase'].startswith('loop'):
            if abs(abs(dq)-.73) > 1e-6:
                failures.append(spec['phase']+': fixed730mAh control mismatch')
    write(out/'main_protocol_audit.json', {'steps': audits, 'failures': failures})
    if failures:
        raise ValueError('; '.join(failures))
    terminal = solution.last_state.copy()
    before = N.fingerprint(terminal)
    np.savez_compressed(out/'terminal_state.npz', y=np.asarray(terminal.all_ys[-1]), t=np.asarray(terminal.all_ts[-1]))
    qref, rawref = rpt.reference(terminal, rest_s=config['virtual_RPT_rest_s'])
    np.savez_compressed(out/'Qref_virtual_RPT.npz', **rawref)
    if N.fingerprint(terminal) != before:
        raise ValueError('Qref virtual RPT changed main state')
    guard(config)
    return {'status': 'completed', **case, 'Q0_Ah': q0['Qref_Ah'], 'Qref_Ah': qref['Qref_Ah'],
            'SOH_percent': 100*qref['Qref_Ah']/q0['Qref_Ah'], 'initial_fingerprint': init_fingerprint,
            'terminal_fingerprint': before, 'virtual_RPTs_main_state_unchanged': True,
            'main_discharge_Ah': discharge_Ah, 'main_charge_Ah': charge_Ah,
            'discharge_EFC': discharge_Ah/q0['Qref_Ah'],
            'absolute_main_throughput_EFC': (charge_Ah+discharge_Ah)/(2*q0['Qref_Ah']),
            'six_loops': True, 'exact_one_EFC_claimed': False,
            'main_steps': audits, 'ageing_and_Qref_wall_s': time.monotonic()-began,
            'Q0_virtual_RPT': q0, 'Qref_virtual_RPT': qref,
            'physical_validation_complete': False, 'AWS_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        config = json.loads((args.output.parent/'declared_numerical_test.json').read_text(encoding='utf-8'))
        began = time.monotonic()
        try:
            result = worker(config, config['cases'][args.worker], args.output)
        except Exception as error:
            result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc(), 'physical_validation_complete': False, 'AWS_started': False}
        result['worker_wall_s'] = time.monotonic()-began
        write(args.output/'report.json', result)
        return 0 if result['status'] == 'completed' else 1
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+600
    while not parameters_ready(args.parameters_folder):
        if time.monotonic() >= deadline:
            write(args.output/'report.json', {'status': 'valid_frozen_parameters_not_ready_within600s',
                'physical_validation_complete': False, 'AWS_started': False})
            return 2
        time.sleep(5)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root/'simulation/rebuilt_lifetime.py', root/'configs/model.json',
               root/'configs/diagnostics.json', args.parameters_folder/'parameters_frozen.json',
               args.parameters_folder/'report.json']
    cases = [{'mesh': mesh, 'rtol': rtol} for mesh in (4,8) for rtol in (1e-6,1e-7)]
    config = {'parameters_path': str((args.parameters_folder/'parameters_frozen.json').resolve()),
              'source_sha256': {str(path.resolve()): digest(path) for path in sources},
              'cases': cases, 'initial_rtol':1e-8, 'initial_atol':1e-10,
              'RPT_rtol':1e-8, 'RPT_atol':1e-10, 'virtual_RPT_rest_s':7230.,
              'protocol_source': 'Kirkaldy2024 Table3 fixed730mAh preparation and6loops, limited numerical test',
              'definitions': short_steps(), 'per_case_limit_s':900, 'total_limit_s':3600,
              'SOH_span_screen_limit_pp':1e-4,
              'interpretation': 'Numerical comparison only; actual RPTs are virtual copies here, not full measured physical chronology'}
    write(args.output/'declared_numerical_test.json', config)
    began = time.monotonic()
    results = []
    for index, case in enumerate(cases):
        folder = args.output/f"mesh{case['mesh']}_rtol{case['rtol']:g}"
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(index),
                   '--parameters-folder', str(args.parameters_folder.resolve()), '--output', str(folder.resolve())]
        remaining = config['total_limit_s']-(time.monotonic()-began)
        if remaining <= 0:
            results.append({'status':'total_wall_limit', **case});break
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(config['per_case_limit_s'],remaining), creationflags=flags)
                result = json.loads((folder/'report.json').read_text(encoding='utf-8'))
                result['exit_code'] = process.returncode
            except subprocess.TimeoutExpired:
                result = {'status':'case_wall_limit', **case}
                write(folder/'report.json', result)
            results.append(result)
    completed = [result for result in results if result['status']=='completed']
    span = max(r['SOH_percent'] for r in completed)-min(r['SOH_percent'] for r in completed) if completed else None
    consistent_initial = all(len({r['initial_fingerprint'] for r in completed if r['mesh']==mesh})==1 for mesh in (4,8))
    all_completed = len(completed)==4
    passed = all_completed and consistent_initial and span <= config['SOH_span_screen_limit_pp']
    report = {'status':'screen_passed' if passed else 'screen_not_passed', 'cases':results,
              'all_four_completed':all_completed, 'same_initial_state_within_each_mesh':consistent_initial,
              'SOH_span_pp':span, 'SOH_span_screen_limit_pp':config['SOH_span_screen_limit_pp'],
              'numerical_short_screen_pass':passed, 'wall_s':time.monotonic()-began,
              'six_loops_not_exact_one_EFC':True, 'physical_validation_complete':False,
              'EOL_validation_complete':False, 'AWS_started':False}
    write(args.output/'report.json',report)
    print(json.dumps({key:value for key,value in report.items() if key!='cases'},indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
