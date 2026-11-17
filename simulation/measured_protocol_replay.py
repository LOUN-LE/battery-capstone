"""Bounded chronological replay of an explicitly sourced measured protocol.

Actual RPTs remain in the physical history. Optional diagnostic branches are
copied measurements and never replace that history. No SOC-window controller,
capacity fit, parameter calibration, or AWS launch is provided here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def normalized_model_options(options):
    """JSON arrays encode PyBaMM tuple-valued options without changing meaning."""
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
    return {key: tuples(value) for key, value in options.items()}


def mesh_keys_for_options(options, parameter_set):
    phases = options.get('particle phases', '1')
    if phases == '1':
        phases = ('1', '1')
    elif isinstance(phases, (list, tuple)):
        phases = tuple(phases)
    else:
        raise ValueError('Supported particle phases are one-phase or explicit negative-composite (2,1)')
    if phases not in (('1', '1'), ('2', '1')):
        raise ValueError('Unknown or unsupported particle-phase configuration')
    if parameter_set == 'Chen2020_composite' and phases != ('2', '1'):
        raise ValueError('Chen2020_composite requires explicit negative two-phase configuration')
    if parameter_set == 'OKane2022' and phases != ('1', '1'):
        raise ValueError('OKane2022 is not a two-phase parameter set')
    if phases == ('2', '1'):
        return {'x_n', 'x_s', 'x_p', 'r_n_prim', 'r_n_sec', 'r_p'}
    return {'x_n', 'x_s', 'x_p', 'r_n', 'r_p'}


def runtime_mesh_points(protocol):
    points = protocol['mesh_points'].copy()
    # PyBaMM 26.8 retains a generic (unused) negative-particle geometry even
    # when both active phases have their own radial coordinates. Give that
    # redundant geometry the primary resolution; do not expose a second,
    # conflicting user-controlled r_n for the composite protocol.
    if 'r_n_prim' in points:
        points['r_n'] = points['r_n_prim']
    return points


def validate_protocol(p):
    allowed = {'cell_model', 'protocol_id', 'source_ref', 'initial_state_source',
               'family', 'parameter_set', 'options', 'mesh_points', 'initial_soc',
               'initial_temperature_K', 'rtol', 'atol', 'max_wall_s', 'steps',
               'parameter_overrides'}
    if not isinstance(p, dict) or set(p)-allowed:
        raise ValueError('Unknown protocol fields or unsupported model profile')
    for key in ('cell_model', 'protocol_id', 'source_ref', 'initial_state_source'):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ValueError(f'Explicit {key} required')
    if p.get('family') not in ('SPMe', 'DFN'):
        raise ValueError('Explicit SPMe or DFN required')
    if not isinstance(p.get('parameter_set'), str) or not p['parameter_set']:
        raise ValueError('Explicit parameter set required')
    if not isinstance(p.get('options'), dict) or p['options'].get('thermal') not in ('isothermal', 'lumped'):
        raise ValueError('Explicit model options and thermal policy required')
    overrides = p.get('parameter_overrides', {})
    if not isinstance(overrides, dict) or any(
            not isinstance(key, str) or not key.strip() or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            for key, value in overrides.items()):
        raise ValueError('Parameter overrides must be a finite numeric mapping')
    required_mesh_keys = mesh_keys_for_options(p['options'], p['parameter_set'])
    if not isinstance(p.get('mesh_points'), dict) or set(p['mesh_points']) != required_mesh_keys:
        raise ValueError('Explicit phase-compatible mesh required: '+','.join(sorted(required_mesh_keys)))
    for value in p['mesh_points'].values():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Mesh counts must be positive integers')
    soc = p.get('initial_soc')
    if isinstance(soc, bool) or not isinstance(soc, (int, float)) or not math.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError('Explicit initial SOC in [0,1] required')
    positive(p.get('initial_temperature_K'), 'initial temperature')
    for key in ('rtol', 'atol'):
        positive(p.get(key), key)
    if not 1 <= positive(p.get('max_wall_s'), 'max wall time') <= 3600:
        raise ValueError('Replay must be bounded to at most one hour')
    if not isinstance(p.get('steps'), list) or not p['steps']:
        raise ValueError('Explicit ordered experimental steps required')
    if not any(s.get('role') == 'rpt' for s in p['steps']):
        raise ValueError('At least one actual RPT step is required')
    for s in p['steps']:
        if s.get('role') not in ('ageing', 'rpt', 'calendar', 'diagnostic'):
            raise ValueError('Each step needs an explicit exposure role')
        if not isinstance(s.get('source_ref'), str) or not s['source_ref'].strip():
            raise ValueError('Each step needs a source reference')
        positive(s.get('temperature_K'), 'step ambient temperature')
        positive(s.get('duration_s'), 'step duration')
        positive(s.get('period_s'), 'output period')
        if s['period_s'] > s['duration_s']:
            raise ValueError('Output period exceeds duration')
        if s.get('operation') not in ('current', 'voltage', 'rest'):
            raise ValueError('Unsupported measured operation')
        if not isinstance(s.get('termination'), list) or not all(isinstance(v, str) and v.strip() for v in s['termination']):
            raise ValueError('Explicit termination list required, including [] for timed steps')
        if not isinstance(s.get('expected_termination'), str) or not s['expected_termination'].strip():
            raise ValueError('Explicit expected termination text required')
        if not s['termination'] and s['expected_termination'] != 'final time':
            raise ValueError('A timed exposure must expect final time')
        if s['operation'] != 'rest':
            value = s.get('value')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Explicit finite current (A) or voltage (V) required')
            if s['operation'] == 'voltage' and value <= 0:
                raise ValueError('Voltage must be positive')
    return p


def replay_sequence(protocol, initial_state, advance, clone=copy.deepcopy):
    """advance(state, step) -> (terminal_state, observation). No hidden reset."""
    validate_protocol(protocol)
    main = initial_state
    records = []
    for step in protocol['steps']:
        branch = step['role'] == 'diagnostic'
        terminal, observation = advance(clone(main) if branch else main, step)
        if terminal is None:
            raise ValueError('Missing terminal physical state')
        if not branch:
            main = terminal
        records.append({'role': step['role'], 'propagated': not branch, **observation})
    return main, records


def worker(p, out):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'
    import numpy as np
    import pybamm
    parameters = pybamm.ParameterValues(p['parameter_set'])
    parameters.update(p.get('parameter_overrides', {}))
    parameters.update({'Initial temperature [K]': p['initial_temperature_K'],
                       'Ambient temperature [K]': p['steps'][0]['temperature_K']})
    family = getattr(pybamm.lithium_ion, p['family'])
    options = normalized_model_options(p['options'])
    mesh_points = runtime_mesh_points(p)
    counter = [0]

    def advance(state, spec):
        counter[0] += 1
        kwargs = {'duration': spec['duration_s'], 'period': spec['period_s'],
                  'temperature': spec['temperature_K'], 'termination': spec['termination'],
                  'skip_ok': False}
        make = getattr(pybamm.step, spec['operation'])
        step = make(**kwargs) if spec['operation'] == 'rest' else make(spec['value'], **kwargs)
        sim = pybamm.Simulation(family(options=options), parameter_values=parameters.copy(),
            experiment=pybamm.Experiment([step]), var_pts=mesh_points,
            solver=pybamm.IDAKLUSolver(rtol=p['rtol'], atol=p['atol']))
        args = {'initial_soc': p['initial_soc']} if state is None else {'starting_solution': state}
        result = sim.solve(calc_esoh=False, **args)
        piece = result.cycles[-1].steps[-1]
        raw = {name: np.asarray(piece[name].entries, dtype=float) for name in
               ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]')}
        raw['_input_state_y'] = np.asarray(state.all_ys[-1], dtype=float).reshape(-1) if state is not None else np.array([])
        raw['_terminal_state_y'] = np.asarray(result.last_state.all_ys[-1], dtype=float).reshape(-1)
        # Keep offending output too; a failed numerical/event check must remain
        # diagnosable instead of leaving only a formatted exception message.
        np.savez_compressed(out/f'step_{counter[0]:04d}.npz', **raw)
        if not all(np.all(np.isfinite(a)) for a in raw.values()):
            raise ValueError('Nonfinite measured replay output')
        times, current = raw['Time [s]'], raw['Current [A]']
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError('Replay step has no positive time interval')
        capacity = raw['Discharge capacity [A.h]']
        integral = float(np.trapezoid(current, times)/3600)
        signed = float(capacity[-1]-capacity[0])
        if abs(integral-signed) > 1e-7+1e-4*abs(signed):
            raise ValueError('Replay current/capacity integral mismatch')
        termination = str(piece.termination)
        if spec['expected_termination'] != termination:
            raise ValueError('Expected experimental endpoint was not reached: '+termination)
        if not spec['termination'] and abs(float(times[-1]-times[0])-spec['duration_s']) > 1e-6:
            raise ValueError('Timed exposure ended early')
        return result.last_state.copy(), {'t_start_s': float(times[0]), 't_end_s': float(times[-1]),
            'signed_capacity_Ah': signed,
            'discharge_Ah': float(np.trapezoid(np.maximum(current, 0), times)/3600),
            'charge_Ah': float(np.trapezoid(np.maximum(-current, 0), times)/3600),
            'termination': termination}

    _, records = replay_sequence(p, None, advance, clone=lambda s: None if s is None else s.copy())
    # Every exposure remains separately identifiable. Actual RPT throughput must
    # never silently disappear into ageing EFC or be discarded from physical time.
    totals = {role: {field: sum(r[field] for r in records if r['role'] == role)
                    for field in ('discharge_Ah', 'charge_Ah')}
              for role in ('ageing', 'rpt', 'calendar', 'diagnostic')}
    return {'status': 'replay_completed', 'records': records, 'throughput_by_role': totals,
            'physical_validation_complete': False, 'calibrated': False, 'AWS_started': False,
            'protocol_sha256': hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pybamm_version': pybamm.__version__, 'effective_mesh_points': mesh_points,
            'derived_mesh_aliases': {'r_n': 'r_n_prim'} if 'r_n_prim' in mesh_points else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    p = validate_protocol(json.loads(args.protocol.read_text(encoding='utf-8-sig')))
    if args.worker:
        try:
            result = worker(p, args.output)
        except Exception as error:
            result = {'status': 'replay_failed', 'error': f'{type(error).__name__}: {error}',
                      'physical_validation_complete': False, 'AWS_started': False}
        (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        return 0 if result['status'] == 'replay_completed' else 1
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output/'protocol.json'
    frozen.write_text(json.dumps(p, indent=2, allow_nan=False), encoding='utf-8')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker',
               '--protocol', str(frozen.resolve()), '--output', str(args.output.resolve())]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (args.output/'worker.log').open('w', encoding='utf-8') as log:
        try:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=p['max_wall_s'], creationflags=flags)
            return run.returncode
        except subprocess.TimeoutExpired:
            (args.output/'report.json').write_text(json.dumps({'status': 'wall_limit',
                'max_wall_s': p['max_wall_s'], 'physical_validation_complete': False,
                'AWS_started': False}), encoding='utf-8')
            return 2


if __name__ == '__main__':
    raise SystemExit(main())
