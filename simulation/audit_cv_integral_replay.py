"""Bounded same-state replay of the locally observed first-CV integral failure.

Reads only the hashed Windows checkpoint produced by short_1efc_v1. Uses the
archived engine for the identical parameter/model/eSOH definitions, and public
PyBaMM Experiment/IDAKLU APIs for six separate 60-second CV solves. Never
advances the source trajectory or relaxes its integral acceptance criterion.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/'work/numerical_repair_20261103'
INPUT = BASE/'short_1efc_v1/m4_f1_I3.0/trajectory'
FROZEN = BASE/'short_1efc_v1/source_snapshot/rebuilt_lifetime.py'
CASES = ((1e-4, 1e-6, 1.), (1e-4, 1e-6, .5), (1e-4, 1e-6, .25),
         (1e-5, 1e-7, 1.), (1e-6, 1e-8, 1.), (1e-6, 1e-8, .25))
VARS = ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def worker(out):
    import casadi
    import numpy as np
    import pickle
    import pybamm
    report = {'status': 'running', 'scope': 'CV sampling/integration diagnostic only',
              'main_trajectory_advanced': False, 'acceptance_criterion_changed': False,
              'rows': [], 'script_sha256': digest(__file__)}
    started = time.monotonic()
    protected = {}
    try:
        commit_path = INPUT/'commit.json'
        commit = json.loads(commit_path.read_text(encoding='utf-8'))
        identity = commit['identity']
        checkpoint_path = INPUT/commit['checkpoint']
        if checkpoint_path.parent.resolve() != INPUT.resolve():
            raise ValueError('Checkpoint path escapes the trusted run')
        if digest(checkpoint_path) != commit['sha256']:
            raise ValueError('Checkpoint digest mismatch')
        if digest(FROZEN) != identity['code_sha256']:
            raise ValueError('Archived engine does not match checkpoint source')
        if identity['platform'] != platform.system() or identity['python'] != platform.python_version():
            raise ValueError('Checkpoint platform/Python mismatch')
        for name, expected in identity['versions'].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError('Checkpoint package mismatch: '+name)
        for name, expected in identity['config_sha256'].items():
            if digest(ROOT/'configs'/name) != expected:
                raise ValueError('Checkpoint configuration mismatch: '+name)
        for path in (commit_path, checkpoint_path, FROZEN,
                     *(ROOT/'configs'/name for name in identity['config_sha256'])):
            protected[str(path)] = digest(path)
        spec = importlib.util.spec_from_file_location('cv_replay_frozen_engine', FROZEN)
        N = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(N)
        # The archive lives outside repo/simulation; only its configuration root
        # is redirected to the hash-verified, identical configs above.
        N.ROOT = ROOT
        with casadi.global_unpickle_context(), checkpoint_path.open('rb') as handle:
            checkpoint = pickle.load(handle)
        if checkpoint['identity'] != identity:
            raise ValueError('Checkpoint content identity mismatch')
        if not (checkpoint['status'] == 'failed' and checkpoint['phase'] == 'CV'
                and identity['mesh'] == 4 and identity['family'] == 'SPMe'
                and identity['physics'] == 'baseline' and identity['current_A'] == 3.):
            raise ValueError('Unexpected failure checkpoint')
        main = checkpoint['solution'].last_state.copy()
        original_fingerprint = N.fingerprint(main)
        original_q = N.scalar(main, 'Discharge capacity [A.h]')
        original_t = N.scalar(main, 'Time [s]')
        report.update(source_identity=identity, checkpoint_sha256=commit['sha256'],
                      frozen_source_sha256=digest(FROZEN), protected_sources=protected,
                      source_fingerprint=original_fingerprint, Qprior_Ah=original_q,
                      tprior_s=original_t, completed_prior_pieces=checkpoint['pieces'])
        save(out/'report.json', report)
        for index, (rtol, atol, period) in enumerate(CASES):
            engine = N.Engine(mesh=4, family='SPMe', physics='baseline',
                              temperature_K=identity['temperature_K'],
                              overrides=identity['overrides'], rtol=rtol, atol=atol)
            if engine.options != identity['options']:
                raise ValueError('Model option mismatch')
            bounds = engine.bounds(main)
            target = bounds[0]+.8*(bounds[1]-bounds[0])
            termination = pybamm.step.CustomTermination('operating SOC',
                lambda variables: pybamm.InputParameter('target')-variables[N.STO])
            step = pybamm.step.voltage(4.2, duration=60., period=period,
                                       termination=[termination])
            model = engine.model()
            pts = {name: int(value*4) for name, value in model.default_var_pts.items()}
            simulation = pybamm.Simulation(model, parameter_values=engine.param.copy(),
                experiment=pybamm.Experiment([(step,)]), var_pts=pts,
                solver=pybamm.IDAKLUSolver(rtol=rtol, atol=atol))
            began = time.monotonic()
            solution = simulation.solve(starting_solution=main.copy(), calc_esoh=False,
                                         inputs={'target': target})
            piece = solution.cycles[-1].steps[-1]
            raw = {name: np.asarray(piece[name].entries, dtype=float) for name in VARS}
            if not all(np.all(np.isfinite(value)) for value in raw.values()):
                raise ValueError('Nonfinite CV observations')
            times, currents, capacities = raw[VARS[0]], raw[VARS[1]], raw[VARS[3]]
            if len(times) < 2 or not np.all(np.diff(times) > 0):
                raise ValueError('Invalid CV output time grid')
            dq = float(capacities[-1]-capacities[0])
            integral = float(np.trapezoid(currents, times)/3600.)
            residual, limit = dq-integral, 1e-7+1e-4*abs(dq)
            name = f'replay_{index:02d}_rtol{rtol:g}_period{period:g}.npz'
            np.savez_compressed(out/name, **raw)
            row = {'rtol': rtol, 'atol': atol, 'output_period_s': period,
                'duration_requested_s': 60., 'duration_actual_s': float(times[-1]-times[0]),
                'sample_count': len(times), 'max_output_gap_s': float(np.max(np.diff(times))),
                'Qprior_Ah': original_q, 'Qstart_Ah': float(capacities[0]),
                'Qend_Ah': float(capacities[-1]),
                'Qstart_minus_prior_Ah': float(capacities[0]-original_q),
                'tstart_minus_prior_s': float(times[0]-original_t),
                'Istart_A': float(currents[0]), 'Iend_A': float(currents[-1]),
                'dq_Ah': dq, 'trapz_Ah': integral, 'residual_Ah': residual,
                'original_limit_Ah': limit, 'passes_original_integral_check': abs(residual) <= limit,
                'termination': str(piece.termination), 'target_stoichiometry': target,
                'elapsed_s': time.monotonic()-began, 'raw': name, 'raw_sha256': digest(out/name),
                'main_unchanged': N.fingerprint(main) == original_fingerprint}
            if not row['main_unchanged']:
                raise ValueError('Replay mutated frozen source state')
            report['rows'].append(row)
            save(out/'report.json', report)
            print(json.dumps(row), flush=True)
            del solution, piece, raw, simulation, model, engine
            gc.collect()
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'], report['error'] = 'failed', repr(exc)
        traceback.print_exc()
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        report['source_hashes_unchanged'] = all(digest(path) == value for path, value in protected.items())
        save(out/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(BASE/'cv_replay'))
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if args.worker:
        worker(out)
        return 0
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {'hard_timeout_s': 240, 'max_replays': len(CASES)}
    with (out/'worker.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run([sys.executable, '-u', __file__, '--worker', '--output', str(out)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=240, check=False)
            receipt['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            receipt.update(returncode=1, timed_out=True)
    receipt['elapsed_s'] = time.monotonic()-started
    save(out/'execution.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
