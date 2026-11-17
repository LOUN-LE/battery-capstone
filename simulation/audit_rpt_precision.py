"""RPT-only precision audit of trusted original Linux 100-EFC checkpoints.

Main trajectories are read-only. No mesh interpolation, re-aging, fitting, or
model-physics changes. Q0 and Qref use identical diagnostic solver settings.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import numpy as np
import pybamm
import battery as B
import run_100efc as R
import lifetime_policy as P


def save(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def fingerprint(sol):
    h = hashlib.sha256()
    for t, y in zip(sol.all_ts, sol.all_ys):
        h.update(np.asarray(t, dtype=float).tobytes())
        h.update(np.asarray(y, dtype=float).tobytes())
    return h.hexdigest()


class AuditCell(B.Cell):
    def solve(self, key, steps, start, inputs=None):
        sol, step, wall = super().solve(key, steps, start, inputs)
        self.reference_solution = sol
        return sol, step, wall


def measure(main, factor, mesh, out, label):
    before = fingerprint(main)
    cell = AuditCell(rtol=B.DEFAULT_RTOL*factor, atol=B.DEFAULT_ATOL*factor, mesh_scale=mesh)
    result = cell.reference_test(main.last_state.copy())
    q = result['Q_cum_Ah']
    if not result['ok'] or not np.isfinite(q) or q <= 0:
        raise ValueError('Reference protocol invalid: '+label)
    if abs(q-result['Q_trapz_Ah']) > 1e-6:
        raise ValueError('Reference capacity integral disagreement')
    if before != fingerprint(main):
        raise ValueError('RPT mutated source: '+label)
    raw = {}
    for j, step in enumerate(cell.reference_solution.cycles[-1].steps):
        for name in ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]'):
            raw[f'{j}_{name}'] = np.asarray(step[name].entries, dtype=float)
    path = out/(label+'.npz')
    np.savez_compressed(path, **raw)
    row = {'label': label, 'factor': factor, 'rtol': cell.rtol, 'atol': cell.atol,
        'mesh_scale': mesh, 'Q_Ah': q, 'reference': result,
        'raw': path.name, 'raw_sha256': P.digest(path), 'source_preserved': True}
    del cell
    gc.collect()
    return row


def summarize(rows, threshold_pp):
    comparisons = []
    for factor in (1., .1, .01):
        selected = [r for r in rows if r['factor'] == factor and r.get('repeat', 0) == 0]
        soh = {str(r['current_A']): 100*r['Qref']['Q_Ah']/r['Q0']['Q_Ah'] for r in selected}
        if len(soh) != 4:
            continue
        comparisons.append({'factor': factor, 'SOH100_percent': soh,
            'delta_3p3_minus_3p0_percentage_points': soh['3.3']-soh['3.0'],
            'ascending_current_order': sorted(soh, key=soh.get)})
    if len(comparisons) < 3:
        return {'complete': False, 'comparisons': comparisons}
    delta = [r['delta_3p3_minus_3p0_percentage_points'] for r in comparisons]
    spread = max(delta)-min(delta)
    finest_delta = delta[-1]
    return {'complete': True, 'comparisons': comparisons,
        'delta_max_minus_min_percentage_points': spread,
        'spread_over_finest_abs_delta': spread/abs(finest_delta) if finest_delta else None,
        'rank_stable': all(r['ascending_current_order'] == comparisons[0]['ascending_current_order'] for r in comparisons),
        'delta_precision_screen_limit_percentage_points': threshold_pp,
        'delta_precision_screen_pass': spread <= threshold_pp,
        'screen_note': 'Predeclared RPT-only screening criterion, not a rigorous error bound. Frozen aging trajectories and mesh4 remain untested for tolerance/mesh convergence.',
        'aging_trajectory_convergence_tested': False,
        'real_cell_validation': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--delta-screen-pp', type=float, default=.0001,
                   help='Predeclared screening limit, approx 9%% of original 0.0011pp signal')
    args = p.parse_args()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Use the original Linux environment, never Windows checkpoint migration')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    report = {'status': 'running', 'scope': 'RPT-only; no main trajectory advancement',
        'new_code_sha256': P.digest(__file__), 'rows': [], 'sources': files,
        'started_epoch': time.time(), 'delta_screen_pp': args.delta_screen_pp}
    try:
        versions = {name: importlib.metadata.version(name) for name in
                    ('pybamm', 'pybammsolvers', 'casadi', 'numpy', 'scipy', 'pyarrow')}
        init_cell = B.Cell(mesh_scale=4)
        initial = init_cell.initial_state().last_state.copy()
        fixed_initial_fingerprint = fingerprint(initial)
        # One frozen initial physical state is shared across all diagnostic tolerances.
        q0s = {factor: measure(initial, factor, 4, out, f'initial_factor_{factor:g}')
               for factor in (1., .1, .01)}
        del init_cell
        report['frozen_initial_fingerprint'] = fixed_initial_fingerprint
        report['Q0_precision'] = list(q0s.values())
        save(out/'report.json', report)
        for current in (3., 3.1, 3.2, 3.3):
            directory = Path(args.input_root)/f'current_{current:.1f}A'
            cp, summary = directory/'checkpoint.pkl', directory/'summary.json'
            meta = json.loads(summary.read_text())
            P.validate_legacy_identity(meta['identity'], B.ROOT, cell=4, current=current,
                mesh=4., versions=versions, python=platform.python_version())
            if abs(meta['ledger']['efc']-100.) > 1e-8:
                raise ValueError('Not a 100-EFC source')
            for path in (cp, summary, directory/'diagnostics.json'):
                files[str(path)] = P.digest(path)
            state = R.load_checkpoint(cp)
            if state['identity'] != meta['identity'] or state['ledger'] != meta['ledger']:
                raise ValueError('Checkpoint/summary mismatch')
            main = state['solution'].last_state.copy()
            ledger_before = copy.deepcopy(state['ledger'])
            for factor in (1., .1, .01):
                ref = measure(main, factor, 4, out, f'current_{current:.1f}_factor_{factor:g}')
                report['rows'].append({'current_A': current, 'factor': factor, 'repeat': 0,
                    'Q0': q0s[factor], 'Qref': ref,
                    'stored_Q0_Ah': state['ledger']['q0_Ah'],
                    'SOH_percent': 100*ref['Q_Ah']/q0s[factor]['Q_Ah']})
                save(out/'report.json', report)
            if current in (3., 3.3):
                ref = measure(main, .01, 4, out, f'current_{current:.1f}_factor_0.01_repeat')
                report['rows'].append({'current_A': current, 'factor': .01, 'repeat': 1,
                    'Q0': q0s[.01], 'Qref': ref,
                    'SOH_percent': 100*ref['Q_Ah']/q0s[.01]['Q_Ah']})
            if state['ledger'] != ledger_before or fingerprint(initial) != fixed_initial_fingerprint:
                raise ValueError('Diagnostic changed source ledger/initial state')
            del main, state
            gc.collect()
        report['comparison'] = summarize(report['rows'], args.delta_screen_pp)
        repeats = []
        for current in (3., 3.3):
            vals = [r['SOH_percent'] for r in report['rows'] if r['current_A'] == current and r['factor'] == .01]
            repeats.append({'current_A': current, 'repeat_spread_pp': max(vals)-min(vals)})
        report['tightest_repeats'] = repeats
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'; report['error'] = repr(exc)
        raise
    finally:
        report['source_hashes_unchanged'] = all(P.digest(k) == v for k, v in files.items())
        report['finished_epoch'] = time.time()
        save(out/'report.json', report)
        print(json.dumps({'status': report['status'], 'comparison': report.get('comparison'),
                          'error': report.get('error'), 'sources_preserved': report['source_hashes_unchanged']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
