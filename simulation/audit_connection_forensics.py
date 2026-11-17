"""Instrumented forensic replay of the trusted cell4 Linux lifetime checkpoint.

Goal: observe WHY the unchanged 1e-10 physical-state connection check trips at the
~1614.94 EFC DIS -> REST_AFTER_DISCHARGE transition. This script NEVER edits physics,
configs, judging logic, or snap_diff return values. It only records side-channel
measurements at every solve/transition. The original controller (lifetime_operation.
run_segment via battery.Cell.solve) decides pass/fail exactly as before.

All outputs go to a NEW, disjoint, empty audit directory. The input checkpoint and
every original file stay byte-for-byte untouched (verified before/after). Pickle is
loaded only on the original Linux environment. Instrumentation is memory-only monkey
patching of in-process callables; installed pybamm files on disk are not modified and
patches are restored in finally. Missing required hook measurements prevent the
audit from claiming complete forensic capture or returning successful audit status.

The design.md F1-F3 hypotheses (nextafter 1-ULP; ULP doubling past 2**24 s) are
RECORDED AS HYPOTHESES TO TEST, not conclusions.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

EXPECTED_CHECKPOINT = '293383f8d13a4c262abf8e2e42d7538d03028fc59cf40601bceec5083afa8d64'
EXPECTED_ARCHIVE = '9f4ea2442b778ea4ee5f4438d78bdd1438dda457b477b0fdb85030a195c1202b'
THRESHOLD_UNCHANGED = 1e-10
POW_2_24 = float(2 ** 24)  # 16777216.0 s; design.md F3 boundary to test


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_manifest(root):
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size,
            'sha256': digest(p)} for p in sorted(root.rglob('*')) if p.is_file()}


def float_hex(x):
    try:
        return float(x).hex()
    except (TypeError, ValueError, OverflowError):
        return str(x)


def ulps_between(a, b):
    """Signed count of representable doubles between a and b (0 if equal)."""
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    ia = struct.unpack('<q', struct.pack('<d', a))[0]
    ib = struct.unpack('<q', struct.pack('<d', b))[0]
    if ia < 0:
        ia = -9223372036854775808 - ia
    if ib < 0:
        ib = -9223372036854775808 - ib
    return ib - ia


def ulp_size(x):
    """Spacing to the next representable double above |x|."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return float(math.nextafter(x, math.inf) - x) if hasattr(math, 'nextafter') else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='/var/lib/battery-capstone/results/lifetime_cell4_3p2A_20261028_v2/trajectory/checkpoint.pkl')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-archive', default='/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')
    p.add_argument('--max-wall-s', type=float, default=900)
    p.add_argument('--max-cycles', type=int, default=3)
    p.add_argument('--max-pieces', type=int, default=4000)
    p.add_argument('--max-memory-gib', type=float, default=24)
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def _slice_report(np, y_from, y_to, slices):
    """Per-variable absolute/relative diff between two y vectors, grouped by model slice."""
    out = {}
    yf = np.ravel(np.asarray(y_from, dtype=float))
    yt = np.ravel(np.asarray(y_to, dtype=float))
    for name, sl in slices.items():
        try:
            a = yf[sl]; b = yt[sl]
        except Exception:
            continue
        if a.shape != b.shape or a.size == 0:
            continue
        d = np.abs(a - b)
        den = float(np.max(np.abs(a)))
        out[name] = {'max_abs_diff': float(np.max(d)),
                     'ref_max_abs': den,
                     'max_rel_diff': float(np.max(d) / den) if den > 0 else (0.0 if not np.any(d) else float('inf')),
                     'size': int(a.size)}
    return out


def _model_slices(model):
    """rhs vs algebraic index slices from a pybamm model, if available."""
    info = {'rhs_names': [], 'alg_names': [], 'y_slices': {}}
    try:
        y_slices = getattr(model, 'y_slices', None)
        if y_slices:
            for var, sls in y_slices.items():
                name = getattr(var, 'name', str(var))
                sl = sls[0] if isinstance(sls, (list, tuple)) else sls
                info['y_slices'][name] = (int(sl.start), int(sl.stop))
    except Exception:
        pass
    try:
        info['len_rhs'] = int(model.len_rhs)
        info['len_alg'] = int(model.len_alg)
        info['rhs_names'] = [getattr(v, 'name', str(v)) for v in model.rhs]
        info['alg_names'] = [getattr(v, 'name', str(v)) for v in model.algebraic]
    except Exception:
        pass
    return info


class ForensicHooks:
    """Observe native solver inputs/outputs without replacing any computed value.

    Keep six recent steps in memory. Write full arrays only on a controller phase
    transition or an offending join; JSONL contains compact summaries otherwise.
    This class also supports a fresh-state hook smoke test, without any pickle.
    """
    REQUIRED = ('y_prev_end', 'y_mapped_before_ic', 'y_after_ic',
                'y0full_at_integrate', 'ydot0full_at_integrate', 't_eval',
                'y_first_output', 't_first_output')

    def __init__(self, emit, sequence):
        import numpy as np
        self.np, self.emit, self.sequence = np, emit, sequence
        self.records, self.installed, self.unavailable = [], [], []
        self.context = None
        self.total_steps = self.complete_steps = 0

    def _vector(self, values):
        return self.np.asarray(values, dtype=float).reshape(-1).copy()

    def _capture(self, rec, name, values):
        rec['_arrays'][name] = self._vector(values)

    def _error(self, rec, where, exc):
        rec.setdefault('measurement_errors', []).append(where + ': ' + str(exc))

    def public(self, rec):
        return {k: v for k, v in rec.items() if k != '_arrays'}

    def install(self):
        from pybamm.solvers.base_solver import BaseSolver
        from pybamm.solvers.idaklu_solver import IDAKLUSolver
        owner = self
        original_step = BaseSolver.step
        original_ic = IDAKLUSolver._set_consistent_initialization
        original_integrate = IDAKLUSolver._integrate

        def consistent(solver, model, time, inputs_list):
            rec = owner.context
            if rec is not None:
                try:
                    owner._capture(rec, 'y_mapped_before_ic', model.y0_list[0])
                    rec['ic_time_s'] = float(time)
                    rec['ic_time_hex'] = float_hex(time)
                    rec['new_model_slices'] = _model_slices(model)
                    rec['ic_input_count'] = len(inputs_list)
                except Exception as exc:
                    owner._error(rec, 'ic_pre', exc)
            result = original_ic(solver, model, time, inputs_list)
            if rec is not None:
                try:
                    owner._capture(rec, 'y_after_ic', model.y0_list[0])
                    owner._capture(rec, 'y0full_after_ic', model.y0full[0])
                    owner._capture(rec, 'ydot0full_after_ic', model.ydot0full[0])
                except Exception as exc:
                    owner._error(rec, 'ic_post', exc)
            return result

        def integrate(solver, model, t_eval, inputs_list=None, t_interp=None, nproc=None):
            rec = owner.context
            if rec is not None:
                try:
                    owner._capture(rec, 't_eval', t_eval)
                    owner._capture(rec, 't_interp', [] if t_interp is None else t_interp)
                    owner._capture(rec, 'y0full_at_integrate', model.y0full[0])
                    owner._capture(rec, 'ydot0full_at_integrate', model.ydot0full[0])
                except Exception as exc:
                    owner._error(rec, 'integrate_pre', exc)
            result = original_integrate(solver, model, t_eval, inputs_list, t_interp, nproc)
            if rec is not None:
                try:
                    first = result[0]
                    owner._capture(rec, 'y_first_output', first.all_ys[0][:, 0])
                    owner._capture(rec, 't_first_output', [float(first.t[0])])
                    rec['first_output_t_s'] = float(first.t[0])
                    rec['first_output_t_hex'] = float_hex(float(first.t[0]))
                except Exception as exc:
                    owner._error(rec, 'integrate_post', exc)
            return result

        def stepped(solver, old_solution, model, dt_, *a, **kw):
            rec = {'event': 'solver_step', 'sequence': owner.sequence(),
                   'dt_s': float(dt_), '_arrays': {}}
            previous_context = owner.context
            owner.context = rec
            try:
                try:
                    old_t = float(old_solution.t[-1])
                    rec.update(prev_t_s=old_t, prev_t_hex=float_hex(old_t),
                               ulp_size_at_prev_t=ulp_size(old_t),
                               previous_model_slices=_model_slices(old_solution.all_models[-1]))
                    owner._capture(rec, 'y_prev_end', old_solution.last_state.all_ys[0])
                except Exception as exc:
                    owner._error(rec, 'step_pre', exc)
                result = original_step(solver, old_solution, model, dt_, *a, **kw)
                return result
            finally:
                owner.context = previous_context
                arrays = rec['_arrays']
                rec['missing_measurements'] = [k for k in owner.REQUIRED if k not in arrays]
                rec['capture_complete'] = not rec['missing_measurements']
                old_slices = rec.get('previous_model_slices', {})
                new_slices = rec.get('new_model_slices', {})
                rec['state_layout_equal'] = old_slices == new_slices
                try:
                    if 't_first_output' in arrays and 'prev_t_s' in rec:
                        rec['ulps_prev_to_new_first'] = ulps_between(rec['prev_t_s'], arrays['t_first_output'][0])
                    comparisons = (('mapping_change', 'y_prev_end', 'y_mapped_before_ic'),
                                   ('ic_change', 'y_mapped_before_ic', 'y_after_ic'),
                                   ('first_output_change', 'y0full_at_integrate', 'y_first_output'))
                    for label, before, after in comparisons:
                        if before not in arrays or after not in arrays:
                            continue
                        if label == 'mapping_change' and not rec['state_layout_equal']:
                            rec[label] = {'skipped': 'state_layout_differs; compare named slices separately'}
                            continue
                        rec[label] = _slice_report(owner.np, arrays[before], arrays[after],
                            {k: slice(*v) for k, v in new_slices.get('y_slices', {}).items()})
                except Exception as exc:
                    owner._error(rec, 'step_summary', exc)
                owner.total_steps += 1
                owner.complete_steps += int(rec['capture_complete'])
                owner.records.append(rec)
                owner.records = owner.records[-6:]
                owner.emit(owner.public(rec))

        for target, name, replacement in ((BaseSolver, 'step', stepped),
                (IDAKLUSolver, '_set_consistent_initialization', consistent),
                (IDAKLUSolver, '_integrate', integrate)):
            self.installed.append((target, name, getattr(target, name)))
            setattr(target, name, replacement)

    def save_latest(self, out, reason):
        paths = []
        for index, rec in enumerate(self.records):
            if rec['sequence'] != self.sequence():
                continue
            name = f"solver_step_{rec['sequence']:06d}_{index}.npz"
            self.np.savez_compressed(out / name, **rec['_arrays'])
            rec['array_artifact'] = name
            save(out / name.replace('.npz', '.json'), {**self.public(rec), 'detail_reason': reason})
            paths.append(name)
        return paths

    def restore(self):
        for target, name, original in reversed(self.installed):
            setattr(target, name, original)
        self.installed.clear()


def replay(args):
    import numpy as np
    import battery as B
    import run_100efc as R
    import run_lifetime as L
    import lifetime_operation as O
    import lifetime_policy as P

    out = Path(args.output_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    began = time.perf_counter()
    stop = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, _frame: stop.update(signal=signum))
    log = (out / 'connections.jsonl').open('w', encoding='utf-8', buffering=1)
    result = {'status': 'initializing', 'checkpoint': str(checkpoint),
              'threshold_unchanged': THRESHOLD_UNCHANGED, 'controller_patched': False,
              'instrumentation_only': True, 'hook_unavailable': [],
              'cache_limitation': 'Original live solver caches were not serialized. Replay starts from the same stored physical state and inputs with newly built caches.',
              'hypotheses': {
                  'F1_nextafter_1ulp': 'DIS-end time and REST first stored time differ by exactly 1 ULP via np.nextafter (base_solver.py L1453). TEST, not assumed.',
                  'F1b_jump_tracks_ulp': 'Connection jump magnitude tracks ULP(t) size. TEST.',
                  'F3_ulp_doubling_past_2**24s': 't_abs beyond 2**24 s = 16777216 s has doubled ULP spacing. TEST.',
                  'conclusion': 'none; measurements below decide'},
              'environment': {'python': platform.python_version(),
                    'platform': platform.platform(), 'machine': platform.machine(),
                    'versions': L.versions(),
                    'thread_environment': {k: os.environ.get(k) for k in
                       ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}}
    active = {'solve': {}, 'cursor': None, 'solves': 0, 'joins': 0,
              'last_diff': None, 'last_join': None, 'last_valid': None,
              'prev_piece_end': None, 'last_step_hooks': []}
    worst = {}
    original_diff = B.snap_diff
    original_solve = None
    cell = None

    def emit(event):
        event.update(audit_wall_s=time.perf_counter() - began)
        log.write(json.dumps(clean(event), ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False) + '\n')

    def arrays(path, values):
        np.savez_compressed(out / path, **{k: np.asarray(v).copy()
                                         for k, v in values.items()})

    def cache_info():
        return {'simulation_cache_count': len(cell._sims),
                'model_builds': cell.builds, 'esoh_calls': cell.esoh_calls,
                'memory': B.memory_mb()}

    step_hooks = ForensicHooks(emit, lambda: active['solves'])

    def snap_diff(sa, sb):
        dif = original_diff(sa, sb)  # EXACT original result returned to O.run_segment.
        active['joins'] += 1
        details = {}
        for key, a in sa.items():
            b = sb[key]
            details[key] = dict(dif[key])
            details[key].update(variable=B.STATE_VARS.get(key),
               previous_shape=list(a.shape), next_shape=list(b.shape),
               included_in_original_jump=key != 'discharge_capacity')
            if a.shape == b.shape:
                absolute = np.abs(a - b)
                idx = int(np.argmax(absolute))
                den = float(np.max(np.abs(a)))
                details[key].update(denominator=den,
                    denominator_rule='max(abs(previous_array)); no absolute-error floor',
                    argmax_abs_flat_index=idx, previous_at_argmax=float(a.flat[idx]),
                    next_at_argmax=float(b.flat[idx]))
                if key != 'discharge_capacity':
                    old = worst.get(key)
                    if old is None or dif[key]['max_rel_diff'] > old['max_rel_diff']:
                        worst[key] = {**details[key], 'join_sequence': active['joins']}
        relevant = [(key, values.get('max_rel_diff', 0.0))
                    for key, values in dif.items() if key != 'discharge_capacity']
        culprit, jump = max(relevant, key=lambda item: item[1], default=(None, 0.0))
        event = {'event': 'connection_comparison', 'join_sequence': active['joins'],
                 'variables': details, 'jump': jump, 'max_variable': culprit,
                 'unchanged_limit': THRESHOLD_UNCHANGED,
                 'would_fail_original_limit': jump > THRESHOLD_UNCHANGED,
                 'recent_solver_step_hooks': [step_hooks.public(r) for r in step_hooks.records],
                 'shape_mismatch': any('status' in values for values in dif.values())}
        active['last_diff'] = clean(details)
        active['last_join'] = clean(event)
        emit(event)
        if jump > THRESHOLD_UNCHANGED or event['shape_mismatch'] or active['solve'].get('phase_transition'):
            event['solver_array_artifacts'] = step_hooks.save_latest(out, 'failure' if jump > THRESHOLD_UNCHANGED else 'phase_transition')
        if jump > THRESHOLD_UNCHANGED or event['shape_mismatch']:
            arrays('failing_join_previous_end.npz', sa)
            arrays('failing_join_new_start.npz', sb)
            save(out / 'failing_join.json', event)
        return dif

    def solve(key, steps, start, inputs=None):
        active['solves'] += 1
        before = {'event': 'solve_request', 'sequence': active['solves'],
                  'key': repr(key), 'kind': key[0] if isinstance(key, tuple) else key,
                  'inputs': copy.deepcopy(inputs or {}),
                  'start_t_s': float(start.t[-1]), 'start_t_hex': float_hex(float(start.t[-1])),
                  'start_t_abs_ge_2pow24': float(start.t[-1]) >= POW_2_24,
                  'ulp_size_at_start_t': ulp_size(float(start.t[-1])),
                  'cache_hit': key in cell._sims, 'resources': cache_info(),
                  'phase_transition': active.get('previous_solve_kind', (active['cursor'] or {}).get('prev_kind')) != (key[0] if isinstance(key, tuple) else key)}
        active['previous_solve_kind'] = key[0] if isinstance(key, tuple) else key
        active['solve'] = before
        emit(before.copy())
        new, st, wall = original_solve(key, steps, start, inputs)
        after = {'termination': str(st.termination), 'classification': B.classify(st),
                 'piece_start_t_s': float(st.t[0]), 'piece_start_t_hex': float_hex(float(st.t[0])),
                 'piece_end_t_s': float(st.t[-1]), 'solve_wall_s': wall}
        if active['prev_piece_end'] is not None:
            after['prev_piece_end_t_s'] = active['prev_piece_end']
            after['ulps_prev_end_to_this_start'] = ulps_between(active['prev_piece_end'], float(st.t[0]))
            after['abs_gap_prev_end_to_this_start_s'] = float(st.t[0]) - active['prev_piece_end']
        active['prev_piece_end'] = float(st.t[-1])
        active['solve'].update(after)
        emit({'event': 'solve_completed', 'sequence': active['solves'], **after})
        return new, st, wall

    def guard():
        if stop['signal'] is not None:
            return 'audit_signal_stop'
        if time.perf_counter() - began >= args.max_wall_s:
            return 'audit_wall_budget'
        if active['solves'] >= args.max_pieces:
            return 'audit_piece_budget'
        if B.memory_mb().get('rss_mb', 0) > args.max_memory_gib * 1024:
            return 'audit_memory_budget'
        return None

    def observe(main, st, kind, bounds, ledger):
        active['last_valid'] = B.snapshot(st, -1)
        active['last_valid_ledger'] = B.ledger(ledger)
        emit({'event': 'piece_accepted', 'sequence': active['solves'], 'kind': kind,
              'ledger': B.ledger(ledger)})

    def diagnose(main, ledger, target):
        before = B.y_end(main)
        book = copy.deepcopy(B.ledger(ledger))
        ref = cell.reference_test(main.last_state.copy())
        q = ref['Q_cum_Ah']
        if (not ref['ok'] or not np.isfinite(q) or q <= 0
                or abs(ref['Q_trapz_Ah'] - q) / q > 1e-4):
            raise RuntimeError('Invalid reference capacity')
        if not np.array_equal(before, B.y_end(main)) or book != B.ledger(ledger):
            raise RuntimeError('Diagnostic modified main state')
        if ledger.q0_Ah is None:
            ledger.q0_Ah = q
        d = {'event': 'reference_test', 'target_efc': target, 'efc': ledger.efc,
             'Qref_Ah': q, 'SOH_percent': 100 * q / ledger.q0_Ah}
        result.setdefault('replay_diagnostics', []).append(d)
        emit(d)
        return 'eol_observed' if d['SOH_percent'] <= run_args.eol_soh else guard()

    try:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('AWS checkpoint replay requires the original Linux environment')
        step_hooks.install()
        source = R.load_checkpoint(checkpoint)
        sig = source['identity']
        run_args = SimpleNamespace(cell=sig['cell'], current=sig['current_A'],
            mesh_scale=sig['mesh_scale'], efc=sig['target_EFC'],
            eol_soh=sig['eol_SOH_percent'],
            rpt_every_efc=sig['rpt_every_after_100_EFC'])
        actual = L.identity(run_args)
        if sig != actual:
            mismatches = {key: {'checkpoint': sig.get(key), 'actual': actual.get(key)}
                          for key in set(sig) | set(actual) if sig.get(key) != actual.get(key)}
            save(out / 'identity_mismatch.json', mismatches)
            raise RuntimeError('Original source/config/dependency identity mismatch')
        if source.get('terminal'):
            raise RuntimeError('Forensic continuation requires a nonterminal checkpoint')
        baseline = {rel: {'expected': expected, 'actual': digest(B.ROOT / rel)}
                    for rel, expected in P.LEGACY_SHA256.items()}
        if any(v['expected'] != v['actual'] for v in baseline.values()):
            raise RuntimeError('Frozen physical baseline differs')
        state = copy.deepcopy({k: v for k, v in source.items() if k != 'solution'})
        sol = source['solution'].copy()
        source_y = B.y_end(source['solution'])
        rs = B.RunState(**state['ledger'])
        cursor = copy.deepcopy(state['cursor'])
        active['cursor'] = cursor
        cell = L.make_cell(run_args)
        original_solve = cell.solve
        cell.solve = solve
        B.snap_diff = snap_diff
        active['last_valid'] = B.snapshot(B.last_step(sol), -1)
        active['last_valid_ledger'] = B.ledger(rs)
        arrays('initial_checkpoint_snapshot.npz', active['last_valid'])
        result.update(status='replaying', checkpoint_identity=sig,
                      frozen_physical_baseline=baseline,
                      initial_ledger=B.ledger(rs), initial_cursor=copy.deepcopy(cursor))
        emit({'event': 'audit_started', 'initial_ledger': B.ledger(rs), 'identity': sig,
              'cache_limitation': result['cache_limitation']})
        targets = P.diagnostic_schedule(run_args.efc, run_args.rpt_every_efc)
        completed = 0
        while completed < args.max_cycles:
            status = guard()
            if status:
                result['status'] = status
                break
            sol, cursor, status, checks, cycle = O.run_segment(cell, sol, rs,
                cursor, run_args.current, run_args.efc, targets, diagnose, observe, guard)
            active['cursor'] = cursor
            emit({'event': 'segment_returned', 'status': status, 'checks': checks,
                  'completed_cycle': cycle, 'ledger': B.ledger(rs)})
            if status != 'cycle_complete':
                result['status'] = status
                break
            completed += 1
        else:
            result['status'] = 'not_reproduced_within_cycle_bound'
        result.update(completed_cycles=completed,
                      source_solution_unchanged_in_memory=bool(np.array_equal(source_y, B.y_end(source['solution']))))
    except BaseException as exc:
        message = str(exc)
        reproduced = isinstance(exc, RuntimeError) and message == 'Physical state connection mismatch'
        result.update(status='original_connection_failure_reproduced' if reproduced else 'audit_exception',
                      exception_type=type(exc).__name__, exception=message,
                      traceback=traceback.format_exc(),
                      original_failure_reproduced=reproduced)
        emit({'event': 'audit_exception', 'exception_type': type(exc).__name__,
              'exception': message, 'traceback': result['traceback']})
    finally:
        B.snap_diff = original_diff
        if original_solve is not None and cell is not None:
            cell.solve = original_solve
        step_hooks.restore()
        if active['last_valid'] is not None:
            arrays('last_accepted_snapshot.npz', active['last_valid'])
        result.update(elapsed_wall_s=time.perf_counter() - began,
                      solves=active['solves'], connection_comparisons=active['joins'],
                      last_join=active['last_join'], worst_by_variable=worst,
                      forensic_measurements_complete=(step_hooks.total_steps > 0 and step_hooks.complete_steps == step_hooks.total_steps),
                      forensic_steps=step_hooks.total_steps, forensic_complete_steps=step_hooks.complete_steps,
                      memory=B.memory_mb())
        save(out / 'replay_result.json', result)
        log.close()
    return 0 if result.get('forensic_measurements_complete') and result.get('status') in ('original_connection_failure_reproduced',
                   'not_reproduced_within_cycle_bound', 'audit_wall_budget',
                   'audit_piece_budget', 'audit_memory_budget', 'audit_signal_stop') else 1


def main():
    args = parser().parse_args()
    if not (0 < args.max_wall_s <= 3600 and 0 < args.max_cycles <= 100
            and 0 < args.max_pieces <= 20000 and 0 < args.max_memory_gib <= 24):
        raise SystemExit('Bounded limits required: wall<=3600s, cycles<=100, pieces<=20000, memory<=24GiB')
    if args._worker:
        raise SystemExit(replay(args))
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    archive = Path(args.source_archive).resolve(strict=True)
    provenance = {'checkpoint_sha256': digest(checkpoint), 'checkpoint_expected_sha256': EXPECTED_CHECKPOINT,
                  'archive_sha256': digest(archive), 'archive_expected_sha256': EXPECTED_ARCHIVE}
    if provenance['checkpoint_sha256'] != EXPECTED_CHECKPOINT or provenance['archive_sha256'] != EXPECTED_ARCHIVE:
        raise SystemExit('Trusted 2026-10-29 checkpoint/archive SHA-256 mismatch; forensic replay refused')
    source_root = checkpoint.parent.parent
    out = Path(args.output_dir).resolve()
    if out == source_root or source_root in out.parents or out in source_root.parents:
        raise SystemExit('Audit output must be disjoint from the original experiment tree')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Audit output directory must be new and empty')
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'source_provenance.json', provenance)
    before = tree_manifest(source_root)
    save(out / 'original_files_before.json', before)
    import lifetime_supervision as S
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    forced = {}

    def forced_stop(reason):
        forced.update(reason=reason, kill_sent=True)
        save(out / 'forced_stop.json', forced)

    worker_code = None
    parent_exception = None
    try:
        worker_code = S.supervise([sys.executable, str(Path(__file__).resolve()),
                   *sys.argv[1:], '--_worker'], out / 'audit.runner.lock',
                   args.max_wall_s, on_forced_stop=forced_stop)
    except BaseException:
        parent_exception = traceback.format_exc()
    finally:
        after = tree_manifest(source_root)
        save(out / 'original_files_after.json', after)
        changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        worker_result = json.loads((out / 'replay_result.json').read_text(encoding='utf-8')) if (out / 'replay_result.json').is_file() else None
        summary = {'audit_started_utc': started_utc, 'audit_finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'trusted_source_provenance': provenance,
            'worker_exit_code': worker_code, 'parent_exception': parent_exception,
            'parent_wall_s': time.perf_counter() - started, 'forced_stop': forced or None,
            'original_root': str(source_root), 'original_file_count': len(before),
            'original_files_unchanged': not changed, 'changed_original_paths': changed,
            'audit_script_sha256': digest(Path(__file__).resolve()),
            'worker_result_path': str(out / 'replay_result.json'),
            'worker_status': (worker_result or {}).get('status'),
            'original_failure_reproduced': (worker_result or {}).get('original_failure_reproduced', False),
            'forensic_measurements_complete': (worker_result or {}).get('forensic_measurements_complete', False),
            'instrumentation_only': True,
            'max_wall_s': args.max_wall_s, 'threshold_unchanged': THRESHOLD_UNCHANGED,
            'output_dir': str(out)}
        save(out / 'audit_summary.json', summary)
        print(json.dumps(clean(summary), ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if worker_code == 0 and not changed and parent_exception is None else 1)


if __name__ == '__main__':
    main()
