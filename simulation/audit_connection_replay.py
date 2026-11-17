"""Forensic replay of a trusted Linux lifetime checkpoint; physical code is unchanged.

All results go to a NEW disjoint audit directory. The original controller decides
whether the unchanged 1e-10 connection test fails. Instrumentation never edits its
return values. A fresh process reconstructs Simulation/solver caches: this is an
explicit limitation when comparing with the original long-lived process.
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
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='/var/lib/battery-capstone/results/lifetime_cell4_3p2A_20261028_v2/trajectory/checkpoint.pkl')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-archive', default='/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')
    p.add_argument('--max-wall-s', type=float, default=600)
    p.add_argument('--max-cycles', type=int, default=3)
    p.add_argument('--max-pieces', type=int, default=2000)
    p.add_argument('--max-memory-gib', type=float, default=24)
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def replay(args):
    # Physics imports are worker-only. Main process can bound a stuck native solve.
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
              'threshold_unchanged': 1e-10, 'controller_patched': False,
              'simulation_caches_reconstructed': True,
              'cache_limitation': 'Original live solver caches were not serialized. Replay starts from the same stored physical state and inputs with newly built caches.',
              'environment': {'python': platform.python_version(),
                    'platform': platform.platform(), 'machine': platform.machine(),
                    'versions': L.versions(),
                    'thread_environment': {k: os.environ.get(k) for k in
                       ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}}
    active = {'solve': {}, 'cursor': None, 'solves': 0, 'joins': 0,
              'last_diff': None, 'last_join': None, 'last_valid': None}
    worst = {}
    original_diff = B.snap_diff
    original_solve = None
    state = None
    sol = None
    rs = None

    def emit(event):
        event.update(audit_wall_s=time.perf_counter()-began)
        log.write(json.dumps(clean(event), ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False) + '\n')

    def arrays(path, values):
        np.savez_compressed(out / path, **{k: np.asarray(v).copy()
                                         for k, v in values.items()})

    def cache_info():
        return {'simulation_cache_count': len(cell._sims),
                'simulation_cache_keys': [repr(k) for k in cell._sims],
                'model_builds': cell.builds, 'esoh_calls': cell.esoh_calls,
                'esoh_wall_s': cell.esoh_wall_s, 'memory': B.memory_mb()}

    def shape_of(step):
        return [list(np.asarray(y).shape) for y in step.all_ys]

    def solve(key, steps, start, inputs=None):
        active['solves'] += 1
        ctx = active['cursor'] or {}
        before = {'event': 'solve_request', 'sequence': active['solves'],
                  'key': repr(key), 'kind': key[0] if isinstance(key, tuple) else key,
                  'steps': [str(s) for s in steps], 'inputs': copy.deepcopy(inputs or {}),
                  'previous_kind': ctx.get('prev_kind'), 'phase': ctx.get('phase'),
                  'cycle_context': copy.deepcopy(ctx.get('cycle')),
                  'source_ledger': B.ledger(rs),
                  'start_t_s': float(start.t[-1]), 'start_y_shapes': shape_of(start),
                  'cache_hit': key in cell._sims, 'resources': cache_info()}
        active['solve'] = before
        emit(before.copy())
        try:
            new, st, wall = original_solve(key, steps, start, inputs)
        except BaseException:
            emit({'event': 'solve_exception', 'sequence': active['solves'],
                  'traceback': traceback.format_exc(), 'resources': cache_info()})
            raise
        after = {'termination': str(st.termination), 'classification': B.classify(st),
                 'piece_start_t_s': float(st.t[0]), 'piece_end_t_s': float(st.t[-1]),
                 'piece_duration_s': float(st.t[-1]-st.t[0]), 'solve_wall_s': wall,
                 'y_shapes': shape_of(st),
                 'event_time': getattr(st, 't_event', None),
                 'solve_time': getattr(st, 'solve_time', None),
                 'integration_time': getattr(st, 'integration_time', None),
                 'set_up_time': getattr(st, 'set_up_time', None),
                 'solver_type': type(cell._sims[key].solver).__name__,
                 'rtol': cell.rtol, 'atol': cell.atol, 'resources_after': cache_info()}
        active['solve'].update(after)
        emit({'event': 'solve_completed', 'sequence': active['solves'], **after})
        return new, st, wall

    def snap_diff(sa, sb):
        dif = original_diff(sa, sb)  # Return EXACT original result to O.run_segment.
        active['joins'] += 1
        details = {}
        for key, a in sa.items():
            b = sb[key]
            details[key] = dict(dif[key])
            details[key].update(variable=B.STATE_VARS.get(key),
               previous_shape=list(a.shape), next_shape=list(b.shape),
               included_in_original_jump=key != 'discharge_capacity')
            if a.shape == b.shape:
                absolute = np.abs(a-b)
                idx = int(np.argmax(absolute))
                den = float(np.max(np.abs(a)))
                details[key].update(denominator=den,
                    denominator_rule='max(abs(previous_array)); no absolute-error floor',
                    argmax_abs_flat_index=idx, previous_at_argmax=float(a.flat[idx]),
                    next_at_argmax=float(b.flat[idx]), previous_min=float(np.min(a)),
                    previous_max=float(np.max(a)), next_min=float(np.min(b)),
                    next_max=float(np.max(b)))
                if key != 'discharge_capacity':
                    old = worst.get(key)
                    if old is None or dif[key]['max_rel_diff'] > old['max_rel_diff']:
                        worst[key] = {**details[key], 'join_sequence': active['joins']}
        relevant = [(key, values.get('max_rel_diff', 0.0))
                    for key, values in dif.items() if key != 'discharge_capacity']
        culprit, jump = max(relevant, key=lambda item: item[1], default=(None, 0.0))
        event = {'event': 'connection_comparison', 'join_sequence': active['joins'],
                 'solve': copy.deepcopy(active['solve']), 'variables': details,
                 'jump': jump, 'max_variable': culprit, 'unchanged_limit': 1e-10,
                 'would_fail_original_limit': jump > 1e-10,
                 'shape_mismatch': any('status' in values for values in dif.values())}
        active['last_diff'] = clean(details)
        active['last_join'] = clean(event)
        emit(event)
        if jump > 1e-10 or event['shape_mismatch']:
            arrays('failing_join_previous_end.npz', sa)
            arrays('failing_join_new_start.npz', sb)
            save(out/'failing_join.json', event)
        return dif

    def guard():
        if stop['signal'] is not None:
            return 'audit_signal_stop'
        if time.perf_counter()-began >= args.max_wall_s:
            return 'audit_wall_budget'
        if active['solves'] >= args.max_pieces:
            return 'audit_piece_budget'
        if B.memory_mb().get('rss_mb', 0) > args.max_memory_gib*1024:
            return 'audit_memory_budget'
        return None

    def observe(main, st, kind, bounds, ledger):
        active['last_valid'] = B.snapshot(st, -1)
        active['last_valid_ledger'] = B.ledger(ledger)
        emit({'event':'piece_accepted', 'sequence':active['solves'], 'kind':kind,
              'ledger': B.ledger(ledger), 'soc_bounds': copy.deepcopy(bounds)})

    def diagnose(main, ledger, target):
        before = B.y_end(main)
        book = copy.deepcopy(B.ledger(ledger))
        ref = cell.reference_test(main.last_state.copy())
        q = ref['Q_cum_Ah']
        if (not ref['ok'] or not np.isfinite(q) or q <= 0
                or abs(ref['Q_trapz_Ah']-q)/q > 1e-4):
            raise RuntimeError('Invalid reference capacity')
        if not np.array_equal(before, B.y_end(main)) or book != B.ledger(ledger):
            raise RuntimeError('Diagnostic modified main state')
        if ledger.q0_Ah is None:
            ledger.q0_Ah = q
        d = {'event':'reference_test','target_efc':target,'efc':ledger.efc,
             'Qref_Ah':q,'SOH_percent':100*q/ledger.q0_Ah,
             'state_max_abs_change':0.0,'ledger_unchanged':True,'reference':ref}
        result.setdefault('replay_diagnostics', []).append(d)
        emit(d)
        return 'eol_observed' if d['SOH_percent'] <= run_args.eol_soh else guard()

    try:
        if not sys.platform.startswith('linux'):
            raise RuntimeError('AWS checkpoint replay requires the original Linux environment')
        source = R.load_checkpoint(checkpoint)
        sig = source['identity']
        run_args = SimpleNamespace(cell=sig['cell'], current=sig['current_A'],
            mesh_scale=sig['mesh_scale'], efc=sig['target_EFC'],
            eol_soh=sig['eol_SOH_percent'],
            rpt_every_efc=sig['rpt_every_after_100_EFC'])
        actual = L.identity(run_args)
        if sig != actual:
            mismatches = {key:{'checkpoint':sig.get(key),'actual':actual.get(key)}
                          for key in set(sig)|set(actual) if sig.get(key) != actual.get(key)}
            save(out/'identity_mismatch.json', mismatches)
            raise RuntimeError('Original source/config/dependency identity mismatch')
        if source.get('terminal'):
            raise RuntimeError('Forensic continuation requires a nonterminal checkpoint')
        baseline = {rel:{'expected':expected, 'actual':digest(B.ROOT/rel)}
                    for rel, expected in P.LEGACY_SHA256.items()}
        if any(v['expected'] != v['actual'] for v in baseline.values()):
            raise RuntimeError('Frozen physical baseline differs')
        # Memory-only copies; the input checkpoint and source files stay untouched.
        state = copy.deepcopy({k:v for k,v in source.items() if k != 'solution'})
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
                      initial_ledger=B.ledger(rs), initial_cursor=copy.deepcopy(cursor),
                      original_source_exception= 'RuntimeError: Physical state connection mismatch')
        emit({'event':'audit_started', 'initial_ledger':B.ledger(rs),
              'identity':sig, 'cache_limitation':result['cache_limitation']})
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
            emit({'event':'segment_returned','status':status,'checks':checks,
                  'completed_cycle':cycle,'ledger':B.ledger(rs)})
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
        emit({'event':'audit_exception','exception_type':type(exc).__name__,
              'exception':message,'traceback':result['traceback']})
    finally:
        B.snap_diff = original_diff
        if original_solve is not None:
            cell.solve = original_solve
        if active['last_valid'] is not None:
            arrays('last_accepted_snapshot.npz', active['last_valid'])
        result.update(elapsed_wall_s=time.perf_counter()-began,
                      solves=active['solves'], connection_comparisons=active['joins'],
                      last_join=active['last_join'], worst_by_variable=worst,
                      last_accepted_ledger=active.get('last_valid_ledger'),
                      current_ledger=B.ledger(rs) if rs else None,
                      memory=B.memory_mb())
        save(out/'replay_result.json', result)
        log.close()
    return 0 if result.get('status') in ('original_connection_failure_reproduced',
                   'not_reproduced_within_cycle_bound','audit_wall_budget',
                   'audit_piece_budget','audit_memory_budget','audit_signal_stop') else 1


def main():
    args = parser().parse_args()
    if not (0 < args.max_wall_s <= 3600 and 0 < args.max_cycles <= 100
            and 0 < args.max_pieces <= 20000 and 0 < args.max_memory_gib <= 24):
        raise SystemExit('Bounded limits required: wall<=3600s, cycles<=100, pieces<=20000, memory<=24GiB')
    if args._worker:
        raise SystemExit(replay(args))
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    archive = Path(args.source_archive).resolve(strict=True)
    expected_checkpoint = '293383f8d13a4c262abf8e2e42d7538d03028fc59cf40601bceec5083afa8d64'
    expected_archive = '9f4ea2442b778ea4ee5f4438d78bdd1438dda457b477b0fdb85030a195c1202b'
    provenance = {'checkpoint_sha256': digest(checkpoint), 'checkpoint_expected_sha256': expected_checkpoint,
                  'archive_sha256': digest(archive), 'archive_expected_sha256': expected_archive}
    if provenance['checkpoint_sha256'] != expected_checkpoint or provenance['archive_sha256'] != expected_archive:
        raise SystemExit('Trusted 2026-10-29 checkpoint/archive SHA-256 mismatch; replay refused')
    source_root = checkpoint.parent.parent
    out = Path(args.output_dir).resolve()
    if out == source_root or source_root in out.parents or out in source_root.parents:
        raise SystemExit('Audit output must be disjoint from the original experiment tree')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Audit output directory must be new and empty')
    out.mkdir(parents=True, exist_ok=True)
    save(out/'source_provenance.json', provenance)
    before = tree_manifest(source_root)
    save(out/'original_files_before.json', before)
    import lifetime_supervision as S
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    forced = {}
    def forced_stop(reason):
        forced.update(reason=reason, kill_sent=True)
        save(out/'forced_stop.json', forced)
    worker_code = None
    parent_exception = None
    try:
        worker_code = S.supervise([sys.executable, str(Path(__file__).resolve()),
                   *sys.argv[1:], '--_worker'], out/'audit.runner.lock',
                   args.max_wall_s, on_forced_stop=forced_stop)
    except BaseException:
        parent_exception = traceback.format_exc()
    finally:
        after = tree_manifest(source_root)
        save(out/'original_files_after.json', after)
        changed = [key for key in sorted(set(before)|set(after)) if before.get(key) != after.get(key)]
        worker_result = json.loads((out/'replay_result.json').read_text(encoding='utf-8')) if (out/'replay_result.json').is_file() else None
        summary = {'audit_started_utc':started_utc, 'audit_finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'trusted_source_provenance':provenance,
            'worker_exit_code':worker_code, 'parent_exception':parent_exception,
            'parent_wall_s':time.perf_counter()-started, 'forced_stop':forced or None,
            'original_root':str(source_root),'original_file_count':len(before),
            'original_files_unchanged':not changed,'changed_original_paths':changed,
            'audit_script_sha256':digest(Path(__file__).resolve()),
            'worker_result_path':str(out/'replay_result.json'),
            'worker_status':(worker_result or {}).get('status'),
            'original_failure_reproduced':(worker_result or {}).get('original_failure_reproduced',False),
            'max_wall_s':args.max_wall_s,'native_hang_kill_after_s':args.max_wall_s+15,
            'threshold_unchanged':1e-10,'output_dir':str(out)}
        save(out/'audit_summary.json', summary)
        print(json.dumps(clean(summary),ensure_ascii=False,allow_nan=False),flush=True)
    raise SystemExit(0 if worker_code == 0 and not changed and parent_exception is None else 1)


if __name__ == '__main__':
    main()
