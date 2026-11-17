"""Bounded local aging/RPT convergence experiment; never resumes old checkpoints.

Each mesh starts afresh. Aging tolerance is varied independently of diagnostic
tolerance. All runs stop at the SAME discharged Ah (finest-grid initial Q0 times
target EFC); diagnostic Q0 is measured on that mesh at the same precision as Qref.
The default aging stage uses one fixed RPT precision and cannot pass a numerical
screen. The full stage adds the diagnostic tolerance sweep. Execution success
is distinct from a numerical screening pass and cell validity.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [{'variant': f'm{mesh}_f{factor:g}', 'mesh': mesh, 'aging_factor': factor}
            for mesh in (4, 8) for factor in (1., .1, .01)]
RPT_LEVELS = ((1e-6, 1e-8), (1e-7, 1e-9), (1e-8, 1e-10))
CHILDREN = set()
CHILDREN_LOCK = threading.Lock()
CANCEL = threading.Event()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sources():
    paths = ['simulation/audit_short_convergence.py', 'simulation/rebuilt_lifetime.py',
             'simulation/convergence_contracts.py', 'configs/model.json',
             'configs/diagnostics.json', 'configs/cells_pilot9.json']
    return {name: digest(ROOT/name) for name in paths}


def snapshot_sources(out, expected):
    snapshot = out/'source_snapshot'
    for relative, expected_hash in expected.items():
        source, destination = ROOT/relative, snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if digest(destination) != expected_hash:
            raise ValueError('Source changed while taking snapshot: '+relative)
    save(snapshot/'manifest.json', expected)


def reuse_anchor(source_dir, out, current_sources):
    source_dir = Path(source_dir).resolve()
    parent_path, anchor_path, raw_path = (source_dir/name for name in
                                         ('report.json', 'anchor.json', 'anchor_rpt.npz'))
    parent, anchor = read(parent_path), read(anchor_path)
    for name in ('simulation/rebuilt_lifetime.py', 'configs/model.json',
                 'configs/diagnostics.json', 'configs/cells_pilot9.json'):
        if parent.get('source_sha256', {}).get(name) != current_sources[name]:
            raise ValueError('Anchor engine/configuration provenance mismatch: '+name)
    if parent.get('python') != platform.python_version() or parent.get('platform') != platform.platform():
        raise ValueError('Anchor environment mismatch')
    for name in ('pybamm', 'pybammsolvers', 'numpy', 'casadi'):
        if parent.get('versions', {}).get(name) != importlib.metadata.version(name):
            raise ValueError('Anchor package mismatch: '+name)
    q0 = anchor.get('Q0_anchor_Ah')
    if not isinstance(q0, (int, float)) or isinstance(q0, bool) or not math.isfinite(q0) or q0 <= 0:
        raise ValueError('Invalid anchor capacity')
    if (anchor.get('mesh'), anchor.get('rtol'), anchor.get('atol')) != (8, 1e-8, 1e-10):
        raise ValueError('Anchor must use mesh8 and the declared tight precision')
    if anchor.get('reference', {}).get('main_unchanged') is not True or not anchor.get('initial_fingerprint'):
        raise ValueError('Anchor lacks initial-state preservation evidence')
    if anchor.get('reference', {}).get('Qref_Ah') != q0:
        raise ValueError('Anchor/reference capacity mismatch')
    if digest(raw_path) != anchor.get('raw_sha256'):
        raise ValueError('Anchor raw digest mismatch')
    if parent.get('anchor') != anchor:
        raise ValueError('Parent report does not attest this anchor')
    (out/'anchor_rpt.npz').write_bytes(raw_path.read_bytes())
    save(out/'anchor.json', anchor)
    return {'returncode': 0, 'reused': True, 'source_directory': str(source_dir),
            'source_parent_sha256': digest(parent_path), 'source_anchor_sha256': digest(anchor_path),
            'raw_sha256': anchor['raw_sha256'], 'elapsed_s': 0.}


def anchor_worker(out):
    print('anchor: importing numerical engine', flush=True)
    import rebuilt_lifetime as N
    import numpy as np
    engine = N.Engine(mesh=8, rtol=1e-8, atol=1e-10)
    initial = engine.start()
    print('anchor: measuring mesh8 initial reference capacity', flush=True)
    rpt, raw = engine.reference(initial)
    np.savez_compressed(out/'anchor_rpt.npz', **raw)
    save(out/'anchor.json', {'Q0_anchor_Ah': rpt['Qref_Ah'], 'mesh': 8,
         'rtol': 1e-8, 'atol': 1e-10, 'initial_fingerprint': N.fingerprint(initial),
         'reference': rpt, 'raw_sha256': digest(out/'anchor_rpt.npz')})


def case_worker(args, out):
    print('case: importing numerical engine', flush=True)
    import casadi
    import numpy as np
    import pickle
    from types import SimpleNamespace
    import rebuilt_lifetime as N
    before_sources = sources()
    variant = next(v for v in VARIANTS if v['variant'] == args.variant)
    mesh, factor = variant['mesh'], variant['aging_factor']
    anchor = read(out/'anchor.json')['Q0_anchor_Ah']
    case_dir = out/f"{args.variant}_I{args.current:.1f}"
    case_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    row = {**variant, 'current_A': args.current, 'target_EFC': args.efc,
           'Q0_anchor_Ah': anchor, 'throughput_target_Ah': anchor*args.efc,
           'stage': args.stage, 'measurements': [], 'execution_valid': False,
           'RPT_convergence_established': False}
    try:
        measurement_levels = RPT_LEVELS[-1:] if args.stage == 'aging' else RPT_LEVELS
        stored_rtol, stored_atol = measurement_levels[-1]
        print(f'case: starting trajectory with RPT rtol={stored_rtol:g}', flush=True)
        result = N.run(SimpleNamespace(output=str(case_dir/'trajectory'), resume=False,
            family='SPMe', physics='baseline', mesh=mesh, current=args.current,
            rtol=1e-4*factor, atol=1e-6*factor,
            initial_rtol=1e-8, initial_atol=1e-10, rpt_rtol=stored_rtol, rpt_atol=stored_atol,
            efc_reference_ah=anchor, max_efc=args.efc, wall_s=args.case_timeout_s*.8,
            boundary_s=60., rpt_every=10., max_pieces=0))
        if result['status'] != 'max_efc_censored' or abs(result['efc']-args.efc) > 1e-8:
            raise ValueError('Trajectory did not reach requested endpoint: '+result['status'])
        if abs(result['rpts'][-1]['efc']-args.efc) > 1e-8:
            raise ValueError('Endpoint RPT missing')
        print('case: trajectory and endpoint RPT completed; checking checkpoint', flush=True)
        # Read only the just-written, hashed local checkpoint from this case.
        commit = read(case_dir/'trajectory/commit.json')
        path = case_dir/'trajectory'/commit['checkpoint']
        if N.sha(path) != commit['sha256'] or commit['identity'] != result['identity']:
            raise ValueError('New local checkpoint identity mismatch')
        with casadi.global_unpickle_context(), path.open('rb') as handle:
            checkpoint = pickle.load(handle)
        if checkpoint['identity'] != result['identity']:
            raise ValueError('Checkpoint content identity mismatch')
        main = checkpoint['solution']
        initial = N.Engine(mesh=mesh, rtol=1e-8, atol=1e-10).start()
        if N.fingerprint(initial) != result['initial_state_fingerprint']:
            raise ValueError('Independent initialization is not reproducible')
        before_main, before_initial = N.fingerprint(main), N.fingerprint(initial)
        row.update(achieved_EFC=result['efc'], discharge_Ah=result['discharge_Ah'],
                   initial_fingerprint=before_initial, endpoint_fingerprint=before_main,
                   physical_time_s=N.scalar(main, 'Time [s]'),
                   initial_physics=N.Engine(mesh=mesh).readout(initial),
                   final_physics=result['latest_physics'],
                   conventional_EFC=result['discharge_Ah']/result['Q0_Ah'],
                   identity=result['identity'], charge_Ah=result['charge_Ah'],
                   cycles=result['cycles'], pieces=result['pieces'])
        for rtol, atol in measurement_levels:
            reused = (rtol, atol) == (stored_rtol, stored_atol)
            if reused:
                print(f'case: reusing existing Q0/endpoint RPT at rtol={rtol:g}', flush=True)
                q0, qr = result['rpts'][0], result['rpts'][-1]
                raw0_path = case_dir/'trajectory/raw/rpt_000000.npz'
                rawr_path = case_dir/'trajectory/raw'/qr['raw']
                if rawr_path.parent.resolve() != (case_dir/'trajectory/raw').resolve():
                    raise ValueError('Endpoint raw path escapes trajectory')
                if digest(rawr_path) != qr['sha256']:
                    raise ValueError('Endpoint RPT raw digest mismatch')
                with np.load(raw0_path, allow_pickle=False) as archive:
                    raw0 = {key: archive[key] for key in archive.files}
                with np.load(rawr_path, allow_pickle=False) as archive:
                    rawr = {key: archive[key] for key in archive.files}
            else:
                print(f'case: measuring independent Q0/Qref at rtol={rtol:g}', flush=True)
                diag = N.Engine(mesh=mesh, rtol=rtol, atol=atol)
                q0, raw0 = diag.reference(initial)
                qr, rawr = diag.reference(main)
            if N.fingerprint(main) != before_main or N.fingerprint(initial) != before_initial:
                raise ValueError('Diagnostic mutated a frozen state')
            name = f'rpt_{rtol:g}.npz'
            np.savez_compressed(case_dir/name,
                **{f'Q0_{k}': v for k, v in raw0.items()},
                **{f'Qref_{k}': v for k, v in rawr.items()})
            row['measurements'].append({'rtol': rtol, 'atol': atol,
                'Q0_Ah': q0['Qref_Ah'], 'Qref_Ah': qr['Qref_Ah'],
                'SOH_percent': 100*qr['Qref_Ah']/q0['Qref_Ah'],
                'main_unchanged': True, 'initial_unchanged': True,
                'Q0_protocol': q0, 'Qref_protocol': qr,
                'reused_trajectory_RPT': reused,
                'raw': name, 'raw_sha256': digest(case_dir/name)})
            save(case_dir/'result.json', row)
        if sources() != before_sources:
            raise ValueError('Source changed during case')
        row['source_sha256'] = before_sources
        row['execution_valid'] = True
        print('case: validation completed', flush=True)
    except BaseException as exc:
        row['error'] = repr(exc)
        if isinstance(exc, N.StepAuditError):
            row['failed_step'] = exc.details
        raise
    finally:
        row['elapsed_s'] = time.monotonic()-started
        save(case_dir/'result.json', row)


def run_child(command, log_path, timeout_s):
    started = time.monotonic()
    if CANCEL.is_set() or timeout_s <= 0:
        return {'returncode': None, 'cancelled': True, 'elapsed_s': 0.}
    with log_path.open('w', encoding='utf-8') as log:
        # Keep process handles so Ctrl+C/global cleanup also stops concurrent
        # workers, rather than merely cancelling their Python Future objects.
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        with CHILDREN_LOCK:
            CHILDREN.add(proc)
        try:
            if CANCEL.is_set():
                proc.kill()
            returncode = proc.wait(timeout=max(0., timeout_s-(time.monotonic()-started)))
            return {'returncode': returncode, 'elapsed_s': time.monotonic()-started,
                    'pid': proc.pid, 'cancelled': CANCEL.is_set()}
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            return {'returncode': None, 'timeout': True, 'elapsed_s': time.monotonic()-started,
                    'pid': proc.pid, 'worker_terminated': proc.poll() is not None}
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            with CHILDREN_LOCK:
                CHILDREN.discard(proc)


def stop_children():
    CANCEL.set()
    with CHILDREN_LOCK:
        running = list(CHILDREN)
    errors = []
    for proc in running:
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append({'pid': proc.pid, 'error': repr(exc)})
    return errors


def assess_aging(rows, target_efc, anchor, diagnostic_level=RPT_LEVELS[0]):
    """Describe valid paired effects; one diagnostic precision cannot pass convergence."""
    expected = {(v['variant'], current) for v in VARIANTS for current in (3., 3.3)}
    indexed, invalid = {}, []
    valid = {}
    specs = {v['variant']: v for v in VARIANTS}
    def number(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)
    for row in rows:
        key = (row.get('variant'), row.get('current_A'))
        error = None
        if key not in expected:
            invalid.append({'variant': key[0], 'current_A': key[1],
                            'error': 'Unexpected variant/current'})
            continue
        if key in indexed:
            # Neither copy of an ambiguous key may contribute to a pair.
            valid.pop(key, None)
            invalid.append({'variant': key[0], 'current_A': key[1],
                            'error': 'Duplicate variant/current; all copies invalid'})
            continue
        indexed[key] = row
        spec = specs[key[0]]
        if (row.get('mesh'), row.get('aging_factor')) != (spec['mesh'], spec['aging_factor']):
            error = 'Variant mesh/aging metadata mismatch'
        if row.get('execution_valid') is not True:
            error = row.get('error', 'Worker did not validate execution')
        for field in ('Q0_anchor_Ah', 'discharge_Ah', 'achieved_EFC', 'throughput_target_Ah',
                      'target_EFC'):
            value = row.get(field)
            if not number(value):
                error = error or 'Missing/invalid ledger: '+field
        if error is None and (abs(row['Q0_anchor_Ah']-anchor) > 1e-12 or
                abs(row['target_EFC']-target_efc) > 1e-8 or
                abs(row['achieved_EFC']-target_efc) > 1e-8 or
                abs(row['achieved_EFC']-row['discharge_Ah']/anchor) > 1e-8 or
                abs(row['discharge_Ah']-anchor*target_efc) > 1e-8 or
                abs(row['throughput_target_Ah']-anchor*target_efc) > 1e-8):
            error = 'Endpoint/common-throughput mismatch'
        measurements = row.get('measurements', [])
        if not isinstance(measurements, list) or len(measurements) != 1:
            error = error or 'Aging stage requires exactly one fixed-precision RPT measurement'
        elif error is None:
            measurement = measurements[0]
            if (measurement.get('rtol'), measurement.get('atol')) != diagnostic_level:
                error = 'Unexpected fixed RPT precision'
            for field in ('Q0_Ah', 'Qref_Ah', 'SOH_percent'):
                value = measurement.get(field)
                if not number(value):
                    error = 'Invalid diagnostic value: '+field
            if error is None and abs(measurement['SOH_percent']-
                    100*measurement['Qref_Ah']/measurement['Q0_Ah']) > 1e-9:
                error = 'Inconsistent fixed-Q0 SOH'
            if not (measurement.get('main_unchanged') is True and
                    measurement.get('initial_unchanged') is True):
                error = 'Missing diagnostic state-preservation evidence'
        if error:
            invalid.append({'variant': key[0], 'current_A': key[1], 'error': error})
        else:
            valid[key] = row
    for mesh in (4, 8):
        keys = [key for key, row in valid.items() if row['mesh'] == mesh]
        q0s = [valid[key]['measurements'][0]['Q0_Ah'] for key in keys]
        if q0s and max(q0s)-min(q0s) > 1e-10:
            for key in keys:
                valid.pop(key)
                invalid.append({'variant': key[0], 'current_A': key[1],
                                'error': 'Diagnostic Q0 differs within the same mesh'})
    for name, current in sorted(expected-set(indexed)):
        invalid.append({'variant': name, 'current_A': current, 'error': 'Case missing'})
    pairs = []
    for variant in VARIANTS:
        keys = [(variant['variant'], current) for current in (3., 3.3)]
        if all(key in valid for key in keys):
            low, high = [valid[key]['measurements'][0] for key in keys]
            pairs.append({**variant, 'SOH_3p0_percent': low['SOH_percent'],
                          'SOH_3p3_percent': high['SOH_percent'],
                          'delta_3p3_minus_3p0_pp': high['SOH_percent']-low['SOH_percent'],
                          'rtol': diagnostic_level[0], 'atol': diagnostic_level[1],
                          'descriptive_only': True})
    return {'execution_complete': len(valid) == len(expected) and not invalid,
            'valid_cases': len(valid), 'expected_cases': len(expected), 'invalid_cases': invalid,
            'paired_descriptive_comparisons': pairs, 'numerical_screen_pass': False,
            'RPT_convergence_established': False, 'long_term_convergence_established': False,
            'note': 'Fixed-RPT-precision aging comparison only. Diagnostic tolerance convergence remains untested; no numerical pass or current ranking claim.'}


def controller(args, out):
    from convergence_contracts import assess
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    original = sources()
    snapshot_sources(out, original)
    deadline = began+args.wall_s
    pool, futures = None, {}
    report = {'status': 'running', 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_sha256': original, 'versions': {n: importlib.metadata.version(n) for n in
            ('pybamm', 'pybammsolvers', 'numpy', 'casadi')},
        'platform': platform.platform(), 'python': platform.python_version(),
        'plan': {'stage': args.stage, 'jobs': args.jobs,
            'variants': VARIANTS, 'currents_A': [3., 3.3], 'target_EFC': args.efc,
            'rpt_levels': RPT_LEVELS[-1:] if args.stage == 'aging' else RPT_LEVELS,
            'absolute_screen_pp': args.abs_screen_pp,
            'relative_screen': args.relative_screen, 'case_timeout_s': args.case_timeout_s,
            'total_timeout_s': args.wall_s,
            'EFC_definition': 'Common mesh8 tight-Q0 discharged-Ah reference across all cases',
            'common_physical_throughput': True, 'cross_mesh_state_transfer': False,
            'initial_solver': {'rtol': 1e-8, 'atol': 1e-10},
            'main_solver': {'rtol': '1e-4 * factor', 'atol': '1e-6 * factor'},
            'duration_scope': 'Short numerical screening only; not lifetime/experimental validation'},
        'rows': [], 'execution': [], 'execution_complete': False,
        'numerical_screen_pass': False, 'long_term_convergence_established': False,
        'RPT_convergence_established': False,
        'experimental_validation_complete': False, 'cloud_used': False}
    save(out/'report.json', report)
    command = [sys.executable, str(Path(__file__).resolve()), '--output', str(out),
               '--efc', str(args.efc), '--case-timeout-s', str(args.case_timeout_s),
               '--stage', args.stage]
    try:
        if args.anchor_source:
            anchor_exec = reuse_anchor(args.anchor_source, out, original)
        else:
            anchor_exec = run_child(command+['--worker', 'anchor'], out/'anchor.log',
                                    min(args.case_timeout_s, deadline-time.monotonic()))
        report['execution'].append({'case': 'anchor', **anchor_exec})
        if anchor_exec['returncode'] != 0:
            raise RuntimeError('Anchor measurement failed; see anchor.log')
        report['anchor'] = read(out/'anchor.json')
        save(out/'report.json', report)
        print(json.dumps({'case': 'anchor', **anchor_exec}), flush=True)
        def execute_case(variant, current, label):
            remaining = deadline-time.monotonic()
            if remaining <= 0 or CANCEL.is_set():
                return {'returncode': None, 'deadline_or_cancelled': True, 'elapsed_s': 0.}
            print(json.dumps({'case': label, 'event': 'started', 'remaining_s': remaining}), flush=True)
            return run_child(command+['--worker', 'case', '--variant', variant['variant'],
                              '--current', str(current)], out/(label+'.log'),
                             min(args.case_timeout_s, remaining))
        pool = ThreadPoolExecutor(max_workers=args.jobs)
        for variant in VARIANTS:
            for current in (3., 3.3):
                label = f"{variant['variant']}_I{current:.1f}"
                future = pool.submit(execute_case, variant, current, label)
                futures[future] = (variant, current, label)
        for future in as_completed(futures):
            variant, current, label = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {'returncode': None, 'error': repr(exc)}
            report['execution'].append({'case': label, **receipt})
            result_path = out/label/'result.json'
            row = read(result_path) if result_path.exists() else {
                **variant, 'current_A': current, 'execution_valid': False,
                'error': 'Worker produced no completed result'}
            if receipt.get('returncode') != 0:
                row['execution_valid'] = False
                row['execution_error'] = receipt
                row.setdefault('error', 'Worker failed, timed out, or was not started')
            report['rows'].append(row)
            save(out/'report.json', report)
            print(json.dumps({'case': label, **receipt}), flush=True)
        if args.stage == 'aging':
            report['comparison'] = assess_aging(report['rows'], args.efc,
                                                report['anchor']['Q0_anchor_Ah'], RPT_LEVELS[-1])
        else:
            report['comparison'] = assess(report['rows'], VARIANTS, args.efc,
                                          args.abs_screen_pp, args.relative_screen)
        for mesh in (4, 8):
            fps = {row['initial_fingerprint'] for row in report['rows']
                   if row.get('mesh') == mesh and row.get('execution_valid') is True}
            if len(fps) > 1:
                raise ValueError('Initial conditions differ within mesh '+str(mesh))
        report['source_hashes_unchanged'] = sources() == original
        if not report['source_hashes_unchanged']:
            raise ValueError('Source changed during experiment')
        report.update(execution_complete=report['comparison']['execution_complete'],
                      numerical_screen_pass=report['comparison']['numerical_screen_pass'])
        if args.stage == 'aging':
            report['status'] = ('completed_aging_stage' if report['execution_complete'] else
                                'completed_aging_with_failed_or_missing_cases')
        else:
            report['status'] = ('screen_passed' if report['numerical_screen_pass'] else
                                'completed_screen_not_passed')
    except BaseException as exc:
        report['status'], report['error'] = 'failed_or_incomplete', repr(exc)
        traceback.print_exc()
    finally:
        cleanup_errors = stop_children()
        for future in futures:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        report['worker_cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report.update(execution_complete=False, numerical_screen_pass=False,
                          status='worker_cleanup_failed')
        report['source_hashes_unchanged'] = sources() == original
        if not report['source_hashes_unchanged']:
            report.update(execution_complete=False, numerical_screen_pass=False,
                          status='source_changed')
        report['elapsed_s'] = time.monotonic()-began
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(out/'report.json', report)
    print(json.dumps({k: report.get(k) for k in
          ('status', 'execution_complete', 'numerical_screen_pass', 'elapsed_s', 'error')}), flush=True)
    # A completed but failed screen must not appear green to an orchestration job.
    return 0 if report['numerical_screen_pass'] else (2 if report['execution_complete'] else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--efc', type=float, default=1.)
    p.add_argument('--case-timeout-s', type=float, default=1800.)
    p.add_argument('--wall-s', type=float, default=7200.)
    p.add_argument('--abs-screen-pp', type=float, default=1e-4)
    p.add_argument('--relative-screen', type=float, default=.1)
    p.add_argument('--stage', choices=('aging', 'full'), default='aging')
    p.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    p.add_argument('--anchor-source', help='Verified local prior report directory; reuse its mesh8 tight Q0')
    p.add_argument('--worker', choices=('anchor', 'case'))
    p.add_argument('--variant', choices=[v['variant'] for v in VARIANTS])
    p.add_argument('--current', type=float, choices=(3., 3.3))
    args = p.parse_args()
    if not (0 < args.efc <= 5 and 0 < args.case_timeout_s <= 1800 and 0 < args.wall_s <= 7200
            and math.isfinite(args.abs_screen_pp) and args.abs_screen_pp > 0
            and 0 < args.relative_screen <= 1):
        p.error('Require bounded short experiment (0<EFC<=5), positive finite limits')
    out = Path(args.output).resolve()
    if args.worker == 'anchor':
        anchor_worker(out)
        return 0
    if args.worker == 'case':
        if args.variant is None or args.current is None:
            p.error('Case worker requires variant and current')
        case_worker(args, out)
        return 0
    return controller(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
