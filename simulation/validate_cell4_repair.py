"""Linux-only bounded migration/resume validation of the trusted mesh4 cell4.

Runs five completed cycles, preserves the rollover checkpoint, resumes for twelve
more, then admits production only from hashes and actual committed observations.
This validator never creates or modifies an original physical checkpoint.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import until80_policy as P

EXPECTED_FAILED_EFC = 1614.938734766
EXPECTED_FAILED_T = 16794912.919245493


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='/var/lib/battery-capstone/results/lifetime_cell4_3p2A_20261028_v2/trajectory/checkpoint.pkl')
    p.add_argument('--source-archive', default='/var/lib/battery-capstone/backups/cell4_3p2A_v2_original_20261029.tar.gz')
    p.add_argument('--output-dir', required=True, help='New audit directory; its branch/ subdirectory holds runner output')
    p.add_argument('--deadline-utc', default='2026-11-02T14:49:00Z')
    p.add_argument('--validation-wall-s', type=float, default=1800.)
    return p.parse_args()


def manifest(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': P.digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def y_sha(solution):
    import numpy as np
    import battery as B
    y = np.asarray(B.y_end(solution), dtype='<f8')
    return hashlib.sha256(y.tobytes()).hexdigest()


def load_checked(branch):
    import run_100efc as R
    summary = P.read(branch/'summary.json')
    P.validate_checkpoint_report(branch, summary)
    state = R.load_checkpoint(branch/'checkpoint.pkl')
    if (P.identity_sha(state['identity']) != summary['identity_sha256']
            or state['commit_index'] != summary['checkpoint_commit_id']
            or state['journal_head_sha256'] != summary['history_commit_id']
            or state['ledger']['efc'] != summary['last_valid_EFC']):
        raise ValueError('Loaded checkpoint does not match committed report')
    return summary, state


def run_phase(command, out, label, end_wall):
    """The runner has its own supervised shutdown; add a bounded outer wait."""
    start = time.monotonic()
    with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=max(1., end_wall-start))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                code = proc.wait(timeout=70.)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                code = 124
    receipt = {'command': command, 'exit_code': code, 'wall_s': time.monotonic()-start,
               'log_sha256': P.digest(out/(label+'.log'))}
    P.save(out/(label+'.json'), receipt)
    if code != 75:
        raise RuntimeError(label+' did not produce a healthy rollover (exit '+str(code)+')')
    return receipt


def validate_history(branch, report, source_ledger):
    """Validate every linked commit and every referenced RPT/parquet artifact."""
    import pyarrow.parquet as pq
    import numpy as np
    previous_hash = None
    previous_ledger = source_ledger
    entries, rpts, transitions, exceedances = [], [], [], []
    last_row = None
    q0 = source_ledger['q0_Ah']
    for index in range(report['checkpoint_commit_id']+1):
        path = branch/'segments'/f'segment_{index:08d}.json'
        e = P.read(path)
        if (e['index'] != index or e['previous_journal_sha256'] != previous_hash
                or e['identity_sha256'] != report['identity_sha256']
                or e['source_checkpoint_sha256'] != P.SOURCE_CHECKPOINT_SHA256):
            raise ValueError('Full journal hash/provenance chain failed at '+str(index))
        ledger = e['ledger']
        if ledger['q0_Ah'] != q0:
            raise ValueError('Q0 changed')
        if index == 0 and ledger != source_ledger:
            raise ValueError('Initial migration ledger differs from source')
        for key in ('efc', 't_abs_s', 'discharge_Ah', 'charge_Ah', 'pieces', 'cycle'):
            if ledger[key] < previous_ledger[key]-1e-10:
                raise ValueError('Nonmonotonic ledger '+key)
        if abs(ledger['efc']-ledger['discharge_Ah']/q0)>1e-8:
            raise ValueError('EFC/charge ledger identity failed')
        checks = e.get('checks') or {}
        if checks:
            if checks.get('connection_definition') != 'physical-initial-state-v2':
                raise ValueError('Wrong connection definition')
            for key, limit in (('physical_join_max_rel', 1e-10),
                               ('time_join_max_error_s', 1e-6),
                               ('native_first_output_error_ratio', 1.)):
                value = checks.get(key)
                if value is None or not math.isfinite(value) or value > limit:
                    raise ValueError('Numerical check failed '+key)
            if checks.get('nonfinite_pieces') != 0:
                raise ValueError('Nonfinite piece recorded')
            exceedances.extend({**item, 'segment_index': index}
                               for item in checks.get('legacy_gate_exceedances', []))
        for a in [e.get('parquet'), *e.get('rpts', [])]:
            if not a:
                continue
            artifact = P.safe_file(branch, a['path'])
            if (not artifact.is_file() or artifact.stat().st_size != a['bytes']
                    or P.digest(artifact) != a['sha256']):
                raise ValueError('Artifact hash/size failed: '+str(artifact))
        for a in e.get('rpts', []):
            point = P.read(P.safe_file(branch, a['path']))
            rpts.append({**point, 'segment_index': index, 'path': a['path']})
            if point.get('origin') == 'copied_parent_RPT':
                continue
            ref = point['reference']; q = point['Qref_Ah']
            if (not ref['ok'] or not math.isfinite(q) or q <= 0
                    or abs(ref['Q_cum_Ah']-q) > 1e-12
                    or abs(ref['Q_trapz_Ah']-q)/q > 1e-4
                    or abs(point['SOH_percent']-100*q/q0) > 1e-10
                    or point.get('state_max_abs_change') != 0.
                    or point.get('ledger_unchanged') is not True):
                raise ValueError('RPT capacity/state preservation failed')
            raw = point['raw_observations']
            raw_path = P.safe_file(branch, raw['path'])
            if (not raw_path.is_file() or raw_path.stat().st_size != raw['bytes']
                    or P.digest(raw_path) != raw['sha256']):
                raise ValueError('RPT raw observation hash/size failed')
            with np.load(raw_path, allow_pickle=False) as observed:
                for phase in ('cc_charge','cv_hold','rest','discharge'):
                    vectors = [observed[phase+'_'+name] for name in
                               ('time_s','current_A','voltage_V','discharge_capacity_Ah')]
                    if (any(v.ndim != 1 or v.size < 2 or not np.all(np.isfinite(v)) for v in vectors)
                            or any(v.shape != vectors[0].shape for v in vectors)
                            or np.any(np.diff(vectors[0]) < 0)):
                        raise ValueError('Invalid RPT raw phase observations')
                t = observed['discharge_time_s']; current = observed['discharge_current_A']
                capacity = observed['discharge_discharge_capacity_Ah']
                raw_integral = float(np.trapezoid(current, t)/3600.)
                raw_capacity = float(capacity[-1]-capacity[0])
                if abs(raw_capacity-q)>1e-10 or abs(raw_integral-q)/q>1e-4:
                    raise ValueError('RPT raw integration/capacity disagrees with reported Qref')
        if e.get('parquet'):
            a = e['parquet']
            table = pq.read_table(P.safe_file(branch, a['path']),
                                  columns=['phase','time_s','current_A','efc','cycle'])
            if table.num_rows != a['rows']:
                raise ValueError('Parquet row count differs')
            data = table.to_pydict()
            if any(not np.all(np.isfinite(data[k])) for k in ('time_s','current_A','efc')):
                raise ValueError('Nonfinite stored observations')
            for n in range(table.num_rows):
                row = {k: data[k][n] for k in data}
                if last_row is not None and last_row['phase'] != row['phase']:
                    transitions.append({'previous_kind': last_row['phase'],
                                        'kind': row['phase'], 'segment_index': index, **row})
                last_row = row
        entries.append(e)
        previous_hash, previous_ledger = P.digest(path), ledger
    if previous_hash != report['history_commit_id']:
        raise ValueError('Final chain tip differs')
    return entries, rpts, transitions, exceedances


def main():
    args = parse()
    if not sys.platform.startswith('linux'):
        raise SystemExit('Original checkpoint validation is Linux-only; no pickle was loaded')
    if not (0 < args.validation_wall_s <= 1800):
        raise SystemExit('Validation wall bound must be <=1800s')
    hard_deadline = min(P.deadline(args.deadline_utc), P.deadline('2026-11-02T14:49:00Z'))
    if hard_deadline <= dt.datetime.now(dt.timezone.utc):
        raise SystemExit('Validation deadline has elapsed')
    out = Path(args.output_dir).resolve()
    source = Path(args.checkpoint).resolve(strict=True)
    archive = Path(args.source_archive).resolve(strict=True)
    source_root = source.parent.parent
    if out == source_root or source_root in out.parents or out in source_root.parents:
        raise SystemExit('Validation outputs must be disjoint from source')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Validation requires a new empty output directory')
    if P.digest(source) != P.SOURCE_CHECKPOINT_SHA256 or P.digest(archive) != P.SOURCE_ARCHIVE_SHA256:
        raise SystemExit('Trusted source hashes differ')
    out.mkdir(parents=True, exist_ok=True)
    before = manifest(source_root)
    P.save(out/'source_manifest_before.json', before)
    began = time.monotonic()
    receipt = {'passed': False, 'mesh_scale': 4, 'test_scope': False,
               'source_checkpoint_sha256': P.SOURCE_CHECKPOINT_SHA256,
               'source_archive_sha256': P.SOURCE_ARCHIVE_SHA256,
               'validator_sha256': P.digest(Path(__file__)),
               'deadline_utc': hard_deadline.isoformat(), 'phase_receipts': []}
    try:
        import run_100efc as R
        import battery as B
        original = R.load_checkpoint(source)
        source_ledger = original['ledger']
        original_y_sha = y_sha(original['solution'])
        prior = float(original['elapsed_wall_s'])
        for name in ('summary.json','outcome.json','FAILED.json'):
            if (source.parent/name).is_file():
                obj = P.read(source.parent/name)
                prior = max(prior, float(obj.get('elapsed_wall_s', 0.)))
        del original
        branch = out/'branch'
        budget = prior+args.validation_wall_s
        common = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).with_name('run_until80.py')),
                  '--output-dir', str(branch), '--cell','4','--current','3.2','--mesh-scale','4',
                  '--tolerance-factor','1.0','--max-efc','1000000',
                  '--max-wall-s',str(budget),'--worker-wall-s',str(args.validation_wall_s),
                  '--deadline-utc',hard_deadline.isoformat(), '--source-archive',str(archive)]
        end_wall = began+min(args.validation_wall_s, (hard_deadline-dt.datetime.now(dt.timezone.utc)).total_seconds())
        receipt['phase_receipts'].append(run_phase(common+['--migrate-checkpoint',str(source),'--smoke-segments','5'],out,'phase1_migration',end_wall))
        first_report, first_state = load_checked(branch)
        first_tip = {'checkpoint_sha256': P.digest(branch/'checkpoint.pkl'),
                     'physical_y_sha256': y_sha(first_state['solution']),
                     'ledger': first_state['ledger'], 'cursor': first_state['cursor'],
                     'commit_index': first_state['commit_index'],
                     'journal_head_sha256': first_state['journal_head_sha256']}
        P.save(out/'phase1_checkpoint_tip.json', first_tip)
        P.save(out/'phase1_summary.json', first_report)
        shutil.copy2(branch/'checkpoint.pkl', out/'phase1_checkpoint.pkl')
        del first_state
        receipt['phase_receipts'].append(run_phase(common+['--resume','--smoke-segments','12'],out,'phase2_resume',end_wall))
        report, final = load_checked(branch)
        entries, rpts, transitions, exceedances = validate_history(branch, report, source_ledger)
        fresh_rpts = [r for r in rpts if r.get('origin') == 'until80_RPT' and r['efc'] >= 1620.-1e-8]
        failed_transition_rows = [x for x in transitions if x['previous_kind']=='DIS' and x['kind']=='REST'
            and abs(x['efc']-EXPECTED_FAILED_EFC)<1e-4 and abs(x['time_s']-EXPECTED_FAILED_T)<.1
            and abs(x['current_A'])<1e-12]
        detailed_joins = [x for x in exceedances if x.get('phase')=='REST_AFTER_DISCHARGE'
            and x.get('previous_kind')=='DIS' and x.get('kind')=='REST'
            and abs(x.get('efc', -1)-EXPECTED_FAILED_EFC)<1e-4
            and x.get('legacy_output_jump', 0)>1e-10
            and 0 <= x.get('physical_initial_jump', math.inf)<=1e-10]
        post_rpt = bool(fresh_rpts) and any(x['previous_kind']=='DIS' and x['kind']=='REST'
            and x['segment_index']>min(r['segment_index'] for r in fresh_rpts)
            and x['efc']>=1620.-1e-8 for x in transitions)
        migration = P.read(branch/'migration_state_receipt.json')
        resumed = P.read(branch/'resume_state_receipt.json')
        checks = {
            'actual_mesh4_baseline_solver': final['identity']['mesh_scale']==4 and final['identity']['numerical_spec']['factor']==1. and not final['identity']['test_scope'],
            'initial_state_migration_equal': migration.get('source_y_sha256')==original_y_sha and migration.get('migrated_y_sha256')==original_y_sha and migration.get('physical_y_bitwise_equal') is True and migration.get('ledger_unchanged') is True and migration.get('cursor_unchanged') is True and migration.get('source_identity_verified') is True,
            'resume_loaded_exact_rollover': resumed.get('checkpoint_sha256')==first_tip['checkpoint_sha256'] and resumed.get('physical_y_sha256')==first_tip['physical_y_sha256'] and resumed.get('ledger')==first_tip['ledger'] and resumed.get('cursor')==first_tip['cursor'] and resumed.get('commit_index')==first_tip['commit_index'] and resumed.get('journal_head_sha256')==first_tip['journal_head_sha256'] and resumed.get('identity_verified') is True and resumed.get('before_first_advance') is True,
            'healthy_final_rollover': final['status']==P.ROLLOVER and not final['terminal'] and not (branch/'FAILED.json').exists(),
            'reached_1620': report['last_valid_EFC']>=1620.,
            'failed_transition_passed': bool(failed_transition_rows) and bool(detailed_joins),
            'rpt_1620_valid': bool(fresh_rpts),
            'post_rpt_transition_passed': bool(post_rpt),
            'fixed_q0': final['ledger']['q0_Ah']==source_ledger['q0_Ah'],
        }
        receipt.update(checks=checks, reached_EFC=report['last_valid_EFC'],
                       last_SOH_percent=report['last_SOH_percent'],
                       last_RPT_EFC=report['last_SOH_measurement_EFC'],
                       failed_transition_passed=checks['failed_transition_passed'],
                       post_rpt_transition_passed=checks['post_rpt_transition_passed'],
                       failed_transition_rows=failed_transition_rows, detailed_joins=detailed_joins,
                       journal_commits=len(entries), validated_RPTs=len(fresh_rpts),
                       final_checkpoint_sha256=report['current_checkpoint_sha256'],
                       history_commit_id=report['history_commit_id'], branch=str(branch))
        receipt['passed'] = all(checks.values())
    except BaseException as exc:
        receipt.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        after = manifest(source_root)
        P.save(out/'source_manifest_after.json', after)
        receipt.update(source_unchanged=(before==after), elapsed_wall_s=time.monotonic()-began,
                       finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        receipt['passed'] = receipt['passed'] and receipt['source_unchanged']
        P.save(out/'admission_receipt.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if receipt['passed'] else 1)


if __name__ == '__main__':
    main()
