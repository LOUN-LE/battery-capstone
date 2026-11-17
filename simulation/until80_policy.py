"""Pure limits, provenance and transactional-file rules for until80 branches."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path

FORMAT = 'until80-compact-v2-initial-state'
LEGACY_OPERATION_SHA256 = 'ee2c46cbd6ee45f174be832642750b04be5db2d53a1b0c94d884f439721ca1ab'
MAX_WALL_S = 336 * 3600.
MAX_WORKER_S = 20 * 3600.
MAX_EFC = 1e6
SOURCE_CHECKPOINT_SHA256 = '293383f8d13a4c262abf8e2e42d7538d03028fc59cf40601bceec5083afa8d64'
SOURCE_ARCHIVE_SHA256 = '9f4ea2442b778ea4ee5f4438d78bdd1438dda457b477b0fdb85030a195c1202b'
ROLLOVER = 'rollover_checkpointed'
CENSORED = {'global_time_censored', 'deadline_censored', 'interrupted_censored'}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            sha.update(block)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def identity_sha(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def deadline(value):
    if value is None:
        return None
    date = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.tzinfo is None or date.utcoffset() is None:
        raise ValueError('Deadline must contain an explicit UTC offset')
    return date.astimezone(dt.timezone.utc)


def validate_limits(args):
    if not (math.isfinite(args.max_wall_s) and 0 < args.max_wall_s <= MAX_WALL_S):
        raise ValueError('Cumulative budget must be positive and <=336h')
    if not (math.isfinite(args.worker_wall_s) and 0 < args.worker_wall_s <= MAX_WORKER_S):
        raise ValueError('Each worker must be positive and <=20h')
    if not (math.isfinite(args.max_efc) and 0 < args.max_efc <= MAX_EFC):
        raise ValueError('Finite EFC ceiling <=1e6 required')
    if args.tolerance_factor not in (1., .1, .01):
        raise ValueError('Only baseline or independently tested tighter solver settings are supported')
    if args.smoke_segments < 0 or args.test_failure_after_pieces < 0 or args.test_rollover_after_pieces < 0:
        raise ValueError('Nonnegative verification controls required')
    if args.test_scope:
        if args.mesh_scale != 1 or args.max_efc > 2 or args.max_wall_s > 600 or args.worker_wall_s > 600:
            raise ValueError('Local test scope is limited to mesh1,2EFC,600s')
        if not (.01 <= args.rpt_every_efc <= 10):
            raise ValueError('Invalid test diagnostic interval')
    else:
        if args.fresh_test_only or args.test_failure_after_pieces or args.test_rollover_after_pieces:
            raise ValueError('Synthetic/fresh modes are unavailable for production')
        if args.rpt_every_efc != 10:
            raise ValueError('Production SOH interval is fixed at10EFC')
        if deadline(args.deadline_utc) is None:
            raise ValueError('Production requires an absolute UTC deadline')
    if not (0 < args.max_memory_gib <= 24 and args.min_free_memory_gib > 0 and args.min_free_disk_gib >= 1):
        raise ValueError('Explicit bounded RAM/disk limits required')
    deadline(args.deadline_utc)


def upcoming_diagnostics(efc, ceiling, interval=10.):
    """A bounded forward window, not a growing all-history schedule."""
    first = (math.floor((efc + 1e-8) / interval) + 1) * interval
    count = max(3, math.ceil(2. / interval) + 1)
    return [first + i*interval for i in range(count) if first + i*interval <= ceiling+1e-8]


def eol_bracket(previous, current, threshold=80.):
    if current is None or current['SOH_percent'] >= threshold:
        return None
    return {'threshold_percent': threshold, 'observed_at_EFC': current['efc'],
        'interval_lower_EFC': previous['efc'] if previous else None,
        'interval_upper_EFC': current['efc'], 'exact_crossing_known': False,
        'rul_from_100_observation_EFC': current['efc']-100 if current['efc'] >= 100 else None}


def validate_control(path, factor, source_sha):
    if factor == 1:
        return None
    if not path:
        raise ValueError('Tighter solver requires a successful independent numerical-control report')
    report = read(path)
    control = report.get('solver_precision_control') or {}
    provenance = report.get('trusted_source_provenance') or {}
    transitions = report.get('event_time_correspondence') or []
    if not (report.get('audit_kind') == 'solver_precision_control'
            and report.get('worker_exit_code') == 0
            and report.get('worker_status') == 'not_reproduced_within_cycle_bound'
            and report.get('original_files_unchanged') is True
            and report.get('original_failure_reproduced') is False
            and report.get('tolerance_factor') == factor
            and control.get('factor') == factor
            and control.get('connection_gate') == 1e-10
            and control.get('physics_changed') is False
            and control.get('connection_gate_changed') is False
            and control.get('simulation_count_when_applied') == 0
            and provenance.get('checkpoint_sha256') == source_sha
            and provenance.get('archive_sha256') == SOURCE_ARCHIVE_SHA256
            and any(t.get('same_baseline_cycle') is True
                    and t.get('phase') == 'REST_AFTER_DISCHARGE'
                    and t.get('previous_kind') == 'DIS' and t.get('kind') == 'REST'
                    for t in transitions)):
        raise ValueError('Numerical-control evidence does not validate this checkpoint/factor')
    return {'sha256': digest(path), 'factor': factor,
            'source_checkpoint_sha256': source_sha,
            'note': 'Numerical control only; not real-cell aging validation'}


def safe_file(root, relative):
    rel = Path(relative)
    target = (root/rel).resolve()
    if rel.is_absolute() or target == root or root not in target.parents:
        raise ValueError('Artifact path escapes branch')
    return target


def validate_checkpoint_report(root, report):
    """No pickle import: coordinator verifies the atomic checkpoint and commit tip."""
    root = Path(root).resolve()
    if report.get('format') != FORMAT:
        raise ValueError('Not an until80 compact result')
    cp = root/'checkpoint.pkl'
    if not cp.is_file() or digest(cp) != report.get('current_checkpoint_sha256'):
        raise ValueError('Checkpoint hash does not match result')
    index = report.get('checkpoint_commit_id')
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('Invalid commit index')
    journal = root/'segments'/f'segment_{index:08d}.json'
    if not journal.is_file() or digest(journal) != report.get('history_commit_id'):
        raise ValueError('Journal head does not match checkpoint report')
    entry = read(journal)
    if (entry.get('index') != index or entry.get('identity_sha256') != report.get('identity_sha256')
            or abs(entry['ledger']['efc']-report['last_valid_EFC']) > 1e-8
            or entry.get('status') != report.get('status')
            or entry.get('source_checkpoint_sha256') != report.get('source_checkpoint_sha256')):
        raise ValueError('Committed journal and checkpoint report disagree')
    for artifact in [entry.get('parquet'), *entry.get('rpts', [])]:
        if artifact:
            p = safe_file(root, artifact['path'])
            if not p.is_file() or p.stat().st_size != artifact['bytes'] or digest(p) != artifact['sha256']:
                raise ValueError('Committed data artifact is missing or changed')
            if p.suffix=='.json' and p.parent.name=='rpt':
                point=read(p)
                raw=point.get('raw_observations')
                if point.get('origin')=='until80_RPT' and not raw:
                    raise ValueError('Committed RPT raw observations missing')
                if raw and point.get('origin')!='copied_parent_RPT':
                    target=safe_file(root,raw['path'])
                    if not target.is_file() or target.stat().st_size!=raw['bytes'] or digest(target)!=raw['sha256']:
                        raise ValueError('Committed RPT raw observations changed')
    return report


def quarantine_uncommitted(root, state):
    """Preserve only this branch's uncommitted files; never unlink source data."""
    root = Path(root).resolve()
    candidates = []
    committed = state['commit_index']
    rpts = state['rpt_count']
    for directory, prefix, limit in ((root/'segments','segment_',committed),(root/'rpt','rpt_',rpts)):
        for p in directory.iterdir():
            if not p.is_file() or not p.name.startswith(prefix):
                continue
            match=re.fullmatch(re.escape(prefix)+r'(\d{8})(?:_raw)?\.(?:json|npz|parquet)(?:\.partial|\.tmp)?',p.name)
            if not match:raise ValueError('Unknown artifact filename; inspect before resume')
            index=int(match.group(1))
            if index > limit or p.name.endswith(('.partial','.tmp')):
                candidates.append(p)
    temp = root/'checkpoint.pkl.tmp'
    if temp.is_file(): candidates.append(temp)
    if not candidates:
        return []
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = root/'quarantine'/stamp
    target.mkdir(parents=True, exist_ok=False)
    moved = []
    for p in candidates:
        relative = p.relative_to(root)
        q = target/relative
        q.parent.mkdir(parents=True, exist_ok=True)
        p.replace(q)
        moved.append({'original':relative.as_posix(),'preserved_at':q.relative_to(root).as_posix()})
    save(target/'manifest.json',{'reason':'Uncommitted artifacts preserved without deletion','files':moved})
    return moved
