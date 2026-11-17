"""Admission-gated, bounded until-80 coordinator. Default/--plan is read-only.

Uses no AWS credential or expiring upload URL. Original data stay read-only;
results remain on EBS. Existing external EC2-stop notifications are unchanged.
336 hours is a budget ceiling, never an SOH-80 arrival estimate. Numerical audit
admission does not establish experimental LG M50 chemistry validity.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

RATE = 0.4032
MAX_WALL_S = 336 * 3600
GIB = 1024**3
TERMINAL = {'eol_observed', 'max_efc_censored', 'global_time_censored',
            'resource_censored', 'worker_failed', 'invalid_worker_contract',
            'interrupted_censored', 'unclean_boot_failed', 'coordinator_failed',
            'source_modified'}
STOP = {'signal': None}


def now():
    return dt.datetime.now(dt.timezone.utc)


def stamp():
    return now().isoformat()


def parse_time(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('UTC deadline/admission time must be timezone-aware')
    return result.astimezone(dt.timezone.utc)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, name, minimum=0):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError('Invalid finite admission number: ' + name)
    return value


def need(condition, message):
    if not condition:
        raise ValueError(message)


def evidence(spec, receipt_dir):
    path = Path(spec['path'])
    path = (receipt_dir/path).resolve() if not path.is_absolute() else path.resolve()
    need(path.is_file(), 'Missing evidence file: ' + str(path))
    need(digest(path) == spec['sha256'], 'Evidence SHA-256 mismatch: ' + str(path))
    return read(path), path


def budget_plan(cost, max_wall_s):
    need(0 < max_wall_s <= MAX_WALL_S, 'Cumulative trajectory cap must be <=336 hours')
    budget = finite(cost['budget_KRW'], 'budget_KRW', 1)
    reserve = finite(cost['reserve_KRW'], 'reserve_KRW')
    fx = finite(cost['planning_USDKRW'], 'planning_USDKRW', 2000)
    buffer = finite(cost['budget_buffer'], 'budget_buffer', 1.10)
    prior = finite(cost['prior_spend_upper_USD'], 'prior_spend_upper_USD')
    observed = finite(cost['observed_billed_USD'], 'observed_billed_USD')
    unbilled = finite(cost['unbilled_estimate_USD'], 'unbilled_estimate_USD')
    need(budget <= 600000 and reserve >= 150000 and reserve < budget,
         'Budget <=600000 KRW and reserve >=150000 KRW required')
    need(finite(cost['rate_USD_h'], 'rate_USD_h') == RATE, 'Reviewed fixed planning rate is 0.4032 USD/h')
    need(observed + unbilled <= prior + 1e-9, 'Observed+unbilled spending exceeds prior allowance')
    reviewed = parse_time(cost['reviewed_utc'])
    need(-300 <= (now()-reviewed).total_seconds() <= 86400, 'Cost admission must be reviewed within 24 hours')
    # Conservatively budget the full cap again, even though source runtime is
    # deducted from available new worker time. Prior spending is not erased.
    compute_upper = RATE * max_wall_s / 3600
    total = reserve + (prior + compute_upper) * fx * buffer
    need(total <= budget, 'Conservative total planning cost exceeds approved KRW budget')
    return {'budget_KRW': budget, 'reserve_KRW': reserve, 'planning_USDKRW': fx,
            'FX_is_live_quote': False, 'budget_buffer': buffer, 'rate_USD_h': RATE,
            'prior_spend_upper_USD': prior, 'compute_upper_USD': compute_upper,
            'planning_total_upper_KRW': total, 'is_AWS_account_hard_spend_cap': False,
            'cost_is_final_bill': False}


def validate_admission(args):
    receipt_path = Path(args.admission_receipt).resolve(strict=True)
    r = read(receipt_path); base = receipt_path.parent
    need(r.get('schema') == 'capstone-until80-admission-v1', 'Unsupported admission schema')
    source = Path(args.source_cp).resolve(strict=True)
    need(r['source_checkpoint_sha256'] == digest(source), 'Admission source checkpoint does not match')
    need(float(r['selected_numerical_factor']) == args.numfactor, 'Selected numerical factor differs')
    refs = r['replay_reports']
    need(len(refs) >= 2, 'Two baseline failure reproductions required')
    for ref in refs:
        data, _ = evidence(ref, base)
        need(data.get('original_failure_reproduced') is True and
             data.get('original_files_unchanged') is True and data.get('worker_exit_code') == 0,
             'Baseline reproduction/source preservation gate failed')
        need(data.get('threshold_unchanged') == 1e-10, 'Baseline connection gate changed')
    aging, _ = evidence(r['aging_report'], base)
    need(aging.get('status') == 'complete' and aging.get('required_readouts_available') is True
         and aging.get('original_source_files_unchanged') is True, 'Aging audit incomplete')
    for name in ('initial', 'legacy100', 'final'):
        state = aging['states'][name]
        reps = state.get('reference_repeats', [])
        need(state.get('source_state_and_ledger_unchanged') is True and len(reps) >= 2,
             'Copied RPT state/repetition gate failed: ' + name)
        need(state.get('repeat_comparison', {}).get('max_minus_min_Ah') == 0,
             'Current admission requires measured exactly reproducible independent RPTs')
        for rep in reps:
            need(rep.get('source_state_and_ledger_unchanged') is True and rep['reference']['ok']
                 and rep.get('existing_integral_check_pass') is True
                 and finite(rep['direct_integral_relative_difference'], 'RPT integral') <= 1e-4,
                 'RPT direct-integral/source preservation check failed')
    precision, precision_path = evidence(r['precision_report'], base)
    result, precision_result_path = evidence(r['precision_worker_result'], base)
    need(precision.get('original_files_unchanged') is True and precision.get('worker_exit_code') == 0
         and precision.get('worker_status') == 'not_reproduced_within_cycle_bound',
         'Selected precision control did not pass the unchanged connection gate')
    need(float(precision.get('tolerance_factor', 0)) == args.numfactor
         and result.get('completed_cycles', 0) >= 3
         and result.get('source_solution_unchanged_in_memory') is True,
         'Selected numerical control must complete at least three cycles from unchanged source')
    applied = result.get('solver_precision_control') or {}
    need(applied.get('connection_gate') == 1e-10 and applied.get('physics_changed') is False
         and applied.get('connection_gate_changed') is False
         and float(applied.get('factor', 0)) == args.numfactor,
         'Precision gate/physics evidence missing or altered')
    need(any(e.get('same_baseline_cycle') is True for e in result.get('event_time_correspondence', [])),
         'Precision control did not reach the matching baseline phase event')
    x4, _ = evidence(r['mesh4_summary'], base)
    x8, _ = evidence(r['mesh8_summary'], base)
    for scale, data in [(4, x4), (8, x8)]:
        ident = data.get('identity', {})
        need(data.get('status') == 'efc_target' and ident.get('mesh_scale') == scale
             and ident.get('cell') == 4 and ident.get('current_A') == 3.2
             and abs(data['ledger']['efc']-100) <= 1e-8, 'Independent 100 EFC mesh summary incomplete')
    def at100(data):
        return next(d for d in data['diagnostics'] if abs(d['efc']-100) < 1e-8)
    mesh_diff = abs(at100(x4)['SOH_percent']-at100(x8)['SOH_percent'])
    need(mesh_diff <= .1, '100 EFC mesh SOH difference exceeds 0.1 percent-point precheck')
    budget = budget_plan(r['cost_admission'], args.max_cumulative_wall_s)
    summary = read(source.parent/'summary.json')
    prior = finite(summary['elapsed_wall_s'], 'source elapsed')
    failure = source.parent/'FAILED.json'
    if failure.is_file():
        prior = max(prior, finite(read(failure).get('elapsed_wall_s', 0), 'source failure elapsed'))
    remaining = args.max_cumulative_wall_s-prior
    need(remaining > 300, 'Cumulative trajectory budget already exhausted')
    return {'receipt_path': str(receipt_path), 'receipt_sha256': digest(receipt_path),
            'source_checkpoint': str(source), 'source_checkpoint_sha256': digest(source),
            'source_cumulative_wall_s': prior, 'maximum_new_wall_s': remaining,
            'precision_report': str(precision_path), 'precision_worker_result': str(precision_result_path),
            'selected_numerical_factor': args.numfactor, 'budget': budget,
            'mesh_scope': 'independent initial-to-100 EFC only', 'mesh_SOH_difference_pp': mesh_diff,
            'chemistry_validated': False,
            'scope_note': 'Admission does not establish real LG M50 chemistry or long/EOL grid convergence.'}


def build_config(args, admission):
    root = Path(args.root).resolve(); repo = Path(args.repo).resolve()
    for source in (repo, Path(admission['source_checkpoint']).parent):
        need(root != source and source not in root.parents and root not in source.parents,
             'New cloud root must be disjoint from source trees')
    unit = 'capstone-until80-' + hashlib.sha256(str(root).encode()).hexdigest()[:10]
    deadline = now()+dt.timedelta(seconds=admission['maximum_new_wall_s'])
    return {'schema': 'capstone-until80-cloud-v1', 'repo': str(repo), 'root': str(root),
            'python': str(Path(args.python).absolute()), 'source_cp': admission['source_checkpoint'],
            'numerical_factor': args.numfactor, 'max_cumulative_wall_s': args.max_cumulative_wall_s,
            'worker_wall_s': args.worker_wall_s, 'max_efc': args.max_efc,
            'deadline_utc': deadline.isoformat(), 'unit': unit, 'admission': admission,
            'created_utc': stamp(), 'chat_required': False, 'S3_automatic_upload': False,
            'SOH80_arrival_estimate_provided': False}


def unit_files(config, script):
    unit = config['unit']; root = config['root']; python = config['python']
    def command(mode):
        return ' '.join(shlex.quote(v) for v in [python, str(script), mode, '--root', root])
    service = ('[Unit]\nDescription=Bounded battery until80 calculation\nAfter=local-fs.target\n'
               '[Service]\nType=exec\nRestart=no\nKillMode=control-group\nTimeoutStopSec=120\n'
               'Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
               'ExecStart='+command('--run')+'\nExecStopPost='+command('--after-stop')+'\n')
    deadline_service = '[Unit]\nDescription=Independent battery UTC deadline\n[Service]\nType=oneshot\nExecStart='+command('--deadline-fire')+'\n'
    calendar = parse_time(config['deadline_utc']).strftime('%Y-%m-%d %H:%M:%S UTC')
    timer = ('[Unit]\nDescription=Persistent battery UTC budget deadline\n[Timer]\nOnCalendar='+calendar+
             '\nAccuracySec=1s\nPersistent=true\nUnit='+unit+'-deadline.service\n[Install]\nWantedBy=timers.target\n')
    boot = ('[Unit]\nDescription=Battery unclean-boot fail-safe\nAfter=local-fs.target\nBefore=multi-user.target\n'
            '[Service]\nType=oneshot\nExecStart='+command('--boot-guard')+'\n[Install]\nWantedBy=multi-user.target\n')
    return {unit+'.service':service, unit+'-deadline.service':deadline_service,
            unit+'-deadline.timer':timer, unit+'-boot-guard.service':boot}


def linux_root():
    need(sys.platform.startswith('linux') and hasattr(os, 'geteuid') and os.geteuid() == 0,
         'Actual cloud actions require Linux root; use --plan elsewhere')


def install(args, config):
    linux_root(); root = Path(config['root'])
    need(not root.exists() or not any(root.iterdir()), 'Use a new empty run root')
    need(Path(config['python']).is_file(), 'Selected venv Python missing')
    for rel in ['simulation/run_until80.py', 'simulation/until80_policy.py']:
        need((Path(config['repo'])/rel).is_file(), 'Missing reviewed worker: '+rel)
    root.mkdir(parents=True, exist_ok=True)
    save(root/'cloud_config.json', config)
    save(root/'cloud_state.json', {'status':'configured', 'terminal':False, 'server_shutdown_lease':False})
    definitions = unit_files(config, Path(__file__).resolve())
    for name, content in definitions.items():
        path = Path('/etc/systemd/system')/name
        need(not path.exists(), 'Existing persistent unit must not be overwritten: '+name)
        path.write_text(content, encoding='utf-8')
    subprocess.run(['systemctl', 'daemon-reload'], check=True, timeout=30)
    subprocess.run(['systemctl', 'enable', config['unit']+'-boot-guard.service'], check=True, timeout=30)
    subprocess.run(['systemctl', 'enable', '--now', config['unit']+'-deadline.timer'], check=True, timeout=30)
    subprocess.run(['systemctl', 'is-active', '--quiet', config['unit']+'-deadline.timer'], check=True, timeout=10)
    return {'status':'installed_not_started', 'service':config['unit']+'.service',
            'deadline_utc':config['deadline_utc'], 'run_root':str(root)}


def resources(root):
    import shutil
    free = shutil.disk_usage(root).free
    available = None
    try:
        m = {l.split(':',1)[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
        available = m['MemAvailable']/m['MemTotal']
    except (OSError, ValueError, KeyError):
        pass
    return {'free_GiB':free/GIB, 'available_ram_fraction':available,
            'limit':free < 2*GIB or (available is not None and available < .08)}


def estimate(config, elapsed_s):
    b = config['admission']['budget']; additional = RATE*max(0, elapsed_s)/3600
    return {'additional_compute_estimate_USD':additional,
            'planning_total_with_reserve_KRW':b['reserve_KRW']+
            (b['prior_spend_upper_USD']+additional)*b['planning_USDKRW']*b['budget_buffer'],
            'is_final_bill':False}


def worker_command(config, resume):
    result = [config['python'], str(Path(config['repo'])/'simulation/run_until80.py'),
              '--output-dir', str(Path(config['root'])/'trajectory'),
              '--tolerance-factor', str(config['numerical_factor']),
              '--numerical-control-report', config['admission']['precision_report'],
              '--max-wall-s', str(config['max_cumulative_wall_s']),
              '--worker-wall-s', str(config['worker_wall_s']),
              '--deadline-utc', config['deadline_utc'], '--max-efc', str(config['max_efc'])]
    return result+(['--resume'] if resume else ['--migrate-checkpoint', config['source_cp']])


def group_members(pgid):
    result = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            p = path.read_text().rsplit(')',1)[1].split()
            if int(p[2]) == pgid and p[0] != 'Z': result.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return result


def stop_group(proc, grace_s=60):
    record = {'TERM':False, 'KILL':False}
    if group_members(proc.pid):
        try: os.killpg(proc.pid, signal.SIGTERM); record['TERM']=True
        except ProcessLookupError: pass
        until = time.monotonic()+grace_s
        while group_members(proc.pid) and time.monotonic() < until:
            proc.poll(); time.sleep(.2)
        if group_members(proc.pid):
            try: os.killpg(proc.pid, signal.SIGKILL); record['KILL']=True
            except ProcessLookupError: pass
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
    record['group_empty'] = not group_members(proc.pid)
    return record


def validate_rollover(code, report, previous=None):
    need(code == 75 and report.get('status') == 'rollover_checkpointed'
         and report.get('terminal') is False, 'Only validated healthy exit75 can restart')
    need(isinstance(report.get('checkpoint_commit_id'), int) and report['checkpoint_commit_id'] >= 1,
         'Rollover commit ID missing')
    need(isinstance(report.get('history_commit_id'), str) and len(report['history_commit_id']) == 64,
         'Committed journal digest missing')
    for key in ['last_valid_EFC', 'cumulative_wall_s', 'last_SOH_measurement_EFC', 'last_SOH_percent']:
        finite(report[key], key)
    need(report['last_SOH_measurement_EFC'] <= report['last_valid_EFC']+1e-8,
         'Last SOH observation lies beyond checkpoint')
    if previous:
        need(report['checkpoint_commit_id'] > previous['checkpoint_commit_id']
             and report['last_valid_EFC'] >= previous['last_valid_EFC']
             and report['cumulative_wall_s'] >= previous['cumulative_wall_s'],
             'Rollover did not advance validated committed history')


def latest_committed_report(trajectory, policy):
    """On a failed worker, an old healthy outcome never overrides latest commit."""
    summary = read(Path(trajectory)/'summary.json')
    policy.validate_checkpoint_report(Path(trajectory), summary)
    return summary


def emit(root, event):
    line = json.dumps(event, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    with (root/'cloud_supervisor.log').open('a', encoding='utf-8') as f: f.write(line+'\n')
    print(line, flush=True)
    try:
        with open('/dev/console', 'a', encoding='utf-8') as console: console.write(line+'\n')
    except OSError:
        pass


def halt():
    try: subprocess.run(['sync'], check=False, timeout=30)
    finally: subprocess.run(['/usr/sbin/shutdown', '-h', 'now'], check=False, timeout=30)


def state_manifest(source):
    return {str(p):digest(p) for p in [source, *(source.parent/n for n in
            ['summary.json', 'diagnostics.json', 'FAILED.json', 'outcome.json'])] if p.is_file()}


def run(config):
    linux_root()
    import fcntl
    root = Path(config['root']); source = Path(config['source_cp'])
    state = read(root/'cloud_state.json')
    need(not state.get('terminal'), 'Terminal cloud run cannot be restarted')
    # A failed duplicate launch must not acquire authority to halt another job.
    lock = Path('/var/lib/battery-capstone/until80-cloud.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for other in ['capstone-audit.service', 'capstone-lifetime.service']:
        need(subprocess.run(['systemctl', 'is-active', '--quiet', other], check=False).returncode != 0,
             'Existing battery calculation still active: '+other)
    before = {}
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    state.update(status='starting', terminal=False, server_shutdown_lease=True,
                 boot_id=boot_id, started_utc=stamp(), deadline_utc=config['deadline_utc'],
                 rollovers=0, jobs_completed=0, chemistry_validated=False,
                 last_worker=None, source_manifest_before=before)
    save(root/'cloud_state.json', state)
    started = time.monotonic(); proc = None; previous = None; resume = False
    deadline = parse_time(config['deadline_utc'])
    sys.path.insert(0, str(Path(config['repo'])/'simulation'))
    try:
        subprocess.run(['systemctl', 'is-active', '--quiet', config['unit']+'-deadline.timer'], check=True, timeout=10)
        before = state_manifest(source)
        state['source_manifest_before'] = before
        save(root/'cloud_state.json', state)
        import until80_policy as policy
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, _frame: STOP.update(signal=number))
        need(digest(config['admission']['receipt_path']) == config['admission']['receipt_sha256'],
             'Admission receipt changed after installation')
        need(digest(source) == config['admission']['source_checkpoint_sha256'], 'Original checkpoint changed')
        while True:
            elapsed = time.monotonic()-started; remaining = (deadline-now()).total_seconds()
            resource = resources(root); cost = estimate(config, elapsed)
            reason = ('global_time_censored' if remaining <= 120 else
                      'resource_censored' if resource['limit'] else
                      'interrupted_censored' if STOP['signal'] else None)
            if cost['planning_total_with_reserve_KRW'] >= config['admission']['budget']['budget_KRW']:
                reason = 'global_time_censored'
            if reason: state['status']=reason; break
            index = state['jobs_completed']+1
            command = worker_command(config, resume)
            with (root/f'worker_{index:04d}.log').open('ab', buffering=0) as log:
                env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                proc = subprocess.Popen(command, cwd=config['repo'], env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', current_worker=index, worker_pid=proc.pid)
                save(root/'cloud_state.json', state)
                emit(root, {'stage':'until80_worker_started', 'worker':index, 'resume':resume,
                            'deadline_utc':config['deadline_utc'], 'utc':stamp()})
                worker_start = time.monotonic(); next_progress = 0.; forced_reason = None
                while proc.poll() is None:
                    elapsed = time.monotonic()-started; resource = resources(root)
                    if STOP['signal']: forced_reason='interrupted_censored'
                    elif (deadline-now()).total_seconds() <= 120: forced_reason='global_time_censored'
                    elif resource['limit']: forced_reason='resource_censored'
                    elif time.monotonic()-worker_start > config['worker_wall_s']+120: forced_reason='worker_failed'
                    if forced_reason:
                        state['termination'] = stop_group(proc); break
                    if elapsed >= next_progress:
                        current_path = root/'trajectory/summary.json'
                        current = read(current_path) if current_path.is_file() else {}
                        progress = {k:current.get(k) for k in ['status', 'last_valid_EFC', 'last_SOH_percent', 'last_SOH_measurement_EFC', 'cumulative_wall_s']}
                        state.update(progress=progress, resources=resource, cost_estimate=estimate(config, elapsed))
                        save(root/'cloud_state.json', state)
                        emit(root, {'stage':'until80_progress', **progress, 'cost_estimate':state['cost_estimate'], 'utc':stamp()})
                        next_progress = elapsed+300
                    time.sleep(1)
                code = proc.wait(timeout=15)
                if group_members(proc.pid):
                    state['termination'] = stop_group(proc); forced_reason = forced_reason or 'worker_failed'
                proc = None
            state['jobs_completed'] = index
            outcome_path = root/'trajectory/outcome.json'
            report = read(outcome_path) if outcome_path.is_file() else {}
            state['last_worker'] = {'returncode':code, 'outcome':report}
            try:
                state['last_committed_report'] = latest_committed_report(root/'trajectory', policy)
            except BaseException as exc:
                state['last_committed_validation_error_type'] = type(exc).__name__
            if forced_reason: state['status']=forced_reason; break
            # Nonzero physics exceptions override any old healthy outcome file.
            if code not in (0, 75, 124): state['status']='worker_failed'; break
            policy.validate_checkpoint_report(root/'trajectory', report)
            state['last_committed_report'] = report
            if code == 75:
                validate_rollover(code, report, previous)
                previous = report; resume=True; state['rollovers'] += 1
                save(root/'cloud_state.json', state)
                continue
            if code == 0 and report.get('terminal') is True and report.get('status') in ('eol_observed', 'max_efc_censored'):
                state['status'] = report['status']; break
            if code == 124 and report.get('terminal') is True:
                state['status'] = ('resource_censored' if 'resource' in report.get('status','') else
                                   'global_time_censored' if 'time' in report.get('status','') else 'interrupted_censored')
                break
            state['status']='invalid_worker_contract'; break
    except BaseException as exc:
        state.update(status='coordinator_failed', error_type=type(exc).__name__, error=str(exc)[:1200])
    finally:
        if proc is not None:
            try: state['termination'] = stop_group(proc)
            except BaseException as exc:
                state.update(status='coordinator_failed', cleanup_error_type=type(exc).__name__)
        try:
            after = state_manifest(source)
            source_unchanged = bool(before) and before == after
        except BaseException as exc:
            after = {}; source_unchanged = False
            state['source_hash_error_type'] = type(exc).__name__
        state.update(source_manifest_after=after, original_sources_unchanged=source_unchanged,
                     terminal=True, current_worker=None, finished_utc=stamp(),
                     wall_s=time.monotonic()-started, cost_estimate=estimate(config,time.monotonic()-started),
                     shutdown_requested=True)
        if before and before != after: state['status']='source_modified'
        try:
            save(root/'cloud_state.json', state)
            last = state.get('last_committed_report') or {}
            emit(root, {'stage':'until80_final', 'status':state['status'], 'utc':stamp(),
                        **{k:last.get(k) for k in ['last_valid_EFC','last_SOH_measurement_EFC','last_SOH_percent','cumulative_wall_s']},
                        'original_sources_unchanged':state['original_sources_unchanged'],
                        'cost_estimate':state['cost_estimate'], 'shutdown_requested':True,
                        'results_location':'EBS only; no expiring S3 upload promise'})
        finally:
            halt()
            lock.close()
    return 0 if state['status'] in ('eol_observed', 'max_efc_censored') else 1


def guard_action(mode, config):
    linux_root(); root = Path(config['root']); path = root/'cloud_state.json'
    if not path.is_file(): return {'status':'no_cloud_state_no_action'}
    state = read(path)
    if not state.get('server_shutdown_lease'): return {'status':'no_shutdown_lease_no_action'}
    if mode != 'after-stop' and state.get('terminal'): return {'status':'already_terminal_recovery_allowed'}
    if mode == 'boot-guard':
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot == state.get('boot_id'): return {'status':'same_boot_no_action'}
        state.update(status='unclean_boot_failed', terminal=True, finished_utc=stamp(),
                     note='No automatic computation resume after an unclean reboot. A later recovery boot is allowed.')
    elif mode == 'deadline-fire':
        if now() < parse_time(config['deadline_utc']): return {'status':'deadline_not_reached'}
        state.update(status='global_time_censored', terminal=True, finished_utc=stamp())
        try:
            subprocess.run(['systemctl','kill','--kill-whom=all','--signal=SIGTERM',config['unit']+'.service'], check=False,timeout=15)
        except (subprocess.TimeoutExpired, OSError):
            pass  # The independent machine stop must not depend on systemctl success.
    elif mode == 'after-stop' and not state.get('terminal'):
        state.update(status='coordinator_failed', terminal=True, finished_utc=stamp(),
                     note='Service exited outside its normal finalizer; recovery is allowed after fail-safe stop.')
    state['shutdown_requested'] = True
    try: save(path, state)
    finally: halt()
    return {'status':state['status'], 'shutdown_requested':True}


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    for option in ['plan','install','run','deadline-fire','boot-guard','after-stop']:
        group.add_argument('--'+option, action='store_true')
    p.add_argument('--repo'); p.add_argument('--root', required=True); p.add_argument('--python')
    p.add_argument('--source-cp'); p.add_argument('--numfactor',type=float,choices=(.1,.01))
    p.add_argument('--admission-receipt');p.add_argument('--max-cumulative-wall-s',type=float,default=MAX_WALL_S)
    p.add_argument('--worker-wall-s',type=float,default=72000);p.add_argument('--max-efc',type=float,default=1000000)
    args=p.parse_args()
    need(300 <= args.worker_wall_s <= 72000 and 0 < args.max_efc <= 1000000, 'Invalid worker/EFC bounds')
    return args


def main():
    args=parse()
    mode = next((m for m in ['run','deadline_fire','boot_guard','after_stop','install'] if getattr(args,m)), 'plan')
    if mode in ('run','deadline_fire','boot_guard','after_stop'):
        config=read(Path(args.root)/'cloud_config.json')
        if mode=='run': return run(config)
        print(json.dumps(guard_action(mode.replace('_','-'),config)));return 0
    for name in ('repo','python','source_cp','numfactor','admission_receipt'):
        need(getattr(args,name) is not None, 'Missing --'+name.replace('_','-'))
    admission=validate_admission(args);config=build_config(args,admission)
    if mode=='install': print(json.dumps(install(args,config),indent=2));return 0
    print(json.dumps({'mode':'plan','no_changes':True,'no_jobs_started':True,'no_shutdown':True,
                      'config':config,'units':unit_files(config,Path(__file__).resolve()),
                      'first_worker_command':worker_command(config,False)},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
