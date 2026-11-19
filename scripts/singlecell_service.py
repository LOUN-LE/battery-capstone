"""One admitted single-cell worker, independent of chat; no AWS credentials stored.

Default is an offline plan. --install starts a detached systemd service only after
receipt, source, budget, and absolute-deadline checks. SNS topic permissions and
email subscription must already exist. No IAM or SNS configuration is changed.
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

UTC = dt.timezone.utc
LATEST_HARD_STOP = dt.datetime(2026, 11, 2, 14, 59, tzinfo=UTC)


def utcnow():
    return dt.datetime.now(UTC)


def parse_utc(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Absolute timezone-aware deadline required')
    return parsed.astimezone(UTC)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, minimum=0):
    result = float(value)
    require(math.isfinite(result) and result >= minimum, 'Invalid ' + name)
    return result


def validate(args, now=None):
    now = now or utcnow()
    hard = parse_utc(args.deadline_utc)
    require(now < hard <= LATEST_HARD_STOP, 'Deadline is expired or exceeds approved today23:59KST')
    reserve_s = number(args.shutdown_reserve_s, 'shutdown reserve', 300)
    soft = hard - dt.timedelta(seconds=reserve_s)
    require((soft-now).total_seconds() >= 120, 'Insufficient time before soft stop')
    require(300 <= args.email_interval_s <= 7200, 'Email interval must be300..7200seconds')
    require(args.region == 'us-east-1', 'Only the reviewed region us-east-1 is allowed')
    require(args.topic_arn.startswith('arn:aws:sns:us-east-1:000000000000:capstone-'), 'Unreviewed SNS topic')
    require(args.s3_prefix.startswith('s3://battery-capstone-backup-000000000000-us-east-1/repair/20261102/'), 'Unreviewed backup prefix')
    repo = Path(args.repo_root).resolve(strict=True)
    source = Path(args.source_cp).resolve(strict=True)
    archive = Path(args.source_archive).resolve(strict=True)
    out = Path(args.output_dir).resolve()
    require(source.parent.parent != out and source.parent.parent not in out.parents and out not in source.parent.parent.parents,
            'Original and new output must be disjoint')
    require(not out.exists() or not any(out.iterdir()), 'New worker output must be empty')
    admission = read(args.admission)
    require(admission.get('schema') == 'singlecell-admission-v1', 'Unsupported admission schema')
    require(admission.get('instance_id') == 'i-00000000000000000', 'Unreviewed instance')
    require(digest(source) == admission['checkpoint_sha256'], 'Checkpoint hash changed')
    require(digest(archive) == admission['archive_sha256'], 'Source archive hash changed')
    require(admission.get('source_hashes'), 'Source hash manifest is required')
    for name, sha in admission['source_hashes'].items():
        path = (repo / name).resolve()
        require(repo in path.parents, 'Source path escapes repository')
        require(digest(path) == sha, 'Source hash changed: ' + name)
    ref = admission['validation_receipt']
    receipt_path = Path(ref['path'])
    if not receipt_path.is_absolute():
        receipt_path = Path(args.admission).resolve().parent / receipt_path
    require(digest(receipt_path) == ref['sha256'], 'Validation receipt hash changed')
    receipt = read(receipt_path)
    require(receipt.get('passed') is True and receipt.get('mesh_scale') == 4
            and receipt.get('source_checkpoint_sha256') == admission['checkpoint_sha256']
            and number(receipt.get('reached_EFC', 0), 'reached EFC') >= 1620
            and receipt.get('failed_transition_passed') is True
            and receipt.get('post_rpt_transition_passed') is True
            and receipt.get('source_unchanged') is True,
            'Original1614→1620RPT→post-RPT-transition validation is not admitted')
    cost = admission['cost']
    rate = number(cost['rate_USD_h'], 'actual hourly rate', .000001)
    prior = number(cost['prior_spend_upper_USD'], 'prior upper spend')
    fx = number(cost['planning_USDKRW'], 'planningFX', 2000)
    buffer = number(cost['budget_buffer'], 'budget buffer', 1.10)
    reserve = number(cost['reserve_KRW'], 'reserveKRW', 150000)
    budget = number(cost['budget_KRW'], 'budgetKRW', 1)
    require(budget <= 600000 and reserve < budget, 'Approved aggregate budget exceeded')
    require(cost.get('rate_verified_utc') and cost.get('prior_spend_reviewed_utc'), 'Authenticated price/spend review times required')
    for key in ('rate_verified_utc', 'prior_spend_reviewed_utc'):
        age = (now - parse_utc(cost[key])).total_seconds()
        require(-300 <= age <= 86400, 'Cost review is stale: ' + key)
    cost_upper = reserve + (prior + rate * (hard-now).total_seconds()/3600) * fx * buffer
    require(cost_upper <= budget, 'Conservative aggregate cost exceeds approved cap')
    source_summary = read(source.parent / 'summary.json')
    source_wall = float(source_summary.get('elapsed_wall_s', 0))
    if (source.parent / 'FAILED.json').exists():
        source_wall = max(source_wall, float(read(source.parent / 'FAILED.json').get('elapsed_wall_s', 0)))
    remaining = (soft-now).total_seconds()
    require(0 < remaining < 20*3600, 'One-worker plan requires remaining time below20h')
    unit = 'capstone-singlecell-' + hashlib.sha256(str(out).encode()).hexdigest()[:10]
    control = out.parent / (out.name + '.service')
    return {'schema': 'singlecell-service-v1', 'created_utc': now.isoformat(),
            'instance_id': admission['instance_id'], 'hard_stop_utc': hard.isoformat(),
            'worker_soft_stop_utc': soft.isoformat(), 'shutdown_reserve_s': reserve_s,
            'source_elapsed_wall_s': source_wall, 'maximum_new_worker_s': remaining,
            'worker_max_cumulative_s': source_wall + remaining,
            'planning_total_upper_KRW': cost_upper, 'rate_USD_h': rate,
            'prior_spend_upper_USD': prior, 'planning_USDKRW': fx,
            'budget_buffer': buffer, 'reserve_KRW': reserve, 'budget_KRW': budget,
            'unit': unit, 'control_dir': str(control), 'output_dir': str(out),
            'admission_sha256': digest(args.admission), 'source_checkpoint_sha256': admission['checkpoint_sha256'],
            'note': 'Hard-stop time is a resource deadline, not an SOH80 arrival estimate'}


def worker_command(args, plan):
    return [args.python, '-B', str(Path(args.repo_root)/'simulation/run_until80.py'),
            '--migrate-checkpoint', args.source_cp, '--source-archive', args.source_archive,
            '--output-dir', args.output_dir, '--cell', '4', '--current', '3.2', '--mesh-scale', '4',
            '--tolerance-factor', '1', '--max-efc', '1000000',
            '--max-wall-s', str(plan['worker_max_cumulative_s']),
            '--worker-wall-s', str(plan['maximum_new_worker_s']),
            '--deadline-utc', plan['worker_soft_stop_utc'], '--rpt-every-efc', '10',
            '--max-memory-gib', '24', '--min-free-memory-gib', '2', '--min-free-disk-gib', '4']


def systemd_units(args, plan, argv):
    unit = plan['unit']
    # Persistent timer executes shutdown directly, even if Python/worker crashes.
    calendar = parse_utc(plan['hard_stop_utc']).strftime('%Y-%m-%d %H:%M:%S UTC')
    command = [args.python, '-B', str(Path(__file__).resolve()), *argv, '--_run']
    def quoted(parts):
        # systemd does not use a shell; percent is its own specifier character.
        return ' '.join('"' + str(x).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"' for x in parts)
    return {unit+'.service': '[Unit]\nDescription=Admitted battery single-cell lifetime\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=exec\nRestart=no\nKillMode=mixed\nTimeoutStopSec=240\nExecStart='+quoted(command)+'\nExecStopPost=/usr/sbin/shutdown -h +1\n',
            unit+'-hardstop.service': '[Unit]\nDescription=Independent battery experiment hard stop\n\n[Service]\nType=oneshot\nExecStart=/usr/sbin/shutdown -h now\n',
            unit+'-hardstop.timer': '[Unit]\nDescription=Absolute battery experiment hard deadline\n\n[Timer]\nOnCalendar='+calendar+'\nAccuracySec=1s\nPersistent=true\nUnit='+unit+'-hardstop.service\n\n[Install]\nWantedBy=timers.target\n'}


def run_command(command, timeout=60):
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          encoding='utf-8', timeout=timeout)


def notify(args, event, status):
    payload = {'event': event, 'utc': utcnow().isoformat(), **status}
    run_command(['aws', '--region', args.region, 'sns', 'publish', '--topic-arn', args.topic_arn,
                 '--subject', 'Battery cell4: ' + event, '--message', json.dumps(payload, ensure_ascii=False),
                 '--no-cli-pager'], timeout=45)


def snapshot(args, plan, event, began):
    out = Path(args.output_dir)
    data = {}
    for name in ('summary.json', 'outcome.json', 'FAILED.json'):
        if (out/name).is_file():
            try:
                current = read(out/name)
                data.update({k: current[k] for k in ('status', 'last_valid_EFC', 'last_SOH_percent',
                    'last_SOH_measurement_EFC', 'eol', 'row_count', 'rpt_count', 'error') if k in current})
            except (OSError, ValueError):
                pass
    elapsed = time.monotonic()-began
    return {'event': event, 'utc': utcnow().isoformat(), 'instance_id': plan['instance_id'],
            'elapsed_worker_wall_s': elapsed, 'estimated_new_compute_USD': elapsed/3600*plan['rate_USD_h'],
            'hard_stop_utc': plan['hard_stop_utc'], 'soft_stop_utc': plan['worker_soft_stop_utc'],
            'output_dir': args.output_dir, **data}


def execute(args):
    # Recheck admission at actual service start. Planning never consumes compute.
    plan = validate(args)
    control = Path(plan['control_dir'])
    control.mkdir(parents=True, exist_ok=True)
    save(control/'plan.json', plan)
    began = time.monotonic()
    events = []
    process = None
    stop = {'requested': False}
    def stop_requested(*_):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    def event(kind, **extra):
        state = {**snapshot(args, plan, kind, began), **extra}
        save(control/'status.json', state)
        events.append(state)
        save(control/'events.json', events)
        try:
            notify(args, kind, state)
        except Exception as exc:
            save(control/'notification_error.json', {'utc': utcnow().isoformat(), 'error': str(exc)})
    def backup():
        result = []
        for source, suffix in ((Path(args.output_dir), 'trajectory'), (control, 'service')):
            if source.exists():
                # Never --delete; originals and earlier S3 objects are retained.
                done = run_command(['aws', '--region', args.region, 's3', 'sync', str(source),
                    args.s3_prefix.rstrip('/')+'/'+suffix+'/', '--exclude', '*.tmp', '--exclude', '*.runner.lock',
                    '--only-show-errors', '--no-cli-pager'], timeout=180 if suffix == 'trajectory' else 45)
                result.append({'source': str(source), 'returncode': done.returncode})
        save(control/'backup_receipt.json', {'utc': utcnow().isoformat(), 'sync_results': result,
             'note': 'AWS CLI sync succeeded; S3 objects remain private; checksum inventory is in trajectory journal'})
    try:
        event('starting')
        with (control/'worker.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(worker_command(args, plan), stdout=log, stderr=subprocess.STDOUT,
                cwd=args.repo_root, start_new_session=True,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
            save(control/'worker_pid.json', {'pid': process.pid, 'started_utc': utcnow().isoformat()})
            last_mail = time.monotonic()
            while process.poll() is None:
                if stop['requested'] or utcnow() >= parse_utc(plan['worker_soft_stop_utc']):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
                    break
                if time.monotonic()-last_mail >= args.email_interval_s:
                    event('progress')
                    last_mail = time.monotonic()
                time.sleep(5)
        backup()
        final = snapshot(args, plan, 'worker_exit', began)
        failed = (Path(args.output_dir)/'FAILED.json').exists() or process.returncode not in (0, 75, 124)
        event('failed' if failed else 'eol_observed' if final.get('status') == 'eol_observed' else 'stopped',
              worker_exit_code=process.returncode, backup_completed=True)
        # Include the final status and backup receipt, written after the first sync.
        run_command(['aws', '--region', args.region, 's3', 'sync', str(control),
            args.s3_prefix.rstrip('/')+'/service/', '--exclude', '*.tmp', '--only-show-errors',
            '--no-cli-pager'], timeout=45)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        event('failed', coordinator_error=repr(exc))
        try:
            backup()
        except Exception as backup_exc:
            save(control/'backup_error.json', {'error': repr(backup_exc), 'utc': utcnow().isoformat()})
        raise
    finally:
        # This is an additional stop path; independent systemd timer remains armed.
        run_command(['/usr/sbin/shutdown', '-h', '+1'], timeout=15)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--source-cp', required=True)
    p.add_argument('--source-archive', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--admission', required=True)
    p.add_argument('--deadline-utc', required=True)
    p.add_argument('--shutdown-reserve-s', type=float, default=600)
    p.add_argument('--region', default='us-east-1')
    p.add_argument('--topic-arn', required=True)
    p.add_argument('--s3-prefix', required=True)
    p.add_argument('--email-interval-s', type=float, default=3600)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--install', action='store_true')
    group.add_argument('--_run', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args._run:
        require(sys.platform.startswith('linux') and os.geteuid() == 0, 'Linux root required')
        return execute(args)
    plan = validate(args)
    argv = [x for x in sys.argv[1:] if x != '--install']
    units = systemd_units(args, plan, argv)
    if not args.install:
        print(json.dumps({'plan': plan, 'worker_command': worker_command(args, plan), 'systemd_units': units}, indent=2))
        return
    require(sys.platform.startswith('linux') and os.geteuid() == 0, 'Linux root required')
    control = Path(plan['control_dir'])
    require(not control.exists(), 'Control directory already exists; do not overwrite a run')
    control.mkdir(parents=True)
    save(control/'plan.json', plan)
    for name, content in units.items():
        dest = Path('/etc/systemd/system')/name
        require(not dest.exists(), 'Existing unit cannot be overwritten: '+name)
        dest.write_text(content, encoding='utf-8')
    run_command(['systemctl', 'daemon-reload'])
    timer = plan['unit']+'-hardstop.timer'
    # Timer is armed and queried BEFORE the compute service starts.
    run_command(['systemctl', 'enable', '--now', timer])
    state = run_command(['systemctl', 'show', timer, '--property=ActiveState,NextElapseUSecRealtime'])
    fields = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
    require(fields.get('ActiveState') == 'active' and fields.get('NextElapseUSecRealtime') not in (None, '', 'n/a'),
            'Independent hard-stop timer did not arm')
    save(control/'hard_timer_receipt.json', {'stdout': state.stdout, 'checked_utc': utcnow().isoformat()})
    run_command(['systemctl', 'start', plan['unit']+'.service'])
    active = run_command(['systemctl', 'show', plan['unit']+'.service', '--property=ActiveState,MainPID'])
    print(json.dumps({'plan':plan, 'hard_timer':state.stdout, 'worker_service':active.stdout}, indent=2))


if __name__ == '__main__':
    main()
