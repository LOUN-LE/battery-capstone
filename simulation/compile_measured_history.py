"""Compile measured C/D history on the naive instrument clock, without inventing rest.

Clock admission requires a complete matching MPR/MPT audit. Run envelopes remain
recorded exposure summaries, not reconstructed programmed cutoff definitions.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(value):
    return value is True or isinstance(value, str) and value.lower() == 'true'


def number(value):
    if value in ('', None):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite recorded history value')
    return result


def naive(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is not None:
        raise ValueError('Expected naive instrument clock; timezone conversion is not authorized')
    return stamp


def basename(path):
    return PurePosixPath(path.replace('\\', '/')).name


def source_key(cell, path):
    return cell, str(PurePosixPath(basename(path)).with_suffix('')).casefold()


def clock_admission(rows, audit, expected_mpt_names):
    files = audit.get('files', [])
    total = audit.get('sources_total')
    pairs = [r for r in rows if truth(r.get('MPT_pair_present'))]
    statuses = {'complete', 'completed', 'all126_binary_clocks_and98_MPT_origins_audited'}
    record_by_path = {r['source_path']: r for r in files}
    checks = {'audit_has_final_status': audit.get('status') in statuses,
              'all_sources_downloaded_and_parsed': isinstance(total, int) and total > 0 and len(files) == len(rows) == audit.get('parsed_count') == audit.get('download_verified_count') == total,
              'all_available_pairs_verified': audit.get('all_available_MPT_clock_mappings_verified') is True and len(pairs) == audit.get('available_MPT_pairs_expected') == audit.get('MPT_pairs_verified') == len(expected_mpt_names),
              'unique_native_sources': len(record_by_path) == len(files) and len({r['source_path'] for r in rows}) == len(rows),
              'each_MPT_mapping_verified': all(truth(r.get('mapping_verified')) for r in pairs),
              'CSV_matches_audited_sources': all(r['source_path'] in record_by_path and record_by_path[r['source_path']].get('stage') == 'clock_parsed' for r in rows)}
    seen = set()
    pair_names = set()
    for row in rows:
        key = source_key(row['cell'], row['source_path'])
        if key in seen:
            checks['unique_canonical_cell_and_full_stem'] = False
        seen.add(key)
        if truth(row.get('MPT_pair_present')):
            pair_names.add(str(PurePosixPath(basename(row['source_path'])).with_suffix('.mpt')))
        record = record_by_path.get(row['source_path'], {})
        clock = record.get('clock', {})
        if (record.get('CRC32_verified') is not True or record.get('local_file') != row.get('local_file')
                or clock.get('cell') != row['cell'] or clock.get('timestamp_naive') != row.get('timestamp_naive')):
            checks['CSV_audit_identity_and_verified_binary_match'] = False
        if truth(row.get('MPT_pair_present')) and clock.get('mapping_verified') is not True:
            checks['each_MPT_mapping_verified'] = False
        if truth(row.get('MPT_pair_present')) != truth(clock.get('MPT_pair_present')):
            checks['CSV_pair_presence_matches_audit'] = False
        if row.get('MPT_absolute_origin_naive', '') != (clock.get('MPT_absolute_origin_naive') or ''):
            checks['CSV_origin_matches_audited_clock'] = False
        fields = ('MPR_rows','MPT_rows','MPR_t_first_s','MPR_t_last_s','MPT_t_first_s','MPT_t_last_s',
                  'time_offset_MPR_minus_MPT_median_s','time_offset_max_residual_s',
                  'current_max_abs_difference_mA','voltage_max_abs_difference_V')
        try:
            if any(number(row.get(field)) != number(clock.get(field)) for field in fields):
                checks['CSV_numerical_clock_evidence_matches_audit'] = False
        except (ValueError, TypeError):
            checks['CSV_numerical_clock_evidence_matches_audit'] = False
    checks.setdefault('unique_canonical_cell_and_full_stem', True)
    checks.setdefault('CSV_audit_identity_and_verified_binary_match', True)
    checks.setdefault('CSV_pair_presence_matches_audit', True)
    checks.setdefault('CSV_origin_matches_audited_clock', True)
    checks.setdefault('CSV_numerical_clock_evidence_matches_audit', True)
    checks['every_timeline_MPT_has_exact_native_pair'] = pair_names == set(expected_mpt_names)
    return checks


def mapped_origin(row):
    if not truth(row.get('mapping_verified')):
        raise ValueError('Cannot use an unverified MPR/MPT time origin')
    origin = naive(row['MPT_absolute_origin_naive'])
    expected = naive(row['timestamp_naive'])+timedelta(seconds=number(row['time_offset_MPR_minus_MPT_median_s']))
    if abs((origin-expected).total_seconds()) > 1e-6:
        raise ValueError('Mapped MPT origin is inconsistent with native origin and measured offset')
    return origin


def make_segment(row, native_path, clock=None, source_kind='timeline_run', raw_sha=None):
    valid = truth(row.get('analysis_valid', True))
    issues = [value for value in (row.get('flags'), row.get('anomaly_reason')) if value]
    start, end, origin = None, None, None
    if valid and row.get('kind') == 'rpt':
        if clock is None:
            issues.append('No verified exact native MPR/MPT origin')
        else:
            origin = mapped_origin(clock)
            first, last = number(row['t_first_s']), number(row['t_last_s'])
            if first < number(clock['MPT_t_first_s'])-1e-6 or last > number(clock['MPT_t_last_s'])+1e-6:
                raise ValueError('Timeline run falls outside the exact verified MPT pair')
            start, end = origin+timedelta(seconds=first), origin+timedelta(seconds=last)
    elif valid and truth(row.get('absolute_time_known')):
        start, end = naive(row['abs_start']), naive(row['abs_end'])
        origin = naive(row['time_reference']) if row.get('time_reference') else None
    if start is not None and end < start:
        valid = False
        issues.append('Recorded run end precedes start')
        start, end = None, None
    net, method = None, None
    if valid:
        for field in ('charge_dq_mAh', 'charge_qqo_mAh', 'charge_trapz_mAh'):
            if row.get(field) not in ('', None):
                net, method = number(row[field])/1000, field
                break
    if not valid:
        issues.append('Quarantined raw record; no physical duration or throughput assigned')
    return {'segment_id': row['run_id'], 'cell': row['cell'], 'kind': row['kind'],
            'group': row.get('group'), 'part': row.get('part'),
            'source_file': row['source_file'], 'native_source_path': native_path,
            'session_path': str(PurePosixPath(native_path).parent), 'file_id': row.get('file_id'),
            'source_kind': source_kind, 'declared_raw_sha256': raw_sha,
            'row_start': int(row['row_start']), 'row_end_exclusive': int(row['row_end_exclusive']),
            'analysis_valid': valid, 'absolute_time_known': start is not None,
            'absolute_start_naive': start.isoformat() if start else None,
            'absolute_end_naive': end.isoformat() if end else None,
            'instrument_origin_naive': origin.isoformat() if origin else None,
            'recorded_t_first_s': number(row.get('t_first_s')), 'recorded_t_last_s': number(row.get('t_last_s')),
            'recorded_sample_span_s': (end-start).total_seconds() if start is not None else None,
            'existing_effective_duration_s': number(row.get('effective_duration_s')) if valid else None,
            'phase': row.get('phase'), 'segment_role': row.get('segment_role'),
            'control_type': row.get('control_type'), 'control_value': number(row.get('control_value')),
            'control_unit': row.get('control_unit'), 'control_basis': row.get('control_basis'),
            'current_direction': row.get('current_direction'), 'I_median_mA': number(row.get('I_median_mA')),
            'T_median_C': number(row.get('T_median_C')), 'T_min_C': number(row.get('T_min_C')), 'T_max_C': number(row.get('T_max_C')),
            'net_recorded_charge_Ah': net, 'net_charge_method': method,
            'programmed_cutoffs_known': False, 'issues': issues}


def coverage_intervals(segments):
    """Use only sample envelopes; inter-sample gaps are never silently filled."""
    intervals = []
    for cell in sorted({s['cell'] for s in segments}):
        rows = sorted((s for s in segments if s['cell'] == cell and s['analysis_valid'] and s['absolute_time_known']),
                      key=lambda s: (s['absolute_start_naive'], s['absolute_end_naive'], s['segment_id']))
        previous, frontier = None, None
        for row in rows:
            start, end = naive(row['absolute_start_naive']), naive(row['absolute_end_naive'])
            if frontier is not None and start > frontier:
                adjacent = previous['source_file'] == row['source_file'] and previous['row_end_exclusive'] == row['row_start']
                intervals.append({'cell': cell, 'kind': 'inter_sample_transition_interval' if adjacent else 'unobserved_gap',
                                  'start_naive': frontier.isoformat(), 'end_naive': start.isoformat(),
                                  'duration_s': (start-frontier).total_seconds(),
                                  'from_segment': previous['segment_id'], 'to_segment': row['segment_id'],
                                  'inferred_operation': None, 'known_rest': False})
            elif frontier is not None and start < frontier:
                intervals.append({'cell': cell, 'kind': 'recorded_interval_overlap',
                                  'start_naive': start.isoformat(), 'end_naive': min(frontier,end).isoformat(),
                                  'duration_s': max(0., (min(frontier,end)-start).total_seconds()),
                                  'from_segment': previous['segment_id'], 'to_segment': row['segment_id'],
                                  'inferred_operation': None, 'known_rest': False})
            if frontier is None or end >= frontier:
                previous, frontier = row, end
    return intervals


def binary_only_segments(row, record, clock_dir, common):
    path = clock_dir/row['local_file']
    if sha(path) != record['sha256']:
        raise ValueError('Binary-only source differs from verified clock audit')
    cols, meta, timestamp = common.read_mpr(path)
    if timestamp is None or timestamp.isoformat() != row['timestamp_naive']:
        raise ValueError('Binary-only absolute clock changed')
    ranges, basis = common.split_runs(cols)
    info = common.describe(row['source_path'])
    segments = []
    for index, (a,b) in enumerate(ranges):
        following = float(cols['t'][ranges[index+1][0]]) if index+1 < len(ranges) else None
        summary = common.summarize_run(cols, a, b, timestamp, following)
        summary.update({'run_id': f"binary:{Path(row['local_file']).stem}#r{index:05d}",
                        'cell': row['cell'], 'kind': 'rpt', 'source_file': basename(row['source_path']),
                        'file_id': 'binary:'+Path(row['local_file']).stem,
                        'group': info['group'], 'part': info['part'], 'analysis_valid': True,
                        'absolute_time_known': True, 'time_reference': timestamp.isoformat(),
                        'phase': 'recorded_binary_only_RPT',
                        'segment_role': 'rest' if summary['control_type'] == 'rest' else 'recorded_sequence',
                        'effective_duration_s': summary['duration_to_next_s'] if following is not None else summary['duration_s'],
                        'flags': 'Recorded non-increasing time' if summary['n_dt_nonpositive'] else ''})
        # This is already an absolute binary source; use the direct-clock branch.
        temporary = dict(summary, kind='binary_rpt')
        segment = make_segment(temporary, row['source_path'], source_kind='binary_only_MPR_run', raw_sha=record['sha256'])
        segment['kind'] = 'rpt'
        if summary['n_dt_nonpositive'] or summary['duration_s'] < 0:
            segment['analysis_valid'] = False
            segment['net_recorded_charge_Ah'] = None
            segment['issues'].append('Binary-only source requires anomaly review')
        segments.append(segment)
    return segments


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def compile_history(timeline_path, clock_dir, output, require_complete=True):
    output.mkdir(parents=True, exist_ok=False)
    paths = {'timeline': timeline_path, 'file_manifest': timeline_path.with_name('file_manifest.csv'),
             'clock_mappings': clock_dir/'clock_mappings.csv', 'clock_audit_report': clock_dir/'clock_audit_report.json'}
    blobs = {k: p.read_bytes() for k,p in paths.items()}
    hashes = {k: hashlib.sha256(v).hexdigest() for k,v in blobs.items()}
    csv_rows = lambda key: list(csv.DictReader(io.StringIO(blobs[key].decode('utf-8-sig'))))
    timeline, mappings, manifest = csv_rows('timeline'), csv_rows('clock_mappings'), csv_rows('file_manifest')
    audit = json.loads(blobs['clock_audit_report'].decode('utf-8'))
    runs = [r for r in timeline if r['row_type'] == 'run']
    mpt_names = {r['source_file'] for r in runs if r['kind'] == 'rpt'}
    checks = clock_admission(mappings, audit, mpt_names)
    complete = all(checks.values())
    if require_complete and not complete:
        report = {'status': 'blocked_incomplete_clock_audit', 'clock_checks': checks,
                  'source_sha256': hashes, 'compiler_completed': False, 'full_history_replay_admissible': False}
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
    mapping = {source_key(r['cell'], r['source_path']): r for r in mappings if truth(r.get('MPT_pair_present'))}
    manifest_by_name = {r['source_file']: r for r in manifest}
    records_by_path = {r['source_path']: r for r in audit['files']}
    segments = []
    for row in runs:
        item = mapping.get(source_key(row['cell'], row['source_file'])) if row['kind'] == 'rpt' else None
        native = item['source_path'] if item else manifest_by_name.get(row['source_file'], {}).get('zip_path', row['source_file'])
        declared_raw_hash = manifest_by_name.get(row['source_file'], {}).get('sha256')
        if item:
            paired = records_by_path[item['source_path']]['clock']
            if paired.get('MPT_sha256') != declared_raw_hash:
                raise ValueError('Timeline MPT source manifest differs from exact verified clock pair')
        segments.append(make_segment(row, native, item, raw_sha=declared_raw_hash))
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'work/preprocess'))
    import common
    parsed_binary = []
    for row in mappings:
        if not truth(row.get('MPT_pair_present')):
            record = records_by_path[row['source_path']]
            added = binary_only_segments(row, record, clock_dir, common)
            segments.extend(added)
            parsed_binary.append({'source_path': row['source_path'], 'sha256': record['sha256'], 'runs': len(added)})
    segments.sort(key=lambda s: (s['cell'], s['absolute_start_naive'] or '9999', s['absolute_end_naive'] or '9999', s['segment_id']))
    intervals = coverage_intervals(segments)
    net_by_role = {}
    for kind in ('ageing','rpt'):
        rows = [s for s in segments if s['kind'] == kind]
        net = [s['net_recorded_charge_Ah'] for s in rows if s['analysis_valid'] and s['net_recorded_charge_Ah'] is not None]
        net_by_role[kind] = {'positive_net_Ah': sum(max(q,0) for q in net),
                             'negative_net_Ah_absolute': sum(max(-q,0) for q in net),
                             'absolute_of_run_net_Ah': sum(abs(q) for q in net),
                             'missing_or_quarantined_run_count': sum(not s['analysis_valid'] or s['net_recorded_charge_Ah'] is None for s in rows),
                             'is_exact_integral_of_absolute_current': False}
    counts = {kind: sum(i['kind'] == kind for i in intervals) for kind in ('unobserved_gap','inter_sample_transition_interval','recorded_interval_overlap')}
    totals = {kind: sum(i['duration_s'] for i in intervals if i['kind'] == kind) for kind in counts}
    same_inputs = all(sha(paths[k]) == h for k,h in hashes.items())
    reasons = ['Programmed cutoffs and exact controller state are not reconstructed from recorded run summaries',
               'Instrument timezone/DST is unknown; naive dates are not UTC or guaranteed elapsed calendar hours',
               'Run net charge totals are not a complete absolute-current throughput integral']
    if not complete:
        reasons.append('Complete exact native origin audit is unavailable')
    if counts['unobserved_gap']:
        reasons.append(f"Unobserved gaps remain: {counts['unobserved_gap']}; never inferred as rest")
    if counts['recorded_interval_overlap']:
        reasons.append(f"Recorded interval overlaps need review: {counts['recorded_interval_overlap']}")
    invalid = [s for s in segments if not s['analysis_valid']]
    if invalid:
        reasons.append(f'Quarantined or invalid runs retained: {len(invalid)}')
    if not same_inputs:
        reasons.append('An input file changed during compilation')
    write_csv(output/'ordered_segments.csv', segments)
    (output/'ordered_segments.json').write_text(json.dumps({'clock_basis': 'naive instrument local clock; timezone unknown',
        'recorded_exposure_only': True, 'segments': segments}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output/'coverage_intervals.csv', intervals)
    write_csv(output/'source_interval_annotations.csv', [r for r in timeline if r['row_type'] != 'run'])
    report = {'status': 'recorded_history_compiled' if same_inputs else 'inputs_changed_during_compilation',
              'compiler_completed': same_inputs, 'clock_origin_audit_complete': complete, 'clock_checks': checks,
              'source_sha256': hashes, 'source_files': {k: str(p.resolve()) for k,p in paths.items()},
              'input_hashes_unchanged': same_inputs, 'compiler_source_sha256': sha(__file__),
              'source_common_sha256': sha(root/'work/preprocess/common.py'), 'binary_only_sources_parsed': parsed_binary,
              'segment_count': len(segments), 'quarantined_runs': len(invalid),
              'quarantined_raw_rows': sum(s['row_end_exclusive']-s['row_start'] for s in invalid),
              'unlocated_runs': sum(not s['absolute_time_known'] for s in segments),
              'interval_counts': counts, 'interval_durations_s': totals,
              'longest_unobserved_intervals': sorted((i for i in intervals if i['kind'] == 'unobserved_gap'), key=lambda i:i['duration_s'], reverse=True)[:20],
              'net_charge_by_exposure_role_Ah': net_by_role, 'full_history_replay_admissible': False,
              'exact_control_protocol_admissible': False, 'admissibility_reasons': reasons,
              'all_unobserved_intervals_inferred_operation': None, 'timezone_identified': False,
              'physical_model_validation_complete': False, 'AWS_started': False}
    report['by_cell'] = {}
    for cell in sorted({s['cell'] for s in segments}):
        cell_segments = [s for s in segments if s['cell'] == cell]
        located = [s for s in cell_segments if s['analysis_valid'] and s['absolute_time_known']]
        cell_intervals = [i for i in intervals if i['cell'] == cell]
        report['by_cell'][cell] = {'segments': len(cell_segments),
                                  'first_observed_naive': min(s['absolute_start_naive'] for s in located) if located else None,
                                  'last_observed_naive': max(s['absolute_end_naive'] for s in located) if located else None,
                                  'unobserved_gap_count': sum(i['kind'] == 'unobserved_gap' for i in cell_intervals),
                                  'unobserved_gap_sum_s': sum(i['duration_s'] for i in cell_intervals if i['kind'] == 'unobserved_gap'),
                                  'inter_sample_transition_sum_s': sum(i['duration_s'] for i in cell_intervals if i['kind'] == 'inter_sample_transition_interval'),
                                  'overlap_count': sum(i['kind'] == 'recorded_interval_overlap' for i in cell_intervals),
                                  'binary_only_RPT_runs': sum(s['source_kind'] == 'binary_only_MPR_run' for s in cell_segments)}
    report['output_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'report.json'}
    (output/'compiler_source_frozen.py').write_bytes(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--timeline', type=Path, default=root/'work/preprocess/out/timeline.csv')
    parser.add_argument('--clock-dir', type=Path, default=root/'work/validation_proposal_20261103/all_rpt_clock_origins_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', dest='require_complete', action='store_true', default=True)
    parser.add_argument('--allow-incomplete', dest='require_complete', action='store_false', help='Diagnostic output only; never grants replay admission')
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        report = compile_history(args.timeline.resolve(), args.clock_dir.resolve(), args.output.resolve(), args.require_complete)
    except Exception as error:
        if not output_existed and args.output.exists() and not (args.output/'report.json').exists():
            (args.output/'failure.json').write_text(json.dumps({'status': 'compilation_failed', 'error': f'{type(error).__name__}: {error}',
                'full_history_replay_admissible': False}), encoding='utf-8')
        raise
    print(json.dumps({k: report.get(k) for k in ('status','compiler_completed','segment_count','quarantined_raw_rows','interval_counts','interval_durations_s','full_history_replay_admissible')}))
    return 0 if report['compiler_completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
