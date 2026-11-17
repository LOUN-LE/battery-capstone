"""Input gates for ageing campaigns. No PDE imports or inferred exposures."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from measured_protocol_replay import validate_protocol


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(ref, base):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Immutable source requires path and sha256')
    if not isinstance(ref['path'], str) or not ref['path'].strip():
        raise ValueError('Source path required')
    path = (Path(base)/ref['path']).resolve()
    if not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64 or digest(path) != ref['sha256']:
        raise ValueError('Source sha256 mismatch: '+str(path))
    return path


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite numeric')
    if positive and value <= 0:
        raise ValueError(name+' must be positive')
    return value


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def check_history(ref, base):
    path = checked_path(ref, base)
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    counts = report.get('interval_counts', {})
    # Inspect evidence, not a caller's reviewed/pass flag. Current real history
    # contains 163 unknown gaps and therefore cannot reach the PDE backend.
    tests = [report.get('status') == 'recorded_history_compiled',
             report.get('compiler_completed') is True,
             report.get('clock_origin_audit_complete') is True,
             bool(report.get('clock_checks')) and all(v is True for v in report['clock_checks'].values()),
             report.get('input_hashes_unchanged') is True,
             report.get('full_history_replay_admissible') is True,
             report.get('exact_control_protocol_admissible') is True,
             counts.get('unobserved_gap') == 0, counts.get('recorded_interval_overlap') == 0,
             report.get('quarantined_runs') == 0, report.get('unlocated_runs') == 0,
             report.get('quarantined_raw_rows') == 0]
    if not all(tests):
        raise ValueError('Measured history admission refused: gaps='+str(counts.get('unobserved_gap'))+
                         '; exact controls, complete clocks and zero unknown exposure required')
    sources, hashes = report.get('source_files', {}), report.get('source_sha256', {})
    if not sources or set(sources) != set(hashes):
        raise ValueError('Measured history admission requires immutable underlying sources')
    tracked = {str(path): digest(path)}
    for key, source in sources.items():
        source_path = checked_path({'path': source, 'sha256': hashes[key]}, path.parent)
        tracked[str(source_path)] = hashes[key]
    return tracked


def validate_campaign(campaign, base):
    allowed = {'schema_version', 'data_kind', 'cell_id', 'condition_id', 'protocol',
               'measurement_steps', 'targets', 'history_admission', 'source_identity'}
    if (not isinstance(campaign, dict) or set(campaign)-allowed
        or type(campaign.get('schema_version')) is not int or campaign['schema_version'] != 1):
        raise ValueError('Unknown campaign fields or unsupported schema')
    c = copy.deepcopy(campaign)
    if c.get('data_kind') not in ('synthetic', 'measured'):
        raise ValueError('Explicit measured or synthetic data_kind required')
    if any(not isinstance(c.get(k), str) or not c[k].strip() for k in ('cell_id', 'condition_id')):
        raise ValueError('Explicit cell and condition identity required')
    p = validate_protocol(c.get('protocol'))
    if c['data_kind'] == 'measured' and any(s['role'] == 'diagnostic' for s in p['steps']):
        raise ValueError('Measured actual history cannot discard diagnostic exposure; every recorded step must propagate')
    measurements = c.get('measurement_steps')
    if not isinstance(measurements, list) or len(measurements) < 2:
        raise ValueError('At least two ordered RPT measurements required')
    indices, ids, signatures = [], [], []
    for m in measurements:
        if not isinstance(m, dict) or set(m)-{'rpt_id', 'step_index', 'kind', 'cutoff_V', 'conditioning_steps'}:
            raise ValueError('Unknown measurement fields')
        index = m.get('step_index')
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(p['steps']):
            raise ValueError('Measurement step index outside ordered protocol')
        rpt = m.get('rpt_id')
        if not isinstance(rpt, str) or not rpt.strip():
            raise ValueError('RPT id must be a nonempty string')
        s = p['steps'][index]
        if s['role'] != 'rpt' or s['operation'] != 'current' or s.get('value', 0) <= 0:
            raise ValueError('Measurement must be an actual positive-current CC discharge RPT')
        kind = m.get('kind')
        if kind == 'timed_software_probe':
            if c['data_kind'] != 'synthetic' or s['termination'] or s['expected_termination'] != 'final time':
                raise ValueError('Timed software probe is synthetic only; it is not full capacity/SOH')
        elif kind == 'full_discharge_capacity':
            cutoff = number(m.get('cutoff_V'), 'RPT discharge cutoff', positive=True)
            if s['termination'] != [f'{cutoff:g} V'] or s['expected_termination'] != f'event: Voltage < {cutoff:g} [V] [experiment]':
                raise ValueError('Full capacity requires exact reviewed discharge voltage event')
            ci = m.get('conditioning_steps')
            if ci != [index-3, index-2, index-1] or index < 3:
                raise ValueError('Full RPT requires explicit consecutive charge/CV/rest conditioning')
            group = [p['steps'][i] for i in ci]
            if (any(g['role'] != 'rpt' or g['temperature_K'] != s['temperature_K'] for g in group)
                or group[0]['operation'] != 'current' or group[0].get('value', 0) >= 0
                or group[1]['operation'] != 'voltage' or group[2]['operation'] != 'rest'
                or not group[0]['termination'] or not group[1]['termination']):
                raise ValueError('RPT charge/CV/rest conditioning incomplete or mismatched')
            cv_voltage = group[1]['value']
            if (cutoff >= cv_voltage or group[0]['termination'] != [f'{cv_voltage:g} V']
                or group[0]['expected_termination'] != f'event: Voltage > {cv_voltage:g} [V] [experiment]'
                or len(group[1]['termination']) != 1 or not group[1]['termination'][0].endswith(' A')
                or group[2]['termination'] or group[2]['expected_termination'] != 'final time'):
                raise ValueError('RPT CC/CV voltage, current endpoint and timed rest must be coherent')
            try:
                cv_current = float(group[1]['termination'][0][:-2])
            except ValueError:
                raise ValueError('Positive explicit CV current cutoff required') from None
            if (not math.isfinite(cv_current) or cv_current <= 0
                or group[1]['expected_termination'] != f'event: abs(Current [A]) < {cv_current:g} [A] [experiment]'):
                raise ValueError('Exact positive CV current event required')
            signatures.append((s['value'], s['temperature_K'], cutoff,
                               group[0]['value'], tuple(group[0]['termination']),
                               cv_voltage, tuple(group[1]['termination']), group[2]['duration_s']))
        else:
            raise ValueError('Explicit full_discharge_capacity or timed_software_probe required')
        indices.append(index); ids.append(rpt)
    if indices != sorted(set(indices)) or len(ids) != len(set(ids)):
        raise ValueError('Unique chronological measurement indices and RPT ids required')
    if len({m['kind'] for m in measurements}) != 1:
        raise ValueError('All normalized measurements must use the same full-capacity or timed-probe quantity')
    if signatures and any(s != signatures[0] for s in signatures):
        raise ValueError('SOH denominator requires the same reference measurement conditions')
    tracked = check_history(c.get('history_admission'), base) if c['data_kind'] == 'measured' else {}
    if c['data_kind'] == 'measured':
        history_path = checked_path(c['history_admission'], base)
        history = json.loads(history_path.read_text(encoding='utf-8-sig'))
        binding = history.get('admitted_protocols', {}).get(c['cell_id'], {})
        if (binding.get('protocol_sha256') != canonical_sha256(p)
            or binding.get('measurement_steps_sha256') != canonical_sha256(measurements)
            or binding.get('ordered_source_refs') != [s['source_ref'] for s in p['steps']]):
            raise ValueError('Measured history admission is not bound to this cell and exact ordered protocol')
    path = checked_path(c.get('targets'), base)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['rpt_id', 'Qref_Ah']:
            raise ValueError('Target CSV must be native rpt_id,Qref_Ah in Ah, with no precomputed SOH')
        rows = list(reader)
    if [r['rpt_id'] for r in rows] != ids:
        raise ValueError('Target rows must exactly match chronological measurement RPT ids')
    capacities = []
    for row in rows:
        try:
            q = float(row['Qref_Ah'])
        except (ValueError, TypeError):
            raise ValueError('Target capacity must be finite positive Ah') from None
        capacities.append(number(q, 'Target capacity', positive=True))
    c['_targets_rows'] = [{'rpt_id': r, 'Qref_Ah': q, 'SOH_percent': 100*q/capacities[0]}
                          for r, q in zip(ids, capacities)]
    if c['data_kind'] == 'measured':
        from physical_campaign_sources import verify_native_targets
        native = verify_native_targets(c, base, c['_targets_rows'])
        tracked.update(native['source_hashes'])
        c['_native_source_sha256'] = native['source_identity_sha256']
        c['_native_source_member_path'] = native['source_member_path']
        c['_native_Q0_Ah'] = native['native_Q0_Ah']
        c['_native_curve_coverage'] = {k: native[k] for k in
            ('native_RPT_count', 'evaluated_RPT_ids', 'full_native_curve_evaluated')}
        for m in measurements:
            s = p['steps'][m['step_index']]
            charge, cv, rest = [p['steps'][i] for i in m['conditioning_steps']]
            if (s['value'] != .5 or s['temperature_K'] != 298.15 or m['cutoff_V'] != 2.5
                or charge['value'] != -1.5 or cv['value'] != 4.2
                or cv['termination'] != ['0.05 A'] or rest['duration_s'] != 7230):
                raise ValueError('Measured native C/10 capacity requires reviewed 25C/0.5A/2.5V RPT with sourced conditioning')
    c['_target_path'] = str(path)
    c['_data_sha256'] = c['targets']['sha256']
    c['_campaign_source_hashes'] = {**tracked, str(path): digest(path)}
    return c


def validate_split(calibration, heldouts):
    cells = {calibration['cell_id']}
    hashes = {calibration['targets']['sha256']}
    native_hashes = {calibration.get('_native_source_sha256')}-{None}
    native_members = {calibration.get('_native_source_member_path')}-{None}
    for heldout in heldouts:
        if heldout['cell_id'] in cells or heldout['targets']['sha256'] in hashes:
            raise ValueError('Fit/heldout leakage: repeated cell or raw target source hash')
        if heldout['data_kind'] != calibration['data_kind']:
            raise ValueError('Cannot combine measured and synthetic validation evidence')
        if (heldout.get('_native_source_sha256') in native_hashes
            or heldout.get('_native_source_member_path') in native_members):
            raise ValueError('Fit/heldout leakage: underlying native summary or archive member repeated')
        cells.add(heldout['cell_id']); hashes.add(heldout['targets']['sha256'])
        if heldout.get('_native_source_sha256'):
            native_hashes.add(heldout['_native_source_sha256'])
            native_members.add(heldout['_native_source_member_path'])
    return True
