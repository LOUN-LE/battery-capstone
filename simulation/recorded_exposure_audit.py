"""Independent interval/charge audit of lossless reconstructed native exposure.

This is not a controller compiler or permission to run a lifetime calculation.
Only eligible observed intervals are integrated; rest-entry stale current,
unknown gaps, clock overlaps and unobserved transitions are never interpolated.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds_us(start, end):
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    return delta.days * 86400000000 + delta.seconds * 1000000 + delta.microseconds


def integrate_window(t, current_mA, eligible, start, end):
    """Integrate each wholly contained eligible linear interval.

    Raw sign is retained (positive means native charging). A zero crossing is
    integrated analytically so positive/negative throughput is not cancelled.
    Noneligible time remains unknown, never zero current.
    """
    import numpy as np
    t = np.asarray(t, dtype=float)
    current = np.asarray(current_mA, dtype=float)
    eligible = np.asarray(eligible)
    if (t.ndim != 1 or current.shape != t.shape or eligible.dtype != np.bool_
            or eligible.shape != (max(0, len(t)-1),)):
        raise ValueError('Paired current/time and boolean interval mask required')
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError('Finite ordered clock window required')
    dt = np.diff(t)
    if np.any(~np.isfinite(t)) or np.any(eligible & ((dt <= 0) | (dt > 60.00001))):
        raise ValueError('Eligible intervals require monotonic dense native clock')
    if np.any(eligible & (~np.isfinite(current[:-1]) | ~np.isfinite(current[1:]))):
        raise ValueError('Eligible current must be finite')
    left = t[:-1]
    right = t[1:]
    use = eligible & (left >= start) & (right <= end) & (right > left)
    width = right[use] - left[use]
    slope = (current[1:][use] - current[:-1][use]) / dt[use]
    a = current[:-1][use] + slope * (left[use]-t[:-1][use])
    b = current[:-1][use] + slope * (right[use]-t[:-1][use])
    signed = (a+b)*width*.5
    positive = np.maximum(signed, 0.)
    negative = np.maximum(-signed, 0.)
    cross = a*b < 0
    positive[cross] = width[cross] * np.maximum(a[cross], b[cross])**2 / (2*np.abs(b[cross]-a[cross]))
    negative[cross] = width[cross] * np.minimum(a[cross], b[cross])**2 / (2*np.abs(b[cross]-a[cross]))
    return dict(valid_duration_s=float(math.fsum(width)),
                signed_native_current_Ah=float(math.fsum(signed)/3600000),
                observed_charge_Ah=float(math.fsum(positive)/3600000),
                observed_discharge_Ah=float(math.fsum(negative)/3600000),
                eligible_interval_count=int(use.sum()))


def audit(path):
    import numpy as np
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if payload.get('schema_version') != 2 or payload.get('data_kind') != 'recorded_exposure_reconstruction':
        raise ValueError('Lossless raw exposure schema2 required')
    if payload.get('pde_replay_admissible') is not False:
        raise ValueError('Raw reconstruction must not claim lifetime admission')
    steps = payload['cell']['steps']
    sources = {s['source_id']: s for s in payload['cell']['source_files']}
    grouped = defaultdict(list)
    ledger = Counter()
    unknown_by_kind = Counter()
    positive_clock_end = None
    for s in steps:
        us = seconds_us(s['absolute_start_naive'], s['absolute_end_naive'])
        if us != s['duration_us'] or us < 0:
            raise ValueError('Native integer clock does not match stored duration')
        if abs(s['duration_s']*1e6-us) > 2.01:
            raise ValueError('Raw float duration and absolute clock disagree')
        if us:
            begin = s['absolute_start_naive']
            if positive_clock_end is not None and begin != positive_clock_end:
                raise ValueError('Positive native clock has an overlap or missing interval')
            positive_clock_end = s['absolute_end_naive']
        ledger['total_duration_us'] += us
        if s['operation'] == 'rest':
            if s.get('rest_current_A') != 0 or s.get('raw_current_interpolation_permitted') is not False:
                raise ValueError('Rest current interpolation is forbidden')
            ledger['recorded_rest_controller_us'] += us
        elif s['operation'] == 'recorded_current_drive':
            grouped[s['source_id']].append(s)
        elif s['operation'] is None:
            ledger['unresolved_us'] += us
            unknown_by_kind[s['kind']] += us
        else:
            raise ValueError('Unknown reconstructed operation')
    integrals = Counter()
    file_receipts = []
    for source_id, source_steps in grouped.items():
        source = sources[source_id]
        raw_path = (path.parent/source['arrays_file']).resolve()
        if not raw_path.is_relative_to(path.parent):
            raise ValueError('Raw arrays must stay in the reconstructed artifact directory')
        if sha(raw_path) != source['arrays_sha256']:
            raise ValueError('Raw array hash changed')
        with np.load(raw_path, allow_pickle=False) as archive:
            t, I, mask, mode, ns = [archive[k] for k in
                ('t_s','I_mA','current_drive_eligible_interval','mode','Ns')]
        if (len(t) != source['rows'] or mask.shape != (max(0,len(t)-1),)
                or mask.dtype != np.bool_):
            raise ValueError('Lossless array shape or mask mismatch')
        if np.any(mask & ((mode[:-1] != mode[1:]) | ~np.isin(mode[:-1],[1,2]) | (ns[:-1] != ns[1:]))):
            raise ValueError('Current mask crosses a recorded rest/mode/Ns boundary')
        charge = discharge = 0.
        for s in source_steps:
            if source['native_sha256'] != s['raw_source_sha256'] or source['arrays_file'] != s['arrays_file']:
                raise ValueError('Step source binding mismatch')
            a, b = s['row_start'], s['row_end_exclusive']
            if not 0 <= a < b <= len(t):
                raise ValueError('Raw row range outside source')
            # Only wholly contained intervals, matching the reconstruction's
            # conservative policy. Partial boundary intervals stay unresolved.
            result = integrate_window(t[a:b], I[a:b], mask[a:b-1], s['start_s'], s['end_s'])
            valid_us = round(result['valid_duration_s']*1e6)
            if valid_us < 0 or valid_us > s['duration_us']+2:
                raise ValueError('Eligible waveform exceeds declared source window')
            valid_us = min(valid_us, s['duration_us'])
            ledger['eligible_recorded_current_us'] += valid_us
            missing = s['duration_us']-valid_us
            ledger['unresolved_us'] += missing
            unknown_by_kind['noneligible_current_sample_interval'] += missing
            for key in ('signed_native_current_Ah','observed_charge_Ah','observed_discharge_Ah'):
                integrals[key] += result[key]
            charge += result['observed_charge_Ah']
            discharge += result['observed_discharge_Ah']
        file_receipts.append(dict(source_id=source_id,arrays_sha256=source['arrays_sha256'],
                                  eligible_observed_charge_Ah=charge,eligible_observed_discharge_Ah=discharge))
    accounted = sum(ledger[k] for k in ('recorded_rest_controller_us','eligible_recorded_current_us','unresolved_us'))
    if accounted != ledger['total_duration_us']:
        raise ValueError('Independent classified time ledger failed')
    return dict(status='independent_raw_exposure_audit_passed',cell_id=payload['cell']['cell_id'],
                inputs_sha256=sha(path),auditor_sha256=sha(__file__),time_ledger_us=dict(ledger),
                unresolved_by_kind_us=dict(unknown_by_kind),eligible_interval_integrals_Ah=dict(integrals),
                source_array_receipts=file_receipts,integer_time_accounting_pass=True,
                physical_validation_complete=False,pde_replay_admissible=False,
                current_integral_scope='eligible observed linear intervals only; excludes unknown current and recorded rest-controller intervals',
                clipping_policy='wholly contained eligible intervals only; partial boundary intervals remain unresolved',
                maximum_eligible_sample_gap_s=60.,
                rest_scope='recorded rest controller; sparse samples are not continuous measured current',
                unobserved_intervals_remain_unknown=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing audit result must be preserved')
    result = audit(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_array_receipts'},indent=2))
