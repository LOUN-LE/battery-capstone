"""Read-only dynamic voltage-shape observations on the recorded sample clock.

No interpolation, time shifting, state initialisation, fitting, resistance inference,
or source-row filtering. Each request stays inside one recorded execution run.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

LOADED_TIMES_S = (1., 10., 30.)
REST_TIMES_S = (1., 10., 30., 60., 600., 1800.)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class RawTrace:
    arrays: Mapping[str, np.ndarray]
    temperature_quality_mask: np.ndarray | None = None

    def __post_init__(self):
        required = ('raw_source_row', 'run_index', 'time_s', 'Ns', 'I_mA', 'V', 'mode')
        missing = [k for k in required if k not in self.arrays]
        if missing:
            raise ValueError('Missing recorded channel(s): ' + ', '.join(missing))
        n = len(self.arrays['time_s'])
        if not n or any(np.asarray(self.arrays[k]).ndim != 1 or len(self.arrays[k]) != n for k in required):
            raise ValueError('Required recorded arrays must be nonempty, one-dimensional and equally sized')
        time = np.asarray(self.arrays['time_s'], dtype=float)
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise ValueError('Nonfinite or nonincreasing raw clock; preserve source and resolve before observation search')
        for key in ('surface_temperature_C', 'dq_mAh', 'Q_minus_Qo_mAh'):
            if key in self.arrays and (np.asarray(self.arrays[key]).ndim != 1 or len(self.arrays[key]) != n):
                raise ValueError('Optional channel shape differs: ' + key)
        if self.temperature_quality_mask is not None and np.asarray(self.temperature_quality_mask).shape != (n,):
            raise ValueError('Temperature quality mask must have one flag per raw row')

    def validate_run(self, run):
        a, b = int(run['row_start']), int(run['row_end_exclusive'])
        if not 0 <= a < b <= len(self.arrays['time_s']):
            raise ValueError('Run row range is invalid')
        ids = np.asarray(self.arrays['run_index'][a:b])
        if np.any(ids != int(run['run_index'])):
            raise ValueError('Run spans different recorded executions; repeated Ns must remain separate')
        if np.any(self.arrays['Ns'][a:b] != int(run['Ns'])):
            raise ValueError('Run Ns disagrees with raw samples')
        if a and self.arrays['run_index'][a-1] == ids[0]:
            raise ValueError('Run begins inside an existing execution')
        if b < len(self.arrays['time_s']) and self.arrays['run_index'][b] == ids[0]:
            raise ValueError('Run ends inside an existing execution')
        return a, b


def point_identity(run):
    return {k: run.get(k) for k in ('file_id', 'part', 'purpose', 'run_index', 'Ns', 'role')}


def first_reached_sample(trace: RawTrace, run, requested_elapsed_s: float):
    """Return the first recorded sample at/after tau, without leaving this run."""
    if isinstance(requested_elapsed_s, bool) or not math.isfinite(requested_elapsed_s) or requested_elapsed_s < 0:
        raise ValueError('Requested elapsed time must be finite and nonnegative')
    a, b = trace.validate_run(run)
    time = trace.arrays['time_s']
    onset = float(time[a])
    base = {**point_identity(run), 'requested_elapsed_s': float(requested_elapsed_s),
            'onset_source_row': int(trace.arrays['raw_source_row'][a]), 'onset_time_s': onset,
            'run_row_start': a, 'run_row_end_exclusive': b,
            'run_first_to_last_duration_s': float(time[b-1]-onset),
            'run_warning_flags': run.get('warning_flags', []),
            'sample_status': 'missing', 'missing_reason': 'requested_time_after_last_recorded_sample_in_this_run',
            'raw_source_row': None, 'raw_array_index': None, 'recorded_time_s': None,
            'actual_elapsed_s': None, 'overshoot_from_request_s': None,
            'I_mA': None, 'V': None, 'surface_temperature_C': None,
            'recorded_mode': None, 'point_quality_flags': []}
    offset = int(np.searchsorted(time[a:b], onset + requested_elapsed_s, side='left'))
    if offset >= b-a:
        return base
    index = a + offset
    flags = []
    if min(index-a, b-1-index) <= 2:
        flags.append('within_two_rows_of_recorded_run_boundary')
    if trace.temperature_quality_mask is not None and bool(trace.temperature_quality_mask[index]):
        flags.append('temperature_adjacent_to_gt2C_sample_jump_diagnostic_mask')
    if 'record_anomaly_mask' in trace.arrays and bool(trace.arrays['record_anomaly_mask'][index]):
        flags.append('preserved_record_anomaly')
    if 'surface_temperature_C' in trace.arrays and finite(trace.arrays['surface_temperature_C'][index]) is None:
        flags.append('nonfinite_recorded_temperature')
    observed_v, observed_i = finite(trace.arrays['V'][index]), finite(trace.arrays['I_mA'][index])
    status = 'observed' if observed_v is not None and observed_i is not None else 'nonfinite_sample'
    return {**base, 'sample_status': status, 'missing_reason': None if status == 'observed' else 'nonfinite_voltage_or_current',
            'raw_source_row': int(trace.arrays['raw_source_row'][index]), 'raw_array_index': index,
            'recorded_time_s': float(time[index]), 'actual_elapsed_s': float(time[index]-onset),
            'overshoot_from_request_s': float(time[index]-onset-requested_elapsed_s),
            'I_mA': observed_i, 'V': observed_v,
            'surface_temperature_C': finite(trace.arrays['surface_temperature_C'][index]) if 'surface_temperature_C' in trace.arrays else None,
            'recorded_mode': int(trace.arrays['mode'][index]), 'point_quality_flags': flags}


def voltage_shapes(trace: RawTrace, run, requested_times, kind: str, preceding_load=None):
    """Voltage changes relative to the first requested sample (normally >=1 s)."""
    times = tuple(requested_times)
    if not times or times[0] != 1. or any(y <= x for x, y in zip(times, times[1:])):
        raise ValueError('Shape times must increase strictly from the explicit 1-second reference')
    points = [first_reached_sample(trace, run, tau) for tau in times]
    reference = points[0]
    rows = []
    for p in points:
        good = p['sample_status'] == 'observed' and reference['sample_status'] == 'observed'
        reason = None if good else ('reference_sample_unavailable' if reference['sample_status'] != 'observed' else 'target_sample_unavailable')
        same_sample = good and p['requested_elapsed_s'] != 1. and p['raw_source_row'] == reference['raw_source_row']
        if same_sample:
            good, reason = False, 'reference_and_target_are_same_recorded_sample_no_resolved_time_difference'
        rows.append({**p, 'observation_kind': kind, 'preceding_load_run_index': preceding_load,
                     'reference_requested_elapsed_s': 1., 'reference_raw_source_row': reference['raw_source_row'],
                     'reference_recorded_time_s': reference['recorded_time_s'], 'reference_actual_elapsed_s': reference['actual_elapsed_s'],
                     'reference_I_mA': reference['I_mA'], 'reference_V': reference['V'],
                     'reference_quality_flags': reference['point_quality_flags'],
                     'shape_status': 'observed' if good else 'missing', 'shape_missing_reason': reason,
                     'delta_voltage_mV_from_1s_selected_sample': 1000*(p['V']-reference['V']) if good else None,
                     'actual_time_since_reference_s': p['recorded_time_s']-reference['recorded_time_s'] if good else None,
                     'is_reference_point': p['requested_elapsed_s'] == 1.,
                     'informative_nonreference_shape': good and p['requested_elapsed_s'] != 1.,
                     'is_resistance': False, 'raw_clock_shifted': False, 'raw_interpolated': False})
    return rows


def c_half_rate_cutoff(trace: RawTrace, run, cutoff_voltage: float, direction: str):
    a, b = trace.validate_run(run)
    time = np.asarray(trace.arrays['time_s'][a:b], dtype=float)
    current = np.asarray(trace.arrays['I_mA'][a:b], dtype=float)
    voltage = np.asarray(trace.arrays['V'][a:b], dtype=float)
    if direction not in ('discharge', 'charge') or not math.isfinite(cutoff_voltage):
        raise ValueError('Finite cutoff and charge/discharge direction required')
    finite_channels = np.all(np.isfinite(current)) and np.all(np.isfinite(voltage))
    hit = np.flatnonzero(voltage <= cutoff_voltage if direction == 'discharge' else voltage >= cutoff_voltage)
    crossing = int(hit[0]) if len(hit) else None
    return {**point_identity(run), 'observation_kind': 'C_half_rate_raw_run_cutoff_diagnostic',
            'direction': direction, 'onset_source_row': int(trace.arrays['raw_source_row'][a]),
            'last_source_row': int(trace.arrays['raw_source_row'][b-1]),
            'first_recorded_time_s': float(time[0]), 'last_recorded_time_s': float(time[-1]),
            'first_to_last_duration_s': float(time[-1]-time[0]),
            'first_voltage_V': finite(voltage[0]), 'last_voltage_V': finite(voltage[-1]),
            'cutoff_voltage_V': cutoff_voltage, 'last_voltage_minus_cutoff_mV': finite(1000*(voltage[-1]-cutoff_voltage)),
            'cutoff_crossing_in_recorded_samples': crossing is not None,
            'first_cutoff_source_row': int(trace.arrays['raw_source_row'][a+crossing]) if crossing is not None else None,
            'first_cutoff_elapsed_s': float(time[crossing]-time[0]) if crossing is not None else None,
            'signed_Ah_trapezoid_within_run': float(np.trapezoid(current, time)/3_600_000) if finite_channels else None,
            'absolute_Ah_trapezoid_within_run': float(np.trapezoid(np.abs(current), time)/3_600_000) if finite_channels else None,
            'instrument_dq_Ah_sum': finite(np.sum(trace.arrays['dq_mAh'][a:b])/1000) if 'dq_mAh' in trace.arrays else None,
            'instrument_Q_minus_Qo_change_Ah': finite((trace.arrays['Q_minus_Qo_mAh'][b-1]-trace.arrays['Q_minus_Qo_mAh'][a])/1000) if 'Q_minus_Qo_mAh' in trace.arrays else None,
            'following_first_sample_time_s': float(trace.arrays['time_s'][b]) if b < len(trace.arrays['time_s']) else None,
            'following_sample_gap_s': float(trace.arrays['time_s'][b]-time[-1]) if b < len(trace.arrays['time_s']) else None,
            'run_warning_flags': run.get('warning_flags', []), 'not_SOHRPT_Q0': True,
            'interpretation': 'Recorded run endpoint and absolute charge only; no interpolated cutoff, endpoint enforcement, or model event success inferred.'}


def parse_runs(path):
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8-sig', newline='')))
    for r in rows:
        for key in ('run_index', 'Ns', 'row_start', 'row_end_exclusive'):
            r[key] = int(r[key])
        r['warning_flags'] = json.loads(r.get('warning_flags') or '[]')
    return rows


def select_loaded_run(part, run):
    if run['control_type'] != 'CC':
        return False
    if part in ('long_02', 'long_03', 'long_04'):
        return run['Ns'] == 5
    if part in ('short_02', 'short_03'):
        return run['Ns'] >= 5
    return False


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def extract(dataset_root, output):
    root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError('Output must be outside the preserved source dataset')
    if (output/'report.json').exists():
        raise ValueError('Preserve existing output: choose a fresh output folder')
    report = read_json(root/'report.json')
    if report.get('only_cell') != 'C' or report.get('rows_removed') != 0:
        raise ValueError('This operator campaign accepts the preserved C-only raw dataset')
    artifacts = read_json(root/'artifact_sha256.json')
    guarded = {Path(p): expected for p, expected in read_json(root/'source_sha256_guard.json')['files'].items()}
    for p in (root/'report.json', root/'source_sha256_guard.json'):
        guarded[p] = artifacts[str(p.relative_to(root))]
    source_locations = []
    for item in report['files']:
        raw = Path(item['raw_npz_path']).resolve()
        if root not in raw.parents:
            raise ValueError('Raw path escapes source dataset')
        folder = raw.parent
        for p in (raw, folder/'runs.csv', folder/'receipt.json', folder/'quality_flags.npz'):
            guarded[p] = artifacts[str(p.relative_to(root))]
        source_locations.append((item, raw, folder))
    for p, expected in guarded.items():
        if sha(p) != expected:
            raise ValueError('Source or decoder hash differs before extraction: ' + str(p))
    loaded, rest, cutoffs, files = [], [], [], []
    for item, raw, folder in source_locations:
        with np.load(raw, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != 'original_MPR_structured_rows'}
        with np.load(folder/'quality_flags.npz', allow_pickle=False) as z:
            if not np.array_equal(z['raw_source_row'], arrays['raw_source_row']):
                raise ValueError('Quality mask row identity differs')
            quality = z['temperature_adjacent_to_gt2C_sample_jump']
        trace = RawTrace(arrays, quality)
        runs = parse_runs(folder/'runs.csv')
        loaded_count, rest_count = 0, 0
        for k, run in enumerate(runs):
            trace.validate_run(run)
            if select_loaded_run(item['part'], run):
                kind = 'continuous_C_half_discharge_shape' if item['part']=='long_02' else 'loaded_step_shape'
                loaded.extend(voltage_shapes(trace, run, LOADED_TIMES_S, kind))
                loaded_count += 1
                if k+1 < len(runs) and runs[k+1]['control_type']=='rest':
                    rest.extend(voltage_shapes(trace, runs[k+1], REST_TIMES_S, 'immediate_post_load_rest_shape', run['run_index']))
                    rest_count += 1
            if item['part']=='long_02' and run['Ns'] in (5, 8) and run['control_type']=='CC':
                cutoffs.append(c_half_rate_cutoff(trace, run, 2.5 if run['Ns']==5 else 4.2,
                                                'discharge' if run['Ns']==5 else 'charge'))
        files.append(dict(file_id=item['file_id'], part=item['part'], rows=len(arrays['time_s']),
                          loaded_runs=loaded_count, immediate_post_load_rest_runs=rest_count,
                          raw_NPZ_sha256=guarded[raw], no_SOC_initialisation_performed=True))
    for p, expected in guarded.items():
        if sha(p) != expected:
            raise ValueError('Source or decoder changed during extraction: ' + str(p))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='C_BoL_dynamic_shapes_extracted', only_cell='C', files=files,
                  loaded_shape_records=loaded, rest_shape_records=rest, C_half_rate_cutoff_records=cutoffs,
                  observed_loaded_shape_points=sum(r['shape_status']=='observed' for r in loaded),
                  missing_loaded_shape_points=sum(r['shape_status']!='observed' for r in loaded),
                  observed_rest_shape_points=sum(r['shape_status']=='observed' for r in rest),
                  missing_rest_shape_points=sum(r['shape_status']!='observed' for r in rest),
                  informative_loaded_nonreference_shapes=sum(r['informative_nonreference_shape'] for r in loaded),
                  informative_rest_nonreference_shapes=sum(r['informative_nonreference_shape'] for r in rest),
                  method='first raw sample at/after requested tau within one recorded run; delta V referenced to selected >=1s sample',
                  no_interpolation=True, no_time_shift=True, no_raw_rows_removed=True,
                  author_R0p1_or_instantaneous_R0_calculated=False, no_model_or_SOC_reset=True,
                  physical_validation_complete=False, all_original_and_decoder_sources_unchanged=True,
                  source_sha256={str(p): expected for p, expected in guarded.items()},
                  operator_sha256=sha(Path(__file__)))
    write_csv(output/'loaded_shapes.csv', loaded)
    write_csv(output/'post_load_rest_shapes.csv', rest)
    write_csv(output/'C_half_rate_cutoff.csv', cutoffs)
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    hashes={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='artifact_sha256.json'}
    (output/'artifact_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = extract(args.dataset_root, args.output)
    print(json.dumps({k:result[k] for k in ('status','observed_loaded_shape_points','missing_loaded_shape_points',
        'observed_rest_shape_points','missing_rest_shape_points','all_original_and_decoder_sources_unchanged')}, ensure_ascii=False))


if __name__=='__main__':
    main()
