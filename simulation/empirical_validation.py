"""Prepare measured capacity targets without inventing an EFC or an EOL.

Summary throughput is absolute charge + discharge ageing Ah. Dividing by twice
fixed Q0 is its canonical EFC definition, not an approximation to discharge EFC.
This module does not fit degradation coefficients or certify a physical model.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite {name}')
    return result


def prepare_summary(rows):
    if not rows:
        raise ValueError('Empty measured summary')
    q0 = finite(rows[0]['C/10 Capacity [mA h]'], 'Q0')
    if q0 <= 0 or finite(rows[0]['Ageing Sets'], 'initial set') != 0:
        raise ValueError('A positive measured RPT0 capacity is required')
    if finite(rows[0]['Charge Throughput [A h]'], 'initial throughput') != 0:
        raise ValueError('Throughput origin must be RPT0')
    points = []
    for row in rows:
        q = finite(row['C/10 Capacity [mA h]'], 'capacity')
        throughput = finite(row['Charge Throughput [A h]'], 'throughput')
        day = finite(row['Days of degradation'], 'day')
        cycles = finite(row['Ageing Cycles'], 'cycles')
        rpt = finite(row['Ageing Sets'], 'RPT')
        if q <= 0 or min(throughput, day, cycles, rpt) < 0:
            raise ValueError('Negative axis or nonpositive measured capacity')
        absolute_efc = throughput / (2 * q0 / 1000)
        point = {'rpt': rpt, 'ageing_cycles': cycles, 'days': day,
                 'capacity_Ah': q / 1000, 'SOH_percent': 100 * q / q0,
                 'absolute_ageing_EFC': absolute_efc,
                 'approximate_EFC': absolute_efc}  # Deprecated historical-name alias only.
        if points and any(point[k] <= points[-1][k] for k in
                          ('rpt', 'ageing_cycles', 'absolute_ageing_EFC')):
            raise ValueError('RPT, cycles and throughput must strictly increase')
        if points and day < points[-1]['days']:
            raise ValueError('Measurement days must not decrease')
        points.append(point)
    return points


def eol_bracket(points, axis):
    """Return observed first crossing interval; never extrapolate or force a crossing."""
    if not points:
        raise ValueError('Empty EOL observations')
    xs = [finite(point[axis], axis) for point in points]
    soh = [finite(point['SOH_percent'], 'SOH') for point in points]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError('EOL axis must strictly increase')
    if soh[0] <= 80:
        return {'status': 'left_censored', 'lower': None, 'upper': xs[0], 'axis': axis}
    for index in range(1, len(points)):
        if soh[index] <= 80:
            return {'status': 'observed', 'lower': xs[index-1],
                    'upper': xs[index], 'axis': axis}
    return {'status': 'right_censored', 'lower': xs[-1],
            'upper': None, 'axis': axis}


MATCH_FIELDS = ('cell_model', 'protocol_id', 'temperature_protocol_id',
                'rpt_protocol_id', 'soh_definition', 'axis_column', 'axis_definition', 'axis_origin')


def compare_curves(measured, predicted, measured_metadata, predicted_metadata, axis):
    """Interpolate predictions at every measured point; reject incomplete coverage.

    Metadata must identify reviewed protocols, not merely matching current labels.
    This produces error metrics, not a validation-pass decision or a fitted model.
    """
    for key in MATCH_FIELDS:
        value = measured_metadata.get(key)
        if value is None or value == '' or predicted_metadata.get(key) != value:
            raise ValueError(f'Missing or different comparison condition: {key}')
    # The selected column is a data key; its scientific definition is separate.
    # E.g. column='absolute_ageing_EFC', definition='Q_abs/(2*fixed_Q0)'.
    # Both must match so cycle counts cannot be relabelled as discharge-based EFC.
    if axis != measured_metadata['axis_column']:
        raise ValueError('Requested axis column differs from reviewed metadata')
    if not measured or len(predicted) < 2:
        raise ValueError('Insufficient comparison data')
    def checked(curve):
        xs = [finite(p[axis], axis) for p in curve]
        ys = [finite(p['SOH_percent'], 'SOH') for p in curve]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError('Comparison axis must strictly increase')
        return xs, ys
    mx, my = checked(measured)
    px, py = checked(predicted)
    if mx[0] < px[0] or mx[-1] > px[-1]:
        raise ValueError('Predictions do not cover all measured observations; no extrapolation')
    errors = []
    for x, y in zip(mx, my):
        index = min(max(bisect.bisect_right(px, x)-1, 0), len(px)-2)
        estimate = py[index] + (py[index+1]-py[index])*(x-px[index])/(px[index+1]-px[index])
        errors.append(estimate-y)
    return {'count': len(errors), 'RMSE_pp': math.sqrt(sum(e*e for e in errors)/len(errors)),
            'max_abs_error_pp': max(map(abs, errors)),
            'mean_error_pp': sum(errors)/len(errors), 'physical_validation_pass': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output folder; preserve earlier evidence')
    prepared, hashes = {}, {}
    for cell in ('C', 'D'):
        path = args.benchmark / f'kirkaldy_expt2_cell{cell}_performance_summary.csv'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as handle:
            prepared[cell] = prepare_summary(list(csv.DictReader(handle)))
    args.output.mkdir(parents=True)
    report = {'source_sha256': hashes, 'calibration_candidate': 'C',
              'held_out_cell': 'D', 'split_status': 'proposed_not_fitted',
              'SOH_definition': '100 * measured_C10_capacity / fixed_measured_RPT0_C10_capacity',
              'axis_column': 'absolute_ageing_EFC',
              'axis_definition': 'charge_plus_discharge_throughput_Ah/(2*RPT0_C10_Ah)',
              'axis_is_approximate': False,
              'axis_provenance': {'source': 'https://zenodo.org/records/10637534',
                  'meaning': 'Authors define charge throughput as absolute charge-plus-discharge ageing Ah, excluding RPT and break-in',
                  'quality_note': 'Canonical definition does not imply exact measurement or reconcile incomplete cycle-table counters'},
              'legacy_aliases': {'approximate_EFC': {'canonical': 'absolute_ageing_EFC', 'deprecated': True,
                  'meaning': 'Historical misnomer, same numeric values; not an approximation to discharge-only EFC'}},
              'physical_validation_complete': False,
              'blockers': ['Audit incomplete cycle-table counters separately; discharge-only EFC is a different axis, not ground truth for this axis',
                           'Resolve RPT absolute time origins for chronological physical replay',
                           'Match ageing SOC/current/temperature and RPT protocols before fitting',
                           'No observed SOH80 endpoint in these C/D summaries'], 'cells': {}}
    for cell, points in prepared.items():
        with (args.output/f'cell{cell}_measured_targets.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(points[0]))
            writer.writeheader(); writer.writerows(points)
        report['cells'][cell] = {'observations': len(points), 'last': points[-1],
                                'EOL_absolute_ageing_EFC': eol_bracket(points, 'absolute_ageing_EFC'),
                                'EOL_ageing_cycles': eol_bracket(points, 'ageing_cycles')}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
