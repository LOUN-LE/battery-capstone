"""Fixed C-fit/D-heldout empirical baselines; never a PyBaMM validation.

The model forms are declared before evaluating D. RPT dates are day-resolution
calendar proxies, not a reconstruction of the unobserved intra-day history.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import scipy
from scipy.optimize import nnls


MODEL_VERSION = 'empirical_C_only_v2_predictor_provenance'
FEATURES = ('sqrt(calendar_days_since_RPT0 / 365)', 'absolute_ageing_EFC / 1000')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {name}')
    return value


def load_summary(path, cell):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise ValueError('At least three native RPT observations required')
    q0 = number(rows[0]['C/10 Capacity [mA h]'], 'Q0') / 1000
    if q0 <= 0 or number(rows[0]['Ageing Sets'], 'first RPT') != 0:
        raise ValueError('Positive measured RPT0 required')
    origin = date.fromisoformat(rows[0]['RPT start date'])
    points = []
    for row in rows:
        q = number(row['C/10 Capacity [mA h]'], 'C10 capacity') / 1000
        throughput = number(row['Charge Throughput [A h]'], 'throughput')
        rpt_date = date.fromisoformat(row['RPT start date'])
        point = {'cell': cell, 'rpt': number(row['Ageing Sets'], 'RPT'),
                 'ageing_cycles': number(row['Ageing Cycles'], 'cycles'),
                 'rpt_date': rpt_date.isoformat(),
                 'calendar_days_since_RPT0': (rpt_date-origin).days,
                 'reported_days_of_degradation': number(row['Days of degradation'], 'degradation days'),
                 'absolute_ageing_throughput_Ah': throughput,
                 'absolute_ageing_EFC': throughput / (2*q0),
                 'Q0_Ah': q0, 'capacity_Ah': q, 'SOH_percent': 100*q/q0}
        if q <= 0 or min(point['calendar_days_since_RPT0'], throughput, point['ageing_cycles']) < 0:
            raise ValueError('Invalid capacity or exposure')
        if points and any(point[key] <= points[-1][key] for key in
                          ('rpt', 'calendar_days_since_RPT0', 'absolute_ageing_EFC')):
            raise ValueError('RPT/date/throughput axis must strictly increase')
        points.append(point)
    if points[0]['absolute_ageing_throughput_Ah'] != 0:
        raise ValueError('Throughput origin must be RPT0')
    return points


def design(points):
    values = []
    for point in points:
        days = number(point['calendar_days_since_RPT0'], 'calendar days')
        efc = number(point['absolute_ageing_EFC'], 'absolute ageing EFC')
        if min(days, efc) < 0:
            raise ValueError('Negative exposure')
        values.append([math.sqrt(days/365), efc/1000])
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class FrozenBaseline:
    version: str
    fit_cell: str
    fit_source_sha256: str
    sqrt_calendar_coefficient_pp: float
    linear_efc_coefficient_pp: float


def fit_calibration(points, fit_cell, source_sha256):
    """The fitter accepts calibration points only; heldout outcomes have no input."""
    if fit_cell != 'C' or not points or any(p['cell'] != fit_cell for p in points):
        raise ValueError('This declared experiment permits C-only calibration')
    x = design(points)
    y = np.asarray([100-number(p['SOH_percent'], 'SOH') for p in points])
    coefficients, _ = nnls(x, y)
    return FrozenBaseline(MODEL_VERSION, fit_cell, source_sha256,
                          float(coefficients[0]), float(coefficients[1]))


def predict(model, points):
    # Prediction consumes only exposure columns, not observed heldout SOH.
    x = design(points)
    coefficients = np.array([model.sqrt_calendar_coefficient_pp, model.linear_efc_coefficient_pp])
    return (100-x@coefficients).tolist()


def metrics(points, predicted):
    if len(points) != len(predicted) or not points:
        raise ValueError('One prediction per native RPT is required')
    actual = np.asarray([number(p['SOH_percent'], 'SOH') for p in points])
    prediction = np.asarray([number(value, 'prediction') for value in predicted])
    error = prediction-actual
    return {'RPT_count': len(points), 'RMSE_pp': float(np.sqrt(np.mean(error**2))),
            'max_abs_error_pp': float(np.max(np.abs(error))),
            'mean_bias_pp': float(np.mean(error)), 'observed_SOH_min_percent': float(actual.min()),
            'observed_SOH_max_percent': float(actual.max()),
            'native_observations_only': True}


def observed_eol(points):
    for index, point in enumerate(points):
        if point['SOH_percent'] <= 80:
            if index == 0:
                return {'status': 'left_censored', 'EFC80': None}
            prior = points[index-1]
            return {'status': 'observed_bracket', 'EFC80': None,
                    'lower_EFC': prior['absolute_ageing_EFC'], 'upper_EFC': point['absolute_ageing_EFC'],
                    'note': 'No interpolation or EOL pass decision is made by this baseline script'}
    return {'status': 'right_censored', 'EFC80': None,
            'last_observed_EFC': points[-1]['absolute_ageing_EFC'],
            'last_observed_SOH_percent': points[-1]['SOH_percent']}


def run(benchmark, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    benchmark = Path(benchmark)
    settings = {'version': MODEL_VERSION, 'calibration_cells': ['C'], 'heldout_cells': ['D'],
                'models': ['constant_initial_SOH_100', 'nonnegative_sqrt_calendar_plus_linear_EFC'],
                'features': FEATURES, 'intercept': 100, 'coefficient_constraint': 'nonnegative',
                'fit_objective': 'unweighted squared SOH residual at native C RPT observations',
                'selection_rule': 'Both a-priori baselines reported; no selection by D results',
                'temperature_or_SOC_sensitivity': 'not modelled; one-condition benchmark only',
                'heldout_Q0_policy': 'D RPT0 capacity only normalizes D SOH and fixed exposure denominator',
                'prediction_information': 'RPT schedule and observed cumulative ageing throughput; not a blind future-EOL forecast',
                'SOH_definition': '100 * C10_capacity / fixed_measured_RPT0_C10_capacity',
                'EFC_definition': 'charge_plus_discharge_ageing_Ah / (2 * fixed_measured_RPT0_C10_Ah)',
                'EFC_scope': 'summary-reported ageing throughput excludes RPT and break-in; not interchangeable with incomplete discharge-table EFC',
                'calendar_proxy': 'RPT start date minus RPT0 start date; date-only precision, not exact chronological replay',
                'reported_Days_of_degradation': 'preserved separately; not assumed equal to calendar elapsed days',
                'physical_model_validation': False}
    (output/'declared_experiment.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    try:
        c_path = benchmark/'kirkaldy_expt2_cellC_performance_summary.csv'
        c_hash = digest(c_path)
        calibration = load_summary(c_path, 'C')
        model = fit_calibration(calibration, 'C', c_hash)
        frozen_path = output/'frozen_C_fit.json'
        frozen_path.write_text(json.dumps({'settings': settings, 'model': asdict(model),
            'predictor_dependency': {'file': Path(__file__).name, 'sha256': digest(__file__)},
            'numerical_dependencies': {'numpy': np.__version__, 'scipy': scipy.__version__}}, indent=2), encoding='utf-8')
        frozen_hash = digest(frozen_path)
        # D is loaded only after the frozen C-fit artifact has been written.
        d_path = benchmark/'kirkaldy_expt2_cellD_performance_summary.csv'
        d_hash = digest(d_path)
        heldout = load_summary(d_path, 'D')
        report = {'status': 'evaluated', 'model_kind': 'empirical_baseline_not_PyBaMM',
                  'physical_validation_complete': False, 'EOL_validation_complete': False,
                  'AWS_started': False, 'fit_source_sha256': c_hash, 'heldout_source_sha256': d_hash,
                  'frozen_fit_sha256_before_heldout_load': frozen_hash, 'frozen_fit_sha256_after_evaluation': None,
                  'code_sha256': digest(__file__), 'models': {},
                  'scope': 'C-calibration and D-outcome holdout, Kirkaldy2024 Experiment2 at same nominal25C/SOC70-85; no heldout-condition validation',
                  'limitations': ['Date-only calendar exposure proxy; exact temperature/RPT/calendar replay not performed',
                                  'No physical degradation coefficients fitted', 'D outcomes were not used in this fit or model selection',
                                  'Previously inspected D summaries mean this is not a newly blinded dataset',
                                  'Ageing-throughput provenance/reconciliation remains separate from this summary-level benchmark'],
                  'cells': {}}
        for name in settings['models']:
            report['models'][name] = {}
            rows = []
            for cell, points in [('C', calibration), ('D', heldout)]:
                predicted = [100.]*len(points) if name == 'constant_initial_SOH_100' else predict(model, points)
                report['models'][name][cell] = {'role': 'calibration' if cell == 'C' else 'heldout',
                                               **metrics(points, predicted)}
                rows.extend({**point, 'predicted_SOH_percent': prediction,
                             'residual_pp': prediction-point['SOH_percent']} for point, prediction in zip(points, predicted))
                report['cells'][cell] = {'Q0_Ah': points[0]['Q0_Ah'], 'EOL': observed_eol(points)}
            with (output/f'{name}_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        report['frozen_fit_sha256_after_evaluation'] = digest(frozen_path)
        if frozen_hash != report['frozen_fit_sha256_after_evaluation']:
            raise RuntimeError('Frozen C model was modified during heldout evaluation')
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return report
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(), 'physical_validation_complete': False}, indent=2), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.benchmark, args.output)
    print(json.dumps({'status': result['status'], 'models': result['models'],
                      'physical_validation_complete': False}, indent=2))


if __name__ == '__main__':
    main()
