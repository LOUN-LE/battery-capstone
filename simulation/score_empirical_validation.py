"""Apply per-cell 1-A/1-B criteria and staged 2-A scope without physical certification."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from empirical_validation import compare_curves


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid finite number: {label}')
    if positive and value <= 0:
        raise ValueError(f'Nonpositive {label}')
    return float(value)


def _strings(values, label):
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f'Explicit nonempty {label} required')
    if len(set(values)) != len(values):
        raise ValueError(f'Duplicate {label}')
    return set(values)


def _provenance(case, evidence_root):
    for field in ('cell_id', 'experiment_id', 'condition_id', 'axis_column'):
        if not isinstance(case.get(field), str) or not case[field]:
            raise ValueError(f'Missing {field}')
    model = case['model']
    if not isinstance(model, dict):
        raise ValueError('Explicit model provenance object required')
    if not isinstance(model.get('id'), str) or not model['id']:
        raise ValueError('Model id required')
    if model.get('kind') not in ('physical', 'empirical_baseline', 'initial_SOH_baseline'):
        raise ValueError('Explicit model kind required')
    fit_cells = _strings(model.get('fit_cell_ids'), 'fit_cell_ids')
    fit_conditions = _strings(model.get('fit_condition_ids'), 'fit_condition_ids')
    if case['cell_id'] in fit_cells:
        raise ValueError('Calibration and validation cells overlap')
    if model.get('frozen_before_evaluation') is not True or model.get('retuned_on_validation') is not False:
        raise ValueError('Model must be frozen before held-out evaluation without retuning')
    if model.get('heldout_initialization') not in ('RPT0_Q0_only', 'no_heldout_inputs'):
        raise ValueError('Held-out initial information policy required')
    path = Path(model['artifact_path'])
    path = path if path.is_absolute() else Path(evidence_root)/path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != model.get('artifact_sha256'):
        raise ValueError('Frozen model artifact SHA-256 mismatch')
    stage = 'stage1' if case['condition_id'] in fit_conditions else 'stage2'
    if stage == 'stage2' and case.get('heldout_condition_axis') not in ('temperature', 'SOC'):
        raise ValueError('Stage2 needs an explicitly sourced untrained temperature or SOC condition')
    # An artifact hash verifies the evaluated bytes. It does not prove the
    # declaration about their creation time or how the investigator used data.
    return {'model_id': model['id'], 'model_kind': model['kind'], 'artifact_sha256': actual,
            'stage': stage,
            'provenance_scope': 'Artifact bytes verified; split and freeze chronology are explicit declarations requiring audit'}


def _crossing(points, axis, threshold):
    if len(points) < 2:
        raise ValueError('At least two observations required for EOL assessment')
    xs = [_number(p[axis], axis) for p in points]
    ys = [_number(p['SOH_percent'], 'SOH') for p in points]
    if any(x < 0 for x in xs) or any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError('EOL axis must be nonnegative and strictly increasing')
    if ys[0] <= threshold:
        return {'status': 'left_censored', 'upper_EFC': xs[0]}
    for index in range(1, len(points)):
        if ys[index] <= threshold:
            low, high = xs[index-1], xs[index]
            value = low + (ys[index-1]-threshold)/(ys[index-1]-ys[index])*(high-low)
            _number(value, 'interpolated EFC80', True)
            _number(high-low, 'EOL interval width', True)
            return {'status': 'observed', 'EFC80': value, 'lower_EFC': low, 'upper_EFC': high,
                    'bracket_width_EFC': high-low,
                    'recrossed_above_threshold': any(y > threshold for y in ys[index+1:])}
    return {'status': 'right_censored', 'lower_EFC': xs[-1]}


def _curve_status(metrics, limits):
    return 'passed' if (metrics['RMSE_pp'] <= _number(limits['SOH_RMSE_pp_max'], 'RMSE limit', True)
                       and metrics['max_abs_error_pp'] <= _number(limits['SOH_max_abs_error_pp_max'], 'max error limit', True)) else 'failed'


def _eol(case, criteria):
    rule = criteria['eol']
    threshold = _number(rule['threshold_SOH_percent'], 'EOL threshold', True)
    axis = case['axis_column']
    measured = _crossing(case['measured'], axis, threshold)
    if measured['status'] != 'observed':
        return {'status': 'not_evaluated', 'reason': 'Measured threshold crossing not observed',
                'measured': measured, 'basic': 'not_evaluated', 'challenge': 'not_evaluated'}
    if (case['measured_metadata'].get('axis_unit') != 'EFC'
            or case['predicted_metadata'].get('axis_unit') != 'EFC'):
        return {'status': 'deferred', 'reason': 'EFC80 requires reviewed EFC-axis units, not cycles or days', 'measured': measured}
    if measured['recrossed_above_threshold']:
        return {'status': 'deferred', 'reason': 'Measured SOH recrosses threshold; crossing stability needs review', 'measured': measured}
    if rule.get('interpolation_rule') != 'linear_first_observed_downward_crossing':
        return {'status': 'deferred', 'reason': 'Supported interpolation rule not fixed', 'measured': measured}
    resolution = rule.get('max_observed_bracket_EFC')
    if resolution is None:
        return {'status': 'deferred', 'reason': 'Observed RPT interval resolution has not been justified', 'measured': measured}
    resolution = _number(resolution, 'maximum observed EOL bracket', True)
    if measured['bracket_width_EFC'] > resolution:
        return {'status': 'deferred', 'reason': 'Observed EOL interval exceeds fixed resolution limit', 'measured': measured}
    predicted = _crossing(case['predicted'], axis, threshold)
    if predicted['status'] != 'observed':
        return {'status': 'not_evaluated', 'reason': 'Prediction does not contain threshold crossing; no extrapolation',
                'measured': measured, 'predicted': predicted, 'basic': 'not_evaluated', 'challenge': 'not_evaluated'}
    if predicted['recrossed_above_threshold']:
        return {'status': 'deferred', 'reason': 'Prediction recrosses threshold; EOL definition needs review',
                'measured': measured, 'predicted': predicted}
    if measured['EFC80'] <= 0:
        raise ValueError('Measured EFC80 must be positive')
    error = 100*abs(predicted['EFC80']-measured['EFC80'])/measured['EFC80']
    _number(error, 'EFC80 relative error')
    basic = 'passed' if error <= _number(criteria['basic']['observed_EFC80_relative_error_percent_max'], 'EOL basic limit', True) else 'failed'
    challenge = 'passed' if error <= _number(criteria['challenge']['observed_EFC80_relative_error_percent_max'], 'EOL challenge limit', True) else 'failed'
    return {'status': basic, 'basic': basic, 'challenge': challenge, 'relative_error_percent': error,
            'measured': measured, 'predicted': predicted, 'interpolation_rule': rule['interpolation_rule']}


def score_cell(case, criteria, evidence_root):
    if not isinstance(case, dict):
        return {'status': 'deferred', 'reason': 'Case must be a JSON object',
                'cell_id': None, 'experiment_id': None, 'condition_id': None,
                'assessment_scope': 'metric_scoring_only', 'physical_validation_complete': False}
    result = {'cell_id': case.get('cell_id'), 'experiment_id': case.get('experiment_id'),
              'condition_id': case.get('condition_id'), 'status': 'deferred',
              'assessment_scope': 'metric_scoring_only', 'physical_validation_complete': False}
    try:
        result.update(_provenance(case, evidence_root))
        for points in (case['measured'], case['predicted']):
            for point in points:
                _number(point[case['axis_column']], 'comparison coordinate')
                _number(point['SOH_percent'], 'SOH')
        metrics = compare_curves(case['measured'], case['predicted'], case['measured_metadata'],
                                 case['predicted_metadata'], case['axis_column'])
        for key in ('RMSE_pp', 'max_abs_error_pp', 'mean_error_pp'):
            _number(metrics[key], key)
        result['metrics'] = metrics
        result['curve_basic'] = _curve_status(metrics, criteria['basic'])
        result['curve_challenge'] = _curve_status(metrics, criteria['challenge'])
        result['measured_SOH_range_percent'] = [min(p['SOH_percent'] for p in case['measured']),
                                                max(p['SOH_percent'] for p in case['measured'])]
        result['RPT_count'] = len(case['measured'])
        result['eol'] = _eol(case, criteria)
        result['status'] = result['curve_basic']
        result['statement'] = ('Observed SOH curve criteria passed; EOL assessed separately' if result['status'] == 'passed'
                               else 'Valid observed SOH comparison failed at least one basic limit')
        if result['model_kind'] != 'physical':
            result['statement'] += '; baseline result does not validate a physical digital twin'
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as error:
        result['status'] = 'deferred'
        result['reason'] = str(error)
    return result


def score_suite(cases, criteria, evidence_root):
    results = [score_cell(case, criteria, evidence_root) for case in cases]
    identities = {(r.get('model_id'), r.get('model_kind'), r.get('artifact_sha256')) for r in results}
    repeated = len({(r['experiment_id'], r['cell_id']) for r in results}) != len(results)
    consistent = bool(results) and len(identities) == 1 and None not in next(iter(identities)) and not repeated
    def stage(number):
        rows = [r for r in results if r.get('stage') == number]
        if not rows:
            return 'not_evaluated'
        if not consistent or any(r['status'] == 'deferred' for r in rows):
            return 'deferred'
        return 'passed' if all(r['status'] == 'passed' for r in rows) else 'failed'
    stages = {'stage1_same_condition': stage('stage1'), 'stage2_new_condition': stage('stage2')}
    scope = ('evaluated_conditions_only' if all(s == 'passed' for s in stages.values()) else
             'same_condition_only' if stages['stage1_same_condition'] == 'passed' else 'no_passed_scope')
    return {'criteria_id': criteria['criteria_id'], 'cells': results, 'stages': stages,
            'same_frozen_model_verified': consistent, 'scope': scope,
            'staged_2A_curve_scope_complete': all(s == 'passed' for s in stages.values()),
            'physical_validation_complete': False, 'assessment_scope': 'metric_scoring_only',
            'warning': 'No universal SOC/temperature/charge-rate or physical certification follows from metric scoring. Baseline passes remain baseline results.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True, help='JSON array of case objects')
    parser.add_argument('--criteria', type=Path, default=Path(__file__).resolve().parents[1]/'configs/empirical_acceptance.json')
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier scoring evidence; use a new output path')
    result = score_suite(json.loads(args.cases.read_text(encoding='utf-8-sig')),
                         json.loads(args.criteria.read_text(encoding='utf-8-sig')), args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
