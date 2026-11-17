"""Frozen C-only ageing fit and post-freeze evaluation of admitted campaigns.

Unknown measured exposure is never replaced by rest. Synthetic software probes
exercise this backend but cannot provide capacity/SOH or lifetime validation.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from physical_campaign_inputs import (checked_path, digest, number, validate_campaign,
                                      validate_split)
import measured_protocol_replay as replay

SUPPORTED_PARAMETERS = {'SEI solvent diffusivity [m2.s-1]': 'm2.s-1',
                        'Initial SEI thickness [m]': 'm'}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def assert_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Frozen source changed: '+path)


def validate_parameters(definitions, protocol):
    if protocol['parameter_set'] != 'OKane2022' or protocol['options'].get('SEI') != 'solvent-diffusion limited':
        raise ValueError('Supported fit requires explicit OKane2022 solvent-diffusion limited SEI')
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('Explicit reviewed calibration parameter definitions required')
    names = []
    for d in definitions:
        if not isinstance(d, dict) or set(d) != {'name', 'lower', 'upper', 'initial', 'scale',
                                                   'difference_step_scaled', 'unit', 'source_ref'}:
            raise ValueError('Each fit parameter requires explicit bounds, initial, scale, step, unit and source')
        name = d['name']
        if name not in SUPPORTED_PARAMETERS or d['unit'] != SUPPORTED_PARAMETERS[name]:
            raise ValueError('Unsupported ageing parameter or unit; controls, geometry and BOL cannot be fitted')
        if name in protocol.get('parameter_overrides', {}):
            raise ValueError('Ageing parameter collides with frozen BOL override')
        if not isinstance(d['source_ref'], str) or not d['source_ref'].strip():
            raise ValueError('Physical bounds and initial value need an explicit source')
        for field in ('lower', 'upper', 'initial', 'scale', 'difference_step_scaled'):
            number(d[field], field, positive=True)
        if not d['lower'] < d['upper'] or not d['lower'] <= d['initial'] <= d['upper']:
            raise ValueError('Invalid physical calibration bounds')
        if d['difference_step_scaled'] >= (d['upper']-d['lower'])/d['scale']:
            raise ValueError('Derivative step must fit inside declared parameter interval')
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ageing parameter')
    return definitions


def campaign_prediction(campaign, result):
    if result.get('status') != 'replay_completed':
        raise ValueError('Replay did not complete')
    rows = []
    for m in campaign['measurement_steps']:
        r = result['records'][m['step_index']]
        if r['role'] != 'rpt' or r['propagated'] is not True or r['charge_Ah'] > 1e-10:
            raise ValueError('Actual RPT state propagation or discharge sign invalid')
        q = number(r['discharge_Ah'], 'Model reference discharge capacity', positive=True)
        rows.append({'rpt_id': m['rpt_id'], 'Qref_Ah': q})
    q0 = rows[0]['Qref_Ah']
    for row, target in zip(rows, campaign['_targets_rows']):
        row.update(normalized_percent=100*row['Qref_Ah']/q0,
                   target_normalized_percent=target['SOH_percent'], target_Qref_Ah=target['Qref_Ah'],
                   capacity_error_Ah=row['Qref_Ah']-target['Qref_Ah'])
    errors = [r['normalized_percent']-r['target_normalized_percent'] for r in rows]
    synthetic = campaign['data_kind'] == 'synthetic'
    full = all(m.get('kind') == 'full_discharge_capacity' for m in campaign['measurement_steps'])
    return {'data_kind': campaign['data_kind'], 'cell_id': campaign['cell_id'],
            'condition_id': campaign['condition_id'], 'measurements': rows,
            'normalization': 'fixed_first_model_RPT_Q0_and_fixed_first_target_RPT_Q0',
            'quantity_label': ('synthetic_reference_capacity_ratio_percent' if full else 'synthetic_probe_ratio_percent')
                              if synthetic else 'reference_capacity_SOH_percent',
            'model_Q0_Ah': q0, 'target_Q0_Ah': campaign['_targets_rows'][0]['Qref_Ah'],
            'Q0_relative_error_percent': 100*(q0/campaign['_targets_rows'][0]['Qref_Ah']-1),
            'native_curve_coverage': campaign.get('_native_curve_coverage'),
            'RMSE_pp': math.sqrt(sum(e*e for e in errors)/len(errors)),
            'max_abs_error_pp': max(map(abs, errors)),
            'throughput_by_role': result['throughput_by_role'],
            'physical_validation_complete': False}


class BudgetExceeded(RuntimeError):
    pass


def fit_campaign(manifest, output, base):
    """No heldout document/data is opened before parameters_frozen.json exists."""
    import numpy as np
    from scipy.optimize import least_squares
    allowed = {'schema_version', 'calibration_campaign', 'heldout_campaigns', 'calibration_parameters',
               'budget', 'objective_scale_pp', 'criteria', 'optimizer_tolerances'}
    if (not isinstance(manifest, dict) or set(manifest) != allowed
        or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1):
        raise ValueError('Explicit strict fit manifest fields required')
    cpath = checked_path(manifest['calibration_campaign'], base)
    c = validate_campaign(json.loads(cpath.read_text(encoding='utf-8-sig')), cpath.parent)
    defs = validate_parameters(manifest['calibration_parameters'], c['protocol'])
    if not isinstance(manifest['heldout_campaigns'], list):
        raise ValueError('Explicit heldout campaign list required, including []')
    budget = manifest['budget']
    if not isinstance(budget, dict) or set(budget) != {'max_model_calls', 'total_wall_s', 'per_candidate_wall_s', 'optimizer_max_nfev'}:
        raise ValueError('Explicit model-call, wall and optimizer budgets required')
    for key, value in budget.items():
        number(value, key, positive=True)
    if any(isinstance(budget[k], bool) or not isinstance(budget[k], int) for k in ('max_model_calls', 'optimizer_max_nfev')):
        raise ValueError('Model-call and optimizer budgets must be integers')
    if budget['per_candidate_wall_s'] > budget['total_wall_s'] or budget['total_wall_s'] > 3600:
        raise ValueError('Total local fit bound is at most one hour and bounds every candidate')
    scale_pp = number(manifest['objective_scale_pp'], 'objective scale pp', positive=True)
    tolerances = manifest['optimizer_tolerances']
    if not isinstance(tolerances, dict) or set(tolerances) != {'ftol', 'xtol', 'gtol'}:
        raise ValueError('Explicit optimizer ftol, xtol and gtol required')
    for name, value in tolerances.items():
        if number(value, name, positive=True) <= np.finfo(float).eps:
            raise ValueError('Optimizer tolerance must exceed floating point epsilon')
    criteria_path = checked_path(manifest['criteria'], base)
    criteria = json.loads(criteria_path.read_text(encoding='utf-8-sig'))
    for key in ('SOH_RMSE_pp_max', 'SOH_max_abs_error_pp_max'):
        number(criteria['basic'][key], key, positive=True)
    hashes = {str(cpath): digest(cpath), str(criteria_path): digest(criteria_path),
              **c['_campaign_source_hashes']}
    for path in (Path(__file__).resolve(), Path(replay.__file__).resolve(),
                 Path(__file__).with_name('physical_campaign_inputs.py').resolve(),
                 Path(__file__).with_name('physical_campaign_sources.py').resolve()):
        hashes[str(path)] = digest(path)
    # Importing a parameter module inspects its implementation, not the heldout
    # observations. Freeze parameter-library version/source before fitting.
    import pybamm
    import scipy
    parameter_module = importlib.import_module('pybamm.input.parameters.lithium_ion.'+c['protocol']['parameter_set'])
    hashes[str(Path(parameter_module.__file__).resolve())] = digest(parameter_module.__file__)
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/'manifest_frozen.json', manifest)
    freeze = {'source_hashes': hashes, 'manifest_sha256': digest(out/'manifest_frozen.json'),
              'python': sys.version, 'pybamm': pybamm.__version__, 'numpy': np.__version__,
              'scipy': scipy.__version__, 'heldout_data_opened': False,
              'objective': 'native_RPT_normalized_capacity_error_pp / declared_scale',
              'data_kind': c['data_kind'], 'parameter_definitions': defs}
    write(out/'experiment_frozen.json', freeze)
    began = time.monotonic(); calls = []; cache = {}; best = [None]
    scales = np.array([d['scale'] for d in defs]); names = [d['name'] for d in defs]
    lower = np.array([d['lower'] for d in defs])/scales
    upper = np.array([d['upper'] for d in defs])/scales
    observed = np.array([r['SOH_percent'] for r in c['_targets_rows']])
    all_hashes = hashes.copy()

    def evaluate(z, campaign, phase):
        values = [float(v) for v in np.asarray(z)*scales]
        key = (campaign['targets']['sha256'], campaign['cell_id'], tuple(values))
        if key in cache:
            return cache[key]
        remaining = budget['total_wall_s']-(time.monotonic()-began)
        if len(calls) >= budget['max_model_calls'] or remaining <= 0:
            raise BudgetExceeded('Predeclared call or total wall budget reached')
        assert_hashes(all_hashes)
        candidate = out/f'candidate_{len(calls)+1:04d}'; candidate.mkdir()
        protocol = copy.deepcopy(campaign['protocol'])
        protocol.setdefault('parameter_overrides', {}).update(dict(zip(names, values)))
        write(candidate/'protocol.json', protocol)
        request = {'protocol': protocol, 'source_hashes': all_hashes}
        write(candidate/'request.json', request)
        record = {'phase': phase, 'cell_id': campaign['cell_id'], 'parameters': dict(zip(names, values)),
                  'candidate_path': candidate.name, 'status': 'started'}
        calls.append(record); write(candidate/'attempt.json', record); started = time.monotonic()
        command = [sys.executable, str(Path(__file__).resolve()), '--candidate',
                   '--manifest', str(candidate/'request.json'), '--output', str(candidate)]
        try:
            with (candidate/'worker.log').open('w', encoding='utf-8') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(remaining, budget['per_candidate_wall_s'], protocol['max_wall_s']),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            result = json.loads((candidate/'report.json').read_text(encoding='utf-8'))
            if run.returncode or result['status'] != 'replay_completed':
                raise ValueError(result.get('error', result['status']))
            prediction = campaign_prediction(campaign, result)
            record.update(status='evaluated', prediction=prediction)
            cache[key] = prediction
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            cache[key] = None
        finally:
            record['wall_s'] = time.monotonic()-started
            write(candidate/'attempt.json', record); write(out/'calls.json', calls)
        assert_hashes(all_hashes)
        return cache[key]

    def residual(z):
        prediction = evaluate(z, c, 'C_calibration')
        if prediction is None:
            return np.full(len(observed), 1e6)
        r = (np.array([v['normalized_percent'] for v in prediction['measurements']])-observed)/scale_pp
        cost = float(r@r)
        if best[0] is None or cost < best[0]['cost']:
            best[0] = {'z': np.array(z), 'cost': cost, 'prediction': prediction}
        return r

    def jacobian(z):
        center = evaluate(z, c, 'C_calibration')
        if center is None:
            raise ValueError('Physical derivative unavailable at failed base candidate')
        f0 = residual(z); columns = []
        for j, d in enumerate(defs):
            h = d['difference_step_scaled']
            h = min(h, upper[j]-z[j]) if upper[j]-z[j] >= h else -min(h, z[j]-lower[j])
            if h == 0:
                raise ValueError('No finite physical derivative interval')
            shifted = np.array(z, copy=True); shifted[j] += h
            if evaluate(shifted, c, 'C_calibration') is None:
                raise ValueError('Physical derivative unavailable at failed perturbation')
            columns.append((residual(shifted)-f0)/h)
        return np.column_stack(columns)

    optimizer = {'success': False, 'status': 'not_started'}; solution = None
    try:
        z0 = np.array([d['initial'] for d in defs])/scales
        initial_jac = jacobian(z0)
        write(out/'initial_sensitivity.json', {'all_physical_calls_valid': True,
            'jacobian': initial_jac.tolist(), 'rank': int(np.linalg.matrix_rank(initial_jac)),
            'parameter_count': len(defs), 'condition': finite_condition(initial_jac)})
        solution = least_squares(residual, z0, jac=jacobian, bounds=(lower, upper),
                                 max_nfev=budget['optimizer_max_nfev'], **tolerances)
        optimizer = {'success': bool(solution.success), 'status': int(solution.status),
                     'message': solution.message, 'nfev': solution.nfev,
                     'jacobian_rank': int(np.linalg.matrix_rank(solution.jac)),
                     'jacobian_condition': finite_condition(solution.jac)}
    except (BudgetExceeded, ValueError) as error:
        optimizer.update(status='incomplete', error=f'{type(error).__name__}: {error}')
    if best[0] is None:
        write(out/'report.json', {'status': 'no_valid_candidate', 'optimizer': optimizer,
              'model_calls': len(calls), 'physical_validation_complete': False, 'AWS_started': False})
        return 1
    selected = best[0]
    selection = 'lowest_valid_C_candidate_not_converged'
    if optimizer['success']:
        final_prediction = evaluate(solution.x, c, 'C_calibration')
        if final_prediction is None:
            raise ValueError('Converged optimizer solution has no valid physical evaluation')
        r = (np.array([v['normalized_percent'] for v in final_prediction['measurements']])-observed)/scale_pp
        selected = {'z': solution.x, 'cost': float(r@r), 'prediction': final_prediction}
        selection = 'converged_optimizer_solution_not_a_Jacobian_probe'
    values = (selected['z']*scales).tolist()
    parameter_freeze = {'parameters': dict(zip(names, values)), 'calibration_cell': c['cell_id'],
                        'calibration_target_sha256': c['targets']['sha256'], 'optimizer': optimizer,
                        'experiment_frozen_sha256': digest(out/'experiment_frozen.json'),
                        'data_kind': c['data_kind'], 'selected_C_objective': selected['cost'],
                        'selection': selection,
                        'heldout_data_opened': False, 'physical_validation_complete': False}
    assert_hashes(all_hashes); write(out/'parameters_frozen.json', parameter_freeze)
    frozen_hash = digest(out/'parameters_frozen.json'); heldout_results = []; admitted = []
    heldout_failures = []
    for ref in manifest['heldout_campaigns']:
        try:
            hpath = checked_path(ref, base)
            h = validate_campaign(json.loads(hpath.read_text(encoding='utf-8-sig')), hpath.parent)
            validate_split(c, admitted+[h]); admitted.append(h)
            all_hashes.update({str(hpath): digest(hpath), **h['_campaign_source_hashes']})
            # Same frozen BOL, physics and measurement definition; condition and
            # experimental exposure may differ, never heldout parameter fitting.
            for key in ('family', 'parameter_set', 'options', 'parameter_overrides'):
                if h['protocol'].get(key, {}) != c['protocol'].get(key, {}):
                    raise ValueError('Heldout physical model or frozen BOL differs')
            validate_parameters(defs, h['protocol'])
            prediction = evaluate(selected['z'], h, 'postfreeze_heldout')
            if prediction is None:
                raise ValueError('Heldout replay failed')
            prediction['basic_curve_limits_satisfied'] = (h['data_kind'] == 'measured'
                and h.get('_native_curve_coverage', {}).get('full_native_curve_evaluated') is True
                and prediction['RMSE_pp'] <= criteria['basic']['SOH_RMSE_pp_max']
                and prediction['max_abs_error_pp'] <= criteria['basic']['SOH_max_abs_error_pp_max'])
            heldout_results.append(prediction)
        except (ValueError, OSError, BudgetExceeded) as error:
            heldout_failures.append({'reference': ref, 'error': f'{type(error).__name__}: {error}'})
    assert_hashes(all_hashes)
    if digest(out/'parameters_frozen.json') != frozen_hash:
        raise ValueError('Frozen calibration changed during heldout evaluation')
    write(out/'report.json', {'status': 'backend_completed' if optimizer['success'] else 'backend_incomplete',
        'data_kind': c['data_kind'],
        'model_kind': 'physical_model', 'calibration': selected['prediction'], 'optimizer': optimizer,
        'calibration_optimizer_converged': optimizer['success'],
        'local_sensitivity_full_rank': optimizer.get('jacobian_rank') == len(defs),
        'parameter_identification_unique': False, 'heldout_results': heldout_results,
        'heldout_failures': heldout_failures, 'model_calls': len(calls), 'wall_s': time.monotonic()-began,
        'source_hashes_unchanged': True, 'parameters_frozen_sha256': frozen_hash,
        'physical_validation_complete': False, 'SOH_lifetime_validation_complete': False,
        'EOL_validation_status': 'not_evaluated', 'AWS_started': False,
        'scope': 'synthetic_backend_runtime_only' if c['data_kind'] == 'synthetic' else 'admitted_campaign_metrics_only',
        'limitations': ['No automatic physical certification from optimizer or curve limits',
                        'Synthetic timed probes measure throughput ratios, not full reference capacity/SOH',
                        'A measured subset is partial coverage and cannot pass the full native curve criterion',
                        'Only explicitly supported SEI parameters can be fitted; no inferred missing exposure']})
    return 0 if optimizer['success'] and not heldout_failures else 1


def finite_condition(matrix):
    import numpy as np
    value = float(np.linalg.cond(matrix))
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if args.candidate:
        try:
            assert_hashes(manifest['source_hashes'])
            result = replay.worker(manifest['protocol'], args.output)
            assert_hashes(manifest['source_hashes'])
        except Exception as error:
            result = {'status': 'replay_failed', 'error': f'{type(error).__name__}: {error}',
                      'physical_validation_complete': False, 'AWS_started': False}
        write(args.output/'report.json', result)
        return 0 if result['status'] == 'replay_completed' else 1
    return fit_campaign(manifest, args.output, args.manifest.resolve().parent)


if __name__ == '__main__':
    raise SystemExit(main())
