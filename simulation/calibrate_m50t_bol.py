"""Bounded C-only effective BOL calibration; never a lifetime/SOH validation.

Transport and degradation coefficients stay fixed. Two parameters change only
initial active capacities and lithium inventory. D is opened after fit freeze.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

F = 96485.33212331


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def derive_initial_updates(p, scale, delta):
    """delta is Ah of particle lithium, NOT Ah of nominal discharge capacity."""
    if not all(math.isfinite(v) for v in (scale, delta)) or scale <= 0:
        raise ValueError('Finite positive loading scale and finite inventory shift required')
    area = p['Electrode width [m]']*p['Electrode height [m]']
    eps_n = p['Negative electrode active material volume fraction']*scale
    eps_p = p['Positive electrode active material volume fraction']*scale
    for side, eps in [('Negative', eps_n), ('Positive', eps_p)]:
        if not 0 < eps <= 1-p[side+' electrode porosity']+1e-12:
            raise ValueError('Active material plus pore fraction exceeds physical volume')
    coefficient_n = F*area*p['Negative electrode thickness [m]']*eps_n/3600
    coefficient_p = F*area*p['Positive electrode thickness [m]']*eps_p/3600
    cn = p['Initial concentration in negative electrode [mol.m-3]']+delta/coefficient_n
    cp = p['Initial concentration in positive electrode [mol.m-3]']
    if not (0 < cn < p['Maximum concentration in negative electrode [mol.m-3]']
            and 0 < cp < p['Maximum concentration in positive electrode [mol.m-3]']):
        raise ValueError('Initial particle concentrations violate stoichiometry bounds')
    updates = {'Negative electrode active material volume fraction': eps_n,
               'Positive electrode active material volume fraction': eps_p,
               'Initial concentration in negative electrode [mol.m-3]': cn}
    return updates, {'target_initial_particle_lithium_Ah': coefficient_n*cn+coefficient_p*cp,
                     'negative_capacity_Ah': coefficient_n*p['Maximum concentration in negative electrode [mol.m-3]'],
                     'positive_capacity_Ah': coefficient_p*p['Maximum concentration in positive electrode [mol.m-3]'],
                     'negative_initial_stoichiometry_before_SOC_mapping': cn/p['Maximum concentration in negative electrode [mol.m-3]']}


def read_curve(path):
    import numpy as np
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    q = np.asarray([float(row['Charge (mA.h)'])/1000 for row in rows])
    v = np.asarray([float(row['Voltage (V)']) for row in rows])
    if len(q) < 2 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError('Invalid measured capacity-voltage curve')
    q = q-q[0]
    if q[-1] <= 0 or not np.all(np.diff(q) > 0):
        raise ValueError('Measured discharge Ah must strictly increase')
    return q, v


class BudgetExceeded(RuntimeError):
    pass


def worker(config, out):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYBAMM_DISABLE_TELEMETRY'] = 'true'
    import numpy as np
    import pybamm
    import scipy
    import importlib
    from scipy.optimize import least_squares, minimize_scalar
    began = time.monotonic()
    if digest(__file__) != config['code_sha256']:
        raise ValueError('Calibration code changed after declared experiment freeze')
    parameter_module = importlib.import_module('pybamm.input.parameters.lithium_ion.OKane2022')
    parameter_source_hash = digest(parameter_module.__file__)
    write(out/'runtime_environment_before_fit.json', {'python': sys.version, 'pybamm': pybamm.__version__,
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'OKane2022_source_sha256': parameter_source_hash, 'code_sha256': digest(__file__)})
    source_c = Path(config['source_C'])
    if digest(source_c) != config['source_C_sha256']:
        raise ValueError('C source changed after experiment freeze')
    measured_q, measured_v = read_curve(source_c)
    sample_q = measured_q[-1]*np.linspace(*config['voltage_capacity_fraction_grid'])
    observed_v = np.interp(sample_q, measured_q, measured_v)
    base = pybamm.ParameterValues('OKane2022')
    cache, attempts = {}, []
    count = [0]
    best = [None]

    def evaluate(x, fine=False, reserve=0):
        x = [float(v) for v in x]
        key = tuple(x)+(bool(fine),)
        if key in cache:
            return cache[key]
        if count[0] >= config['max_model_calls']-reserve or time.monotonic()-began > config['max_wall_s']-60:
            raise BudgetExceeded('Predeclared model-call or wall budget reached')
        count[0] += 1
        number = count[0]
        stem = out/f'candidate_{number:03d}'
        record = {'call': number, 's_Q': x[0], 'delta_Q_Li_Ah': x[1], 'fine_mesh': fine,
                  'status': 'started', 'raw_file': stem.name+'.npz'}
        write(stem.with_suffix('.json'), record)
        started = time.monotonic()
        try:
            updates, inventory = derive_initial_updates(base, *x)
            p = base.copy()
            p.update(updates)
            p.update({'Initial temperature [K]': 298.15, 'Ambient temperature [K]': 298.15})
            model = pybamm.lithium_ion.SPMe(options=config['options'])
            steps = [pybamm.step.current(-1.5, duration=20000, termination='4.2 V', skip_ok=False),
                     pybamm.step.voltage(4.2, duration=20000, termination='0.05 A', skip_ok=False),
                     pybamm.step.rest(config['pre_discharge_rest_s']),
                     pybamm.step.current(.5, duration=50000, termination='2.5 V', skip_ok=False)]
            if config['initial_rest_s'] > 0:
                steps.insert(0, pybamm.step.rest(config['initial_rest_s']))
            resolution = config['fine'] if fine else config['fit']
            experiment = pybamm.Experiment(steps, period=resolution['output_period_s'], temperature=298.15)
            sim = pybamm.Simulation(model, parameter_values=p, experiment=experiment,
                                   var_pts=resolution['mesh_points'],
                                   solver=pybamm.IDAKLUSolver(rtol=resolution['rtol'], atol=resolution['atol']))
            solution = sim.solve(initial_soc=config['initial_soc_assumption'], calc_esoh=False)
            pieces = [cycle.steps[-1] for cycle in solution.cycles]
            discharge = pieces[-1]
            q = np.asarray(discharge['Discharge capacity [A.h]'].entries, dtype=float)
            q = q-q[0]
            v = np.asarray(discharge['Voltage [V]'].entries, dtype=float)
            t = np.asarray(discharge['Time [s]'].entries, dtype=float)
            current = np.asarray(discharge['Current [A]'].entries, dtype=float)
            initial_li = float(solution['Total lithium in particles [mol]'].entries[0])*F/3600
            np.savez_compressed(stem.with_suffix('.npz'), Q_Ah=q, V=v, time_s=t, current_A=current)
            if not all(np.all(np.isfinite(a)) for a in (q, v, t, current)) or len(q) < 2 or not np.all(np.diff(q) > 0):
                raise ValueError('Nonfinite or nonmonotonic model discharge')
            expected = config['expected_terminations']
            actual = [str(piece.termination) for piece in pieces]
            if actual != expected:
                raise ValueError('A required RPT endpoint was not reached: '+str(actual))
            if abs(initial_li-inventory['target_initial_particle_lithium_Ah']) > config['inventory_conservation_tolerance_Ah']:
                raise ValueError('SOC initialization did not conserve specified particle lithium inventory')
            integrated_q = float(np.trapezoid(current, t)/3600)
            if abs(integrated_q-q[-1]) > 1e-7+1e-4*q[-1]:
                raise ValueError('Discharge current/capacity integral mismatch')
            if q[-1] < sample_q[-1]:
                raise ValueError('Candidate does not cover the predeclared voltage sampling interval')
            voltage_residual = (np.interp(sample_q, q, v)-observed_v)/(config['voltage_weight_scale_V']*math.sqrt(len(sample_q)))
            capacity_residual = (q[-1]-measured_q[-1])/config['capacity_weight_scale_Ah']
            residual = np.r_[voltage_residual, capacity_residual]
            record.update(status='evaluated', objective_sum_squares=float(residual@residual),
                          Q_model_Ah=float(q[-1]), C_measured_Q_Ah=float(measured_q[-1]),
                          capacity_error_percent=float(100*(q[-1]/measured_q[-1]-1)),
                          voltage_RMSE_V_on_fixed_fit_grid=float(np.sqrt(np.mean((np.interp(sample_q,q,v)-observed_v)**2))),
                          parameter_updates=updates, inventory=inventory,
                          actual_initial_particle_lithium_Ah=initial_li,
                          initialization_inventory_residual_Ah=initial_li-inventory['target_initial_particle_lithium_Ah'],
                          expected_terminations=actual)
            if not fine and (best[0] is None or record['objective_sum_squares'] < best[0]['objective_sum_squares']):
                best[0] = record.copy()
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            # Fixed-length penalty, never candidate-specific clipping/extrapolation.
            residual = np.full(len(sample_q)+1, 1000+float(np.linalg.norm(x)))
        record['wall_s'] = time.monotonic()-started
        write(stem.with_suffix('.json'), record)
        attempts.append(record)
        write(out/'candidate_journal.json', attempts)
        cache[key] = residual, record
        gc.collect()
        return residual, record

    baseline_residual, baseline = evaluate(config['baseline'])
    starts = config['starts']
    bounds = np.asarray(config['bounds'], dtype=float)
    h = np.asarray(config['jacobian_absolute_steps'], dtype=float)

    def function(x):
        return evaluate(x, reserve=4)[0]

    sensitivity_records = {}
    def jacobian(x, reserve=4):
        initial, first_record = evaluate(x, reserve=reserve)
        sampled = [first_record]
        columns = []
        for index in range(2):
            stepped = np.asarray(x, dtype=float).copy()
            delta = h[index] if x[index]+h[index] <= bounds[1,index] else -h[index]
            stepped[index] += delta
            residual, record = evaluate(stepped, reserve=reserve)
            sampled.append(record)
            columns.append((residual-initial)/delta)
        details = {'all_candidate_evaluations_valid': all(r['status'] == 'evaluated' for r in sampled),
                   'sampled_calls': [r['call'] for r in sampled]}
        sensitivity_records[tuple(float(v) for v in x)] = details
        if not details['all_candidate_evaluations_valid']:
            raise ValueError('A sensitivity sample failed; penalty Jacobian is not physical sensitivity')
        return np.column_stack(columns)

    first_jac = jacobian(starts[0])
    scaled_jac = first_jac*np.asarray(config['identifiability_parameter_scales'])
    singular = np.linalg.svd(scaled_jac, compute_uv=False)
    sensitivity = {'parameter_scaled_singular_values': singular.tolist(),
                   'numerical_rank': int(np.linalg.matrix_rank(scaled_jac)),
                   'condition_number': float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
                   'rank_is_not_structural_identifiability_proof': True,
                   'finite_difference_absolute_steps': h.tolist(),
                   **sensitivity_records[tuple(float(v) for v in starts[0])]}
    write(out/'initial_sensitivity.json', sensitivity)
    searches = []
    for index, start in enumerate(starts):
        try:
            result = least_squares(function, start, jac=jacobian, bounds=bounds,
                                   x_scale=config['optimization_parameter_scales'],
                                   ftol=1e-4, xtol=1e-4, gtol=1e-4,
                                   max_nfev=config['max_nfev_by_start'][index])
            searches.append({'start': start, 'x': result.x.tolist(), 'success': bool(result.success),
                             'message': str(result.message), 'cost': float(result.cost),
                             'optimality': float(result.optimality), 'nfev': int(result.nfev)})
        except (BudgetExceeded, ValueError) as error:
            searches.append({'start': start, 'status': 'budget_limit', 'reason': str(error)})
            break
    write(out/'optimization_searches.json', searches)
    profiles = []
    for fixed_scale in config.get('profile_fixed_scales', []):
        sampled = []
        def profile_objective(delta):
            residual, record = evaluate([fixed_scale, delta], reserve=4)
            sampled.append(record['call'])
            return float(residual@residual)
        try:
            result = minimize_scalar(profile_objective, bounds=(bounds[0,1], bounds[1,1]),
                                     method='bounded', options={'maxiter': 8, 'xatol': .005})
            profiles.append({'fixed_s_Q': fixed_scale, 'optimized_delta_Q_Li_Ah': float(result.x),
                             'objective_sum_squares': float(result.fun), 'success': bool(result.success),
                             'sampled_calls': sampled, 'message': str(result.message)})
        except BudgetExceeded as error:
            profiles.append({'fixed_s_Q': fixed_scale, 'status': 'budget_limit',
                             'sampled_calls': sampled, 'reason': str(error)})
            break
    write(out/'conditional_profiles.json', {'scope': 'Bounded C-only conditional nuisance optimization, not confidence intervals',
                                           'profiles': profiles})
    if best[0] is None:
        raise RuntimeError('No physically admissible candidate was evaluated')
    chosen = best[0]
    params = [chosen['s_Q'], chosen['delta_Q_Li_Ah']]
    try:
        final_jac = jacobian(params, reserve=1)
        final_singular = np.linalg.svd(final_jac*np.asarray(config['identifiability_parameter_scales']), compute_uv=False)
        final_sensitivity = {'parameter_scaled_singular_values': final_singular.tolist(),
                             'numerical_rank': int(np.linalg.matrix_rank(final_jac)),
                             'condition_number': float(final_singular[0]/final_singular[-1]) if final_singular[-1] > 0 else None,
                             'rank_is_not_structural_identifiability_proof': True,
                             **sensitivity_records[tuple(params)]}
    except (BudgetExceeded, ValueError) as error:
        final_sensitivity = {'status': 'not_evaluated', 'reason': str(error)}
    write(out/'final_sensitivity.json', final_sensitivity)
    frozen = {'scope': 'Effective C-RPT0 BOL calibration candidate, not identified material parameters or lifetime model',
              'fit_source_C_sha256': config['source_C_sha256'], 'code_sha256': config['code_sha256'],
              'declared_experiment_sha256': digest(out/'declared_experiment.json'),
              'runtime_environment_before_fit_sha256': digest(out/'runtime_environment_before_fit.json'),
              's_Q': params[0], 'delta_Q_Li_Ah': params[1],
              'parameter_updates': chosen['parameter_updates'], 'inventory': chosen['inventory'],
              'transport_and_degradation_parameters_changed': False,
              'C_only_fit': True, 'D_opened_before_freeze': False,
              'selected_by': 'lowest C-only fixed-objective residual among bounded candidates'}
    write(out/'parameters_frozen.json', frozen)
    frozen_sha = digest(out/'parameters_frozen.json')
    fine_residual, fine = evaluate(params, fine=True)
    verification = {'fine_candidate': fine, 'frozen_parameters_sha256': frozen_sha,
                    'C_and_D_absolute_capacity_voltage_only': {},
                    'SOH_curve_validation_complete': False, 'EOL_validation_complete': False,
                    'no_model_selection_or_refit_based_on_D': True}
    if fine['status'] == 'evaluated':
        with np.load(out/fine['raw_file']) as raw:
            fine_q, fine_v = raw['Q_Ah'], raw['V']
        for cell, path in [('C', source_c), ('D', Path(config['source_D_after_freeze']))]:
            expected_hash = config['source_C_sha256'] if cell == 'C' else config['source_D_expected_sha256']
            if digest(path) != expected_hash:
                raise ValueError('Measured raw source does not match frozen official receipt: '+cell)
            mq, mv = read_curve(path)
            if cell == 'D' and digest(path) == config['source_C_sha256']:
                raise ValueError('Held-out raw D is the calibration raw C')
            covered = (mq >= fine_q[0]) & (mq <= fine_q[-1])
            errors = np.interp(mq[covered], fine_q, fine_v)-mv[covered]
            verification['C_and_D_absolute_capacity_voltage_only'][cell] = {
                'source_sha256': digest(path), 'capacity_measured_Ah': float(mq[-1]),
                'capacity_model_Ah': float(fine_q[-1]), 'capacity_relative_error_percent': float(100*(fine_q[-1]/mq[-1]-1)),
                'voltage_coverage_fraction': float(np.mean(covered)),
                'voltage_RMSE_V_on_covered_points': float(np.sqrt(np.mean(errors**2))),
                'voltage_max_abs_error_V_on_covered_points': float(np.max(np.abs(errors))),
                'not_an_ageing_SOH_error': True}
            if digest(path) != expected_hash:
                raise ValueError('Measured source changed during validation: '+cell)
        verification['coarse_to_fine_capacity_difference_Ah'] = fine['Q_model_Ah']-chosen['Q_model_Ah']
    if digest(out/'parameters_frozen.json') != frozen_sha:
        raise ValueError('Frozen parameters changed during verification')
    if digest(__file__) != config['code_sha256'] or digest(parameter_module.__file__) != parameter_source_hash:
        raise ValueError('Code or base parameter source changed during execution')
    report = {'status': ('bounded_BOL_calibration_and_screen_completed' if fine['status'] == 'evaluated'
                         else 'fine_verification_failed'), 'model_calls': count[0],
              'baseline': baseline, 'chosen_C_candidate': chosen, 'initial_sensitivity': sensitivity,
              'final_sensitivity': final_sensitivity, 'conditional_profiles': profiles,
              'optimization_searches': searches, 'verification': verification,
              'optimizer_converged_for_all_starts': bool(searches) and all(s.get('success') for s in searches),
              'wall_s': time.monotonic()-began, 'physical_validation_complete': False,
              'SOH_curve_validation_complete': False, 'EOL_validation_complete': False, 'AWS_started': False,
              'limitations': config['limitations']}
    write(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--warm-start-run', type=Path)
    parser.add_argument('--protocol-evidence', type=Path)
    args = parser.parse_args()
    if args.worker:
        config = json.loads((args.output/'declared_experiment.json').read_text(encoding='utf-8'))
        try:
            report = worker(config, args.output)
            print(json.dumps({'status': report['status'], 'model_calls': report['model_calls'], 'wall_s': report['wall_s']}))
            return 0 if report['status'] == 'bounded_BOL_calibration_and_screen_completed' else 1
        except Exception as error:
            write(args.output/'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                               'traceback': traceback.format_exc(), 'physical_validation_complete': False})
            return 1
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    c_path = root/'work/external_m50_benchmark/kirkaldy_expt2_cellC_RPT0_C10.csv'
    model_path = Path(__file__).resolve().parents[1]/'configs/model.json'
    receipt_path = root/'work/external_m50_benchmark/kirkaldy_minimal_extract_manifest.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    d_receipt = [entry for entry in receipt['files'] if entry['local_file'] == 'kirkaldy_expt2_cellD_RPT0_C10.csv']
    if len(d_receipt) != 1 or '/cell D/' not in d_receipt[0]['path']:
        raise ValueError('Canonical raw D source receipt missing or ambiguous')
    starts = [[.97, 0.], [.93, -.1]]
    warm_provenance = None
    if args.warm_start_run:
        warm_path = args.warm_start_run.resolve()/'optimization_searches.json'
        searches = json.loads(warm_path.read_text(encoding='utf-8'))
        if len(searches) < 2 or any('x' not in search for search in searches[:2]):
            raise ValueError('Two C-only previous optimization endpoints required')
        starts = [search['x'] for search in searches[:2]]
        warm_provenance = {'C_only_optimization_searches': str(warm_path), 'sha256': digest(warm_path),
                           'D_metrics_not_read_for_warm_start': True}
    initial_rest_s, pre_discharge_rest_s, protocol_evidence = 0, 3600, None
    if args.protocol_evidence:
        evidence_path = args.protocol_evidence.resolve()
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        if evidence['cell'] != 'C' or evidence['processed_C_source_sha256'] != digest(c_path):
            raise ValueError('Source-derived protocol evidence does not apply to the C fit curve')
        if digest(evidence['source_native_path']) != evidence['source_native_sha256']:
            raise ValueError('Native protocol source changed after duration extraction')
        initial_rest_s = float(evidence['initial_rest_s'])
        pre_discharge_rest_s = float(evidence['pre_discharge_rest_s'])
        if not all(math.isfinite(v) and v > 0 for v in (initial_rest_s, pre_discharge_rest_s)):
            raise ValueError('Source-derived rest durations must be positive and finite')
        protocol_evidence = {'path': str(evidence_path), 'sha256': digest(evidence_path),
                             'source_native_sha256': evidence['source_native_sha256'],
                             'observed_initial_boundary_span_s': evidence['initial_rest_observed_boundary_span_s'],
                             'observed_pre_discharge_boundary_span_s': evidence['pre_discharge_rest_observed_boundary_span_s'],
                             'rounding_policy': evidence['rounding_policy'],
                             'paper_vs_records_discrepancy': evidence['paper_vs_records_discrepancy']}
    config = {'scope': 'C-only two-effective-parameter BOL candidate search',
              'source_C': str(c_path), 'source_C_sha256': digest(c_path),
              'source_D_after_freeze': str(c_path.with_name('kirkaldy_expt2_cellD_RPT0_C10.csv')),
              'source_D_expected_sha256': d_receipt[0]['sha256'],
              'canonical_source_receipt_sha256': digest(receipt_path),
              'source_D_native_archive_member': d_receipt[0]['path'],
              'code_sha256': digest(__file__), 'model_config_sha256': digest(model_path),
              'options': json.loads(model_path.read_text(encoding='utf-8'))['options'],
              'parameter_set': 'OKane2022', 'family': 'SPMe', 'initial_soc_assumption': .2,
              'initial_rest_s': initial_rest_s, 'pre_discharge_rest_s': pre_discharge_rest_s,
              'pre_discharge_rest_step_index': 3 if initial_rest_s > 0 else 2,
              'protocol_evidence': protocol_evidence,
              'protocol_basis': 'C-native-record-derived rest durations; nominal voltage/current endpoints' if protocol_evidence else 'paper nominal1h rest assumption',
              'bounds': [[.85, -.4], [1., .4]], 'baseline': [1., 0.],
              'starts': starts, 'max_nfev_by_start': [20, 20], 'warm_start_provenance': warm_provenance,
              'max_model_calls': 120, 'max_wall_s': 1800, 'profile_fixed_scales': [.86, .925, .99],
              'voltage_capacity_fraction_grid': [.02, .8, 81],
              'voltage_weight_scale_V': .02, 'capacity_weight_scale_Ah': .02,
              'weight_meaning': 'Engineering optimization objective only, not user SOH/lifetime acceptance limits',
              'jacobian_absolute_steps': [.001, .005], 'identifiability_parameter_scales': [.05, .1],
              'optimization_parameter_scales': [.05, .1], 'inventory_conservation_tolerance_Ah': 1e-5,
              'fit': {'mesh_points': dict(x_n=10, x_s=10, x_p=10, r_n=20, r_p=20),
                      'rtol': 1e-6, 'atol': 1e-8, 'output_period_s': 10},
              'fine': {'mesh_points': dict(x_n=20, x_s=20, x_p=20, r_n=40, r_p=40),
                       'rtol': 1e-7, 'atol': 1e-9, 'output_period_s': 1},
              'expected_terminations': ['event: Voltage > 4.2 [V] [experiment]',
                                        'event: abs(Current [A]) < 0.05 [A] [experiment]',
                                        'final time', 'event: Voltage < 2.5 [V] [experiment]'],
              'limitations': ['Assumed fresh SOC0.2 and no five-formation-cycle replay',
                              'Nominal25C isothermal rather than measured temperature history',
                              's_Q and inventory offset are effective initial-condition parameters, not unique measured material properties',
                              'Parameter bounds are declared physically admissible search limits, not literature confidence intervals',
                              'No transport/degradation acceleration or lifetime fitting',
                              'D raw RPT0 is opened only after freeze; no use of D ageing outcomes',
                              'Budget-limited optimizer termination or rank2 does not establish parameter identifiability']}
    if initial_rest_s > 0:
        config['expected_terminations'].insert(0, 'final time')
        config['limitations'].append('Paper1h rest differs from native C2h+30s; discrepancy cause unknown, rounded source durations used')
    write(out/'declared_experiment.json', config)
    (out/'calibration_source_frozen.py').write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(out)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (out/'worker.log').open('w', encoding='utf-8') as handle:
        try:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=config['max_wall_s'], creationflags=flags)
            return result.returncode
        except subprocess.TimeoutExpired:
            write(out/'failure.json', {'status': 'wall_limit', 'max_wall_s': config['max_wall_s'],
                                      'physical_validation_complete': False, 'AWS_started': False})
            return 2


if __name__ == '__main__':
    raise SystemExit(main())
