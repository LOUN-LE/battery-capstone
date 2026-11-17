"""Consolidate local evidence without promoting software/BOL/baselines to lifetime validation.

Current policy excludes AWS and team release. A synthetic-tested backend is not
a measured physical ageing fit or independent SOH/lifetime validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROLES = ('proposal', 'criteria', 'software', 'BOL', 'BOL_post_audit', 'calibrated_numerics', 'numerical_declaration',
         'history', 'baseline_scores', 'clock', 'clock_post_audit', 'EOL_data',
         'physical_backend_report', 'physical_backend_assessment', 'physical_backend_experiment', 'physical_backend_parameters')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(paths, root):
    """Lock reviewed input bytes explicitly before running the assessor."""
    root = Path(root).resolve()
    result = {}
    for role in ROLES:
        path = Path(paths[role]).resolve()
        result[role] = {'path': str(path.relative_to(root)), 'sha256': digest(path)}
    return result


def checked_inputs(manifest, root):
    if set(manifest) != set(ROLES):
        raise ValueError('Exactly the required evidence roles must be supplied')
    root = Path(root).resolve()
    data, sources = {}, {}
    for role in ROLES:
        declared = manifest[role]
        if not isinstance(declared, dict) or set(declared) != {'path', 'sha256'}:
            raise ValueError('Explicit path and SHA required for '+role)
        path = (root/declared['path']).resolve()
        path.relative_to(root)  # No outside-workspace evidence paths.
        expected = declared['sha256']
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or not path.is_file():
            raise ValueError('Missing evidence or malformed SHA for '+role)
        actual = digest(path)
        if actual != expected:
            raise ValueError('Evidence hash drift for '+role)
        sources[role] = {'path': str(path.relative_to(root)), 'sha256': actual}
        data[role] = None if role == 'proposal' else json.loads(path.read_text(encoding='utf-8-sig'))
    return data, sources


def all_checks(value):
    return isinstance(value, dict) and bool(value) and all(flag is True for flag in value.values())


def backend_runtime_stage(data, sources):
    report, assessment = data['physical_backend_report'], data['physical_backend_assessment']
    experiment, parameters = data['physical_backend_experiment'], data['physical_backend_parameters']
    if (report.get('parameters_frozen_sha256') != sources['physical_backend_parameters']['sha256'] or
            parameters.get('experiment_frozen_sha256') != sources['physical_backend_experiment']['sha256']):
        raise ValueError('Backend report, frozen parameters and experiment do not belong together')
    if (assessment.get('actual_model_calls') != report.get('model_calls') or
            assessment.get('optimizer') != report.get('optimizer') or
            parameters.get('optimizer') != report.get('optimizer') or
            assessment.get('fitted_D') != parameters.get('parameters', {}).get('SEI solvent diffusivity [m2.s-1]')):
        raise ValueError('Backend runtime assessment belongs to different fit evidence')
    if (report.get('data_kind') != 'synthetic' or report.get('scope') != 'synthetic_backend_runtime_only' or
            report.get('physical_validation_complete') is not False or report.get('SOH_lifetime_validation_complete') is not False):
        raise ValueError('Current-phase backend evidence must remain explicitly synthetic, not a physical certification')
    hashes = experiment.get('source_hashes', {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Backend frozen source hashes required')
    changed = [name for name, expected in hashes.items() if not Path(name).is_file() or digest(Path(name)) != expected]
    runtime_pass = (report.get('status') == 'backend_completed' and report.get('data_kind') == 'synthetic' and
        report.get('scope') == 'synthetic_backend_runtime_only' and report.get('physical_validation_complete') is False and
        report.get('SOH_lifetime_validation_complete') is False and report.get('source_hashes_unchanged') is True and
        report.get('optimizer', {}).get('success') is True and not report.get('heldout_failures') and
        bool(report.get('heldout_results')) and len(report.get('calibration', {}).get('measurements', [])) == 2 and
        assessment.get('software_checks_pass') is True and all_checks(assessment.get('checks')) and
        assessment.get('checks', {}).get('two_actual_full_RPT_measurements') is True and
        assessment.get('physical_validation_complete') is False and assessment.get('SOH_lifetime_validation_complete') is False)
    return {'status': ('historical_synthetic_runtime' if changed else
                       'synthetic_runtime_verified_current_sources' if runtime_pass else 'synthetic_runtime_incomplete'),
            'scope': 'Synthetic full-RPT backend execution only; no independent recovery or measured-cell validation claim',
            'optimizer_complete': report.get('optimizer', {}).get('success') is True,
            'current_source_hashes_match': not changed, 'changed_source_paths': changed,
            'actual_model_calls': report.get('model_calls'), 'physical_validation_complete': False}


def assess(manifest, root):
    data, sources = checked_inputs(manifest, root)
    criteria = data['criteria']
    if criteria.get('source', {}).get('sha256') != sources['proposal']['sha256']:
        raise ValueError('Criteria are not tied to the supplied proposal PDF')
    if criteria.get('basic') != {'SOH_RMSE_pp_max': 1., 'SOH_max_abs_error_pp_max': 2.,
                                'observed_EFC80_relative_error_percent_max': 10.}:
        raise ValueError('This assessor requires the user-selected basic1A criteria')
    clock, clock_post = data['clock'], data['clock_post_audit']
    if clock_post.get('clock_audit_report_SHA256') != sources['clock']['sha256']:
        raise ValueError('Clock post-audit belongs to different evidence')
    bol_verification = data['BOL'].get('verification', {})
    bol_post = data['BOL_post_audit']
    if bol_post.get('fine_direct_RPT_capacity_Ah') != bol_verification.get('fine_candidate', {}).get('Q_model_Ah'):
        raise ValueError('BOL post-audit does not match the supplied fine candidate')
    for cell in ('C', 'D'):
        audited = bol_post.get('raw_sources', {}).get(cell, {}).get('actual_sha256')
        reported = bol_verification.get('C_and_D_absolute_capacity_voltage_only', {}).get(cell, {}).get('source_sha256')
        if not isinstance(audited, str) or audited != reported:
            raise ValueError('BOL post-audit raw cell source mismatch: '+cell)
    declaration = data['numerical_declaration']
    declared_hashes = declaration.get('source_sha256', {})
    if not declared_hashes:
        raise ValueError('Numerical declaration has no frozen source hashes')
    normalized = {}
    for name, expected in declared_hashes.items():
        path = Path(name).resolve(); relative = path.relative_to(Path(root).resolve())
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Numerical declared source drift: '+str(relative))
        normalized[path] = expected
    if normalized.get((Path(root)/sources['BOL']['path']).resolve()) != sources['BOL']['sha256']:
        raise ValueError('Numerical experiment used a different BOL report')
    parameters = Path(declaration.get('parameters_path', '')).resolve()
    if normalized.get(parameters) != bol_verification.get('frozen_parameters_sha256'):
        raise ValueError('Numerical experiment used different frozen BOL parameters')
    if not all(suite.get('criteria_id') == criteria['criteria_id'] for suite in
               data['baseline_scores'].get('score_suites', {}).values()):
        raise ValueError('Baseline scores use different acceptance criteria')
    software = data['software']; suites = software.get('suites', [])
    software_hashes = software.get('source_hashes', {})
    if not isinstance(software_hashes, dict):
        raise ValueError('Software source-hash manifest must be a mapping')
    normalized_software_sources = set()
    for name, expected in software_hashes.items():
        path = (Path(root)/name).resolve(); path.relative_to(Path(root).resolve())
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Software tested source drift: '+name)
        normalized_software_sources.add(path)
    assessor_covered = (Path(root)/'repo/simulation/assess_m50t_readiness.py').resolve() in normalized_software_sources
    software_pass = (software.get('all_pass') is True and software.get('source_hashes_stable') is True and
        bool(software_hashes) and assessor_covered and bool(suites) and
        all(s.get('passed') is True and s.get('exit_code') == 0 and s.get('tests', 0) > 0 for s in suites)
        and software.get('total_tests') == sum(s['tests'] for s in suites))
    clock_pass = (clock.get('all_available_MPT_clock_mappings_verified') is True and
        clock.get('parsed_count') == 126 and clock.get('MPT_pairs_verified') == 98 and
        clock_post.get('all_pass') is True and all_checks(clock_post.get('checks')))
    bol_audit = data['BOL_post_audit']
    bol_completed = (data['BOL'].get('status') == 'bounded_BOL_calibration_and_screen_completed' and
        bol_audit.get('all_execution_checks_pass') is True and all_checks(bol_audit.get('checks')))
    numerical = data['calibrated_numerics']
    cases = numerical.get('cases', [])
    span, limit = numerical.get('SOH_span_pp'), numerical.get('SOH_span_screen_limit_pp')
    case_sohs = [c.get('SOH_percent') for c in cases]
    valid_sohs = bool(case_sohs) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in case_sohs)
    initial_states_match = all(len({c.get('initial_fingerprint') for c in cases if c.get('mesh') == mesh}) == 1
                               for mesh in (4, 8)) and all(
        isinstance(c.get('initial_fingerprint'), str) and re.fullmatch('[0-9a-f]{64}', c['initial_fingerprint']) for c in cases)
    numerical_pass = (numerical.get('status') == 'screen_passed' and numerical.get('numerical_short_screen_pass') is True and
        numerical.get('all_four_completed') is True and numerical.get('same_initial_state_within_each_mesh') is True and
        len(cases) == 4 and {(c.get('mesh'), c.get('rtol')) for c in cases} ==
        {(4, 1e-6), (4, 1e-7), (8, 1e-6), (8, 1e-7)} and all(
            c.get('status') == 'completed' and c.get('exit_code') == 0 and
            c.get('virtual_RPTs_main_state_unchanged') is True for c in cases) and
        valid_sohs and initial_states_match and isinstance(span, (int, float)) and isinstance(limit, (int, float)) and
        math.isclose(max(case_sohs)-min(case_sohs), span, rel_tol=1e-12, abs_tol=1e-12) and
        math.isfinite(span) and math.isfinite(limit) and 0 <= span <= limit == 1e-4)
    baseline = data['baseline_scores']
    baseline_cells = [cell for suite in baseline.get('score_suites', {}).values() for cell in suite.get('cells', [])]
    if baseline.get('model_kind') != 'empirical_baseline' or not baseline_cells:
        raise ValueError('Expected explicitly labelled empirical baseline evidence')
    if baseline.get('physical_validation_complete') is not False or any(
            cell.get('physical_validation_complete') is not False for cell in baseline_cells):
        raise ValueError('Baseline metrics must not be labelled physical validation')
    eol_files = data['EOL_data'].get('files', [])
    if data['EOL_data'].get('status') != 'selected_members_retrieved_and_verified' or not eol_files:
        raise ValueError('Verified measured EOL data summary required, even if no crossing exists')
    observed = [f.get('cell') for f in eol_files if f.get('EOL80', {}).get('status') == 'observed_bracket']
    history = data['history']
    backend = backend_runtime_stage(data, sources)
    stages = {
        'software_regression': {'status': 'passed' if software_pass else 'failed',
            'total_tests': software.get('total_tests'), 'scope': 'software only',
            'current_source_hashes_checked': len(software_hashes), 'assessor_in_tested_source_manifest': assessor_covered},
        'native_clock_mapping': {'status': 'passed' if clock_pass else 'failed', 'scope': 'clock/source identity only'},
        'initial_capacity_voltage': {'status': 'screen_completed' if bol_completed else 'deferred',
            'scope': 'BOL execution/initial-state fit only; no lifetime acceptance',
            'limitations': data['BOL'].get('limitations', [])},
        'calibrated_short_numerics': {'status': 'short_screen_passed' if numerical_pass else 'deferred',
            'scope': 'finite short numerical check only', 'reported_status': numerical.get('status')},
        'measured_history': {'status': 'recorded_history_compiled' if history.get('compiler_completed') is True else 'deferred',
            'full_history_replay_admissible': history.get('full_history_replay_admissible') is True,
            'limitations': history.get('admissibility_reasons', [])},
        'empirical_baseline': {'status': 'evaluated', 'scope': 'baseline metrics only',
            'cells': [{key: cell.get(key) for key in ('model_id', 'cell_id', 'condition_id', 'stage', 'status', 'metrics')}
                      for cell in baseline_cells]},
        'physical_backend_software': backend,
        'physical_ageing_SOH': {'status': 'not_evaluated',
            'reasons': ['Matched measured physical ageing calibration and held-out prediction artifacts are absent',
                        'Backend is implemented; supplied evidence covers synthetic full-RPT execution only',
                        'A BOL fit, a short numerical screen and an empirical baseline cannot substitute for them']},
        'physical_EFC80': {'status': 'not_evaluated', 'observed_measured_cells': observed,
            'reason': 'Measured crossings alone are not matched frozen physical-model EOL predictions' if observed else
                      'No observed crossing in supplied data; extrapolated EOL cannot pass observed lifetime validation'}}
    if history.get('full_history_replay_admissible') is not True:
        stages['physical_ageing_SOH']['reasons'].append('Full recorded-history physical replay is not admissible')
    return {'status': 'deferred_incomplete_physical_validation', 'input_sources': sources, 'stages': stages,
            'code_fully_complete': False, 'release_ready': False, 'physical_validation_complete': False,
            'AWS_launch_allowed_current_policy': False, 'AWS_started': False,
            'generalization': 'No validated physical-model generalization. Baseline results apply only to reported cases.',
            'assessor_source_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
        report = assess(manifest, args.evidence_root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {'status': 'invalid_or_unmatched_evidence', 'reason': f'{type(error).__name__}: {error}',
                  'code_fully_complete': False, 'release_ready': False, 'AWS_launch_allowed_current_policy': False}
    report['manifest_sha256'] = digest(args.manifest) if args.manifest.is_file() else None
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'code_fully_complete', 'release_ready', 'AWS_launch_allowed_current_policy')}, indent=2))
    return 2  # Incomplete release is intentional and must never be a successful launch gate.


if __name__ == '__main__':
    raise SystemExit(main())
