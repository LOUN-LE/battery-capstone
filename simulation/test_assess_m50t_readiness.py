import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from assess_m50t_readiness import ROLES, assess, digest, make_manifest


def fixture(root):
    proposal = root/'proposal.pdf'
    proposal.write_bytes(b'synthetic proposal fixture')
    parameters = root/'parameters.json'; parameters.write_text('{"parameter": 1}')
    tested_source = root/'repo/simulation/assess_m50t_readiness.py'
    tested_source.parent.mkdir(parents=True)
    tested_source.write_bytes(Path(__file__).with_name('assess_m50t_readiness.py').read_bytes())
    criteria = {'criteria_id': 'fixture_1A', 'source': {'sha256': digest(proposal)},
                'basic': {'SOH_RMSE_pp_max': 1., 'SOH_max_abs_error_pp_max': 2.,
                          'observed_EFC80_relative_error_percent_max': 10.}}
    baseline_cell = {'cell_id': 'D', 'condition_id': 'same', 'model_id': 'baseline', 'stage': 'stage1',
                     'status': 'passed', 'physical_validation_complete': False, 'metrics': {'RMSE_pp': .2}}
    data = {'criteria': criteria,
        'software': {'all_pass': True, 'source_hashes_stable': True,
                     'source_hashes': {'repo/simulation/assess_m50t_readiness.py': digest(tested_source)},
                     'total_tests': 2, 'suites': [{'passed': True, 'exit_code': 0, 'tests': 2}]},
        'BOL': {'status': 'bounded_BOL_calibration_and_screen_completed',
                'verification': {'fine_candidate': {'Q_model_Ah': 4.8}, 'frozen_parameters_sha256': digest(parameters),
                    'C_and_D_absolute_capacity_voltage_only': {
                        'C': {'source_sha256': '1'*64}, 'D': {'source_sha256': '2'*64}}}},
        'BOL_post_audit': {'all_execution_checks_pass': True, 'checks': {'execution': True},
                          'fine_direct_RPT_capacity_Ah': 4.8,
                          'raw_sources': {'C': {'actual_sha256': '1'*64}, 'D': {'actual_sha256': '2'*64}}},
        'calibrated_numerics': {'status': 'screen_passed', 'numerical_short_screen_pass': True, 'all_four_completed': True,
            'same_initial_state_within_each_mesh': True, 'SOH_span_pp': 1e-7, 'SOH_span_screen_limit_pp': 1e-4,
            'cases': [{'mesh': mesh, 'rtol': rtol, 'status': 'completed', 'exit_code': 0,
                       'virtual_RPTs_main_state_unchanged': True, 'initial_fingerprint': f'{mesh:064x}',
                       'SOH_percent': 99.9+(1e-7 if mesh == 8 else 0)} for mesh in (4, 8) for rtol in (1e-6, 1e-7)]},
        'history': {'compiler_completed': True, 'full_history_replay_admissible': False,
                    'admissibility_reasons': ['unobserved gaps']},
        'baseline_scores': {'model_kind': 'empirical_baseline', 'physical_validation_complete': False,
            'score_suites': {'baseline': {'criteria_id': 'fixture_1A', 'cells': [baseline_cell]}}},
        'clock': {'all_available_MPT_clock_mappings_verified': True, 'parsed_count': 126, 'MPT_pairs_verified': 98},
        'EOL_data': {'status': 'selected_members_retrieved_and_verified', 'files': [
            {'cell': 'D', 'EOL80': {'status': 'right_censored'}}]}}
    paths = {'proposal': proposal}
    for key, value in data.items():
        path = root/(key+'.json'); path.write_text(json.dumps(value), encoding='utf-8'); paths[key] = path
    path = root/'clock_post_audit.json'
    path.write_text(json.dumps({'clock_audit_report_SHA256': digest(paths['clock']),
                               'all_pass': True, 'checks': {'hashes': True}}), encoding='utf-8')
    paths['clock_post_audit'] = path
    path = root/'numerical_declaration.json'
    path.write_text(json.dumps({'parameters_path': str(parameters), 'source_sha256': {
        str(paths['BOL']): digest(paths['BOL']), str(parameters): digest(parameters)}}), encoding='utf-8')
    paths['numerical_declaration'] = path
    backend_source = root/'backend_source.py'; backend_source.write_text('SYNTHETIC_VERSION = 1')
    def put(role, value):
        path = root/(role+'.json'); path.write_text(json.dumps(value), encoding='utf-8'); paths[role] = path
    put('physical_backend_experiment', {'source_hashes': {str(backend_source): digest(backend_source)}})
    put('physical_backend_parameters', {'parameters': {'SEI solvent diffusivity [m2.s-1]': 2.5e-22},
        'optimizer': {'success': True}, 'experiment_frozen_sha256': digest(paths['physical_backend_experiment'])})
    put('physical_backend_report', {'status': 'backend_completed', 'data_kind': 'synthetic',
        'scope': 'synthetic_backend_runtime_only', 'model_calls': 5, 'source_hashes_unchanged': True,
        'parameters_frozen_sha256': digest(paths['physical_backend_parameters']), 'optimizer': {'success': True},
        'heldout_failures': [], 'heldout_results': [{}], 'calibration': {'measurements': [{}, {}]},
        'physical_validation_complete': False, 'SOH_lifetime_validation_complete': False})
    put('physical_backend_assessment', {'actual_model_calls': 5, 'optimizer': {'success': True}, 'fitted_D': 2.5e-22,
        'software_checks_pass': True, 'checks': {'two_actual_full_RPT_measurements': True},
        'physical_validation_complete': False, 'SOH_lifetime_validation_complete': False})
    return paths, make_manifest(paths, root)


class ReadinessTests(unittest.TestCase):
    def test_all_software_BOL_and_baseline_green_do_not_make_physical_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); _, manifest = fixture(root)
            result = assess(manifest, root)
            self.assertEqual(result['stages']['software_regression']['status'], 'passed')
            self.assertEqual(result['stages']['physical_backend_software']['status'], 'synthetic_runtime_verified_current_sources')
            self.assertFalse(result['code_fully_complete'])
            self.assertFalse(result['release_ready'])
            self.assertFalse(result['AWS_launch_allowed_current_policy'])
            self.assertEqual(result['stages']['physical_ageing_SOH']['status'], 'not_evaluated')
            self.assertEqual(result['stages']['physical_EFC80']['status'], 'not_evaluated')

    def test_missing_or_hash_drifted_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, manifest = fixture(root)
            broken = dict(manifest); broken.pop('history')
            with self.assertRaises(ValueError):
                assess(broken, root)
            paths['history'].write_text('{}')
            with self.assertRaises(ValueError):
                assess(manifest, root)

    def test_unmatched_clock_post_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            paths['clock_post_audit'].write_text(json.dumps({'clock_audit_report_SHA256': '0'*64}))
            with self.assertRaises(ValueError):
                assess(make_manifest(paths, root), root)

    def test_proposal_or_criteria_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['criteria'].read_text())
            value['source']['sha256'] = '0'*64
            paths['criteria'].write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                assess(make_manifest(paths, root), root)

    def test_observed_EOL_is_data_availability_not_model_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            paths['EOL_data'].write_text(json.dumps({'status': 'selected_members_retrieved_and_verified',
                                                   'files': [{'cell': 'E', 'EOL80': {'status': 'observed_bracket'}}]}))
            result = assess(make_manifest(paths, root), root)
            self.assertEqual(result['stages']['physical_EFC80']['observed_measured_cells'], ['E'])
            self.assertEqual(result['stages']['physical_EFC80']['status'], 'not_evaluated')

    def test_BOL_post_audit_from_different_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['BOL_post_audit'].read_text())
            value['fine_direct_RPT_capacity_Ah'] = 5.1
            paths['BOL_post_audit'].write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                assess(make_manifest(paths, root), root)

    def test_frozen_numerical_parameter_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            (root/'parameters.json').write_text('{"parameter": 2}')
            with self.assertRaises(ValueError):
                assess(make_manifest(paths, root), root)

    def test_baseline_mislabelled_physical_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['baseline_scores'].read_text())
            value['physical_validation_complete'] = True
            paths['baseline_scores'].write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                assess(make_manifest(paths, root), root)

    def test_numerical_green_flag_cannot_hide_mismatched_initial_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['calibrated_numerics'].read_text())
            value['cases'][0]['initial_fingerprint'] = '0'*64
            paths['calibrated_numerics'].write_text(json.dumps(value))
            result = assess(make_manifest(paths, root), root)
            self.assertEqual(result['stages']['calibrated_short_numerics']['status'], 'deferred')

    def test_tested_software_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            (root/'repo/simulation/assess_m50t_readiness.py').write_text('CHANGED = True')
            with self.assertRaisesRegex(ValueError, 'Software tested source drift'):
                assess(make_manifest(paths, root), root)

    def test_software_source_stability_flag_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['software'].read_text()); value['source_hashes_stable'] = False
            paths['software'].write_text(json.dumps(value))
            self.assertEqual(assess(make_manifest(paths, root), root)['stages']['software_regression']['status'], 'failed')

    def test_historical_backend_sources_are_not_a_current_runtime_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            (root/'backend_source.py').write_text('SYNTHETIC_VERSION = 2')
            result = assess(make_manifest(paths, root), root)
            backend = result['stages']['physical_backend_software']
            self.assertEqual(backend['status'], 'historical_synthetic_runtime')
            self.assertFalse(backend['current_source_hashes_match'])
            self.assertFalse(result['release_ready'])

    def test_backend_frozen_parameter_experiment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            value = json.loads(paths['physical_backend_parameters'].read_text())
            value['experiment_frozen_sha256'] = '0'*64
            paths['physical_backend_parameters'].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'do not belong together'):
                assess(make_manifest(paths, root), root)

    def test_optimizer_incomplete_cannot_be_a_current_runtime_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); paths, _ = fixture(root)
            parameters = json.loads(paths['physical_backend_parameters'].read_text())
            parameters['optimizer'] = {'success': False}
            paths['physical_backend_parameters'].write_text(json.dumps(parameters))
            report = json.loads(paths['physical_backend_report'].read_text())
            report.update(optimizer={'success': False}, parameters_frozen_sha256=digest(paths['physical_backend_parameters']))
            paths['physical_backend_report'].write_text(json.dumps(report))
            assessment = json.loads(paths['physical_backend_assessment'].read_text())
            assessment['optimizer'] = {'success': False}
            paths['physical_backend_assessment'].write_text(json.dumps(assessment))
            result = assess(make_manifest(paths, root), root)
            self.assertEqual(result['stages']['physical_backend_software']['status'], 'synthetic_runtime_incomplete')
            self.assertFalse(result['stages']['physical_backend_software']['optimizer_complete'])


if __name__ == '__main__':
    unittest.main()
