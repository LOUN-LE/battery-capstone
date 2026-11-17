import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from empirical_validation import MATCH_FIELDS
from score_empirical_validation import score_cell, score_suite


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        artifact = self.root/'frozen.json'
        artifact.write_text('{"fit":"C only"}')
        self.criteria = json.loads((Path(__file__).resolve().parents[1]/'configs/empirical_acceptance.json').read_text(encoding='utf-8'))
        metadata = {k: k for k in MATCH_FIELDS}
        metadata.update(axis_column='EFC', axis_unit='EFC')
        self.case = {'cell_id': 'D', 'experiment_id': 'Expt2', 'condition_id': '25C_70_85',
                     'axis_column': 'EFC', 'measured_metadata': metadata,
                     'predicted_metadata': metadata.copy(),
                     'measured': [{'EFC': 0, 'SOH_percent': 100}, {'EFC': 100, 'SOH_percent': 90}],
                     'predicted': [{'EFC': 0, 'SOH_percent': 100}, {'EFC': 100, 'SOH_percent': 90}],
                     'model': {'id': 'frozen_model', 'kind': 'empirical_baseline',
                               'artifact_path': 'frozen.json', 'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                               'fit_cell_ids': ['C'], 'fit_condition_ids': ['25C_70_85'],
                               'frozen_before_evaluation': True, 'retuned_on_validation': False,
                               'heldout_initialization': 'RPT0_Q0_only'}}

    def score(self):
        return score_cell(self.case, self.criteria, self.root)

    def test_valid_curve_and_missing80_are_separate(self):
        result = self.score()
        self.assertEqual(result['curve_basic'], 'passed')
        self.assertEqual(result['eol']['status'], 'not_evaluated')
        self.assertFalse(result['physical_validation_complete'])
        self.assertIn('baseline result', result['statement'])

    def test_both_curve_limits_required_at_inclusive_boundary(self):
        self.case['predicted'][1]['SOH_percent'] = 92
        result = self.score()
        self.assertEqual(result['metrics']['max_abs_error_pp'], 2)
        self.assertEqual(result['curve_basic'], 'failed')  # RMSE=sqrt(2)>1
        self.case['predicted'][1]['SOH_percent'] = 91
        self.assertEqual(self.score()['curve_basic'], 'passed')
        self.assertEqual(self.score()['curve_challenge'], 'failed')

    def test_validation_cell_cannot_be_calibration_cell(self):
        self.case['model']['fit_cell_ids'] = ['D']
        self.assertEqual(self.score()['status'], 'deferred')

    def test_flags_do_not_replace_artifact_evidence(self):
        self.case['model']['artifact_sha256'] = '0'*64
        self.assertEqual(self.score()['status'], 'deferred')
        self.case['model']['artifact_path'] = 'not_found.json'
        self.assertEqual(self.score()['status'], 'deferred')

    def test_retuning_and_future_heldout_initialization_rejected(self):
        self.case['model']['retuned_on_validation'] = True
        self.assertEqual(self.score()['status'], 'deferred')
        self.case['model']['retuned_on_validation'] = False
        self.case['model']['heldout_initialization'] = 'use_all_heldout_capacity'
        self.assertEqual(self.score()['status'], 'deferred')

    def test_bad_input_is_deferred_not_failed_or_censored(self):
        self.case['measured'][1]['SOH_percent'] = float('nan')
        self.assertEqual(self.score()['status'], 'deferred')

    def test_observed80_with_unjustified_interval_is_deferred(self):
        for points in (self.case['measured'], self.case['predicted']):
            points[1]['SOH_percent'] = 70
        self.assertEqual(self.score()['eol']['status'], 'deferred')

    def test_locked_linear_EFC80_and_error(self):
        self.criteria['eol']['max_observed_bracket_EFC'] = 100
        self.case['measured'][1]['SOH_percent'] = 70
        self.case['predicted'][1]['SOH_percent'] = 72
        eol = self.score()['eol']
        self.assertAlmostEqual(eol['measured']['EFC80'], 200/3)
        self.assertAlmostEqual(eol['relative_error_percent'], 50/7)
        self.assertEqual(eol['basic'], 'passed')
        self.assertEqual(eol['challenge'], 'failed')
        self.assertEqual(eol['measured']['bracket_width_EFC'], 100)

    def test_cycles_or_days_cannot_be_called_EFC80(self):
        self.criteria['eol']['max_observed_bracket_EFC'] = 100
        self.case['measured_metadata']['axis_unit'] = 'cycles'
        self.case['predicted_metadata']['axis_unit'] = 'cycles'
        for points in (self.case['measured'], self.case['predicted']):
            points[1]['SOH_percent'] = 70
        self.assertEqual(self.score()['eol']['status'], 'deferred')

    def test_one_bad_cell_cannot_hide_in_pooled_mean(self):
        bad = copy.deepcopy(self.case)
        bad['cell_id'] = 'E'
        bad['predicted'][1]['SOH_percent'] = 95
        result = score_suite([self.case, bad], self.criteria, self.root)
        self.assertEqual(result['stages']['stage1_same_condition'], 'failed')
        self.assertFalse(result['staged_2A_curve_scope_complete'])

    def test_scope_staged_and_same_frozen_model(self):
        result = score_suite([self.case], self.criteria, self.root)
        self.assertEqual(result['scope'], 'same_condition_only')
        second = copy.deepcopy(self.case)
        second.update(cell_id='E', condition_id='40C_70_85', heldout_condition_axis='temperature')
        result = score_suite([self.case, second], self.criteria, self.root)
        self.assertTrue(result['staged_2A_curve_scope_complete'])
        self.assertFalse(result['physical_validation_complete'])
        second['model']['id'] = 'retuned_model'
        result = score_suite([self.case, second], self.criteria, self.root)
        self.assertEqual(result['scope'], 'no_passed_scope')

    def test_duplicate_cell_does_not_count_as_multiple_validation_cells(self):
        result = score_suite([self.case, self.case], self.criteria, self.root)
        self.assertEqual(result['stages']['stage1_same_condition'], 'deferred')

    def test_bad_provenance_shape_and_boolean_values_defer(self):
        self.case['model'] = None
        self.assertEqual(self.score()['status'], 'deferred')
        self.assertEqual(score_cell([], self.criteria, self.root)['status'], 'deferred')

    def test_wide_observed_interval_and_recrossing_defer(self):
        self.criteria['eol']['max_observed_bracket_EFC'] = 50
        self.case['measured'][1]['SOH_percent'] = 70
        self.case['predicted'][1]['SOH_percent'] = 70
        self.assertEqual(self.score()['eol']['status'], 'deferred')
        self.criteria['eol']['max_observed_bracket_EFC'] = 100
        self.case['measured'].append({'EFC': 200, 'SOH_percent': 81})
        self.case['predicted'].append({'EFC': 200, 'SOH_percent': 81})
        self.assertIn('recrosses', self.score()['eol']['reason'])

    def test_prediction_without_EFC80_is_not_extrapolated(self):
        self.criteria['eol']['max_observed_bracket_EFC'] = 100
        self.case['measured'][1]['SOH_percent'] = 70
        self.assertEqual(self.score()['eol']['status'], 'not_evaluated')

    def test_boolean_point_and_nonfinite_derived_metrics_defer(self):
        self.case['predicted'][1]['SOH_percent'] = True
        self.assertEqual(self.score()['status'], 'deferred')
        self.case['predicted'][1]['SOH_percent'] = 1e300
        self.assertEqual(self.score()['status'], 'deferred')

    def test_renamed_condition_or_current_only_does_not_meet_stage2(self):
        self.case['condition_id'] = 'different_current_only'
        self.case['heldout_condition_axis'] = 'current'
        self.assertEqual(self.score()['status'], 'deferred')


if __name__ == '__main__':
    unittest.main()
