"""Tests of actual objective/normalization and fit/heldout separation; no PDE."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from physical_ageing_campaign import (BudgetExceeded, campaign_prediction, digest, fit_campaign,
                                     validate_campaign, validate_parameters)
from test_physical_campaign_admission import fixture as synthetic_campaign


class CampaignFitTests(unittest.TestCase):
    def test_only_supported_active_degradation_parameters_with_physical_bounds(self):
        protocol = {'parameter_set': 'OKane2022', 'options': {'SEI': 'solvent-diffusion limited'}}
        definition = {'name': 'SEI solvent diffusivity [m2.s-1]', 'lower': 2e-22, 'upper': 3e-22,
                      'initial': 2.5e-22, 'scale': 1e-22, 'difference_step_scaled': .1,
                      'unit': 'm2.s-1', 'source_ref': 'synthetic narrow nominal interval'}
        self.assertEqual(validate_parameters([definition], protocol), [definition])
        for change in ({'name': 'Current function [A]'}, {'unit': 'A'}, {'lower': 0},
                       {'upper': 2e-22}, {'initial': float('nan')}, {'source_ref': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_parameters([{**definition, **change}], protocol)
        with self.assertRaisesRegex(ValueError, 'collides'):
            validate_parameters([definition], {**protocol, 'parameter_overrides': {definition['name']: 2.5e-22}})
        with self.assertRaisesRegex(ValueError, 'explicit'):
            validate_parameters([definition], {**protocol, 'options': {'SEI': 'none'}})

    def test_fixed_model_Q0_is_not_rescaled_to_heldout_target_Q0(self):
        c = {'data_kind': 'synthetic', 'cell_id': 'D', 'condition_id': 'toy',
             'measurement_steps': [{'rpt_id': 'RPT0', 'step_index': 0}, {'rpt_id': 'RPT1', 'step_index': 1}],
             '_targets_rows': [{'Qref_Ah': 2, 'SOH_percent': 100}, {'Qref_Ah': 1.6, 'SOH_percent': 80}]}
        result = {'status': 'replay_completed', 'records': [
            {'role': 'rpt', 'propagated': True, 'charge_Ah': 0, 'discharge_Ah': 1},
            {'role': 'rpt', 'propagated': True, 'charge_Ah': 0, 'discharge_Ah': .9}], 'throughput_by_role': {}}
        report = campaign_prediction(c, result)
        self.assertEqual(report['model_Q0_Ah'], 1)
        self.assertEqual(report['Q0_relative_error_percent'], -50)
        self.assertEqual(report['measurements'][1]['normalized_percent'], 90)
        self.assertEqual(report['measurements'][1]['capacity_error_Ah'], -.7000000000000001)
        self.assertFalse(report['physical_validation_complete'])
        self.assertEqual(report['quantity_label'], 'synthetic_probe_ratio_percent')
        result['records'][1]['propagated'] = False
        with self.assertRaisesRegex(ValueError, 'propagation'):
            campaign_prediction(c, result)

    def test_C_only_optimizer_and_D_open_only_after_parameter_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign_c = synthetic_campaign(base)
            campaign_c['protocol']['options']['SEI'] = 'solvent-diffusion limited'
            Path(campaign_c['targets']['path']).write_text('rpt_id,Qref_Ah\nRPT0,1\nRPT1,0.975\n', encoding='utf-8')
            campaign_c['targets']['sha256'] = digest(campaign_c['targets']['path'])
            campaign_d = copy.deepcopy(campaign_c); campaign_d['cell_id'] = 'D'
            (base/'D.csv').write_text('rpt_id,Qref_Ah\nRPT0,2\nRPT1,1.6\n', encoding='utf-8')
            campaign_d['targets'] = {'path': 'D.csv', 'sha256': digest(base/'D.csv')}
            for label, value in [('C', campaign_c), ('D', campaign_d)]:
                (base/f'{label}.json').write_text(json.dumps(value), encoding='utf-8')
            criteria = {'basic': {'SOH_RMSE_pp_max': 1, 'SOH_max_abs_error_pp_max': 2}}
            (base/'criteria.json').write_text(json.dumps(criteria), encoding='utf-8')
            ref = lambda name: {'path': name, 'sha256': digest(base/name)}
            manifest = {'schema_version': 1, 'calibration_campaign': ref('C.json'),
                'heldout_campaigns': [ref('D.json')], 'criteria': ref('criteria.json'), 'objective_scale_pp': 1,
                'optimizer_tolerances': {'ftol': 1e-10, 'xtol': 1e-10, 'gtol': 1e-10},
                'calibration_parameters': [{'name': 'SEI solvent diffusivity [m2.s-1]',
                    'lower': 2e-22, 'upper': 3e-22, 'initial': 2.2e-22, 'scale': 1e-22,
                    'difference_step_scaled': .1, 'unit': 'm2.s-1', 'source_ref': 'synthetic nominal test'}],
                'budget': {'max_model_calls': 30, 'total_wall_s': 120, 'per_candidate_wall_s': 60,
                           'optimizer_max_nfev': 10}}
            out = base/'out'; observed_phases = []
            def fake_run(command, **kwargs):
                candidate = Path(command[command.index('--output')+1])
                request = json.loads(Path(command[command.index('--manifest')+1]).read_text(encoding='utf-8'))
                attempt = json.loads((candidate/'attempt.json').read_text(encoding='utf-8'))
                observed_phases.append(attempt['phase'])
                self.assertEqual((out/'parameters_frozen.json').exists(), attempt['phase'] == 'postfreeze_heldout')
                value = request['protocol']['parameter_overrides']['SEI solvent diffusivity [m2.s-1]']
                self.assertGreaterEqual(value, 2e-22)
                self.assertLessEqual(value, 3e-22)
                records = [{'role': s['role'], 'propagated': True, 'charge_Ah': 0,
                            'discharge_Ah': 1 if i == 0 else 1-value*1e20}
                           for i, s in enumerate(request['protocol']['steps'])]
                result = {'status': 'replay_completed', 'records': records, 'throughput_by_role': {}}
                (candidate/'report.json').write_text(json.dumps(result), encoding='utf-8')
                return SimpleNamespace(returncode=0)
            with patch('physical_ageing_campaign.subprocess.run', side_effect=fake_run):
                self.assertEqual(fit_campaign(manifest, out, base), 0)
            result = json.loads((out/'report.json').read_text(encoding='utf-8'))
            frozen = json.loads((out/'parameters_frozen.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(frozen['parameters']['SEI solvent diffusivity [m2.s-1]']/1e-22, 2.5)
            self.assertGreater(result['heldout_results'][0]['max_abs_error_pp'], 17)
            self.assertEqual(observed_phases[-1], 'postfreeze_heldout')
            self.assertEqual(result['heldout_results'][0]['Q0_relative_error_percent'], -50)
            self.assertFalse(result['physical_validation_complete'])
            self.assertFalse(result['heldout_results'][0]['basic_curve_limits_satisfied'])
            # A lower-cost finite-difference point must not inherit convergence
            # from a different optimizer solution. This reproduces that bug.
            import numpy as np
            def different_solution(residual, initial, jac, **kwargs):
                residual(np.array([2.5]))
                matrix = jac(np.array([2.2]))
                return SimpleNamespace(success=True, status=1, message='explicit unit fixture',
                    nfev=1, jac=matrix, x=np.array([2.2]))
            out = base/'out_solution_identity'
            with patch('physical_ageing_campaign.subprocess.run', side_effect=fake_run), \
                 patch('scipy.optimize.least_squares', side_effect=different_solution):
                self.assertEqual(fit_campaign(manifest, out, base), 0)
            selected = json.loads((out/'parameters_frozen.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(selected['parameters']['SEI solvent diffusivity [m2.s-1]']/1e-22, 2.2)
            self.assertEqual(selected['selection'], 'converged_optimizer_solution_not_a_Jacobian_probe')
            # A valid earlier candidate does not turn budget exhaustion into
            # successful completion or a zero exit status.
            def exhausted(residual, initial, **kwargs):
                residual(initial)
                raise BudgetExceeded('explicit unit fixture')
            out = base/'out_budget_exhausted'
            with patch('physical_ageing_campaign.subprocess.run', side_effect=fake_run), \
                 patch('scipy.optimize.least_squares', side_effect=exhausted):
                self.assertEqual(fit_campaign(manifest, out, base), 1)
            incomplete = json.loads((out/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(incomplete['status'], 'backend_incomplete')
            self.assertFalse(incomplete['calibration_optimizer_converged'])


if __name__ == '__main__':
    unittest.main()
