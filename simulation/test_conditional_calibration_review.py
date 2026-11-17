import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from conditional_calibration_review import (CalibrationNotAccepted,bind_targets,bound_review,
    pinned,require_lifetime_ready,review_result,score_curve,validate_criteria)
from conditional_ageing_calibration import normalized_errors,run


CRITERIA=dict(criteria_id='LG_M50T_1A_2A_20261103',basic=dict(SOH_RMSE_pp_max=1.,SOH_max_abs_error_pp_max=2.))


def curve(cell='C',pred=(5.,4.9),target=(5.,4.5),coefficient=2.5e-21):
    errors=normalized_errors(list(pred),list(target))
    return dict(cell_id=cell,inputs_sha256=hashlib.sha256(cell.encode()).hexdigest(),diffusivity_m2_s=coefficient,measurements=[dict(rpt_id=f'RPT{i}',Qref_Ah=p,
        target_Qref_Ah=t,termination='event: Voltage < 2.5 [V] [experiment]') for i,(p,t) in enumerate(zip(pred,target))],
        errors_pp=errors,RMSE_pp=math.sqrt(sum(e*e for e in errors)/len(errors)),max_abs_error_pp=max(map(abs,errors)),
        empty_boundaries=[],endpoint_time_mismatches=[])


class ReviewTests(unittest.TestCase):
    def manifest(self):return dict(data_kind='conditional_observed_program_route',nominal_diffusivity_m2_s=2.5e-22,log10_multiplier_bounds=[-1,1])

    def report(self):return dict(C=curve(),D=curve('D'),D_used_for_refit=False,physical_validation_complete=False)

    def frozen(self,report):
        return dict(diffusivity_m2_s=report['C']['diffusivity_m2_s'],
                    C_inputs_sha256=report['C']['inputs_sha256'],C_result=copy.deepcopy(report['C']))

    def test_converged_boundary_bad_fit_not_accepted(self):
        report=self.report()
        r=review_result(report,self.frozen(report),self.manifest(),CRITERIA)
        self.assertEqual(r['status'],'conditional_calibration_not_accepted')
        self.assertTrue(r['parameter_bound_review']['at_upper_bound'])
        self.assertFalse(r['all_curve_metrics_pass'])
        self.assertEqual(r['D_evaluation_role'],'postfreeze_diagnostic_after_C_metric_failure')

    def test_good_conditional_curve_still_cannot_export(self):
        report=self.report()
        for c in ('C','D'):report[c]=curve(c,pred=(5.,4.5),coefficient=2.5e-22)
        r=review_result(report,self.frozen(report),self.manifest(),CRITERIA)
        self.assertTrue(r['conditional_diagnostics_accepted'])
        r['lifetime_parameter_export_allowed']=True;r['physical_validation_complete']=True
        with self.assertRaises(CalibrationNotAccepted):require_lifetime_ready(r)

    def test_reported_zero_metrics_cannot_hide_capacity_error(self):
        r=curve();r.update(RMSE_pp=0,max_abs_error_pp=0)
        with self.assertRaisesRegex(ValueError,'disagrees'):score_curve(r,CRITERIA)

    def test_objective_vector_cannot_disagree_with_raw_capacity(self):
        r=curve();r['errors_pp']=[0.,0.]
        with self.assertRaisesRegex(ValueError,'objective errors'):score_curve(r,CRITERIA)

    def test_existing_criteria_ID_cannot_be_relaxed(self):
        c=copy.deepcopy(CRITERIA);c['basic']['SOH_RMSE_pp_max']=10.
        with self.assertRaisesRegex(ValueError,'altered limits'):validate_criteria(c)

    def test_nonfinite_capacity_and_boolean_refused(self):
        for v in (float('nan'),float('inf'),True):
            r=curve();r['measurements'][1]['Qref_Ah']=v
            with self.assertRaises(ValueError):score_curve(r,CRITERIA)
            with self.assertRaises(ValueError):normalized_errors([5,v],[5,4.5])

    def test_missing_initial_rpt_refused(self):
        r=curve();r['measurements'][0]['rpt_id']='RPT1'
        with self.assertRaises(ValueError):score_curve(r,CRITERIA)

    def test_wrong_capacity_event_refused(self):
        r=curve();r['measurements'][1]['termination']='final time'
        with self.assertRaises(ValueError):score_curve(r,CRITERIA)

    def test_parameter_tampering_in_D_refused(self):
        r=self.report();r['D']['diffusivity_m2_s']=2.4e-21
        with self.assertRaisesRegex(ValueError,'parameter mismatch'):review_result(r,self.frozen(r),self.manifest(),CRITERIA)

    def test_scored_C_cannot_be_swapped_after_parameter_freeze(self):
        report=self.report();frozen=self.frozen(report)
        # Recomputed metrics are internally consistent; only source-candidate
        # binding can reveal that this is a different trajectory.
        report['C']=curve(pred=(5.,4.8))
        with self.assertRaisesRegex(ValueError,'frozen optimizer candidate'):
            review_result(report,frozen,self.manifest(),CRITERIA)

    def test_frozen_input_hash_cannot_alias_another_C_input(self):
        report=self.report();frozen=self.frozen(report)
        frozen['C_inputs_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Frozen C input identity'):
            review_result(report,frozen,self.manifest(),CRITERIA)

    def test_target_alias_or_changed_Q_refused(self):
        r=curve();r['inputs_sha256']='a';payload=dict(data_kind='conditional_observed_program_route',cell=dict(cell_id='C',measurements=[dict(rpt_id='RPT0',target_Qref_Ah=5),dict(rpt_id='RPT1',target_Qref_Ah=4.4)]))
        with self.assertRaisesRegex(ValueError,'target identity'):bind_targets(r,payload,'a')

    def test_Q0_bias_not_erased_by_normalization(self):
        r=curve(pred=(5.,4.5),target=(4.,3.6));s=score_curve(r,CRITERIA)
        self.assertTrue(s['curve_metrics_pass']);self.assertEqual(s['Q0_bias_percent'],25)

    def test_near_bound_diagnostic_uses_log_coordinate(self):
        self.assertTrue(bound_review(2.4999975286280114e-21,self.manifest())['at_upper_bound'])

    def test_missing_pinned_criteria_fails_before_PDE(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text(json.dumps(self.manifest()))
            with patch('conditional_ageing_calibration.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError,'immutable'):run(p,Path(tmp)/'results')
                process.assert_not_called()

    def test_rejected_fit_retains_report_freezes_optimizer_solution_and_exits_as_not_accepted(self):
        # Import before patching subprocess, which numpy/platform also uses on Windows.
        import scipy.optimize
        import numpy as np
        from conditional_calibration_review import sha
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);out=root/'results';events=[]
            manifest=self.manifest()
            for cell,key in [('C','calibration'),('D','heldout')]:
                p=root/f'{cell}.json';p.write_text(json.dumps(dict(data_kind='conditional_observed_program_route',
                    cell=dict(cell_id=cell,measurements=[dict(rpt_id='RPT0',target_Qref_Ah=5.),dict(rpt_id='RPT1',target_Qref_Ah=4.5)]))))
                manifest[key]=dict(path=p.name,sha256=sha(p))
            p=root/'criteria.json';p.write_text(json.dumps(CRITERIA));manifest['criteria']=dict(path=p.name,sha256=sha(p))
            manifest.update(total_wall_s=300,max_model_calls=10,per_candidate_wall_s=60,optimizer_max_nfev=10,assumptions=['software fixture only'])
            p=root/'manifest.json';p.write_text(json.dumps(manifest))
            def fake_candidate(command,**kwargs):
                src=Path(command[command.index('--inputs')+1]);dest=Path(command[command.index('--output')+1]);dest.mkdir()
                coefficient=float(command[command.index('--diffusivity')+1]);cell=src.stem
                if cell=='D':
                    frozen=json.loads((out/'parameters_frozen.json').read_text())
                    self.assertEqual(frozen['diffusivity_m2_s'],2.5e-21)
                    self.assertFalse(frozen['C_curve_metrics_pass'])
                    self.assertFalse(frozen['lifetime_parameter_export_allowed'])
                result=curve(cell,coefficient=coefficient);result['inputs_sha256']=sha(src)
                (dest/'result.json').write_text(json.dumps(result));events.append(cell)
                return subprocess.CompletedProcess(command,0,'','')
            def fake_optimizer(objective,*args,**kwargs):
                objective([0.]);objective([1.]);objective([.05]) # Last call is an FD probe, not the optimum.
                return SimpleNamespace(success=True,x=np.array([1.]),message='test optimizer convergence')
            with patch('scipy.optimize.least_squares',side_effect=fake_optimizer),patch('conditional_ageing_calibration.subprocess.run',side_effect=fake_candidate):
                with self.assertRaises(CalibrationNotAccepted):run(p,out)
            result=json.loads((out/'report.json').read_text())
            self.assertEqual(events,['C','C','C','D'])
            self.assertEqual(result['status'],'conditional_calibration_not_accepted')
            self.assertFalse(result['physical_validation_complete'])
            self.assertFalse(result['ground_truth_label_export_allowed'])
            self.assertEqual(len(json.loads((out/'evaluation_history.json').read_text())),4)


if __name__=='__main__':unittest.main()
