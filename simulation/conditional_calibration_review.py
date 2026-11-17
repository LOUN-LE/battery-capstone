"""Independent conditional-fit scoring. Optimizer convergence is not acceptance.

This reviewer never promotes an assumed history to a validated lifetime model.
It recomputes capacity-normalized metrics and binds them to immutable inputs.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


class CalibrationNotAccepted(RuntimeError):
    """Calculation completed, but its calibration diagnostics were not accepted."""


def positive(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Finite positive capacity/criterion/parameter required')
    return float(value)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref,base):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:
        raise ValueError('Explicit immutable path/sha256 reference required')
    path=(Path(base)/ref['path']).resolve()
    if sha(path)!=ref['sha256']:raise ValueError('Pinned input changed: '+str(path))
    return path


def validate_criteria(criteria):
    for k in ('SOH_RMSE_pp_max','SOH_max_abs_error_pp_max'):positive(criteria['basic'][k])
    if criteria.get('criteria_id')!='LG_M50T_1A_2A_20261103':
        raise ValueError('Reviewed LG M50T acceptance criteria required')
    if criteria['basic']['SOH_RMSE_pp_max']!=1. or criteria['basic']['SOH_max_abs_error_pp_max']!=2.:
        raise ValueError('Criteria ID cannot be reused with relaxed or altered limits')
    return criteria


def bind_targets(result,payload,expected_hash):
    cell=payload['cell']
    if (payload.get('data_kind')!='conditional_observed_program_route'
        or result.get('inputs_sha256')!=expected_hash or result['cell_id']!=cell['cell_id']):
        raise ValueError('Conditional candidate input identity mismatch')
    actual=result['measurements'];expected=cell['measurements']
    if len(actual)!=len(expected):raise ValueError('Incomplete RPT coverage')
    for a,e in zip(actual,expected):
        if a['rpt_id']!=e['rpt_id'] or not math.isclose(positive(a['target_Qref_Ah']),positive(e['target_Qref_Ah']),rel_tol=1e-12,abs_tol=0):
            raise ValueError('Native target identity/value mismatch')


def score_curve(result,criteria):
    rows=result['measurements'];ids=[r['rpt_id'] for r in rows]
    if len(rows)<2 or ids!=[f'RPT{i}' for i in range(len(rows))]:
        raise ValueError('Chronological complete RPT0-based capacity series required')
    pred=[positive(r['Qref_Ah']) for r in rows];target=[positive(r['target_Qref_Ah']) for r in rows]
    if any(r.get('termination')!='event: Voltage < 2.5 [V] [experiment]' for r in rows):
        raise ValueError('Independent 2.5V capacity endpoint required at every RPT')
    values=[dict(rpt_id=i,model_Qref_Ah=p,target_Qref_Ah=t,model_SOH_percent=100*p/pred[0],
                 target_SOH_percent=100*t/target[0],error_pp=100*(p/pred[0]-t/target[0])) for i,p,t in zip(ids,pred,target)]
    errors=[v['error_pp'] for v in values]
    if 'errors_pp' in result:
        stored=result['errors_pp']
        if len(stored)!=len(errors) or any(isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a)
            or not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9) for a,b in zip(stored,errors)):
            raise ValueError('Stored objective errors disagree with raw capacity series')
    rmse=math.sqrt(sum(e*e for e in errors)/len(errors));maximum=max(map(abs,errors))
    for key,value in [('RMSE_pp',rmse),('max_abs_error_pp',maximum)]:
        stored=result.get(key)
        if isinstance(stored,bool) or not isinstance(stored,(int,float)) or not math.isfinite(stored) or not math.isclose(stored,value,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Stored metric disagrees with independently recomputed capacity series: '+key)
    basic=criteria['basic'];worst=max(values,key=lambda r:abs(r['error_pp']))
    # Empty boundaries are separate from the >1s endpoint-difference list.
    omitted=[e for e in result.get('empty_boundaries',[]) if e.get('observed_empty') is False]
    omitted_s=sum((datetime.fromisoformat(e['recorded_end'])-datetime.fromisoformat(e['recorded_start'])).total_seconds() for e in omitted)
    return dict(RMSE_pp=rmse,max_abs_error_pp=maximum,worst_rpt_id=worst['rpt_id'],
        curve_metrics_pass=rmse<=basic['SOH_RMSE_pp_max'] and maximum<=basic['SOH_max_abs_error_pp_max'],
        Q0_bias_percent=100*(pred[0]/target[0]-1),last_model_SOH_percent=values[-1]['model_SOH_percent'],
        last_target_SOH_percent=values[-1]['target_SOH_percent'],last_error_pp=errors[-1],
        model_empty_recorded_nonempty_count=len(omitted),recorded_duration_omitted_by_empty_model_steps_s=omitted_s,
        endpoint_time_difference_over1s_count=len(result.get('endpoint_time_mismatches',[])),
        rows=values,normalization='each series uses its own fixed RPT0; zero RPT0 SOH error does not imply Q0 agreement')


def bound_review(value,manifest):
    nominal=positive(manifest['nominal_diffusivity_m2_s']);value=positive(value)
    low,high=manifest['log10_multiplier_bounds']
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (low,high)) or low>=high:
        raise ValueError('Finite ordered log10 search bounds required')
    z=math.log10(value/nominal)
    if z<low-1e-10 or z>high+1e-10:raise ValueError('Selected parameter outside declared bounds')
    # Engineering diagnostic tolerance in log10(multiplier), NOT an accuracy criterion.
    tolerance=1e-3
    return dict(multiplier=value/nominal,log10_multiplier=z,log10_bounds=[low,high],
                bound_diagnostic_tolerance_log10=tolerance,
                at_lower_bound=z-low<=tolerance,at_upper_bound=high-z<=tolerance,
                requires_bound_review=min(z-low,high-z)<=tolerance)


def review_result(report,frozen,manifest,criteria):
    validate_criteria(criteria)
    if manifest.get('data_kind')!='conditional_observed_program_route':raise ValueError('Conditional scope required')
    if report.get('D_used_for_refit') is not False:raise ValueError('Heldout refit/leakage status unresolved')
    if report.get('physical_validation_complete') is not False:raise ValueError('Conditional exposure cannot certify physical validity')
    coefficient=positive(frozen['diffusivity_m2_s'])
    for cell in ('C','D'):
        if report[cell].get('cell_id')!=cell or not math.isclose(positive(report[cell]['diffusivity_m2_s']),coefficient,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Post-freeze cell or parameter mismatch')
    # Equal coefficients alone do not identify the selected model trajectory.
    # Bind the scored C curve to the complete candidate frozen by the optimizer,
    # not another probe/replay that happened to use the same coefficient.
    c_hash=report['C'].get('inputs_sha256')
    if (not isinstance(c_hash,str) or len(c_hash)!=64
        or any(ch not in '0123456789abcdef' for ch in c_hash)
        or frozen.get('C_inputs_sha256')!=c_hash):
        raise ValueError('Frozen C input identity mismatch')
    if frozen.get('C_result')!=report['C']:
        raise ValueError('Scored C result differs from the frozen optimizer candidate')
    scores={cell:score_curve(report[cell],criteria) for cell in ('C','D')}
    bounds=bound_review(coefficient,manifest)
    passed=all(c['curve_metrics_pass'] for c in scores.values())
    diagnostic_ok=passed and not bounds['requires_bound_review']
    return dict(schema_version=1,status='conditional_calibration_not_accepted' if not diagnostic_ok else 'conditional_curve_metrics_passed_physical_unverified',
        optimizer_convergence_is_not_acceptance=True,criteria_id=criteria['criteria_id'],basic_criteria=criteria['basic'],
        cells=scores,parameter_bound_review=bounds,all_curve_metrics_pass=passed,
        conditional_diagnostics_accepted=diagnostic_ok,
        D_evaluation_role='postfreeze_diagnostic_after_C_metric_failure' if not scores['C']['curve_metrics_pass'] else 'postfreeze_heldout_conditional_evaluation',
        physical_validation_complete=False,lifetime_parameter_export_allowed=False,ground_truth_label_export_allowed=False,
        blockers=([f'{cell}_curve_metrics_failed' for cell,s in scores.items() if not s['curve_metrics_pass']]
                  +(['parameter_at_search_bound'] if bounds['requires_bound_review'] else [])
                  +['unknown_exposure_and_branch_assumptions','long_history_numerical_convergence_unverified']),
        EFC80='not_evaluated_no_accepted_prediction',data_kind='conditional_observed_program_route')


def require_lifetime_ready(assessment):
    # No assumed-history result is eligible, even if someone changes a summary flag.
    if (assessment.get('data_kind')=='conditional_observed_program_route'
        or assessment.get('physical_validation_complete') is not True
        or assessment.get('all_curve_metrics_pass') is not True
        or assessment.get('lifetime_parameter_export_allowed') is not True):
        raise CalibrationNotAccepted('Lifetime/ground-truth export refused: physical admission and accepted independent validation required')


def main():
    a=argparse.ArgumentParser();a.add_argument('--review',type=Path,required=True)
    a.add_argument('--manifest',type=Path,required=True);a.add_argument('--criteria',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--require-lifetime-ready',action='store_true')
    args=a.parse_args();payload=json.loads(args.review.read_text(encoding='utf-8-sig'))
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'));criteria=json.loads(args.criteria.read_text(encoding='utf-8-sig'))
    for cell,key in [('C','calibration'),('D','heldout')]:
        path=pinned(manifest[key],args.manifest.parent)
        bind_targets(payload['report'][cell],json.loads(path.read_text(encoding='utf-8-sig')),manifest[key]['sha256'])
    frozen_bytes=json.dumps(payload['frozen'],indent=2,allow_nan=False).encode('utf-8')
    if hashlib.sha256(frozen_bytes).hexdigest()!=payload['report']['parameters_frozen_sha256']:
        raise ValueError('Frozen parameter content hash mismatch')
    assessment=review_result(payload['report'],payload['frozen'],manifest,criteria)
    assessment['source_sha256']={str(p.resolve()):sha(p) for p in (args.review,args.manifest,args.criteria,Path(__file__))}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'assessment.json').write_text(json.dumps(assessment,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.output/'RPT_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['cell']+list(assessment['cells']['C']['rows'][0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for cell,s in assessment['cells'].items():
            for row in s['rows']:w.writerow(dict(cell=cell,**row))
    print(json.dumps({k:v for k,v in assessment.items() if k not in ('cells','source_sha256')},indent=2))
    if args.require_lifetime_ready:require_lifetime_ready(assessment)


if __name__=='__main__':
    try:main()
    except CalibrationNotAccepted as e:
        print(str(e));raise SystemExit(2)
