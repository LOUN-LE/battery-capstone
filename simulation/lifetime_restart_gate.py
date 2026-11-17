"""Strict evidence gate for the exact measured-history replay route.

Does not replace the historical synthetic assessor. Recomputes curve scores and
requires measured campaign admission; conditional scenarios cannot pass. Passing
this gate is not a claim that unobserved EFC80 has been validated.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from physical_campaign_inputs import validate_campaign
from score_empirical_validation import score_suite


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite numeric')
    if positive and value <= 0:
        raise ValueError(name+' must be positive')
    return float(value)


def checked(ref, root):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Explicit immutable path and sha256 required')
    path=(root/ref['path']).resolve();path.relative_to(root)
    if sha(path)!=ref['sha256']:
        raise ValueError('Evidence changed: '+str(path))
    return path


def native_efc_coordinates(campaign, base):
    # validate_campaign has already verified the original archive member identity.
    from fit_empirical_baselines import load_summary
    path=checked(campaign['source_identity']['summary'],Path(base).resolve())
    return {f"RPT{int(p['rpt'])}":p['absolute_ageing_EFC'] for p in load_summary(path,campaign['cell_id'])}


def assess(manifest, root):
    root=Path(root).resolve();reasons=[];details={};loaded={}
    roles=('proposal','criteria','candidate','code_sources','software','validation','numerics','checkpoint')
    if manifest.get('schema_version') != 1 or set(manifest)-set(roles)-{'schema_version','admission_route'}:
        reasons.append('manifest: unknown schema or fields')
    if manifest.get('admission_route')!='exact_measured_history_replay':
        reasons.append('manifest: this gate only assesses the exact measured-history replay route')
    for role in roles:
        try:
            path=checked(manifest.get(role),root)
            loaded[role]=path if role=='proposal' else json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError,TypeError,KeyError) as exc:
            reasons.append(role+': '+str(exc))
    candidate_hash=manifest.get('candidate',{}).get('sha256')
    sources=loaded.get('code_sources')
    def bound(record):
        if record.get('candidate_sha256')!=candidate_hash or record.get('code_sources')!=sources:
            raise ValueError('Run is not bound to this candidate and exact tested code')
    try:
        criteria=loaded['criteria']
        if criteria['basic']!={'SOH_RMSE_pp_max':1.,'SOH_max_abs_error_pp_max':2.,'observed_EFC80_relative_error_percent_max':10.}:
            raise ValueError('User basic1A limits changed')
        if criteria['source']['sha256']!=sha(loaded['proposal']):
            raise ValueError('Criteria do not match user proposal')
        if criteria['eol']['threshold_SOH_percent']!=80 or criteria['challenge_is_required'] is not False:
            raise ValueError('Fixed threshold or required scope changed')
        if criteria['eol']['max_observed_bracket_EFC'] is not None:
            raise ValueError('EOL resolution requires a separately reviewed gate-schema update; no arbitrary interval limit')
        details['criteria_fixed']=True
    except (KeyError,TypeError,ValueError) as exc:reasons.append('criteria: '+str(exc))
    try:
        if not isinstance(sources,dict) or not sources:raise ValueError('Nonempty current code source manifest required')
        for path,digest in sources.items():checked({'path':path,'sha256':digest},root)
        for module in ('lifetime_restart_gate.py','physical_campaign_inputs.py','score_empirical_validation.py'):
            matches=[v for p,v in sources.items() if Path(p).name==module]
            if matches!=[sha(Path(__file__).with_name(module))]:raise ValueError('Gate/dependency missing from current source coverage: '+module)
        s=loaded['software'];suites=s['suites']
        if not suites or any(x.get('exit_code')!=0 or type(x.get('test_count')) is not int or x['test_count']<=0 for x in suites):
            raise ValueError('Passing nonempty test suites required')
        if s.get('all_pass') is not True or s.get('source_hashes_stable') is not True or s.get('frozen_six_unchanged') is not True:
            raise ValueError('Software execution or frozen source integrity failed')
        if s['total_tests']!=sum(x['test_count'] for x in suites):raise ValueError('Test count mismatch')
        if s.get('tested_code_sources')!=sources:raise ValueError('Test results do not identify these exact code bytes')
        details['current_software_pass']=True
    except (KeyError,TypeError,ValueError,OSError) as exc:reasons.append('software: '+str(exc))
    cases=[];maximum_EFC=0.
    try:
        items=loaded['validation']
        if not isinstance(items,list) or not items:raise ValueError('Actual measured held-out runs are absent')
        for item in items:
            campaign_path=checked(item['campaign'],root)
            campaign=json.loads(campaign_path.read_text(encoding='utf-8-sig'))
            if campaign.get('data_kind')!='measured':raise ValueError('Synthetic or conditional history cannot become measured admission')
            admitted=validate_campaign(campaign,campaign_path.parent)
            run=json.loads(checked(item['run'],root).read_text(encoding='utf-8-sig'));bound(run)
            if run.get('data_kind')!='measured' or run.get('status')!='measured_campaign_complete':
                raise ValueError('Completed actual measured campaign run required')
            if run.get('campaign_sha256')!=item['campaign']['sha256']:
                raise ValueError('Prediction uses a different admitted history')
            case=json.loads(checked(item['case'],root).read_text(encoding='utf-8-sig'))
            if case['model']['kind']!='physical' or case['model']['artifact_sha256']!=candidate_hash:
                raise ValueError('Baseline or different fitted model cannot pass physical restart gate')
            if case['cell_id']!=admitted['cell_id'] or case['condition_id']!=admitted['condition_id']:
                raise ValueError('Cell or condition does not match admitted source')
            if case['axis_column']!='EFC' or case['measured_metadata'].get('axis_unit')!='EFC':
                raise ValueError('Reviewed EFC coordinates required')
            measured=run['measured'];predicted=run['predicted']
            native=admitted['_targets_rows']
            native_efc=native_efc_coordinates(campaign,campaign_path.parent)
            if run.get('evaluation_axis_policy')!='native_ageing_throughput_at_RPT':
                raise ValueError('Explicit native ageing-throughput RPT comparison policy required')
            if len(measured)!=len(native) or [x['rpt_id'] for x in measured]!=[x['rpt_id'] for x in native]:
                raise ValueError('Entire chronological native RPT set must be evaluated')
            for x,n in zip(measured,native):
                if x['Qref_Ah']!=n['Qref_Ah']:raise ValueError('Measured capacity differs from immutable native targets')
                if not math.isclose(finite(x['EFC'],'measured EFC'),native_efc[x['rpt_id']],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Measured EFC differs from native ageing-throughput coordinate')
            if len(predicted)!=len(native) or [x['rpt_id'] for x in predicted]!=[x['rpt_id'] for x in native]:
                raise ValueError('Prediction omitted a measured RPT')
            for x in predicted:
                if not math.isclose(finite(x['EFC'],'prediction evaluation EFC'),native_efc[x['rpt_id']],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Prediction evaluated at a warped RPT throughput coordinate')
            for series,key in ((measured,'measured'),(predicted,'predicted')):
                q0=finite(series[0]['Qref_Ah'],'fixed Q0',True)
                recomputed=[dict(EFC=finite(x['EFC'],'EFC'),SOH_percent=100*finite(x['Qref_Ah'],'Qref',True)/q0) for x in series]
                if case[key]!=recomputed:raise ValueError('SOH curve is not derived from actual run capacities and fixed Q0')
            maximum_EFC=max(maximum_EFC,max(x['EFC'] for x in measured))
            cases.append(case)
        score=score_suite(cases,loaded['criteria'],root);details['curve_scores']=score
        if score['staged_2A_curve_scope_complete'] is not True or score['same_frozen_model_verified'] is not True:
            raise ValueError('Each cell and both independent stages must pass the fixed criteria')
        if any(c['eol']['status'] not in ('passed','not_evaluated') or
               (c['eol'].get('measured',{}).get('status')=='observed' and c['eol']['status']!='passed') for c in score['cells']):
            raise ValueError('Observed EFC80 comparison failed or measurement resolution remains unjustified')
        details['SOH80_validation_complete']=any(c['eol']['status']=='passed' for c in score['cells'])
    except (KeyError,TypeError,ValueError,OSError) as exc:reasons.append('measured_validation: '+str(exc))
    try:
        n=loaded['numerics'];bound(n)
        if n.get('scope')!='full_validation_exposure':raise ValueError('A short numerical screen cannot certify long exposure')
        declaration=json.loads(checked(n['declaration'],root).read_text(encoding='utf-8-sig'));bound(declaration)
        limit=finite(declaration['SOH_difference_limit_pp'],'declared numerical limit',True)
        if finite(n['completed_EFC'],'numerical exposure')<maximum_EFC or maximum_EFC<=0:
            raise ValueError('Numerical check does not cover actual validated exposure')
        pairs=n['pairs']
        if not pairs:raise ValueError('Actual paired mesh/tolerance curves required')
        for pair in pairs:
            if pair['coarse_config']==pair['fine_config']:raise ValueError('Distinct resolution settings required')
            a=np.asarray(pair['coarse_SOH_percent'],float);b=np.asarray(pair['fine_SOH_percent'],float)
            efc=np.asarray(pair['EFC'],float)
            if a.ndim!=1 or a.shape!=b.shape or a.shape!=efc.shape or len(a)<2 or not np.isfinite([a,b,efc]).all() or not np.all(np.diff(efc)>0):
                raise ValueError('Finite paired chronological numerical observations required')
            if efc[-1]<maximum_EFC or np.max(np.abs(a-b))>limit:raise ValueError('Numerical range or declared limit failed')
        details['long_exposure_numerics_pass']=True
    except (KeyError,TypeError,ValueError,OSError) as exc:reasons.append('numerics: '+str(exc))
    try:
        c=loaded['checkpoint'];bound(c)
        declaration=json.loads(checked(c['declaration'],root).read_text(encoding='utf-8-sig'));bound(declaration)
        tolerance=finite(declaration['state_comparison_atol'],'declared state tolerance',True)
        checked(c['checkpoint_payload'],root)
        if c.get('scope')!='current_candidate_actual_stop_resume' or c.get('fresh_initialization') is not True:
            raise ValueError('Current-candidate actual stop/resume evidence required; old cell4 cannot substitute')
        for name in ('saved_vs_loaded','uninterrupted_vs_resumed'):
            a=np.asarray(c[name]['left'],float);b=np.asarray(c[name]['right'],float)
            if not a.size or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.max(abs(a-b))>tolerance:
                raise ValueError('Physical-state roundtrip/resumed trajectory mismatch: '+name)
        if finite(c['Q0_before_Ah'],'Q0',True)!=finite(c['Q0_after_Ah'],'Q0',True):raise ValueError('Resume changed fixed Q0')
        if c['virtual_RPT_main_state_before']!=c['virtual_RPT_main_state_after'] or not c['virtual_RPT_main_state_before']:
            raise ValueError('Virtual measurement mutated or omitted main state evidence')
        details['checkpoint_pass']=True
    except (KeyError,TypeError,ValueError,OSError) as exc:reasons.append('checkpoint: '+str(exc))
    return dict(status='lifetime_restart_admitted' if not reasons else 'lifetime_restart_deferred',
        lifetime_restart_admitted=not reasons,reasons=reasons,evidence=details,
        SOH80_validation_complete=details.get('SOH80_validation_complete',False),
        scope='Exact measured-history replay route only; no unobserved lifetime certification',
        user_acceptance='Per-cell SOH RMSE<=1pp and max<=2pp; both independent stages; observed EFC80 assessed separately',
        engineering_admission_policy='Complete admitted replay, current-source numerical and checkpoint evidence; these are not additional PDF error thresholds',
        alternative_condition_level_validation='Not assessed here. Unknown historic exposure is not a mathematical proof that all lifetime validation routes are impossible.',
        AWS_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('Use a new result path')
    result=assess(json.loads(a.manifest.read_text(encoding='utf-8-sig')),a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['lifetime_restart_admitted'] else 2)
