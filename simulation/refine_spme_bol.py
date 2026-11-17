"""C-only BOL objective-window refinement; no new mechanisms or bound expansion.

Reuses the audited BOL worker and its inventory-conserving helper unchanged.
The sole fitting change is a fixed measured absolute-Ah voltage grid spanning
2--98% instead of 2--80%. Initial SOC remains the declared assumption 0.2.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def build_config(previous,previous_report):
    p=copy.deepcopy(previous)
    if (p['family']!='SPMe' or p['parameter_set']!='OKane2022' or p['initial_soc_assumption']!=.2
        or p['initial_rest_s']!=120 or p['pre_discharge_rest_s']!=7230
        or p['bounds']!=[[.85,-.4],[1.,.4]]):
        raise ValueError('Audited fixed BOL protocol, initial SOC and unchanged bounds required')
    chosen=previous_report['chosen_C_candidate']
    p.update(scope='C-only expanded 2--98 percent absolute-Ah BOL refinement; SOC fixed0.2',
        voltage_capacity_fraction_grid=[.02,.98,97],
        starts=[[chosen['s_Q'],chosen['delta_Q_Li_Ah']],[.925,.15]],
        max_nfev_by_start=[10,10],max_model_calls=59,max_wall_s=540,profile_fixed_scales=[])
    p['limitations']=p['limitations']+[
        'Voltage fitting grid expanded to fixed measured2--98%; last2% remains a separate audit',
        'InitialSOC0.2 is deliberately fixed to isolate objective-window change',
        'Old and new full covered-curve scores are reported on absolute Ah; no curve-axis stretching',
        'A bound-constrained optimum cannot establish unique electrode-property identification']
    return p


def score_absolute_curve(model_q,model_v,measured_q,measured_v):
    import numpy as np
    for q,v in [(model_q,model_v),(measured_q,measured_v)]:
        if len(q)<2 or len(q)!=len(v) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)) or not np.all(np.diff(q)>0):
            raise ValueError('Finite strictly increasing unwarped Ah curve required')
    mask=(measured_q>=model_q[0])&(measured_q<=model_q[-1])
    if not np.any(mask):raise ValueError('No absolute-Ah overlap')
    errors=np.interp(measured_q[mask],model_q,model_v)-measured_v[mask]
    f=measured_q[mask]/measured_q[-1];bins=[]
    for lo,hi in [(0,.02),(.02,.8),(.8,.95),(.95,1.0000001)]:
        select=(f>=lo)&(f<hi)
        bins.append(dict(measured_capacity_fraction=[lo,min(hi,1.)],n=int(select.sum()),
                         RMSE_V=float(np.sqrt(np.mean(errors[select]**2))) if np.any(select) else None))
    return dict(capacity_model_Ah=float(model_q[-1]),capacity_measured_Ah=float(measured_q[-1]),
        capacity_error_percent=float(100*(model_q[-1]/measured_q[-1]-1)),
        covered_fraction=float(np.mean(mask)),full_covered_voltage_RMSE_V=float(np.sqrt(np.mean(errors**2))),
        max_abs_voltage_error_V=float(np.max(np.abs(errors))),segments=bins)


def full_rpt(config,parameter_updates,out):
    import numpy as np
    import pybamm
    out.mkdir(exist_ok=False)
    p=pybamm.ParameterValues('OKane2022');p.update(parameter_updates)
    p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
    steps=[pybamm.step.rest(120),pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),
        pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),pybamm.step.rest(7230),
        pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False)]
    sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,
        experiment=pybamm.Experiment(steps,period=1,temperature=298.15),var_pts=config['fine']['mesh_points'],
        solver=pybamm.IDAKLUSolver(rtol=config['fine']['rtol'],atol=config['fine']['atol']))
    solution=sim.solve(initial_soc=.2,calc_esoh=False)
    pieces=[c.steps[-1] for c in solution.cycles]
    if [str(p.termination) for p in pieces]!=config['expected_terminations']:
        raise ValueError('Full RPT event mismatch')
    stages=[]
    for i,piece in enumerate(pieces):
        raw={n:np.asarray(piece[n].entries,dtype=float) for n in ('Time [s]','Voltage [V]','Current [A]','Discharge capacity [A.h]')}
        np.savez_compressed(out/f'step_{i+1:04d}.npz',**raw)
        stages.append(dict(stage_index=i,model_duration_s=float(raw['Time [s]'][-1]-raw['Time [s]'][0]),termination=str(piece.termination)))
    write(out/'report.json',dict(status='full_RPT_completed',stages=stages,
        capacity_Ah=float(pieces[-1]['Discharge capacity [A.h]'].entries[-1]-pieces[-1]['Discharge capacity [A.h]'].entries[0]),
        initial_SOC=.2,physical_validation_complete=False))
    return stages


def run(root,output):
    import numpy as np
    import calibrate_m50t_bol as old
    root=Path(root).resolve();output=Path(output).resolve();previous=root/'work/validation_proposal_20261103/physical_BOL_fit_v3'
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    previous_config=json.loads((previous/'declared_experiment.json').read_text(encoding='utf-8-sig'))
    previous_report=json.loads((previous/'report.json').read_text(encoding='utf-8-sig'))
    config=build_config(previous_config,previous_report)
    if sha(Path(old.__file__))!=config['code_sha256']:raise ValueError('Audited BOL worker changed; do not silently reuse it')
    wrapper_hash=sha(__file__)
    write(output/'declared_experiment.json',config)
    write(output/'refinement_provenance.json',dict(wrapper_sha256=wrapper_hash,worker_sha256=sha(old.__file__),
        previous_config_sha256=sha(previous/'declared_experiment.json'),previous_report_sha256=sha(previous/'report.json'),
        sole_fit_change='voltage fit window2--80% to2--98% at fixed measured absolute Ah',
        physics_options_changed=False,parameter_bounds_expanded=False,initial_SOC_changed=False))
    report=old.worker(config,output)
    scores={};old_raw=np.load(previous/'candidate_034.npz');new_raw=np.load(output/report['verification']['fine_candidate']['raw_file'])
    for cell,key in [('C','source_C'),('D','source_D_after_freeze')]:
        q,v=old.read_curve(Path(config[key]));scores[cell]={
            'previous_v3':score_absolute_curve(old_raw['Q_Ah'],old_raw['V'],q,v),
            'refined':score_absolute_curve(new_raw['Q_Ah'],new_raw['V'],q,v)}
    frozen=json.loads((output/'parameters_frozen.json').read_text(encoding='utf-8-sig'))
    phases=full_rpt(config,frozen['parameter_updates'],output/'full_RPT') if time.monotonic()-began<570 else None
    result=dict(status='expanded_window_refinement_completed',scores=scores,
        selected_parameters={k:frozen[k] for k in ('s_Q','delta_Q_Li_Ah')},
        parameter_at_inventory_upper_bound=.4-frozen['delta_Q_Li_Ah']<.001,
        C_full_curve_RMSE_improved=scores['C']['refined']['full_covered_voltage_RMSE_V']<scores['C']['previous_v3']['full_covered_voltage_RMSE_V'],
        C_capacity_absolute_error_improved=abs(scores['C']['refined']['capacity_error_percent'])<abs(scores['C']['previous_v3']['capacity_error_percent']),
        optimizer_converged_for_all_starts=report['optimizer_converged_for_all_starts'],
        source_unchanged=wrapper_hash==sha(__file__) and sha(old.__file__)==config['code_sha256'],
        model_calls=report['model_calls']+(1 if phases is not None else 0),wall_s=time.monotonic()-began,
        fixed_initial_SOC=.2,full_RPT_phase_artifacts_saved=phases is not None,
        physical_validation_complete=False,SOH_validation_complete=False,AWS_started=False)
    write(output/'refinement_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    p=a.parse_args();run(p.root,p.output)
