"""C-only effective initial SOC from a native pre-charge rest endpoint.

Material parameters stay frozen. An endpoint-matched homogeneous state does
not reconstruct formation, concentration gradients or the measured relaxation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def compare_phase(model_t,model_v,observed_t,observed_v):
    """Phase-relative absolute seconds only; no time/voltage-axis stretching."""
    import numpy as np
    mt=np.asarray(model_t)-model_t[0];ot=np.asarray(observed_t)-observed_t[0]
    if not np.all(np.diff(mt)>0) or not np.all(np.diff(ot)>0):raise ValueError('Monotonic phase clocks required')
    mask=(ot>=0)&(ot<=mt[-1]);e=np.interp(ot[mask],mt,model_v)-np.asarray(observed_v)[mask]
    if not len(e) or not np.all(np.isfinite(e)):raise ValueError('No finite covered phase samples')
    return dict(voltage_RMSE_mV=float(np.sqrt(np.mean(e*e))*1000),voltage_max_abs_error_mV=float(max(abs(e))*1000),
        model_duration_s=float(mt[-1]),observed_sample_span_s=float(ot[-1]),duration_difference_s=float(mt[-1]-ot[-1]),
        observed_sample_coverage_fraction=float(np.mean(mask)),time_axis='phase_relative_seconds_no_stretch')


def load_native(root,cell):
    import numpy as np
    sys.path.insert(0,str(root/'work/preprocess'))
    import common
    name=f'NDK - LG M50 deg - exp 2,2 - rig 3 - 25degC - cell {cell} - BoL - RPT0_long_01_MB_CA{4 if cell=="C" else 5}.mpt'
    path=root/'work/external_m50_benchmark/raw_protocol'/name
    before=sha(path);cols,meta,_=common.read_mpt(path)
    ranges,_=common.split_runs(cols)
    first=ranges[:6]
    if [int(cols['ns'][a]) for a,b in first]!=list(range(6)):raise ValueError('Unexpected native RPT sequence')
    phases=[]
    for group in ([0],[1],[2],[3,4],[5]):
        indices=np.concatenate([np.arange(*first[i]) for i in group]);t=cols['t'][indices]
        if not np.all(np.diff(t)>0):raise ValueError('Native initial RPT has nonincreasing time')
        phases.append(dict(t=np.asarray(t,dtype=float),v=np.asarray(cols['V'][indices],dtype=float),
                           i=-np.asarray(cols['I'][indices],dtype=float)/1000,T=np.asarray(cols['T'][indices],dtype=float)))
    if sha(path)!=before:raise ValueError('Native source changed while reading')
    return phases,dict(path=str(path),sha256=before,parser=meta['parser'],phase_Ns=[[0],[1],[2],[3,4],[5]])


def run(root,parameters_path,output):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import numpy as np
    import pybamm
    from scipy.optimize import brentq
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    parameters_path=Path(parameters_path).resolve();material_hash=sha(parameters_path)
    material=json.loads(parameters_path.read_text(encoding='utf-8-sig'))
    if not material.get('C_only_fit') or material.get('transport_and_degradation_parameters_changed') is not False:
        raise ValueError('Frozen C-only effective BOL material required')
    config=json.loads((root/'work/validation_proposal_20261103/physical_BOL_fit_v3/declared_experiment.json').read_text())
    options=config['options'];mesh=config['fine']['mesh_points']
    c,c_source=load_native(root,'C')
    if max(abs(c[0]['i']))>1e-6:raise ValueError('Native pre-charge phase is not rest')
    from joint_spme_bol_fit import select_rest_anchor
    target,anchor_policy=select_rest_anchor(c[0]);anchor_policy.update(policy_id='declared_rest_interior_last3_v1',declared_rest_duration_s=120.,excluded_boundary_rows=len(anchor_policy['boundary_candidates']),raw_preserved=True);samples=[]
    p=pybamm.ParameterValues('OKane2022');p.update(material['parameter_updates'])
    p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
    def simulate(soc,full):
        definitions=[pybamm.step.rest(120,skip_ok=False)]
        if full:definitions.extend([pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),
            pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),pybamm.step.rest(7230,skip_ok=False),
            pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False)])
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=p.copy(),
            experiment=pybamm.Experiment(definitions,period=1,temperature=298.15),var_pts=mesh,
            solver=pybamm.IDAKLUSolver(rtol=1e-8,atol=1e-10))
        sol=sim.solve(initial_soc=float(soc),calc_esoh=False)
        pieces=[x.steps[-1] for x in sol.cycles]
        actual=[str(x.termination) for x in pieces]
        expected=config['expected_terminations'] if full else ['final time']
        if actual!=expected:raise ValueError('Initial RPT endpoint mismatch '+str(actual))
        raw=[dict(t=np.asarray(x['Time [s]'].entries,dtype=float),v=np.asarray(x['Voltage [V]'].entries,dtype=float),
                  i=np.asarray(x['Current [A]'].entries,dtype=float),q=np.asarray(x['Discharge capacity [A.h]'].entries,dtype=float)) for x in pieces]
        return raw
    def residual(soc):
        raw=simulate(soc,False)[0];value=float(raw['v'][-1]);samples.append(dict(soc=float(soc),rest_terminal_model_V=value,residual_V=value-target))
        np.savez_compressed(out/f'rest_fit_{len(samples):02d}.npz',**raw)
        write(out/'rest_fit_journal.json',samples)
        return value-target
    fitted=float(brentq(residual,.05,.6,xtol=1e-6,maxiter=12))
    frozen=dict(effective_initial_soc=fitted,source_C=c_source,material_parameters_sha256=material_hash,
        material_parameters_file=str(parameters_path),fit_observation='last3 zero-current C rest samples inside declared120.0s control window',anchor_policy=anchor_policy,anchor_policy_id='declared_rest_interior_last3_v1',declared_rest_duration_s=120.,excluded_boundary_rows=len(anchor_policy['boundary_candidates']),raw_preserved=True,source_C_sha256=c_source['sha256'],
        target_voltage_V=target,target_voltage_span_V=float(np.ptp(anchor_policy['selected_voltage_V'])),
        assumptions=['homogeneous SPMe initial concentration','25C isothermal; native surface temperature differs',
                     'formation/previous gradients unknown; terminal voltage matching is not initial-state identification'],
        D_used_for_fit=False,physical_validation_complete=False,code_sha256=sha(__file__))
    write(out/'parameters_effective_initialization.json',frozen);freeze_hash=sha(out/'parameters_effective_initialization.json')
    report=dict(status='initial_state_comparison_completed',initialization=frozen,scenarios={},
                physical_validation_complete=False,SOH_curve_validation_complete=False)
    # D is opened only after the C-only initialization has been frozen.
    d,d_source=load_native(root,'D')
    for label,soc in [('previous_assumption',.2),('C_rest_endpoint_fit',fitted)]:
        raw=simulate(soc,True)
        np.savez_compressed(out/f'{label}_full_RPT.npz',**{f'phase{i}_{k}':v for i,row in enumerate(raw) for k,v in row.items()})
        result={}
        for cell,native,source in [('C',c,c_source),('D',d,d_source)]:
            scores=[compare_phase(r['t'],r['v'],m['t'],m['v']) for r,m in zip(raw,native)]
            for score,m in zip(scores,native):score['recorded_surface_temperature_C']={name:float(f(m['T'])) for name,f in [('min',np.min),('median',np.median),('max',np.max)]}
            result[cell]=dict(source=source,phase_order=['initial_rest','CC_charge','CV','pre_discharge_rest','C10_discharge'],phases=scores,
                             model_C10_capacity_Ah=float(raw[-1]['q'][-1]-raw[-1]['q'][0]),D_initial_state_assumed_equal_to_C=cell=='D')
        report['scenarios'][label]=dict(initial_soc=soc,comparison=result)
    if sha(parameters_path)!=material_hash or sha(out/'parameters_effective_initialization.json')!=freeze_hash:
        raise ValueError('Frozen material/initialization changed during evaluation')
    write(out/'report.json',report)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--parameters',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();r=run(args.root,args.parameters,args.output);print(json.dumps(dict(status=r['status'],effective_initial_soc=r['initialization']['effective_initial_soc']),indent=2))
