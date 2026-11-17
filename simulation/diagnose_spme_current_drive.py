"""C precharge falsification test using recorded current, with unchanged V limits.

No nominal-capacity current substitution, time stretching, or end-capacity fit.
Measured surface temperature is an imposed diagnostic input, not predicted heat.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from initialize_spme_from_rest import load_native,sha,write
from initialization_contract import validate_initialization_anchor


def run(root,initialization,output):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import numpy as np
    import pybamm
    root=Path(root).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    init=json.loads(Path(initialization).read_text(encoding='utf-8-sig'))
    validate_initialization_anchor(init)
    material_path=Path(init['material_parameters_file'])
    if sha(material_path)!=init['material_parameters_sha256']:raise ValueError('Material parameter source changed')
    material=json.loads(material_path.read_text(encoding='utf-8-sig'))
    native,source=load_native(root,'C')
    t=np.concatenate([r['t'] for r in native[:2]]);t=t-t[0]
    current=np.concatenate([r['i'] for r in native[:2]])
    voltage=np.concatenate([r['v'] for r in native[:2]])
    temperature=np.concatenate([r['T'] for r in native[:2]])+273.15
    if not np.all(np.diff(t)>0) or not all(np.all(np.isfinite(x)) for x in (t,current,voltage,temperature)):
        raise ValueError('Finite monotonic recorded drive required')
    if not np.all((temperature>273.15)&(temperature<323.15)):raise ValueError('Initial-phase temperature quality gate failed')
    config=json.loads((root/'work/validation_proposal_20261103/physical_BOL_fit_v3/declared_experiment.json').read_text())
    report=dict(status='recorded_current_precharge_diagnostic_completed',source=source,
        initialization_sha256=sha(initialization),code_sha256=sha(__file__),
        initialization_contract_sha256=sha(Path(__file__).with_name('initialization_contract.py')),
        anchor_policy_id=init['anchor_policy_id'],scenarios={},physical_validation_complete=False,
        scope='recorded initial rest and CC only; excludes CV/discharge; voltage limit never raised',
        current_interpolation='linear between recorded samples, including 2ms transition; raw sign inverted mA to A',
        surface_T_is_not_coolant_or_predicted_temperature=True)
    for label in ('isothermal_25C','imposed_recorded_surface_T'):
        p=pybamm.ParameterValues('OKane2022');p.update(material['parameter_updates'])
        p.update({'Current function [A]':pybamm.Interpolant(t,current,pybamm.t),
                  'Initial temperature [K]':298.15 if label=='isothermal_25C' else float(temperature[0]),
                  'Ambient temperature [K]':298.15 if label=='isothermal_25C' else pybamm.Interpolant(t,temperature,pybamm.t)})
        limits={k:float(p[k]) for k in ('Lower voltage cut-off [V]','Upper voltage cut-off [V]')}
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,
            var_pts=config['fine']['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=1e-8,atol=1e-10))
        try:
            sol=sim.solve(t_eval=[0,float(t[-1])],t_interp=np.arange(0,t[-1]+.5,1)[np.arange(0,t[-1]+.5,1)<=t[-1]],
                          initial_soc=init['effective_initial_soc'])
            mt=np.asarray(sol['Time [s]'].entries);mv=np.asarray(sol['Voltage [V]'].entries)
            mi=np.asarray(sol['Current [A]'].entries);mask=(t>=mt[0])&(t<=mt[-1])
            e=np.interp(t[mask],mt,mv)-voltage[mask]
            np.savez_compressed(output/(label+'.npz'),model_t=mt,model_v=mv,model_i=mi,
                                observed_t=t,observed_v=voltage,observed_i=current,observed_surface_T=temperature)
            report['scenarios'][label]=dict(status='solved',termination=str(sol.termination),
                observed_end_s=float(t[-1]),model_end_s=float(mt[-1]),
                recorded_interval_completed=bool(mt[-1]>=t[-1]-1e-6),
                elapsed_time_coverage_fraction=float(mt[-1]/t[-1]),sample_coverage_fraction=float(mask.mean()),
                voltage_RMSE_mV_on_covered_samples=float(np.sqrt(np.mean(e*e))*1000),voltage_limits=limits)
        except Exception as e:
            report['scenarios'][label]=dict(status='failed',error_type=type(e).__name__,error=str(e),voltage_limits=limits)
    write(output/'report.json',report)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--initialization',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();print(json.dumps(run(args.root,args.initialization,args.output),indent=2))
