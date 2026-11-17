"""Bounded native-preparation/fixed-Ah SPMe numerical screen, not lifetime proof.

Actual prefix RPTs propagate. A terminal virtual RPT is one copied branch whose
five steps propagate within that copy. Frozen measured_protocol_replay is reused
for validation/main state routing and is not edited.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import measured_protocol_replay as replay
from initialization_contract import validate_initialization_anchor


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def completed_screen_case(result):
    """A saved success report cannot override a failed worker process."""
    if (result.get('status')!='short_screen_completed' or result.get('process_exit_code')!=0
            or result.get('source_hashes_unchanged') is not True
            or result.get('copied_RPT_main_state_unchanged') is not True):
        return False
    values=[result.get(k) for k in ('Q0_Ah','Qref_Ah','SOH_percent')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        return False
    q0,qref,soh=values
    return math.isclose(soh,100*qref/q0,rel_tol=0,abs_tol=1e-10)


def guard(config):
    for path,expected in config['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Frozen source changed during screen: '+path)


def phase(name,operation,value,duration,termination,expected,role):
    s=dict(role=role,source_ref='Native C RPT0/F004 preparation and fixed730mAh bounded numerical probe: '+name,
           phase=name,operation=operation,duration_s=duration,termination=termination,
           expected_termination=expected,period_s=1.,temperature_K=298.15)
    if operation!='rest':s['value']=value
    return s


def full_rpt(role,prefix):
    return [phase(prefix+'_initial_rest','rest',None,120.,[],'final time',role),
            phase(prefix+'_CC','current',-1.5,20000.,['4.2 V'],'event: Voltage > 4.2 [V] [experiment]',role),
            phase(prefix+'_CV','voltage',4.2,20000.,['0.05 A'],'event: abs(Current [A]) < 0.05 [A] [experiment]',role),
            phase(prefix+'_rest','rest',None,7230.,[],'final time',role),
            phase(prefix+'_C10','current',.5,50000.,['2.5 V'],'event: Voltage < 2.5 [V] [experiment]',role)]


def make_protocol(parameters,initialization,model_config,case):
    soc=initialization['effective_initial_soc']
    if isinstance(soc,bool) or not isinstance(soc,(int,float)) or not math.isfinite(soc) or not 0<=soc<=1:
        raise ValueError('Finite effective initial SOC required')
    overrides=parameters['parameter_updates']
    allowed={'Negative electrode active material volume fraction','Positive electrode active material volume fraction',
             'Initial concentration in negative electrode [mol.m-3]'}
    if set(overrides)!=allowed:raise ValueError('Only the frozen three BOL material updates are supported')
    if parameters.get('C_only_fit') is not True:raise ValueError('C-only frozen material parameters required')
    initial=full_rpt('rpt','initial_actual_RPT')
    prep=[phase('preparation_initial_rest','rest',None,3600.,[],'final time','ageing'),
          phase('preparation_CC','current',-1.5,20000.,['4.2 V'],'event: Voltage > 4.2 [V] [experiment]','ageing'),
          phase('preparation_CV','voltage',4.2,20000.,['0.05 A'],'event: abs(Current [A]) < 0.05 [A] [experiment]','ageing'),
          phase('preparation_postCV_rest','rest',None,7200.,[],'final time','ageing'),
          phase('preparation_730mAh_discharge','current',1.5,1752.,['2.5 V'],'final time','ageing'),
          phase('preparation_post_discharge_rest','rest',None,120.,[],'final time','ageing')]
    loops=[]
    for i in range(2):
        loops.extend([phase(f'loop{i+1}_discharge','current',5.,525.6,['2.5 V'],'final time','ageing'),
                      phase(f'loop{i+1}_charge','current',-1.5,1752.,['4.2 V'],'final time','ageing')])
    p=dict(cell_model='LG M50T effective C-only BOL candidate; bounded short numerical screen',
        protocol_id='native_preparation_fixed730mAh_two_loops_'+case['name'],
        source_ref='Native C RPT0 + F004 prep durations; Kirkaldy2024 Table3 fixed-Ah cycle basis',
        initial_state_source='C-only pre-charge-rest voltage effective SOC; unknown formation/gradients remain assumptions',
        family='SPMe',parameter_set=model_config['parameter_set'],options=copy.deepcopy(model_config['options']),
        mesh_points=dict(x_n=case['x_points'],x_s=case['x_points'],x_p=case['x_points'],r_n=case['r_points'],r_p=case['r_points']),
        initial_soc=soc,initial_temperature_K=298.15,rtol=case['rtol'],atol=case['rtol']/100,
        max_wall_s=600,parameter_overrides=copy.deepcopy(overrides),steps=initial+prep+loops+full_rpt('diagnostic','copied_postage_RPT'))
    return replay.validate_protocol(p)


def replay_terminal_diagnostic_group(protocol,initial_state,advance,clone=copy.deepcopy,fingerprint=None):
    """Main prefix uses frozen replay routing; one terminal copy carries its RPT."""
    replay.validate_protocol(protocol)
    first=next((i for i,s in enumerate(protocol['steps']) if s['role']=='diagnostic'),len(protocol['steps']))
    if first==len(protocol['steps']):raise ValueError('Terminal diagnostic RPT group required')
    if any(s['role']!='diagnostic' for s in protocol['steps'][first:]):
        raise ValueError('Only a terminal contiguous diagnostic group is supported')
    main_protocol=copy.deepcopy(protocol);main_protocol['steps']=protocol['steps'][:first]
    main,records=replay.replay_sequence(main_protocol,initial_state,advance,clone=clone)
    before=fingerprint(main) if fingerprint else None
    branch=clone(main)
    for s in protocol['steps'][first:]:
        branch,obs=advance(branch,s)
        if branch is None:raise ValueError('Missing terminal diagnostic state')
        records.append(dict(role='diagnostic',propagated=False,branch_propagated=True,**obs))
    after=fingerprint(main) if fingerprint else None
    if fingerprint and before!=after:raise ValueError('Copied diagnostic RPT mutated the main state')
    return main,records,dict(main_fingerprint_before=before,main_fingerprint_after=after,
                             copied_RPT_main_state_unchanged=before==after if fingerprint else None)


def summarize(records):
    q0=next(r['discharge_Ah'] for r in records if r['phase']=='initial_actual_RPT_C10')
    qref=next(r['discharge_Ah'] for r in records if r['phase']=='copied_postage_RPT_C10')
    if not all(math.isfinite(v) and v>0 for v in (q0,qref)):raise ValueError('Positive full RPT capacities required')
    groups={}
    for group,prefix in [('actual_initial_RPT','initial_actual_RPT'),('preparation','preparation'),('cycling_only','loop'),('copied_diagnostic_RPT','copied_postage_RPT')]:
        rows=[r for r in records if r['phase'].startswith(prefix)]
        groups[group]={key:sum(r[key] for r in rows) for key in ('charge_Ah','discharge_Ah')}
        groups[group]['absolute_throughput_Ah']=groups[group]['charge_Ah']+groups[group]['discharge_Ah']
    return dict(Q0_Ah=q0,Qref_Ah=qref,SOH_percent=100*qref/q0,throughput_by_group=groups,
        cycling_only_absolute_throughput_EFC_model_fixed_Q0=groups['cycling_only']['absolute_throughput_Ah']/(2*q0),
        main_absolute_throughput_EFC_including_preparation_model_fixed_Q0=sum(groups[g]['absolute_throughput_Ah'] for g in ['preparation','cycling_only'])/(2*q0),
        EFC_denominator='fixed model initial C/10 RPT capacity; model numerical-probe axis, not an observed EOL label',
        SOH_exposure_scope='actual initial RPT followed by native preparation and two loops; diagnostic copy is not propagated to main',
        physical_validation_complete=False,full_measured_history_reproduced=False,EFC80='not_evaluated')


def worker(config,case,out):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import numpy as np
    import pybamm
    guard(config);started=time.monotonic()
    p=make_protocol(config['parameters'],config['initialization'],config['model_config'],case)
    write(out/'protocol.json',p)
    parameters=pybamm.ParameterValues(p['parameter_set']);parameters.update(p['parameter_overrides'])
    parameters.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
    counter=[0]
    def fingerprint(state):
        return hashlib.sha256(np.asarray(state.all_ys[-1],dtype=float).tobytes()+np.asarray(state.all_ts[-1],dtype=float).tobytes()).hexdigest()
    def advance(state,spec):
        counter[0]+=1
        kwargs=dict(duration=spec['duration_s'],period=spec['period_s'],termination=spec['termination'],temperature=spec['temperature_K'],skip_ok=False)
        make=getattr(pybamm.step,spec['operation']);step=make(**kwargs) if spec['operation']=='rest' else make(spec['value'],**kwargs)
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=replay.normalized_model_options(p['options'])),
            parameter_values=parameters.copy(),var_pts=p['mesh_points'],experiment=pybamm.Experiment([step]),
            solver=pybamm.IDAKLUSolver(rtol=p['rtol'],atol=p['atol']))
        args={'initial_soc':p['initial_soc']} if state is None else {'starting_solution':state}
        result=sim.solve(calc_esoh=False,**args);piece=result.cycles[-1].steps[-1]
        raw={name:np.asarray(piece[name].entries,dtype=float) for name in ('Time [s]','Current [A]','Voltage [V]','Discharge capacity [A.h]')}
        raw['_input_state_y']=np.asarray(state.all_ys[-1],dtype=float).reshape(-1) if state is not None else np.array([])
        raw['_terminal_state_y']=np.asarray(result.last_state.all_ys[-1],dtype=float).reshape(-1)
        np.savez_compressed(out/f'step_{counter[0]:03d}_{spec["phase"]}.npz',**raw)
        if not all(np.all(np.isfinite(a)) for a in raw.values()):raise ValueError('Nonfinite raw step output')
        t,I,Q=raw['Time [s]'],raw['Current [A]'],raw['Discharge capacity [A.h]']
        if len(t)<2 or not np.all(np.diff(t)>0):raise ValueError('Non-increasing step time')
        dq=float(Q[-1]-Q[0]);integral=float(np.trapezoid(I,t)/3600)
        if abs(dq-integral)>1e-7+1e-4*abs(dq):raise ValueError('Current/capacity integral discrepancy')
        if str(piece.termination)!=spec['expected_termination']:raise ValueError('Unexpected endpoint at '+spec['phase']+': '+str(piece.termination))
        if spec['expected_termination']=='final time' and abs(float(t[-1]-t[0])-spec['duration_s'])>1e-6:
            raise ValueError('Fixed-duration exposure terminated early')
        if 'loop' in spec['phase'] or '730mAh' in spec['phase']:
            if abs(abs(dq)-.73)>1e-6:raise ValueError('Fixed730mAh step did not pass730mAh')
        obs=dict(phase=spec['phase'],t_start_s=float(t[0]),t_end_s=float(t[-1]),signed_capacity_Ah=dq,
            charge_Ah=float(np.trapezoid(np.maximum(-I,0),t)/3600),discharge_Ah=float(np.trapezoid(np.maximum(I,0),t)/3600),
            termination=str(piece.termination),input_fingerprint=fingerprint(state) if state is not None else None,
            terminal_fingerprint=fingerprint(result.last_state))
        write(out/f'step_{counter[0]:03d}_{spec["phase"]}.json',obs)
        return result.last_state.copy(),obs
    main,records,copy_audit=replay_terminal_diagnostic_group(p,None,advance,clone=lambda s:None if s is None else s.copy(),fingerprint=fingerprint)
    report=dict(status='short_screen_completed',case=case,**summarize(records),records=records,**copy_audit,
                source_hashes_unchanged=True,pybamm_version=pybamm.__version__,elapsed_s=time.monotonic()-started)
    guard(config);write(out/'report.json',report);return report


def run(parameters_path,initialization_path,output):
    output.mkdir(parents=True,exist_ok=False)
    initialization=read_json(initialization_path)
    validate_initialization_anchor(initialization)
    if parameters_path is None:parameters_path=Path(initialization['material_parameters_file'])
    if sha(parameters_path)!=initialization['material_parameters_sha256']:raise ValueError('Initialization/material identity mismatch')
    parameters=read_json(parameters_path)
    model_path=Path(__file__).resolve().parents[1]/'configs/model.json'
    sources=[parameters_path,initialization_path,model_path,Path(__file__),Path(replay.__file__),
             Path(__file__).with_name('initialization_contract.py')]
    source=initialization.get('source_C',{})
    if source:
        if sha(source['path'])!=source['sha256']:raise ValueError('C initialization observation source changed')
        sources.append(Path(source['path']))
    config=dict(parameters=parameters,initialization=initialization,model_config=read_json(model_path),
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        cases=[dict(name='x10_r20_rtol1e-7',x_points=10,r_points=20,rtol=1e-7),
               dict(name='x20_r40_rtol1e-8',x_points=20,r_points=40,rtol=1e-8)],
        numerical_screen_tolerance_pp=1e-4,criterion_scope='predeclared practical short numerical screen, not certified error bound',
        physical_validation_complete=False,AWS_started=False)
    write(output/'declaration.json',config);results=[]
    for case in config['cases']:
        dest=output/case['name'];dest.mkdir()
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker','--config',str(output/'declaration.json'),
                 '--case',case['name'],'--output',str(dest)]
        with (dest/'worker.log').open('w',encoding='utf-8') as log:
            try:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                result=read_json(dest/'report.json')
                result['process_exit_code']=process.returncode
            except subprocess.TimeoutExpired:result=dict(status='wall_limit',max_wall_s=600,physical_validation_complete=False)
        results.append(result)
    guard(config)
    completed=[r for r in results if completed_screen_case(r)]
    spread=abs(completed[0]['SOH_percent']-completed[1]['SOH_percent']) if len(completed)==2 else None
    summary=dict(status='short_numerical_screen_completed' if len(completed)==2 else 'short_numerical_screen_incomplete',cases=results,
        SOH_span_pp=spread,numerical_screen_tolerance_pp=1e-4,
        numerical_screen_pass=spread is not None and spread<=1e-4,
        criterion_scope=config['criterion_scope'],independent_initialization_per_mesh=True,
        actual_initial_RPT_propagated=True,diagnostic_group_is_one_contiguous_copied_branch=True,
        physical_validation_complete=False,full_measured_history_reproduced=False,EFC80='not_evaluated',AWS_started=False)
    write(output/'report.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
    return 0 if summary['numerical_screen_pass'] else 2


def main():
    a=argparse.ArgumentParser();a.add_argument('--parameters',type=Path);a.add_argument('--initialization',type=Path)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--worker',action='store_true');a.add_argument('--config',type=Path);a.add_argument('--case')
    args=a.parse_args()
    if not args.worker:
        if args.initialization is None:a.error('--initialization is required')
        return run(args.parameters,args.initialization,args.output)
    else:
        try:
            config=read_json(args.config);case=next(c for c in config['cases'] if c['name']==args.case)
            worker(config,case,args.output)
        except Exception as e:
            write(args.output/'report.json',dict(status='short_screen_failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),physical_validation_complete=False))
            raise


if __name__=='__main__':raise SystemExit(main())
