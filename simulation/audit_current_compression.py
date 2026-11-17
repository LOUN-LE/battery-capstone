"""Real-PDE, bounded paired test; input compression is never exact replay."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from bounded_current_waveform import compress, verified_setpoint, POLICY
from run_corrected_history_trial import compile_route, waveform
from run_exposure_scenario_smoke import read,write,sha


def worker(a):
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import pybamm
    from composite_candidate_parameters import build_frozen_candidate
    w=a.root.resolve()/'work/validation_proposal_20261103/latest_spme_review_20261103'
    scenario=w/'raw_exposure_scenarios_v1/C_all_unresolved_rest_25C.json'
    cand=w/'composite_SiDn_C_fit_v2/parameters_frozen.json';anchor=w/'joint_initial_fit_v3/parameters_effective_initialization.json'
    _,raw,plan,_,arrays,pr=compile_route(a.root.resolve(),scenario,1)
    rawsteps=read(raw)['cell']['steps'];programs={x['whole_file_sha256']:x for x in read(pr)}
    p,options,meta=build_frozen_candidate(cand,anchor)
    options['SEI']=('solvent-diffusion limited','none');options['SEI porosity change']='false'
    mesh=dict(x_n=10,x_s=10,x_p=10,r_n=20,r_n_prim=20,r_n_sec=20,r_p=20)
    paths=[Path(__file__),Path(__file__).with_name('bounded_current_waveform.py'),Path(__file__).with_name('run_corrected_history_trial.py'),scenario,raw,pr]
    guards={str(x.resolve()):sha(x) for x in paths};guards.update(meta['source_sha256'])
    result=dict(status='running',policy=POLICY,physical_validation_complete=False,exact_replay_admitted=False,
                purpose='short engineering input-approximation screen, not long-exposure convergence',
                capacity_difference_screen_pp=1e-4,
                screen_limit_source='existing audit_calibrated_short_protocol.py numerical screen; not user physical acceptance',
                mesh=mesh,rtol=1e-7,atol=1e-9,source_sha256=guards,cases=[])
    write(a.output/'declared_experiment.json',result)
    selected=[]
    for item in plan:
        if item['kind']!='eligible_recorded_current':continue
        if a.source_prefix and not item['source_ref'].startswith(a.source_prefix):continue
        s=item['source_step'];ar=arrays.get(s)
        cc=verified_setpoint(rawsteps[s['raw_step_index']],s,ar,programs)
        if cc is None or any(np.sign(x[1])==np.sign(cc) for x in selected):continue
        table=waveform(s,ar);reduced,cm=compress(table,cc)
        if cm['compressed'] and cm['raw_samples']>=500 and table[-1,0]<=2000:
            selected.append((item,cc,table,reduced,cm))
        if len(selected)==2:break
    if len(selected)!=2:raise ValueError('Both source current signs needed')
    for case,(item,cc,full,reduced,cm) in enumerate(selected):
        np.savez_compressed(a.output/f'case{case}_input.npz',raw=full,compressed=reduced)
        variants={};saved={}
        for label,table in [('raw',full),('compressed',reduced)]:
            controls=[pybamm.step.current(table,duration=table[-1,0],skip_ok=False,period=1),
                      pybamm.step.rest(600,skip_ok=False,period=10),
                      pybamm.step.current(-.5,duration=86400,termination='4.2 V',skip_ok=False,period=10),
                      pybamm.step.voltage(4.2,duration=21600,termination='0.05 A',skip_ok=False,period=10),
                      pybamm.step.rest(3600,skip_ok=False,period=10),
                      pybamm.step.current(.5,duration=86400,termination='2.5 V',skip_ok=False,period=10)]
            sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=p.copy(),var_pts=mesh,
                experiment=pybamm.Experiment([tuple(controls)],temperature=298.15),solver=pybamm.IDAKLUSolver(rtol=1e-7,atol=1e-9))
            started=time.monotonic();print(case,label,'build start',flush=True)
            sim.build_for_experiment();built=time.monotonic();print(case,label,'built',built-started,flush=True)
            sol=sim.solve(calc_esoh=False);finished=time.monotonic();pieces=sol.cycles[-1].steps
            if len(pieces)!=6 or pieces[-1].termination!='event: Voltage < 2.5 [V] [experiment]':raise ValueError('Independent RPT incomplete')
            first=pieces[0];t,I,V,q=[np.asarray(first[v].entries,dtype=float) for v in ('Time [s]','Current [A]','Voltage [V]','Discharge capacity [A.h]')]
            qref=np.asarray(pieces[-1]['Discharge capacity [A.h]'].entries,dtype=float)
            y=np.asarray(first.last_state.all_ys[-1],dtype=float).reshape(-1)
            np.savez_compressed(a.output/f'case{case}_{label}.npz',t_s=t,I_A=I,V=V,Q_Ah=q,terminal_y=y,RPT_Q_Ah=qref)
            saved[label]=(t,V,y)
            variants[label]=dict(build_s=built-started,solve_s=finished-built,Qref_Ah=float(qref[-1]-qref[0]),
                                 applied_signed_Ah=float(q[-1]-q[0]),termination=str(first.termination))
            print(case,label,json.dumps(variants[label]),flush=True)
        t,v,y=saved['raw'];tt,vv,yy=saved['compressed']
        dv=np.interp(t,tt,vv)-v;diffpp=100*(variants['compressed']['Qref_Ah']/variants['raw']['Qref_Ah']-1)
        result['cases'].append(dict(source_ref=item['source_ref'],source_step=item['source_step'],compression=cm,variants=variants,
            voltage_max_difference_mV=float(max(abs(dv))*1000),voltage_RMS_difference_mV=float(np.sqrt(np.mean(dv*dv))*1000),
            terminal_state_max_scaled_difference=float(max(abs(yy-y)/np.maximum(1.,abs(y)))),
            Qref_relative_difference_pp=float(diffpp),short_screen_pass=abs(diffpp)<=result['capacity_difference_screen_pp']))
        write(a.output/'report.json',result)
    guards.update(arrays.guards);result.update(source_sha256=guards,source_hashes_unchanged=all(sha(x)==h for x,h in guards.items()),
        status='completed',short_screen_pass=all(x['short_screen_pass'] for x in result['cases']))
    write(a.output/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--max-wall-s',type=int,default=300);p.add_argument('--source-prefix',default='');a=p.parse_args()
    if a.worker:worker(a)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        try:
            with (a.output/'worker.log').open('w',encoding='utf-8') as f:
                r=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],stdout=f,stderr=subprocess.STDOUT,
                    timeout=a.max_wall_s,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            receipt=dict(status='finished',exit_code=r.returncode)
        except subprocess.TimeoutExpired:receipt=dict(status='terminated_at_budget',max_wall_s=a.max_wall_s)
        write(a.output/'run_receipt.json',receipt);print(json.dumps(receipt));raise SystemExit(0 if receipt.get('exit_code')==0 else 2)
