"""C-only bounded counterexample test on charge, discharge and relaxed OCV.

Recharge and the six-hour rest endpoint are calibration data in this experiment.
C/2, GITT and D are not opened. Engineering weights are not noise estimates.
"""
from __future__ import annotations
import argparse,json,math,os,time,sys
from pathlib import Path
from joint_spme_bol_fit import BOUNDS as MATERIAL_BOUNDS,derive_updates,homogeneous_state,select_rest_anchor,interpolate_required,sha,write

# The first three physical bounds are unchanged. Dn range is an engineering search domain.
BOUNDS=[MATERIAL_BOUNDS[0]+[-1.],MATERIAL_BOUNDS[1]+[1.]]

def scale_diffusivity(original,multiplier):
    if not math.isfinite(multiplier) or multiplier<=0:raise ValueError('Positive finite diffusivity multiplier required')
    if callable(original):
        def diffusivity(sto,T):return original(sto,T)*multiplier
        return diffusivity
    return original*multiplier

def select_shape_samples(raw_t,raw_v,cutoff_raw_s):
    import numpy as np
    t=np.asarray(raw_t)-cutoff_raw_s;v=np.asarray(raw_v);requested=[60.,600.,3600.,10800.,21600.];rows=[]
    if not np.all(np.diff(t)>0):raise ValueError('Increasing unchanged raw shape clock required')
    for threshold in requested:
        index=int(np.searchsorted(t,threshold))
        if index>=len(t):
            if threshold<21600:raise ValueError('Required early shape observation is absent')
            rows.append(dict(requested_s=threshold,raw_index=None,coverage=False));continue
        rows.append(dict(requested_s=threshold,raw_index=index,raw_t_s=float(raw_t[index]),derived_elapsed_s=float(t[index]),raw_voltage_V=float(v[index]),coverage=True))
    return rows

def protocol_steps(pybamm):
    return [pybamm.step.rest(120,skip_ok=False),
        pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),
        pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),
        pybamm.step.rest(7230,skip_ok=False),
        pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False),
        pybamm.step.rest(21600,skip_ok=False),pybamm.step.rest(30,skip_ok=False),
        pybamm.step.current(-.5,duration=50000,termination='4.2 V',skip_ok=False)]

def stable_rest_endpoint(t,v,duration_s=21600,window_s=600):
    import numpy as np
    t=np.asarray(t)-t[0];v=np.asarray(v);mask=(t>=duration_s-window_s)&(t<=duration_s)
    if mask.sum()<3:raise ValueError('At least3 native samples required in declared final600s rest window')
    return float(np.median(v[mask])),dict(sample_count=int(mask.sum()),median_V=float(np.median(v[mask])),span_V=float(np.ptp(v[mask])),selected_time_first_s=float(t[mask][0]),selected_time_last_s=float(t[mask][-1]))

def metric_residuals(raw,obs):
    import numpy as np
    cc,cv,dis,rest,charge=raw[1],raw[2],raw[4],raw[5],raw[7]
    qc=-(cc['q']-cc['q'][0]);qd=dis['q']-dis['q'][0];qr=-(charge['q']-charge['q'][0])
    tc=float(cc['t'][-1]-cc['t'][0]);tv=float(cv['t'][-1]-cv['t'][0]);tr=float(charge['t'][-1]-charge['t'][0])
    cvt=cv['t']-cv['t'][0];rvoltage,rmeta=stable_rest_endpoint(rest['t'],rest['v'])
    curves={'CC_voltage':(interpolate_required(qc,cc['v'],obs['CC_q'])-obs['CC_v'])/.02,
            'CV_current':(interpolate_required(cvt,cv['i'],obs['CV_t'])-obs['CV_i'])/.05,
            'C10_discharge_voltage':(interpolate_required(qd,dis['v'],obs['C10_q'])-obs['C10_v'])/.02,
            'C10_recharge_voltage':(interpolate_required(qr,charge['v'],obs['recharge_q'])-obs['recharge_v'])/.02}
    scalars={'CC_time':(tc-obs['CC_duration_s'])/300.,'CC_Ah':(qc[-1]-obs['CC_total_Ah'])/.02,
             'CV_time':(tv-obs['CV_duration_s'])/300.,'C10_discharge_Ah':(qd[-1]-obs['C10_total_Ah'])/.02,
             'relaxed_voltage':(rvoltage-obs['rest_median_V'])/.02,
             'C10_recharge_Ah':(qr[-1]-obs['recharge_total_Ah'])/.02,'C10_recharge_time':(tr-obs['recharge_duration_s'])/300.}
    rest_time=rest['t']-rest['t'][0];shape=interpolate_required(rest_time,rest['v'],obs['rest_shape_t']);curves['relaxation_shape']=(shape-obs['rest_shape_v'])/.02
    terminal_time=float(obs['rest_shape_terminal_t']);terminal_covered=rest_time[0]<=terminal_time<=rest_time[-1];terminal_prediction=float(np.interp(terminal_time,rest_time,rest['v'])) if terminal_covered else None
    blocks=[a/math.sqrt(len(a)) for a in curves.values()]+[np.array([v]) for v in scalars.values()]
    residual=np.concatenate(blocks);components={k:float(np.mean(v*v)) for k,v in curves.items()};components.update({k:float(v*v) for k,v in scalars.items()})
    details=dict(components=components,CC_duration_s=tc,CV_duration_s=tv,CC_transferred_Ah=float(qc[-1]),C10_capacity_Ah=float(qd[-1]),post_C10_rest_model_median_V=rvoltage,post_C10_rest_observed_median_V=obs['rest_median_V'],post_C10_rest_residual_mV=1000*(rvoltage-obs['rest_median_V']),post_C10_rest_window=rmeta,recharge_capacity_Ah=float(qr[-1]),recharge_duration_s=tr,CC_grid_voltage_RMSE_mV=float(np.sqrt(np.mean(curves['CC_voltage']**2))*20),CV_grid_current_RMSE_A=float(np.sqrt(np.mean(curves['CV_current']**2))*.05),C10_grid_voltage_RMSE_mV=float(np.sqrt(np.mean(curves['C10_discharge_voltage']**2))*20),recharge_grid_voltage_RMSE_mV=float(np.sqrt(np.mean(curves['C10_recharge_voltage']**2))*20))
    details.update(rest_shape_times_s=obs['rest_shape_t'].tolist(),rest_shape_model_V=shape.tolist(),rest_shape_observed_V=obs['rest_shape_v'].tolist(),rest_shape_residual_mV=(1000*(shape-obs['rest_shape_v'])).tolist(),rest_shape_terminal_diagnostic=dict(actual_elapsed_s=terminal_time if terminal_time>=0 else None,covered=bool(terminal_covered),model_V=terminal_prediction,observed_V=float(obs['rest_shape_terminal_v']),extrapolated=False,used_in_loss=False))
    return residual,details

def run(root,output,max_calls=140,max_wall_s=1200):
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import numpy as np,pybamm
    from scipy.optimize import least_squares
    from initialize_spme_from_rest import load_native
    from diagnose_low_rate_recharge import raw_charge_Ah
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();work=root/'work/validation_proposal_20261103';latest=work/'latest_spme_review_20261103';cfgpath=work/'physical_BOL_fit_v3/declared_experiment.json';config=json.loads(cfgpath.read_text());oldpath=work/'physical_BOL_fit_v3/parameters_frozen.json';old=json.loads(oldpath.read_text());jointpath=latest/'joint_initial_fit_v3/parameters_frozen.json';joint=json.loads(jointpath.read_text());native,csource=load_native(root,'C');target,anchor=select_rest_anchor(native[0]);sys.path.insert(0,str(root/'work/preprocess'));import common
    cols,meta,_=common.read_mpt(Path(csource['path']));runs,_=common.split_runs(cols)
    if [int(cols['ns'][a]) for a,b in runs]!=list(range(10)):raise ValueError('Expected C long01 Ns0..9')
    n=[dict(t=cols['t'][a:b],v=cols['V'][a:b],i=-cols['I'][a:b]/1000.) for a,b in runs]
    def ah(phase,sign):return sign*raw_charge_Ah(phase['t'],-phase['i']*1000.)
    ccq=ah(n[1],1);dq=ah(n[5],-1);rq=ah(n[8],1);ccg=np.linspace(0,.9*ccq[-1],81);dg=np.linspace(.02*dq[-1],.98*dq[-1],97);rg=np.linspace(.02*rq[-1],.98*rq[-1],97);cvt=n[2]['t']-n[2]['t'][0];cvg=np.linspace(0,cvt[-1],81);rest_v,rest_meta=stable_rest_endpoint(n[6]['t'],n[6]['v'])
    obs=dict(CC_q=ccg,CC_v=np.interp(ccg,ccq,n[1]['v']),CV_t=cvg,CV_i=np.interp(cvg,cvt,n[2]['i']),C10_q=dg,C10_v=np.interp(dg,dq,n[5]['v']),recharge_q=rg,recharge_v=np.interp(rg,rq,n[8]['v']),CC_duration_s=float(n[1]['t'][-1]-n[1]['t'][0]),CV_duration_s=float(cvt[-1]),CC_total_Ah=float(ccq[-1]),C10_total_Ah=float(dq[-1]),rest_median_V=rest_v,recharge_total_Ah=float(rq[-1]),recharge_duration_s=float(n[8]['t'][-1]-n[8]['t'][0]))
    shape_rows=select_shape_samples(n[6]['t'],n[6]['v'],float(n[5]['t'][-1]))
    for row in shape_rows:
        row['raw_source_row']=runs[6][0]+row['raw_index'] if row['coverage'] else None
    obs.update(rest_shape_t=np.array([a['derived_elapsed_s'] for a in shape_rows[:4]]),rest_shape_v=np.array([a['raw_voltage_V'] for a in shape_rows[:4]]),rest_shape_terminal_t=float(shape_rows[4]['derived_elapsed_s']) if shape_rows[4]['coverage'] else -1.,rest_shape_terminal_v=float(shape_rows[4]['raw_voltage_V']) if shape_rows[4]['coverage'] else 0.);materialbest=json.loads((latest/'bidirectional_BOL_fit_v1/parameters_frozen.json').read_text());base=pybamm.ParameterValues('OKane2022');original_Dn=base['Negative particle diffusivity [m2.s-1]']
    starting=[[old['s_Q'],old['s_Q'],old['delta_Q_Li_Ah'],-1.],materialbest['chosen_C']['parameters']+[math.log10(.2)],[.94,.90,.30,math.log10(.2)]];guards={str(p.resolve()):sha(p) for p in [Path(csource['path']),Path(__file__),Path(__file__).with_name('joint_spme_bol_fit.py'),Path(__file__).with_name('diagnose_low_rate_recharge.py'),Path(__file__).with_name('fit_spme_bidirectional_bol.py'),latest/'bidirectional_BOL_fit_v1/parameters_frozen.json',oldpath,jointpath,cfgpath]};declaration=dict(scope='C-only4parameter material/Dn estimation; rest-shape/recharge included in C calibration',model='SPMe',parameter_set='OKane2022',bounds=BOUNDS,variables=['s_n','s_p','delta_Q_Li_Ah','log10_Dn_multiplier'],diffusivity_multiplier_domain='0.1..10 engineering range, not literature prior; original sto/T/Ea function multiplied intact',base_driver_source=dict(path=str(Path(__file__).with_name('fit_spme_bidirectional_bol.py')),sha256=sha(Path(__file__).with_name('fit_spme_bidirectional_bol.py'))),shape_clock_origin='last raw C10discharge cutoff timestamp; derived elapsed only; raw times unmodified',rest_shape_rows=shape_rows,shape_loss_policy='first4actual selected raw rows groupmean/V20mV; final21600s sample diagnostic only because terminal median already weighted',starting_points=starting,options=config['options'],C_source=csource,anchor=anchor,anchor_policy_id='declared_rest_interior_last3_v1',rest_endpoint_observation=rest_meta,grid_policy='unchanged absoluteAh CC0-90% and discharge2-98%; recharge2-98%; CVfixed observed seconds; no extrapolation or stretching',scales=dict(V=.02,Ah=.02,time_s=300,I_A=.05),loss_policy='Each voltage/current curve contributes mean squared normalized residual; each scalar contributes squared normalized residual. Original11metric terms unchanged; added one four-sample relaxation-shape group. Endpoint sample not added to loss.',fit_mesh=config['fit']['mesh_points'],fit_rtol=config['fit']['rtol'],fit_atol=config['fit']['atol'],fit_output_period_s=10,fine_mesh=config['fine']['mesh_points'],fine_rtol=1e-8,fine_atol=1e-10,fine_output_period_s=1,max_model_calls=max_calls,max_wall_s=max_wall_s,temperature='25C isothermal diagnostic; raw surface temperature preserved, not imposed',C2_GITT_D_used_for_fit=False,source_sha256=guards)
    write(out/'declared_experiment.json',declaration);np.savez_compressed(out/'fixed_observations.npz',**obs);np.savez_compressed(out/'native_observations.npz',**{f'Ns{i}_{k}':v for i,row in enumerate(n) for k,v in row.items()});snap=out/'source_snapshots';snap.mkdir()
    for i,path in enumerate(guards):p=Path(path);(snap/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    expected=list(config['expected_terminations'])+['final time','final time','event: Voltage > 4.2 [V] [experiment]'];count=0;cache={};journal=[];best=None;lastJ=None;fixed_len=81+81+97+97+7+4
    def evaluate(x,fine=False,reserve=8):
        nonlocal count,best
        key=tuple(float(z) for z in x)+(bool(fine),)
        if key in cache:return cache[key]
        if count>=max_calls-reserve or time.monotonic()-began>max_wall_s-90:raise RuntimeError('Declared model/wall budget exhausted')
        count+=1;number=count;record=dict(call=number,parameters=[float(z) for z in x],fine=fine,base_parameter_set='OKane2022',status='started');write(out/f'candidate_{number:03d}.json',record)
        try:
            updates,inventory=derive_updates(base,*x[:3]);p=base.copy();p.update(updates);p.update({'Negative particle diffusivity [m2.s-1]':scale_diffusivity(original_Dn,10.**float(x[3]))});state,state_meta=homogeneous_state(p,inventory,target);p.update(state);p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15});res=config['fine'] if fine else config['fit'];sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,experiment=pybamm.Experiment(protocol_steps(pybamm),period=1 if fine else 10,temperature=298.15),var_pts=res['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=1e-8 if fine else res['rtol'],atol=1e-10 if fine else res['atol']));sol=sim.solve(calc_esoh=False);pieces=[a.steps[-1] for a in sol.cycles];raw=[{k:np.asarray(a[v].entries) for k,v in [('t','Time [s]'),('v','Voltage [V]'),('i','Current [A]'),('q','Discharge capacity [A.h]')]} for a in pieces];np.savez_compressed(out/f'candidate_{number:03d}.npz',**{f'phase{i}_{k}':v for i,row in enumerate(raw) for k,v in row.items()})
            if len(pieces)!=8 or [str(a.termination) for a in pieces]!=expected:raise ValueError('Native eightphase event sequence mismatch')
            if abs(raw[0]['v'][-1]-target)>.001:raise ValueError('Initial rest endpoint misses declared anchor')
            record.update(negative_diffusivity_multiplier=10.**float(x[3]),C10_capacity_Ah=float(raw[4]['q'][-1]-raw[4]['q'][0]),recharge_capacity_Ah=float(-(raw[7]['q'][-1]-raw[7]['q'][0])),post_C10_rest_model_median_V=stable_rest_endpoint(raw[5]['t'],raw[5]['v'])[0])
            if fine:
                summaries=[]
                channels=['Loss of lithium inventory [%]','Loss of capacity to negative SEI [A.h]','Loss of capacity to negative SEI on cracks [A.h]','Loss of capacity to negative lithium plating [A.h]','Total lithium lost [mol]','Total lithium in particles [mol]','Total lithium in electrolyte [mol]','Total lithium [mol]','Negative electrode stoichiometry','Positive electrode stoichiometry','X-averaged negative particle surface stoichiometry','X-averaged positive particle surface stoichiometry','X-averaged negative electrode open-circuit potential [V]','X-averaged positive electrode open-circuit potential [V]','X-averaged negative electrode reaction overpotential [V]','X-averaged positive electrode reaction overpotential [V]','Loss of active material in negative electrode [%]','Loss of active material in positive electrode [%]']
                for pi,piece in enumerate(pieces):
                    values={};info={};missing=[]
                    for name in channels:
                        try:
                            value=np.asarray(piece[name].entries);values[name]=value;info[name]=dict(first=float(value.reshape(-1)[0]),last=float(value.reshape(-1)[-1]))
                        except KeyError:missing.append(name)
                    values['_terminal_state_y']=np.asarray(piece.last_state.all_ys[-1]).reshape(-1);np.savez_compressed(out/f'candidate_{number:03d}_internal_phase{pi}.npz',**values);summaries.append(dict(phase=pi,channels=info,unavailable_channels=missing))
                record['internal_state_diagnostics']=summaries
            residual,details=metric_residuals(raw,obs);record.update(status='evaluated',objective_sum_squares=float(residual@residual),updates=updates,initial_state_updates=state,initial_state=state_meta,inventory=inventory,**details)
            if not fine and (best is None or record['objective_sum_squares']<best['objective_sum_squares']):best=record.copy()
        except Exception as e:record.update(status='failed',error=type(e).__name__+': '+str(e));residual=np.full(fixed_len,1000+float(np.linalg.norm(x)))
        record['wall_s']=time.monotonic()-began;write(out/f'candidate_{number:03d}.json',record);journal.append(record);write(out/'journal.json',journal);cache[key]=(residual,record);return residual,record
    # Equilibrium coulomb-balance screening is algebraic only: no side-reaction loss,
    # no concentration gradient, and observed netAh rather than forced model capacity.
    netcharge=float(raw_charge_Ah(cols['t'][:runs[5][1]],cols['I'][:runs[5][1]])[-1]);static=[]
    for sn in [.85,.925,1.]:
        for sp in [.85,.925,1.]:
            for delta in [-.4,0.,.4]:
                item=dict(parameters=[sn,sp,delta],observed_net_charge_Ah=netcharge)
                try:
                    u,inv=derive_updates(base,sn,sp,delta);p=base.copy();p.update(u);_,m=homogeneous_state(p,inv,target);x=m['negative_stoichiometry']+netcharge/inv['Q_n_Ah'];y=m['positive_stoichiometry']-netcharge/inv['Q_p_Ah']
                    if not 0<x<1 or not 0<y<1:raise ValueError('Observed coulomb balance exceeds physical stoichiometry under no-side-loss assumption')
                    value=p['Positive electrode OCP [V]'](y)-p['Negative electrode OCP [V]'](x);value=float(np.asarray(value.evaluate() if hasattr(value,'evaluate') else value).item());item.update(status='screened',equilibrium_voltage_V=value,residual_mV=1000*(value-rest_v),x=x,y=y)
                except Exception as e:item.update(status='infeasible_or_outside_assumptions',error=str(e))
                static.append(item)
    write(out/'static_equilibrium_screen.json',dict(scope='27point algebraic sensitivity, not PDE feasibility proof',assumptions=['no side-reaction charge loss','homogeneous stoichiometry','observed charge balance, not independently predicted cutoff'],cases=static))
    import inspect,hashlib
    original_source=inspect.getsource(original_Dn) if callable(original_Dn) else str(original_Dn)
    probe=original_Dn(.5,298.15) if callable(original_Dn) else original_Dn
    probe=float(np.asarray(probe.evaluate() if hasattr(probe,'evaluate') else probe).item())
    write(out/'original_diffusivity_contract.json',dict(source=original_source,source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),nominal_probe_sto=.5,nominal_probe_T_K=298.15,nominal_probe_m2_s=probe,sto_and_temperature_dependence_unchanged=True,activation_energy_not_refitted=True))
    screens=[]
    for start in starting[:2]:
        _,record=evaluate(start);screens.append(record)
    write(out/'initial_candidate_screen.json',screens);print('초기 두 후보와 허용범위27점 정적진단 완료',flush=True)
    def jac(x):
        nonlocal lastJ
        f,r=evaluate(x)
        if r['status']!='evaluated':raise ValueError('Failed candidate cannot define physical Jacobian')
        columns=[]
        for i,h in enumerate([.001,.001,.005,.025]):
            step=h if x[i]+h<=BOUNDS[1][i] else-h;z=np.array(x).copy();z[i]+=step;g,q=evaluate(z)
            if q['status']!='evaluated':
                step=-step;z=np.array(x).copy();z[i]+=step
                if not BOUNDS[0][i]<=z[i]<=BOUNDS[1][i]:raise ValueError('No admissible finite difference direction')
                g,q=evaluate(z)
            if q['status']!='evaluated':raise ValueError('Both physical sensitivity directions failed')
            columns.append((g-f)/step)
        lastJ=np.column_stack(columns);return lastJ
    searches=[]
    for start in starting:
        try:
            result=least_squares(lambda x:evaluate(x)[0],start,jac=jac,bounds=BOUNDS,x_scale=[.05,.05,.1,.4],max_nfev=20,ftol=1e-4,xtol=1e-4,gtol=1e-4);searches.append(dict(start=start,x=result.x.tolist(),success=bool(result.success),message=str(result.message),nfev=result.nfev))
        except (ValueError,RuntimeError) as e:searches.append(dict(start=start,status='stopped',reason=str(e)))
        write(out/'searches.json',searches)
        if count>=max_calls-8 or time.monotonic()-began>max_wall_s-90:break
    if best is None:raise RuntimeError('No candidate covers all declared observed intervals')
    frozen=dict(scope='C-only4parameter material/Dn training candidate; recharge and rest-shape used for estimation',role='C_training_candidate_not_physical_validation',recharge_used_for_training=True,rest_shape_used_for_training=True,base_parameter_set='OKane2022',chosen_C=best,source_C=csource,low_rate_observations_used_in_fit=True,transport_changed=True,transport_change='negative particle diffusivity multiplier only; original sto/T function retained',negative_diffusivity_multiplier=10.**best['parameters'][3],D_opened_before_freeze=False,C2_GITT_D_used_for_fit=False,parameter_bounds=BOUNDS,source_sha256=sha(__file__));write(out/'parameters_frozen.json',frozen);freeze=sha(out/'parameters_frozen.json');fine=[]
    for x in [starting[0],best['parameters']]:
        _,r=evaluate(x,fine=True,reserve=0);fine.append(r)
    J=None
    if lastJ is not None:
        s=np.linalg.svd(lastJ*np.array([.05,.05,.1,.4]),compute_uv=False);J=dict(scaled_singular_values=s.tolist(),scaled_condition=float(s[0]/s[-1]),rank=int(np.linalg.matrix_rank(lastJ)),structural_identifiability=False,confidence_interval=False,profile_likelihood=False)
    report=dict(status='C_only_material_Dn_shape_fit_evaluated',model_calls=count,wall_s=time.monotonic()-began,initial_candidate_screen=screens,coarse_chosen=best,fine_retained=fine[0],fine_chosen=fine[1],searches=searches,local_sensitivity=J,parameter_at_bounds=[min(abs(v-lo),abs(hi-v))<.001 for v,lo,hi in zip(best['parameters'],*BOUNDS)],all_guarded_sources_unchanged=all(sha(path)==h for path,h in guards.items()),parameters_frozen_unchanged=sha(out/'parameters_frozen.json')==freeze,C2_GITT_D_used_for_fit=False,physical_validation_complete=False,SOH_validation_complete=False,AWS_started=False);write(out/'report.json',report);print(json.dumps({k:report[k] for k in ['status','model_calls','wall_s','parameter_at_bounds','all_guarded_sources_unchanged']},ensure_ascii=False));return report
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--max-model-calls',type=int,default=140);a.add_argument('--max-wall-s',type=int,default=1200);v=a.parse_args();run(v.root,v.output,v.max_model_calls,v.max_wall_s)
