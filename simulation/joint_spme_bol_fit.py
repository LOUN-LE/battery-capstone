"""Bounded C-only joint initial SPMe fit on unwarped Ah and time axes.

Independent electrode loading factors replace the common-loading restriction.
A lithium-conserving homogeneous equilibrium root anchors the initial voltage;
it is not reconstruction of formation or concentration gradients.
"""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path

F=96485.33212331001
BOUNDS=[[.85,.85,-.4],[1.,1.,.4]]

def sha(path):
 import hashlib
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def derive_updates(p,s_n,s_p,delta):
 if not all(math.isfinite(float(x)) for x in (s_n,s_p,delta)) or not .85<=s_n<=1 or not .85<=s_p<=1 or not -.4<=delta<=.4:raise ValueError('Unchanged bounded electrode/inventory domain required')
 A=p['Electrode width [m]']*p['Electrode height [m]'];a={};coeff={}
 for side,scale in [('Negative',s_n),('Positive',s_p)]:
  active=p[side+' electrode active material volume fraction']*scale
  if not 0<active<=1-p[side+' electrode porosity']+1e-12:raise ValueError('Active plus pore volume invalid')
  a[side+' electrode active material volume fraction']=active
  coeff[side]=F*A*p[side+' electrode thickness [m]']*active/3600
 cn=p['Initial concentration in negative electrode [mol.m-3]']+delta/coeff['Negative'];cp=p['Initial concentration in positive electrode [mol.m-3]']
 if not 0<cn<p['Maximum concentration in negative electrode [mol.m-3]']:raise ValueError('Inventory shift outside physical stoichiometry')
 a['Initial concentration in negative electrode [mol.m-3]']=cn
 return a,dict(Q_n_Ah=coeff['Negative']*p['Maximum concentration in negative electrode [mol.m-3]'],Q_p_Ah=coeff['Positive']*p['Maximum concentration in positive electrode [mol.m-3]'],Q_Li_Ah=coeff['Negative']*cn+coeff['Positive']*cp)

def homogeneous_state(p,inventory,target):
 from scipy.optimize import brentq
 Qn,Qp,Li=[inventory[k] for k in ('Q_n_Ah','Q_p_Ah','Q_Li_Ah')]
 lo=max(1e-6,(Li-Qp*(1-1e-6))/Qn);hi=min(1-1e-6,(Li-Qp*1e-6)/Qn)
 if hi<=lo:raise ValueError('No feasible lithium-conserving stoichiometry interval')
 def voltage(x):
  y=(Li-Qn*x)/Qp;a=p['Positive electrode OCP [V]'](y)-p['Negative electrode OCP [V]'](x)
  import numpy as np
  return float(np.asarray(a.evaluate() if hasattr(a,'evaluate') else a).item())
 def at(v):return float(brentq(lambda x:voltage(x)-v,lo,hi,xtol=1e-10,maxiter=50))
 x=at(target);y=(Li-Qn*x)/Qp;x0=at(2.5);x100=at(4.2)
 if not x0<x<x100:raise ValueError('Initial rest state outside reviewed voltage window')
 if abs(Qn*x+Qp*y-Li)>1e-10:raise ValueError('Initial lithium inventory was not conserved')
 return {'Initial concentration in negative electrode [mol.m-3]':x*p['Maximum concentration in negative electrode [mol.m-3]'],'Initial concentration in positive electrode [mol.m-3]':y*p['Maximum concentration in positive electrode [mol.m-3]']},dict(negative_stoichiometry=x,positive_stoichiometry=y,equilibrium_SOC=(x-x0)/(x100-x0),initial_equilibrium_V=voltage(x),target_rest_endpoint_V=target,lithium_inventory_residual_Ah=Qn*x+Qp*y-Li)

def interpolate_required(x,y,grid):
 import numpy as np
 if len(x)<2 or not np.all(np.diff(x)>0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Finite monotonic model axis required')
 if grid[0]<x[0]-1e-10 or grid[-1]>x[-1]+1e-10:raise ValueError('Candidate does not cover fixed observed fitting interval; extrapolation refused')
 return np.interp(grid,x,y)

def select_rest_anchor(phase,duration_s=120.):
 import numpy as np
 t=np.asarray(phase['t'])-phase['t'][0];v=np.asarray(phase['v']);i=np.asarray(phase['i']);mask=(t<=duration_s)&(t>=0)&(abs(i)<=1e-6)
 if int(mask.sum())<3:raise ValueError('At least3 observed rest samples inside declared control window required')
 indices=np.flatnonzero(mask)[-3:];target=float(np.mean(v[indices]))
 return target,dict(rule='last3 zero-current samples inside declared120.0s rest; no raw deletion',duration_s=float(duration_s),selected_indices=indices.tolist(),selected_times_s=t[indices].tolist(),selected_voltage_V=v[indices].tolist(),boundary_candidates=[dict(index=int(k),t_s=float(t[k]),v_V=float(v[k]),i_A=float(i[k])) for k in np.flatnonzero(~mask)],retired_last3mean_V=float(np.mean(v[-3:])))


def run(root,output,max_calls=140,max_wall_s=900,max_nfev=30):
 for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
 os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
 import numpy as np
 import pybamm
 from scipy.optimize import least_squares
 from initialize_spme_from_rest import load_native,compare_phase
 from calibrate_m50t_bol import read_curve
 root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
 work=root/'work/validation_proposal_20261103';old=work/'physical_BOL_fit_v3';previous=json.loads((old/'parameters_frozen.json').read_text());oldconfig=json.loads((old/'declared_experiment.json').read_text());native,source=load_native(root,'C');cpath=root/'work/external_m50_benchmark/kirkaldy_expt2_cellC_RPT0_C10.csv';qobs,vobs=read_curve(cpath)
 native_q=np.r_[0.,np.cumsum(-.5*(native[1]['i'][1:]+native[1]['i'][:-1])*np.diff(native[1]['t'])/3600)]
 if not np.all(np.diff(native_q)>0):raise ValueError('Raw C charging Ah must increase')
 cc_grid=np.linspace(0,.9*native_q[-1],81);cc_v=np.interp(cc_grid,native_q,native[1]['v']);cv_time=native[2]['t']-native[2]['t'][0];cv_grid=np.linspace(0,cv_time[-1],81);cv_i=np.interp(cv_grid,cv_time,native[2]['i']);q_grid=np.linspace(.02*qobs[-1],.98*qobs[-1],97);q_v=np.interp(q_grid,qobs,vobs);target,anchor_meta=select_rest_anchor(native[0]);cc_time=float(native[1]['t'][-1]-native[1]['t'][0]);cv_duration=float(cv_time[-1]);base=pybamm.ParameterValues('OKane2022')
 declaration=dict(scope='C-only stagewise-constrained joint initial fit; homogeneous rest-voltage initialization',model='SPMe',parameter_set='OKane2022',options=oldconfig['options'],bounds=BOUNDS,variables=['negative_loading_scale','positive_loading_scale','inventory_offset_Ah'],starting_points=[[previous['s_Q'],previous['s_Q'],previous['delta_Q_Li_Ah']],[.9269659660008397,.9224942420352057,.1364153120959527]],engineering_scales=dict(V=.02,Ah=.02,time_s=300,I_A=.05),fixed_grid=dict(CC_charged_Ah=cc_grid.tolist(),CV_elapsed_s=cv_grid.tolist(),C10_discharged_Ah=q_grid.tolist()),target_native_CC_charge_Ah=float(native_q[-1]),target_CC_duration_s=cc_time,target_CV_duration_s=cv_duration,target_rest_V=target,rest_anchor=anchor_meta,retired_initial_anchor='joint_initial_fit_v2',source_C=source,source_C10=dict(path=str(cpath),sha256=sha(cpath)),initial_state='lithium-conserving homogeneous equilibrium root at measured terminal rest voltage; rest endpoint must agree within1mV',max_model_calls=max_calls,max_wall_s=max_wall_s,max_nfev_each_start=max_nfev,temperature='25C isothermal diagnostic; not raw surface temperature replay',D_used_for_fit=False,source_sha256=sha(__file__))
 write(out/'declared_experiment.json',declaration);np.savez_compressed(out/'fixed_observations.npz',CC_q=cc_grid,CC_v=cc_v,CV_t=cv_grid,CV_i=cv_i,C10_q=q_grid,C10_v=q_v)
 cache={};journal=[];count=0;best=None;fixed_len=len(cc_grid)+len(cv_grid)+len(q_grid)+4
 def evaluate(x,fine=False,reserve=5):
  nonlocal count,best
  key=tuple(float(v) for v in x)+(bool(fine),)
  if key in cache:return cache[key]
  if count>=max_calls-reserve or time.monotonic()-began>max_wall_s-60:raise RuntimeError('Declared model/wall budget exhausted')
  count+=1;number=count;record=dict(call=number,parameters=[float(v) for v in x],fine=fine,status='started');write(out/f'candidate_{number:03d}.json',record)
  try:
   updates,inventory=derive_updates(base,*x);p=base.copy();p.update(updates);p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15});state,state_meta=homogeneous_state(p,inventory,target);p.update(state)
   steps=[pybamm.step.rest(120,skip_ok=False),pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),pybamm.step.rest(7230,skip_ok=False),pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False)]
   resolution=oldconfig['fine'] if fine else oldconfig['fit'];sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=oldconfig['options']),parameter_values=p,experiment=pybamm.Experiment(steps,period=resolution['output_period_s'],temperature=298.15),var_pts=resolution['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=resolution['rtol'],atol=resolution['atol']))
   sol=sim.solve(calc_esoh=False);pieces=[c.steps[-1] for c in sol.cycles]
   raw=[dict(t=np.asarray(c['Time [s]'].entries),v=np.asarray(c['Voltage [V]'].entries),i=np.asarray(c['Current [A]'].entries),q=np.asarray(c['Discharge capacity [A.h]'].entries)) for c in pieces];np.savez_compressed(out/f'candidate_{number:03d}.npz',**{f'phase{i}_{k}':v for i,row in enumerate(raw) for k,v in row.items()})
   if [str(c.termination) for c in pieces]!=oldconfig['expected_terminations']:raise ValueError('Native RPT event mismatch')
   if abs(raw[0]['v'][-1]-target)>.001:raise ValueError('Homogeneous initialization misses native rest endpoint by over1mV')
   cc,cv,dis=raw[1],raw[2],raw[4];qc=-(cc['q']-cc['q'][0]);qd=dis['q']-dis['q'][0];tc=cc['t'][-1]-cc['t'][0];tv=cv['t'][-1]-cv['t'][0]
   vcs=interpolate_required(qc,cc['v'],cc_grid);ics=interpolate_required(cv['t']-cv['t'][0],cv['i'],cv_grid);vds=interpolate_required(qd,dis['v'],q_grid)
   residual=np.r_[(vcs-cc_v)/(.02*np.sqrt(len(cc_grid))),(tc-cc_time)/300.,(qc[-1]-native_q[-1])/.02,(ics-cv_i)/(.05*np.sqrt(len(cv_grid))),(tv-cv_duration)/300.,(vds-q_v)/(.02*np.sqrt(len(q_grid))),(qd[-1]-qobs[-1])/.02]
   phases=[compare_phase(a['t'],a['v'],b['t'],b['v']) for a,b in zip(raw,native)]
   cvmask=cv_time<=tv;di=np.interp(cv_time[cvmask],cv['t']-cv['t'][0],cv['i'])-native[2]['i'][cvmask]
   record.update(status='evaluated',objective_sum_squares=float(residual@residual),updates=updates,initial_state_updates=state,initial_state=state_meta,inventory=inventory,CC_duration_s=float(tc),CV_duration_s=float(tv),CC_transferred_Ah=float(qc[-1]),C10_capacity_Ah=float(qd[-1]),C10_capacity_bias_percent=float(100*(qd[-1]/qobs[-1]-1)),CV_current_RMSE_A=float(np.sqrt(np.mean(di*di))),phases=phases,fixed_interval_coverage_complete=True)
   if not fine and (best is None or record['objective_sum_squares']<best['objective_sum_squares']):best=record.copy()
  except Exception as error:
   record.update(status='failed',error=type(error).__name__+': '+str(error));residual=np.full(fixed_len,1000+float(np.linalg.norm(x)))
  record['wall_s']=time.monotonic()-began;write(out/f'candidate_{number:03d}.json',record);journal.append(record);write(out/'journal.json',journal);cache[key]=residual,record
  return residual,record
 def fun(x):return evaluate(x)[0]
 lastJ=None
 def jac(x):
  nonlocal lastJ
  f,record=evaluate(x)
  if record['status']!='evaluated':raise ValueError('Cannot calculate physical sensitivity from failed candidate')
  cols=[]
  for i,h in enumerate([.001,.001,.005]):
   step=h if x[i]+h<=BOUNDS[1][i] else -h;z=np.array(x).copy();z[i]+=step;g,r=evaluate(z)
   if r['status']!='evaluated':
    z=np.array(x).copy();step=-step;z[i]+=step
    if not BOUNDS[0][i]<=z[i]<=BOUNDS[1][i]:raise ValueError('No admissible finite difference direction')
    g,r=evaluate(z)
   if r['status']!='evaluated':raise ValueError('Both sensitivity directions failed; no penalty Jacobian')
   cols.append((g-f)/step)
  lastJ=np.column_stack(cols);return lastJ
 baseline_x=declaration['starting_points'][0]
 _,baseline_fine=evaluate(baseline_x,fine=True)
 write(out/'retained_v3_fine_baseline.json',baseline_fine)
 print('제어시간 내부 anchor를 사용한 retained v3 fine RPT 평가 완료',flush=True)
 searches=[]
 for start in declaration['starting_points']:
  try:
   opt=least_squares(fun,start,jac=jac,bounds=BOUNDS,x_scale=[.05,.05,.1],max_nfev=max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4);searches.append(dict(start=start,x=opt.x.tolist(),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev)))
  except (RuntimeError,ValueError) as e:searches.append(dict(start=start,status='stopped',reason=str(e)));break
 write(out/'searches.json',searches)
 if best is None:raise RuntimeError('No valid joint candidate')
 frozen=dict(scope='C-only joint initial candidate, not uniquely identified material or a lifetime model',chosen_C=best,source_C=source,transport_changed=False,parameter_bounds=BOUNDS,D_opened_before_freeze=False,source_sha256=sha(__file__))
 write(out/'parameters_frozen.json',frozen);freeze=sha(out/'parameters_frozen.json');_,fine=evaluate(best['parameters'],fine=True,reserve=0)
 if fine['status']!='evaluated':raise RuntimeError('Fine candidate failed fixed coverage/endpoint validation')
 data=np.load(out/f"candidate_{fine['call']:03d}.npz");comparisons={}
 for cell in ('C','D'):
  observed,src=load_native(root,cell);phases=[]
  for i,b in enumerate(observed):phases.append(compare_phase(data[f'phase{i}_t'],data[f'phase{i}_v'],b['t'],b['v']))
  dq,dv=read_curve(root/f'work/external_m50_benchmark/kirkaldy_expt2_cell{cell}_RPT0_C10.csv');mq=data['phase4_q']-data['phase4_q'][0];mask=dq<=mq[-1];e=np.interp(dq[mask],mq,data['phase4_v'])-dv[mask]
  comparisons[cell]=dict(phases=phases,source=src,C10_capacity_model_Ah=float(mq[-1]),C10_capacity_measured_Ah=float(dq[-1]),C10_capacity_bias_percent=float(100*(mq[-1]/dq[-1]-1)),C10_voltage_RMSE_mV=float(np.sqrt(np.mean(e*e))*1000),C10_voltage_covered_fraction=float(mask.mean()))
 Jmeta=None
 if lastJ is not None:
  singular=np.linalg.svd(lastJ*np.array([.05,.05,.1]),compute_uv=False);Jmeta=dict(singular_values=singular.tolist(),rank=int(np.linalg.matrix_rank(lastJ)),condition_number=float(singular[0]/singular[-1]),structural_identifiability=False,note='Last valid optimizer local Jacobian; not necessarily final candidate profile or confidence interval')
 result=dict(status='joint_initial_candidate_evaluated',rest_anchor=anchor_meta,retained_v3_fine_baseline=baseline_fine,model_calls=count,wall_s=time.monotonic()-began,coarse_chosen_C=best,fine_candidate=fine,comparisons=comparisons,searches=searches,local_sensitivity=Jmeta,parameter_at_bounds=[min(abs(x-lo),abs(hi-x))<.001 for x,lo,hi in zip(best['parameters'],*BOUNDS)],parameters_frozen_unchanged=sha(out/'parameters_frozen.json')==freeze,source_unchanged=sha(__file__)==declaration['source_sha256'],D_used_for_fit=False,physical_validation_complete=False,SOH_validation_complete=False,AWS_started=False)
 write(out/'report.json',result);print(json.dumps(dict(status=result['status'],calls=count,wall_s=result['wall_s'],parameters=best['parameters'],C=comparisons['C'],D=comparisons['D']),ensure_ascii=False,indent=2));return result

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--max-model-calls',type=int,default=140);a.add_argument('--max-wall-s',type=int,default=900);a.add_argument('--max-nfev',type=int,default=30);args=a.parse_args();run(args.root,args.output,args.max_model_calls,args.max_wall_s,args.max_nfev)
