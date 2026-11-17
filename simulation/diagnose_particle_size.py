"""Three fresh PSD structure counterfactuals; no parameter estimation."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
import numpy as np,pybamm
ROOT=None;OUT=None
from fit_spme_bidirectional_bol import protocol_steps
from joint_spme_bol_fit import sha,write,interpolate_required
CASES=[('single',None),('PSD_default_NE',.3),('PSD_broader_NE',.6)]
OPTIONS={'thermal':'isothermal','surface form':'algebraic','SEI':'none','lithium plating':'none','loss of active material':'none','SEI porosity change':'false','lithium plating porosity change':'false'}
F=96485.33212331001

EXPECTED_EVENTS=['final time','event: Voltage > 4.2 [V] [experiment]',
 'event: abs(Current [A]) < 0.05 [A] [experiment]','final time',
 'event: Voltage < 2.5 [V] [experiment]','final time','final time',
 'event: Voltage > 4.2 [V] [experiment]']

def validate_events(events):
 if list(events)!=EXPECTED_EVENTS:
  raise ValueError('Native event mismatch: '+repr(list(events)))

def observed():
 p=ROOT/'work/validation_proposal_20261103/latest_spme_review_20261103/bidirectional_BOL_fit_v1/native_observations.npz';return np.load(p)
def capacity(t,i):return np.r_[0.,np.cumsum(.5*(i[1:]+i[:-1])*np.diff(t)/3600)]
def curve(mx,mv,ox,ov):
 mask=(ox>=mx[0])&(ox<=mx[-1]);e=np.interp(ox[mask],mx,mv)-ov[mask]
 return dict(RMSE_mV=float(np.sqrt(np.mean(e*e))*1000),max_abs_mV=float(max(abs(e))*1000),coverage_fraction=float(mask.mean()),covered_samples=int(mask.sum()),total_samples=len(ox))
def at_initial(a,n):
 a=np.asarray(a,dtype=float);return a.reshape(n,-1)[:,0] if a.ndim>1 else a.reshape(n)
def raw_quad(function,lo,hi,count):
 if not np.isfinite([lo,hi]).all() or not 0<=lo<hi:
  raise ValueError('Finite nonnegative ordered radius bounds required')
 if isinstance(count,bool) or not isinstance(count,int) or count<2:
  raise ValueError('At least two size quadrature cells required')
 edges=np.linspace(lo,hi,count+1);nodes=.5*(edges[:-1]+edges[1:]);widths=np.diff(edges)
 # The installed helper returns a PyBaMM symbolic function, not a NumPy ufunc.
 # Midpoint nodes are strictly positive even when the declared lower bound is 0.
 v=function(pybamm.Vector(nodes))
 values=np.asarray(v.evaluate() if hasattr(v,'evaluate') else v,dtype=float).reshape(-1)
 if values.shape!=nodes.shape or not np.all(np.isfinite(values)) or np.any(values<0):
  raise ValueError('Finite nonnegative paired density samples required')
 mass=float(values@widths)
 if not mass>0:raise ValueError('Positive size density mass required')
 normalized=values/mass
 return dict(nodes=count,raw_area_density_mass=mass,area_mean_m=float((nodes*normalized)@widths),area_sd_m=float(np.sqrt(((nodes-float((nodes*normalized)@widths))**2*normalized)@widths)),all_values_finite=bool(np.all(np.isfinite(values))))
def worker(index):
 label,sd=CASES[index];out=OUT/label;out.mkdir(exist_ok=True);began=time.monotonic();decl=json.loads((OUT/'declared_experiment.json').read_text());init=json.loads(Path(decl['initialization_file']).read_text());material=json.loads(Path(decl['material_file']).read_text());p=pybamm.ParameterValues('OKane2022');p.update(material['parameter_updates']);p.update(init['initial_state_updates']);p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15});options=dict(OPTIONS);options['particle size']='single' if sd is None else ('distribution','single');originalR=float(p['Negative particle radius [m]']);originalRp=float(p['Positive particle radius [m]'])
 if sd is not None:p=pybamm.get_size_distribution_parameters(p,sd_n=sd,sd_p=.3)
 expected_mol=sum(float(p[f'Initial concentration in {side} electrode [mol.m-3]'])*float(p[f'{side.capitalize()} electrode active material volume fraction'])*float(p[f'{side.capitalize()} electrode thickness [m]'])*float(p['Electrode width [m]'])*float(p['Electrode height [m]']) for side in ['negative','positive']);record=dict(label=label,status='started',options=options,sd_n=sd,nominal_negative_mean_m=originalR,positive_radius_m=originalRp,expected_initial_particle_Li_mol=expected_mol,expected_initial_particle_Li_Ah=expected_mol*F/3600,parameter_fit_performed=False,initialization='direct reviewed retained cn/cp; fresh states, no checkpoint mapping', model_solve_calls=1, physical_validation_complete=False, ground_truth_export_admitted=False);write(out/'report.json',record)
 model=pybamm.lithium_ion.SPMe(options=options);varpts={'x_n':10,'x_s':10,'x_p':10,'r_n':20,'r_p':20}
 if sd is not None:varpts['R_n']=16
 sim=pybamm.Simulation(model,parameter_values=p,experiment=pybamm.Experiment(protocol_steps(pybamm),period=10,temperature=298.15),var_pts=varpts,solver=pybamm.IDAKLUSolver(rtol=1e-7,atol=1e-9));solution=sim.solve(calc_esoh=False);pieces=[c.steps[-1] for c in solution.cycles];raw=[{k:np.asarray(c[v].entries) for k,v in [('t','Time [s]'),('v','Voltage [V]'),('i','Current [A]'),('q','Discharge capacity [A.h]')]} for c in pieces]
 record['phase_count']=len(pieces)
 for k,data in enumerate(raw):np.savez_compressed(out/f'phase_{k}.npz',**data)
 if len(pieces)!=8:raise ValueError('Did not complete8nativephases')
 expected=EXPECTED_EVENTS
 record['events']=[str(c.termination) for c in pieces];record['expected_events']=EXPECTED_EVENTS
 write(out/'report.json',record)
 validate_events(record['events'])
 first=pieces[0];initial_li=float(np.asarray(first['Total lithium in particles [mol]'].entries).reshape(-1)[0]);last_li=float(np.asarray(pieces[-1]['Total lithium in particles [mol]'].entries).reshape(-1)[-1]);record.update(initial_particle_Li_model_mol=initial_li,initial_Li_budget_relative_error=(initial_li-expected_mol)/expected_mol,total_particle_Li_end_mol=last_li,total_particle_Li_change_mol=last_li-initial_li)
 write(out/'report.json',record)
 if abs(initial_li-expected_mol)>1e-8:raise ValueError('Fresh initial lithium budget mismatch')
 if sd is not None:
  R=at_initial(first['Negative particle sizes [m]'].entries,16);fa=at_initial(first['X-averaged negative area-weighted particle-size distribution [m-1]'].entries,16);fv=at_initial(first['X-averaged negative volume-weighted particle-size distribution [m-1]'].entries,16);fn=at_initial(first['X-averaged negative number-based particle-size distribution [m-1]'].entries,16);mesh=sim.mesh['negative particle size'];widths=np.diff(mesh.edges);area_mean=float((R*fa)@widths);bulk=at_initial(first['Average negative particle concentration distribution [mol.m-3]'].entries,16);avg_c=float((bulk*fv)@widths);geom=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Negative electrode thickness [m]'])*float(p['Negative electrode active material volume fraction']);computed_negative_li=geom*avg_c
  stats=dict(radius_nodes=R.tolist(),cell_widths=widths.tolist(),area_density_normalization=float(fa@widths),volume_density_normalization=float(fv@widths),number_density_normalization=float(fn@widths),area_mean_m=area_mean,area_mean_relative_to_nominal=area_mean/originalR,volume_mean_m=float((R*fv)@widths),number_mean_m=float((R*fn)@widths),volume_weight_relation_max_abs=float(np.max(abs(fv-R*fa/area_mean))),initial_volume_weighted_c_mol_m3=avg_c,initial_negative_Li_integral_mol=computed_negative_li,initial_negative_Li_parameter_mol=geom*float(p['Initial concentration in negative electrode [mol.m-3]']),radius_nodes_positive=bool(np.all(R>0)),quad16=raw_quad(p['Negative area-weighted particle-size distribution [m-1]'],float(p['Negative minimum particle radius [m]']),float(p['Negative maximum particle radius [m]']),16),quad32=raw_quad(p['Negative area-weighted particle-size distribution [m-1]'],float(p['Negative minimum particle radius [m]']),float(p['Negative maximum particle radius [m]']),32))
  np.savez_compressed(out/'initial_PSD_weights.npz',R_m=R,width_m=widths,f_area_per_m=fa,f_volume_per_m=fv,f_number_per_m=fn,initial_ravg_c_distribution_mol_m3=bulk);record['distribution_checks']=stats;write(out/'report.json',record)
  if any(abs(stats[k]-1)>1e-6 for k in ['area_density_normalization','volume_density_normalization','number_density_normalization']):raise ValueError('Actual model distribution normalization failure')
  if abs(computed_negative_li-stats['initial_negative_Li_parameter_mol'])>1e-8:raise ValueError('Size-integrated initial lithium mismatch')
 n=observed();dq=capacity(n['Ns5_t'],n['Ns5_i']);cq=capacity(n['Ns8_t'],-n['Ns8_i']);md=raw[4]['q']-raw[4]['q'][0];mc=-(raw[7]['q']-raw[7]['q'][0]);rt=raw[5]['t']-raw[5]['t'][0];restmedian=float(np.median(raw[5]['v'][rt>=21000]));native_rt=n['Ns6_t']-n['Ns5_t'][-1];idx=[int(np.searchsorted(native_rt,t)) for t in [60,600,3600,10800]];times=native_rt[idx];shape_v=n['Ns6_v'][idx];shape_model=interpolate_required(rt,raw[5]['v'],times);restobs=float(np.median(n['Ns6_v'][(n['Ns6_t']-n['Ns6_t'][0])>=21000]));record.update(status='fresh_structure_diagnostic_complete',events=[str(c.termination) for c in pieces],C10_capacity_Ah=float(md[-1]),C10_observed_Ah=float(dq[-1]),C10_capacity_bias_percent=float(100*(md[-1]/dq[-1]-1)),C10_voltage=curve(md,raw[4]['v'],dq,n['Ns5_v']),recharge_capacity_Ah=float(mc[-1]),recharge_observed_Ah=float(cq[-1]),recharge_capacity_bias_percent=float(100*(mc[-1]/cq[-1]-1)),recharge_voltage=curve(mc,raw[7]['v'],cq,n['Ns8_v']),rest_median_V=restmedian,rest_observed_median_V=restobs,rest_median_residual_mV=1000*(restmedian-restobs),shape_times_s=times.tolist(),shape_raw_indices=idx,shape_raw_V=shape_v.tolist(),shape_model_V=shape_model.tolist(),shape_residual_mV=(1000*(shape_model-shape_v)).tolist(),shape_RMSE_mV=float(np.sqrt(np.mean((shape_model-shape_v)**2))*1000),wall_s=time.monotonic()-began,physical_validation_complete=False,C2_capacity_preservation_tested=False)
 write(out/'report.json',record);return record

def main():
 OUT.mkdir(exist_ok=False);work=ROOT/'work/validation_proposal_20261103';latest=work/'latest_spme_review_20261103';initpath=latest/'joint_initial_fit_v3/parameters_effective_initialization.json';materialpath=work/'physical_BOL_fit_v3/parameters_frozen.json';helper=Path(pybamm.__file__).parent/'parameters/size_distribution_parameters.py';native=latest/'bidirectional_BOL_fit_v1/native_observations.npz';rawsource=Path(json.loads(initpath.read_text())['source_C']['path']);sources=[Path(__file__),Path(__file__).with_name('joint_spme_bol_fit.py'),initpath,materialpath,helper,native,rawsource,ROOT/'repo/simulation/fit_spme_bidirectional_bol.py'];guards={str(p.resolve()):sha(p) for p in sources};declaration=dict(scope='three fresh-only structure counterfactuals; no fitting, no ageing acceptance',case_definitions=CASES,common_options=OPTIONS,PSD_option=['distribution','single'],size_domain='negative only, positive single',initialization_file=str(initpath),material_file=str(materialpath),source_sha256=guards,pybamm_version=pybamm.__version__,input_mean_radii_unchanged=True,default_width_is_validated_M50T=False,broader_width_is_engineering_perturbation=True,mesh=dict(x=10,r=20,R_n=16),output_period_s=10,rtol=1e-7,atol=1e-9,exact8phase_source=str(ROOT/'repo/simulation/fit_spme_bidirectional_bol.py'),max_solves=3,max_wall_s=360,SEI_plating_LAM_active=False,source_density_quadrature='positive actual size-cell centers and finite-volume widths; no R0 endpoint evaluation',stock_or_Ah_axes_compensated=False,C2_GITT_D_used=False,AWS_started=False);write(OUT/'declared_experiment.json',declaration);snap=OUT/'source_snapshots';snap.mkdir()
 for number,path in enumerate(sources):
  if path.suffix in ('.py','.json'):(snap/f'{number:02d}_{path.name}').write_bytes(path.read_bytes())
 began=time.monotonic();records=[]
 for index,(label,sd) in enumerate(CASES):
  folder=OUT/label;folder.mkdir();remaining=360-(time.monotonic()-began)
  if remaining<=0:records.append(dict(label=label,status='not_started_budget'));break
  try:
   with (folder/'worker.log').open('w',encoding='utf-8') as log:completed=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--root',str(ROOT),'--output',str(OUT),'--case',str(index)],stdout=log,stderr=subprocess.STDOUT,timeout=remaining,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   records.append(dict(label=label,exit_code=completed.returncode,report=str(folder/'report.json')))
  except subprocess.TimeoutExpired:records.append(dict(label=label,status='time_budget_exhausted'));break
  print('fresh PSD case completed:',label,flush=True)
 result=dict(status='fresh_structure_diagnostics_complete' if len(records)==3 and all(r.get('exit_code')==0 for r in records) else 'failed_or_incomplete',cases=records,wall_s=time.monotonic()-began,all_guarded_sources_unchanged=all(sha(Path(p))==v for p,v in guards.items()),solves_started=len(records),physical_validation_complete=False,parameter_fitting_performed=False,AWS_started=False);write(OUT/'run_receipt.json',result);print(json.dumps(result,ensure_ascii=False));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--case',type=int,choices=range(3));a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args()
 ROOT=v.root.resolve();OUT=v.output.resolve()
 if v.case is None:
  result=main();raise SystemExit(0 if result['status']=='fresh_structure_diagnostics_complete' and result['all_guarded_sources_unchanged'] else 2)
 else:
  try:worker(v.case)
  except Exception as e:
   import traceback
   traceback.print_exc();folder=OUT/CASES[v.case][0];failure=dict(status='failed',error_type=type(e).__name__,error=str(e),physical_validation_complete=False)
   write(folder/'failure.json',failure)
   report_path=folder/'report.json';record=json.loads(report_path.read_text()) if report_path.exists() else {}
   record.update(failure);write(report_path,record);raise
