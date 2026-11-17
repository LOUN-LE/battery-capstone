"""C-only continuous F045→F046 C/2 diagnostic with independent2.5V cutoff.

No fresh-cell restart at a file boundary. The .01s interfile rest is an
explicit unobserved-exposure assumption; this driver cannot declare physical
acceptance. Observations never determine model event durations.
"""
from __future__ import annotations
import argparse,json,os,time,sys,math,inspect
from pathlib import Path

def sha(p):
 import hashlib
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def frozen_material_scales(chosen):
 """Fourth fit coordinate is log10(Dn multiplier), never a linear factor."""
 scales=list(chosen['parameters'])
 if len(scales) not in (3,4) or any(not math.isfinite(float(x)) for x in scales):raise ValueError('Three material coordinates plus optional log10Dn required')
 factor=chosen.get('negative_diffusivity_multiplier')
 if len(scales)==4:
  if factor is None or not math.isfinite(float(factor)) or float(factor)<=0:raise ValueError('Four-coordinate freeze requires positive physical Dn multiplier metadata')
  if not math.isclose(float(factor),10.**float(scales[3]),rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Frozen physical Dn multiplier disagrees with log10 coordinate')
 if factor is not None and (not math.isfinite(float(factor)) or float(factor)<=0):raise ValueError('Finite positive frozen Dn multiplier required')
 return scales[:3],None if factor is None else float(factor)

def resolve_diffusivity(*,multiplier=None,source=None,frozen_multiplier=None):
 if multiplier is not None and source is not None:raise ValueError('Select one Dn override')
 if source not in (None,'ORegan2022'):raise ValueError('Unreviewed Dn source')
 if multiplier is not None and (isinstance(multiplier,bool) or not math.isfinite(float(multiplier)) or float(multiplier)<=0):raise ValueError('Finite positive engineering Dn factor required')
 if frozen_multiplier is not None and (multiplier is not None or source is not None):raise ValueError('Explicit Dn override conflicts with frozen Dn metadata; use a separate retained-material diagnostic')
 if frozen_multiplier is not None:return dict(source='OKane2022',multiplier=float(frozen_multiplier),origin='frozen_C_multiplier')
 return dict(source=source or 'OKane2022',multiplier=1. if multiplier is None else float(multiplier),origin='explicit_published_function_hybrid' if source else ('explicit_engineering_multiplier' if multiplier is not None else 'unchanged_baseline'))

def apply_diffusivity(p,spec):
 """Replace only Dn; ORegan does not select a complete material set."""
 import numpy as np,pybamm
 key='Negative particle diffusivity [m2.s-1]'
 original=p[key]
 function=pybamm.ParameterValues('ORegan2022')[key] if spec['source']=='ORegan2022' else original
 function_path=Path(inspect.getsourcefile(function)).resolve()
 factor=spec['multiplier']
 if factor!=1.:
  def scaled(sto,T):return factor*function(sto,T)
  selected=scaled
 else:selected=function
 p.update({key:selected})
 points=[.02,.0355,.05,.1,.3,.6,.9]
 values=[]
 for sto in points:
  value=selected(sto,298.15)
  values.append(float(np.asarray(value.evaluate() if hasattr(value,'evaluate') else value).item()))
 return dict(**spec,key=key,unit='m2/s',function_name=function.__name__,function_source_path=str(function_path),function_source_sha256=sha(function_path),reference_temperature_K=298.15,sample_stoichiometries=points,sample_Dn_m2_s=values,whole_parameter_set_replaced=False)

def programmed_current_A(row):
 factor={0:1.,1:.001}.get(row['ctrl1_val_unit'])
 if factor is None:raise ValueError('Unreviewed native current unit')
 return float(row['ctrl1_val'])*factor
def specifications():
 return [
 dict(file='F045',Ns=0,operation='rest',duration=120.),dict(file='F045',Ns=1,operation='current',value=-1.5,duration=20000.,termination='4.2 V'),dict(file='F045',Ns=2,operation='voltage',value=4.2,duration=20000.,termination='0.05 A'),dict(file='F045',Ns=3,operation='rest',duration=7200.),dict(file='F045',Ns=4,operation='rest',duration=30.),dict(file='F045',Ns=5,operation='current',value=.5,duration=50000.,termination='2.5 V',is_capacity=True),dict(file='F045',Ns=6,operation='rest',duration=21600.),dict(file='F045',Ns=7,operation='rest',duration=30.),dict(file='F045',Ns=8,operation='current',value=-.5,duration=50000.,termination='4.2 V'),dict(file='F045',Ns=9,operation='rest',duration=600.),dict(file='gap',Ns=None,operation='rest',duration=.01,assumption='Unobserved.01s interval modeled as zero-current rest for diagnosis only'),dict(file='F046',Ns=0,operation='rest',duration=120.),dict(file='F046',Ns=1,operation='current',value=-1.5,duration=20000.,termination='4.2 V',allow_initial_event=True),dict(file='F046',Ns=2,operation='voltage',value=4.2,duration=20000.,termination='0.05 A'),dict(file='F046',Ns=3,operation='rest',duration=7200.),dict(file='F046',Ns=4,operation='rest',duration=30.),dict(file='F046',Ns=5,operation='current',value=2.5,duration=20000.,termination='2.5 V',is_capacity=True)]
def score(mx,my,ox,oy):
 import numpy as np
 mask=(ox>=mx[0])&(ox<=mx[-1]);e=np.interp(ox[mask],mx,my)-oy[mask]
 if not len(e):raise ValueError('No observed overlap')
 return dict(RMSE=float(np.sqrt(np.mean(e*e))),max_abs=float(max(abs(e))),coverage_fraction=float(mask.mean()),covered_samples=int(mask.sum()),total_samples=int(len(mask)))
def validate_native_arrays(d):
 import numpy as np
 required=['time_s','I_mA','V','mode','Ns','control_mixed_V_or_mA']
 if any(k not in d for k in required):raise ValueError('Required native NPZ current/voltage/control contract missing')
 n=len(d['time_s'])
 if n<3 or any(len(d[k])!=n for k in required):raise ValueError('Native channel length mismatch')
 if not np.all(np.diff(d['time_s'])>0) or any(not np.all(np.isfinite(d[k])) for k in required):raise ValueError('Nonfinite or nonincreasing native observations')
 if list(dict.fromkeys(d['Ns'].astype(int).tolist()))!=list(range(10)):raise ValueError('Expected F046 Ns0..9')
def run(root,output,*,case_selection='both',negative_diffusivity_multiplier=None,negative_diffusivity_source=None,joint_parameters=None):
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
 os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
 import numpy as np,pybamm
 from lithium_inventory_audit import audit_plating_capacity
 from joint_spme_bol_fit import derive_updates,homogeneous_state,select_rest_anchor
 from initialize_spme_from_rest import load_native,compare_phase
 root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();base=root/'work/validation_proposal_20261103';latest=base/'latest_spme_review_20261103';folder=latest/'dynamic_C0_data_v1/F046_long_02';receipt=json.loads((folder/'receipt.json').read_text());native=np.load(folder/'raw.npz');validate_native_arrays(native);f46=Path(receipt['source_identity']['MPT_local_path']);f45=Path(receipt['previous_file_context']['MPT_local_path']);sys.path.insert(0,str(root/'work/preprocess'));import common
 cols,_,_=common.read_mpt(f45);runs,_=common.split_runs(cols)
 if [int(cols['ns'][a]) for a,b in runs]!=list(range(10)):raise ValueError('F045 Ns sequence mismatch')
 if sha(f45)!=receipt['previous_file_context']['MPT_sha256'] or sha(f46)!=receipt['source_identity']['MPT_sha256']:raise ValueError('Raw MPT source changed')
 f46cols,_,_=common.read_mpt(f46)
 for k,nk,tol in [('t','time_s',1e-8),('I','I_mA',1e-4),('V','V',1e-7),('mode','mode',0),('ns','Ns',0),('ctrl_mixed','control_mixed_V_or_mA',1e-4)]:
  if np.max(np.abs(f46cols[k]-native[nk]))>tol:raise ValueError('Native NPZ/MPT channel mismatch: '+k)
 gap=float(native['time_s'][0]-cols['t'][-1])
 if abs(gap-.01)>1e-7:raise ValueError('Expected explicit.01s interfile gap; no silent exposure substitution')
 decoded_path=base/'metadata_control_probe_v2/decoded_programs_raw.json';allrows=json.loads(decoded_path.read_text());programs={}
 for fid,ident in [('F045',receipt['previous_file_context']),('F046',receipt['source_identity'])]:
  source=Path(ident['MPR_local_path'])
  if sha(source)!=ident['MPR_sha256']:raise ValueError('MPR hash changed')
  row=next(x for x in allrows if x['whole_file_sha256']==ident['MPR_sha256']);programs[fid]=row
  if len(row['raw_program_rows'])!=10:raise ValueError('Expected10 native MB program rows')
 # MPR type/unit semantics are checked against recorded modes and setpoints.
 for fid,row in programs.items():
  prog=row['raw_program_rows']
  for ns,expected in [(0,120),(3,7200),(4,30),(6,21600),(7,30),(9,600 if fid=='F045' else 1800)]:
   q=prog[ns];unit={1:3600.,2:60.,3:1.}.get(q['lim1_value_unit'])
   if q['ctrl_type']!=4 or q['lim1_type']!=0 or unit is None or abs(q['lim1_value']*unit-expected)>1e-5:raise ValueError('Rest program mismatch')
  for ns,amp,volt,discharge in [(1,1.5,4.2,0),(5,.5 if fid=='F045' else 2.5,2.5,1),(8,.5 if fid=='F045' else 2.5,4.2,0)]:
   q=prog[ns]
   if q['ctrl_type']!=0 or abs(programmed_current_A(q)-amp)>2e-6 or abs(q['lim1_value']-volt)>2e-6 or q['charge/discharge']!=discharge:raise ValueError('CC program mismatch')
  q=prog[2]
  if q['ctrl_type']!=1 or abs(q['ctrl1_val']-4.2)>2e-6 or q['lim1_type']!=12 or q['lim1_value']!=50 or q['lim1_value_unit']!=1:raise ValueError('CV50mA program mismatch')
 oldfile=base/'physical_BOL_fit_v3/parameters_frozen.json';newfile=Path(joint_parameters).resolve() if joint_parameters else latest/'joint_initial_fit_v3/parameters_frozen.json';old=json.loads(oldfile.read_text());configpath=base/'physical_BOL_fit_v3/declared_experiment.json';config=json.loads(configpath.read_text());selected_cases=[]
 if case_selection not in ('both','retained_v3','joint_C_frozen'):raise ValueError('Unknown case selection')
 if case_selection in ('both','retained_v3'):selected_cases.append(('retained_v3',[old['s_Q'],old['s_Q'],old['delta_Q_Li_Ah']],None))
 if case_selection in ('both','joint_C_frozen'):
  new=json.loads(newfile.read_text());scales,frozen_factor=frozen_material_scales(new['chosen_C']);selected_cases.append(('joint_C_frozen',scales,frozen_factor))
 # Resolve all requested cases before starting any expensive solve.
 selected_cases=[(label,scales,resolve_diffusivity(multiplier=negative_diffusivity_multiplier,source=negative_diffusivity_source,frozen_multiplier=factor)) for label,scales,factor in selected_cases]
 paths=[oldfile,f45,f46,folder/'raw.npz',folder/'receipt.json',decoded_path,configpath,Path(__file__),Path(__file__).with_name('joint_spme_bol_fit.py'),Path(__file__).with_name('initialize_spme_from_rest.py'),Path(__file__).with_name('lithium_inventory_audit.py')]
 if case_selection in ('both','joint_C_frozen'):paths.append(newfile)
 source_hashes={str(p):sha(p) for p in paths};c,csource=load_native(root,'C');target,anchor=select_rest_anchor(c[0]);specs=specifications();contract=dict(scope='C-only continuous conditioning diagnostic; gap is assumption, not physical validation',specifications=specs,file_gap_s=gap,file_gap_assumption='rest at25C, unobserved',programs=programs,source_hashes=source_hashes,anchor_policy=anchor,anchor_policy_id='declared_rest_interior_last3_v1',axis_policy='absolute model time/Ah with phase-relative scoring; no time/Ah stretching',requested_case_selection=case_selection,negative_diffusivity_cases={label:spec for label,_,spec in selected_cases},D_used=False);write(out/'protocol_contract.json',contract);results={}
 snapshots=out/'source_snapshots';snapshots.mkdir()
 for index,path in enumerate(paths):
  if path.suffix in ('.py','.json'):(snapshots/f'{index:02d}_{path.name}').write_bytes(path.read_bytes())
 for label,scales,diffusivity in selected_cases:
  case=out/label;case.mkdir();p=pybamm.ParameterValues('OKane2022');updates,inv=derive_updates(p,*scales);p.update(updates);state,meta=homogeneous_state(p,inv,target);p.update(state);p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15});Dn=apply_diffusivity(p,diffusivity);function_source=Path(Dn['function_source_path']);source_hashes[str(function_source)]=Dn['function_source_sha256'];(snapshots/f'{label}_{function_source.name}').write_bytes(function_source.read_bytes());write(case/'parameters_used.json',dict(material_updates=updates,initial_concentration_updates=state,inventory=inv,negative_diffusivity=Dn,temperature_K=298.15,initialization_SOC_remap=False));steps=[]
  for s in specs:
   make=getattr(pybamm.step,s['operation']);kw=dict(duration=s['duration'],skip_ok=s.get('allow_initial_event',False))
   if 'termination'in s:kw['termination']=s['termination']
   steps.append(make(**kw) if s['operation']=='rest' else make(s['value'],**kw))
  resolution=config['fine'];sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,experiment=pybamm.Experiment([tuple(steps)],period=1,temperature=298.15),var_pts=resolution['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=resolution['rtol'],atol=resolution['atol']));sol=sim.solve(calc_esoh=False);pieces=sol.cycles[-1].steps;phases=[]
  for i,piece in enumerate(pieces):
   spec=specs[i]
   if isinstance(piece,pybamm.EmptySolution):
    if not spec.get('allow_initial_event') or spec.get('is_capacity'):raise RuntimeError('Unapproved skipped native step')
    phases.append(dict(spec=spec,status='model_initial_event',model_duration_s=0.,observed_single_sample=True,observed_span_s=0.,physical_reproduction_pass=False));continue
   raw={k:np.asarray(piece[v].entries) for k,v in [('t','Time [s]'),('i','Current [A]'),('v','Voltage [V]'),('q','Discharge capacity [A.h]')]};raw['model_terminal_state_y']=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1);internal={};missing=[]
   for channel in ['Negative electrode stoichiometry','Positive electrode stoichiometry','Bulk open-circuit voltage [V]','Loss of lithium inventory [%]','Total lithium lost [mol]','Loss of lithium to negative SEI [mol]','Loss of lithium to negative lithium plating [mol]','Volume-averaged negative lithium plating concentration [mol.m-3]','Volume-averaged negative dead lithium concentration [mol.m-3]','Loss of capacity to negative lithium plating [A.h]']:
    try:internal[channel]=np.asarray(piece[channel].entries,dtype=float)
    except KeyError:missing.append(channel)
   stock_summary={}
   plating_keys=['Volume-averaged negative lithium plating concentration [mol.m-3]','Volume-averaged negative dead lithium concentration [mol.m-3]','Loss of capacity to negative lithium plating [A.h]']
   if all(key in internal for key in plating_keys):
    audit=audit_plating_capacity(internal[plating_keys[0]],internal[plating_keys[1]],internal[plating_keys[2]],electrode_area_m2=p['Electrode width [m]']*p['Electrode height [m]'],electrode_thickness_m=p['Negative electrode thickness [m]'],faraday_C_per_mol=float(pybamm.constants.F.value))
    plated,dead=audit['plated_Ah'],audit['dead_Ah'];stock_error=audit['max_abs_residual_Ah']
    if not audit['capacity_identity_pass']:raise RuntimeError('Plated+dead stock disagrees with PyBaMM plating-capacity variable')
    raw.update(reversible_plated_stock_Ah=plated,dead_lithium_stock_Ah=dead,plated_plus_dead_stock_Ah=plated+dead)
    stock_summary=dict(reversible_plated_first_Ah=float(plated.reshape(-1)[0]),reversible_plated_last_Ah=float(plated.reshape(-1)[-1]),dead_first_Ah=float(dead.reshape(-1)[0]),dead_last_Ah=float(dead.reshape(-1)[-1]),sum_stock_vs_PyBaMM_capacity_max_abs_error_Ah=stock_error,interpretation='partially reversible plated stock and dead stock are separate; the summed PyBaMM capacity-loss label is not wholly irreversible loss',volume_basis='width*height*negative_thickness; no active-material volume fraction')
   np.savez_compressed(case/f'phase_{i:02d}_{spec["file"]}_Ns{spec["Ns"]}.npz',**raw,**internal)
   if 'termination'in spec and str(piece.termination)=='final time':raise RuntimeError('Event-only step reached watchdog')
   phases.append(dict(spec=spec,status='simulated',termination=str(piece.termination),model_duration_s=float(raw['t'][-1]-raw['t'][0]),internal_end={name:float(value.reshape(-1)[-1]) for name,value in internal.items()},missing_internal_channels=missing,plating_stock=stock_summary))
  if len(pieces)!=len(specs):raise RuntimeError('Incomplete conditioning route')
  last=pieces[-1]
  if str(last.termination)!='event: Voltage < 2.5 [V] [experiment]':raise RuntimeError('Independent C/2 2.5V cutoff missing')
  raw=np.load(case/'phase_16_F046_Ns5.npz');mask=native['Ns']==5;ot=native['time_s'][mask]-native['time_s'][mask][0];oi=-native['I_mA'][mask]/1000.;ov=native['V'][mask];oq=np.r_[0.,np.cumsum(.5*(oi[1:]+oi[:-1])*np.diff(ot)/3600)];mt=raw['t']-raw['t'][0];mq=raw['q']-raw['q'][0];q=float(mq[-1]);integral=float(np.trapezoid(raw['i'],raw['t'])/3600)
  if abs(q-integral)>1e-7+1e-5*q:raise RuntimeError('Model C/2 charge integral mismatch')
  result=dict(status='conditional_continuous_C_half_rate_evaluated',scales=scales,initial_state=meta,initial_concentration_updates=state,initial_inventory=inv,negative_diffusivity=Dn,mesh_points=resolution['mesh_points'],rtol=resolution['rtol'],atol=resolution['atol'],output_period_s=1.,temperature_K=298.15,options=config['options'],phases=phases,C_half_Q_model_Ah=q,C_half_Q_observed_Ah=float(oq[-1]),capacity_bias_percent=float(100*(q/oq[-1]-1)),C_half_model_duration_s=float(mt[-1]),C_half_observed_duration_s=float(ot[-1]),C_half_time_difference_s=float(mt[-1]-ot[-1]),voltage_vs_absolute_Ah=score(mq,raw['v'],oq,ov),voltage_vs_elapsed_s=score(mt,raw['v'],ot,ov),event=str(last.termination),initialization_reset_count=1,interfile_fresh_reset=False,file_gap_assumption=True,physical_validation_complete=False,SOH_validation_complete=False,calibration_used_this_curve=False)
  results[label]=result;write(case/'report.json',result);print(label,'C/2 Q=',q,'observed=',float(oq[-1]),flush=True)
 if any(sha(Path(path))!=h for path,h in source_hashes.items()):raise RuntimeError('Frozen source changed during validation')
 report=dict(status='conditional_continuous_C_half_rate_diagnostic_complete',cases=results,source_hashes=source_hashes,source_unchanged=True,code_sha256=sha(__file__),wall_s=time.monotonic()-began,D_used=False,physical_validation_complete=False,AWS_started=False);write(out/'report.json',report);return report
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--case',dest='case_selection',choices=('both','retained_v3','joint_C_frozen'),default='both');a.add_argument('--joint-parameters',type=Path);g=a.add_mutually_exclusive_group();g.add_argument('--negative-diffusivity-multiplier',type=float);g.add_argument('--negative-diffusivity-source',choices=('ORegan2022',));v=a.parse_args();run(v.root,v.output,case_selection=v.case_selection,negative_diffusivity_multiplier=v.negative_diffusivity_multiplier,negative_diffusivity_source=v.negative_diffusivity_source,joint_parameters=v.joint_parameters)
