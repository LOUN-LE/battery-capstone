"""Conditional F045->F046->F047 continuous GITT voltage-shape diagnostic.

MB Q-reference code0 remains unresolved. An explicit route uses either the
observed run span or the paper-defined200mAh at fixed2.5A (288s), with2.5V fallback.
Actual RPT capacity discharges cannot be skipped. No physical/label admission.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time,sys
from datetime import datetime,timedelta
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def validate_gitt_route(ns):
    if list(ns)!=[0,1,2,3]+[4,5,6]*25+[8]:raise ValueError('Expected observed25 repeats of Ns4/5/6; goto Ns7 is not a recorded exposure')

def allow_empty(spec):
    return not spec.get('is_capacity',False) and bool(spec.get('allow_initial_event') or spec.get('conditional_pulse'))

def model_point(local_t,v,query):
    import numpy as np
    t,v=np.asarray(local_t,dtype=float),np.asarray(v,dtype=float)
    if len(t)<2 or t.shape!=v.shape or not np.all(np.diff(t)>0):raise ValueError('Monotone model curve required')
    return float(np.interp(query,t,v)) if t[0]<=query<=t[-1] else None

def pulse_charge_status(actual_Ah,target_Ah,tolerance=1e-7):
    if actual_Ah>target_Ah+tolerance:raise ValueError('Pulse exceeds paper charge target')
    return dict(paper_charge_target_met=abs(actual_Ah-target_Ah)<=tolerance,
                voltage_early_cutoff_before_paper_charge=actual_Ah<target_Ah-tolerance)

def observation_mode(decoded,*,is_capacity,observed_duration_s,pulse_endpoint_policy=None):
    unresolved=any(l['quantity']=='unresolved_native_charge_reference' for l in decoded['limits'])
    if unresolved:
        if is_capacity:raise ValueError('RPT capacity cannot use a conditional pulse endpoint')
        if pulse_endpoint_policy=='paper-200mAh':
            if abs(decoded.get('value',0)-2.5)>1e-12:raise ValueError('Paper route requires constant2.5A discharge')
            return dict(duration=.2*3600/decoded['value'],termination='2.5 V',conditional_pulse=True,
                Q_reference_admitted=False,pulse_duration_from_observation=False,paper_charge_target_Ah=.2,
                paper_duration_equivalent_to_fixed_current_capacity=True)
        if pulse_endpoint_policy!='recorded-duration':raise ValueError('Explicit noncapacity pulse endpoint policy required')
        if observed_duration_s<=0:raise ValueError('A conditional pulse requires positive observed exposure span')
        return dict(duration=float(observed_duration_s),termination='2.5 V',conditional_pulse=True,
            Q_reference_admitted=False,pulse_duration_from_observation=True)
    timed=[l['value'] for l in decoded['limits'] if l['quantity']=='time_s']
    if decoded['operation']=='rest':
        if len(timed)!=1:raise ValueError('One native rest time required')
        return dict(duration=timed[0])
    limits=[l for l in decoded['limits'] if l['quantity'] in ('voltage_V','absolute_current_A')]
    if len(limits)!=1:raise ValueError('One reviewed noncapacity endpoint required')
    l=limits[0]
    # The low-rate0.5A recharge takes about34ks. This is a numerical watchdog,
    # never an observed-time endpoint; all CC/CV steps use the existing50ks cap.
    return dict(duration=50000.,
                termination=f"{l['value']} V" if l['quantity']=='voltage_V' else f"{l['value']} A")

def prepare(root,output,*,pulse_endpoint_policy):
    import numpy as np
    from native_program_contracts import decode_mb
    from dynamic_observation_operator import parse_runs
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=root/'work/validation_proposal_20261103';data=base/'latest_spme_review_20261103/dynamic_C0_data_v1'
    r46=read(data/'F046_long_02/receipt.json');r47=read(data/'F047_long_03/receipt.json')
    identities={'F045':r46['previous_file_context'],'F046':r46['source_identity'],'F047':r47['source_identity']}
    program_path=base/'metadata_control_probe_v2/decoded_programs_raw.json';programs=read(program_path)
    sys.path.insert(0,str(root/'work/preprocess'));import common
    sources={};all_runs={};specs=[];guards={str(program_path):sha(program_path),str(Path(__file__)):sha(__file__)}
    paperpath=base/'paper_RPT_section.txt'
    if pulse_endpoint_policy=='paper-200mAh':
        if not paperpath.exists():raise ValueError('Preserved direct2024 paper RPT evidence required')
        guards[str(paperpath)]=sha(paperpath)
    for fid,identity in identities.items():
        mpt,mpr=Path(identity['MPT_local_path']),Path(identity['MPR_local_path'])
        if sha(mpt)!=identity['MPT_sha256'] or sha(mpr)!=identity['MPR_sha256']:raise ValueError('Native source changed')
        for p in (mpt,mpr):guards[str(p)]=sha(p)
        program=next(p for p in programs if p['whole_file_sha256']==identity['MPR_sha256'])
        cols,meta,_=common.read_mpt(mpt);ranges,_=common.split_runs(cols)
        if fid=='F045':
            if [int(cols['ns'][a]) for a,b in ranges]!=list(range(10)):raise ValueError('Expected F045 Ns0..9')
            arrays=dict(time_s=cols['t'],V=cols['V'],I_mA=cols['I'],Ns=cols['ns'],mode=cols['mode'],
                raw_source_row=np.arange(len(cols['t'])),run_index=np.zeros(len(cols['t']),dtype=int),
                dq_mAh=cols['dq'],surface_temperature_C=cols['T'])
            runs=[]
            for i,(a,b) in enumerate(ranges):
                arrays['run_index'][a:b]=i
                runs.append(dict(file_id=fid,part='long_01',run_index=i,Ns=int(cols['ns'][a]),row_start=a,row_end_exclusive=b,
                    warning_flags=[],role='conditioning_and_actual_RPT'))
        else:
            folder=data/('F046_long_02' if fid=='F046' else 'F047_long_03')
            npz=folder/'raw.npz';guards[str(npz)]=sha(npz);guards[str(folder/'runs.csv')]=sha(folder/'runs.csv')
            with np.load(npz,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files if k!='original_MPR_structured_rows'}
            runs=parse_runs(folder/'runs.csv')
            for key,col,tol in [('time_s','t',1e-8),('V','V',1e-7),('I_mA','I',1e-4),('Ns','ns',0),('mode','mode',0)]:
                if np.max(np.abs(arrays[key]-cols[col]))>tol:raise ValueError('MPR/MPT channel mismatch '+fid+' '+key)
        if fid=='F046' and [r['Ns'] for r in runs]!=list(range(10)):raise ValueError('Expected F046 Ns0..9')
        if fid=='F047':validate_gitt_route([r['Ns'] for r in runs])
        all_runs[fid]=runs
        stamp=identity['clock']['timestamp_naive']
        sources[fid]=dict(identity=identity,program=program,arrays=arrays,runs=runs,timestamp_naive=stamp)
        np.savez_compressed(out/(fid+'_native.npz'),**arrays)
    if len({s['timestamp_naive'] for s in sources.values()})!=1:
        raise ValueError('This raw-global-time implementation requires the audited common native origin')
    gaps=[]
    previous=None
    for fid,source in sources.items():
        arrays,runs=source['arrays'],source['runs']
        if previous is not None:
            prior=sources[previous]
            left=datetime.fromisoformat(prior['timestamp_naive'])+timedelta(seconds=float(prior['arrays']['time_s'][-1]))
            right=datetime.fromisoformat(source['timestamp_naive'])+timedelta(seconds=float(arrays['time_s'][0]))
            # Keep unrounded native-file clocks when their audited origins coincide.
            gap=float(arrays['time_s'][0]-prior['arrays']['time_s'][-1]) if prior['timestamp_naive']==source['timestamp_naive'] else (right-left).total_seconds()
            if gap<=0:raise ValueError('Unreviewed nonpositive file gap')
            spec=dict(file='gap',Ns=None,run_index=None,operation='rest',duration=gap,
                from_file=previous,to_file=fid,unobserved=True,assumption='zero-current25C rest solely for this conditional diagnostic')
            gaps.append(spec);specs.append(spec)
        for run in runs:
            a,b=run['row_start'],run['row_end_exclusive'];ns=run['Ns'];raw=source['program']['raw_program_rows'][ns]
            decoded=decode_mb(raw,preserve_branch_metadata=True,sequence_count=source['program']['sequence_count'],preserve_charge_metadata=True)
            observed_modes=np.unique(arrays['mode'][a:b]).tolist()
            if observed_modes!=[{'current':1,'voltage':2,'rest':3}[decoded['operation']]]:
                raise ValueError('Native control and recorded mode disagree: '+fid+' Ns'+str(ns))
            if decoded['operation']=='current':
                measured=float(np.median(arrays['I_mA'][a:b]))/1000
                if abs(measured+decoded['value'])>.01:raise ValueError('Native current unit/direction differs from observed run')
            duration=float(arrays['time_s'][b-1]-arrays['time_s'][a]);capacity=fid in ('F045','F046') and ns==5
            mode=observation_mode(decoded,is_capacity=capacity,observed_duration_s=duration,pulse_endpoint_policy=pulse_endpoint_policy)
            spec=dict(file=fid,Ns=ns,run_index=run['run_index'],operation=decoded['operation'],native_control_decoded=decoded,
                native_control_row=raw,observed_row_start=a,observed_row_end_exclusive=b,observed_duration_s=duration,
                observed_to_next_first_s=float(arrays['time_s'][b]-arrays['time_s'][a]) if b<len(arrays['time_s']) else None,
                observed_samples=b-a,observed_single_sample_duration_zero=duration==0.,is_capacity=capacity,
                observed_native_start_s=float(arrays['time_s'][a]),observed_native_end_s=float(arrays['time_s'][b-1]),
                source_warning_flags=run.get('warning_flags',[]),**mode)
            if b-a>=2:
                spec['zero_current_boundary_voltage_jump_mV']=1000*float(arrays['V'][b-1]-arrays['V'][b-2]) if abs(float(arrays['I_mA'][b-1]))<=.001 and abs(float(arrays['I_mA'][b-2]))<=.001 else None
            if 'value' in decoded:spec['value']=decoded['value']
            if fid=='F046' and ns==1:spec['allow_initial_event']=True
            specs.append(spec)
        previous=fid
    if len(specs)!=102:raise ValueError('Expected102 continuous protocol steps')
    pulses=[s for s in specs if s['file']=='F047' and s['Ns']==5]
    for number,s in enumerate(pulses,1):
        a,b=s['observed_row_start'],s['observed_row_end_exclusive'];a47=sources['F047']['arrays']
        s['pulse_one_based']=number;s['observed_dq_Ah']=float(abs(a47['dq_mAh'][a:b].sum()/1000))
        s['observed_Q_trapezoid_Ah']=float(abs(np.trapezoid(a47['I_mA'][a:b],a47['time_s'][a:b])/3600000))
    contract=dict(status='conditional_gitt_protocol_prepared',specifications=specs,gaps=gaps,
        source_sha256=guards,source_identities={k:v['identity'] for k,v in sources.items()},
        raw_program_rows={k:v['program']['raw_program_rows'] for k,v in sources.items()},
        pulse_endpoint_policy=pulse_endpoint_policy,native_Q_reference_admitted=False,
        model_Q_path_basis='paper200mAh at fixed2.5A equals288s; independent2.5V earlier stop' if pulse_endpoint_policy=='paper-200mAh' else 'observed pulse duration; not independent capacity prediction',
        paper_defined_path_is_not_binary_Q_code_decoding=True,
        native_loop_program=sources['F047']['program']['raw_program_rows'][7],observed_route_not_predicted_loop=True,
        planned_fresh_cases=2,planned_model_solve_calls=2,planned_steps_per_solve=102,
        prepared_native_sha256={fid:sha(out/(fid+'_native.npz')) for fid in sources},
        audited_common_native_origin=sources['F045']['timestamp_naive'],
        zero_observed_duration_is_not_zero_exposure=True,raw_rows_deleted=0,raw_clock_shifted=False,
        physical_validation_complete=False,ground_truth_export_admitted=False)
    write(out/'protocol_contract.json',contract)
    write(out/'pulse_native_summary.json',[{k:s[k] for k in ('pulse_one_based','run_index','observed_duration_s','observed_dq_Ah',
        'observed_Q_trapezoid_Ah','observed_native_start_s','observed_native_end_s','source_warning_flags')} for s in pulses])
    return contract

def run_case(root,prepared,output,*,label,joint_parameters=None):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import pybamm
    import numpy as np
    from joint_spme_bol_fit import derive_updates,homogeneous_state,select_rest_anchor
    from initialize_spme_from_rest import load_native
    from validate_c_half_rate import frozen_material_scales,resolve_diffusivity,apply_diffusivity
    from dynamic_observation_operator import RawTrace,voltage_shapes,LOADED_TIMES_S,REST_TIMES_S
    from lithium_inventory_audit import audit_plating_capacity
    root,prepared,out=Path(root).resolve(),Path(prepared).resolve(),Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();contract=read(prepared/'protocol_contract.json');guards=dict(contract['source_sha256'])
    guards[str(prepared/'protocol_contract.json')]=sha(prepared/'protocol_contract.json')
    for name in ('joint_spme_bol_fit.py','initialize_spme_from_rest.py','validate_c_half_rate.py','dynamic_observation_operator.py','lithium_inventory_audit.py'):
        p=root/'repo/simulation'/name;guards[str(p)]=sha(p)
    report=dict(status='started',case=label,physical_validation_complete=False,ground_truth_export_admitted=False,
        native_Q_reference_admitted=False,model_solve_calls=0,source_initialization_count=1)
    write(out/'report.json',report);phases=[];observations=[]
    try:
        for p,digest in guards.items():
            if sha(p)!=digest:raise ValueError('Prepared source changed: '+p)
        oldpath=root/'work/validation_proposal_20261103/physical_BOL_fit_v3/parameters_frozen.json';old=read(oldpath)
        if label=='retained_v3':scales=[old['s_Q'],old['s_Q'],old['delta_Q_Li_Ah']];factor=None;materialpath=oldpath
        elif label=='joint_C_frozen':
            materialpath=Path(joint_parameters).resolve();chosen=read(materialpath)
            if chosen.get('D_opened_before_freeze') is not False:raise ValueError('C-only frozen candidate required')
            scales,factor=frozen_material_scales(chosen['chosen_C'])
        else:raise ValueError('Named fixed material case required')
        guards[str(materialpath)]=sha(materialpath)
        configpath=root/'work/validation_proposal_20261103/physical_BOL_fit_v3/declared_experiment.json';config=read(configpath);guards[str(configpath)]=sha(configpath)
        p=pybamm.ParameterValues('OKane2022');updates,inventory=derive_updates(p,*scales);p.update(updates)
        initial,source=load_native(root,'C');target,anchor=select_rest_anchor(initial[0]);state,state_meta=homogeneous_state(p,inventory,target);p.update(state)
        p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
        Dn=apply_diffusivity(p,resolve_diffusivity(frozen_multiplier=factor));guards[Dn['function_source_path']]=Dn['function_source_sha256']
        area=p['Electrode width [m]']*p['Electrode height [m]'];thickness=p['Negative electrode thickness [m]'];F=96485.33212331001
        data={}
        for fid in ('F045','F046','F047'):
            path=prepared/(fid+'_native.npz')
            if sha(path)!=contract['prepared_native_sha256'][fid]:raise ValueError('Prepared native array changed since contract creation')
            guards[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as saved:data[fid]={k:saved[k] for k in saved.files}
        specs=contract['specifications'];steps=[]
        for spec in specs:
            args=dict(duration=spec['duration'],skip_ok=allow_empty(spec))
            if 'termination' in spec:args['termination']=spec['termination']
            constructor=getattr(pybamm.step,spec['operation'])
            steps.append(constructor(**args) if spec['operation']=='rest' else constructor(spec['value'],**args))
        profile=config['fine']
        report.update(material_parameters_path=str(materialpath),material_sha256=sha(materialpath),scales=scales,
            material_updates=updates,initial_concentration_updates=state,initial_state=state_meta,anchor_policy=anchor,
            target_anchor_V=target,inventory=inventory,negative_diffusivity=Dn,mesh_points=profile['mesh_points'],
            rtol=profile['rtol'],atol=profile['atol'],period_s=1.,temperature_K=298.15,options=config['options'],
            preceding_all_main_states_propagated=True,per_pulse_fresh_initialization=False,
            initialization_before_F045_only=True,raw_phase_comparison_clock='phase-local actual raw sample elapsed; full raw/model global times also retained',
            source_sha256=guards)
        snapshots=out/'source_snapshots';snapshots.mkdir()
        for number,(name,digest) in enumerate(guards.items()):
            path=Path(name)
            if path.suffix in ('.py','.json') and path.stat().st_size<2000000:(snapshots/f'{number:02d}_{path.name}').write_bytes(path.read_bytes())
        write(out/'declared_experiment.json',report)
        model=pybamm.lithium_ion.SPMe(options=config['options'])
        sim=pybamm.Simulation(model,parameter_values=p,experiment=pybamm.Experiment([tuple(steps)],period=1,temperature=298.15),
            var_pts=profile['mesh_points'],solver=pybamm.IDAKLUSolver(rtol=profile['rtol'],atol=profile['atol']))
        report['model_solve_calls']=1;write(out/'report.json',report);solution=sim.solve(calc_esoh=False)
        pieces=solution.cycles[-1].steps
        channels=['Negative electrode stoichiometry','Positive electrode stoichiometry','X-averaged negative particle surface stoichiometry',
            'X-averaged positive particle surface stoichiometry','Bulk open-circuit voltage [V]','Loss of lithium inventory [%]',
            'Total lithium lost [mol]','Loss of lithium to negative SEI [mol]','Loss of lithium to negative lithium plating [mol]',
            'Volume-averaged negative lithium plating concentration [mol.m-3]',
            'Volume-averaged negative dead lithium concentration [mol.m-3]','Loss of capacity to negative lithium plating [A.h]']
        previous=None;raw_global_origin=float(data['F045']['time_s'][0])
        runs47={s['run_index']:s for s in specs if s['file']=='F047'}
        for index,piece in enumerate(pieces):
            spec=specs[index];record=dict(step=index,spec=spec)
            if isinstance(piece,pybamm.EmptySolution):
                if not allow_empty(spec):raise ValueError('Actual RPT capacity/exposure step cannot be skipped')
                record.update(status='model_initial_event',model_duration_s=0.,observed_duration_s=spec.get('observed_duration_s'),
                    model_zero_duration_is_not_observed_zero=True,unreproduced_observed_exposure=True,
                    model_transferred_Ah=0.,paper_charge_target_met=False)
                if spec['file']=='F047' and spec['Ns']==5:
                    raw=data['F047'];trace=RawTrace(raw)
                    run=dict(file_id='F047',part='long_03',purpose='GITT',run_index=spec['run_index'],Ns=5,
                        row_start=spec['observed_row_start'],row_end_exclusive=spec['observed_row_end_exclusive'],warning_flags=spec['source_warning_flags'],role='pulse')
                    for point in voltage_shapes(trace,run,LOADED_TIMES_S,'loaded_voltage_shape'):
                        observations.append(dict(**point,model_step=index,model_V_at_actual_raw_phase_elapsed=None,
                            model_voltage_error_mV=None,model_delta_mV_from_same_reference=None,model_minus_raw_delta_mV=None,
                            model_covers_observation=False,model_missing_reason='model_initial_event_not_observed_zero_exposure',
                            raw_global_time_s=point['recorded_time_s'],model_global_sample_time_s=None,model_extrapolated=False))
                    write(out/'shape_observations.json',observations)
                phases.append(record);write(out/'phase_journal.json',phases);continue
            t,v,I,q=[np.asarray(piece[k].entries,dtype=float) for k in ('Time [s]','Voltage [V]','Current [A]','Discharge capacity [A.h]')]
            internal={};missing=[]
            for channel in channels:
                try:internal[channel]=np.asarray(piece[channel].entries,dtype=float).reshape(-1)
                except KeyError:missing.append(channel)
            plating=audit_plating_capacity(internal[channels[-3]],internal[channels[-2]],internal[channels[-1]],
                electrode_area_m2=area,electrode_thickness_m=thickness,faraday_C_per_mol=F)
            if not plating['capacity_identity_pass']:raise ValueError('Plated+dead electrode-volume capacity identity failed')
            fields=dict(model_t_s=t,model_V=v,model_I_A=I,model_discharge_capacity_Ah=q,
                model_terminal_state_y=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1),
                reversible_plated_Ah=plating['plated_Ah'],dead_lithium_Ah=plating['dead_Ah'],
                plating_identity_residual_Ah=plating['residual_Ah'],**internal)
            if previous is not None:
                if abs(q[0]-previous[1])>1e-9 or abs(t[0]-previous[0])>1e-5:raise ValueError('Main state clock/capacity discontinuity')
            previous=(t[-1],q[-1])
            if spec['file']!='gap':
                raw=data[spec['file']];a,b=spec['observed_row_start'],spec['observed_row_end_exclusive']
                fields.update(observed_native_t_s=raw['time_s'][a:b],observed_global_elapsed_from_F045_s=raw['time_s'][a:b]-raw_global_origin,
                    observed_V=raw['V'][a:b],observed_native_I_mA=raw['I_mA'][a:b],observed_source_row=raw['raw_source_row'][a:b])
                record.update(observed_duration_s=spec['observed_duration_s'],
                    model_minus_raw_global_start_s=float(t[0]-(raw['time_s'][a]-raw_global_origin)))
            np.savez_compressed(out/f'phase_{index:03d}.npz',**fields)
            duration=float(t[-1]-t[0]);record.update(status='simulated',model_termination=str(piece.termination),model_start_global_s=float(t[0]),
                model_end_global_s=float(t[-1]),model_duration_s=duration,model_transferred_Ah=float(abs(q[-1]-q[0])),
                internal_end={k:float(a[-1]) for k,a in internal.items()},missing_channels=missing,
                reversible_plated_Ah=float(plating['plated_Ah'][-1]),dead_lithium_Ah=float(plating['dead_Ah'][-1]),
                plating_capacity_identity_pass=True,plating_identity_max_abs_residual_Ah=plating['max_abs_residual_Ah'])
            if spec.get('is_capacity') and str(piece.termination)!='event: Voltage < 2.5 [V] [experiment]':raise ValueError('Actual RPT independent2.5V cutoff failed')
            if spec.get('conditional_pulse'):
                record.update(model_minus_observed_duration_s=duration-spec['observed_duration_s'],
                    observed_exposure_not_fully_covered=duration+1e-6<spec['observed_duration_s'])
                if 'paper_charge_target_Ah' in spec:
                    target_q=spec['paper_charge_target_Ah'];actual_q=abs(float(q[-1]-q[0]))
                    full=str(piece.termination)=='final time'
                    if (full and abs(actual_q-target_q)>1e-7) or actual_q>target_q+1e-7:
                        raise ValueError('Constant-current paper charge/time identity failed')
                    record.update(paper_charge_target_Ah=target_q,paper_Q_minus_model_Q_Ah=target_q-actual_q,
                        **pulse_charge_status(actual_q,target_q))
            elif 'termination' in spec and str(piece.termination)=='final time':raise ValueError('Independent controller event reached watchdog')
            if spec['file']=='F047' and spec['Ns'] in (5,6):
                raw=data['F047'];trace=RawTrace(raw);run=dict(file_id='F047',part='long_03',purpose='GITT',run_index=spec['run_index'],Ns=spec['Ns'],
                    row_start=spec['observed_row_start'],row_end_exclusive=spec['observed_row_end_exclusive'],warning_flags=spec['source_warning_flags'],role='pulse' if spec['Ns']==5 else 'relaxation')
                times=LOADED_TIMES_S if spec['Ns']==5 else REST_TIMES_S
                points=voltage_shapes(trace,run,times,'loaded_voltage_shape' if spec['Ns']==5 else 'postpulse_relaxation',
                    preceding_load=spec['run_index']-1 if spec['Ns']==6 else None)
                reference=points[0];pred_ref=model_point(t-t[0],v,reference['actual_elapsed_s']) if reference['sample_status']=='observed' else None
                for point in points:
                    pred=model_point(t-t[0],v,point['actual_elapsed_s']) if point['sample_status']=='observed' else None
                    delta=1000*(pred-pred_ref) if pred is not None and pred_ref is not None else None
                    observations.append(dict(**point,model_step=index,model_V_at_actual_raw_phase_elapsed=pred,
                        model_voltage_error_mV=1000*(pred-point['V']) if pred is not None else None,
                        model_delta_mV_from_same_reference=delta,
                        model_minus_raw_delta_mV=delta-point['delta_voltage_mV_from_1s_selected_sample'] if delta is not None and point['shape_status']=='observed' else None,
                        model_covers_observation=pred is not None,raw_global_time_s=point['recorded_time_s'],
                        model_global_sample_time_s=float(t[0]+point['actual_elapsed_s']) if pred is not None else None,
                        model_global_query_time_s=float(t[0]+point['actual_elapsed_s']) if point['actual_elapsed_s'] is not None else None,
                        model_extrapolated=False))
            phases.append(record);write(out/'phase_journal.json',phases);write(out/'shape_observations.json',observations)
        if len(pieces)!=102:raise ValueError('Model stopped before complete102-step conditional route')
        metrics={}
        for kind in ('loaded_voltage_shape','postpulse_relaxation'):
            rows=[r for r in observations if r['observation_kind']==kind and r['informative_nonreference_shape']]
            covered=[r for r in rows if r['model_minus_raw_delta_mV'] is not None]
            metrics[kind]=dict(raw_nonreference_points=len(rows),model_covered_points=len(covered),
                shape_RMSE_mV=float(np.sqrt(np.mean([r['model_minus_raw_delta_mV']**2 for r in covered]))) if covered else None,
                shape_max_abs_error_mV=max((abs(r['model_minus_raw_delta_mV']) for r in covered),default=None))
        report.update(status='conditional_continuous_C_GITT_diagnostic_complete',phase_count=len(pieces),phase_results=phases,
            observation_metrics=metrics,shape_observation_count=len(observations),
            model_initial_event_count=sum(r['status']=='model_initial_event' for r in phases),
            conditional_pulse_underexposure_count=sum(r.get('observed_exposure_not_fully_covered',False) or r.get('unreproduced_observed_exposure',False) for r in phases))
    except Exception as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc),partial_phase_count=len(phases))
    report['wall_seconds']=time.monotonic()-started
    report['source_unchanged']=all(sha(p)==digest for p,digest in guards.items()) if guards else None
    report['source_sha256']=guards
    if report['source_unchanged'] is False:report.update(status='failed_source_integrity')
    write(out/'report.json',report)
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--prepare-only',action='store_true');a.add_argument('--prepared',type=Path)
    a.add_argument('--pulse-endpoint-policy',choices=('recorded-duration','paper-200mAh'));a.add_argument('--case',choices=('retained_v3','joint_C_frozen'))
    a.add_argument('--joint-parameters',type=Path);v=a.parse_args()
    if v.prepare_only:
        if v.pulse_endpoint_policy is None:a.error('Explicit recorded-duration policy required while Q-reference unresolved')
        r=prepare(v.root,v.output,pulse_endpoint_policy=v.pulse_endpoint_policy)
    else:
        if v.prepared is None or v.case is None:a.error('Prepared source contract and fixed case required')
        r=run_case(v.root,v.prepared,v.output,label=v.case,joint_parameters=v.joint_parameters)
    print(json.dumps({k:r.get(k) for k in ('status','case','wall_seconds','model_solve_calls','phase_count','error')}))
    raise SystemExit(0 if r['status'] in ('conditional_gitt_protocol_prepared','conditional_continuous_C_GITT_diagnostic_complete') else 2)
