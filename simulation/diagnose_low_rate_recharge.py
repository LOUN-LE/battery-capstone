"""One fixed fine-mesh C0 RPT/rest/recharge diagnostic, without calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from initialization_contract import validate_initialization_anchor


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def compare_axis(model_x,model_v,observed_x,observed_v):
    import numpy as np
    mx,mv,ox,ov=[np.asarray(x,dtype=float) for x in (model_x,model_v,observed_x,observed_v)]
    if len(mx)<2 or len(ox)<2 or not np.all(np.diff(mx)>0) or not np.all(np.diff(ox)>0):
        raise ValueError('Independent monotone absolute axes required')
    if not all(np.all(np.isfinite(x)) for x in (mx,mv,ox,ov)):
        raise ValueError('Finite model/observation curves required')
    mask=(ox>=mx[0])&(ox<=mx[-1]);residual=np.full(len(ox),np.nan)
    residual[mask]=np.interp(ox[mask],mx,mv)-ov[mask]
    values=residual[mask]
    summary=dict(observed_sample_count=len(ox),covered_sample_count=int(mask.sum()),
        observed_sample_coverage_fraction=float(mask.mean()),
        model_axis_start=float(mx[0]),model_axis_end=float(mx[-1]),
        observed_axis_start=float(ox[0]),observed_axis_end=float(ox[-1]),
        model_minus_observed_endpoint=float(mx[-1]-ox[-1]),
        voltage_RMSE_mV=float(np.sqrt(np.mean(values**2))*1000) if len(values) else None,
        voltage_max_abs_error_mV=float(np.max(abs(values))*1000) if len(values) else None,
        voltage_mean_error_mV=float(np.mean(values)*1000) if len(values) else None,
        raw_observations_interpolated=False,model_values_evaluated_at_observed_axis=True,
        axes_stretched=False,extrapolated=False)
    interior=mask.copy();interior[:2]=False;interior[-2:]=False
    summary['interior_excluding_two_boundary_rows_RMSE_mV_diagnostic_only']=float(np.sqrt(np.mean(residual[interior]**2))*1000) if interior.any() else None
    return summary,mask,residual


def raw_charge_Ah(t,native_i_mA):
    import numpy as np
    t,i=np.asarray(t,dtype=float),np.asarray(native_i_mA,dtype=float)
    if len(t)<2 or not np.all(np.diff(t)>0) or not np.all(np.isfinite(i)):
        raise ValueError('Finite recorded current on increasing raw time required')
    return np.r_[0.,np.cumsum((i[1:]+i[:-1])*.5*np.diff(t))/3_600_000]


def run(root,initialization,output,joint_parameters=None):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    import numpy as np
    import pybamm
    root=Path(root).resolve();initpath=Path(initialization).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();init=read(initpath);validate_initialization_anchor(init)
    materialpath=Path(init['material_parameters_file']);material=read(materialpath)
    if sha(materialpath)!=init['material_parameters_sha256'] or material.get('C_only_fit') is not True or material.get('transport_and_degradation_parameters_changed') is not False:
        raise ValueError('Unchanged C-only retained material required')
    source=Path(init['source_C']['path'])
    if sha(source)!=init['source_C']['sha256']:raise ValueError('Native initialization source differs')
    sys.path.insert(0,str(root/'work/preprocess'))
    import common
    cols,meta,_=common.read_mpt(source);ranges,basis=common.split_runs(cols)
    if [int(cols['ns'][a]) for a,b in ranges[:10]]!=list(range(10)):
        raise ValueError('Expected native C0 long01 Ns0..9 sequence')
    a0,b0=ranges[0];relative=cols['t'][a0:b0]-cols['t'][a0]
    select=np.flatnonzero((relative<=120.)&(relative>=0)&(np.abs(cols['I'][a0:b0])<=.001))[-3:]
    policy=init['anchor_policy']
    if len(select)!=3 or select.tolist()!=policy['selected_indices']:
        raise ValueError('Actual native control-interior anchor rows differ')
    if not np.allclose(relative[select],policy['selected_times_s'],rtol=0,atol=1e-12) or not np.allclose(cols['V'][a0:b0][select],policy['selected_voltage_V'],rtol=0,atol=1e-12):
        raise ValueError('Actual native anchor times or voltages differ')
    if abs(float(np.mean(cols['V'][a0:b0][select]))-init['target_voltage_V'])>1e-10:
        raise ValueError('Actual anchor target differs')
    configpath=root/'work/validation_proposal_20261103/physical_BOL_fit_v3/declared_experiment.json'
    config=read(configpath)
    mesh={'x_n':20,'x_s':20,'x_p':20,'r_n':40,'r_p':40}
    if config['fine']['mesh_points']!=mesh:raise ValueError('Requested fine x20/r40 differs from frozen fine profile')
    jointpath=Path(joint_parameters).resolve() if joint_parameters else None
    joint=read(jointpath) if jointpath else None
    if joint is not None:
        if joint.get('D_opened_before_freeze') is not False or joint.get('transport_changed') is not False or joint.get('chosen_C',{}).get('status')!='evaluated':
            raise ValueError('Previously frozen C-only joint candidate with unchanged transport required')
        if joint['source_C']['sha256']!=init['source_C']['sha256']:
            raise ValueError('Joint candidate native source differs from reviewed anchor')
        chosen=joint['chosen_C'];updates=chosen['updates'];state=chosen['initial_state_updates']
        target_inventory=chosen['inventory']['Q_Li_Ah']
        reported_soc=chosen['initial_state']['equilibrium_SOC']
        if abs(chosen['initial_state']['target_rest_endpoint_V']-init['target_voltage_V'])>1e-10:
            raise ValueError('Joint candidate initial voltage differs from declared anchor')
    else:
        updates=material['parameter_updates'];state=init.get('initial_state_updates',{})
        target_inventory=material['inventory']['target_initial_particle_lithium_Ah']
        reported_soc=init['effective_initial_soc']
    statekeys={'Initial concentration in negative electrode [mol.m-3]','Initial concentration in positive electrode [mol.m-3]'}
    if set(state)!=statekeys or not all(math.isfinite(float(x)) for x in state.values()):
        raise ValueError('Joint homogeneous lithium-conserving initial concentration pair required')
    p=pybamm.ParameterValues('OKane2022');p.update(updates);p.update(state)
    p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
    inventory=0.
    for side in ('negative','positive'):
        title=side.capitalize();c=p[f'Initial concentration in {side} electrode [mol.m-3]'];maximum=p[f'Maximum concentration in {side} electrode [mol.m-3]']
        if not 0<c<maximum:raise ValueError('Initial concentration outside physical material range')
        inventory+=c*p[f'{title} electrode active material volume fraction']*p[f'{title} electrode thickness [m]']*p['Electrode width [m]']*p['Electrode height [m]']*96485.33212331001/3600
    if abs(inventory-target_inventory)>1e-8:raise ValueError('Initial pair changed retained lithium inventory')
    frozen=root/'work/validation_proposal_20261103/calibrated_short_numerics_v1/source_snapshots'
    guardpaths=[source,materialpath,initpath,configpath,Path(common.__file__),Path(common.BioLogic.__file__),
        Path(__file__).with_name('initialization_contract.py'),Path(__file__),Path(pybamm.__file__).parent/'input/parameters/lithium_ion/OKane2022.py']+list(frozen.glob('*'))
    if jointpath:guardpaths.append(jointpath)
    guards={str(path.resolve()):sha(path) for path in guardpaths if path.is_file()}
    names=['actual_RPT_initial_rest','actual_RPT_CC','actual_RPT_CV','actual_RPT_pre_discharge_rest','actual_RPT_C10_discharge',
           'post_discharge_rest_6h','pre_recharge_rest_30s','C10_recharge']
    native_groups=[[0],[1],[2],[3,4],[5],[6],[7],[8]]
    steps=[pybamm.step.rest(120,skip_ok=False),
        pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),
        pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),
        pybamm.step.rest(7230,skip_ok=False),
        pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False),
        pybamm.step.rest(21600,skip_ok=False),pybamm.step.rest(30,skip_ok=False),
        pybamm.step.current(-.5,duration=50000,termination='4.2 V',skip_ok=False)]
    expected=list(config['expected_terminations'])+['final time','final time','event: Voltage > 4.2 [V] [experiment]']
    declaration=dict(scope='one retained material/initial-anchor diagnostic; no fitting or hysteresis assumption',
        root=str(root),source_C=str(source),source_C_sha256=sha(source),source_parser=meta['parser'],split_basis=basis,
        initialization_file=str(initpath),initialization_sha256=sha(initpath),anchor_policy_id=init['anchor_policy_id'],
        selected_anchor_rows=select.tolist(),material_parameters_file=str(jointpath or materialpath),material_sha256=sha(jointpath or materialpath),
        retained_material_anchor_provenance_file=str(materialpath),joint_parameters_used=jointpath is not None,
        joint_candidate_frozen_before_low_rate_evaluation=jointpath is not None,
        low_rate_observations_used_in_fit=False,additional_C_internal_diagnostic_not_independent_cell=True,
        effective_initial_soc_report_only=reported_soc,material_updates_applied=updates,initial_state_updates_applied=state,
        solve_initial_soc_remapping_used=False,initial_particle_lithium_Ah=inventory,initial_particle_lithium_residual_Ah=inventory-target_inventory,
        model='SPMe',parameter_set='OKane2022',options=config['options'],mesh_points=mesh,rtol=1e-8,atol=1e-10,
        output_period_s=1.,temperature_K=298.15,temperature_scope='25C isothermal diagnostic, measured surface T is preserved but not imposed',
        phase_names=names,native_phase_Ns=native_groups,expected_terminations=expected,
        all_phases_one_contiguous_Experiment=True,main_state_reset_after_RPT=False,
        source_sha256=guards,physical_validation_complete=False,D_used=False,AWS_started=False)
    write(out/'declared_experiment.json',declaration)
    snapshots=out/'source_snapshots';snapshots.mkdir()
    snapshot_receipt=[]
    for k,path in enumerate([Path(__file__),Path(__file__).with_name('initialization_contract.py'),initpath,materialpath,configpath]+([jointpath] if jointpath else [])):
        destination=snapshots/f'{k:02d}_{path.name}';destination.write_bytes(path.read_bytes())
        snapshot_receipt.append(dict(source_path=str(path.resolve()),snapshot=str(destination.relative_to(out)),sha256=sha(destination)))
    write(out/'source_snapshot_receipt.json',snapshot_receipt)
    report=dict(status='started',**declaration)
    try:
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,
            experiment=pybamm.Experiment(steps,period=1,temperature=298.15),var_pts=mesh,
            solver=pybamm.IDAKLUSolver(rtol=1e-8,atol=1e-10))
        solution=sim.solve(calc_esoh=False)
        pieces=[cycle.steps[-1] for cycle in solution.cycles]
        records=[];all_t=[];all_v=[]
        for index,piece in enumerate(pieces):
            model={key:np.asarray(piece[name].entries,dtype=float) for key,name in
                [('t','Time [s]'),('v','Voltage [V]'),('i','Current [A]'),('q','Discharge capacity [A.h]')]}
            internal={};internal_summary={};unavailable=[]
            channels=['X-averaged negative electrode open-circuit potential [V]',
                'X-averaged positive electrode open-circuit potential [V]',
                'X-averaged negative electrode reaction overpotential [V]',
                'X-averaged positive electrode reaction overpotential [V]',
                'Negative electrode stoichiometry','Positive electrode stoichiometry',
                'X-averaged negative particle surface stoichiometry','X-averaged positive particle surface stoichiometry',
                'Loss of lithium inventory [%]','Loss of active material in negative electrode [%]',
                'Loss of active material in positive electrode [%]','Total lithium lost [mol]']
            for channel in channels:
                try:
                    value=np.asarray(piece[channel].entries,dtype=float);internal[channel]=value
                    internal_summary[channel]=dict(first=float(value.reshape(-1)[0]),last=float(value.reshape(-1)[-1]))
                except KeyError:unavailable.append(channel)
            np.savez_compressed(out/f'internal_state_{index}.npz',**internal)
            indices=np.concatenate([np.arange(*ranges[ns]) for ns in native_groups[index]])
            native={key:np.asarray(cols[col][indices],dtype=float) for key,col in
                [('recorded_t','t'),('v','V'),('i_mA','I'),('surface_T_C','T'),('dq_mAh','dq'),('Q_minus_Qo_mAh','qqo'),('mode','mode')] if cols.get(col) is not None}
            native['elapsed_from_file_start_s']=native['recorded_t']-float(cols['t'][0]);native['raw_source_row']=indices
            np.savez_compressed(out/f'phase_{index}_{names[index]}.npz',
                **{'model_'+k:v for k,v in model.items()},**{'observed_'+k:v for k,v in native.items()},
                model_terminal_state_y=np.asarray(piece.last_state.all_ys[-1],dtype=float).reshape(-1))
            local,mask_local,error_local=compare_axis(model['t']-model['t'][0],model['v'],native['recorded_t']-native['recorded_t'][0],native['v'])
            absolute,mask_abs,error_abs=compare_axis(model['t'],model['v'],native['elapsed_from_file_start_s'],native['v'])
            np.savez_compressed(out/f'comparison_{index}.npz',local_coverage=mask_local,local_voltage_error_V=error_local,
                absolute_elapsed_same_phase_coverage=mask_abs,absolute_elapsed_same_phase_voltage_error_V=error_abs)
            record=dict(phase=index,name=names[index],native_Ns=native_groups[index],model_termination=str(piece.termination),
                expected_termination=expected[index],event_matches=str(piece.termination)==expected[index],
                observed_raw_row_first=int(indices[0]),observed_raw_row_last=int(indices[-1]),
                model_start_elapsed_s=float(model['t'][0]),model_end_elapsed_s=float(model['t'][-1]),
                observed_start_elapsed_s=float(native['elapsed_from_file_start_s'][0]),observed_end_elapsed_s=float(native['elapsed_from_file_start_s'][-1]),
                model_minus_observed_start_s=float(model['t'][0]-native['elapsed_from_file_start_s'][0]),
                phase_local_seconds_no_stretch=local,absolute_file_elapsed_same_phase_no_alignment=absolute,
                raw_surface_temperature_C={k:float(f(native['surface_T_C'])) for k,f in [('min',np.min),('median',np.median),('max',np.max)]},
                model_first_voltage_V=float(model['v'][0]),model_last_voltage_V=float(model['v'][-1]),
                observed_first_voltage_V=float(native['v'][0]),observed_last_voltage_V=float(native['v'][-1]),
                internal_model_channels=internal_summary,unavailable_internal_channels=unavailable,
                raw_preserved=True)
            if index in (4,7):
                sign=1 if index==4 else -1
                mq=sign*(model['q']-model['q'][0]);oq=sign*-raw_charge_Ah(native['recorded_t'],native['i_mA'])
                score,qmask,qerror=compare_axis(mq,model['v'],oq,native['v'])
                record['absolute_transferred_Ah_no_stretch']=score
                record['model_transferred_Ah']=float(mq[-1]);record['observed_trapezoid_transferred_Ah']=float(oq[-1])
                record['observed_instrument_dq_Ah']=float(abs(native['dq_mAh'].sum()/1000))
                np.savez_compressed(out/f'capacity_comparison_{index}.npz',model_transferred_Ah=mq,
                    observed_transferred_Ah=oq,coverage=qmask,error_V=qerror)
            if index>=5:
                record['native_boundary_neighbours']=[dict(raw_source_row=int(k),recorded_t_s=float(cols['t'][k]),Ns=int(cols['ns'][k]),
                    I_mA=float(cols['I'][k]),V=float(cols['V'][k]),surface_T_C=float(cols['T'][k]),recorded_mode=int(cols['mode'][k]),
                    position='before_phase' if k<indices[0] else ('after_phase' if k>indices[-1] else 'inside_boundary_region'))
                    for k in sorted(set(range(max(0,int(indices[0])-2),min(len(cols['t']),int(indices[0])+3)))|
                                    set(range(max(0,int(indices[-1])-2),min(len(cols['t']),int(indices[-1])+3))))]
            records.append(record);all_t.extend(model['t'].tolist());all_v.extend(model['v'].tolist())
            write(out/'phase_journal.json',records)
        all_t,all_v=np.asarray(all_t),np.asarray(all_v)
        # Only model boundary duplicates are coalesced, keeping the later phase's value.
        keep=np.r_[np.diff(all_t)>0,True];all_t,all_v=all_t[keep],all_v[keep]
        if len(pieces)!=8:raise ValueError('Experiment ended before all eight actual phases')
        if not all(r['event_matches'] for r in records):raise ValueError('Native event sequence mismatch')
        if abs(records[0]['model_last_voltage_V']-init['target_voltage_V'])>.001:raise ValueError('Initial rest endpoint misses reviewed anchor by over1mV')
        a,b=ranges[8];global_score,global_mask,global_error=compare_axis(all_t,all_v,cols['t'][a:b]-cols['t'][0],cols['V'][a:b])
        np.savez_compressed(out/'whole_clock_recharge_comparison.npz',model_t=all_t,model_v=all_v,
            observed_recorded_t=cols['t'][a:b],observed_voltage_V=cols['V'][a:b],coverage=global_mask,error_V=global_error)
        report.update(status='low_rate_recharge_diagnostic_complete',phase_results=records,
            actual_initial_RPT_propagated=True,rest_and_recharge_contiguous_state=True,events_all_match=True,
            global_elapsed_recharge_vs_full_trajectory=global_score,
            initial_rest_voltage_error_mV=1000*(records[0]['model_last_voltage_V']-init['target_voltage_V']),
            model_only_duplicate_boundary_samples_coalesced=int((~keep).sum()),
            hysteresis_or_single_cause_identified=False,interpretation='Voltage/RPT direction mismatch is observed only; OCP hysteresis, transport, thermal and unknown history causes are not uniquely identified.')
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                      completed_phase_journal_preserved=(out/'phase_journal.json').exists())
    report['all_guarded_sources_unchanged']=all(sha(path)==expected_hash for path,expected_hash in guards.items())
    report['wall_seconds']=time.monotonic()-started
    write(out/'report.json',report)
    write(out/'artifact_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_sha256.json'})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--initialization',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--joint-parameters',type=Path)
    args=parser.parse_args();result=run(args.root,args.initialization,args.output,args.joint_parameters)
    print(json.dumps({k:result.get(k) for k in ('status','wall_seconds','error','all_guarded_sources_unchanged')},ensure_ascii=False))
    raise SystemExit(0 if result['status']=='low_rate_recharge_diagnostic_complete' else 2)
