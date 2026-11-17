"""Seven fresh engineering perturbations of cutoff, rest and recharge.

No fitting, literature prior, new accepted material or physical pass is produced.
The original low-rate/joint-fit drivers are imported read-only and never edited.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

CASES=[('baseline',1.,1.,1.),('Dn_x0p1',.1,1.,1.),('Dn_x10',10.,1.,1.),
       ('i0n_x0p1',1.,.1,1.),('i0n_x10',1.,10.,1.),
       ('i0p_x0p1',1.,1.,.1),('i0p_x10',1.,1.,10.)]
CHANNELS=['Negative electrode stoichiometry','Positive electrode stoichiometry',
    'X-averaged negative particle surface stoichiometry','X-averaged positive particle surface stoichiometry',
    'Bulk open-circuit voltage [V]','Surface open-circuit voltage [V]','Battery open-circuit voltage [V]',
    'X-averaged negative electrode reaction overpotential [V]',
    'X-averaged positive electrode reaction overpotential [V]']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def scale_function(original,factor):
    if factor<=0:raise ValueError('Positive engineering factor required')
    if factor==1:return original
    if callable(original):
        def changed(*args):return factor*original(*args)
        return changed
    return factor*original

def validate_fine_reference(fine,init,initialization_sha256):
    checks=dict(retained_material=fine.get('material_sha256')==init['material_parameters_sha256'],
        native_source=fine.get('source_C_sha256')==init['source_C']['sha256'],
        initial_concentration_pair=fine.get('initial_state_updates_applied')==init['initial_state_updates'],
        effective_soc=fine.get('effective_initial_soc_report_only')==init['effective_initial_soc'],
        anchor_policy=fine.get('anchor_policy_id')==init['anchor_policy_id'],
        anchor_rows=fine.get('selected_anchor_rows')==init['anchor_policy']['selected_indices'],
        joint_candidate_not_used=fine.get('joint_parameters_used',False) is False)
    if not all(checks.values()):raise ValueError('Fine retained-v3 physical input mismatch: '+str(checks))
    return dict(physical_inputs_exactly_match=True,checks=checks,
        whole_initialization_file_sha256_matches=fine['initialization_sha256']==initialization_sha256,
        fine_recorded_initialization_sha256=fine['initialization_sha256'],current_initialization_sha256=initialization_sha256,
        metadata_hash_difference_disclosed=True)

def tail_median(t,values,*,tail_seconds=120.,exclude_last_rows=0):
    import numpy as np
    t,v=np.asarray(t,dtype=float),np.asarray(values,dtype=float)
    if t.ndim!=1 or t.shape!=v.shape or not len(t):raise ValueError('Paired scalar trace required')
    select=t>=t[-1]-tail_seconds
    if exclude_last_rows:select[-exclude_last_rows:]=False
    if not np.any(select):raise ValueError('No tail samples')
    return dict(median=float(np.median(v[select])),sample_count=int(select.sum()),
        tail_seconds=tail_seconds,exclude_last_rows=exclude_last_rows,
        first_selected_time_s=float(t[select][0]),last_selected_time_s=float(t[select][-1]))

def run_case(root,initialization,fine_reference,output,case_id):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    spec=next(c for c in CASES if c[0]==case_id)
    report=dict(status='started',case_id=case_id,engineering_factors=dict(Dn=spec[1],i0n=spec[2],i0p=spec[3]),
        fresh_start=True,all_phases_state_continuous=True,model_solve_called=False,
        perturbation_scope='one-at-a-time engineering counterexample; not literature prior, identified material, fit or acceptance',
        physical_validation_complete=False,D_used=False,AWS_started=False)
    write(out/'report.json',report);started=time.monotonic()
    guards={}
    try:
        import numpy as np
        import pybamm
        from initialization_contract import validate_initialization_anchor
        from diagnose_low_rate_recharge import compare_axis,raw_charge_Ah
        root=Path(root).resolve();initpath=Path(initialization).resolve();finepath=Path(fine_reference).resolve()
        init=read(initpath);validate_initialization_anchor(init);fine=read(finepath)
        materialpath=Path(init['material_parameters_file']);material=read(materialpath)
        if sha(materialpath)!=init['material_parameters_sha256']:raise ValueError('Retained material changed')
        if abs(init['effective_initial_soc']-.36966180929725573)>1e-12:raise ValueError('Reviewed .3696618093 anchor required')
        source=Path(init['source_C']['path'])
        if sha(source)!=init['source_C']['sha256']:raise ValueError('Raw C source changed')
        fine_input_check=validate_fine_reference(fine,init,sha(initpath))
        configpath=root/'work/validation_proposal_20261103/physical_BOL_fit_v3/declared_experiment.json';config=read(configpath)
        sys.path.insert(0,str(root/'work/preprocess'));import common
        cols,meta,_=common.read_mpt(source);ranges,basis=common.split_runs(cols)
        if [int(cols['ns'][a]) for a,b in ranges[:10]]!=list(range(10)):raise ValueError('Native eight-phase protocol source mismatch')
        guardpaths=[Path(__file__),Path(__file__).with_name('diagnose_low_rate_recharge.py'),Path(__file__).with_name('joint_spme_bol_fit.py'),
            initpath,materialpath,source,configpath,finepath,Path(common.__file__)]
        guards={str(p):sha(p) for p in guardpaths}
        write(out/'source_hashes_before.json',guards)
        snapshots=out/'source_snapshots';snapshots.mkdir()
        for k,p in enumerate((Path(__file__),initpath,materialpath,configpath)):(snapshots/f'{k}_{p.name}').write_bytes(p.read_bytes())
        p=pybamm.ParameterValues('OKane2022');p.update(material['parameter_updates']);p.update(init['initial_state_updates'])
        p.update({'Initial temperature [K]':298.15,'Ambient temperature [K]':298.15})
        for name,factor in zip(('Negative particle diffusivity [m2.s-1]','Negative electrode exchange-current density [A.m-2]',
                               'Positive electrode exchange-current density [A.m-2]'),spec[1:]):
            p.update({name:scale_function(p[name],factor)})
        steps=[pybamm.step.rest(120,skip_ok=False),pybamm.step.current(-1.5,duration=20000,termination='4.2 V',skip_ok=False),
            pybamm.step.voltage(4.2,duration=20000,termination='0.05 A',skip_ok=False),pybamm.step.rest(7230,skip_ok=False),
            pybamm.step.current(.5,duration=50000,termination='2.5 V',skip_ok=False),pybamm.step.rest(21600,skip_ok=False),
            pybamm.step.rest(30,skip_ok=False),pybamm.step.current(-.5,duration=50000,termination='4.2 V',skip_ok=False)]
        expected=fine['expected_terminations'];names=fine['phase_names'];groups=fine['native_phase_Ns']
        mesh={'x_n':10,'x_s':10,'x_p':10,'r_n':20,'r_p':20}
        report.update(model='SPMe',parameter_set='OKane2022',options=config['options'],mesh_points=mesh,
            rtol=1e-7,atol=1e-9,output_period_s=10.,temperature_K=298.15,
            initial_effective_soc=init['effective_initial_soc'],initialization_sha256=sha(initpath),
            material_sha256=sha(materialpath),source_C_sha256=sha(source),fine_reference_sha256=sha(finepath),
            expected_terminations=expected,phase_names=names,native_phase_Ns=groups,main_RPT_state_reset=False,
            fine_reference_input_match=fine_input_check)
        sim=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=config['options']),parameter_values=p,
            experiment=pybamm.Experiment(steps,period=10,temperature=298.15),var_pts=mesh,
            solver=pybamm.IDAKLUSolver(rtol=1e-7,atol=1e-9))
        report['model_solve_called']=True;write(out/'report.json',report)
        solution=sim.solve(calc_esoh=False)
        phases=[]
        for index,cycle in enumerate(solution.cycles):
            if index>=8:raise ValueError('Unexpected extra phase')
            piece=cycle.steps[-1]
            t,v,I,q=[np.asarray(piece[k].entries,dtype=float) for k in ('Time [s]','Voltage [V]','Current [A]','Discharge capacity [A.h]')]
            internal={};missing=[]
            for name in CHANNELS:
                try:internal[name]=np.asarray(piece[name].entries,dtype=float).reshape(-1)
                except KeyError:missing.append(name)
            rows=np.concatenate([np.arange(*ranges[n]) for n in groups[index]])
            ot,ov,oi=[np.asarray(cols[k][rows],dtype=float) for k in ('t','V','I')]
            local,mask,error=compare_axis(t-t[0],v,ot-ot[0],ov)
            np.savez_compressed(out/f'phase_{index}.npz',model_t_s=t,model_V=v,model_I_A=I,model_q_Ah=q,
                observed_native_t_s=ot,observed_V=ov,observed_native_I_mA=oi,observed_source_row=rows,
                local_comparison_mask=mask,local_voltage_error_V=error,**internal)
            record=dict(phase=index,name=names[index],model_duration_s=float(t[-1]-t[0]),observed_duration_s=float(ot[-1]-ot[0]),
                model_termination=str(piece.termination),expected_termination=expected[index],event_matches=str(piece.termination)==expected[index],
                model_first_V=float(v[0]),model_last_V=float(v[-1]),observed_first_V=float(ov[0]),observed_last_V=float(ov[-1]),
                local_seconds=local,internal_end={name:float(value[-1]) for name,value in internal.items()},missing_channels=missing)
            if index in (4,7):
                sign=1 if index==4 else -1
                mq=sign*(q-q[0]);oq=-sign*raw_charge_Ah(ot,oi)
                score,qmask,qerror=compare_axis(mq,v,oq,ov)
                record.update(model_transferred_Ah=float(mq[-1]),observed_transferred_Ah=float(oq[-1]),
                    raw_instrument_dq_Ah=float(abs(cols['dq'][rows].sum()/1000)),absolute_Ah=score)
                np.savez_compressed(out/f'capacity_{index}.npz',model_q_Ah=mq,observed_q_Ah=oq,
                    covered=qmask,model_minus_raw_V=qerror)
            if index==5:
                record.update(model_tail120s=tail_median(t,v),observed_tail120s=tail_median(ot,ov),
                    observed_tail120s_exclude2_boundary_rows=tail_median(ot,ov,exclude_last_rows=2),
                    equilibrium_tail120s=tail_median(t,internal['Bulk open-circuit voltage [V]']),
                    surface_ocv_tail120s=tail_median(t,internal['Surface open-circuit voltage [V]']))
            phases.append(record);write(out/'phase_journal.json',phases)
        report.update(phase_results=phases,phase_count=len(phases),events_all_match=len(phases)==8 and all(p['event_matches'] for p in phases))
        if not report['events_all_match']:raise ValueError('Early or mismatched event; available phases preserved')
        report['status']='engineering_perturbation_diagnostic_complete'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),partial_phase_journal_preserved=(out/'phase_journal.json').exists())
    report['wall_seconds']=time.monotonic()-started
    after={str(p):sha(p) for p in guards};write(out/'source_hashes_after.json',after)
    report['guarded_source_count']=len(guards)
    report['all_guarded_sources_unchanged']=guards==after if guards else None
    if guards and guards!=after:
        report.update(status='failed_source_integrity',error='Guarded source changed during case')
    write(out/'report.json',report)
    return report

def run_suite(root,initialization,fine_reference,output,wall_budget_s=360):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();receipts=[]
    for case_id,*_ in CASES:
        remaining=wall_budget_s-(time.monotonic()-start)
        if remaining<=1:
            receipts.append(dict(case_id=case_id,status='not_started_wall_budget_exhausted'));continue
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--root',str(root),'--initialization',str(initialization),
            '--fine-reference',str(fine_reference),'--output',str(out/case_id),'--case-id',case_id]
        try:
            p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=remaining)
            (out/(case_id+'.log')).write_text(p.stdout,encoding='utf-8')
            receipts.append(dict(case_id=case_id,status='child_returned',exit_code=p.returncode,
                report=read(out/case_id/'report.json') if (out/case_id/'report.json').exists() else None))
        except subprocess.TimeoutExpired as exc:
            (out/(case_id+'.log')).write_text(str(exc.stdout or ''),encoding='utf-8')
            receipts.append(dict(case_id=case_id,status='wall_timeout',timeout_s=remaining,
                report=read(out/case_id/'report.json') if (out/case_id/'report.json').exists() else None))
        write(out/'suite_journal.json',receipts)
    successful=sum(r.get('exit_code')==0 and (r.get('report') or {}).get('status')=='engineering_perturbation_diagnostic_complete' for r in receipts)
    report=dict(status='engineering_screen_finished' if successful==len(CASES) else 'engineering_screen_incomplete',receipts=receipts,wall_seconds=time.monotonic()-start,
        wall_budget_s=wall_budget_s,maximum_model_solve_calls=7,
        actual_solve_calls=sum(bool((r.get('report') or {}).get('model_solve_called')) for r in receipts),
        successful_cases=successful,
        physical_validation_complete=False,calibration_performed=False,engineering_perturbations_not_literature_priors=True,
        baseline_fine_comparison_required_before_interpretation=True)
    write(out/'report.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--initialization',type=Path,required=True);parser.add_argument('--fine-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=[c[0] for c in CASES])
    parser.add_argument('--wall-budget-s',type=float,default=360.)
    args=parser.parse_args()
    result=run_case(args.root,args.initialization,args.fine_reference,args.output,args.case_id) if args.case_id else run_suite(
        args.root,args.initialization,args.fine_reference,args.output,args.wall_budget_s)
    print(json.dumps({k:result.get(k) for k in ('status','case_id','wall_seconds','actual_solve_calls','successful_cases','error')},ensure_ascii=False))
    raise SystemExit(0 if result['status'] in ('engineering_perturbation_diagnostic_complete','engineering_screen_finished') else 2)
