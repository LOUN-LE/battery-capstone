"""DFN-only contracts and immutable checkpoints; historical SPMe artifacts remain separate."""
from pathlib import Path
import hashlib,json,math,os,pickle
import numpy as np
from run_exposure_scenario_smoke import read,write,sha
from corrected_history_checkpoint import state_layout
from diagnostic_model_family import validate_model_family,verify_rest_equations
from runtime_physics_identity import capture,validate_identical,digest

def validate_contract(path,multiplier,reviewed_sha=None,executing=False):
    c=read(path)
    if validate_model_family(c.get('model_family'))!='DFN':raise ValueError('DFN pilot requires explicit DFN model family')
    if c.get('role')!='C_RPT0_RPT1_conditional_DFN_interstitial_SEI_pilot' or c.get('cell')!='C' or c.get('stop_after_rpt')!=1:
        raise ValueError('Only the declared C RPT0/RPT1 pilot is supported')
    if c['mechanism']!='interstitial-diffusion limited' or c['SEI_lithium_interstitial_diffusivity_reference_m2_s']!=1e-20 or c['SEI_activation_energy_J_mol']!=38000.:
        raise ValueError('Declared mechanism/reference/Ea changed; new reviewed driver required')
    if isinstance(multiplier,bool) or not math.isfinite(multiplier) or not c['common_primary_secondary_multiplier_bounds'][0]<=multiplier<=c['common_primary_secondary_multiplier_bounds'][1]:
        raise ValueError('Finite multiplier inside predeclared engineering bounds required')
    if c['physical_validation_complete'] is not False or c['lifetime_restart_admitted'] is not False:
        raise ValueError('Diagnostic contract must not claim physical admission')
    if executing and reviewed_sha!=sha(path):raise ValueError('Root-reviewed contract SHA256 required before PDE execution')
    return c

def prepare_parameters(candidate,anchor,contract,multiplier):
    from composite_candidate_parameters import build_frozen_candidate
    import pybamm,inspect
    from pybamm.models.submodels.interface.sei.sei_growth import SEIGrowth
    p,options,meta=build_frozen_candidate(candidate,anchor)
    reference=contract['SEI_lithium_interstitial_diffusivity_reference_m2_s']
    for prefix in ('Primary: ','Secondary: '):
        if p[prefix+'SEI lithium interstitial diffusivity [m2.s-1]']!=reference or p[prefix+'SEI solvent diffusivity [m2.s-1]']!=2.5e-22:
            raise ValueError('Unexpected default coefficient; never transfer solvent multiplier')
    p.update({prefix+name:value for prefix in ('Primary: ','Secondary: ') for name,value in [
        ('SEI lithium interstitial diffusivity [m2.s-1]',reference*multiplier),('SEI growth activation energy [J.mol-1]',contract['SEI_activation_energy_J_mol'])]})
    options.update({'SEI':('interstitial-diffusion limited','none'),'SEI porosity change':'false'})
    physical_sources=[Path(__file__),Path(__file__).with_name('run_dfn_history_pilot.py'),Path(__file__).with_name('diagnostic_model_family.py'),Path(inspect.getfile(SEIGrowth)),
        *[Path(__file__).with_name(n) for n in ('composite_candidate_parameters.py','joint_spme_bol_fit.py','initialization_contract.py')]]
    identity=capture(p,physical_sources)
    return p,options,meta,identity

def is_rest_item(item):
    return item['kind'] in ('recorded_rest_controller','assumed_exposure') or (item['kind']=='native_RPT_control' and item['spec']['operation']=='rest')

def rest_batches(plan,start=0,max_steps=64):
    """Every resumable batch ends at explicit rest; final RPT may end under load."""
    if max_steps<1:raise ValueError('Positive batch size required')
    begin=start
    while begin<len(plan):
        ceiling=min(begin+max_steps,len(plan))
        if ceiling==len(plan):end=ceiling
        else:
            choices=[i+1 for i in range(begin,ceiling) if is_rest_item(plan[i])]
            if not choices:raise ValueError('No safe rest checkpoint within declared batch size')
            end=choices[-1]
        yield begin,end
        begin=end

def voltage_extrema(values,limits=(2.5,4.2)):
    v=np.asarray(values,dtype=float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('Finite nonempty voltage samples required')
    low,high=map(float,limits);minimum=float(v.min());maximum=float(v.max())
    return dict(min_V=minimum,max_V=maximum,below_nominal_V=max(0.,low-minimum),above_nominal_V=max(0.,maximum-high),
                nominal_limit_violation=minimum<low or maximum>high)

def raw_voltage(item,rawsteps,arrays):
    if item['kind']=='assumed_exposure':return dict(available=False,reason='Unobserved interval; no invented rawV')
    source=item['native_source'] if item['kind']=='native_RPT_control' else item['source_step']
    rawstep=rawsteps[item['source_step']['raw_step_index']]
    a=arrays.get(source);path=(arrays.directory/source['arrays_file']).resolve()
    if 'V_V' not in a:
        with np.load(path,allow_pickle=False) as z:a.update(V_V=z['V_V'],analysis_valid_row=z['analysis_valid_row'])
    window=rawstep if item['kind']=='native_RPT_control' else item['source_step']
    indices=np.arange(window['row_start'],window['row_end_exclusive'])
    if 'start_s' in window:indices=indices[(a['t_s'][indices]>=window['start_s'])&(a['t_s'][indices]<=window['end_s'])]
    valid=a['analysis_valid_row'][indices]&np.isfinite(a['V_V'][indices]);v=a['V_V'][indices][valid]
    if not len(v):return dict(available=False,reason='No valid rawV in source window',raw_rows=len(indices))
    return dict(available=True,raw_rows=len(indices),valid_voltage_rows=len(v),source_array_sha256=source['arrays_sha256'],**voltage_extrema(v))

def validate_charge_residuals(signed,absolute,contract):
    limit=contract['solver_diagnostic_guards']['signed_and_absolute_charge_max_residual_Ah']
    if not np.isfinite([signed,absolute]).all() or max(abs(signed),abs(absolute))>limit:
        raise ValueError('Solver charge residual exceeds declared diagnostic guard')

def native_control_evidence(spec,current,voltage,duration,termination,contract,evidence=None):
    """Check actual native control and the declared endpoint, including CV."""
    I=np.asarray(current,dtype=float);V=np.asarray(voltage,dtype=float)
    guards=contract['solver_diagnostic_guards'];evidence={} if evidence is None else evidence
    if spec['operation']=='voltage':
        evidence['applied_voltage_max_error_V']=float(np.max(abs(V-spec['value'])))
        if evidence['applied_voltage_max_error_V']>guards['applied_voltage_max_error_V']:
            raise ValueError('Native CV applied voltage differs from declared control')
    allowed={}
    for value in spec['termination']:
        number,unit=value.split();number=float(number)
        if unit=='V':
            comparator='>' if spec['value']<0 else '<'
            allowed[f'event: Voltage {comparator} {number:g} [V] [experiment]']=('V',number)
        elif unit=='A':allowed[f'event: abs(Current [A]) < {number:g} [A] [experiment]']=('A',number)
        else:raise ValueError('Unsupported native termination unit')
    if termination=='final time':
        if not spec['timed_endpoint'] or abs(duration-spec['duration_s'])>2e-6:
            raise ValueError('Native RPT event reached watchdog or timed duration differs')
        evidence['native_endpoint_verified']='declared timed endpoint'
    elif termination in allowed:
        unit,target=allowed[termination]
        residual=abs(float(V[-1] if unit=='V' else abs(I[-1]))-target)
        evidence.update(native_endpoint_verified=termination,native_terminal_residual=residual,native_terminal_unit=unit)
        if residual>guards['native_event_terminal_residual_V_or_A']:
            raise ValueError('Native terminal voltage/current does not match its event')
    else:raise ValueError('Unexpected native termination event')
    return evidence

def transition_state_evidence(previous,piece):
    """The controller can add an algebraic current; physical differential state cannot reset."""
    old=previous.all_models[-1];new=piece.all_models[0]
    def differential_layout(model):
        return [(str(key),[(s.start,s.stop,s.step) for s in model.y_slices[key]]) for key in model.rhs]
    if differential_layout(old)!=differential_layout(new) or old.len_rhs!=new.len_rhs:
        raise ValueError('Transferred physical differential state layout differs')
    before=np.asarray(previous.all_ys[-1],dtype=float)[:old.len_rhs,-1]
    after=np.asarray(piece.all_ys[0],dtype=float)[:new.len_rhs,0]
    if not np.allclose(before,after,rtol=1e-12,atol=1e-12):
        raise ValueError('Transferred physical differential state was reset or changed')
    return dict(differential_state_continuity_verified=True,max_abs_differential_change=float(np.max(abs(after-before))),
        prior_algebraic_state_count=int(old.len_alg),next_algebraic_state_count=int(new.len_alg),
        scope='PyBaMM maps shared states; controller-specific algebraic current may be added or removed')

def atomic_json(path,value):
    path=Path(path);pending=path.with_suffix('.partial.json');write(pending,value);os.replace(pending,path)

def record_worker_termination(out,receipt):
    """The termination receipt does not depend on a potentially torn live report."""
    out=Path(out);receipt=dict(receipt)
    pointer=out/'latest_checkpoint.json'
    if pointer.exists():
        try:receipt['latest_committed_checkpoint']=read(pointer)
        except (ValueError,OSError) as exc:receipt['checkpoint_pointer_read_error']=repr(exc)
    atomic_json(out/'run_receipt.json',receipt)
    if receipt['status']=='worker_terminated_at_wall_budget':
        try:
            record=read(out/'report.json');record.update(status='terminated_at_declared_wall_budget',partial_active_batch_not_committed=True)
            atomic_json(out/'report.json',record)
        except (ValueError,OSError) as exc:
            receipt['live_report_annotation_error']=repr(exc);atomic_json(out/'run_receipt.json',receipt)
    return receipt

def bind_rpt_artifact(path,measurement):
    with np.load(path,allow_pickle=False) as z:
        q=z['Q_Ah'];v=z['V'];actual=float(q[-1]-q[0])
        if actual<=0 or actual!=measurement['Qref_Ah'] or abs(float(v[-1])-2.5)>1e-6:
            raise ValueError('Independent RPT raw capacity/cutoff differs')
    measurement.update(raw_artifact=str(Path(path).resolve()),raw_artifact_sha256=sha(path))

def verify_rpt_artifacts(measurements):
    for m in measurements:
        path=Path(m['raw_artifact'])
        if sha(path)!=m['raw_artifact_sha256']:raise ValueError('Fixed Q0/RPT raw artifact changed')
        bind_rpt_artifact(path,dict(m))

def candidate_key(contract_sha,multiplier,identity):
    return digest(dict(model_family='DFN',contract_sha256=contract_sha,Dint_multiplier=multiplier,runtime_identity_sha256=identity['sha256']))

def commit_checkpoint(out,state,report,plan,journal,measurements,identity):
    """Commit a self-consistent immutable snapshot; interrupted staging is ignored."""
    if float(np.asarray(state['Current [A]'].entries).reshape(-1)[-1])!=0:raise ValueError('Only zero-current rest checkpoints can be committed')
    if report.get('model_family')!='DFN':raise ValueError('Checkpoint report must declare DFN model family')
    verify_rpt_artifacts(measurements)
    n=report['completed_steps']
    if not 0<n<len(plan) or len(journal)!=n or [r['step'] for r in journal]!=list(range(n)):
        raise ValueError('Only a complete, contiguous, unfinished batch prefix can be committed')
    if not is_rest_item(plan[n-1]):raise ValueError('Checkpoint plan must end at an explicit rest boundary')
    if report['runtime_identity_sha256']!=identity['sha256']:raise ValueError('Checkpoint runtime declaration differs')
    parent=out/'checkpoints';parent.mkdir(exist_ok=True)
    target=parent/f'step_{n:06d}'
    # Windows may deny a directory rename even when all child-file writes work.
    # A checkpoint exists only after its final manifest is atomically published.
    target.mkdir(exist_ok=False);staging=target
    with (staging/'state.pkl').open('wb') as f:pickle.dump(state,f)
    y=np.asarray(state.all_ys[-1],dtype=float).reshape(-1)
    np.savez_compressed(staging/'terminal.npz',y=y,t=np.asarray(state.t),Q=np.asarray(state['Discharge capacity [A.h]'].entries))
    for name,value in [('journal.json',journal),('measurements.json',measurements),('report.json',report),('runtime_identity.json',identity)]:write(staging/name,value)
    manifest=dict(schema_version=1,role='same_candidate_DFN_interstitial_pilot_checkpoint',model_family='DFN',output_directory=str(out.resolve()),
        candidate_key=report['candidate_key'],contract_sha256=report['contract_sha256'],runtime_identity_sha256=identity['sha256'],
        plan_sha256=digest(plan),completed_steps=n,files={p.name:sha(p) for p in staging.iterdir()},
        checkpoint_current_A=0.,fixed_Q0_Ah=measurements[0]['Qref_Ah'] if measurements else None,
        source_sha256=dict(report['source_sha256']))
    write(staging/'manifest.partial.json',manifest)
    os.replace(staging/'manifest.partial.json',target/'manifest.json')
    receipt=dict(directory=str(target.resolve()),manifest_sha256=sha(target/'manifest.json'),candidate_key=report['candidate_key'],completed_steps=n)
    temporary=out/'latest_checkpoint.partial.json';write(temporary,receipt);os.replace(temporary,out/'latest_checkpoint.json')
    return receipt

def validate_manifest(directory,expected_sha,current,plan,identity,root):
    directory=Path(directory).resolve();work=(root/'work/validation_proposal_20261103/latest_spme_review_20261103').resolve();directory.relative_to(work)
    path=directory/'manifest.json'
    if sha(path)!=expected_sha:raise ValueError('Explicit trusted manifest SHA mismatch')
    m=read(path)
    if m.get('model_family')!='DFN' or current.get('model_family')!='DFN':raise ValueError('Cross-family or undeclared checkpoint model family refused before deserialization')
    if m.get('role')!='same_candidate_DFN_interstitial_pilot_checkpoint' or m['candidate_key']!=current['candidate_key'] or m['contract_sha256']!=current['contract_sha256']:
        raise ValueError('Checkpoint belongs to different candidate or contract')
    if m['plan_sha256']!=digest(plan) or not 0<m['completed_steps']<len(plan):raise ValueError('Checkpoint chronological plan/prefix differs')
    if not is_rest_item(plan[m['completed_steps']-1]):raise ValueError('Checkpoint plan does not end at explicit rest')
    if m['runtime_identity_sha256']!=identity['sha256']:raise ValueError('Checkpoint actual parameter/runtime identity differs')
    for filename,h in m['files'].items():
        p=(directory/filename).resolve();p.relative_to(directory)
        if sha(p)!=h:raise ValueError('Checkpoint component hash differs: '+filename)
    for source,h in m['source_sha256'].items():
        if sha(source)!=h:raise ValueError('Checkpoint source changed: '+source)
    if read(directory/'report.json').get('model_family')!='DFN':raise ValueError('Checkpoint report model family differs before deserialization')
    previous_output=Path(m['output_directory']).resolve()
    if directory.parent.parent!=previous_output:raise ValueError('Checkpoint output-owner mismatch')
    receipt=read(previous_output/'run_receipt.json')
    if receipt.get('status') not in ('worker_finished','worker_terminated_at_wall_budget'):raise ValueError('Predecessor worker termination is not recorded')
    if receipt.get('status')=='worker_finished' and receipt.get('exit_code')==0:raise ValueError('Completed candidate must not be resumed from older prefix')
    try:live_status=read(previous_output/'report.json').get('status')
    except (ValueError,OSError):live_status=None
    if live_status=='dfn_interstitial_pilot_complete':raise ValueError('Completed candidate must not be resumed from older prefix')
    validate_identical(read(directory/'runtime_identity.json'),identity)
    return m

def load_checkpoint(directory,expected_sha,current,plan,identity,root,p,options,mesh):
    directory=Path(directory).resolve();m=validate_manifest(directory,expected_sha,current,plan,identity,root)
    journal=read(directory/'journal.json');measurements=read(directory/'measurements.json');n=m['completed_steps']
    if len(journal)!=n or [r['step'] for r in journal]!=list(range(n)):raise ValueError('Checkpoint journal prefix is inconsistent')
    if [v['rpt_id'] for v in measurements]!=['RPT'+str(x['measurement']['rpt']) for x in plan[:n] if x.get('measurement') is not None]:raise ValueError('Checkpoint RPT sequence differs')
    verify_rpt_artifacts(measurements)
    if m['fixed_Q0_Ah']!=(measurements[0]['Qref_Ah'] if measurements else None):raise ValueError('Manifest fixed Q0 differs from its bound RPT0 artifact')
    # Only this explicitly digest-bound, own-generated local checkpoint is deserialised.
    state=pickle.loads((directory/'state.pkl').read_bytes())
    with np.load(directory/'terminal.npz',allow_pickle=False) as z:
        y=np.asarray(state.all_ys[-1],dtype=float).reshape(-1);q=float(np.asarray(state['Discharge capacity [A.h]'].entries).reshape(-1)[-1]);t=float(state.t[-1])
        if not np.array_equal(y,z['y']) or not np.isfinite(y).all() or t!=float(z['t'][-1]) or q!=float(z['Q'].reshape(-1)[-1]):raise ValueError('Terminal state/clock/charge mismatch')
    proof=verify_rest_equations(state,p,options,mesh,model_family='DFN')
    proof.update(predecessor_checkpoint=str(directory),manifest_sha256=expected_sha,candidate_key=current['candidate_key'],completed_prefix=n,
        initial_SOC_reset=False,fixed_Q0_Ah=m['fixed_Q0_Ah'],terminal_state_exact=True)
    return state,journal,measurements,(t,q),n,proof
