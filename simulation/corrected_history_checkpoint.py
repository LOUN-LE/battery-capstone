"""Resume only an integrity-bound corrected trial with identical physical state.

Pickles are accepted only below this local work tree with a caller-supplied
verified digest. This is not a general-purpose or untrusted checkpoint loader.
"""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
from run_exposure_scenario_smoke import read,sha


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def check_contract(previous,current,old_plan,plan,completed):
    if not isinstance(completed,int) or not 0<completed<len(plan):
        raise ValueError('Resume requires a nonempty incomplete chronological prefix')
    for key in ('candidate_metadata','options','mesh','rtol','atol','period_s','temperature_C','SEI_parameter_status'):
        if canonical(previous.get(key))!=canonical(current.get(key)):
            raise ValueError('Checkpoint physical/numerical contract changed: '+key)
    if canonical(old_plan[:completed])!=canonical(plan[:completed]):
        raise ValueError('Checkpoint chronological input prefix changed')


def state_layout(model):
    return [(str(k),[(s.start,s.stop,s.step) for s in v]) for k,v in model.y_slices.items()]


def verify_rest_equations(state,parameters,options,mesh):
    """Authenticate legacy physical parameters through the stored discretisation.

The legacy checkpoint lacks a full ParameterValues fingerprint. Its last step
must be rest so a fresh equivalent controller can be built without assuming a
different time-dependent input. No PDE solve or reinitialisation is performed.
"""
    import pybamm
    from pybamm.expression_tree.operations.serialise_kernel import encode
    if float(np.asarray(state['Current [A]'].entries).reshape(-1)[-1])!=0:
        raise ValueError('Legacy equation authentication requires a zero-current checkpoint')
    probe=pybamm.Simulation(pybamm.lithium_ion.SPMe(options=options),parameter_values=parameters.copy(),
                           var_pts=mesh,experiment=pybamm.Experiment([pybamm.step.rest(.004)],temperature=298.15))
    probe.build_for_experiment();new=next(iter(probe.steps_to_built_models.values()));old=state.all_models[-1]
    if state_layout(new)!=state_layout(old):raise ValueError('Checkpoint state layout differs')
    fingerprints={}
    for name in ('concatenated_rhs','concatenated_algebraic','mass_matrix','concatenated_initial_conditions'):
        before=canonical(encode(getattr(old,name)));after=canonical(encode(getattr(new,name)))
        if before!=after:raise ValueError('Checkpoint processed physical equations changed: '+name)
        fingerprints[name]=hashlib.sha256(before.encode()).hexdigest()
    for name in ('Voltage [V]','Negative electrode primary open-circuit potential [V]',
                 'Negative electrode secondary open-circuit potential [V]','Positive electrode open-circuit potential [V]'):
        before=canonical(encode(old.get_processed_variable(name)));after=canonical(encode(new.get_processed_variable(name)))
        if before!=after:raise ValueError('Checkpoint processed voltage/OCP observer changed: '+name)
        fingerprints[name]=hashlib.sha256(before.encode()).hexdigest()
    before=canonical(encode(old.events));after=canonical(encode(new.events))
    if before!=after:raise ValueError('Checkpoint processed model events changed')
    fingerprints['events']=hashlib.sha256(before.encode()).hexdigest()
    return dict(discretised_rest_equations_exact=True,equation_sha256=fingerprints,
                scope='same processed rest equations, mass matrix, initial conditions, voltage/OCP observers, events and state layout; legacy runtime versions remain unrecorded')


def measurement_evidence(directory,plan,measurements,completed):
    """Bind fixed Q0 and every completed RPT to independently saved raw PDE Q."""
    result=[]
    for measured,item_index in zip(measurements,[i for i,x in enumerate(plan[:completed]) if x.get('measurement') is not None]):
        owner=Path(directory);name=f'step_{item_index:06d}.npz';seen=set()
        while not (owner/name).exists():
            if str(owner) in seen:raise ValueError('Checkpoint RPT provenance cycle')
            seen.add(str(owner));report=read(owner/'report.json');prior=report.get('resume_proof',{}).get('predecessor')
            if not prior:raise ValueError('Checkpoint raw RPT artifact missing')
            owner=Path(prior).resolve()
        artifact=owner/name
        with np.load(artifact,allow_pickle=False) as z:
            q=np.asarray(z['Q_Ah'],dtype=float);voltage=np.asarray(z['V'],dtype=float)
            actual=float(q[-1]-q[0])
            if not np.isfinite(actual) or actual<=0 or actual!=measured['Qref_Ah']:
                raise ValueError('Fixed Q0/RPT capacity differs from saved raw PDE charge')
            if abs(float(voltage[-1])-2.5)>1e-6:raise ValueError('Saved RPT lacks independent 2.5V endpoint')
        result.append(dict(rpt_id=measured['rpt_id'],Qref_Ah=actual,path=str(artifact),sha256=sha(artifact)))
    if len(result)!=len(measurements):raise ValueError('RPT raw-artifact series incomplete')
    return result


def load_resume(directory,expected_sha,root,current,plan,runtime_identity=None):
    directory=Path(directory).resolve();work=(root/'work/validation_proposal_20261103/latest_spme_review_20261103').resolve()
    directory.relative_to(work)
    checkpoint=directory/'last_state.pkl'
    if not expected_sha or sha(checkpoint)!=expected_sha:
        raise ValueError('Explicit trusted checkpoint SHA256 mismatch')
    previous=read(directory/'report.json');old_plan=read(directory/'route.json')['plan'];n=previous['completed_steps']
    check_contract(previous,current,old_plan,plan,n)
    if previous.get('runtime_identity_sha256'):
        from runtime_physics_identity import validate_identical
        old_identity=read(directory/'runtime_identity.json')
        if old_identity['sha256']!=previous['runtime_identity_sha256']:raise ValueError('Recorded runtime identity changed')
        if runtime_identity is None:raise ValueError('Current actual parameter/runtime identity missing')
        validate_identical(old_identity,runtime_identity)
    if previous.get('checkpoint_sha256',expected_sha)!=expected_sha:
        raise ValueError('Checkpoint digest differs from recorded terminal checkpoint')
    if previous.get('status') not in ('terminated_at_declared_600s_budget','terminated_at_declared_wall_budget','running','failed'):
        raise ValueError('Predecessor is not an unfinished trial')
    receipt=read(directory/'run_receipt.json')
    if receipt.get('status') not in ('worker_terminated_at_wall_budget','worker_finished'):
        raise ValueError('Predecessor termination receipt missing')
    revisions=[]
    for name,digest in previous['source_sha256'].items():
        path=Path(name)
        if sha(path)==digest:continue
        if path.name not in ('run_corrected_history_trial.py','corrected_history_checkpoint.py'):
            raise ValueError('Predecessor source/data changed: '+name)
        snapshots=list((directory/'source_snapshots').glob('*_'+path.name))
        if len(snapshots)!=1 or sha(snapshots[0])!=digest:
            raise ValueError('Prior driver revision cannot be authenticated')
        revisions.append(dict(file=name,previous_sha256=digest,current_sha256=sha(path),
                              scope='new runtime compression/resume; immutable physical contract and executed prefix verified'))
    journal=read(directory/'phase_journal.json');measured=read(directory/'measurements.json')
    if len(journal)!=n or [x['step'] for x in journal]!=list(range(n)):
        raise ValueError('Noncontiguous completed checkpoint journal')
    if canonical(measured)!=canonical(previous['measurements']):
        raise ValueError('Checkpoint fixed-Q0 measurement history differs')
    expected_ids=['RPT'+str(x['measurement']['rpt']) for x in plan[:n] if x.get('measurement') is not None]
    if [x['rpt_id'] for x in measured]!=expected_ids:
        raise ValueError('Checkpoint measured RPT chronology differs')
    evidence=measurement_evidence(directory,plan,measured,n)
    state=pickle.loads(checkpoint.read_bytes())
    terminal=previous.get('terminal_state_artifact',f'step_{n-1:06d}.npz')
    with np.load(directory/terminal,allow_pickle=False) as z:
        y=np.asarray(state.all_ys[-1],dtype=float).reshape(-1)
        if not np.array_equal(y,z['terminal_y']) or not np.isfinite(y).all():
            raise ValueError('Checkpoint differs from saved terminal physical states')
        if float(state.t[-1])!=float(z['t_s'][-1]):raise ValueError('Checkpoint terminal clock changed')
        q=float(np.asarray(state['Discharge capacity [A.h]'].entries).reshape(-1)[-1])
        if q!=float(z['Q_Ah'][-1]):raise ValueError('Checkpoint accumulated charge changed')
        last=(float(z['t_s'][-1]),q)
    for key,value in current['options'].items():
        if canonical(state.all_models[-1].options[key])!=canonical(value):
            raise ValueError('Checkpoint embedded model option differs: '+key)
    proof=dict(predecessor=str(directory),checkpoint_sha256=expected_sha,completed_prefix=n,
               fixed_Q0_Ah=measured[0]['Qref_Ah'],terminal_time_s=last[0],terminal_charge_Ah=last[1],
               terminal_state_count=len(y),source_revisions=revisions,
               declared_physical_contract_identical=True,terminal_state_exact=True,initial_SOC_reset=False,
               runtime_identity_recorded_in_predecessor=bool(previous.get('runtime_identity_sha256')),
               RPT_raw_artifacts=evidence)
    return state,journal,measured,last,n,proof
