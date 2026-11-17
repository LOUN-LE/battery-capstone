"""Compile an explicitly assumed protocol route, never measured-history admission.

The observed Ns order supplies the route; the model supplies native CC/CV
endpoints. Unknown intervals are assumed zero current at a declared temperature.
This is conditional parameter estimation, not reconstruction of missing data.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from numbers import Integral,Real
from pathlib import Path
import sys
from native_program_contracts import decode_mb


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_uniform_recorded_mode(summary, source_ref):
    """Never replace a mixed-mode Ns envelope with its majority controller."""
    try:
        parts=[part.split('x') for part in summary.split(';')]
        modes={int(mode) for mode,count in parts if int(count)>0}
    except (AttributeError,TypeError,ValueError):
        raise ValueError('Invalid recorded mode summary: '+source_ref) from None
    if len(modes)!=1 or not modes <= {1,2,3}:
        raise ValueError('Mixed or unsupported recorded modes require raw subsegment review: '+source_ref)
    return next(iter(modes))


def recorded_mode_blocks(times, modes, *, max_sample_gap_s):
    """Diagnostic mode/clock split; intervals remain observations, not controls.

    A large sampling interval is marked for review even if both endpoint modes
    agree. This helper never fills its interior with rest or constant current.
    """
    times=list(times);modes=list(modes)
    if not times or len(times)!=len(modes):raise ValueError('Paired nonempty time/mode records required')
    if (isinstance(max_sample_gap_s,bool) or not isinstance(max_sample_gap_s,Real)
        or not math.isfinite(max_sample_gap_s) or max_sample_gap_s<=0):raise ValueError('Explicit finite sampling-gap threshold required')
    if any(isinstance(t,bool) or not isinstance(t,Real) or not math.isfinite(t) for t in times):
        raise ValueError('Finite recorded times required')
    if any(isinstance(m,bool) or not isinstance(m,Integral) for m in modes):raise ValueError('Integer recorded modes required')
    times=[float(t) for t in times];modes=[int(m) for m in modes]
    starts=[0];clock_breaks=[];sampling_intervals=[]
    for i in range(1,len(times)):
        dt=times[i]-times[i-1]
        if dt<=0:clock_breaks.append(dict(left_row=i-1,right_row=i,dt_s=dt))
        if modes[i]!=modes[i-1] or dt<=0:starts.append(i)
        if dt>max_sample_gap_s:
            sampling_intervals.append(dict(left_row=i-1,right_row=i,dt_s=dt,
                left_mode=modes[i-1],right_mode=modes[i],operation_between_samples=None,
                reviewed_continuous_exposure=False))
    blocks=[]
    for a,b in zip(starts,starts[1:]+[len(times)]):
        blocks.append(dict(row_start=a,row_end_exclusive=b,recorded_mode=modes[a],
            first_sample_s=times[a],last_sample_s=times[b-1],sample_span_s=times[b-1]-times[a],
            next_boundary_kind=('clock_break' if times[b]<=times[b-1] else 'recorded_mode_change') if b<len(times) else None,
            next_mode_boundary_bracket_s=[times[b-1],times[b]] if b<len(times) and times[b]>times[b-1] else None))
    return dict(blocks=blocks,sampling_intervals_for_review=sampling_intervals,clock_breaks=clock_breaks,
                physical_replay_admissible=False)


def native_step(control, observed_s):
    """Convert constant-current charge limits to time without using target Qref."""
    timed, termination = [], []
    for limit in control['limits']:
        q, v = limit['quantity'], limit['value']
        if q == 'time_s': timed.append(v)
        elif q == 'absolute_step_charge_Ah':
            if control['operation'] != 'current' or not control['value']:
                raise ValueError('Charge/time equivalence requires nonzero constant current')
            timed.append(3600*v/abs(control['value']))
        elif q == 'unresolved_native_charge_reference':
            # Conditioning exposure uses observed CC duration, never a capacity
            # target from a full RPT. The unknown controller reference is retained.
            if observed_s<=0:raise ValueError('Unresolved charge reference without positive observed duration')
            timed.append(observed_s)
        elif q == 'voltage_V': termination.append(f'{v:.7g} V')
        elif q == 'absolute_current_A': termination.append(f'{v:.7g} A')
        else: raise ValueError('Unsupported native endpoint')
    if not timed and not termination: raise ValueError('Unbounded native step')
    # Event-only steps have a generous watchdog; reaching it is a failure.
    duration = min(timed) if timed else max(3*observed_s+3600, 60000)
    return dict(operation=control['operation'], value=control.get('value'),
                duration_s=duration, termination=termination,
                timed_endpoint=bool(timed), native_limits=control['limits'])


def compile_inputs(root, output, gap_temperature_C=25):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    w = root/'work/validation_proposal_20261103'
    paths = {k:w/v for k,v in {
        'programs':'metadata_control_probe_v2/decoded_programs_raw.json',
        'segments':'measured_history_v2/ordered_segments.json',
        'clock':'all_rpt_clock_origins_v1/clock_mappings.csv',
        'anchors':'clock_completion_review_v1/C10_measurement_anchors.json',
        'reference':'cloud_validation_v1/reference_protocol.json'}.items()}
    hashes = {k:sha(p) for k,p in paths.items()}
    programs = json.loads(paths['programs'].read_text(encoding='utf-8-sig'))
    clocks = list(csv.DictReader(paths['clock'].open(encoding='utf-8-sig')))
    local_names = {Path(c['local_file']).name:Path(c['source_path']).name for c in clocks}
    by_name = {}
    for p in programs:
        path = root/Path(p['path'])
        if sha(path) != p['whole_file_sha256']: raise ValueError('Native source changed')
        name = local_names.get(path.name, path.name)
        if name in by_name: raise ValueError('Ambiguous native filename')
        by_name[name] = p
    with (root/'work/preprocess/out/timeline.csv').open(encoding='utf-8-sig') as f:
        timeline = {r['run_id']:r for r in csv.DictReader(f) if r['row_type']=='run'}
    for obs in timeline.values():
        if obs.get('analysis_valid')=='True':
            require_uniform_recorded_mode(obs.get('modes_present'),obs['run_id'])
    segments = json.loads(paths['segments'].read_text())['segments']
    anchors = json.loads(paths['anchors'].read_text())['measurements']
    anchor_by = {(a['cell'],Path(a['MPR_source_path']).name,a['row_start']):a for a in anchors}
    sys.path.insert(0,str(root/'work/preprocess'))
    import common
    binary_ns = {};binary_modes={}
    source_hashes = {p['path']:p['whole_file_sha256'] for p in programs}
    cells = []
    for cell in ('C','D'):
        steps, measurements = [], []
        rows = sorted((s for s in segments if s['cell']==cell and s['analysis_valid']
                       and s['absolute_time_known']),key=lambda s:s['absolute_start_naive'])
        previous = None
        unknown_count, sample_transition_count = 0,0
        for s in rows:
            start = datetime.fromisoformat(s['absolute_start_naive'])
            if previous:
                delta = (start-datetime.fromisoformat(previous['absolute_end_naive'])).total_seconds()
                if delta < -1e-6: raise ValueError('Recorded overlap')
                if delta > 0:
                    adjacent = (previous['native_source_path']==s['native_source_path']
                                and previous['row_end_exclusive']==s['row_start'])
                    # Finite sampling transitions are separately disclosed assumptions too.
                    unknown_count += not adjacent
                    sample_transition_count += adjacent
                    steps.append(dict(operation='rest',value=None,duration_s=delta,
                        termination=[],timed_endpoint=True,role='assumed_transition' if adjacent else 'assumed_gap',
                        temperature_K=273.15+gap_temperature_C,source_ref=f"{previous['segment_id']} -> {s['segment_id']}",
                        assumption='zero current; no SOC reset; fixed declared temperature',native_limits=[]))
            name = Path(s['native_source_path']).name
            p = by_name[name]
            if s['segment_id'] in timeline:
                obs = timeline[s['segment_id']]
                ns = int(obs['Ns'])
            else:
                if name not in binary_ns:
                    cols,_,_ = common.read_mpr(root/Path(p['path']))
                    binary_ns[name] = cols['ns']
                    binary_modes[name] = cols['mode']
                mode_values=set(int(v) for v in binary_modes[name][s['row_start']:s['row_end_exclusive']])
                if len(mode_values)!=1 or not mode_values <= {1,2,3}:
                    raise ValueError('Mixed or unsupported binary modes require raw subsegment review: '+s['segment_id'])
                ns = int(binary_ns[name][s['row_start']])
                obs = None
            r = p['raw_program_rows'][ns]
            if p['technique']=='MB':
                d = decode_mb(r,preserve_branch_metadata=True,sequence_count=p['sequence_count'],preserve_charge_metadata=True)
            elif p['technique']=='CP':
                # CP current has its recorded signed convention (positive charging).
                current = float(r['Is'])*(1 if r['Is_unit']==0 else .001 if r['Is_unit']==1 else float('nan'))
                if not abs(current)>0: raise ValueError('Unsupported CP current')
                d = dict(operation='current',value=-current,limits=[dict(quantity='time_s',value=float(r['ts']),comparator='greater')])
                if r['dQM']>0:
                    if r['dQM_unit'] != 1: raise ValueError('Unknown CP capacity unit')
                    d['limits'].append(dict(quantity='absolute_step_charge_Ah',value=float(r['dQM'])*.001,comparator='greater'))
                if r['EM']>0:d['limits'].append(dict(quantity='voltage_V',value=float(r['EM']),comparator='greater' if current>0 else 'less'))
            else:raise ValueError('Unsupported technique')
            expected = {'CC':'current','CV':'voltage','rest':'rest'}.get(s['control_type'])
            if expected != d['operation']:raise ValueError('Native/observed mode mismatch '+s['segment_id'])
            if d['operation']!='rest':
                recorded=float(s['control_value'])*(-.001 if d['operation']=='current' else 1)
                tolerance=1e-6 if d['operation']=='current' else 1e-4
                if abs(d['value']-recorded)>tolerance:raise ValueError('Native/observed setpoint mismatch '+s['segment_id'])
            step=native_step(d,float(s['recorded_sample_span_s']))
            step.update(role='recorded_program_route',temperature_K=298.15,source_ref=s['segment_id'],
                        native_source_sha256=p['whole_file_sha256'],native_Ns=ns,
                        recorded_start=s['absolute_start_naive'],recorded_end=s['absolute_end_naive'])
            key=(cell,name,s['row_start'])
            if key in anchor_by:
                a=anchor_by[key]
                if (d['operation']!='current' or abs(d['value']-.5)>1e-9
                    or step['termination']!=['2.5 V'] or step['timed_endpoint']):
                    raise ValueError('Full C/10 endpoint mismatch')
                measurements.append(dict(rpt_id=f"RPT{a['rpt']}",step_index=len(steps),
                    target_Qref_Ah=a['performance_summary_C10_capacity_Ah']))
                step['role']='actual_RPT_capacity'
            steps.append(step)
            previous=s
        if len(measurements)!=13 or measurements[0]['rpt_id']!='RPT0':raise ValueError('Incomplete native RPT targets')
        cells.append(dict(cell_id=cell,steps=steps,measurements=measurements,
                          unknown_intervals_assumed_rest=unknown_count,sampling_transitions_assumed_rest=sample_transition_count))
    payload=dict(schema_version=1,data_kind='conditional_observed_program_route',
        physical_validation_complete=False,measured_history_admissible=False,
        assumptions=['Observed Ns order retained; native goto/stop branches NOT independently predicted',
                     'Each observed native control executed to its model endpoint; calendar times can differ',
                     'MB charge-reference limits unresolved: those CC exposures use observed sample duration, NOT native charge interpretation',
                     'All unknown gaps and inter-sample transitions assigned zero current; no SOC reset',
                     'Recorded phases assumed isothermal 25C; gap temperature explicitly declared',
                     'Formation history unknown; initial SOC 0.2 and fixed C-only BOL inventory',
                     'No extrapolated EFC80 or accepted ground-truth labels'],
        gap_temperature_C=gap_temperature_C,source_sha256=hashes,native_source_sha256=source_hashes,
        reference=json.loads(paths['reference'].read_text()),cells=cells)
    target=output/'inputs.json'
    target.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    for c in cells:
        split={k:v for k,v in payload.items() if k!='cells'}
        split['cell']=c
        (output/f"cell_{c['cell_id']}.json").write_text(json.dumps(split,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    report=dict(status='conditional_inputs_compiled',inputs_sha256=sha(target),
                physical_validation_complete=False,measured_history_admissible=False,
                cells=[{k:v for k,v in c.items() if k not in ('steps','measurements')} |
                       dict(steps=len(c['steps']),measurements=len(c['measurements'])) for c in cells])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();print(json.dumps(compile_inputs(args.root,args.output),indent=2))
