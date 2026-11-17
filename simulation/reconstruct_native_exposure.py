"""Lossless raw-mode exposure reconstruction, without inventing a controller.

Every positive interval is accounted once. Recorded rest-controller time is
preserved, even when Ns is unchanged, and stale rest-entry current is never
linearly integrated across a one-hour sampling interval. The resulting schema2
is an evidence/drive source, NOT admission to a physical lifetime replay.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def partition_run(times,modes,ns,current_mA,control=None,*,max_sample_gap_s=60.):
    """Return blocks and explicit boundary/sparse intervals covering one run.

    Mode3 means a recorded rest controller, not a measured current waveform.
    Rest sample conflicts remain visible. Large same-mode1 intervals are unknown
    current exposure and are never filled with the nominal current.
    """
    import numpy as np
    t=np.asarray(times,dtype=float);m=np.asarray(modes);n=np.asarray(ns);I=np.asarray(current_mA,dtype=float)
    if t.ndim!=1 or not len(t) or any(np.asarray(v).shape!=t.shape for v in (m,n,I)):
        raise ValueError('Paired one-dimensional raw sample arrays required')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(m)) or not np.all(np.isfinite(n)):
        raise ValueError('Finite raw clock/mode/Ns required')
    if np.any(m!=np.floor(m)) or np.any(n!=np.floor(n)):raise ValueError('Integer raw mode/Ns required')
    if isinstance(max_sample_gap_s,bool) or not math.isfinite(max_sample_gap_s) or max_sample_gap_s<=0:
        raise ValueError('Finite positive sampling review threshold required')
    if control is None:c=None
    else:
        c=np.asarray(control,dtype=float)
        if c.shape!=t.shape:raise ValueError('Control/raw row shape mismatch')
    dt=np.diff(t);mode_change=m[1:]!=m[:-1];ns_change=n[1:]!=n[:-1]
    cut=mode_change|ns_change|(dt<=0)|(dt>max_sample_gap_s)
    starts=np.r_[0,np.flatnonzero(cut)+1];ends=np.r_[starts[1:],len(t)]
    eligible=np.zeros(max(0,len(t)-1),dtype=bool)
    finite_current=np.isfinite(I)
    # Quantitative drive interpolation is limited to observed, continuous,
    # same-controller intervals; opposite-sign CC entry samples remain unresolved.
    if len(t)>1:
        eligible=(~cut)&np.isin(m[:-1],[1,2])&finite_current[:-1]&finite_current[1:]
        if c is not None:
            cc=m==1
            conflict=cc&np.isfinite(c)&(c!=0)&(I*c<=0)
            eligible&=~conflict[:-1]&~conflict[1:]
    blocks=[];intervals=[];clock_breaks=[]
    for a,b in zip(starts.tolist(),ends.tolist()):
        mode=int(m[a]);rest=mode==3
        keep=eligible[a:b-1]
        q=float(np.sum((I[a:b-1][keep]+I[a+1:b][keep])*.5*dt[a:b-1][keep])/3600000) if b-a>1 else 0.
        # No current integral is asserted for rest: its entry I may be stale.
        blocks.append(dict(kind='recorded_mode_block',row_start=a,row_end_exclusive=b,recorded_mode=mode,native_Ns=int(n[a]),
            start_s=float(t[a]),end_s=float(t[b-1]),duration_s=float(t[b-1]-t[a]),
            controller_label={1:'CC',2:'CV',3:'rest'}.get(mode,'unsupported'),
            operation='rest' if rest else 'recorded_current_drive' if mode in (1,2) else None,
            rest_current_A=0. if rest else None,
            raw_current_interpolation_permitted=mode in (1,2),
            current_drive_valid_interval_count=int(np.sum(eligible[a:b-1])),
            current_drive_unresolved_interval_count=int(max(0,b-a-1)-np.sum(eligible[a:b-1])) if mode in (1,2) else 0,
            recorded_current_integral_Ah_valid_intervals=q if mode in (1,2) else None,
            rest_current_sample_conflict_count=int(np.sum(~finite_current[a:b]|(abs(I[a:b])>1.))) if rest else 0,
            current_sample_conflict_threshold_mA=1. if rest else None))
        if b<len(t):
            j=b-1;delta=float(dt[j])
            if delta<=0:
                clock_breaks.append(dict(left_row=j,right_row=b,left_time_s=float(t[j]),right_time_s=float(t[b]),dt_s=delta))
                continue
            same_rest=int(m[j])==int(m[b])==3 and int(n[j])==int(n[b])
            sparse=delta>max_sample_gap_s
            label='sparse_rest_controller_interval' if sparse and same_rest else 'unresolved_sparse_current_interval' if sparse else 'controller_transition_interval'
            conflict=bool(not np.isfinite(I[j]) or not np.isfinite(I[b]) or abs(I[j])>1 or abs(I[b])>1) if same_rest else None
            intervals.append(dict(kind=label,row_start=j,row_end_exclusive=b+1,left_row=j,right_row=b,
                start_s=float(t[j]),end_s=float(t[b]),duration_s=delta,
                left_mode=int(m[j]),right_mode=int(m[b]),left_Ns=int(n[j]),right_Ns=int(n[b]),
                operation='rest' if same_rest else None,rest_current_A=0. if same_rest else None,
                raw_current_interpolation_permitted=False,recorded_current_integral_Ah=None,
                recorded_mode3_current_conflict=conflict,
                continuity_policy='recorded mode3 controller held between sparse samples; raw current conflict retained' if same_rest else 'unresolved; no current/rest inserted',
                fully_observed_continuous_current=False))
    accounted=math.fsum(b['duration_s'] for b in blocks)+math.fsum(g['duration_s'] for g in intervals)
    expected=float(t[-1]-t[0])
    reconciled=not clock_breaks and math.isclose(accounted,expected,rel_tol=0,abs_tol=1e-6)
    return dict(blocks=blocks,intervals=intervals,clock_breaks=clock_breaks,
                current_drive_eligible_intervals=eligible,
                source_span_s=expected,accounted_span_s=accounted,duration_reconciliation_passed=reconciled,
                full_physical_replay_admissible=False)


def _naive(value):
    t=datetime.fromisoformat(value)
    if t.tzinfo is not None:raise ValueError('Native instrument clock must remain naive')
    return t


def reconcile_clock_episodes(partition,source_start_s,source_end_s):
    """Account overlapping clock episodes once, without choosing raw values.

    The reporting envelope remains first-to-last audited sample time. Regions
    covered by two positive-time clock episodes are explicitly ambiguous; both
    row ranges remain referenced, and no current is assigned to the overlap.
    """
    import numpy as np
    original=partition['blocks']+partition['intervals']
    if source_end_s<source_start_s:raise ValueError('A reversed source envelope has no admitted physical clock')
    if any(p['start_s']<source_start_s-1e-8 or p['end_s']>source_end_s+1e-8 for p in original):
        raise ValueError('Clock episode extends outside audited source envelope; manual clock review required')
    edges=sorted({source_start_s,source_end_s}|{p['start_s'] for p in original}|{p['end_s'] for p in original})
    parts=[];ambiguous=[]
    for left,right in zip(edges,edges[1:]):
        if right<=left:continue
        active=[p for p in original if p['start_s']<=left and p['end_s']>=right and p['end_s']>p['start_s']]
        if len(active)==1:
            p=dict(active[0]);p.update(start_s=left,end_s=right,duration_s=right-left,
                raw_part_start_s=active[0]['start_s'],raw_part_end_s=active[0]['end_s'],
                time_window_clip_required=left!=active[0]['start_s'] or right!=active[0]['end_s'])
        else:
            references=[{k:a[k] for k in ('kind','row_start','row_end_exclusive','start_s','end_s')} for a in active]
            p=dict(kind='ambiguous_clock_overlap_interval' if len(active)>1 else 'unresolved_clock_interval',
                start_s=left,end_s=right,duration_s=right-left,operation=None,raw_current_interpolation_permitted=False,
                row_start=min(a['row_start'] for a in active) if active else 0,
                row_end_exclusive=max(a['row_end_exclusive'] for a in active) if active else 0,
                overlapping_raw_parts=references,current_drive_assigned=False)
            ambiguous.append(p)
        parts.append(p)
    for p in original:
        if p['start_s']==p['end_s']:
            parts.append(dict(p,kind='zero_span_raw_observation',operation=None,raw_current_interpolation_permitted=False))
    parts.sort(key=lambda p:(p['start_s'],p['end_s'],p['row_start']))
    eligible=partition['current_drive_eligible_intervals'].copy()
    # The caller retains all raw arrays. It may use only eligible intervals
    # after excluding overlapping episodes and duplicate/reversed boundary points.
    return dict(parts=parts,ambiguous_intervals=ambiguous,current_drive_eligible_intervals=eligible,
        accounted_span_s=math.fsum(p['duration_s'] for p in parts),
        duration_reconciliation_passed=math.isclose(math.fsum(p['duration_s'] for p in parts),source_end_s-source_start_s,rel_tol=0,abs_tol=1e-6),
        clock_breaks_retained=partition['clock_breaks'])


def _microseconds(start,end):
    d=end-start
    return d.days*86400000000+d.seconds*1000000+d.microseconds


def reconstruct(root,output,*,max_sample_gap_s=60.):
    import numpy as np
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    raw_out=out/'raw_sources';raw_out.mkdir()
    w=root/'work/validation_proposal_20261103'
    paths=dict(segments=w/'measured_history_v2/ordered_segments.json',programs=w/'metadata_control_probe_v2/decoded_programs_raw.json',
               clock=w/'all_rpt_clock_origins_v1/clock_mappings.csv',anchors=w/'clock_completion_review_v1/C10_measurement_anchors.json',
               legacy_intervals=w/'measured_history_v2/coverage_intervals.csv')
    initial_hashes={k:sha(p) for k,p in paths.items()}
    segments=json.loads(paths['segments'].read_text(encoding='utf-8-sig'))['segments']
    programs=json.loads(paths['programs'].read_text(encoding='utf-8-sig'))
    with paths['clock'].open(encoding='utf-8-sig',newline='') as f:clocks=list(csv.DictReader(f))
    aliases={Path(c['local_file']).name:Path(c['source_path']).name for c in clocks}
    by_name={}
    for p in programs:
        path=root/p['path'];name=aliases.get(path.name,path.name)
        if name in by_name:raise ValueError('Ambiguous raw source filename')
        by_name[name]=p
    by_file=defaultdict(list)
    for s in segments:by_file[Path(s['native_source_path']).name].append(s)
    sys.path.insert(0,str(root/'work/preprocess'));import common
    compiled=[];sources=[];quarantines=[];run_checks=[];mixed_checks=[]
    all_clock_breaks=[];clock_reconciliation=[]
    for file_index,(name,runs) in enumerate(sorted(by_file.items())):
        p=by_name[name];path=root/p['path']
        if sha(path)!=p['whole_file_sha256']:raise ValueError('Raw MPR changed '+name)
        cols,meta,stamp=common.read_mpr(path)
        if stamp is None or stamp.tzinfo is not None:raise ValueError('Native timestamp missing or not naive')
        t=np.asarray(cols['t'],dtype=float);m=np.asarray(cols['mode']);ns=np.asarray(cols['ns']);I=np.asarray(cols['I'],dtype=float)
        ctrl=cols['ctrl_mixed'] if cols['ctrl_mixed'] is not None else cols['ctrl_mA']
        coverage=np.zeros(len(t),dtype=np.uint8);valid=np.zeros(len(t),dtype=bool);eligible=np.zeros(max(0,len(t)-1),dtype=bool)
        arrays=dict(t_s=cols['t'],mode=cols['mode'],Ns=cols['ns'],I_mA=cols['I'])
        for key,source in [('V','V_V'),('T','T_C'),('dq','dq_mAh'),('qqo','Q_minus_Q0_mAh'),('ctrl_mixed','control_V_or_mA'),('ctrl_mA','control_mA'),('ctrl_V','control_V')]:
            if cols.get(key) is not None:arrays[source]=cols[key]
        relative_npz=f'raw_sources/raw_{file_index:03d}.npz'
        source_entry=dict(source_id=f'raw_{file_index:03d}',native_source_name=name,native_path=p['path'],native_sha256=p['whole_file_sha256'],
                          rows=len(t),timestamp_naive=stamp.isoformat(),arrays_file=relative_npz,columns_used=meta['columns_used'])
        for s in sorted(runs,key=lambda s:s['row_start']):
            a,b=s['row_start'],s['row_end_exclusive'];coverage[a:b]+=1
            if not s['analysis_valid'] or not s['absolute_time_known']:
                quarantines.append(dict(source_ref=s['segment_id'],source_id=source_entry['source_id'],row_start=a,row_end_exclusive=b,
                    raw_time_min_s=float(np.min(t[a:b])),raw_time_max_s=float(np.max(t[a:b])),
                    recorded_modes=sorted(set(int(v) for v in m[a:b])),physical_duration_assigned=False,issues=s['issues']))
                continue
            valid[a:b]=True
            first=stamp+timedelta(seconds=float(t[a]));last=stamp+timedelta(seconds=float(t[b-1]))
            if (abs((first-_naive(s['absolute_start_naive'])).total_seconds())>2e-6
                or abs((last-_naive(s['absolute_end_naive'])).total_seconds())>2e-6):
                raise ValueError('Native clock no longer matches audited origin '+s['segment_id'])
            split=partition_run(t[a:b],m[a:b],ns[a:b],I[a:b],None if ctrl is None else ctrl[a:b],max_sample_gap_s=max_sample_gap_s)
            if split['clock_breaks']:
                resolved=reconcile_clock_episodes(split,float(t[a]),float(t[b-1]))
                raw_parts=resolved['parts'];local_eligible=split['current_drive_eligible_intervals'].copy()
                for interval in resolved['ambiguous_intervals']:
                    local_eligible&=~((t[a:b-1]<interval['end_s'])&(t[a+1:b]>interval['start_s']))
                for boundary in split['clock_breaks']:
                    for instant in (boundary['left_time_s'],boundary['right_time_s']):
                        local_eligible&=~((t[a:b-1]==instant)|(t[a+1:b]==instant))
                clock_reconciliation.append(dict(source_ref=s['segment_id'],cell=s['cell'],clock_breaks=split['clock_breaks'],
                    ambiguous_overlap_duration_s=sum(g['duration_s'] for g in resolved['ambiguous_intervals']),
                    source_span_s=split['source_span_s'],accounted_span_s=resolved['accounted_span_s'],
                    duration_reconciliation_passed=resolved['duration_reconciliation_passed'],raw_values_chosen_or_deleted=False))
                reconciled=resolved['duration_reconciliation_passed']
            else:
                raw_parts=split['blocks']+split['intervals'];local_eligible=split['current_drive_eligible_intervals'];reconciled=split['duration_reconciliation_passed']
            eligible[a:b-1]=local_eligible
            items=[]
            for part in raw_parts:
                item={k:v for k,v in part.items()}
                if part['operation']=='recorded_current_drive':
                    c,d=part['row_start'],part['row_end_exclusive']
                    take=local_eligible[c:d-1].copy()
                    in_window=(t[a+c:a+d-1]>=part['start_s'])&(t[a+c+1:a+d]<=part['end_s'])
                    take&=in_window
                    durations=np.diff(t[a+c:a+d])[take]
                    left=I[a+c:a+d-1][take];right=I[a+c+1:a+d][take]
                    valid_duration=float(np.sum(durations))
                    item.update(recorded_current_integral_Ah_valid_intervals=float(np.sum((left+right)*.5*durations)/3600000),
                        current_drive_valid_interval_count=int(np.sum(take)),
                        current_drive_unresolved_interval_count=int(np.sum(in_window)-np.sum(take)),
                        current_drive_valid_duration_s=valid_duration,
                        current_drive_unresolved_duration_s=max(0.,part['duration_s']-valid_duration))
                absolute_start=stamp+timedelta(seconds=part['start_s']);absolute_end=stamp+timedelta(seconds=part['end_s'])
                item.update(source_ref=s['segment_id'],source_id=source_entry['source_id'],source_kind=s['kind'],phase=s['phase'],group=s['group'],
                            row_start=a+part['row_start'],row_end_exclusive=a+part['row_end_exclusive'],
                            absolute_start_naive=absolute_start.isoformat(),absolute_end_naive=absolute_end.isoformat(),
                            duration_us=_microseconds(absolute_start,absolute_end),arrays_file=relative_npz,
                            raw_source_sha256=p['whole_file_sha256'],programmed_control_admission=False)
                if 'left_row' in item:item['left_row']+=a;item['right_row']+=a
                items.append(item)
            items.sort(key=lambda i:(i['absolute_start_naive'],i['absolute_end_naive'],i['row_start']))
            expected_us=_microseconds(first,last);sum_us=sum(i['duration_us'] for i in items)
            positive_items=[i for i in items if i['duration_us']>0]
            overlap=any(_naive(left['absolute_end_naive'])>_naive(right['absolute_start_naive']) for left,right in zip(positive_items,positive_items[1:]))
            passed=reconciled and sum_us==expected_us and not overlap
            check=dict(source_ref=s['segment_id'],cell=s['cell'],source_sample_span_s=split['source_span_s'],source_span_us=expected_us,
                       partition_sum_us=sum_us,overlap_count=int(overlap),passed=passed)
            run_checks.append(check)
            if not passed:raise ValueError('Raw run duration reconciliation failed '+s['segment_id'])
            modes=sorted(set(int(v) for v in m[a:b]))
            if len(modes)>1:
                rest_us=sum(i['duration_us'] for i in items if i['operation']=='rest')
                mixed_checks.append(dict(source_ref=s['segment_id'],cell=s['cell'],native_modes=modes,source_span_us=expected_us,
                    partition_sum_us=sum_us,recorded_rest_controller_us=rest_us,rest_restored=rest_us>0,passed=passed,
                    intervals=[{k:i[k] for k in ('kind','row_start','row_end_exclusive','duration_us','operation')} for i in items]))
            compiled.append(dict(cell=s['cell'],source_ref=s['segment_id'],native_source_name=name,source_id=source_entry['source_id'],
                row_start=a,row_end_exclusive=b,absolute_start_naive=first.isoformat(),absolute_end_naive=last.isoformat(),items=items))
        if np.any(coverage!=1):raise ValueError('Native row coverage lost or duplicated '+name)
        arrays.update(analysis_valid_row=valid,current_drive_eligible_interval=eligible)
        np.savez_compressed(out/relative_npz,**arrays)
        source_entry['arrays_sha256']=sha(out/relative_npz);sources.append(source_entry)
        for k in np.flatnonzero(np.diff(t)<=0):
            all_clock_breaks.append(dict(source_id=source_entry['source_id'],left_row=int(k),right_row=int(k+1),
                left_time_s=float(t[k]),right_time_s=float(t[k+1]),left_valid=bool(valid[k]),right_valid=bool(valid[k+1])))
    if initial_hashes!={k:sha(p) for k,p in paths.items()}:raise ValueError('Source metadata changed during reconstruction')
    with paths['legacy_intervals'].open(encoding='utf-8-sig',newline='') as f:legacy=list(csv.DictReader(f))
    expected_unknown={(r['cell'],r['from_segment'],r['to_segment']):r for r in legacy if r['kind']=='unobserved_gap'}
    anchors=json.loads(paths['anchors'].read_text(encoding='utf-8-sig'))['measurements']
    cells={};summary={};unknown_keys=set()
    for cell in ('C','D'):
        runs=sorted((r for r in compiled if r['cell']==cell),key=lambda r:(r['absolute_start_naive'],r['absolute_end_naive'],r['source_ref']))
        steps=[];unknown=[];boundaries=[];previous=None
        for r in runs:
            if previous is not None:
                end=_naive(previous['absolute_end_naive']);start=_naive(r['absolute_start_naive']);gap_us=_microseconds(end,start)
                if gap_us<0:raise ValueError('Cross-run overlap remains in reconstruction')
                if gap_us>0:
                    adjacent=previous['native_source_name']==r['native_source_name'] and previous['row_end_exclusive']==r['row_start']
                    item=dict(kind='Ns_transition_interval' if adjacent else 'unobserved_gap',operation=None,duration_us=gap_us,duration_s=gap_us/1e6,
                        absolute_start_naive=end.isoformat(),absolute_end_naive=start.isoformat(),source_ref=f"{previous['source_ref']} -> {r['source_ref']}",
                        from_segment=previous['source_ref'],to_segment=r['source_ref'],raw_current_interpolation_permitted=False,
                        assumption=None,physical_operation_between_samples=None)
                    if adjacent:boundaries.append(item)
                    else:
                        key=(cell,previous['source_ref'],r['source_ref']);unknown_keys.add(key)
                        if key not in expected_unknown:raise ValueError('Unexpected inter-run unknown interval')
                        if abs(gap_us/1e6-float(expected_unknown[key]['duration_s']))>2e-6:raise ValueError('Unknown interval duration changed')
                        unknown.append(item)
                    steps.append(item)
            steps.extend(r['items']);previous=r
        total_us=sum(s['duration_us'] for s in steps)
        expected_us=_microseconds(_naive(runs[0]['absolute_start_naive']),_naive(runs[-1]['absolute_end_naive']))
        positive_steps=[s for s in steps if s['duration_us']>0]
        overlap=sum(_naive(a['absolute_end_naive'])>_naive(b['absolute_start_naive']) for a,b in zip(positive_steps,positive_steps[1:]))
        if total_us!=expected_us or overlap:raise ValueError('Cell clock reconciliation failed '+cell)
        mixed=[c for c in mixed_checks if c['cell']==cell]
        payload=dict(schema_version=2,data_kind='recorded_exposure_reconstruction',physical_validation_complete=False,
            measured_history_admissible=False,pde_replay_admissible=False,programmed_control_admission=False,
            exposure_integrity=dict(raw_mode_boundaries_preserved=True,internal_intervals_accounted=True,
                duration_reconciliation_passed=True,source_rows_preserved=True,assumed_intervals_disclosed=True),
            unresolved=['163 original inter-file/internal unknown intervals remain unknown','MB programmed capacity reference and branching remain unadmitted',
                        'Rest/current transition sample conflicts remain visible','Sparse sample intervals do not prove every continuous current value'],
            cell=dict(cell_id=cell,experiment='Expt2',clock_basis='naive audited instrument clock; no timezone/DST conversion',
                steps=steps,measurements=[a for a in anchors if a['cell']==cell],source_files=[s for s in sources if any(r['source_id']==s['source_id'] for r in runs)]),
            quarantine=[q for q in quarantines if any(r['source_id']==q['source_id'] for r in runs)],
            source_metadata_sha256=initial_hashes)
        write(out/f'cell_{cell}.json',payload);cells[cell]=payload
        summary[cell]=dict(step_count=len(steps),source_runs=len(runs),total_source_clock_span_us=expected_us,total_reconstructed_duration_us=total_us,
            duration_reconciliation_passed=True,overlap_count=overlap,unobserved_gap_count=len(unknown),unobserved_gap_sum_s=sum(g['duration_us'] for g in unknown)/1e6,
            uncertain_Ns_transition_count=len(boundaries),mixed_mode_source_runs=len(mixed),
            restored_rest_inside_mixed_Ns_s=sum(m['recorded_rest_controller_us'] for m in mixed)/1e6,
            recorded_rest_controller_total_s=sum(s['duration_us'] for s in steps if s['operation']=='rest')/1e6,
            sparse_rest_controller_interval_count=sum(s['kind']=='sparse_rest_controller_interval' for s in steps),
            unresolved_sparse_current_interval_count=sum(s['kind']=='unresolved_sparse_current_interval' for s in steps))
    if unknown_keys!=set(expected_unknown):raise ValueError('Original163 unknown intervals changed or lost')
    if len(mixed_checks)!=16 or not all(c['passed'] and c['rest_restored'] for c in mixed_checks):raise ValueError('Expected16 mixed-mode sources not restored')
    write(out/'source_manifest.json',dict(raw_sources=sources,metadata_sha256=initial_hashes,quarantined_records=quarantines,raw_clock_breaks=all_clock_breaks))
    write(out/'mixed_mode_reconciliation.json',dict(sources=mixed_checks,all16_pass=True))
    write(out/'run_duration_reconciliation.json',dict(run_checks=run_checks,all_pass=all(c['passed'] for c in run_checks)))
    write(out/'clock_anomaly_reconciliation.json',dict(sources=clock_reconciliation,raw_clock_breaks_retained=True,
        overlapping_episode_time_never_double_counted=True,all_pass=all(c['duration_reconciliation_passed'] for c in clock_reconciliation)))
    report=dict(status='raw_mode_clock_reconstruction_completed',schema_version=2,source_files=len(sources),source_runs=len(run_checks),
        cells=summary,original_unknown_intervals_preserved=len(unknown_keys),mixed_mode_source_runs_restored=len(mixed_checks),
        quarantined_raw_rows=sum(q['row_end_exclusive']-q['row_start'] for q in quarantines),raw_clock_break_count=len(all_clock_breaks),
        duration_reconciliation_all_pass=True,overlap_count=0,raw_current_waveform_preserved_losslessly=True,
        mode3_stale_current_not_linearly_integrated=True,physical_validation_complete=False,pde_replay_admissible=False,
        programmed_Q_and_branching_admitted=False,source_sha256=initial_hashes,compiler_sha256=sha(Path(__file__)))
    write(out/'report.json',report)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--max-sample-gap-s',type=float,default=60.);args=a.parse_args()
    print(json.dumps(reconstruct(args.root,args.output,max_sample_gap_s=args.max_sample_gap_s),indent=2))
