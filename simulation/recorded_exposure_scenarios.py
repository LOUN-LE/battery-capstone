"""Compile explicit diagnostic assumptions around preserved raw exposure.

This data is deliberately not a conditional_ageing_calibration input. It cannot
admit a PDE or physical lifetime run. A policy is required for every unresolved
positive-time interval, including holes inside an observed-current block.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def duration_us(start,end):
    delta=end-start
    return delta.days*86400000000+delta.seconds*1000000+delta.microseconds


def _clock(value):
    value=datetime.fromisoformat(value)
    if value.tzinfo is not None:
        raise ValueError('Audited native clock must remain timezone-naive')
    return value


def validate_policies(policies):
    if not isinstance(policies,dict):
        raise ValueError('Explicit per-kind policy dictionary required')
    result={}
    for kind,p in policies.items():
        if not isinstance(kind,str) or not kind or not isinstance(p,dict):
            raise ValueError('Named unresolved kinds require explicit policy objects')
        if set(p)-{'policy_id','rationale','operation','temperature_C','current_A'}:
            raise ValueError('Unsupported assumption field; no hidden solver or capacity targets are accepted')
        if not isinstance(p.get('policy_id'),str) or not p['policy_id'].strip():
            raise ValueError('Each policy needs a nonempty policy_id')
        if not isinstance(p.get('rationale'),str) or not p['rationale'].strip():
            raise ValueError('Each policy needs a disclosed rationale')
        temp=p.get('temperature_C')
        if isinstance(temp,bool) or not isinstance(temp,(int,float)) or not math.isfinite(temp) or temp<=-273.15:
            raise ValueError('Each assumed operation needs a finite temperature_C above absolute zero')
        operation=p.get('operation')
        if operation=='rest':
            if 'current_A' in p and p['current_A']!=0:
                raise ValueError('A rest policy cannot prescribe nonzero current')
            current=0.
        elif operation=='constant_current':
            current=p.get('current_A')
            if isinstance(current,bool) or not isinstance(current,(int,float)) or not math.isfinite(current):
                raise ValueError('Constant-current assumptions require explicit finite current_A')
        else:
            raise ValueError('Assumption operation must be explicit rest or constant_current')
        result[kind]=dict(policy_id=p['policy_id'],rationale=p['rationale'],operation=operation,
                          current_A=float(current),temperature_C=float(temp))
    return result


def eligible_windows(t,current,mask,*,row_start,row_end_exclusive,start_s,end_s):
    """Keep complete eligible raw intervals; never clip or extend an interval."""
    import numpy as np
    t=np.asarray(t,dtype=float);current=np.asarray(current,dtype=float);mask=np.asarray(mask)
    if (t.ndim!=1 or current.shape!=t.shape or mask.dtype!=np.bool_
            or mask.shape!=(max(0,len(t)-1),)):
        raise ValueError('Paired raw arrays and boolean interval mask required')
    a,b=row_start,row_end_exclusive
    if not 0<=a<b<=len(t) or not math.isfinite(start_s) or not math.isfinite(end_s) or end_s<start_s:
        raise ValueError('Valid raw row and clock windows required')
    dt=np.diff(t[a:b]);local=mask[a:b-1]
    if np.any(~np.isfinite(t[a:b])) or np.any(local & ((dt<=0)|(dt>60.00001))):
        raise ValueError('Eligible current requires dense monotonic source intervals')
    if np.any(local & (~np.isfinite(current[a:b-1])|~np.isfinite(current[a+1:b]))):
        raise ValueError('Eligible raw current must be finite')
    use=local&(t[a:b-1]>=start_s)&(t[a+1:b]<=end_s)
    indices=np.flatnonzero(use)+a
    if not len(indices):return []
    cuts=np.flatnonzero(np.diff(indices)!=1)+1
    chunks=np.split(indices,cuts)
    windows=[]
    for c in chunks:
        left,right=int(c[0]),int(c[-1]+1)
        q=float(np.sum((current[left:right]+current[left+1:right+1])*.5*np.diff(t[left:right+1]))/3600000)
        windows.append(dict(row_start=left,row_end_exclusive=right+1,start_s=float(t[left]),end_s=float(t[right]),
                            signed_native_current_Ah=q,eligible_interval_count=right-left))
    return windows


def compile_scenario(inputs,policies,*,scenario_id):
    import numpy as np
    path=Path(inputs).resolve()
    original_sha=sha(path)
    payload=json.loads(path.read_text(encoding='utf-8-sig'))
    if (payload.get('schema_version')!=2 or payload.get('data_kind')!='recorded_exposure_reconstruction'
            or payload.get('pde_replay_admissible') is not False
            or payload.get('physical_validation_complete') is not False):
        raise ValueError('PDE-inadmissible raw reconstruction schema2 required')
    integrity=payload.get('exposure_integrity',{})
    if not all(integrity.get(k) is True for k in ('raw_mode_boundaries_preserved','internal_intervals_accounted',
               'duration_reconciliation_passed','source_rows_preserved','assumed_intervals_disclosed')):
        raise ValueError('Raw reconstruction integrity evidence required')
    if not isinstance(scenario_id,str) or not scenario_id.strip():
        raise ValueError('Explicit scenario_id required')
    policies=validate_policies(policies)
    sources={s['source_id']:s for s in payload['cell']['source_files']}
    if len(sources)!=len(payload['cell']['source_files']):raise ValueError('Duplicate source_id')
    cache=OrderedDict();receipts={};result=[];assumptions=[];inherited=[]
    ledger=Counter();unknown_by_kind=Counter();original_unknown_by_kind=Counter();counts=Counter()
    def arrays(source_id):
        if source_id in cache:
            cache.move_to_end(source_id)
            return cache[source_id]
        source=sources[source_id];raw_path=(path.parent/source['arrays_file']).resolve()
        if not raw_path.is_relative_to(path.parent):raise ValueError('Raw arrays escaped artifact directory')
        if sha(raw_path)!=source['arrays_sha256']:raise ValueError('Raw array hash changed')
        with np.load(raw_path,allow_pickle=False) as archive:
            values={k:archive[k] for k in ('t_s','I_mA','mode','Ns','current_drive_eligible_interval')}
        t,I,m,n,mask=[values[k] for k in ('t_s','I_mA','mode','Ns','current_drive_eligible_interval')]
        if (t.ndim!=1 or len(t)!=source['rows'] or any(v.shape!=t.shape for v in (I,m,n))
                or mask.dtype!=np.bool_ or mask.shape!=(max(0,len(t)-1),)):
            raise ValueError('Lossless source shape mismatch')
        if np.any(mask & ((m[:-1]!=m[1:])|~np.isin(m[:-1],[1,2])|(n[:-1]!=n[1:]))):
            raise ValueError('Eligible mask crosses native mode/Ns boundary')
        if sha(raw_path)!=source['arrays_sha256']:raise ValueError('Raw array changed while loading')
        receipts[source_id]=dict(arrays_file=source['arrays_file'],arrays_sha256=source['arrays_sha256'],
            native_sha256=source['native_sha256'])
        cache[source_id]=values
        if len(cache)>2:cache.popitem(last=False)
        return values
    def base(s,index,start,end):
        us=duration_us(start,end)
        if us<0:raise ValueError('Negative scenario interval')
        return dict(raw_step_index=index,source_ref=s.get('source_ref'),raw_kind=s['kind'],
                    absolute_start_naive=start.isoformat(),absolute_end_naive=end.isoformat(),duration_us=us)
    def assumed(s,index,start,end,kind):
        us=duration_us(start,end)
        if not us:return
        if kind not in policies:raise ValueError('Missing explicit unresolved policy for '+kind)
        policy=policies[kind]
        item=base(s,index,start,end)
        item.update(kind='assumed_exposure',unresolved_kind=kind,assumption_id=f'A{len(assumptions)+1:06d}',
                    policy_id=policy['policy_id'],assumed_operation=policy['operation'],
                    assumed_current_A=policy['current_A'],assumed_temperature_C=policy['temperature_C'],
                    policy_sha256=canonical_sha(policy),measured_operation_claim=False)
        result.append(item)
        assumptions.append(dict(item,rationale=policy['rationale']))
        ledger['assumed_unresolved_us']+=us;unknown_by_kind[kind]+=us;counts[kind]+=1
    previous_end=None
    for index,s in enumerate(payload['cell']['steps']):
        start,end=_clock(s['absolute_start_naive']),_clock(s['absolute_end_naive'])
        us=duration_us(start,end)
        if us!=s['duration_us'] or us<0:raise ValueError('Raw integer duration changed')
        if us:
            if previous_end is not None and start!=previous_end:raise ValueError('Raw clock has an overlap or unaccounted interval')
            previous_end=end
        ledger['raw_total_duration_us']+=us
        if s['operation'] is None:
            original_unknown_by_kind[s['kind']]+=us
            if us:assumed(s,index,start,end,s['kind'])
            else:result.append(dict(base(s,index,start,end),kind='zero_span_raw_observation',operation=None))
            continue
        source=sources.get(s.get('source_id'))
        if (source is None or source['arrays_file']!=s.get('arrays_file')
                or source['native_sha256']!=s.get('raw_source_sha256')):
            raise ValueError('Raw step/source binding mismatch')
        data=arrays(s['source_id']);a,b=s['row_start'],s['row_end_exclusive']
        if not 0<=a<b<=source['rows']:raise ValueError('Raw step rows outside source')
        reference=dict(source_id=s['source_id'],arrays_file=s['arrays_file'],
            arrays_sha256=source['arrays_sha256'],raw_source_sha256=s['raw_source_sha256'])
        if s['operation']=='rest':
            if (s.get('rest_current_A')!=0 or s.get('raw_current_interpolation_permitted') is not False
                    or np.any(data['mode'][a:b]!=3)):
                raise ValueError('Recorded rest must retain mode3 and forbid raw-current interpolation')
            item=dict(base(s,index,start,end),kind='recorded_rest_controller',recorded_rest_current_A=0.,
                row_start=a,row_end_exclusive=b,raw_current_interpolation_permitted=False,
                raw_temperature_samples_preserved=True,temperature_continuity_assumed=False,
                **reference)
            result.append(item);ledger['recorded_rest_controller_us']+=us
            if s['kind']=='sparse_rest_controller_interval' or s.get('rest_current_sample_conflict_count',0):
                inherited.append(dict(raw_step_index=index,source_ref=s['source_ref'],duration_us=us,
                    interpretation='recorded mode3 rest controller; not continuous measured zero current',
                    continuity_policy=s.get('continuity_policy'),raw_current_conflicts_retained=True))
        elif s['operation']=='recorded_current_drive':
            windows=eligible_windows(data['t_s'],data['I_mA'],data['current_drive_eligible_interval'],
                row_start=a,row_end_exclusive=b,start_s=s['start_s'],end_s=s['end_s'])
            origin=_clock(source['timestamp_naive']);cursor=start
            for window in windows:
                left=origin+timedelta(seconds=window['start_s']);right=origin+timedelta(seconds=window['end_s'])
                if left<cursor or right>end:raise ValueError('Eligible window exceeds canonical raw clock')
                assumed(s,index,cursor,left,'noneligible_current_sample_interval')
                item=dict(base(s,index,left,right),kind='eligible_recorded_current',
                    native_positive_current='charge',model_current_sign_conversion_pending=True,
                    raw_temperature_samples_preserved=True,temperature_continuity_assumed=False,
                    capacity_target_reinitialised=False,**reference,**window)
                result.append(item);ledger['eligible_recorded_current_us']+=item['duration_us']
                cursor=right
            assumed(s,index,cursor,end,'noneligible_current_sample_interval')
        else:raise ValueError('Unsupported raw operation')
    if sha(path)!=original_sha:raise ValueError('Raw reconstruction changed while compiling')
    # Include even sources with only unresolved/zero observations in the hash chain.
    for source_id in sources:
        if source_id not in receipts:arrays(source_id)
    if sum(ledger[k] for k in ('recorded_rest_controller_us','eligible_recorded_current_us','assumed_unresolved_us'))!=ledger['raw_total_duration_us']:
        raise ValueError('Scenario integer-time ledger failed')
    if sum(s['duration_us'] for s in result)!=ledger['raw_total_duration_us']:
        raise ValueError('Compiled scenario lost or duplicated time')
    used=set(unknown_by_kind)
    return dict(schema_version=1,data_kind='diagnostic_exposure_scenario',scenario_id=scenario_id,
        pde_replay_admissible=False,physical_validation_complete=False,measured_history_admissible=False,
        conditional_ageing_input_compatible=False,programme_Q_and_branch_admission=False,
        compiler_sha256=sha(__file__),raw_reconstruction_sha256=original_sha,raw_reconstruction_path=str(path),
        cell_id=payload['cell']['cell_id'],policies=policies,policies_sha256=canonical_sha(policies),
        unused_policy_kinds=sorted(set(policies)-used),source_array_receipts=receipts,
        time_ledger_us=dict(ledger),assumed_unresolved_by_kind_us=dict(unknown_by_kind),
        assumed_interval_count_by_kind=dict(counts),original_unknown_by_kind_us=dict(original_unknown_by_kind),
        assumption_inventory=assumptions,inherited_controller_interpretations=inherited,steps=result,
        all_time_accounted_once=True,raw_current_waveform_preserved=True,
        temperature_scope='native T samples retained for recorded exposure; policies apply only to explicitly unresolved time',
        diagnostic_scope='compile and exact time ledger only; no PDE, calibration, SOH prediction or physical pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--policies',type=Path,required=True)
    parser.add_argument('--scenario-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Existing diagnostic must be preserved')
    result=compile_scenario(args.inputs,json.loads(args.policies.read_text(encoding='utf-8-sig')),scenario_id=args.scenario_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','assumption_inventory','source_array_receipts','inherited_controller_interpretations')},indent=2))
