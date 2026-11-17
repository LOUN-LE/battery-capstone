"""Trace batteryDAT's public pulse observation rules without installing it.

The operator is a numerical mirror of the inspected main-branch functions,
not evidence of the unpublished script/version used to build a study summary.
Raw arrays are not modified or rebased. Nonfinite upstream results stay failed.
"""
from __future__ import annotations
import math


def observed_resistance_rows(time_s,voltage,current_mA,ns_changes,*,timestep=None):
    """Trace r0_calc_dis or resist_time on a default positional input index.

    The open time window and its first file-wide match are intentional. They
    can select a preceding zero-current row at timestep=.1; no correction is
    silently applied. Original floating dtypes of V/I are retained.
    """
    import numpy as np
    t=np.asarray(time_s);v=np.asarray(voltage);current=np.asarray(current_mA);flags=np.asarray(ns_changes)
    if t.ndim!=1 or any(a.shape!=t.shape for a in (v,current,flags)) or not len(t):
        raise ValueError('Paired raw one-dimensional pulse arrays required')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(current)):
        raise ValueError('Finite raw samples required for this diagnostic')
    if timestep is not None and (isinstance(timestep,bool) or not math.isfinite(timestep) or timestep<=0):
        raise ValueError('Positive explicit observation timestep required')
    candidates=np.flatnonzero((flags==1)&(current<0))
    result=[]
    for ordinal,flag_row in enumerate(candidates,1):
        row=int(flag_row)
        if row<10 or row+1>=len(t):raise ValueError('Pulse lacks the upstream baseline/forward sample context')
        shifted=bool(abs(current[row-1]-current[row])<abs(current[row-2]-current[row-1]))
        row-=int(shifted)
        ocv=v[row-10:row-1].mean()
        entry=dict(pulse_one_based=ordinal,pulse_zero_based=ordinal-1,flag_row=int(flag_row),
            adjusted_row=row,row_shifted_left=shifted,baseline_row_start=row-10,baseline_row_end_exclusive=row-1,
            baseline_sample_count=9,baseline_voltage_V=float(ocv),prior_current_mA=float(current[row-1]),
            operation='r0_calc_dis' if timestep is None else 'resist_time',timestep_argument_s=timestep)
        if timestep is None:end=row
        else:
            origin=t[row-1];spacing=t[row+1]-t[row]
            lower=origin+timestep-spacing;upper=origin+timestep+spacing
            endpoints=np.flatnonzero((t>lower)&(t<upper))
            entry.update(time_window_origin_row=row-1,time_window_origin_s=float(origin),
                inferred_sample_spacing_s=float(spacing),time_window_lower_open_s=float(lower),
                time_window_upper_open_s=float(upper),time_window_candidate_count=len(endpoints))
            if not len(endpoints):
                entry.update(status='upstream_no_endpoint_skip',resistance_Ohm=None)
                result.append(entry);continue
            end=int(endpoints[0])
        # Keep the upstream scalar dtype path. A float32 MPR input and a text
        # export parsed as float64 can legitimately differ in the last digits.
        denominator=(current[row-1]-current[end])/1000
        with np.errstate(divide='ignore',invalid='ignore'):
            resistance=np.divide(ocv-v[end],denominator)
        finite=bool(np.isfinite(resistance))
        entry.update(endpoint_row=end,endpoint_time_s=float(t[end]),endpoint_voltage_V=float(v[end]),
            endpoint_current_mA=float(current[end]),denominator_A=float(denominator),
            elapsed_from_flag_row_s=float(t[end]-t[flag_row]),
            elapsed_from_adjusted_row_s=float(t[end]-t[row]),
            resistance_Ohm=float(resistance) if finite else None,
            status='finite' if finite else 'upstream_nonfinite_resistance',zero_denominator=bool(denominator==0),
            nonfinite_value=None if finite else str(resistance))
        result.append(entry)
    return result
