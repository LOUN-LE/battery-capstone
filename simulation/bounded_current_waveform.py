"""Declared approximation of verified ageing CC; never a measured replay claim.

The raw data are untouched. Knots retain source timestamps and first/last two
samples. An integral correction preserves signed Ah and a full source-grid
check bounds continuous piecewise-linear current error, not only knot error.
"""
from __future__ import annotations
import numpy as np
from native_program_contracts import decode_mb

POLICY = dict(name='ageing_CC_linear_Ah_preserving_v1',
              max_error_fraction_native_setpoint=.001,
              interior_near_CC_fraction=.005, min_duration_s=30., min_samples=20,
              interpretation='engineering input approximation; not instrument accuracy or physical acceptance',
              excluded='RPT/GITT/PUL/CV/mixed mode or Ns/unverified native control')


def verified_setpoint(rawstep, source_step, arrays, programs):
    """Return a native PyBaMM-sign CC value only for a single eligible ageing run."""
    if (rawstep.get('source_kind') != 'ageing'
            or rawstep.get('kind') != 'recorded_mode_block'
            or rawstep.get('recorded_mode') != 1
            or source_step.get('kind') != 'eligible_recorded_current'):
        return None
    lo, hi = source_step['row_start'], source_step['row_end_exclusive']
    mode, ns = arrays['mode'][lo:hi], arrays['Ns'][lo:hi]
    eligible=arrays.get('current_drive_eligible_interval')
    if eligible is None or eligible.dtype!=np.bool_ or not np.all(eligible[lo:hi-1]):
        return None
    if (not len(mode) or not np.all(mode == 1) or not np.all(ns == ns[0])
            or int(ns[0]) != rawstep['native_Ns']):
        return None
    program = programs.get(source_step['raw_source_sha256'])
    if program is None:
        return None
    row=program['raw_program_rows'][int(ns[0])]
    if program['technique']=='CP':
        # Same audited absolute-current/native-sign fields used by the existing
        # CP interpreter. No CP endpoint or capacity controller is substituted.
        if row.get('Is_vs')!=2 or row.get('Is_unit') not in (0,1):return None
        value=-float(row['Is'])*(1 if row['Is_unit']==0 else .001)
        return value if np.isfinite(value) and value!=0 else None
    if program['technique']!='MB':return None
    decoded = decode_mb(row,preserve_branch_metadata=True,
                        preserve_charge_metadata=True,sequence_count=program['sequence_count'])
    value = decoded.get('value')
    if decoded['operation'] != 'current' or value is None or not np.isfinite(value) or value == 0:
        return None
    return float(value)


def absolute_area(t,current):
    """Exact integral of abs(piecewise-linear I), including zero crossings."""
    a,b=current[:-1],current[1:];dt=np.diff(t)
    same=a*b>=0;area=dt*(abs(a)+abs(b))/2
    area[~same]=dt[~same]*(a[~same]**2+b[~same]**2)/(2*(abs(a[~same])+abs(b[~same])))
    return float(area.sum())


def compress(table, native_setpoint):
    """Deterministic bounded polygon; ineligible/irreducible data stay raw."""
    table = np.asarray(table, dtype=float)
    if (table.ndim != 2 or table.shape[1] != 2 or len(table) < 2
            or not np.isfinite(table).all() or not np.all(np.diff(table[:, 0]) > 0)):
        raise ValueError('Finite, strictly increasing source waveform required')
    t, current = table.T
    raw_area = float(np.trapezoid(current, t))
    raw_abs=absolute_area(t,current)/3600
    info = dict(policy=POLICY, compressed=False, raw_samples=len(t), knots=len(t),
                raw_signed_Ah=raw_area/3600, max_error_A=0., rms_error_A=0.,
                integral_residual_Ah=0., source_duration_s=float(t[-1]-t[0]),
                raw_absolute_Ah=raw_abs,compressed_absolute_Ah=raw_abs,absolute_Ah_residual=0.,
                model_sign_convention='positive discharge, negative charge')
    if native_setpoint is None:
        info['reason'] = 'native ageing CC not verified'
        return table.copy(), info
    if not np.isfinite(native_setpoint) or native_setpoint == 0:
        raise ValueError('Finite nonzero native setpoint required')
    tolerance = abs(native_setpoint)*POLICY['max_error_fraction_native_setpoint']
    info.update(native_setpoint_A=native_setpoint, error_limit_A=tolerance)
    if np.any(current*np.sign(native_setpoint)<0):
        info['reason']='source sign reversal is outside CC approximation scope'
        return table.copy(),info
    if (len(t) < POLICY['min_samples'] or t[-1]-t[0] < POLICY['min_duration_s']
            or np.max(abs(current[2:-2]-native_setpoint), initial=0.) >
               abs(native_setpoint)*POLICY['interior_near_CC_fraction']):
        info['reason'] = 'too short or interior is not near constant CC'
        return table.copy(), info
    protected = {0, 1, len(t)-2, len(t)-1}
    selected = protected | {len(t)//2}
    while len(selected) < len(t):
        indices = np.array(sorted(selected))
        tt, yy = t[indices], current[indices].copy()
        free = np.array([i not in protected for i in indices])
        # Trapezoidal integral is linear in its ordinate values.
        weights = np.empty(len(tt)); weights[0]=(tt[1]-tt[0])/2; weights[-1]=(tt[-1]-tt[-2])/2
        weights[1:-1]=(tt[2:]-tt[:-2])/2
        yy[free] += (raw_area-float(np.trapezoid(yy, tt)))/float(weights[free].sum())
        error = np.interp(t, tt, yy)-current
        worst = int(np.argmax(abs(error)))
        if abs(error[worst]) <= tolerance:
            candidate = np.column_stack([tt, yy])
            if np.any(yy*np.sign(native_setpoint)<0):
                info['reason']='approximation would reverse current sign'
                return table.copy(),info
            residual = (float(np.trapezoid(yy, tt))-raw_area)/3600
            reduced_abs=absolute_area(tt,yy)/3600
            if abs(residual) > 1e-12*max(1., abs(raw_area/3600)):
                raise ValueError('Signed Ah integral correction failed')
            if not np.array_equal(candidate[[0,1,-2,-1]], table[[0,1,-2,-1]]):
                raise ValueError('Protected edge samples changed')
            # Error is linear between source timestamps: this covers the continuum.
            dt=np.diff(t); e0,error1=error[:-1],error[1:]
            mean_square=float(np.sum(dt*(e0*e0+e0*error1+error1*error1)/3)/(t[-1]-t[0]))
            info.update(compressed=True,knots=len(tt),max_error_A=float(max(abs(error))),
                        rms_error_A=float(np.sqrt(mean_square)),integral_residual_Ah=residual,
                        compressed_absolute_Ah=reduced_abs,absolute_Ah_residual=reduced_abs-raw_abs,
                        first_last_two_samples_exact=True,reason='bounded linear source-time approximation')
            return candidate, info
        # If an already retained node is worst due to integral correction, add
        # the next largest unretained error. Never get stuck on that node.
        if worst in selected:
            available=np.array([i for i in range(len(t)) if i not in selected])
            worst=int(available[np.argmax(abs(error[available]))])
        selected.add(worst)
    info['reason']='no reduction satisfies the declared bound'
    return table.copy(), info
