"""Reject retired rest-boundary anchors before starting a new PDE diagnostic."""
from __future__ import annotations
import math

ANCHOR_POLICY = 'declared_rest_interior_last3_v1'


def validate_initialization_anchor(initialization):
    if initialization.get('anchor_policy_id') != ANCHOR_POLICY:
        raise ValueError('Retired or undeclared initial rest anchor: use the declared-rest interior anchor, not the final boundary row')
    duration = initialization.get('declared_rest_duration_s')
    if isinstance(duration, bool) or duration != 120.:
        raise ValueError('Reviewed native initial rest duration must be 120 seconds')
    soc = initialization.get('effective_initial_soc')
    if isinstance(soc, bool) or not isinstance(soc,(int,float)) or not math.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError('Finite effective initial SOC required')
    source = initialization.get('source_C',{})
    if not source.get('path') or not isinstance(source.get('sha256'),str) or len(source['sha256']) != 64:
        raise ValueError('Native C initialization source binding required')
    policy = initialization.get('anchor_policy',{})
    times, voltages = policy.get('selected_times_s',[]), policy.get('selected_voltage_V',[])
    if (policy.get('policy_id') != ANCHOR_POLICY or initialization.get('raw_preserved') is not True
            or len(times) != 3 or len(voltages) != 3
            or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0 <= t <= duration for t in times)
            or any(b <= a for a,b in zip(times,times[1:]))
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 2.5 <= v <= 4.2 for v in voltages)):
        raise ValueError('Three ordered control-interior rest observations and preserved raw rows required')
    target = initialization.get('target_voltage_V')
    if isinstance(target,bool) or not isinstance(target,(int,float)) or not math.isfinite(target) or abs(target-math.fsum(voltages)/3)>1e-10:
        raise ValueError('Rest anchor voltage does not match declared observations')
