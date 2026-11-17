"""Narrow native-control decoding for explicitly conditional calibration inputs.

Raw enum semantics are crosschecked on this dataset, not a vendor-version
certification. This module never admits unknown exposure as measured history.
"""
import math


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Finite numeric native control required')
    return float(x)


def current_A(value, unit):
    if unit not in (0, 1):
        raise ValueError('Unreviewed current unit')
    return finite(value)*(1 if unit == 0 else .001)


def decode_mb(row, *, preserve_branch_metadata=False, sequence_count=None, preserve_charge_metadata=False):
    kind = row['ctrl_type']
    if kind not in (0, 1, 4):
        raise ValueError('Only executed CC/CV/rest rows supported; loops must be unrolled from observations')
    result = {'operation': {0: 'current', 1: 'voltage', 4: 'rest'}[kind], 'limits': [],
              'measured_history_admissible': False, 'original_program_revision_verified': False}
    if kind == 0:
        if row['apply_I/C'] != 0 or row['ctrl1_val_vs'] != 2 or row['charge/discharge'] not in (0, 1):
            raise ValueError('Only absolute-current control with explicit direction supported')
        magnitude = current_A(row['ctrl1_val'], row['ctrl1_val_unit'])
        if magnitude <= 0: raise ValueError('Positive programmed CC magnitude required')
        result['value'] = magnitude * (-1 if row['charge/discharge'] == 0 else 1)
    elif kind == 1:
        if row['ctrl1_val_unit'] != 0 or row['ctrl1_val_vs'] != 4:
            raise ValueError('Only absolute cell-voltage control supported')
        result['value'] = finite(row['ctrl1_val'])
        if result['value'] <= 0: raise ValueError('Positive CV voltage required')
    n = row['lim_nb']
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 3:
        raise ValueError('One to three explicit native limits required')
    for i in range(1, n+1):
        pre = f'lim{i}_'; t, comp, unit = (row[pre+k] for k in ('type','comp','value_unit'))
        value = finite(row[pre+'value'])
        action = row[pre+'action']
        if action != 0 and (not preserve_branch_metadata or action not in (1, 2)):
            raise ValueError('Unreviewed branch/action or nonpositive native limit')
        if comp not in (0, 1) or value <= 0:
            raise ValueError('Unreviewed branch/action or nonpositive native limit')
        if t == 0 and comp == 0 and unit in (1, 2, 3):
            limit = {'quantity': 'time_s', 'value': value*{1:3600,2:60,3:1}[unit], 'comparator': 'greater'}
        elif t == 1 and unit == 0:
            limit = {'quantity': 'voltage_V', 'value': value, 'comparator': 'greater' if comp == 0 else 'less'}
        elif t == 5 and comp == 0 and unit == 1 and kind == 0:
            # Unit alone is insufficient: lim*_Q selects a capacity reference.
            # Never interpret a fraction/reference as an absolute mAh limit.
            if not preserve_charge_metadata:
                raise ValueError('Native charge-reference semantics unresolved')
            limit = {'quantity':'unresolved_native_charge_reference','value':value,
                     'native_Q_code':row[pre+'Q'],'native_unit_code':unit,'comparator':'greater'}
        elif t == 12 and comp == 1 and kind == 1:
            limit = {'quantity': 'absolute_current_A', 'value': current_A(value,unit), 'comparator': 'less'}
        else:
            raise ValueError('Unreviewed native limit enum combination')
        if action:
            target = row[pre+'seq']
            if (not isinstance(sequence_count, int) or isinstance(sequence_count, bool)
                or isinstance(target, bool) or not isinstance(target, int)
                or (action == 1 and not 0 <= target < sequence_count)):
                raise ValueError('Branch metadata requires bounded native sequence identity')
            limit.update(native_action_code=action, native_target_sequence=target,
                         branch_execution_review_required=True)
        result['limits'].append(limit)
    if kind == 4 and any(l['quantity'] != 'time_s' for l in result['limits']):
        raise ValueError('Unreviewed rest endpoint')
    return result
