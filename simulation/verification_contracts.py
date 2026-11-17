"""Pure checks for numerical-reference verification reports."""
import math


def _finite(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def strict_trajectory_valid(trajectory):
    """A trajectory error estimate requires a completed run and valid diagnostics."""
    return (trajectory.get('status') == 'efc_target'
            and all(trajectory.get(k) is True for k in
                    ('initial_state_unchanged', 'main_state_unchanged', 'ledger_unchanged'))
            and all(trajectory.get(k, {}).get('ok') is True for k in
                    ('Q0_reference_test', 'reference_test', 'R10'))
            and all(_finite(trajectory.get(k), positive=True) for k in
                    ('Q0_Ah', 'Q5_Ah', 'R10_ohm'))
            and all(_finite(trajectory.get(k)) for k in ('dQ5_Ah', 'dR10_ohm', 'dSOH_pp')))


def reference_variants_valid(variants):
    """Capacity similarity cannot rescue an RPT that missed its cutoff events."""
    return bool(variants) and all(
        item.get('ok') is True and
        isinstance(item.get('Q_Ah'), (int, float)) and
        math.isfinite(item['Q_Ah']) and item['Q_Ah'] > 0
        for item in variants)


def pilot_result_complete(result):
    """A partial result or failed diagnostic is not a successful pilot cell."""
    if result.get('failures') != [] or 'memory' not in result:
        return False
    cell_id = result.get('cell', {}).get('id')
    if not isinstance(cell_id, int) or isinstance(cell_id, bool):
        return False
    q0 = result.get('Q0', {})
    at5 = result.get('at_5EFC', {})
    initial = result.get('numerical_initial', {})
    strict = at5.get('strict_from_same_state', {})
    required = [q0.get('Q0_Ah'), at5.get('Q5_Ah'),
                at5.get('R10', {}).get('R10_ohm'), strict.get('dQ5_Ah'), strict.get('dR10_ohm')]
    if not all(_finite(x) for x in required):
        return False
    if not all(initial.get(k, {}).get('ok') is True and
               initial[k].get('main_state_unchanged') is True for k in ('tol_x0.01', 'mesh_x2')):
        return False
    if not all(isinstance(initial[k].get(n), (int, float)) and math.isfinite(initial[k][n])
               for k in ('tol_x0.01', 'mesh_x2') for n in ('dQ0_Ah', 'dR10_ohm')):
        return False
    if not (q0.get('reference_test', {}).get('ok') is True and
            result.get('R10_initial', {}).get('ok') is True and
            _finite(result.get('R10_initial', {}).get('R10_ohm'), positive=True) and
            result.get('initial_state_unchanged') is True):
        return False
    if not (strict.get('reference_test', {}).get('ok') is True and
            strict.get('R10', {}).get('ok') is True and
            _finite(strict.get('Q5_Ah'), positive=True) and
            _finite(strict.get('R10_ohm'), positive=True) and
            strict.get('main_state_unchanged') is True and strict.get('ledger_unchanged') is True):
        return False
    if cell_id == 4 and not strict_trajectory_valid(result.get('trajectory_strict', {})):
        return False
    return (required[0] > 0 and required[1] > 0 and required[2] > 0 and
            at5.get('reference_test', {}).get('ok') is True and
            at5.get('R10', {}).get('ok') is True and
            at5.get('main_state_unchanged') is True and at5.get('ledger_unchanged') is True)
