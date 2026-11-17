"""Fail-closed numerical screens for a paired, short aging convergence study.

This module deliberately has no PyBaMM dependency. Passing these empirical
screens is neither an error bound nor evidence of a real battery lifetime.
"""
from __future__ import annotations

import math
from collections import defaultdict


DIAGNOSTIC_LEVELS = ((1e-6, 1e-8), (1e-7, 1e-9), (1e-8, 1e-10))
MESHES = (4, 8)
AGING_FACTORS = (1.0, 0.1, 0.01)
CURRENTS = (3.0, 3.3)


def _number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def _span(values):
    return max(values) - min(values)


def assess(rows, expected_variants, target_efc, abs_limit_pp=1e-4,
           relative_limit=0.1):
    """Assess exactly 2 meshes x 3 aging tolerances x 2 charging currents.

    ``expected_variants`` is a list of dictionaries with ``variant``, ``mesh``,
    and ``aging_factor``. Each row has the fields documented in the harness:
    its identity, a common Q0 throughput anchor, achieved throughput/EFC, and
    three RPT measurements. All variants stop at the same physical discharge
    throughput; diagnostic Q0 is mesh/diagnostic-tolerance specific.

    Missing or malformed input returns an incomplete, failing report instead
    of silently dropping a failed run. ``execution_valid`` is the worker's
    explicit attestation that prescribed events and physical checks passed.
    The function independently checks numerical, ledger, and pairing fields.
    """
    result = {
        'execution_complete': False,
        'numerical_screen_pass': False,
        'signal_resolved': False,
        'long_term_convergence_established': False,
        'EOL_convergence_established': False,
        'real_cell_validation': False,
        'screen_note': ('Predeclared short-run empirical screens, not a rigorous '
                        'error bound. A pass does not validate 100 EFC, EOL, '
                        'or experimental degradation.'),
        'errors': [],
    }
    errors = result['errors']
    for name, value in (('target_efc', target_efc), ('abs_limit_pp', abs_limit_pp),
                        ('relative_limit', relative_limit)):
        if not _number(value, positive=True):
            errors.append('Invalid positive finite setting: ' + name)
    if errors:
        return result
    result.update(target_EFC=target_efc, absolute_limit_pp=abs_limit_pp,
                  relative_limit=relative_limit)
    if not isinstance(expected_variants, (list, tuple)):
        errors.append('expected_variants must be an explicit list of variant dictionaries')
        return result
    variants = {}
    combinations = set()
    for spec in expected_variants:
        if not isinstance(spec, dict):
            errors.append('Malformed expected variant')
            continue
        name, mesh, factor = (spec.get(k) for k in ('variant', 'mesh', 'aging_factor'))
        if not isinstance(name, str) or not name or name in variants:
            errors.append('Missing or duplicate expected variant name')
            continue
        if mesh not in MESHES or not _number(factor) or factor not in AGING_FACTORS:
            errors.append('Unsupported expected mesh/aging factor: ' + name)
            continue
        if (mesh, factor) in combinations:
            errors.append('Duplicate expected mesh/aging factor: ' + name)
        combinations.add((mesh, factor))
        variants[name] = (mesh, factor)
    if combinations != {(m, f) for m in MESHES for f in AGING_FACTORS}:
        errors.append('Expected variants must cover mesh4/8 and all three aging factors')
    if not isinstance(rows, (list, tuple)):
        errors.append('rows must be a list')
        return result
    result['expected_rows'] = len(variants) * len(CURRENTS)
    result['received_rows'] = len(rows)
    indexed = {}
    parsed_measurements = {}
    anchors, target_throughputs = [], []
    q0_by_mesh_and_diagnostic = defaultdict(list)
    for index, row in enumerate(rows):
        prefix = f'row[{index}]'
        if not isinstance(row, dict):
            errors.append(prefix + ': malformed row')
            continue
        name, current = row.get('variant'), row.get('current_A')
        if not isinstance(name, str) or name not in variants:
            errors.append(prefix + ': unexpected variant')
            continue
        if not _number(current) or current not in CURRENTS:
            errors.append(prefix + ': unexpected current')
            continue
        key = (name, current)
        if key in indexed:
            errors.append(prefix + ': duplicate variant/current')
            continue
        indexed[key] = row
        prefix = f'{name}/{current:g}A'
        if (row.get('mesh'), row.get('aging_factor')) != variants[name]:
            errors.append(prefix + ': variant metadata mismatch')
        if row.get('execution_valid') is not True:
            errors.append(prefix + ': worker did not validate execution/events')
        required = ('target_EFC', 'achieved_EFC', 'throughput_target_Ah',
                    'discharge_Ah', 'Q0_anchor_Ah')
        ledger_valid = all(_number(row.get(k), positive=True) for k in required)
        if not ledger_valid:
            errors.append(prefix + ': nonfinite/missing/nonpositive ledger value')
        else:
            anchor = row['Q0_anchor_Ah']
            throughput = row['throughput_target_Ah']
            anchors.append(anchor)
            target_throughputs.append(throughput)
            if abs(row['target_EFC'] - target_efc) > 1e-8:
                errors.append(prefix + ': requested EFC mismatch')
            if abs(row['achieved_EFC'] - target_efc) > 1e-8:
                errors.append(prefix + ': achieved EFC mismatch')
            if abs(row['discharge_Ah'] - throughput) > 1e-8:
                errors.append(prefix + ': discharge throughput mismatch')
            if abs(throughput - target_efc * anchor) > 1e-8:
                errors.append(prefix + ': throughput target is inconsistent with anchor')
            if abs(row['achieved_EFC'] - row['discharge_Ah'] / anchor) > 1e-8:
                errors.append(prefix + ': achieved EFC is inconsistent with throughput')
        measurements = row.get('measurements')
        if not isinstance(measurements, list):
            errors.append(prefix + ': missing RPT measurements')
            continue
        measured = {}
        for measurement in measurements:
            if not isinstance(measurement, dict):
                errors.append(prefix + ': malformed RPT measurement')
                continue
            rtol, atol = measurement.get('rtol'), measurement.get('atol')
            if not _number(rtol) or not _number(atol):
                errors.append(prefix + ': nonfinite/missing RPT tolerance')
                continue
            level = (rtol, atol)
            if level not in DIAGNOSTIC_LEVELS:
                errors.append(prefix + ': unexpected RPT tolerance')
                continue
            if level in measured:
                errors.append(prefix + ': duplicate RPT tolerance')
                continue
            measured[level] = measurement
            if not all(_number(measurement.get(k), positive=True)
                       for k in ('Q0_Ah', 'Qref_Ah', 'SOH_percent')):
                errors.append(prefix + ': nonfinite/missing/nonpositive RPT value')
                continue
            if measurement.get('main_unchanged') is not True:
                errors.append(prefix + ': RPT changed or failed to verify main state')
            if 'ok' in measurement and measurement['ok'] is not True:
                errors.append(prefix + ': RPT cutoff validation failed')
            computed_soh = 100 * measurement['Qref_Ah'] / measurement['Q0_Ah']
            if abs(measurement['SOH_percent'] - computed_soh) > 1e-9:
                errors.append(prefix + ': SOH inconsistent with fixed Q0 and Qref')
            q0_by_mesh_and_diagnostic[(variants[name][0], level)].append(measurement['Q0_Ah'])
        if set(measured) != set(DIAGNOSTIC_LEVELS):
            errors.append(prefix + ': incomplete RPT tolerance sweep')
        parsed_measurements[key] = measured
    expected_keys = {(v, c) for v in variants for c in CURRENTS}
    missing = expected_keys - set(indexed)
    if missing:
        errors.append('Missing variant/current rows: ' + ', '.join(
            f'{v}/{c:g}A' for v, c in sorted(missing)))
    if anchors and _span(anchors) > 1e-12:
        errors.append('Q0 throughput anchor differs across variants')
    if target_throughputs and _span(target_throughputs) > 1e-8:
        errors.append('Physical throughput target differs across variants')
    for (mesh, level), values in q0_by_mesh_and_diagnostic.items():
        if _span(values) > 1e-10:
            errors.append(f'mesh{mesh}/{level}: diagnostic Q0 differs across currents/aging tolerances')
    if errors:
        return result

    result['execution_complete'] = True
    result['Q0_anchor_Ah'] = anchors[0]
    result['throughput_target_Ah'] = target_throughputs[0]
    pairs = []
    all_deltas = []
    all_sohs = {current: [] for current in CURRENTS}
    for name, (mesh, factor) in variants.items():
        comparisons = []
        for level in DIAGNOSTIC_LEVELS:
            values = {c: parsed_measurements[(name, c)][level]['SOH_percent']
                      for c in CURRENTS}
            delta = values[3.3] - values[3.0]
            all_deltas.append(delta)
            for current, value in values.items():
                all_sohs[current].append(value)
            comparisons.append({'rtol': level[0], 'atol': level[1],
                                'SOH_3p0_percent': values[3.0],
                                'SOH_3p3_percent': values[3.3],
                                'delta_3p3_minus_3p0_pp': delta})
        spans = {str(c): _span([parsed_measurements[(name, c)][level]['SOH_percent']
                               for level in DIAGNOSTIC_LEVELS]) for c in CURRENTS}
        pairs.append({'variant': name, 'mesh': mesh, 'aging_factor': factor,
                      'diagnostic_comparisons': comparisons,
                      'rpt_delta_span_pp': _span([r['delta_3p3_minus_3p0_pp'] for r in comparisons]),
                      'rpt_individual_SOH_span_pp': spans,
                      'finest_diagnostic_delta_pp': comparisons[-1]['delta_3p3_minus_3p0_pp']})
    indexed_pairs = {(p['mesh'], p['aging_factor']): p for p in pairs}
    finest_delta = indexed_pairs[(8, .01)]['finest_diagnostic_delta_pp']
    limit = min(abs_limit_pp, relative_limit * abs(finest_delta))
    finest_deltas = [p['finest_diagnostic_delta_pp'] for p in pairs]
    total_span = _span(finest_deltas)
    max_rpt_span = max(p['rpt_delta_span_pp'] for p in pairs)
    individual_spans = {str(c): _span(all_sohs[c]) for c in CURRENTS}
    rank_stable = all(d > 0 for d in all_deltas) or all(d < 0 for d in all_deltas)
    checks = {
        'rank_sign_stable_and_nonzero': rank_stable,
        'aging_and_mesh_delta_span_within_limit': total_span <= limit,
        'RPT_delta_span_within_limit': max_rpt_span <= limit,
        'individual_SOH_span_within_absolute_limit': max(individual_spans.values()) <= abs_limit_pp,
    }
    result.update(
        paired_comparisons=pairs,
        finest_reference={'mesh': 8, 'aging_factor': .01,
                          'rtol': DIAGNOSTIC_LEVELS[-1][0],
                          'atol': DIAGNOSTIC_LEVELS[-1][1],
                          'delta_3p3_minus_3p0_pp': finest_delta},
        effective_delta_limit_pp=limit,
        aging_and_mesh_delta_span_pp=total_span,
        max_RPT_delta_span_pp=max_rpt_span,
        max_RPT_individual_SOH_span_pp=max(
            max(p['rpt_individual_SOH_span_pp'].values()) for p in pairs),
        all_settings_individual_SOH_span_pp=individual_spans,
        rank_stable=rank_stable,
        aging_delta_span_by_mesh_pp={str(mesh): _span([
            indexed_pairs[(mesh, f)]['finest_diagnostic_delta_pp'] for f in AGING_FACTORS])
            for mesh in MESHES},
        mesh_delta_difference_by_aging_factor_pp={str(f): abs(
            indexed_pairs[(8, f)]['finest_diagnostic_delta_pp'] -
            indexed_pairs[(4, f)]['finest_diagnostic_delta_pp']) for f in AGING_FACTORS},
        checks=checks,
        screen_failures=[name for name, passed in checks.items() if not passed],
        numerical_screen_pass=all(checks.values()),
        signal_resolved=all(checks.values()),
    )
    return result
