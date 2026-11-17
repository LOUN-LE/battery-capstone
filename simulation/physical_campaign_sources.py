"""Bind measured native capacity targets to reviewed original archive members."""
from __future__ import annotations
import json
import math
from pathlib import Path
from types import SimpleNamespace
from physical_campaign_inputs import checked_path, digest, number

# These two original retrieval receipts were independently reviewed locally.
# A caller cannot manufacture a new receipt containing only pass/CRC flags.
REVIEWED_RECEIPT_SHA256 = {
    '87134a62851a570091c5fadad99c8085a778014264d3c3f1df6e9838b040aa80',
    'e9c8de9a648d334819d365615aa698621ddb7f1f309cb6eb115d1a95442d0468'}


def verify_native_targets(campaign, base, rows):
    identity = campaign.get('source_identity')
    required = {'kind', 'summary', 'receipt', 'temperature_setpoint_C', 'SOC_percent'}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError('Measured targets require reviewed official source identity and native capacity binding')
    if identity['kind'] != 'Kirkaldy2024_Expt2_performance_summary':
        raise ValueError('Unverified generic measured source format is not admitted')
    temperature = number(identity['temperature_setpoint_C'], 'Source temperature')
    if temperature not in (10, 25, 40) or identity['SOC_percent'] != [70, 85]:
        raise ValueError('Unreviewed native summary condition')
    expected = f'Kirkaldy2024_Expt2_{temperature:g}C_SOC70_85'
    if campaign['condition_id'] != expected:
        raise ValueError('Campaign condition differs from source identity')
    summary = checked_path(identity['summary'], base)
    receipt = checked_path(identity['receipt'], base)
    if digest(receipt) not in REVIEWED_RECEIPT_SHA256:
        raise ValueError('Unreviewed official archive identity or receipt bytes')
    # Reuse the independent original archive/member identity check. The
    # sentinel model makes this a source verification, not baseline fitting.
    from evaluate_frozen_baselines import verify_heldout_sources
    from fit_empirical_baselines import load_summary
    entry = {'cell_id': campaign['cell_id'], 'summary_csv': str(summary),
             'source_receipt': str(receipt), 'condition': {'temperature_setpoint_C': temperature}}
    verified = verify_heldout_sources([entry], SimpleNamespace(fit_cell='__source_verification_only__',
                                                              fit_source_sha256=''))[campaign['cell_id']]
    native = load_summary(summary, campaign['cell_id'])
    if not rows or rows[0].get('rpt_id') != 'RPT0':
        raise ValueError('Measured SOH requires first native RPT0 as its fixed Q0 denominator')
    capacities = {}
    for point in native:
        index = point['rpt']
        if index != int(index):
            raise ValueError('Native summary RPT identity is not an integer')
        capacities[f'RPT{int(index)}'] = point['capacity_Ah']
    for row in rows:
        if row['rpt_id'] not in capacities or not math.isclose(
                row['Qref_Ah'], capacities[row['rpt_id']], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Target capacity/RPT does not match original native summary in Ah')
    native_ids = list(capacities)
    evaluated_ids = [r['rpt_id'] for r in rows]
    indices = [native_ids.index(r) for r in evaluated_ids]
    if indices != sorted(set(indices)):
        raise ValueError('Native target RPT rows must be unique and chronological')
    tracked = {str(summary): digest(summary), str(receipt): digest(receipt)}
    for name in ('physical_campaign_sources.py', 'evaluate_frozen_baselines.py', 'fit_empirical_baselines.py'):
        source = Path(__file__).with_name(name).resolve()
        tracked[str(source)] = digest(source)
    return {'source_hashes': tracked, 'source_identity_sha256': digest(summary),
            'source_member_path': verified['source_member_path'],
            'native_Q0_Ah': capacities['RPT0'], 'native_RPT_count': len(native_ids),
            'evaluated_RPT_ids': evaluated_ids, 'full_native_curve_evaluated': evaluated_ids == native_ids,
            'verified_cell_id': verified['verified_cell_id'],
            'verified_temperature_setpoint_C': verified['verified_temperature_setpoint_C']}
