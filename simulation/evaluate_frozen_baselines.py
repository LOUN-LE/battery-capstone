"""Evaluate a previously frozen C-only baseline; no fit/retune operation exists here."""
from __future__ import annotations
import argparse
import csv
from dataclasses import fields
import json
import math
from pathlib import Path
import re
import traceback
from urllib.parse import unquote

from fit_empirical_baselines import (FrozenBaseline, MODEL_VERSION, digest, load_summary,
                                     predict, metrics, observed_eol)
from score_empirical_validation import score_suite


def load_frozen(path):
    blob = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    dependency = blob.get('predictor_dependency', {})
    if dependency.get('file') != 'fit_empirical_baselines.py' or dependency.get('sha256') != digest(Path(__file__).with_name('fit_empirical_baselines.py')):
        raise ValueError('Frozen predictor code SHA-256 mismatch or provenance missing')
    values = blob['model']
    if set(values) != {field.name for field in fields(FrozenBaseline)}:
        raise ValueError('Unexpected frozen model schema')
    model = FrozenBaseline(**values)
    if model.version != MODEL_VERSION or model.fit_cell != 'C':
        raise ValueError('Expected the declared C-only baseline version')
    coefficients = (model.sqrt_calendar_coefficient_pp, model.linear_efc_coefficient_pp)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in coefficients):
        raise ValueError('Invalid nonnegative baseline coefficients')
    return model


def metadata(condition):
    required = ('condition_id', 'experiment_id', 'temperature_setpoint_C',
                'SOC_percent', 'source_refs')
    if any(key not in condition for key in required) or not condition['source_refs']:
        raise ValueError('Reviewed condition metadata and sources are required')
    if condition['SOC_percent'] != [70, 85]:
        raise ValueError('This benchmark applies only the reviewed Expt2 SOC70-85 protocol')
    if condition['temperature_setpoint_C'] not in (10, 25, 40):
        raise ValueError('Unreviewed Expt2 temperature')
    expected_condition = f"Kirkaldy2024_Expt2_{condition['temperature_setpoint_C']}C_SOC70_85"
    if condition['experiment_id'] != 'Kirkaldy2024_Expt2' or condition['condition_id'] != expected_condition:
        raise ValueError('Experiment and condition identity must match reviewed Expt2 temperature/SOC')
    return {'cell_model': 'LG M50T (LG GBM50T2170)',
            'protocol_id': 'Kirkaldy2024_Expt2_fixed730mAh_0.3Ccharge_1Cdischarge',
            'temperature_protocol_id': f"base_cooled_setpoint_{condition['temperature_setpoint_C']}C",
            'rpt_protocol_id': 'Kirkaldy2024_C10_RPT_25C',
            'soh_definition': '100*C10_capacity/fixed_measured_RPT0_C10_capacity',
            'axis_column': 'absolute_ageing_EFC',
            'axis_definition': 'summary_charge_plus_discharge_ageing_Ah/(2*fixed_measured_RPT0_C10_Ah)',
            'axis_origin': 'RPT0_zero_ageing_throughput', 'axis_unit': 'EFC'}


def verify_heldout_sources(manifest, model):
    """Verify raw identity before any scoring; cell labels alone are insufficient.

    Source receipts are archival evidence, not cryptographic attestations from a
    remote authority. Preserve their bytes and source paths for independent audit.
    """
    if not manifest or len({entry['cell_id'] for entry in manifest}) != len(manifest):
        raise ValueError('Nonempty unique held-out cell manifest required')
    seen = set()
    verified = {}
    for entry in manifest:
        cell = entry['cell_id']
        path = Path(entry['summary_csv']).resolve()
        raw_hash = digest(path)
        if raw_hash == model.fit_source_sha256:
            raise ValueError('Calibration source bytes reused as held-out data')
        if raw_hash in seen:
            raise ValueError('Duplicate raw source bytes across held-out aliases')
        seen.add(raw_hash)
        if cell == model.fit_cell:
            raise ValueError('Calibration cell cannot be an independent held-out cell')
        receipt_path = Path(entry['source_receipt']).resolve()
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        url = unquote(receipt.get('source_archive_url', receipt.get('archive_url', '')))
        if not url.startswith('https://zenodo.org/records/10637534/files/Expt 2,2 - C-based Degradation 2.zip'):
            raise ValueError('Unreviewed source archive identity')
        matches = [r for r in receipt.get('files', []) if r.get('sha256') == raw_hash]
        if len(matches) != 1 or matches[0].get('crc32_verified') is not True:
            raise ValueError('Raw file not uniquely matched to a verified download record')
        source = matches[0]
        source_path = source.get('source_path', source.get('path', ''))
        identity = re.fullmatch(r'Expt 2,2 - C-based Degradation 2/Summary Data/Performance Summary/Expt 2,2 - cell ([A-F]) \((10|25|40)degC\) - Processed Data\.csv', source_path)
        if identity is None:
            raise ValueError('Source record is not a reviewed Expt2 performance-summary member')
        actual_cell, actual_temperature = identity.group(1), int(identity.group(2))
        if actual_cell != cell or actual_temperature != entry['condition']['temperature_setpoint_C']:
            raise ValueError('Declared cell/temperature differs from authoritative source member identity')
        if source.get('cell', cell) != cell or source.get('metadata', {}).get('Cell', cell) != cell:
            raise ValueError('Source receipt cell fields disagree')
        verified[cell] = {'source_sha256': raw_hash, 'receipt_path': str(receipt_path),
                          'receipt_sha256': digest(receipt_path), 'source_member_path': source_path,
                          'verified_cell_id': actual_cell, 'verified_temperature_setpoint_C': actual_temperature}
    return verified


def evaluate_frozen(model_path, manifest, criteria, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(model_path).resolve()
    before = digest(model_path)
    model = load_frozen(model_path)
    predictor_path = Path(__file__).with_name('fit_empirical_baselines.py')
    predictor_before = digest(predictor_path)
    verified_sources = verify_heldout_sources(manifest, model)
    cases_by_model = {name: [] for name in ('constant_initial_SOH_100', 'nonnegative_sqrt_calendar_plus_linear_EFC')}
    report = {'status': 'evaluated', 'physical_validation_complete': False,
              'model_kind': 'empirical_baseline', 'frozen_model_sha256_before': before,
              'frozen_predictor_sha256_before': predictor_before,
              'source_manifest': manifest, 'cells': {}, 'score_suites': {},
              'no_refit_performed': True, 'temperature_or_SOC_factor_added': False,
              'prediction_information': 'Known native RPT date and measured ageing-throughput schedule, not blind future EOL',
              'limitation': 'Matched nominal-condition metadata is source-reviewed; baseline has no electrochemical/thermal dynamics',
              'AWS_started': False}
    for entry in manifest:
        cell = entry['cell_id']
        if cell == model.fit_cell:
            raise ValueError('Calibration C cannot be presented as an independent held-out cell')
        meta = metadata(entry['condition'])
        source = Path(entry['summary_csv']).resolve()
        source_hash = digest(source)
        points = load_summary(source, cell)
        rows = []
        report['cells'][cell] = {**verified_sources[cell], 'source_sha256': source_hash,
                                 'condition': entry['condition'], 'EOL': observed_eol(points), 'models': {}}
        for name in cases_by_model:
            values = [100.]*len(points) if name == 'constant_initial_SOH_100' else predict(model, points)
            report['cells'][cell]['models'][name] = metrics(points, values)
            rows.extend({**point, 'model_id': name, 'predicted_SOH_percent': value,
                         'residual_pp': value-point['SOH_percent']} for point, value in zip(points, values))
            cases_by_model[name].append({'cell_id': cell,
                'experiment_id': entry['condition']['experiment_id'],
                'condition_id': entry['condition']['condition_id'], 'axis_column': 'absolute_ageing_EFC',
                'heldout_condition_axis': 'temperature' if entry['condition']['temperature_setpoint_C'] != 25 else None,
                'measured': [{'absolute_ageing_EFC': p['absolute_ageing_EFC'], 'SOH_percent': p['SOH_percent']} for p in points],
                'predicted': [{'absolute_ageing_EFC': p['absolute_ageing_EFC'], 'SOH_percent': value} for p, value in zip(points, values)],
                'measured_metadata': meta, 'predicted_metadata': meta.copy(),
                'model': {'id': name, 'kind': 'initial_SOH_baseline' if name == 'constant_initial_SOH_100' else 'empirical_baseline',
                    'artifact_path': str(model_path), 'artifact_sha256': before, 'fit_cell_ids': ['C'],
                    'fit_condition_ids': ['Kirkaldy2024_Expt2_25C_SOC70_85'],
                    'frozen_before_evaluation': True, 'retuned_on_validation': False,
                    'heldout_initialization': 'RPT0_Q0_only'}})
        if source_hash != digest(source):
            raise RuntimeError('Measured source changed during evaluation')
        with (output/f'cell_{cell}_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for name, cases in cases_by_model.items():
        (output/f'{name}_cases.json').write_text(json.dumps(cases, indent=2, allow_nan=False), encoding='utf-8')
        report['score_suites'][name] = score_suite(cases, criteria, output)
    after = digest(model_path)
    if before != after:
        raise RuntimeError('Frozen model changed during evaluation')
    if predictor_before != digest(predictor_path):
        raise RuntimeError('Frozen predictor implementation changed during evaluation')
    report['frozen_model_sha256_after'] = after
    report['frozen_predictor_sha256_after'] = digest(predictor_path)
    report['code_sha256'] = digest(__file__)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--criteria', type=Path, default=Path(__file__).resolve().parents[1]/'configs/empirical_acceptance.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        report = evaluate_frozen(args.model, json.loads(args.manifest.read_text(encoding='utf-8-sig')),
                                 json.loads(args.criteria.read_text(encoding='utf-8-sig')), args.output)
    except Exception as error:
        if not existed and args.output.exists():
            (args.output/'failure.json').write_text(json.dumps({'status': 'failed',
                'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                'physical_validation_complete': False}, indent=2), encoding='utf-8')
        raise
    print(json.dumps({'status': report['status'], 'cells': report['cells'],
                      'physical_validation_complete': False}, indent=2))


if __name__ == '__main__':
    main()
