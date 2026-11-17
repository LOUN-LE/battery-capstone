"""C-only structural candidate screen; no fit, D access, or lifetime claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    source = 'Kirkaldy2024 DOI10.1016/j.jpowsour.2024.234185 section2.4; nominal RPT conditions, not exact historical replay'
    def step(operation, value, duration, termination, expected):
        item = {'role': 'rpt', 'source_ref': source, 'operation': operation,
                'temperature_K': 298.15, 'duration_s': duration, 'period_s': 1,
                'termination': termination, 'expected_termination': expected}
        if value is not None:
            item['value'] = value
        return item
    return {'cell_model': 'LG M50T structural candidate; Chen2020_composite is not a validated M50T lifetime set',
            'protocol_id': 'C_RPT0_composite_structural_screen_v1', 'source_ref': source,
            'initial_state_source': 'Assumed fresh SOC0.2, no formation-history replay; full precharge follows',
            'family': 'SPMe', 'parameter_set': 'Chen2020_composite',
            'options': {'particle phases': ['2', '1'],
                        'open-circuit potential': [['single', 'current sigmoid'], 'single'],
                        'thermal': 'isothermal'},
            'mesh_points': {'x_n': 5, 'x_s': 5, 'x_p': 5, 'r_n_prim': 10, 'r_n_sec': 10, 'r_p': 10},
            'initial_soc': .2, 'initial_temperature_K': 298.15,
            'rtol': 1e-6, 'atol': 1e-8, 'max_wall_s': 600,
            'steps': [step('current', -1.5, 20000, ['4.2 V'], 'event: Voltage > 4.2 [V] [experiment]'),
                      step('voltage', 4.2, 20000, ['0.05 A'], 'event: abs(Current [A]) < 0.05 [A] [experiment]'),
                      step('rest', None, 3600, [], 'final time'),
                      step('current', .5, 50000, ['2.5 V'], 'event: Voltage < 2.5 [V] [experiment]')]}


def read_C_curve(path):
    import numpy as np
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    q = np.array([float(r['Charge (mA.h)'])/1000 for r in rows])
    q -= q[0]
    v = np.array([float(r['Voltage (V)']) for r in rows])
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)) or not np.all(np.diff(q) > 0):
        raise ValueError('C measured curve is not valid and strictly increasing')
    return q, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capstone = Path(__file__).resolve().parents[2]
    source = capstone/'work/external_m50_benchmark/kirkaldy_expt2_cellC_RPT0_C10.csv'
    adapter = Path(__file__).with_name('measured_protocol_replay.py')
    p = protocol()
    p_path = out/'declared_protocol.json'
    p_path.write_text(json.dumps(p, indent=2), encoding='utf-8')
    declared = {'source_C': str(source), 'source_C_sha256': digest(source),
                'protocol_sha256': digest(p_path), 'adapter_sha256': digest(adapter),
                'screen_code_sha256': digest(__file__), 'selection': 'C-only structural candidate; D is not read',
                'degradation_parameter_changes': False, 'fitting_performed': False}
    (out/'declared_screen.json').write_text(json.dumps(declared, indent=2), encoding='utf-8')
    began = time.monotonic()
    command = [sys.executable, str(adapter), '--protocol', str(p_path), '--output', str(out/'replay')]
    with (out/'controller.log').open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=630)
    receipt = json.loads((out/'replay/report.json').read_text(encoding='utf-8'))
    report = {'scope': 'Single C RPT0 absolute capacity/voltage structural screening only',
              'status': 'screen_failed', 'D_read': False, 'fit_performed': False,
              'transport_or_degradation_parameters_changed': False, 'physical_validation_complete': False,
              'SOH_curve_validation_complete': False, 'EOL_validation_complete': False,
              'AWS_started': False, 'wall_s': time.monotonic()-began,
              'replay_exit_code': result.returncode, 'replay_status': receipt.get('status'),
              'declaration': declared,
              'limitations': ['Different model structure and parameter set from OKane2022 single-phase candidate',
                  'No replay of five formation cycles or measured temperature history',
                  'Fresh SOC0.2 assumption and nominal25C full precharge RPT, not exact laboratory chronology',
                  'Small mesh screen only; no mesh/tolerance convergence established',
                  'This result is not a 13-RPT SOH curve or independent-cell verification']}
    if result.returncode == 0 and receipt.get('status') == 'replay_completed':
        import numpy as np
        mq, mv = read_C_curve(source)
        with np.load(out/'replay/step_0004.npz') as raw:
            q = raw['Discharge capacity [A.h]'].copy()
            q -= q[0]
            v = raw['Voltage [V]'].copy()
        mask = (mq >= q[0]) & (mq <= q[-1])
        error = np.interp(mq[mask], q, v)-mv[mask]
        report.update(status='C_only_BOL_screen_completed', C_measured_Q_Ah=float(mq[-1]),
                      model_Q_Ah=float(q[-1]), capacity_relative_error_percent=float(100*(q[-1]/mq[-1]-1)),
                      full_observed_voltage_coverage_fraction=float(np.mean(mask)),
                      covered_native_points=int(np.sum(mask)), observed_native_points=len(mq),
                      voltage_RMSE_V_on_covered_points=float(np.sqrt(np.mean(error**2))),
                      voltage_max_abs_error_V_on_covered_points=float(np.max(np.abs(error))),
                      expected_endpoint_checks_passed=True)
        with (out/'C_RPT0_residuals.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['measured_Q_Ah', 'measured_V', 'predicted_V', 'residual_V'])
            writer.writerows(zip(mq[mask], mv[mask], mv[mask]+error, error))
    else:
        report['error'] = receipt.get('error', 'Replay did not complete')
    if digest(source) != declared['source_C_sha256'] or digest(adapter) != declared['adapter_sha256']:
        report.update(status='source_changed_during_screen', error='C source or replay adapter hash changed')
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'C_only_BOL_screen_completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
