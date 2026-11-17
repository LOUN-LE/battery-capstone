"""Independent physics/control checks; raw findings are retained even on failure."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import rebuilt_lifetime as N
import pybamm


def historical_sei(capstone):
    evidence = json.loads((capstone/'work/evidence/audit_core_evidence_20261029.json').read_text())
    findings = json.loads((capstone/'work/evidence/audit_core_findings_20261029.json').read_text())
    # From installed solvent-diffusion-limited RHS, with fixed area and T=Tref:
    # dL/dt = Vbar*D*c/(z*L). It is a calendar-time law with no SOC/current term.
    p = pybamm.ParameterValues('OKane2022')
    k = p['SEI partial molar volume [m3.mol-1]']*p['SEI solvent diffusivity [m2.s-1]']
    k *= p['Bulk solvent concentration [mol.m-3]']/p['Ratio of lithium moles to SEI moles']
    l0 = p['Initial SEI thickness [m]']
    result = []
    for name in ('initial', 'legacy100', 'final'):
        t = evidence['aging']['states'][name]['source_ledger']['t_abs_s']
        observed = findings['states'][name]['SEI_thickness_stats']['mean']
        predicted = float(np.sqrt(l0*l0+2*k*t))
        result.append({'state': name, 'time_s': t, 'observed_SEI_m': observed,
            'analytic_SEI_m': predicted, 'relative_residual': abs(observed-predicted)/predicted})
    return {'growth_constant_m2_s': k, 'points': result,
        'all_within_0p1_percent': all(r['relative_residual'] < .001 for r in result),
        'interpretation': 'Small SEI growth is predicted by the selected RHS; no state reset is needed to explain these measurements. This is not validation of real-cell aging.'}


def sequence_tests():
    e = N.Engine(rtol=1e-7, atol=1e-9)
    initial = e.start()
    before = N.fingerprint(initial)
    bounds = e.bounds(initial)
    direct, _, _, _ = e.advance(initial, 'CC', 600., 3.2, bounds)
    split = initial
    for _ in range(10):
        split, _, _, _ = e.advance(split, 'CC', 60., 3.2, e.bounds(split))
    a, b = e.readout(direct), e.readout(split)
    metrics = {k: abs(a[k]-b[k]) for k in a}
    # Finite-current voltage/capacity tolerances are explicitly numerical screening
    # criteria, not the old relative 1e-10 start-state definition.
    passed = metrics['V'] < .001 and metrics['Q_signed_Ah'] < 1e-5 and metrics['SEI_m'] < 1e-12
    rest, _, _, _ = e.advance(initial, 'REST', 600., 3.2, bounds)
    r = e.readout(rest)
    k = (e.param['SEI partial molar volume [m3.mol-1]']*e.param['SEI solvent diffusivity [m2.s-1]']
         *e.param['Bulk solvent concentration [mol.m-3]']/e.param['Ratio of lithium moles to SEI moles'])
    l = e.readout(initial)['SEI_m']
    predicted = float(np.sqrt(l*l+1200*k))
    return {'native600_vs_10x60_abs_differences': metrics, 'screen_pass': passed,
        'rest_SEI_m': r['SEI_m'], 'rest_analytic_SEI_m': predicted,
        'rest_analytic_relative_error': abs(r['SEI_m']-predicted)/predicted,
        'same_time_CC_vs_rest_SEI_abs_difference_m': abs(a['SEI_m']-r['SEI_m']),
        'source_unchanged': before == N.fingerprint(initial)}


def reference_and_models(out, capstone):
    import pandas as pd
    data_path = capstone/'work/external_m50_benchmark/chen_cell02.csv'
    data = pd.read_csv(data_path, skiprows=13)
    measured = data[data['Step'] == 7]
    temperature = float(measured['Temperature Cell [degC]'].median())+273.15
    rows = []
    for family, physics in [('SPMe', 'baseline'), ('DFN', 'baseline'), ('DFN', 'coupled')]:
        e = N.Engine(family=family, physics=physics, temperature_K=temperature)
        main = e.start()
        before = N.fingerprint(main)
        ref, raw = e.reference(main, rest_s=7200)
        q = raw['3_Discharge capacity [A.h]']; q = q-q[0]
        v = raw['3_Voltage [V]']
        observed_q = measured['Capacity [Ah]'].to_numpy()
        observed_v = measured['Voltage [V]'].to_numpy()
        valid = (observed_q >= q[0]) & (observed_q <= q[-1])
        residual = np.interp(observed_q[valid], q, v)-observed_v[valid]
        row = {'family': family, 'physics': physics, 'temperature_K': temperature,
            'temperature_policy': 'constant measured discharge median; not temperature-history replay',
            'Q_model_Ah': ref['Qref_Ah'], 'Q_measured_Ah': float(observed_q[-1]),
            'capacity_relative_error_percent': 100*(ref['Qref_Ah']/observed_q[-1]-1),
            'voltage_RMSE_V': float(np.sqrt(np.mean(residual**2))),
            'voltage_max_abs_error_V': float(np.max(np.abs(residual))),
            'coverage_fraction': float(np.mean(valid)),
            'source_preserved': before == N.fingerprint(main),
            'initial_readout': e.readout(main), 'reference': ref,
            'degradation_validation': False}
        np.savez_compressed(out/f'chen_{family}_{physics}.npz', **raw)
        rows.append(row)
    return {'data_sha256': N.sha(data_path), 'results': rows,
        'scope': 'Initial voltage/capacity screening only. M50T aging remains unvalidated; median-temperature approximation and preconditioning differ from exact replay.'}


def resume_test(out):
    import casadi
    import pickle
    def args(path, **kw):
        d = dict(output=str(path), resume=False, family='SPMe', physics='baseline', mesh=1,
                 current=3.2, max_efc=.7, wall_s=900., boundary_s=60., rpt_every=10., max_pieces=0)
        d.update(kw)
        return SimpleNamespace(**d)
    direct = N.run(args(out/'direct'))
    pause = N.run(args(out/'resumed', max_pieces=30))
    resumed = N.run(args(out/'resumed', resume=True))
    def state(path):
        commit = json.loads((path/'commit.json').read_text())
        with casadi.global_unpickle_context(), (path/commit['checkpoint']).open('rb') as f:
            return pickle.load(f)
    a, b = state(out/'direct'), state(out/'resumed')
    read_a, read_b = N.Engine().readout(a['solution']), N.Engine().readout(b['solution'])
    differences = {k: abs(read_a[k]-read_b[k]) for k in read_a}
    checks = {
        'pause_healthy': pause['status'] == 'ready',
        'same_terminal_reason': direct['status'] == resumed['status'] == 'max_efc_censored',
        'same_main_time': differences['t_s'] < 1e-6,
        'same_efc': abs(a['efc']-b['efc']) < 1e-9,
        'same_fixed_Q0': a['Q0_Ah'] == b['Q0_Ah'],
        'same_stage': a['phase'] == b['phase'],
        'same_voltage': differences['V'] < 1e-6,
        'same_SEI': differences['SEI_m'] < 1e-13,
        'bounded_checkpoint_count': len(list((out/'resumed').glob('checkpoint_*.pkl'))) <= 3,
    }
    return {'checks': checks, 'all_pass': all(checks.values()),
            'abs_readout_differences': differences, 'final_EFC': resumed['efc'],
            'final_stage': resumed['phase'], 'final_physics': resumed['latest_physics']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    capstone = N.ROOT.parent
    report = {'code_sha256': N.sha(N.__file__), 'experimental_validation_complete': False,
              'aws_restarted': False, 'started_epoch': time.time()}
    source_paths = [N.ROOT/'simulation'/n for n in
        ('battery.py', 'lifetime_operation.py', 'run_100efc.py', 'run_lifetime.py', 'run_until80.py')]
    desktop = Path.home()/'Desktop'/'배터리'
    source_paths += [desktop/'simulation/battery.py', desktop/'configs/model.json']
    report['preserved_sources_sha256'] = {str(f): N.sha(f) for f in source_paths}
    tests = [('historical_SEI', lambda: historical_sei(capstone)),
             ('sequence', sequence_tests),
             ('Chen_initial_screen', lambda: reference_and_models(out, capstone)),
             ('resume', lambda: resume_test(out))]
    for name, call in tests:
        try:
            report[name] = call()
        except Exception as exc:
            report[name] = {'error': repr(exc)}
        N.write_json(out/'report.json', report)
    report['preserved_sources_unchanged'] = all(N.sha(f) == v for f, v in
        ((Path(k), v) for k, v in report['preserved_sources_sha256'].items()))
    report['finished_epoch'] = time.time()
    N.write_json(out/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
