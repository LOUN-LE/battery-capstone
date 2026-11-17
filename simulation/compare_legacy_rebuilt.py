"""Same-input independent comparison, not a production dependency."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import casadi
import pickle
import battery as B
import rebuilt_lifetime as N


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--efc', type=float, default=5.01)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'legacy_engine_sha256': N.sha(B.__file__),
              'rebuilt_engine_sha256': N.sha(N.__file__), 'target_EFC': args.efc,
              'rtol': 1e-5, 'atol': 1e-7, 'mesh_scale': 1, 'charge_A': 3.2}
    try:
        cell = B.Cell(rtol=1e-5, atol=1e-7, mesh_scale=1)
        sol = cell.initial_state()
        original = B.y_end(sol).copy()
        rpt0 = cell.reference_test(sol.last_state.copy())
        if not np.array_equal(B.y_end(sol), original):
            raise ValueError('Legacy reference mutated main state')
        rs = B.RunState(4, 'independent_legacy_comparison', q0_Ah=rpt0['Q_cum_Ah'],
                        t_abs_s=float(sol.t[-1]))
        history = []
        def diag(main, ledger, target):
            before, book = B.y_end(main).copy(), copy.deepcopy(B.ledger(ledger))
            r = cell.reference_test(main.last_state.copy())
            if not np.array_equal(before, B.y_end(main)) or B.ledger(ledger) != book:
                raise ValueError('Legacy reference changed main/ledger')
            history.append({'EFC': ledger.efc, 'Qref_Ah': r['Q_cum_Ah'],
                            'SOH_percent': 100*r['Q_cum_Ah']/ledger.q0_Ah})
        sol, cycles, status, checks = B.run_operation(cell, sol, rs, 3.2, 5., 60.,
            efc_stop=args.efc, output_period_s=1., diagnostic_efcs=[5.,10.], on_diagnostic=diag)
        read_old = N.Engine().readout(sol)
        report['legacy'] = {'status': status, 'ledger': B.ledger(rs), 'physics': read_old,
                            'RPTs': history, 'checks': checks}
        N.write_json(out/'report.json', report)
        result = N.run(SimpleNamespace(output=str(out/'rebuilt'), resume=False, family='SPMe',
            physics='baseline', mesh=1, current=3.2, max_efc=args.efc, wall_s=1200.,
            boundary_s=60., rpt_every=10., max_pieces=0))
        commit = json.loads((out/'rebuilt/commit.json').read_text())
        with casadi.global_unpickle_context(), (out/'rebuilt'/commit['checkpoint']).open('rb') as f:
            rebuilt = pickle.load(f)
        read_new = N.Engine().readout(rebuilt['solution'])
        report['rebuilt'] = {'status': result['status'], 'Q0_Ah': result['Q0_Ah'],
                             'EFC': result['efc'], 'physics': read_new, 'RPTs': result['rpts']}
        report['absolute_differences'] = {k: abs(read_old[k]-read_new[k]) for k in read_new}
        report['Q0_abs_difference_Ah'] = abs(rs.q0_Ah-result['Q0_Ah'])
        if history:
            report['SOH_at_5_abs_difference_percentage_points'] = abs(
                history[0]['SOH_percent']-result['rpts'][1]['SOH_percent'])
        report['scope'] = 'Matching short trajectory can exclude some implementation faults only. It cannot validate a complete lifetime or real-cell degradation.'
    except Exception as exc:
        report['error'] = repr(exc)
        N.write_json(out/'report.json', report)
        raise
    N.write_json(out/'report.json', report)
    print(json.dumps({k: report.get(k) for k in
          ('absolute_differences', 'Q0_abs_difference_Ah', 'SOH_at_5_abs_difference_percentage_points')}, indent=2))


if __name__ == '__main__':
    main()
