"""Real diagnostic check: all four phases honor configured observation period."""
import argparse
from pathlib import Path
import numpy as np
import rebuilt_lifetime as N

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
e = N.Engine()
sol = e.start()
reference, raw = e.reference(sol)
gaps = [float(np.max(np.diff(raw[f'{i}_Time [s]']))) for i in range(4)]
report = {'engine_sha256': N.sha(N.__file__), 'reference': reference,
          'requested_period_s': e.diag['output_period_s'], 'max_gap_s_by_phase': gaps,
          'passed': all(0 < d <= e.diag['output_period_s']+1e-8 for d in gaps),
          'cloud_deployed': False, 'long_lifetime_validated': False}
N.write_json(out/'report.json', report)
if not report['passed']:
    raise SystemExit('Configured RPT output period was not applied')
print(report)
