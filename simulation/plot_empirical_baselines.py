"""Standalone scientific plots of every held-out cell, including failed cases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--prefix', default='heldout')
    args = parser.parse_args()
    report_path = args.evidence/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if report['model_kind'] != 'empirical_baseline':
        raise ValueError('Expected empirical baseline report')
    cells = list(report['cells'])
    source_hashes = {report_path.name: hashlib.sha256(report_path.read_bytes()).hexdigest()}
    for kind in ('curves', 'residuals'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
        fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.08, hspace=.44, wspace=.35)
        for cell, axis in zip(cells, axes.flat):
            path = args.evidence/f'cell_{cell}_predictions.csv'
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            empirical = [r for r in rows if r['model_id'] == 'nonnegative_sqrt_calendar_plus_linear_EFC']
            constant = [r for r in rows if r['model_id'] == 'constant_initial_SOH_100']
            x = [float(r['absolute_ageing_EFC']) for r in empirical]
            if kind == 'curves':
                axis.plot(x, [float(r['SOH_percent']) for r in empirical], 'o', color='black', ms=4, label='Measured native RPT')
                axis.plot(x, [float(r['predicted_SOH_percent']) for r in empirical], '-', color='#1765ae', label='Frozen C-only empirical fit')
                axis.plot(x, [float(r['predicted_SOH_percent']) for r in constant], '--', color='#999999', label='Constant 100 baseline')
                axis.set_ylabel('SOH (%)')
            else:
                axis.axhline(0, color='black', lw=.7)
                axis.axhspan(-2, 2, alpha=.09, color='#318947', label='Basic maximum-error band')
                axis.plot(x, [float(r['residual_pp']) for r in empirical], 'o-', color='#1765ae', ms=4, label='Empirical prediction - measured')
                axis.plot(x, [float(r['residual_pp']) for r in constant], '--', color='#999999', label='Constant prediction - measured')
                axis.set_ylabel('Residual (percentage points)')
            temp = report['cells'][cell]['condition']['temperature_setpoint_C']
            m = report['cells'][cell]['models']['nonnegative_sqrt_calendar_plus_linear_EFC']
            axis.set_title(f"Cell {cell}, {temp} C | RMSE {m['RMSE_pp']:.2f} pp, max {m['max_abs_error_pp']:.2f} pp", fontsize=10)
            axis.set_xlabel('Ageing EFC: total Ah / (2 fixed Q0)', fontsize=9)
            axis.grid(alpha=.25)
        for axis in list(axes.flat)[len(cells):]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=7.5, loc='best')
        fig.suptitle('Frozen 25 C empirical baseline on held-out cells; no temperature refit\nNot a physical-model or EOL validation', fontsize=13, y=.97)
        path = args.evidence/f'{args.prefix}_{kind}.png'
        if path.exists():
            raise ValueError('Preserve existing plot')
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
    receipt = {'source_sha256': source_hashes,
               'figure_interpretation': 'Lines connect predictions only for display; metrics use native RPTs, not interpolated samples',
               'plot_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'physical_validation_complete': False}
    (args.evidence/f'{args.prefix}_plot_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
