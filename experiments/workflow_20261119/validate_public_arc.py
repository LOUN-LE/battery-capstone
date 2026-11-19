import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path('<USER_HOME>/Documents/Codex/2026-10-30/xzoq/outputs/capstone/repo')
sys.path.insert(0, str(REPO))
from simulation.thermal_abuse_validation import Criterion, import_ohneseit_arc, read_trace, summarize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/thermal_abuse_validation'
OUTPUT.mkdir(parents=True, exist_ok=True)
checksums = {'NMC_SOC100_M1': 'e4d0734e203b6c2b614b9acf5ee95d40',
             'NMC_SOC100_M2': '65e381a24e80d583b73edddcce4eea74'}
# A descriptive research marker. The cited paper uses a different apparatus-specific TR definition.
criterion = Criterion(0, 1, 0, 60,
    'Research marker: first recorded exothermal rate >= 1 C/min, no persistence. Not the paper thermal-runaway definition.')
summaries = []
for name, checksum in checksums.items():
    source = ROOT / 'work/thermal_abuse_raw' / f'{name}.txt'
    target = OUTPUT / f'{name}.csv'
    metadata = OUTPUT / f'{name}.metadata.json'
    import_ohneseit_arc(source, target, metadata, checksum)
    trace = read_trace(target, metadata, criterion.max_gap_s)
    result = summarize(trace, criterion)
    (OUTPUT / f'{name}.report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    summaries.append({'record': name, 'samples': result['samples'],
        'first_recorded_temperature_C': float(trace.temperature_C[0]),
        'recorded_peak_temperature_C': result['recorded_peak_temperature_C'],
        'recorded_peak_rate_C_per_min': result['recorded_peak_rate_C_per_min'],
        'research_rate_marker_time_s': result['criterion_crossing']['onset_s'],
        'research_rate_marker_temperature_C': result['criterion_crossing']['onset_temperature_C'],
        'thermal_runaway_label': '', 'fire_label': '',
        'M50T_validation': 'cell_identity_unverified',
        'trace_complete': False, 'source_sha256': result['metadata']['raw_sha256']})
with (OUTPUT / 'public_arc_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
report = {'data_ingestion_validated': True, 'published_md5_verified_files': 2,
          'thermal_runaway_model_implemented': False, 'M50T_thermal_abuse_validation_passed': False,
          'fire_prediction_validated': False, 'data': summaries,
          'current_model': 'SPMeT lumped operating heat model; no calibrated abuse reaction kinetics, internal-short, venting/combustion, or pack propagation model',
          'missing_for_extension': ['Confirmed target cell and aging state/SOC',
              'Abuse heat source and cooling boundary histories',
              'Matched observation location and time origin',
              'Calorimetry-based reaction parameters plus independent repeat tests',
              'Observed runaway onset definition and annotations',
              'Flame/venting/heat-release observations for fire labels'],
          'sources': ['https://zenodo.org/records/7707929',
              'https://doi.org/10.3390/batteries9050237',
              'https://www.nasa.gov/wp-content/uploads/2024/01/battery-failure-databank.pdf',
              'https://docs.pybamm.org/en/stable/source/examples/notebooks/models/thermal-models.html']}
(OUTPUT / 'readiness_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
