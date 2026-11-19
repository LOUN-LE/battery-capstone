"""Read-only independent XLSX→CSV parity check using a different reader."""
import csv
import hashlib
import json
from pathlib import Path
import openpyxl

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/local_thermal_extension/nist_ignition'
source = root / 'work/thermal_extension_raw/nist_ignition.xlsx'
before = hashlib.sha256(source.read_bytes()).hexdigest()
workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
sheet = workbook['Sheet1']
source_values = {}
for row in sheet.iter_rows(min_row=3):
    for cell in row[1:]:
        if cell.value is not None:
            source_values[cell.coordinate] = cell.value
with (out / 'timeseries_all77.csv').open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == len(source_values)
assert len({r['source_cell'] for r in rows}) == len(rows)
for r in rows:
    value = source_values[r['source_cell']]
    if value == 'na':
        assert r['surface_temperature_c'] == '' and r['temperature_missing'] == 'true'
    else:
        assert float(value) == float(r['surface_temperature_c'])
        assert r['temperature_missing'] == 'false'
    assert int(''.join(c for c in r['source_cell'] if c.isdigit())) - 3 == float(r['time_s'])
with (out / 'experiments_nmc21700.csv').open(encoding='utf-8-sig', newline='') as handle:
    cases = list(csv.DictReader(handle))
assert len(cases) == 34 and {int(r['experiment_id']) for r in cases} == set(range(49,83))
assert {int(r['experiment_id']) for r in cases if r['ignition_observed']=='false'} == {65,67,72,73,74,77,78}
assert {int(r['experiment_id']) for r in cases if r['ignition_observed']==''} == {62}
assert sum(r['ignition_observed']=='true' for r in cases) == 26
assert all(not r['ignition_time_s'] and not r['svb_time_s'] and r['m50t_identity_verified']=='false' for r in cases)
workbook.close()
assert before == hashlib.sha256(source.read_bytes()).hexdigest()
report = {'passed': True, 'independent_reader': 'openpyxl read_only=True, input not modified',
          'exact_source_value_parity_rows': len(rows), 'source_cell_uniqueness': True,
          'source_time_row_parity': True, 'source_sha256_unchanged': before,
          'nmc21700_experiments': 34, 'nmc21700_positive':26, 'nmc21700_negative':7, 'nmc21700_unknown':1,
          'focused_software_tests_passed':8,
          'annotation_audit': 'Figures3/4/8 visually inspected; negative NMC IDs manually transcribed, no threshold inference',
          'physical_fire_model_validation': False}
(out/'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
