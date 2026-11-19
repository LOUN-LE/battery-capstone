from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

ROOT = Path(r'<USER_HOME>\Documents\Codex\2026-11-15\dl')
BASE = ROOT / 'outputs/local_thermal_extension'
REPO = Path(r'<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
final = BASE/'dl_pilot_final'
report = json.loads((final/'report.json').read_text(encoding='utf-8'))
assert report['status'] == 'local_training_and_inference_completed'
assert report['integrity']['source_snapshot_code_plan_unchanged_during_run']
assert report['included_experiments'] == 33 and report['excluded_unknown_experiments'] == 1
assert report['metrics']['mlp']['test']['false_negative'] == 3
for old in ['dl_pilot', 'dl_pilot_v2']:
    assert sha(BASE/old/'predictions.csv') == sha(final/'predictions.csv')
    assert sha(BASE/old/'split_manifest.csv') == sha(final/'split_manifest.csv')
    assert sha(BASE/old/'experiment_plan.json') == sha(final/'experiment_plan.json')
(BASE/'source_code/simulation').mkdir(parents=True, exist_ok=True)
(BASE/'source_code/simulation/__init__.py').write_text('',encoding='utf-8')
modules = ['ignition_training.py','test_ignition_training.py','nist_ignition_dataset.py','test_nist_ignition_dataset.py',
           'arc_rate_model.py','test_arc_rate_model.py','failure_databank.py','verify_failure_databank.py','thermal_abuse_validation.py']
for name in modules:
    source = REPO/'simulation'/name
    if source.exists():
        shutil.copy2(source, BASE/'source_code/simulation'/name)
(BASE/'source_code/requirements.txt').write_text('numpy==2.4.6\ntorch==2.14.1\nscikit-learn==1.9.1\nmatplotlib==3.11.2\n',encoding='utf-8')
(BASE/'source_code/README.txt').write_text('Run from this source_code directory with Python 3.11 and the recorded dependencies.\n'
    'python -m simulation.ignition_training --nist-input ../nist_ignition/experiments_nmc21700.csv --output new_pilot_run\n'
    'python -m unittest simulation.test_ignition_training -v\n'
    'The original repository and data files are preserved. Do not overwrite a scored run.\n',encoding='utf-8')
(BASE/'failure_databank/source_data').mkdir(exist_ok=True)
shutil.copy2(ROOT/'work/thermal_extension_raw/battery_failure_databank_revision2_feb24.xlsx',
             BASE/'failure_databank/source_data/battery_failure_databank_revision2_feb24.xlsx')
rows = [
    {'dataset':'NIST_NMC21700','independent_experiments':34,'time_rows':38229,'usable_binary_ignition_labels':33,'label_source':'observed_in_paper','purpose':'conditional experimental ignition learning','path':'nist_ignition/experiments_nmc21700.csv'},
    {'dataset':'NIST_all','independent_experiments':77,'time_rows':83925,'usable_binary_ignition_labels':75,'label_source':'observed_in_paper','purpose':'two distinct chemistries; not all used in pilot','path':'nist_ignition/experiments_all77.csv'},
    {'dataset':'FBD_M50_BV','independent_experiments':25,'time_rows':'','usable_binary_ignition_labels':0,'label_source':'calorimeter_heat_mass','purpose':'energy regression candidate; no fire label','path':'failure_databank/m50_records.csv'},
    {'dataset':'ARC_NMC21700','independent_experiments':2,'time_rows':230,'usable_binary_ignition_labels':0,'label_source':'exotherm_temperature_rate','purpose':'limited empirical rate model','path':'arc_rate/report.json'},
    {'dataset':'AWS_pilot32','independent_experiments':0,'time_rows':63761,'usable_binary_ignition_labels':0,'label_source':'simulation','purpose':'32 numeric scenarios; plating/temperature only','path':'../aws32/scenario_summary.csv'},
]
with (BASE/'dataset_index.csv').open('w',encoding='utf-8-sig',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
verification={'final_run':'dl_pilot_final','all_three_runs_same_predictions':True,'all_three_runs_same_split_and_plan':True,
              'model_tuned_after_test':False,'retained_runs_reason':'Input snapshot/hash/adapter provenance validation improvements only',
              'final_model_report_sha256':sha(final/'report.json'),'final_input_sha256':sha(final/'training_input_snapshot.csv')}
(BASE/'release_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
files=sorted(p for p in BASE.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and '__pycache__' not in p.parts)
(BASE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(BASE).as_posix()}\n' for p in files),encoding='utf-8')
archive=ROOT/'outputs/capstone_local_experiments_and_training_20261115.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[BASE/'SHA256SUMS.txt']:
        z.write(p,p.relative_to(BASE).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for p in files:
        assert hashlib.sha256(z.read(p.relative_to(BASE).as_posix())).hexdigest()==sha(p)
print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'sha256':sha(archive)},ensure_ascii=False))
