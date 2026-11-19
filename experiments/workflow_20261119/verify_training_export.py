from pathlib import Path
import csv,json,hashlib,zipfile,collections,shutil
root=Path(__file__).resolve().parents[1]
out=root/'outputs/aws1000_delivery_20261116'
with (root/'work/aws1000_retrieved/sharing/scenario_summary.csv').open(encoding='utf-8-sig',newline='') as f: source={r['scenario_id']:r for r in csv.DictReader(f)}
path=out/'딥러닝_학습용_열화_1000.csv'
with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
mapping={'x_ambient_temperature_C':'ambient_temperature_C','x_charge_current_A':'charge_current_A','x_initial_soc':'initial_soc','x_heat_transfer_W_m2K':'heat_transfer_W_m2K','y_sei_loss_increase_Ah':'sei_loss_Ah_net_change','y_dead_lithium_increase_mol_m3':'dead_lithium_mol_m3_net_change','y_plating_inventory_net_change_Ah':'plating_inventory_Ah_net_change'}
assert len(rows)==len({r['scenario_id'] for r in rows})==1000
for r in rows:
    s=source[r['scenario_id']]
    assert r['dataset_split']==s['dataset_split']
    assert all(float(r[a])==float(s[b]) for a,b in mapping.items())
    assert int(r['y_plating_event'])==int(s['plating_event']=='True')
    assert not any(v=='' for v in r.values())
assert 'screening_priority' not in rows[0]
zip_path=root/'outputs/딥러닝_학습용_1000건_전달.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['딥러닝_학습용_열화_1000.csv','학습용CSV_설명.txt']:z.write(out/name,name)
with zipfile.ZipFile(zip_path) as z: assert z.testzip() is None
shutil.copy2(root/'outputs/aws1000/independent_data_audit.json',out/'independent_data_audit.json')
all_zip=root/'outputs/캡스톤_1000건_CSV_검토_시간보고서.zip'
with zipfile.ZipFile(all_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file(): z.write(p,p.name)
with zipfile.ZipFile(all_zip) as z: assert z.testzip() is None
print(json.dumps({'rows':1000,'numeric_and_boolean_values_exactly_match_source':True,'splits':dict(collections.Counter(r['dataset_split'] for r in rows)),'training_zip_bytes':zip_path.stat().st_size,'all_zip_bytes':all_zip.stat().st_size},ensure_ascii=False))
