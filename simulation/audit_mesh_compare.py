"""Compare independent same-protocol mesh runs; no simulated states are remapped."""
import argparse,json
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def compare(a,b):
 for key in ('cell','current_A','efc_target','output_period_s','versions','python','tolerance'):
  if a['identity'][key]!=b['identity'][key]:raise ValueError('Protocol/environment mismatch: '+key)
 if a['identity']['mesh_scale']==b['identity']['mesh_scale']:raise ValueError('Different independent grids required')
 qa=a['ledger']['q0_Ah'];qb=b['ledger']['q0_Ah']; points=[]
 amap={round(d['target_efc'],8):d for d in a['diagnostics']}
 for d in b['diagnostics']:
  old=amap.get(round(d['target_efc'],8))
  if old:points.append({'target_EFC':d['target_efc'],'mesh4_measured_EFC':old['efc'],'mesh8_measured_EFC':d['efc'],'mesh4_Qref_Ah':old['Qref_Ah'],'mesh8_Qref_Ah':d['Qref_Ah'],'Qref_abs_diff_Ah':abs(old['Qref_Ah']-d['Qref_Ah']),'mesh4_SOH_percent':old['SOH_percent'],'mesh8_SOH_percent':d['SOH_percent'],'SOH_abs_diff_pp':abs(old['SOH_percent']-d['SOH_percent'])})
 pairs=[]
 ac={c['cycle']:c for c in a['cycles']}
 for c in b['cycles']:
  old=ac.get(c['cycle'])
  if old and 'charge_time_s' in c and 'charge_time_s' in old:pairs.append({'cycle':c['cycle'],'charge_time_abs_diff_s':abs(old['charge_time_s']-c['charge_time_s'])})
 return {'mesh4_EFC':a['ledger']['efc'],'mesh8_EFC':b['ledger']['efc'],'mesh4_status':a['status'],'mesh8_status':b['status'],'Q0_abs_diff_Ah':abs(qa-qb),'Q0_relative_diff':abs(qa-qb)/qa,'RPT_comparison':points,'max_SOH_abs_diff_pp':max((p['SOH_abs_diff_pp'] for p in points),default=None),'paired_charge_cycles':len(pairs),'max_paired_charge_time_diff_s':max((p['charge_time_abs_diff_s'] for p in pairs),default=None),'EOL_convergence_established':False,'long_term_convergence_established':False,'note':'Only common diagnostic targets and same-cycle charge times are compared. Raw differences do not establish 1610-EFC or EOL accuracy. Each grid uses its own Q0; no state projection is used.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--mesh4-dir',required=True);p.add_argument('--mesh8-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 x=read(Path(a.mesh4_dir)/'summary.json');y=read(Path(a.mesh8_dir)/'summary.json')
 r=compare(x,y);Path(a.output).write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k!='RPT_comparison'}))
if __name__=='__main__':main()
