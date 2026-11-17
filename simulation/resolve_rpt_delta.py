"""Frozen local states: isolate measurement error from aging integration."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json
import pickle
import time
from pathlib import Path
import casadi
import numpy as np
import rebuilt_lifetime as N

def main(mesh=4):
    base=Path(__file__).resolve().parents[2]/'work/numerical_repair_20261103'
    out=base/f'rpt_delta_resolution_fixed_mesh{mesh}'
    out.mkdir(exist_ok=False)
    rows=[]
    initial=N.Engine(mesh=mesh,rtol=1e-8,atol=1e-10).start()
    for tol in (1e-7,1e-8):
        e=N.Engine(mesh=mesh,rtol=tol,atol=tol/100)
        q0,raw=e.reference(initial)
        np.savez_compressed(out/f'initial_{tol}.npz',**raw)
        for factor in (.1,.01):
            for current in (3.,3.3):
                directory=base/('aging_stage' if mesh==4 else 'mesh8_completion')/f'm{mesh}_f{factor:g}_I{current:.1f}'/'trajectory'
                deadline=time.monotonic()+1800
                while not (directory.parent/'result.json').exists() or not json.loads((directory.parent/'result.json').read_text()).get('execution_valid'):
                    if time.monotonic()>deadline: raise TimeoutError('Valid endpoint not available')
                    time.sleep(1)
                commit=json.loads((directory/'commit.json').read_text())
                path=directory/commit['checkpoint']
                if N.sha(path)!=commit['sha256']: raise ValueError('Checkpoint hash mismatch')
                with casadi.global_unpickle_context(),path.open('rb') as f: state=pickle.load(f)
                if state['identity']!=commit['identity'] or abs(state['efc']-1)>1e-8:
                    raise ValueError('Identity/endpoint mismatch')
                if N.fingerprint(initial)!=state['initial_state_fingerprint']:
                    raise ValueError('Initial physical state mismatch')
                ref,raw=e.reference(state['solution'])
                np.savez_compressed(out/f'f{factor}_I{current}_{tol}.npz',**raw)
                rows.append(dict(aging_factor=factor,current_A=current,rpt_rtol=tol,
                    Q0_Ah=q0['Qref_Ah'],Qref_Ah=ref['Qref_Ah'],
                    SOH_percent=100*ref['Qref_Ah']/q0['Qref_Ah'],
                    main_unchanged=ref['main_unchanged'],checkpoint_sha256=commit['sha256']))
                N.write_json(out/'report.json',dict(status='running',rows=rows))
                print(rows[-1],flush=True)
    N.write_json(out/'report.json',dict(status='completed',rows=rows))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=int,choices=(4,8),default=4)
    main(p.parse_args().mesh)
