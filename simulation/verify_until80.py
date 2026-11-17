"""Fresh local mesh1 verification of compact until80 transactions; not a life test."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import battery as B
import run_100efc as R
import run_lifetime as L
import lifetime_policy as LP
import until80_policy as P


def command(script,*arguments,code=0):
    result=subprocess.run([sys.executable,'-X','utf8',str(script),*map(str,arguments)],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    if result.returncode!=code:
        raise RuntimeError(f'Expected exit{code}, got{result.returncode}: {result.stdout[-5000:]} {result.stderr[-5000:]}')
    return result


def hashes(root):
    return {p.relative_to(root).as_posix():P.digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('New verification directory required')
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    script=B.ROOT/'simulation/run_until80.py'
    report={'all_pass':False,'scope':'Synthetic local mesh1<=.15EFC regression; no actual1614AWS migration or life validation'}
    try:
        assert P.upcoming_diagnostics(1614.34,1e6)==[1620.,1630.,1640.]
        assert P.upcoming_diagnostics(999995.,1e6)==[1e6]
        for factor in (.1,.01):
            try:P.validate_control(None,factor,'not-a-real-hash')
            except ValueError:pass
            else:raise AssertionError('Missing tighter-tolerance evidence accepted')
        example=P.eol_bracket({'efc':1610.,'SOH_percent':80.1},{'efc':1620.,'SOH_percent':79.9})
        assert example['interval_lower_EFC']==1610 and example['interval_upper_EFC']==1620 and not example['exact_crossing_known']
        assert P.eol_bracket(None,{'efc':10.,'SOH_percent':99.}) is None
        assert P.eol_bracket(None,{'efc':10.,'SOH_percent':80.}) is None
        common=['--test-scope','--mesh-scale',1,'--max-wall-s',600,'--worker-wall-s',600]
        seed=out/'seed'
        command(script,*common,'--fresh-test-only','--max-efc',.05,'--rpt-every-efc',.05,'--output-dir',seed)
        seed_state=R.load_checkpoint(seed/'checkpoint.pkl');P.validate_checkpoint_report(seed,P.read(seed/'summary.json'))
        assert abs(seed_state['ledger']['efc']-.05)<1e-8
        # A trusted, explicitly synthetic lifetime-v1 container with a real fresh
        # physical state. It checks migration semantics, not the actual AWS SHA.
        source=out/'synthetic_legacy'/'trajectory';source.mkdir(parents=True)
        legacy_args=SimpleNamespace(cell=4,current=3.2,mesh_scale=1,efc=2.,eol_soh=80.,rpt_every_efc=10.)
        points=[P.read(path) for path in sorted((seed/'rpt').glob('*.json'))]
        legacy_identity=L.identity(legacy_args)
        legacy_identity['source_sha256']['simulation/lifetime_operation.py']=P.LEGACY_OPERATION_SHA256
        old={'identity':legacy_identity,'solution':seed_state['solution'],
            'ledger':seed_state['ledger'],'cursor':seed_state['cursor'],
            'elapsed_wall_s':seed_state['cumulative_wall_s'],'source_elapsed_wall_s':0.,
            'cycles':[],'checks':[],'files':[],'diagnostics':points,'terminal':False,'status':'running'}
        R.save_checkpoint(source/'checkpoint.pkl',old)
        P.save(source/'summary.json',{k:v for k,v in old.items() if k!='solution'})
        P.save(source/'diagnostics.json',points)
        P.save(source/'FAILED.json',{'status':'failed','elapsed_wall_s':old['elapsed_wall_s']+.01,
                                    'error':'synthetic fixture, not a real failure'})
        source_before=hashes(source)
        continued=[*common,'--max-efc',.15,'--rpt-every-efc',.05]
        resumed=out/'resumed'
        command(script,*continued,'--migrate-checkpoint',source/'checkpoint.pkl','--output-dir',resumed,
                '--test-rollover-after-pieces',1,code=75)
        pause=P.read(resumed/'outcome.json');P.validate_checkpoint_report(resumed,pause)
        paused=R.load_checkpoint(resumed/'checkpoint.pkl')
        assert pause['status']==P.ROLLOVER and not pause['terminal']
        assert paused['cursor']['phase']=='DISCHARGE'
        assert pause['source_elapsed_wall_s']>=old['elapsed_wall_s']+.01
        # New process must preserve these uncommitted artifacts, not delete them.
        fake1=resumed/'segments/segment_99999999.parquet.partial';fake1.write_bytes(b'preserve-partial')
        fake2=resumed/'rpt/rpt_99999999.json';fake2.write_bytes(b'preserve-rpt')
        command(script,*continued,'--resume','--output-dir',resumed)
        final=P.read(resumed/'outcome.json');P.validate_checkpoint_report(resumed,final)
        assert final['status']=='max_efc_censored' and final['terminal']
        assert not fake1.exists() and not fake2.exists()
        preserved=[p.read_bytes() for p in (resumed/'quarantine').rglob('*') if p.is_file() and p.name!='manifest.json']
        assert b'preserve-partial' in preserved and b'preserve-rpt' in preserved
        direct=out/'direct'
        command(script,*continued,'--migrate-checkpoint',source/'checkpoint.pkl','--output-dir',direct)
        a=R.load_checkpoint(resumed/'checkpoint.pkl');b=R.load_checkpoint(direct/'checkpoint.pkl')
        cell=L.make_cell(legacy_args)
        aa,bb=B.last_step(a['solution']),B.last_step(b['solution']);q0=a['ledger']['q0_Ah']
        degradation_a=cell.degradation_Ah(aa);degradation_b=cell.degradation_Ah(bb)
        delta={'EFC':abs(a['ledger']['efc']-b['ledger']['efc']),
               'time_s':abs(a['ledger']['t_abs_s']-b['ledger']['t_abs_s']),
               'voltage_V':abs(B.scalar(aa,B.VARS['voltage'])-B.scalar(bb,B.VARS['voltage'])),
               'SOC_fraction':abs(cell.soc(aa)-cell.soc(bb)),
               'aging_relative_Q0':max(abs(degradation_a[k]-degradation_b[k])/q0 for k in ('sei_Ah','plated_Ah','dead_Ah'))}
        assert delta['EFC']<1e-8 and delta['time_s']<1e-3 and delta['voltage_V']<1e-3
        assert delta['SOC_fraction']<1e-4 and delta['aging_relative_Q0']<1e-4
        forbidden={'cycles','checks','diagnostics','files'}
        assert not forbidden.intersection(a)
        assert a['rpt_count']>=3 and a['latest_rpt']['efc']<=a['ledger']['efc']+1e-8
        assert (resumed/'checkpoint.pkl').stat().st_size < (seed/'checkpoint.pkl').stat().st_size*2+1048576
        # A broken commit tip must not qualify for automatic rollover/resume.
        corrupt=copy.deepcopy(final);corrupt['current_checkpoint_sha256']='0'*64
        try:P.validate_checkpoint_report(resumed,corrupt)
        except ValueError:pass
        else:raise AssertionError('Corrupt checkpoint report accepted')
        # Tolerance controls must not be silently bypassed during migration.
        command(script,*continued,'--migrate-checkpoint',source/'checkpoint.pkl','--output-dir',out/'no_control',
                '--tolerance-factor',.1,code=1)
        assert not (out/'no_control/checkpoint.pkl').exists()
        invalid=out/'invalid_legacy'/'trajectory';invalid.mkdir(parents=True)
        bad=copy.copy(old);bad['identity']=copy.deepcopy(old['identity']);bad['identity']['current_A']=3.3
        R.save_checkpoint(invalid/'checkpoint.pkl',bad)
        command(script,*continued,'--migrate-checkpoint',invalid/'checkpoint.pkl','--output-dir',out/'refused',code=1)
        assert not (out/'refused/checkpoint.pkl').exists()
        # An explicit test sentinel must fail-stop and preserve the last checkpoint.
        failed=out/'failed'
        command(script,*common,'--fresh-test-only','--max-efc',.05,'--rpt-every-efc',.05,
                '--test-failure-after-pieces',1,'--output-dir',failed,code=1)
        failure=P.read(failed/'FAILED.json')
        assert failure['status']=='failed' and failure['terminal'] and failure['right_censored'] is None
        assert 'test_sentinel_failure' in failure['error'] and failure['last_valid_EFC']==0.
        assert list((failed/'quarantine').rglob('*.partial'))
        # A failed branch is not eligible for unattended retry.
        before_failed=P.digest(failed/'checkpoint.pkl')
        command(script,*common,'--resume','--max-efc',.05,'--rpt-every-efc',.05,'--output-dir',failed,code=1)
        assert P.digest(failed/'checkpoint.pkl')==before_failed
        assert hashes(source)==source_before
        assert all(P.digest(B.ROOT/k)==v for k,v in LP.LEGACY_SHA256.items())
        report.update(all_pass=True,source_unchanged=True,frozen_physics_unchanged=True,
            checkpoint_no_history_keys=True,source_elapsed_included=True,
            resumed_vs_continuous=delta,quarantine_preserves_uncommitted=True,
            corrupt_commit_rejected=True,missing_precision_evidence_rejected=True,
            mismatched_migration_rejected=True,unexpected_failure_is_fatal=True,
            failed_branch_retry_refused=True,until80_policy={'next_at1614':P.upcoming_diagnostics(1614.34,1e6),
              'eol_example':example,'max_cumulative_h':P.MAX_WALL_S/3600},
            output_rows=final['row_count'],rpt_count=final['rpt_count'],
            checkpoint_bytes=(resumed/'checkpoint.pkl').stat().st_size,
            wall_s=time.perf_counter()-started)
    finally:P.save(out/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
