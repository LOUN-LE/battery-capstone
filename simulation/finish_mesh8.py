"""User-authorized local mesh8 completion; no AWS calls."""
from pathlib import Path
from types import SimpleNamespace
import audit_short_convergence as H

if __name__ == '__main__':
    base = Path(__file__).resolve().parents[2]/'work/numerical_repair_20261103'
    H.VARIANTS = [v for v in H.VARIANTS if v['mesh'] == 8 and v['aging_factor'] != 1.]
    args = SimpleNamespace(stage='aging', jobs=2, efc=1., case_timeout_s=1800.,
        wall_s=7200., abs_screen_pp=1e-4, relative_screen=.1,
        anchor_source=str(base/'short_1efc_v2'))
    raise SystemExit(H.controller(args, base/'mesh8_completion'))
