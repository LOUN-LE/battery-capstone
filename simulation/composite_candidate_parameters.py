"""Frozen exploratory composite parameters; never grants physical admission."""
from pathlib import Path
import json
import numpy as np
from joint_spme_bol_fit import derive_updates, sha

F = 96485.33212331001
FRESH_OPTIONS = {'particle phases': ('2', '1'),
    'open-circuit potential': (('single', 'current sigmoid'), 'single'),
    'thermal': 'isothermal', 'SEI': 'none', 'lithium plating': 'none',
    'loss of active material': 'none'}


def silicon_diffusivity_bonkile(sto, T):
    import pybamm
    return 3e-16 * np.exp(48200 / pybamm.constants.R * (1 / 298.15 - 1 / T))


def number(value):
    return float(np.asarray(value.evaluate() if hasattr(value, 'evaluate') else value).item())


def build_frozen_candidate(frozen_path, anchor_path):
    import pybamm
    from scipy.optimize import least_squares
    from initialization_contract import validate_initialization_anchor
    frozen_path, anchor_path = Path(frozen_path).resolve(), Path(anchor_path).resolve()
    frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
    anchor = json.loads(anchor_path.read_text(encoding='utf-8-sig'))
    if (frozen.get('role') != 'C_training_candidate_not_physical_validation'
            or frozen.get('D_opened_before_freeze') is not False
            or frozen.get('C2_GITT_used_for_fit') is not False):
        raise ValueError('Declared C-only exploratory frozen candidate required')
    validate_initialization_anchor(anchor)
    params = frozen['chosen_C']['parameters']
    if len(params) != 3:
        raise ValueError('Three reference material/inventory parameters required')
    _, target = derive_updates(pybamm.ParameterValues('OKane2022'), *params)
    p = pybamm.ParameterValues('Chen2020_composite')
    area = p['Electrode width [m]'] * p['Electrode height [m]']
    qn = [area * p['Negative electrode thickness [m]']
          * p[prefix + 'Negative electrode active material volume fraction']
          * p[prefix + 'Maximum concentration in negative electrode [mol.m-3]'] * F / 3600
          for prefix in ('Primary: ', 'Secondary: ')]
    qp = (area * p['Positive electrode thickness [m]']
          * p['Positive electrode active material volume fraction']
          * p['Maximum concentration in positive electrode [mol.m-3]'] * F / 3600)
    sn, sp = target['Q_n_Ah'] / sum(qn), target['Q_p_Ah'] / qp
    updates = {prefix + 'Negative electrode active material volume fraction':
               p[prefix + 'Negative electrode active material volume fraction'] * sn
               for prefix in ('Primary: ', 'Secondary: ')}
    updates['Positive electrode active material volume fraction'] = p['Positive electrode active material volume fraction'] * sp
    p.update(updates)
    qg, qs = [q * sn for q in qn]
    qp *= sp
    target_v = anchor['target_voltage_V']

    def residual(x):
        ug = number(p['Primary: Negative electrode OCP [V]'](x[0]))
        us = number(p['Secondary: Negative electrode OCP [V]'](x[1]))
        up = number(p['Positive electrode OCP [V]'](x[2]))
        return np.array([ug - us, up - ug - target_v,
                         qg*x[0] + qs*x[1] + qp*x[2] - target['Q_Li_Ah']])

    roots = []
    for start in ([.4, .5, .6], [.1, .2, .8], [.6, .5, .4]):
        solved = least_squares(residual, start, bounds=([.001]*3, [.999]*3),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
        if not solved.success or max(abs(residual(solved.x))) > 1e-8:
            raise ValueError('Composite inventory/phase-potential/terminal anchor root failed')
        roots.append(solved.x)
    if any(np.max(abs(x-roots[0])) > 1e-7 for x in roots):
        raise ValueError('Initial phase root differs across declared starts')
    x = roots[0]
    state = {'Primary: Initial concentration in negative electrode [mol.m-3]':
                 x[0]*p['Primary: Maximum concentration in negative electrode [mol.m-3]'],
             'Secondary: Initial concentration in negative electrode [mol.m-3]':
                 x[1]*p['Secondary: Maximum concentration in negative electrode [mol.m-3]'],
             'Initial concentration in positive electrode [mol.m-3]':
                 x[2]*p['Maximum concentration in positive electrode [mol.m-3]']}
    p.update(state)
    p.update({'Secondary: Negative particle diffusivity [m2.s-1]': silicon_diffusivity_bonkile,
              'Initial temperature [K]': 298.15, 'Ambient temperature [K]': 298.15})
    meta = dict(data_kind='diagnostic_frozen_composite_candidate',
                physical_validation_complete=False, ground_truth_export_admitted=False,
                reference_parameters=params, inventory_target=target,
                phase_Q_Ah=dict(graphite=qg, silicon=qs, positive=qp),
                volume_updates=updates, initial_state_updates=state,
                initial_stoichiometries=x.tolist(), initial_residual=residual(x).tolist(),
                target_anchor_V=target_v,
                source_sha256={str(frozen_path): sha(frozen_path), str(anchor_path): sha(anchor_path),
                               str(Path(__file__).resolve()): sha(__file__)},
                candidate_status='C-only candidate; failed C2 transfer; not physically accepted',
                Si_D_source='https://raw.githubusercontent.com/pybamm-team/PyBaMM/main/packages/pybamm/src/pybamm/input/parameters/lithium_ion/Bonkile2024.py',
                Si_D_ref_m2_s=3e-16, Si_D_Ea_J_mol=48200,
                actual_M50T_phase_OCP_verified=False)
    return p, dict(FRESH_OPTIONS), meta
