"""C-only unfitted SEI/eSOH feasibility diagnostic, never an RPT/lifetime fit."""
import argparse
import csv
from datetime import date
import hashlib
import inspect
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bol-parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pybamm
    from pybamm.models.submodels.interface.sei import sei_growth, base_sei, sei_thickness
    from pybamm.models.submodels.porosity import reaction_driven_porosity
    import pybamm.models.full_battery_models.lithium_ion.electrode_soh as esoh_source
    capstone = Path(__file__).resolve().parents[2]
    c_path = capstone/'work/external_m50_benchmark/kirkaldy_expt2_cellC_performance_summary.csv'
    frozen_hash = digest(args.bol_parameters)
    frozen = json.loads(args.bol_parameters.read_text(encoding='utf-8-sig'))
    with c_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    first_date = date.fromisoformat(rows[0]['RPT start date'])
    elapsed_s = np.array([(date.fromisoformat(row['RPT start date'])-first_date).days*86400 for row in rows], dtype=float)
    measured_soh = np.array([100*float(row['C/10 Capacity [mA h]'])/float(rows[0]['C/10 Capacity [mA h]']) for row in rows])
    p = pybamm.ParameterValues('OKane2022')
    p.update(frozen['parameter_updates'])
    F = float(pybamm.constants.F.evaluate())
    Vbar = float(p['SEI partial molar volume [m3.mol-1]'])
    D = float(p['SEI solvent diffusivity [m2.s-1]'])
    csol = float(p['Bulk solvent concentration [mol.m-3]'])
    z = float(p['Ratio of lithium moles to SEI moles'])
    L0 = float(p['Initial SEI thickness [m]'])
    eps_s = float(p['Negative electrode active material volume fraction'])
    eps_e = float(p['Negative electrode porosity'])
    radius = float(p['Negative particle radius [m]'])
    a = 3*eps_s/radius
    volume = float(p['Electrode width [m]']*p['Electrode height [m]']*p['Negative electrode thickness [m]'])
    k = Vbar*D*csol/z
    L = np.sqrt(L0*L0+2*k*elapsed_s)
    delta_moles = volume*a*(L-L0)*z/Vbar
    loss_Ah = delta_moles*F/3600
    porosity = eps_e-a*(L-L0)
    options = json.loads((capstone/'repo/configs/model.json').read_text(encoding='utf-8'))['options']
    model = pybamm.lithium_ion.SPMe(options=options)
    solver = pybamm.lithium_ion.ElectrodeSOHSolver(p, param=model.param, options=model.options)
    Qn = float(p.evaluate(model.param.n.Q_init))
    Qp = float(p.evaluate(model.param.p.Q_init))
    Qli0 = float(p.evaluate(model.param.Q_Li_particles_init))
    capacities = []
    for loss in loss_Ah:
        solution = solver.solve({'Q_n': Qn, 'Q_p': Qp, 'Q_Li': Qli0-float(loss)})
        capacities.append(float(solution['Capacity [A.h]']))
    normalized = 100*np.array(capacities)/capacities[0]
    # Log-parameter sensitivities of Li-loss only, no fit and no C outcomes used.
    coefficient = F/3600*z*volume*a/Vbar
    sensitivity = np.column_stack((coefficient*k*elapsed_s/L,
                                    coefficient*(L0*L0/L-L0)))
    singular = np.linalg.svd(sensitivity, compute_uv=False)
    norms = np.linalg.norm(sensitivity, axis=0)
    cosine = float(sensitivity[:,0]@sensitivity[:,1]/np.prod(norms))
    target = args.output/'C_unfitted_SEI_eSOH_diagnostic.csv'
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['RPT', 'calendar_days', 'measured_C10_SOH_percent', 'unfitted_SEI_thickness_m',
                         'SEI_lithium_loss_Ah', 'SEI_only_porosity', 'zero_current_eSOH_capacity_Ah',
                         'normalized_zero_current_eSOH_capacity_percent'])
        writer.writerows(zip([r['Ageing Sets'] for r in rows], elapsed_s/86400, measured_soh,
                              L, loss_Ah, porosity, capacities, normalized))
    modules = (sei_growth, base_sei, sei_thickness, reaction_driven_porosity, esoh_source)
    source_files = [inspect.getfile(module) for module in modules]
    report = {'scope': 'Unfitted C-only, constant25C, constant-surface-area, SEI-only analytic diagnostic with zero-current eSOH mapping',
              'fit_performed': False, 'production_parameters_changed': False, 'D_read': False,
              'physical_validation_complete': False, 'SOH_curve_validation_complete': False,
              'EOL_validation_complete': False, 'source_C_sha256': digest(c_path),
              'frozen_BOL_sha256': frozen_hash, 'diagnostic_code_sha256': digest(__file__),
              'installed_primary_code_sha256': {path: digest(path) for path in source_files},
              'pybamm_version': pybamm.__version__,
              'formulas': {'Arrhenius': 'exp(E_SEI/R*(1/Tref-1/T)); weighted time tau=integral(Arrhenius dt)',
                           'k_reference': 'Vbar*D_sol*c_sol/z', 'L': 'sqrt(L0^2+2*k_reference*tau)',
                           'SEI_lithium_loss_mol': 'V_negative*a*z*(L-L0)/Vbar',
                           'SEI_lithium_loss_Ah': 'SEI_lithium_loss_mol*F/3600',
                           'SEI_only_porosity': 'epsilon0-a*(L-L0)',
                           'eSOH': 'Q=ElectrodeSOHSolver(Q_n,Q_p,Q_Li0-SEI_loss_Ah); not a finite-current RPT'},
              'fixed_parameters': {'D_sol_m2_per_s': D, 'L0_m': L0, 'k_m2_per_s': k,
                  'surface_area_per_volume_per_m': a, 'negative_electrode_volume_m3': volume,
                  'initial_particle_lithium_Ah': Qli0, 'Qn_Ah': Qn, 'Qp_Ah': Qp},
              'last_default_diagnostic': {'calendar_days': float(elapsed_s[-1]/86400),
                  'SEI_thickness_m': float(L[-1]), 'SEI_Li_loss_Ah': float(loss_Ah[-1]),
                  'SEI_only_porosity': float(porosity[-1]),
                  'normalized_zero_current_eSOH_percent': float(normalized[-1]),
                  'measured_C10_SOH_percent': float(measured_soh[-1])},
              'local_log_D_log_L0_sensitivity': {'rank': int(np.linalg.matrix_rank(sensitivity)),
                  'singular_values_Ah': singular.tolist(), 'condition_number': float(singular[0]/singular[-1]),
                  'column_cosine': cosine,
                  'meaning': 'Local mathematical diagnostic at fixed defaults, not parameter identifiability under actual experimental uncertainty'},
              'joint_fit_status': 'not_attempted_not_defensible_as_physical_calibration_yet',
              'blocking_reasons': ['Frozen BOL lithium-inventory offset is at declared upper search bound; initial inventory/active loading tradeoff remains',
                  'C RPT dates have day resolution and unknown intra-day ageing/RPT/storage temperature history',
                  'Physical RPT0 SEI thickness/formation history is not measured or reconstructed',
                  'Zero-current eSOH omits finite-current cutoff, SEI resistance and pore-transport effects',
                  'Other Li-loss and LAM mechanisms cannot be uniquely assigned to SEI using capacity alone'],
              'identifiability_limits': ['If 2*k*tau << L0^2, Li loss is proportional to D_sol*tau/L0: the D/L0 ratio dominates',
                  'If times span the crossover and surface area, temperature history, initial inventory and SEI-only mechanism are known, ideal continuous data can distinguish the two parameters locally',
                  'Rank2 in this assumed diagnostic does not establish practical joint identification from the measured13SOHpoints'],
              'needed_before_fitting': ['Resolve physical chronology and temperature-weighted exposure',
                  'Resolve BOL inventory-bound profile or propagate its uncertainty rather than fixing it as known',
                  'Independently constrain initial SEI thickness or a growth coefficient; inspect LLI/LAM/resistance data',
                  'Validate any candidate against full finite-current RPT and held-out cells/conditions']}
    if digest(args.bol_parameters) != frozen_hash:
        raise RuntimeError('Frozen BOL parameters changed during diagnostic')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'fit_performed': False, 'last_default_diagnostic': report['last_default_diagnostic'],
                      'joint_fit_status': report['joint_fit_status'],
                      'local_sensitivity': report['local_log_D_log_L0_sensitivity']}, indent=2))


if __name__ == '__main__':
    main()
