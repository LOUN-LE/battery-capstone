"""Explicit model-family selection and same-family rest-state authentication."""
from __future__ import annotations

import copy
import hashlib
import json


def validate_model_family(value):
    if not isinstance(value, str) or value not in ('SPMe', 'DFN'):
        raise ValueError('Explicit model family must be SPMe or DFN')
    return value


def make_model(family, options):
    import pybamm
    family = validate_model_family(family)

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value

    normalized = {key: tuples(value) for key, value in copy.deepcopy(options).items()}
    return getattr(pybamm.lithium_ion, family)(options=normalized)


def verify_rest_equations(state, parameters, options, mesh, *, model_family):
    """Build, but do not solve, an equivalent rest controller for this family.

    No interpolation, concentration reconstruction, initial-SOC reset or transfer
    between model families or grids is allowed by this authentication path.
    """
    import numpy as np
    import pybamm
    from pybamm.expression_tree.operations.serialise_kernel import encode
    from corrected_history_checkpoint import state_layout

    family = validate_model_family(model_family)
    old = state.all_models[-1]
    if not isinstance(old, getattr(pybamm.lithium_ion, family)):
        raise ValueError('Checkpoint embedded model family differs from '+family)
    if float(np.asarray(state['Current [A]'].entries).reshape(-1)[-1]) != 0:
        raise ValueError('Equation authentication requires a zero-current checkpoint')
    probe = pybamm.Simulation(make_model(family, options), parameter_values=parameters.copy(),
        var_pts=mesh, experiment=pybamm.Experiment([pybamm.step.rest(.004)], temperature=298.15))
    probe.build_for_experiment()
    new = next(iter(probe.steps_to_built_models.values()))
    if state_layout(new) != state_layout(old):
        raise ValueError('Checkpoint state layout differs')

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    fingerprints = {}

    def compare(name, before, after):
        before, after = canonical(encode(before)), canonical(encode(after))
        if before != after:
            raise ValueError('Checkpoint processed physical equations changed: '+name)
        fingerprints[name] = hashlib.sha256(before.encode()).hexdigest()

    for name in ('concatenated_rhs', 'concatenated_algebraic', 'mass_matrix', 'concatenated_initial_conditions'):
        compare(name, getattr(old, name), getattr(new, name))
    observers = ['Voltage [V]', 'Positive electrode open-circuit potential [V]']
    if tuple(options.get('particle phases', ('1', '1'))) == ('2', '1'):
        observers.extend(('Negative electrode primary open-circuit potential [V]',
                          'Negative electrode secondary open-circuit potential [V]'))
    else:
        observers.append('Negative electrode open-circuit potential [V]')
    for name in observers:
        compare(name, old.get_processed_variable(name), new.get_processed_variable(name))
    compare('events', old.events, new.events)
    return dict(model_family=family, discretised_rest_equations_exact=True, equation_sha256=fingerprints,
        scope='Same-family processed rest equations, state layout, initial conditions, voltage/OCP observers and events; no PDE solve or state transfer')
