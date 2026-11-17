"""Opt-in v2 continuity observations, using the actual differential initial state.

No physical parameters, solver tolerances, stored solution samples, or inherited
battery.py files are changed. The legacy controller remains in force until a
caller explicitly selects InitialStateCell AND consumes connection_snapshot().
The first solver output is retained as an independent diagnostic observation.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np
import pybamm
import battery as B

DEFINITION = 'physical-initial-state-v2'
CONNECTION_LIMIT = 1e-10


def native_first_output_sanity(initial_y, first_y, len_rhs, rtol, atol):
    """Check first differential output using the unchanged solver tolerances.

    This is an accuracy sanity check, distinct from the 1e-10 physical transfer
    gate. IDA may legitimately initialize algebraic states for a new current or
    voltage mode; only differential states must stay within this error scale.
    Shape and finiteness checks cover the complete native state vector.
    """
    a, b = np.asarray(initial_y, dtype=float), np.asarray(first_y, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not 0 < len_rhs <= a.size:
        raise RuntimeError('Native first-output state dimension mismatch')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise RuntimeError('Nonfinite native first-output state')
    atol = np.broadcast_to(np.asarray(atol, dtype=float), a.shape)
    if (not math.isfinite(rtol) or rtol <= 0
            or not np.all(np.isfinite(atol)) or np.any(atol <= 0)):
        raise RuntimeError('Invalid solver tolerances for first-output sanity check')
    scale = atol[:len_rhs] + rtol * np.abs(a[:len_rhs])
    ratios = np.abs(b[:len_rhs] - a[:len_rhs]) / scale
    maximum = float(np.max(ratios))
    if not math.isfinite(maximum) or maximum > 1.0:
        raise RuntimeError('First differential solver output exceeds solver error scale')
    return dict(max_scaled_differential_error=maximum, rtol=float(rtol),
                atol_min=float(np.min(atol)), atol_max=float(np.max(atol)),
                differential_state_count=int(len_rhs),
                algebraic_state_count=int(a.size-len_rhs),
                algebraic_states='finite, but allowed to initialize for new operating mode')


def validated_difference(previous, initialized):
    """Fail closed on invalid data; keep the original relative error definition."""
    if set(previous) != set(initialized) or not previous:
        raise RuntimeError('Physical state variable mismatch')
    for name in previous:
        a, b = np.asarray(previous[name]), np.asarray(initialized[name])
        if a.shape != b.shape or a.size == 0:
            raise RuntimeError('Physical state shape mismatch')
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise RuntimeError('Nonfinite physical initial state')
    difference = B.snap_diff(previous, initialized)
    jump = max((v['max_rel_diff'] for k, v in difference.items()
                if k != 'discharge_capacity'), default=0.0)
    if not math.isfinite(jump) or jump > CONNECTION_LIMIT:
        raise RuntimeError('Physical state connection mismatch')
    return difference


def differential_variables_only(model, variable_names):
    """Internal IDA initialization may change algebraic states: reject their use."""
    for name in variable_names.values():
        expression = model.get_processed_variable_or_event(name)
        for symbol in expression.pre_order():
            if isinstance(symbol, (pybamm.Time, pybamm.StateVectorDot)):
                raise RuntimeError('Continuity variable depends on time/derivative: '+name)
            if isinstance(symbol, pybamm.StateVector):
                if any(s.start < 0 or s.stop > model.len_rhs
                       for s in symbol.y_slices):
                    raise RuntimeError('Continuity variable depends on algebraic state: '+name)


class InitialStateIDAKLU(pybamm.IDAKLUSolver):
    """Record y0 after Python initialization, immediately before native integration.

    PyBaMM copies experiment solvers shallowly. The shared, bounded sink therefore
    receives the record from the actual step solver, rather than a template.
    """
    def __init__(self, *args, capture_sink=None, variable_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_sink = capture_sink if capture_sink is not None else deque(maxlen=1)
        self.variable_names = dict(B.STATE_VARS if variable_names is None else variable_names)
        self._validated_models = set()

    def _integrate(self, model, t_eval, inputs_list, t_interp=None, **kwargs):
        if len(inputs_list) != 1 or len(model.y0full) != 1:
            raise RuntimeError('Continuity capture requires exactly one input/state set')
        if model not in self._validated_models:
            differential_variables_only(model, self.variable_names)
            self._validated_models.add(model)
        time = float(t_eval[0])
        y0 = np.asarray(model.y0full[0], dtype=float).reshape(-1, 1).copy()
        if not math.isfinite(time) or not np.all(np.isfinite(y0)):
            raise RuntimeError('Nonfinite solver initial state')
        if y0.shape[0] != model.len_rhs + model.len_alg:
            raise RuntimeError('Solver initial state dimension mismatch')
        initial = pybamm.Solution(np.array([time]), y0, model, inputs_list[0])
        snapshot = {key: B.val(initial, name, 0).copy()
                    for key, name in self.variable_names.items()}
        if any(a.size == 0 or not np.all(np.isfinite(a)) for a in snapshot.values()):
            raise RuntimeError('Nonfinite physical initial snapshot')
        self.capture_sink.append(dict(model=model, t_s=time, snapshot=snapshot,
                                      len_rhs=model.len_rhs, len_alg=model.len_alg,
                                      initial_y=y0[:, 0].copy()))
        return super()._integrate(model, t_eval, inputs_list, t_interp, **kwargs)


class InitialStateCell(B.Cell):
    """Opt-in Cell adapter; physics and all diagnostic operations are inherited."""
    connection_definition = DEFINITION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_sink = deque(maxlen=1)
        self._connection_step = None
        self._connection_record = None
        self.last_connection_audit = None
        self.last_reference_observations = None

    def sim(self, key, steps):
        if key not in self._sims:
            self._sims[key] = pybamm.Simulation(
                B.make_model(), parameter_values=self.param.copy(),
                experiment=pybamm.Experiment([tuple(steps)]),
                solver=InitialStateIDAKLU(rtol=self.rtol, atol=self.atol,
                                          capture_sink=self._initial_sink),
                var_pts=self.var_pts)
            self.builds += 1
        return self._sims[key]

    def solve(self, key, steps, start, inputs=None):
        self._initial_sink.clear()
        self._connection_step = self._connection_record = None
        result = super().solve(key, steps, start, inputs)
        step = result[1]
        times=np.asarray(step.t,dtype=float)
        if (times.size < 2 or not np.all(np.isfinite(times))
                or float(times[-1]) <= float(times[0])):
            raise RuntimeError('Invalid or zero-duration solver step time range')
        if not self._initial_sink:
            raise RuntimeError('Missing solver initial-state continuity record')
        record = self._initial_sink[-1]
        if record['model'] is not step.all_models[0] or record['t_s'] != float(step.t[0]):
            raise RuntimeError('Solver initial-state record does not match returned step')
        record['first_output_sanity'] = native_first_output_sanity(
            record['initial_y'], np.asarray(step.all_ys[0],dtype=float)[:,0],
            record['len_rhs'], self.rtol, self.atol)
        self._connection_step, self._connection_record = step, record
        if key == 'REF':
            self.last_reference_observations = [
                {name: np.asarray(part[var].entries,dtype=float).reshape(-1).copy()
                 for name,var in {'time_s':'Time [s]','current_A':'Current [A]',
                     'voltage_V':'Voltage [V]','discharge_capacity_Ah':'Discharge capacity [A.h]'}.items()}
                for part in result[0].cycles[-1].steps]
        return result

    def connection_snapshot(self, step):
        if step is not self._connection_step or self._connection_record is None:
            raise RuntimeError('Missing or stale initial-state continuity record')
        record = self._connection_record
        initial = {k: a.copy() for k, a in record['snapshot'].items()}
        raw = B.snapshot(step, 0)
        # Check shape/finiteness even though the raw output diagnostic has no v2 gate.
        for key in initial:
            if raw[key].shape != initial[key].shape or not np.all(np.isfinite(raw[key])):
                raise RuntimeError('Invalid first solver output for continuity audit')
        raw_difference = B.snap_diff(initial, raw)
        self.last_connection_audit = dict(
            definition=DEFINITION, threshold=CONNECTION_LIMIT,
            initial_t_s=record['t_s'], first_output_t_s=float(step.t[0]),
            len_rhs=record['len_rhs'], len_alg=record['len_alg'],
            native_first_output_sanity=record['first_output_sanity'],
            first_output_vs_initial=raw_difference,
            first_output_vs_initial_max_rel=max(
                (v['max_rel_diff'] for k, v in raw_difference.items()
                 if k != 'discharge_capacity'), default=0.0))
        return initial
