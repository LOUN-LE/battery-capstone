"""Meaningful opt-in continuity regression tests; no legacy pickle is read."""
import unittest
from collections import deque
import numpy as np
import pybamm
import battery as B
from connection_initial_state import (InitialStateIDAKLU, InitialStateCell,
    validated_difference, differential_variables_only, native_first_output_sanity)


class InitialStateTests(unittest.TestCase):
    def test_exact_and_zero_initial_states(self):
        for a in (np.array([0., 0.]), np.array([0., 1., 2.])):
            self.assertEqual(validated_difference({'x':a}, {'x':a.copy()})['x']['max_rel_diff'],0)

    def test_injected_physical_jump_rejected_at_unchanged_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'connection mismatch'):
            validated_difference({'x':np.array([1.])}, {'x':np.array([1.+2e-10])})
        with self.assertRaisesRegex(RuntimeError, 'connection mismatch'):
            validated_difference({'x':np.array([0.])}, {'x':np.array([1e-100])})

    def test_shape_keys_and_nonfinite_rejected(self):
        cases=[({'x':np.ones(2)}, {'x':np.ones(1)}),
               ({'x':np.ones(1)}, {'y':np.ones(1)}),
               ({'x':np.array([np.nan])}, {'x':np.ones(1)}),
               ({'x':np.ones(1)}, {'x':np.array([np.inf])})]
        for a,b in cases:
            with self.subTest(a=a,b=b), self.assertRaises(RuntimeError):
                validated_difference(a,b)

    def test_native_guard_rejects_corrupt_differential_output(self):
        with self.assertRaisesRegex(RuntimeError,'exceeds solver error scale'):
            native_first_output_sanity(np.array([1.,2.]),np.array([1.01,2.]),
                                       1,1e-4,1e-6)
        # A new current/voltage mode may initialize algebraic values freely.
        result=native_first_output_sanity(np.array([1.,2.]),np.array([1.,20.]),
                                         1,1e-4,1e-6)
        self.assertEqual(result['max_scaled_differential_error'],0.)
        for a,b in ((np.array([1.]),np.array([np.inf])),
                    (np.array([1.]),np.array([1.,2.]))):
            with self.assertRaises(RuntimeError):
                native_first_output_sanity(a,b,1,1e-4,1e-6)

    def test_large_absolute_time_does_not_change_captured_initial_state(self):
        model=pybamm.BaseModel()
        x=pybamm.Variable('x')
        model.rhs={x:-x}; model.initial_conditions={x:0.0268640808567}
        model.variables={'x':x}
        sink=deque(maxlen=1)
        solver=InitialStateIDAKLU(capture_sink=sink, variable_names={'x':'x'})
        sim=pybamm.Simulation(model,solver=solver)
        solved=sim.solve([0,1]); built=sim.built_model
        y=np.asarray(solved.last_state.all_ys[0]).copy()
        for start_time in (1.,16794912.91924549):
            old=pybamm.Solution(np.array([start_time]),y.copy(),built,{})
            result=solver.step(old,built,60,save=False,t_interp=np.arange(61))
            record=sink[-1]
            np.testing.assert_array_equal(record['initial_y'],y[:,0])
            validated_difference({'x':y[:,0]},record['snapshot'])
            raw=B.val(result,'x',0)
            self.assertTrue(np.all(np.isfinite(raw)))
        self.assertGreater(B.compare(y[:,0],raw)['max_rel_diff'],1e-10)

    def test_real_cell_adapter_transition_and_source_state_preserved(self):
        cell=InitialStateCell()
        initial=cell.initial_state()
        start=initial.last_state
        start_y=B.y_end(start).copy()
        # A true model transition, INIT(rest) -> DIS at a fixed physical boundary.
        end=float(start.t[-1])+1.
        result,step,_=cell.solve(('DIS',5.,1.,1.),[B.step_dis(5.,1.,1.)],
             start,{B.T_STOP:end,B.X_LO:0.0})
        snapshot=cell.connection_snapshot(step)
        validated_difference(B.snapshot(B.last_step(initial),-1),snapshot)
        self.assertEqual(cell.last_connection_audit['threshold'],1e-10)
        np.testing.assert_array_equal(B.y_end(start),start_y)
        # A stale step must never silently reuse a later/earlier initialization.
        with self.assertRaisesRegex(RuntimeError,'stale'):
            cell.connection_snapshot(B.last_step(initial))


if __name__=='__main__':
    unittest.main(verbosity=2)
