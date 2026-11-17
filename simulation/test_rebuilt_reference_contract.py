"""Exercise real RPT validation using deterministic solution-shaped outputs."""
import copy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import rebuilt_lifetime as rebuilt
from rebuilt_lifetime import Engine, fingerprint


class FakeMain:
    def __init__(self):
        self.all_ts = [np.array([0.])]
        self.all_ys = [np.array([[0.2], [0.7]])]

    def copy(self):
        return copy.deepcopy(self)


class FakeStep:
    def __init__(self, times, currents, voltages, start_capacity, termination):
        times = np.asarray(times, dtype=float)
        currents = np.asarray(currents, dtype=float)
        increments = .5 * (currents[:-1] + currents[1:]) * np.diff(times) / 3600
        capacities = start_capacity + np.r_[0., np.cumsum(increments)]
        self.variables = {
            'Time [s]': times,
            'Current [A]': currents,
            'Voltage [V]': np.asarray(voltages, dtype=float),
            'Discharge capacity [A.h]': capacities,
        }
        self.termination = termination

    def __getitem__(self, variable):
        return SimpleNamespace(entries=self.variables[variable])


def fixture():
    main = FakeMain()
    charge = FakeStep([0, 1, 2], [-1.5] * 3, [4.0, 4.1, 4.2], 0.,
                      'event: Voltage > 4.2 [V] [experiment]')
    cv = FakeStep([2, 3, 4], [-.2, -.1, -.05], [4.2] * 3,
                  charge.variables['Discharge capacity [A.h]'][-1],
                  'event: abs(Current [A]) < 0.05 [A] [experiment]')
    rest = FakeStep([4, 5, 6], [0.] * 3, [4.2, 4.1, 4.05],
                    cv.variables['Discharge capacity [A.h]'][-1], 'final time')
    discharge = FakeStep([6, 7, 8], [.5] * 3, [4.05, 3.0, 2.5],
                         rest.variables['Discharge capacity [A.h]'][-1],
                         'event: Voltage < 2.5 [V] [experiment]')
    pieces = [charge, cv, rest, discharge]
    result = SimpleNamespace(cycles=[SimpleNamespace(steps=pieces)])
    simulation = SimpleNamespace(solve=Mock(return_value=result))
    engine = object.__new__(Engine)
    engine.diag = {
        'charge_current_A': 1.5, 'charge_voltage_V': 4.2,
        'cv_cutoff_current_A': .05, 'rest_s': 2.,
        'discharge_current_A': .5, 'discharge_cutoff_V': 2.5,
        'output_period_s': 1.,
    }
    engine.sim = Mock(return_value=simulation)
    return engine, main, pieces, result, simulation


class RebuiltReferenceContract(unittest.TestCase):
    def test_complete_reference_is_measured_on_an_unchanged_copy(self):
        engine, main, _, _, simulation = fixture()
        before = fingerprint(main)
        report, raw = engine.reference(main)
        self.assertAlmostEqual(report['Qref_Ah'], 1. / 3600., places=14)
        self.assertAlmostEqual(report['integral_Ah'], report['Qref_Ah'], places=14)
        self.assertTrue(report['main_unchanged'])
        self.assertEqual(fingerprint(main), before)
        self.assertEqual(len(raw), 16)
        self.assertIsNot(simulation.solve.call_args.kwargs['starting_solution'], main)
        self.assertFalse(simulation.solve.call_args.kwargs['calc_esoh'])

    def test_wrong_voltage_event_cannot_pass_by_containing_Voltage(self):
        for phase, wrong_event in ((0, 'event: Voltage > 4.1 [V] [experiment]'),
                                   (3, 'event: Voltage < 2.6 [V] [experiment]')):
            with self.subTest(phase=phase):
                engine, main, pieces, _, _ = fixture()
                pieces[phase].termination = wrong_event
                self.assertIn('Voltage', wrong_event)
                with self.assertRaisesRegex(ValueError, 'prescribed events'):
                    engine.reference(main)

    def test_shortened_rest_with_valid_event_and_time_grid_is_rejected(self):
        engine, main, pieces, _, _ = fixture()
        pieces[2].variables['Time [s]'] = np.array([4., 4.5, 5.])
        with self.assertRaisesRegex(ValueError, 'rest duration incomplete'):
            engine.reference(main)

    def test_output_gap_exceeding_requested_period_is_rejected(self):
        engine, main, pieces, _, _ = fixture()
        pieces[0].variables['Time [s]'] = np.array([0., .5, 2.])
        with self.assertRaisesRegex(ValueError, 'output period not honored'):
            engine.reference(main)

    def test_solver_mutation_of_main_state_is_rejected(self):
        engine, main, _, result, simulation = fixture()

        def mutate_main(**kwargs):
            main.all_ys[0][0, 0] += .1
            return result

        simulation.solve.side_effect = mutate_main
        with self.assertRaisesRegex(ValueError, 'mutated main trajectory'):
            engine.reference(main)

    def test_nonfinite_RPT_output_in_any_phase_is_rejected(self):
        for phase, variable in ((0, 'Voltage [V]'), (1, 'Current [A]'),
                                (2, 'Time [s]'), (3, 'Discharge capacity [A.h]')):
            with self.subTest(phase=phase, variable=variable):
                engine, main, pieces, _, _ = fixture()
                pieces[phase].variables[variable][1] = np.nan
                with self.assertRaisesRegex(ValueError, 'Nonfinite reference output'):
                    engine.reference(main)


class RebuiltCliStatusContract(unittest.TestCase):
    def invoke(self, status):
        args = SimpleNamespace(output='unused mocked output')
        parser = Mock()
        parser.parse_args.return_value = args
        result = {'status': status, 'efc': .25, 'Q0_Ah': 5., 'rpts': []}
        with patch.object(rebuilt, 'parser', return_value=parser), \
                patch.object(rebuilt, 'run', return_value=result) as run, \
                patch('sys.stdout', new_callable=io.StringIO) as stdout:
            code = rebuilt.main()
            printed = json.loads(stdout.getvalue())
        parser.parse_args.assert_called_once_with()
        run.assert_called_once_with(args)
        self.assertEqual(printed, result)
        return code

    def test_incomplete_failed_or_unknown_run_status_returns_nonzero(self):
        for status in ('wall_limit', 'interrupted', 'operating_condition_failed',
                       'failed', 'unrecognized_status'):
            with self.subTest(status=status):
                self.assertEqual(self.invoke(status), 2)

    def test_bounded_target_ready_or_virtual_EOL_returns_zero(self):
        for status in ('max_efc_censored', 'ready', 'virtual_eol_observed'):
            with self.subTest(status=status):
                self.assertEqual(self.invoke(status), 0)


if __name__ == '__main__':
    unittest.main()
