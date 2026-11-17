import unittest

from diagnose_particle_size import EXPECTED_EVENTS, validate_events, raw_quad
import pybamm


class ParticleSizeEventTests(unittest.TestCase):
    def test_absolute_current_event_matches_installed_controller(self):
        validate_events(EXPECTED_EVENTS)
        self.assertEqual(EXPECTED_EVENTS[2],
                         'event: abs(Current [A]) < 0.05 [A] [experiment]')

    def test_retired_signed_current_alias_is_rejected(self):
        events = list(EXPECTED_EVENTS)
        events[2] = 'event: Current > -0.05 [A] [experiment]'
        with self.assertRaisesRegex(ValueError, 'Native event mismatch'):
            validate_events(events)

    def test_watchdog_cannot_pass_as_voltage_cutoff(self):
        for index in (1, 2, 4, 7):
            events = list(EXPECTED_EVENTS)
            events[index] = 'final time'
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_events(events)

    def test_truncated_or_reordered_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_events(EXPECTED_EVENTS[:-1])
        events = list(EXPECTED_EVENTS)
        events[1], events[4] = events[4], events[1]
        with self.assertRaises(ValueError):
            validate_events(events)

    def test_installed_symbolic_lognormal_accepts_midpoint_vectors(self):
        p = pybamm.get_size_distribution_parameters(
            pybamm.ParameterValues('OKane2022'), sd_n=.3)
        function = p['Negative area-weighted particle-size distribution [m-1]']
        low = p['Negative minimum particle radius [m]']
        high = p['Negative maximum particle radius [m]']
        coarse = raw_quad(function, low, high, 16)
        fine = raw_quad(function, low, high, 32)
        self.assertTrue(coarse['all_values_finite'])
        self.assertAlmostEqual(coarse['raw_area_density_mass'], 1, delta=.003)
        self.assertAlmostEqual(coarse['area_mean_m'],
                               p['Negative particle radius [m]'], delta=1e-8)
        self.assertAlmostEqual(coarse['area_mean_m'], fine['area_mean_m'], delta=1e-8)

    def test_invalid_density_cannot_be_normalized_into_a_pass(self):
        for function in (lambda r: r*0, lambda r: r*0-1):
            with self.subTest(function=function), self.assertRaises(ValueError):
                raw_quad(function, 0, 1, 16)
        with self.assertRaises(ValueError):
            raw_quad(lambda r: r, -1, 1, 16)


if __name__ == '__main__':
    unittest.main()
