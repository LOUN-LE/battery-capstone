"""Regression checks for thermal boundary separation and irregular sampling."""
import unittest

import numpy as np
import pybamm
from scipy.integrate import trapezoid

from .thermal_validation import metrics, thermal_parameters


class ThermalValidationTests(unittest.TestCase):
    def test_initial_and_boundary_are_independent(self):
        params = pybamm.ParameterValues("OKane2022")
        params.update(thermal_parameters(35.0, 20.0, 10.0))
        params.update({"Current function [A]": 0})
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        sim = pybamm.Simulation(model, parameter_values=params,
                               solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12,
                                                         options={"dt_max": .1}))
        clock = np.linspace(0, 60, 601)
        sol = sim.solve([0, 60], initial_soc=0.5, t_interp=clock)
        temperature = sol["Volume-averaged cell temperature [C]"].entries
        self.assertAlmostEqual(float(temperature[0]), 35.0, places=8)
        self.assertGreater(float(temperature[-1]), 20.0)
        self.assertLess(float(temperature[-1]), 35.0)
        heat = np.asarray(sol["Volume-averaged total heating [W.m-3]"](clock))
        cooling = np.asarray(sol["Surface total cooling [W.m-3]"](clock))
        contact = np.asarray(sol["Lumped contact resistance heating [W.m-3]"](clock))
        capacity = np.asarray(sol["Volume-averaged effective heat capacity [J.K-1.m-3]"](clock))
        integrated_rise = float(trapezoid((heat + contact + cooling) / capacity, clock))
        self.assertAlmostEqual(float(temperature[-1] - temperature[0]), integrated_rise, places=5)

    def test_adiabatic_zero_current_preserves_temperature(self):
        params = pybamm.ParameterValues("OKane2022")
        params.update(thermal_parameters(35.0, 20.0, 0.0))
        params.update({"Current function [A]": 0})
        sim = pybamm.Simulation(pybamm.lithium_ion.SPMe(options={"thermal": "lumped"}),
                               parameter_values=params)
        sol = sim.solve([0, 60], initial_soc=0.5)
        temperature = sol["Volume-averaged cell temperature [C]"].entries
        self.assertLess(float(np.max(abs(temperature - 35.0))), 1e-7)

    def test_invalid_boundaries(self):
        for values in ((True, 25, 10), (25, np.nan, 10), (-273.15, 25, 10), (25, 25, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                thermal_parameters(*values)

    def test_irregular_observation_density_does_not_reweight_time(self):
        t = np.array([0, .01, .02, 1, 2], dtype=float)
        result = metrics(t, np.ones(5), np.zeros(5))
        self.assertEqual(result["time_weighted_rmse"], 1)
        with self.assertRaises(ValueError):
            metrics(np.array([0, 0, 1]), np.ones(3), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
