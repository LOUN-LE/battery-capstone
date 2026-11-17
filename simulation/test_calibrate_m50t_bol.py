import unittest
from calibrate_m50t_bol import F, derive_initial_updates


def parameters():
    return {'Electrode width [m]': 1.58, 'Electrode height [m]': .065,
            'Negative electrode thickness [m]': 8.52e-5, 'Positive electrode thickness [m]': 7.56e-5,
            'Negative electrode active material volume fraction': .75, 'Positive electrode active material volume fraction': .665,
            'Negative electrode porosity': .25, 'Positive electrode porosity': .335,
            'Initial concentration in negative electrode [mol.m-3]': 29866.,
            'Initial concentration in positive electrode [mol.m-3]': 17038.,
            'Maximum concentration in negative electrode [mol.m-3]': 33133.,
            'Maximum concentration in positive electrode [mol.m-3]': 63104.}


class BOLMathTests(unittest.TestCase):
    def test_inventory_uses_scaled_active_fraction_and_preserves_NP(self):
        p = parameters()
        _, base = derive_initial_updates(p, 1., 0)
        updates, adjusted = derive_initial_updates(p, .9, -.2)
        self.assertAlmostEqual(adjusted['target_initial_particle_lithium_Ah'], .9*base['target_initial_particle_lithium_Ah']-.2)
        self.assertAlmostEqual(adjusted['negative_capacity_Ah']/adjusted['positive_capacity_Ah'],
                               base['negative_capacity_Ah']/base['positive_capacity_Ah'])
        recovered_delta = (updates['Initial concentration in negative electrode [mol.m-3]']-p['Initial concentration in negative electrode [mol.m-3]'])*F*1.58*.065*8.52e-5*(.75*.9)/3600
        self.assertAlmostEqual(recovered_delta, -.2)

    def test_all_declared_bound_corners_are_physical(self):
        for scale in (.85, 1):
            for delta in (-.4, .4):
                updates, _ = derive_initial_updates(parameters(), scale, delta)
                self.assertLess(updates['Initial concentration in negative electrode [mol.m-3]'], 33133.)

    def test_invalid_volume_or_stoichiometry_is_rejected(self):
        for scale, delta in ((1.01, 0), (0, 0), (.9, 5), (.9, -10), (float('nan'), 0)):
            with self.assertRaises(ValueError):
                derive_initial_updates(parameters(), scale, delta)

    def test_no_transport_or_degradation_keys_are_updated(self):
        updates, _ = derive_initial_updates(parameters(), .95, .1)
        self.assertEqual(set(updates), {'Negative electrode active material volume fraction',
                                       'Positive electrode active material volume fraction',
                                       'Initial concentration in negative electrode [mol.m-3]'})


if __name__ == '__main__':
    unittest.main()
