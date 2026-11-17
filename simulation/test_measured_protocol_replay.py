import copy
import unittest
from measured_protocol_replay import validate_protocol, replay_sequence, normalized_model_options, runtime_mesh_points


def protocol():
    return {'cell_model': 'test-cell', 'protocol_id': 'fixture', 'source_ref': 'fixture',
            'initial_state_source': 'fixture-known-empty', 'family': 'SPMe',
            'parameter_set': 'OKane2022', 'options': {'thermal': 'isothermal'},
            'mesh_points': dict(x_n=5, x_s=5, x_p=5, r_n=10, r_p=10),
            'initial_soc': 0, 'initial_temperature_K': 298.15,
            'rtol': 1e-6, 'atol': 1e-8, 'max_wall_s': 60,
            'steps': [{'role': 'rpt', 'source_ref': 'fixture', 'operation': 'rest',
                       'duration_s': 5, 'period_s': 1, 'temperature_K': 298.15,
                       'termination': [], 'expected_termination': 'final time'}]}


class MeasuredReplayTests(unittest.TestCase):
    def test_explicit_fields_required(self):
        for key in ('initial_soc', 'initial_state_source', 'initial_temperature_K', 'mesh_points'):
            p = protocol(); del p[key]
            with self.assertRaises(ValueError):
                validate_protocol(p)
        for key in ('temperature_K', 'termination', 'expected_termination', 'source_ref'):
            p = protocol(); del p['steps'][0][key]
            with self.assertRaises(ValueError):
                validate_protocol(p)

    def test_actual_rpt_cannot_be_omitted(self):
        p = protocol(); p['steps'][0]['role'] = 'diagnostic'
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_nonfinite_or_unbounded_protocol_rejected(self):
        for key, value in (('initial_soc', float('nan')), ('max_wall_s', 3601), ('rtol', 0)):
            p = protocol(); p[key] = value
            with self.assertRaises(ValueError):
                validate_protocol(p)

    def test_rpt_terminal_state_propagates_but_diagnostic_does_not(self):
        p = protocol()
        p['steps'] = [dict(p['steps'][0], role=role) for role in ('ageing', 'rpt', 'diagnostic', 'ageing')]
        observed = []
        def advance(state, step):
            observed.append((step['role'], state['exposure']))
            state['exposure'] += 1
            return state, {'exposure': state['exposure']}
        terminal, records = replay_sequence(p, {'exposure': 0}, advance)
        self.assertEqual(observed, [('ageing', 0), ('rpt', 1), ('diagnostic', 2), ('ageing', 2)])
        self.assertEqual(terminal['exposure'], 3)
        self.assertEqual([r['propagated'] for r in records], [True, True, False, True])

    def test_negative_current_is_explicit_charge(self):
        p = protocol(); p['steps'][0].update(operation='current', value=-1.5)
        self.assertIs(validate_protocol(p), p)
        del p['steps'][0]['value']
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_parameter_overrides_and_unknown_profiles_rejected(self):
        for overrides in ([], {'parameter': float('nan')}, {'parameter': '1e-3'}, {'parameter': True}):
            p = protocol(); p['parameter_overrides'] = overrides
            with self.assertRaises(ValueError):
                validate_protocol(p)
        p = protocol(); p['physics'] = 'coupled'
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_explicit_negative_composite_mesh_and_parameter_compatibility(self):
        p = protocol()
        p['parameter_set'] = 'Chen2020_composite'
        with self.assertRaises(ValueError):
            validate_protocol(p)
        p['options']['particle phases'] = ['2', '1']
        with self.assertRaises(ValueError):
            validate_protocol(p)  # old r_n key is insufficient
        p['mesh_points'] = dict(x_n=5, x_s=5, x_p=5, r_n_prim=10, r_n_sec=10, r_p=10)
        self.assertIs(validate_protocol(p), p)
        p['mesh_points']['r_n'] = 10
        with self.assertRaises(ValueError):
            validate_protocol(p)  # do not silently mix one- and two-phase radii
        del p['mesh_points']['r_n']
        p['parameter_set'] = 'OKane2022'
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_unknown_particle_phase_values_are_rejected(self):
        for phases in ('2', ['3', '1'], ['1', '2'], [2, 1], ['2'], True):
            p = protocol(); p['options']['particle phases'] = phases
            with self.assertRaises(ValueError):
                validate_protocol(p)

    def test_json_tuple_options_preserve_exact_phase_meaning(self):
        options = {'thermal': 'isothermal', 'particle phases': ['2', '1'],
                   'open-circuit potential': [['single', 'current sigmoid'], 'single']}
        result = normalized_model_options(options)
        self.assertEqual(result['particle phases'], ('2', '1'))
        self.assertEqual(result['open-circuit potential'], (('single', 'current sigmoid'), 'single'))
        self.assertEqual(options['particle phases'], ['2', '1'])
        p = protocol(); p['options']['particle phases'] = ['1', '1']
        self.assertIs(validate_protocol(p), p)

    def test_composite_redundant_geometry_resolution_is_derived_not_user_mixed(self):
        p = protocol()
        p['mesh_points'] = dict(x_n=5, x_s=5, x_p=5, r_n_prim=10, r_n_sec=12, r_p=10)
        points = runtime_mesh_points(p)
        self.assertEqual(points['r_n'], 10)
        self.assertEqual(points['r_n_sec'], 12)
        self.assertNotIn('r_n', p['mesh_points'])
        self.assertEqual(runtime_mesh_points(protocol()), protocol()['mesh_points'])


if __name__ == '__main__':
    unittest.main()
