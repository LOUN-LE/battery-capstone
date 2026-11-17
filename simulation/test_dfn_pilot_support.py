"""DFN isolation, fresh-state selection and checkpoint provenance checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import dfn_pilot_support as support
from diagnostic_model_family import make_model, validate_model_family, verify_rest_equations
from run_exposure_scenario_smoke import read, write, sha
from runtime_physics_identity import digest

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'work/validation_proposal_20261103/latest_spme_review_20261103/interstitial_history_pilot_20261105/contract_v2.json'
TEST_TMP = ROOT/'work/dfn_checkpoint_test_tmp'
TEST_TMP.mkdir(exist_ok=True)


class Values:
    def __init__(self, values): self.entries = np.asarray(values)


class FakeState:
    def __init__(self):
        self.all_ys = [np.array([[1.], [2.]])]
        self.t = np.array([5.])

    def __getitem__(self, key): return Values([0.])


class DFNPilotTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        out = root/'work/validation_proposal_20261103/latest_spme_review_20261103/dfn_chunk'
        out.mkdir(parents=True)
        content = dict(actual_parameter_values={'Dint': 1e-20}, runtime_versions={'python': 'test'}, source_sha256={})
        identity = dict(schema_version=1, sha256=digest(content), **content)
        report = dict(model_family='DFN', candidate_key=support.candidate_key('a'*64, 1., identity),
            contract_sha256='a'*64, runtime_identity_sha256=identity['sha256'], completed_steps=1, source_sha256={})
        plan = [{'kind': 'recorded_rest_controller'}, {'kind': 'native_RPT_control', 'measurement': {'rpt': 0}}]
        receipt = support.commit_checkpoint(out, FakeState(), report, plan, [{'step': 0}], [], identity)
        cp = Path(receipt['directory'])
        write(out/'run_receipt.json', {'status': 'worker_terminated_at_wall_budget'})
        write(out/'report.json', {'status': 'checked_budget_boundary', 'model_family': 'DFN'})
        return root, out, cp, receipt, identity, report, plan

    def test_family_selection_and_options_are_explicit(self):
        for invalid in (None, '', 'dfn', 'SPM', True, ['DFN']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): validate_model_family(invalid)
        options = {'thermal': 'isothermal', 'particle phases': ['2', '1']}
        original = copy.deepcopy(options)
        dfn = make_model('DFN', options)
        self.assertEqual(type(dfn).__name__, 'DFN')
        self.assertEqual(options, original)
        self.assertEqual(type(make_model('SPMe', {'thermal': 'isothermal'})).__name__, 'SPMe')

    def test_legacy_contract_cannot_run_as_DFN(self):
        with self.assertRaisesRegex(ValueError, 'family'): support.validate_contract(ORIGINAL, 1000.)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            contract = read(ORIGINAL)
            contract.update(model_family='DFN', role='C_RPT0_RPT1_conditional_DFN_interstitial_SEI_pilot')
            path = Path(tmp)/'contract.json'
            write(path, contract)
            support.validate_contract(path, 1000., sha(path), True)
            with self.assertRaisesRegex(ValueError, 'reviewed'): support.validate_contract(path, 1000., 'wrong', True)
            contract['model_family'] = 'SPMe'
            write(path, contract)
            with self.assertRaisesRegex(ValueError, 'DFN'): support.validate_contract(path, 1000.)

    def test_cross_family_manifest_is_rejected_before_pickle(self):
        for family in (None, 'SPMe', 'unknown'):
            with self.subTest(family=family), tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
                root, out, cp, receipt, identity, report, plan = self.fixture(tmp)
                manifest = read(cp/'manifest.json')
                manifest['model_family'] = family
                write(cp/'manifest.json', manifest)
                with patch.object(support.pickle, 'loads', side_effect=AssertionError('pickle must not load')) as load:
                    with self.assertRaisesRegex(ValueError, 'family'):
                        support.load_checkpoint(cp, sha(cp/'manifest.json'), report, plan, identity, root, None, {}, {})
                    load.assert_not_called()

    def test_checkpoint_component_and_family_are_both_bound(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root, out, cp, receipt, identity, report, plan = self.fixture(tmp)
            manifest = support.validate_manifest(cp, receipt['manifest_sha256'], report, plan, identity, root)
            self.assertEqual(manifest['model_family'], 'DFN')
            current = dict(report, model_family='SPMe')
            with self.assertRaisesRegex(ValueError, 'family'):
                support.validate_manifest(cp, receipt['manifest_sha256'], current, plan, identity, root)
            (cp/'terminal.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'component hash'):
                support.validate_manifest(cp, receipt['manifest_sha256'], report, plan, identity, root)

    def test_DFN_namespace_is_distinct_from_SPMe_candidate(self):
        from interstitial_pilot_support import candidate_key as old_key
        identity = {'sha256': 'a'*64}
        self.assertNotEqual(support.candidate_key('b'*64, 1000., identity), old_key('b'*64, 1000., identity))

    def test_DFN_rest_checkpoint_authenticates_and_rejects_SPMe(self):
        import pybamm
        options = {'thermal': 'isothermal'}
        parameters = pybamm.ParameterValues('Chen2020')
        mesh = {'x_n': 5, 'x_s': 5, 'x_p': 5, 'r_n': 5, 'r_p': 5}
        simulation = pybamm.Simulation(make_model('DFN', options), parameter_values=parameters.copy(), var_pts=mesh,
            experiment=pybamm.Experiment([pybamm.step.current(.1, duration=1),
                pybamm.step.voltage(4.2, duration=1), pybamm.step.rest(1)]),
            solver=pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-9))
        state = simulation.solve(calc_esoh=False).last_state
        proof = verify_rest_equations(state, parameters, options, mesh, model_family='DFN')
        self.assertTrue(proof['discretised_rest_equations_exact'])
        with self.assertRaisesRegex(ValueError, 'family'):
            verify_rest_equations(state, parameters, options, mesh, model_family='SPMe')


if __name__ == '__main__': unittest.main()
