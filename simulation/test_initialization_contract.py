import unittest
from initialization_contract import validate_initialization_anchor


class InitializationContractTests(unittest.TestCase):
    def fixture(self):
        return dict(anchor_policy_id='declared_rest_interior_last3_v1',declared_rest_duration_s=120.,
                    effective_initial_soc=.37,source_C=dict(path='native.mpt',sha256='a'*64),raw_preserved=True,
                    target_voltage_V=3.62,anchor_policy=dict(policy_id='declared_rest_interior_last3_v1',
                        selected_times_s=[90.,100.,110.],selected_voltage_V=[3.62]*3))

    def test_reviewed_anchor_passes(self):
        validate_initialization_anchor(self.fixture())

    def test_retired_last_three_boundary_anchor_rejected(self):
        value=self.fixture();value.pop('anchor_policy_id')
        with self.assertRaisesRegex(ValueError,'Retired'):
            validate_initialization_anchor(value)

    def test_nonfinite_initial_state_rejected(self):
        for soc in [True,float('nan'),float('inf'),-.1,1.1]:
            value=self.fixture();value['effective_initial_soc']=soc
            with self.subTest(soc=soc),self.assertRaises(ValueError):
                validate_initialization_anchor(value)

    def test_changed_native_duration_rejected(self):
        value=self.fixture();value['declared_rest_duration_s']=3600.
        with self.assertRaises(ValueError):validate_initialization_anchor(value)

    def test_unbound_observation_rejected(self):
        value=self.fixture();value['source_C']={}
        with self.assertRaises(ValueError):validate_initialization_anchor(value)

    def test_next_control_boundary_observation_rejected(self):
        value=self.fixture();value['anchor_policy']['selected_times_s'][-1]=120.044
        with self.assertRaises(ValueError):validate_initialization_anchor(value)

    def test_relabelled_old_anchor_mean_rejected(self):
        value=self.fixture();value['target_voltage_V']=3.63427
        with self.assertRaises(ValueError):validate_initialization_anchor(value)


if __name__ == '__main__':unittest.main()
