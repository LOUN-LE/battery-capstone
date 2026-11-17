import unittest
from conditional_history_inputs import native_step,require_uniform_recorded_mode,recorded_mode_blocks
from conditional_ageing_calibration import normalized_errors,controller_boundary_empty_allowed,validate_replay_input_integrity


class ConditionalHistoryTests(unittest.TestCase):
    def test_mixed_mode_Ns_cannot_be_compiled_as_majority_CC(self):
        with self.assertRaisesRegex(ValueError,'raw subsegment review'):
            require_uniform_recorded_mode('1x1756;3x63','F025#r00103')
        self.assertEqual(require_uniform_recorded_mode('1x1756','pure_CC'),1)

    def test_sampling_gap_is_reviewed_not_filled_as_rest(self):
        r=recorded_mode_blocks([0.,.1,.102,3600.102,7200.102,7200.2,7201.2],
                              [1,1,3,3,3,1,1],max_sample_gap_s=60.)
        self.assertEqual([b['recorded_mode'] for b in r['blocks']],[1,3,1])
        self.assertEqual(len(r['sampling_intervals_for_review']),2)
        self.assertTrue(all(g['operation_between_samples'] is None for g in r['sampling_intervals_for_review']))
        self.assertFalse(r['physical_replay_admissible'])

    def test_clock_reversal_splits_a_same_mode_block(self):
        r=recorded_mode_blocks([100.,101.,0.,1.],[1,1,1,1],max_sample_gap_s=60.)
        self.assertEqual(len(r['blocks']),2)
        self.assertEqual(r['clock_breaks'][0]['dt_s'],-101.)
        self.assertEqual(r['blocks'][0]['next_boundary_kind'],'clock_break')
        self.assertIsNone(r['blocks'][0]['next_mode_boundary_bracket_s'])

    def test_charge_limit_preserves_pulse_and_native_capacity(self):
        c=dict(operation='current',value=5,limits=[dict(quantity='absolute_step_charge_Ah',value=.73)])
        step=native_step(c,9999)
        self.assertAlmostEqual(step['duration_s'],525.6)
        self.assertEqual(step['value'],5)
        self.assertTrue(step['timed_endpoint'])

    def test_full_capacity_not_forced_to_observed_duration(self):
        c=dict(operation='current',value=.5,limits=[dict(quantity='voltage_V',value=2.5)])
        s=native_step(c,34794)
        self.assertEqual(s['termination'],['2.5 V'])
        self.assertFalse(s['timed_endpoint'])
        self.assertGreater(s['duration_s'],34794)

    def test_soh_uses_each_fixed_initial_capacity(self):
        errors=normalized_errors([5,4.5],[4,3.2])
        self.assertEqual(errors[0],0)
        self.assertAlmostEqual(errors[1],10)

    def test_incomplete_fit_series_refused(self):
        with self.assertRaises(ValueError):normalized_errors([5],[4])

    def test_cc_time_and_charge_take_first_endpoint(self):
        c=dict(operation='current',value=-1.5,limits=[dict(quantity='time_s',value=1752),
            dict(quantity='absolute_step_charge_Ah',value=.73),dict(quantity='voltage_V',value=4.2)])
        s=native_step(c,900)
        self.assertEqual(s['duration_s'],1752)
        self.assertEqual(s['termination'],['4.2 V'])

    def test_unresolved_capacity_uses_observed_exposure_not_false_unit(self):
        c=dict(operation='current',value=5,limits=[dict(quantity='unresolved_native_charge_reference',value=.15)])
        self.assertEqual(native_step(c,525.406025)['duration_s'],525.406025)

    def test_voltage_limited_pulse_mismatch_is_logged_but_capacity_cannot_skip(self):
        s=dict(operation='current',termination=['4.2 V'],native_limits=[dict(quantity='voltage_V',value=4.2)])
        self.assertTrue(controller_boundary_empty_allowed(s,is_capacity_measurement=False))
        self.assertFalse(controller_boundary_empty_allowed(s,is_capacity_measurement=True))
        self.assertFalse(controller_boundary_empty_allowed(dict(operation='rest',termination=[],native_limits=[]),is_capacity_measurement=False))

    def test_legacy_Ns_aggregated_cloud_inputs_cannot_restart(self):
        p=dict(schema_version=1,data_kind='conditional_observed_program_route',measured_history_admissible=False)
        with self.assertRaisesRegex(ValueError,'Legacy'):validate_replay_input_integrity(p)

    def test_new_schema_without_raw_boundary_accounting_cannot_run(self):
        p=dict(schema_version=2,data_kind='conditional_observed_program_route',measured_history_admissible=False,
               exposure_integrity=dict(raw_mode_boundaries_preserved=True))
        with self.assertRaisesRegex(ValueError,'integrity'):validate_replay_input_integrity(p)


if __name__=='__main__':unittest.main()
