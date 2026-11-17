import unittest
import numpy as np
from reconstruct_native_exposure import partition_run,reconcile_clock_episodes


class RawModeReconstructionTests(unittest.TestCase):
    def test_stale_rest_entry_current_does_not_create_750mAh(self):
        r=partition_run([0.,.1,.102,3600.102,7200.102,7200.2,7201.2],[1,1,3,3,3,1,1],[7]*7,
                        [1500,1500,1500,0,0,1500,1500],[1500]*7,max_sample_gap_s=60)
        rest=[g for g in r['intervals'] if g['kind']=='sparse_rest_controller_interval']
        self.assertEqual(sum(g['duration_s'] for g in rest),7200.)
        self.assertTrue(rest[0]['recorded_mode3_current_conflict'])
        self.assertTrue(all(g['recorded_current_integral_Ah'] is None for g in rest))
        self.assertTrue(r['duration_reconciliation_passed'])
        self.assertAlmostEqual(r['accounted_span_s'],7201.2)
        self.assertFalse(r['current_drive_eligible_intervals'][2])

    def test_large_CC_gap_is_not_filled_with_CC(self):
        r=partition_run([0.,1.,3601.,3602.],[1]*4,[6]*4,[-5000]*4,[-5000]*4,max_sample_gap_s=60)
        g=r['intervals'][0]
        self.assertEqual(g['kind'],'unresolved_sparse_current_interval')
        self.assertIsNone(g['operation'])
        self.assertEqual(g['duration_s'],3600.)
        self.assertTrue(r['duration_reconciliation_passed'])

    def test_sign_conflicting_CC_entry_current_is_not_interpolated(self):
        r=partition_run([0.,1.,2.],[1]*3,[7]*3,[-5000,1500,1500],[1500]*3)
        self.assertEqual(r['current_drive_eligible_intervals'].tolist(),[False,True])
        self.assertAlmostEqual(r['blocks'][0]['recorded_current_integral_Ah_valid_intervals'],1500/3600000)

    def test_Ns_change_is_a_separate_uncertain_transition(self):
        r=partition_run([0.,1.,1.01,2.],[1]*4,[6,6,7,7],[-5000,-5000,1500,1500],[-5000,-5000,1500,1500])
        self.assertEqual([b['native_Ns'] for b in r['blocks']],[6,7])
        self.assertAlmostEqual(r['intervals'][0]['duration_s'],.01)
        self.assertIsNone(r['intervals'][0]['operation'])
        self.assertTrue(r['duration_reconciliation_passed'])

    def test_clock_reversal_cannot_claim_reconciled_physical_time(self):
        r=partition_run([100.,101.,0.,1.],[1]*4,[7]*4,[1500]*4,[1500]*4)
        self.assertFalse(r['duration_reconciliation_passed'])
        self.assertEqual(len(r['clock_breaks']),1)
        self.assertFalse(r['full_physical_replay_admissible'])

    def test_overlapping_clock_episodes_are_accounted_once_without_selecting_values(self):
        raw=partition_run([0.,1.,2.,1.,2.,3.],[1]*6,[7]*6,[1500]*6,[1500]*6)
        fixed=reconcile_clock_episodes(raw,0.,3.)
        self.assertTrue(fixed['duration_reconciliation_passed'])
        self.assertEqual(fixed['accounted_span_s'],3.)
        self.assertEqual(fixed['ambiguous_intervals'][0]['duration_s'],1.)
        self.assertIsNone(fixed['ambiguous_intervals'][0]['operation'])
        self.assertEqual(len(fixed['ambiguous_intervals'][0]['overlapping_raw_parts']),2)

    def test_duplicate_clock_point_is_preserved_without_extra_positive_time(self):
        raw=partition_run([0.,1.,1.,2.],[1]*4,[7]*4,[1500]*4,[1500]*4)
        fixed=reconcile_clock_episodes(raw,0.,2.)
        self.assertEqual(fixed['accounted_span_s'],2.)
        self.assertEqual(len(fixed['clock_breaks_retained']),1)

    def test_unresolved_clock_reset_outside_source_envelope_refused(self):
        raw=partition_run([100.,101.,0.,1.],[1]*4,[7]*4,[1500]*4,[1500]*4)
        with self.assertRaises(ValueError):reconcile_clock_episodes(raw,100.,1.)

    def test_raw_current_waveform_is_not_averaged(self):
        r=partition_run([0.,1.,2.],[2]*3,[2]*3,[1500.,1000.,500.])
        self.assertTrue(np.all(r['current_drive_eligible_intervals']))
        self.assertAlmostEqual(r['blocks'][0]['recorded_current_integral_Ah_valid_intervals'],2000/3600000)

    def test_invalid_source_shapes_refused(self):
        with self.assertRaises(ValueError):partition_run([0.,1.],[1],[7,7],[1500,1500])


if __name__=='__main__':unittest.main()
