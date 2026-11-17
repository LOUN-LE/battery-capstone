import unittest
from audit_history_gaps import classify


class GapTests(unittest.TestCase):
    def case(self, seconds=.01, same=False):
        from datetime import datetime, timedelta
        start = datetime(2020, 1, 1)
        gap = dict(cell='C', from_segment='a', to_segment='b', duration_s=str(seconds),
                   start_naive=start.isoformat(), end_naive=(start+timedelta(seconds=seconds)).isoformat())
        segments = {'a': dict(cell='C', source_file='one', kind='rpt'),
                    'b': dict(cell='C', source_file='one' if same else 'two', kind='rpt')}
        return gap, segments

    def test_tiny_transition_never_proves_rest(self):
        row = classify(*self.case())
        self.assertEqual(row['category'], 'subsecond_file_transition')
        self.assertFalse(row['admitted_as_rest'])
        self.assertFalse(row['operation_resolved'])

    def test_anomaly_distinct_from_external_storage(self):
        self.assertEqual(classify(*self.case(7, True))['category'], 'within_file_record_loss')

    def test_long_gap_remains_unknown(self):
        row = classify(*self.case(90000))
        self.assertEqual(row['category'], 'long_unobserved_exposure')
        self.assertFalse(row['temperature_resolved'])

    def test_mismatched_clock_refused(self):
        g, s = self.case(); g['duration_s'] = '20'
        with self.assertRaises(ValueError): classify(g, s)

    def test_cross_cell_refused(self):
        g, s = self.case(); s['b']['cell'] = 'D'
        with self.assertRaises(ValueError): classify(g, s)
