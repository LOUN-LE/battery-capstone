import copy
from datetime import datetime
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compile_measured_history import (clock_admission, mapped_origin, source_key, make_segment,
                                       coverage_intervals, binary_only_segments, naive, main)


def clock_fixture():
    path = 'Raw Data/Performance Checks/BoL/Slow/C_RPT0_long_01.mpr'
    clock = {'cell': 'C', 'timestamp_naive': '2020-10-13T00:00:00',
             'MPT_pair_present': True, 'mapping_verified': True,
             'MPT_absolute_origin_naive': '2020-10-13T00:00:01',
             'time_offset_MPR_minus_MPT_median_s': 1.,
             'MPR_t_first_s': 1., 'MPR_t_last_s': 101.,
             'MPT_t_first_s': 0., 'MPT_t_last_s': 100.}
    row = {k: str(v) if v is not None else '' for k,v in clock.items()}
    row.update(source_path=path, local_file='files/native.mpr')
    audit = {'status': 'completed', 'sources_total': 1, 'download_verified_count': 1, 'parsed_count': 1,
             'available_MPT_pairs_expected': 1, 'MPT_pairs_verified': 1,
             'all_available_MPT_clock_mappings_verified': True,
             'files': [{'source_path': path, 'local_file': 'files/native.mpr', 'CRC32_verified': True,
                        'stage': 'clock_parsed', 'clock': clock}]}
    return row, audit


def run_row():
    return {'run_id': 'F1#r0', 'source_file': 'C_RPT0_long_01.mpt', 'cell': 'C',
            'kind': 'rpt', 'group': '0', 'part': 'long_01', 'row_start': '0', 'row_end_exclusive': '3',
            'analysis_valid': 'True', 't_first_s': '0', 't_last_s': '10',
            'control_type': 'rest', 'control_value': '', 'I_median_mA': '0',
            'charge_dq_mAh': '0'}


class ChronologyTests(unittest.TestCase):
    def test_complete_clock_audit_is_required(self):
        row, audit = clock_fixture()
        self.assertTrue(all(clock_admission([row], audit, {'C_RPT0_long_01.mpt'}).values()))
        audit['status'] = 'in_progress'
        self.assertFalse(all(clock_admission([row], audit, {'C_RPT0_long_01.mpt'}).values()))

    def test_drifted_CSV_origin_and_offset_cannot_override_audited_pair(self):
        row, audit = clock_fixture()
        row['MPT_absolute_origin_naive'] = '2020-10-13T00:00:02'
        row['time_offset_MPR_minus_MPT_median_s'] = '2'
        self.assertEqual(mapped_origin(row), datetime(2020,10,13,0,0,2))  # internally self-consistent
        checks = clock_admission([row], audit, {'C_RPT0_long_01.mpt'})
        self.assertFalse(checks['CSV_origin_matches_audited_clock'])
        self.assertFalse(checks['CSV_numerical_clock_evidence_matches_audit'])

    def test_full_BoL_Fast_and_Slow_stems_remain_distinct(self):
        self.assertNotEqual(source_key('C', 'C_RPT0_long_01.mpr'), source_key('C', 'C_RPT0_short_01.mpt'))
        self.assertEqual(source_key('C', 'folder/C_RPT0_long_01.mpr'), source_key('C','C_RPT0_long_01.mpt'))
        row, _ = clock_fixture()
        segment = make_segment(run_row(), row['source_path'], row)
        self.assertIn('BoL/Slow', segment['session_path'])
        self.assertEqual(segment['absolute_start_naive'], '2020-10-13T00:00:01')
        self.assertFalse(segment['programmed_cutoffs_known'])

    def test_aware_clock_and_unverified_mapping_are_rejected(self):
        with self.assertRaises(ValueError):
            naive('2020-10-13T00:00:00+01:00')
        row, _ = clock_fixture(); row['mapping_verified'] = 'False'
        with self.assertRaises(ValueError):
            mapped_origin(row)

    def test_six_zero_rows_remain_quarantined_not_rest(self):
        row = run_row()
        row.update(analysis_valid='False', kind='ageing', row_start='148694', row_end_exclusive='148700',
                   t_first_s='0', t_last_s='0', charge_dq_mAh='0',
                   abs_start='2021-02-05T00:00:00', abs_end='2021-02-05T00:00:00')
        segment = make_segment(row, 'cycling6.mpr')
        self.assertFalse(segment['analysis_valid'])
        self.assertIsNone(segment['absolute_start_naive'])
        self.assertIsNone(segment['net_recorded_charge_Ah'])
        self.assertEqual(segment['row_end_exclusive']-segment['row_start'], 6)

    def test_gaps_sampling_transitions_and_overlap_are_separate(self):
        def seg(id, start, end, a, b, file='same'):
            return {'segment_id': id,'cell':'C','source_file':file,'row_start':a,'row_end_exclusive':b,
                    'analysis_valid':True,'absolute_time_known':True,
                    'absolute_start_naive':f'2020-01-01T00:00:{start:02d}',
                    'absolute_end_naive':f'2020-01-01T00:00:{end:02d}'}
        records = coverage_intervals([seg('a',0,5,0,2),seg('b',6,10,2,4),seg('c',17,20,10,12),seg('d',19,25,12,15)])
        self.assertEqual([r['kind'] for r in records], ['inter_sample_transition_interval','unobserved_gap','recorded_interval_overlap'])
        self.assertEqual(records[1]['duration_s'],7)
        self.assertTrue(all(r['known_rest'] is False and r['inferred_operation'] is None for r in records))

    def test_binary_only_measured_rest_is_really_parsed_and_keeps_RPT_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'raw.mpr').write_bytes(b'verified binary fixture')
            timestamp = datetime(2020,1,1)
            row = {'cell':'C','local_file':'raw.mpr','timestamp_naive':timestamp.isoformat(),
                   'source_path':'Raw/BoL/Slow/C_RPT0_long_05.mpr'}
            def summary(*args):
                return {'row_start':0,'row_end_exclusive':2,'Ns':0,'t_first_s':0.,'t_last_s':21600.,
                        'duration_s':21600.,'duration_to_next_s':None,'abs_start':timestamp.isoformat(),
                        'abs_end':'2020-01-01T06:00:00','control_type':'rest', 'control_value':None,
                        'I_median_mA':0.,'charge_trapz_mAh':0.,'n_dt_nonpositive':0}
            common = SimpleNamespace(read_mpr=lambda path: ({'t':[0,21600]}, {}, timestamp),
                                     split_runs=lambda cols: ([(0,2)],'fixture'),
                                     describe=lambda path: {'group':0,'part':'long_05'}, summarize_run=summary)
            record = {'sha256': hashlib.sha256((root/'raw.mpr').read_bytes()).hexdigest()}
            result = binary_only_segments(row,record,root,common)
            self.assertEqual(result[0]['kind'],'rpt')
            self.assertEqual(result[0]['segment_role'],'rest')
            self.assertEqual(result[0]['recorded_sample_span_s'],21600)
            self.assertEqual(result[0]['source_kind'],'binary_only_MPR_run')

    def test_existing_output_directory_is_not_modified_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'existing'; path.mkdir()
            (path/'marker.txt').write_text('preserve me')
            with patch('sys.argv', ['compile_measured_history.py','--output',str(path)]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual({p.name for p in path.iterdir()}, {'marker.txt'})


if __name__ == '__main__':
    unittest.main()
