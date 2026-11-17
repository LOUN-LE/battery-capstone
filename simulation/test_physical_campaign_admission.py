"""Independent admission checks; no real PDE, optimization, AWS or inferred rest."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from physical_campaign_inputs import canonical_sha256, check_history, validate_campaign, validate_split
from physical_ageing_campaign import validate_campaign as public_validate_campaign


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT/'work/validation_proposal_20261103'


def ref(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def step(operation, value=None, termination=None, expected='final time', role='rpt'):
    result = {'role': role, 'source_ref': 'explicit unit-test fixture, not real measured exposure',
              'operation': operation, 'duration_s': 10, 'period_s': 1,
              'temperature_K': 298.15, 'termination': termination or [], 'expected_termination': expected}
    if value is not None:
        result['value'] = value
    return result


def fixture(directory, cell='C', final_capacity=4.9, full=False):
    target = directory/f'{cell}_targets.csv'
    target.write_text(f'rpt_id,Qref_Ah\nRPT0,5.0\nRPT1,{final_capacity}\n', encoding='utf-8')
    if full:
        group = [step('current', -1.5, ['4.2 V'], 'event: Voltage > 4.2 [V] [experiment]'),
                 step('voltage', 4.2, ['0.05 A'], 'event: abs(Current [A]) < 0.05 [A] [experiment]'),
                 step('rest'), step('current', .5, ['2.5 V'], 'event: Voltage < 2.5 [V] [experiment]')]
        steps = copy.deepcopy(group)+copy.deepcopy(group)
        measurements = [{'rpt_id': f'RPT{i}', 'step_index': 4*i+3, 'kind': 'full_discharge_capacity',
                         'cutoff_V': 2.5, 'conditioning_steps': [4*i, 4*i+1, 4*i+2]} for i in range(2)]
    else:
        steps = [step('current', .5), step('rest', role='ageing'), step('current', .5)]
        measurements = [{'rpt_id': 'RPT0', 'step_index': 0, 'kind': 'timed_software_probe'},
                        {'rpt_id': 'RPT1', 'step_index': 2, 'kind': 'timed_software_probe'}]
    protocol = {'cell_model': 'explicit synthetic fixture', 'protocol_id': 'unit-protocol',
        'source_ref': 'unit fixture only', 'initial_state_source': 'known synthetic initial SOC',
        'family': 'SPMe', 'parameter_set': 'OKane2022', 'options': {'thermal': 'isothermal',
            'SEI': 'solvent-diffusion limited'}, 'mesh_points': dict(x_n=5,x_s=5,x_p=5,r_n=10,r_p=10),
        'initial_soc': .9, 'initial_temperature_K': 298.15, 'rtol': 1e-6, 'atol': 1e-8,
        'max_wall_s': 60, 'steps': steps}
    return {'schema_version': 1, 'data_kind': 'synthetic', 'cell_id': cell, 'condition_id': 'unit-condition',
            'protocol': protocol, 'measurement_steps': measurements, 'targets': ref(target)}


class PhysicalAdmissionTests(unittest.TestCase):
    def test_backend_public_gate_is_the_independently_tested_gate(self):
        self.assertIs(public_validate_campaign, validate_campaign)

    def test_valid_synthetic_admission_is_not_physical_validation(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            case = validate_campaign(fixture(Path(folder)), ROOT)
            self.assertEqual(case['_targets_rows'][0]['SOH_percent'], 100)
            self.assertEqual(case['data_kind'], 'synthetic')

    def test_current_actual_163_gap_history_is_refused(self):
        history = ROOT/'work/validation_proposal_20261103/measured_history_v2/report.json'
        self.assertEqual(json.loads(history.read_text(encoding='utf-8'))['interval_counts']['unobserved_gap'], 163)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            case = fixture(Path(folder), full=True)
            case.update(data_kind='measured', history_admission=ref(history))
            with self.assertRaisesRegex(ValueError, 'history admission.*gaps=163'):
                validate_campaign(case, ROOT)

    def test_same_raw_target_relabelled_as_other_cell_is_leakage(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            directory = Path(folder)
            calibration = validate_campaign(fixture(directory, 'C'), ROOT)
            heldout = validate_campaign(fixture(directory, 'D'), ROOT)
            self.assertNotEqual(calibration['cell_id'], heldout['cell_id'])
            self.assertEqual(calibration['targets']['sha256'], heldout['targets']['sha256'])
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_split(calibration, [heldout])

    def test_distinct_cell_distinct_native_source_split_is_admitted(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            directory = Path(folder)
            calibration = validate_campaign(fixture(directory, 'C', 4.9), ROOT)
            heldout = validate_campaign(fixture(directory, 'D', 4.8), ROOT)
            self.assertTrue(validate_split(calibration, [heldout]))

    def test_actual_RPT_positive_CC_is_required(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            original = fixture(Path(folder))
            for change in ({'value': -.5}, {'role': 'diagnostic'}, {'operation': 'rest'}):
                case = copy.deepcopy(original)
                case['protocol']['steps'][0].update(change)
                with self.assertRaisesRegex(ValueError, 'actual positive-current CC discharge RPT'):
                    validate_campaign(case, ROOT)

    def test_measured_cannot_mislabel_timed_probe_as_capacity(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            case = fixture(Path(folder)); case['data_kind'] = 'measured'
            with self.assertRaisesRegex(ValueError, 'synthetic only'):
                validate_campaign(case, ROOT)

    def test_measured_cannot_discard_recorded_step_as_virtual_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            case = fixture(Path(folder), full=True)
            case['data_kind'] = 'measured'
            case['protocol']['steps'].append(step('rest', role='diagnostic'))
            with self.assertRaisesRegex(ValueError, 'cannot discard diagnostic exposure'):
                validate_campaign(case, ROOT)

    def test_native_target_hash_drift_is_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            case = fixture(Path(folder))
            Path(case['targets']['path']).write_text('rpt_id,Qref_Ah\nRPT0,5\nRPT1,4.0\n')
            with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                validate_campaign(case, ROOT)

    def test_full_RPT_reference_condition_changes_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            original = fixture(Path(folder), full=True)
            validate_campaign(original, ROOT)
            for change in ({'value': -1.0}, {'termination': ['4.1 V']}):
                case = copy.deepcopy(original)
                case['protocol']['steps'][4].update(change)
                with self.assertRaises(ValueError):
                    validate_campaign(case, ROOT)

    def test_full_RPT_requires_actual_immediate_conditioning(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            original = fixture(Path(folder), full=True)
            for index, change in ((0, {'role': 'diagnostic'}), (1, {'operation': 'rest'}),
                                  (2, {'temperature_K': 300.15})):
                case = copy.deepcopy(original)
                case['protocol']['steps'][index].update(change)
                with self.assertRaisesRegex(ValueError, 'conditioning'):
                    validate_campaign(case, ROOT)

    def test_history_gate_uses_actual_compiler_overlap_key(self):
        # Contract fixture only, not a real measured-history admission or PDE run.
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            directory = Path(folder)
            source = directory/'synthetic_source.txt'; source.write_text('unit fixture only')
            history = directory/'synthetic_admission.json'
            record = {'status': 'recorded_history_compiled', 'compiler_completed': True,
                'clock_origin_audit_complete': True, 'clock_checks': {'unit_fixture': True},
                'input_hashes_unchanged': True, 'full_history_replay_admissible': True,
                'exact_control_protocol_admissible': True,
                'interval_counts': {'unobserved_gap': 0, 'recorded_interval_overlap': 0},
                'quarantined_runs': 0, 'unlocated_runs': 0, 'quarantined_raw_rows': 0,
                'source_files': {'source': str(source)}, 'source_sha256': {'source': ref(source)['sha256']}}
            history.write_text(json.dumps(record))
            checked = check_history(ref(history), ROOT)
            self.assertIn(str(source.resolve()), checked)

    def test_history_admission_cannot_authorize_a_changed_ordered_protocol(self):
        # Admission-contract fixture only. It never reaches a PDE or native-source fit.
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            directory = Path(folder)
            case = fixture(directory, full=True); case['data_kind'] = 'measured'
            source = directory/'source.txt'; source.write_text('unit contract fixture')
            record = {'status': 'recorded_history_compiled', 'compiler_completed': True,
                'clock_origin_audit_complete': True, 'clock_checks': {'unit_fixture': True},
                'input_hashes_unchanged': True, 'full_history_replay_admissible': True,
                'exact_control_protocol_admissible': True,
                'interval_counts': {'unobserved_gap': 0, 'recorded_interval_overlap': 0},
                'quarantined_runs': 0, 'unlocated_runs': 0, 'quarantined_raw_rows': 0,
                'source_files': {'source': str(source)}, 'source_sha256': {'source': ref(source)['sha256']},
                'admitted_protocols': {'C': {'protocol_sha256': canonical_sha256(case['protocol']),
                    'measurement_steps_sha256': canonical_sha256(case['measurement_steps']),
                    'ordered_source_refs': [s['source_ref'] for s in case['protocol']['steps']]}}}
            history = directory/'history.json'; history.write_text(json.dumps(record))
            case['history_admission'] = ref(history)
            case['protocol']['initial_soc'] = .8
            with self.assertRaisesRegex(ValueError, 'not bound to this cell and exact ordered protocol'):
                validate_campaign(case, ROOT)


if __name__ == '__main__':
    unittest.main()
