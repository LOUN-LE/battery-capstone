import json
from pathlib import Path
import tempfile
import unittest

from audit_calibrated_short_protocol import short_steps, parameters_ready, guard, digest


class CalibratedShortProtocolTests(unittest.TestCase):
    def test_fixed_quantity_not_dynamic_soc_protocol(self):
        steps = short_steps()
        loops = [s for s in steps if s['phase'].startswith('loop')]
        self.assertEqual(len(loops), 12)
        self.assertEqual(sum(s['kind'] == 'rest' for s in steps), 2)
        self.assertEqual(sum(s['duration'] for s in steps if s['kind'] == 'rest'), 14400)
        for step in loops:
            self.assertAlmostEqual(abs(step['value'])*step['duration']/3600, .73)
            self.assertEqual(step['expected'], 'final time')
        total_discharge = sum(s['value']*s['duration']/3600 for s in steps
                              if s['kind'] == 'current' and s['value'] > 0)
        self.assertAlmostEqual(total_discharge, 5.11)

    def test_source_guard_rejects_changed_frozen_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen.json'
            path.write_text('original frozen parameter bytes')
            config = {'source_sha256': {str(path): digest(path)}}
            guard(config)
            path.write_text('modified parameters')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                guard(config)

    def test_unfinished_or_unconverged_source_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertFalse(parameters_ready(folder))
            (folder/'parameters_frozen.json').write_text(json.dumps({'C_only_fit': True}))
            (folder/'report.json').write_text(json.dumps({'status': 'bounded_BOL_calibration_and_screen_completed',
                'optimizer_converged_for_all_starts': False,
                'verification': {'fine_candidate': {'status': 'evaluated'}}}))
            self.assertFalse(parameters_ready(folder))


if __name__ == '__main__':
    unittest.main()
