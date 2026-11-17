import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from reference_failure_capture import reference_with_failure_capture, RAW_NAMES


class FakePiece:
    termination = 'final time'
    def __init__(self, missing=()):
        self.missing = missing
    def __getitem__(self, key):
        if key in self.missing:
            raise KeyError(key)
        return SimpleNamespace(entries=np.array([0., 1.]))


def fake_solution(missing=()):
    return SimpleNamespace(cycles=[SimpleNamespace(steps=[FakePiece(missing)])])


class CaptureTests(unittest.TestCase):
    def test_current_partial_samples_saved_and_original_error_reraised(self):
        original = ValueError('sentinel invalid RPT endpoint')
        engine = SimpleNamespace(sims={})
        def reference(main, **kwargs):
            engine.sims[('reference', 1)] = SimpleNamespace(solution=fake_solution())
            raise original
        engine.reference = reference
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'capture'
            with self.assertRaises(ValueError) as raised:
                reference_with_failure_capture(engine, None, output=out, rest_s=1)
            self.assertIs(raised.exception, original)
            report = json.loads((out/'reference_receipt.json').read_text())
            self.assertEqual(report['status'], 'reference_failed')
            self.assertEqual(report['sample_availability'], 'current_call_samples_available')
            with np.load(out/report['records'][0]['file']) as raw:
                self.assertEqual(set(raw.files), set(RAW_NAMES))
            self.assertEqual(original.reference_failure_receipt_path, str((out/'reference_receipt.json').resolve()))

    def test_preexisting_cached_solution_is_not_misattributed(self):
        engine = SimpleNamespace(sims={('reference', 1): SimpleNamespace(solution=fake_solution())})
        def reference(main):
            raise RuntimeError('solve failed before replacing old cached result')
        engine.reference = reference
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'capture'
            with self.assertRaises(RuntimeError):
                reference_with_failure_capture(engine, None, output=out)
            report = json.loads((out/'reference_receipt.json').read_text())
            self.assertEqual(report['sample_availability'], 'cached_samples_only_not_attributed_to_current_failure')
            self.assertEqual(report['current_call_sample_files'], 0)

    def test_genuinely_missing_samples_are_explicit(self):
        engine = SimpleNamespace(sims={})
        def reference(main):
            raise RuntimeError('no solution produced')
        engine.reference = reference
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'capture'
            with self.assertRaises(RuntimeError):
                reference_with_failure_capture(engine, None, output=out)
            report = json.loads((out/'reference_receipt.json').read_text())
            self.assertEqual(report['sample_availability'], 'no_samples_available')
            self.assertEqual(report['records'], [])

    def test_partial_field_availability_is_not_fabricated(self):
        engine = SimpleNamespace(sims={})
        def reference(main):
            engine.sims[('reference', 1)] = SimpleNamespace(solution=fake_solution(['Voltage [V]']))
            raise ValueError('some observables absent')
        engine.reference = reference
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'capture'
            with self.assertRaises(ValueError):
                reference_with_failure_capture(engine, None, output=out)
            report = json.loads((out/'reference_receipt.json').read_text())
            self.assertIn('Voltage [V]', report['records'][0]['unavailable_fields'])

    def test_success_return_value_is_unchanged(self):
        expected = ({'Qref_Ah': 5.}, {'raw': np.array([1.])})
        engine = SimpleNamespace(sims={}, reference=lambda main: expected)
        with tempfile.TemporaryDirectory() as directory:
            result = reference_with_failure_capture(engine, None, output=Path(directory)/'capture')
            self.assertIs(result, expected)


if __name__ == '__main__':
    unittest.main()
