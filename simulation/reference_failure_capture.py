"""Preserve available Engine.reference failure output without changing its verdict.

This helper is separate from the completed numerical-v1 execution. It must be
called explicitly by future callers; earlier runs did not use it.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


RAW_NAMES = ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_sims(engine):
    for key, simulation in getattr(engine, 'sims', {}).items():
        if isinstance(key, tuple) and key and key[0] == 'reference':
            yield key, simulation


def _solution(simulation):
    try:
        return simulation.solution
    except Exception:
        return None


def _pieces(solution):
    cycles = getattr(solution, 'cycles', None)
    found = False
    if cycles:
        for cycle_index, cycle in enumerate(cycles):
            if cycle is None:
                continue
            steps = getattr(cycle, 'steps', None)
            if steps:
                for step_index, step in enumerate(steps):
                    if step is not None:
                        found = True
                        yield cycle_index, step_index, step
            else:
                found = True
                yield cycle_index, None, cycle
    if not found:
        yield None, None, solution


def capture_available_reference_outputs(engine, output, prior_solution_ids):
    """Capture every available field; explicitly flag unchanged cached solutions."""
    output = Path(output)
    records = []
    extraction_errors = []
    current_files = cached_files = 0
    for sim_index, (key, simulation) in enumerate(_reference_sims(engine)):
        solution = _solution(simulation)
        if solution is None:
            continue
        # PyBaMM may leave the previous successful solution cached after a solve
        # fails. Those bytes are useful history but not evidence of this failure.
        is_current = prior_solution_ids.get(id(simulation)) != id(solution)
        attribution = 'current_call_saved_solution' if is_current else 'preexisting_cached_solution_not_attributed_to_current_failure'
        for cycle_index, step_index, piece in _pieces(solution):
            raw = {}
            missing = {}
            for name in RAW_NAMES:
                try:
                    raw[name] = np.asarray(piece[name].entries, dtype=float)
                except Exception as error:
                    missing[name] = f'{type(error).__name__}: {error}'
            if not raw:
                extraction_errors.append({'simulation_key': repr(key), 'cycle_index': cycle_index,
                                          'step_index': step_index, 'unavailable_fields': missing})
                continue
            stem = f"sim{sim_index:02d}_cycle{cycle_index if cycle_index is not None else 'none'}_step{step_index if step_index is not None else 'none'}.npz"
            path = output/stem
            np.savez_compressed(path, **raw)
            times = raw.get('Time [s]', np.array([])).reshape(-1)
            interval = bool(len(times) >= 2 and np.all(np.isfinite(times)) and times[-1] > times[0])
            records.append({'file': stem, 'sha256': _sha(path), 'simulation_key': repr(key),
                            'simulation_type': type(simulation).__name__, 'solution_type': type(solution).__name__,
                            'cycle_index': cycle_index, 'step_index': step_index,
                            'attribution': attribution, 'available_fields': list(raw),
                            'unavailable_fields': missing, 'time_samples': len(times),
                            'has_positive_time_interval': interval,
                            'all_saved_arrays_finite': all(np.all(np.isfinite(value)) for value in raw.values()),
                            'reported_termination': str(getattr(piece, 'termination', 'unavailable'))})
            if is_current:
                current_files += 1
            else:
                cached_files += 1
    availability = ('current_call_samples_available' if current_files else
                    'cached_samples_only_not_attributed_to_current_failure' if cached_files else 'no_samples_available')
    return {'sample_availability': availability, 'current_call_sample_files': current_files,
            'cached_prior_sample_files': cached_files, 'records': records,
            'extraction_errors': extraction_errors}


def reference_with_failure_capture(engine, main, *, output, rest_s=None, context=None):
    """Return Engine.reference unchanged, or re-raise its original exception.

    A receipt path is added as an exception note/attribute. Capturing a failed
    experiment never changes it into a successful reference measurement.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior = {id(simulation): id(solution) for _, simulation in _reference_sims(engine)
             if (solution := _solution(simulation)) is not None}
    receipt_path = output/'reference_receipt.json'
    try:
        result = engine.reference(main, rest_s=rest_s) if rest_s is not None else engine.reference(main)
    except Exception as original:
        receipt = {'status': 'reference_failed', 'error_type': type(original).__name__,
                   'error_message': str(original), 'traceback': traceback.format_exc(),
                   'engine_type': type(engine).__name__, 'requested_rest_s': rest_s,
                   'caller_context': context or {}, 'helper_code_sha256': _sha(__file__),
                   'physical_validation_complete': False,
                   'source_may_include_starting_solution_context': True,
                   'note': 'Saved simulator output can include starting-solution context, partial phases or a complete wrong-endpoint experiment; it is not an accepted RPT'}
        try:
            receipt.update(capture_available_reference_outputs(engine, output, prior))
            receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
        except Exception as capture_error:
            receipt['capture_error'] = f'{type(capture_error).__name__}: {capture_error}'
            receipt.setdefault('sample_availability', 'capture_failed_availability_unknown')
            try:
                receipt_path.write_text(json.dumps(receipt, indent=2, default=str), encoding='utf-8')
            except Exception as write_error:
                original.add_note(f'Failure receipt could not be written: {type(write_error).__name__}: {write_error}')
        if receipt_path.exists():
            original.add_note(f'Reference failure samples/context receipt: {receipt_path.resolve()}')
            try:
                original.reference_failure_receipt_path = str(receipt_path.resolve())
            except Exception:
                pass
        raise
    receipt_path.write_text(json.dumps({'status': 'reference_completed',
        'helper_code_sha256': _sha(__file__), 'physical_validation_complete': False,
        'note': 'Original return value is unchanged; the caller retains responsibility for normal raw output and scientific validation'}, indent=2), encoding='utf-8')
    return result
