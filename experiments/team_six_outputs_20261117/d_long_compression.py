"""Bounded waveform reduction for verified constant-current steps in D's ageing route."""

from hashlib import file_digest
from itertools import islice
import json
from pathlib import Path


def _sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def compression_fingerprints():
    """Return the exact adapter and legacy compression code hashes for run provenance."""
    import bounded_current_waveform

    paths = (Path(__file__).resolve(), Path(bounded_current_waveform.__file__).resolve())
    return {str(path): _sha256(path) for path in paths}


def _read_route_input(route, filename):
    matches = [(Path(name), expected) for name, expected in route.fingerprints.items()
               if Path(name).name == filename]
    if len(matches) != 1:
        raise ValueError(f"Expected one fingerprinted route input named {filename}")
    path, expected = matches[0]
    payload = path.read_bytes()
    from hashlib import sha256
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"D route input changed: {path}")
    return json.loads(payload.decode("utf-8-sig"))


def compressed_steps(route, start_index=0):
    """Yield route controls with original clocks and bounded ageing-current approximations."""
    from bounded_current_waveform import compress, verified_setpoint

    if route.scenario["cell_id"] != "D":
        raise ValueError("The waveform adapter is scoped to Imperial cell D")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index <= len(route.plan):
        raise ValueError("Resume index is outside the D plan")
    raw_steps = _read_route_input(route, "cell_D.json")["cell"]["steps"]
    programs = {program["whole_file_sha256"]: program
                for program in _read_route_input(route, "decoded_programs_raw.json")}
    for spec in islice(route.steps(), start_index, None):
        if spec["kind"] == "eligible_recorded_current":
            source = route.plan[spec["index"]]["source_step"]
            native = raw_steps[source["raw_step_index"]]
            setpoint = verified_setpoint(native, source, route._arrays.get(source), programs)
            if setpoint is not None:
                table, audit = compress(spec["value"], setpoint)
                spec["compression"] = audit
                if audit["compressed"]:
                    table.setflags(write=False)
                    spec["value"] = table
        yield spec
