"""Read the Imperial D-cell history for a conditional RPT0–RPT1 replay."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
import sys


SCENARIO_DIRECTORY = Path(
    "work/validation_proposal_20261103/latest_spme_review_20261103/"
    "raw_exposure_scenarios_v1"
)


def _sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


@dataclass
class DRoute:
    plan: tuple
    scenario: dict
    fingerprints: dict
    initial_anchor: dict
    summary: dict
    _arrays: object = field(repr=False)
    _waveform: object = field(repr=False)

    def verify_fingerprints(self):
        """Check the local inputs used to construct this route against their hashes."""
        for name, expected in self.fingerprints.items():
            if _sha256(Path(name)) != expected:
                raise ValueError(f"D route input changed: {name}")

    def steps(self):
        """Yield independent control specifications in instrument-clock order."""
        for index, entry in enumerate(self.plan):
            source = entry["source_step"]
            kind = entry["kind"]
            observed_s = source["duration_us"] / 1e6
            spec = {
                "index": index,
                "kind": kind,
                "source_ref": entry["source_ref"],
                "duration_s": observed_s,
                "observed_duration_s": observed_s,
                "canonical_duration_us": source["duration_us"],
                "absolute_start_naive": source["absolute_start_naive"],
                "absolute_end_naive": source["absolute_end_naive"],
                "termination": (),
                "timed_endpoint": True,
                "measurement": entry.get("measurement"),
                "temperature_C": None,
                "assumption_id": source.get("assumption_id"),
                "raw_kind": source["raw_kind"],
            }
            if kind == "native_RPT_control":
                native = entry["spec"]
                spec.update(
                    control=native["operation"],
                    value=native["value"],
                    duration_s=native["duration_s"],
                    termination=tuple(native["termination"]),
                    timed_endpoint=native["timed_endpoint"],
                    native_limits=tuple(native["native_limits"]),
                )
            elif kind == "eligible_recorded_current":
                table = self._waveform(source, self._arrays.get(source))
                table.setflags(write=False)
                spec.update(control="current", value=table, duration_s=float(table[-1, 0]))
            elif kind == "recorded_rest_controller":
                raw = self._arrays.get(source)
                rows = slice(source["row_start"], source["row_end_exclusive"])
                if source["recorded_rest_current_A"] != 0 or not (raw["mode"][rows] == 3).all():
                    raise ValueError(f"Mode 3 rest changed at {entry['source_ref']}")
                spec.update(control="rest", value=None)
            elif kind == "assumed_exposure":
                spec.update(control="rest", value=None, temperature_C=source["assumed_temperature_C"])
            else:
                raise ValueError(f"Unexpected D route control kind {kind}")
            yield spec

    def __iter__(self):
        return self.steps()


def _audit_plan(plan, gap_temperature_C):
    previous_end = None
    counts = Counter()
    anchors = {}
    by_kind_us = Counter()
    for index, entry in enumerate(plan):
        kind = entry["kind"]
        source = entry["source_step"]
        start = datetime.fromisoformat(source["absolute_start_naive"])
        end = datetime.fromisoformat(source["absolute_end_naive"])
        duration_us = source["duration_us"]
        measured_us = round((end - start).total_seconds() * 1e6)
        if duration_us <= 0 or abs(duration_us - measured_us) > 1:
            raise ValueError(f"Invalid instrument duration at {entry['source_ref']}")
        if previous_end is not None and start != previous_end:
            raise ValueError(f"Instrument interval is discontinuous at {entry['source_ref']}")
        previous_end = end
        counts[kind] += 1
        by_kind_us[kind] += duration_us
        if kind == "assumed_exposure" and (
            source["assumed_operation"] != "rest"
            or source["assumed_current_A"] != 0
            or source["assumed_temperature_C"] != gap_temperature_C
        ):
            raise ValueError(f"Undeclared exposure assumption at {entry['source_ref']}")
        measurement = entry.get("measurement")
        if measurement is not None:
            rpt = measurement["rpt"]
            if measurement["cell"] != "D" or rpt in anchors:
                raise ValueError("Unexpected or duplicate RPT measurement")
            anchors[rpt] = {
                "index": index,
                "source_ref": entry["source_ref"],
                "capacity_Ah": measurement["performance_summary_C10_capacity_Ah"],
            }
    if list(anchors) != [0, 1] or anchors[1]["index"] != len(plan) - 1:
        raise ValueError("D route must finish at the first ageing-set RPT discharge")
    return {
        "cell_id": "D",
        "gap_temperature_C": gap_temperature_C,
        "planned_steps": len(plan),
        "step_counts": dict(counts),
        "duration_us_by_kind": dict(by_kind_us),
        "measurements": anchors,
        "clock_basis": "Naive instrument clock; no timezone/DST conversion",
        "recorded_temperature_scope": "Native cell-surface temperature has no implied ambient boundary",
        "gap_policy": "Unobserved intervals use the scenario's zero-current rest and temperature",
        "physical_validation_complete": False,
        "measured_history_admissible": False,
    }


def get_route(root, gap_temperature_C=25):
    """Compile the D-cell route through RPT1 without building or solving a model."""
    if gap_temperature_C not in (10, 25, 40):
        raise ValueError("The retained D-cell scenarios use 10, 25 or 40 °C")
    temperature = int(gap_temperature_C)
    root = Path(root).resolve()
    simulation = root / "repo/simulation"
    if not (simulation / "run_corrected_history_trial.py").is_file():
        raise FileNotFoundError(f"Legacy capstone simulation unavailable at {simulation}")
    if str(simulation) not in sys.path:
        sys.path.insert(0, str(simulation))

    from initialize_spme_from_rest import load_native
    from joint_spme_bol_fit import select_rest_anchor
    from run_corrected_history_trial import compile_route, waveform

    path = root / SCENARIO_DIRECTORY / f"D_all_unresolved_rest_{temperature}C.json"
    scenario, raw_path, plan, zero, arrays, programs = compile_route(root, path, 1)
    if scenario["cell_id"] != "D":
        raise ValueError("Expected Imperial Expt 2,2 cell D")
    summary = _audit_plan(plan, temperature)
    summary["zero_span_annotations"] = len(zero)

    phases, source = load_native(root, "D")
    rpt0 = plan[summary["measurements"][0]["index"]]["measurement"]
    if source["sha256"] != rpt0["MPT_SHA256"]:
        raise ValueError("The D initial rest must come from the compiled RPT0 source")
    anchor_voltage, policy = select_rest_anchor(phases[0])
    initial_anchor = {
        "cell_id": "D",
        "target_voltage_V": anchor_voltage,
        "anchor_policy_id": "declared_rest_interior_last3_v1",
        "declared_rest_duration_s": 120.0,
        "anchor_policy": policy,
        "source_D": source,
        "raw_preserved": True,
    }
    control_sources = (
        "run_corrected_history_trial.py",
        "run_exposure_scenario_smoke.py",
        "initialize_spme_from_rest.py",
        "joint_spme_bol_fit.py",
        "native_program_contracts.py",
        "conditional_history_inputs.py",
    )
    inputs = (Path(__file__).resolve(), path, raw_path, programs, *(simulation / name for name in control_sources))
    fingerprints = {str(file): _sha256(file) for file in inputs}
    fingerprints[source["path"]] = source["sha256"]
    for step in plan:
        array_ref = step.get("native_source", step["source_step"])
        if "arrays_file" in array_ref:
            file = (raw_path.parent / array_ref["arrays_file"]).resolve()
            file.relative_to(raw_path.parent)
            digest = array_ref["arrays_sha256"]
            if fingerprints.setdefault(str(file), digest) != digest:
                raise ValueError(f"Conflicting native D-input hashes for {file}")
    scenario_metadata = {
        key: scenario[key] for key in (
            "scenario_id", "cell_id", "data_kind", "policies", "policies_sha256",
            "raw_reconstruction_sha256", "measured_history_admissible",
        )
    }
    return DRoute(tuple(plan), scenario_metadata, fingerprints, initial_anchor, summary, arrays, waveform)
