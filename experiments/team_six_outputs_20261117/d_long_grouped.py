"""Join adjacent D ageing-current controls without dropping short transitions."""

import numpy as np


MAX_PHASES = 128
MAX_TRANSITION_S = 0.01
MAX_RAMP_S = 1e-6
TEMPERATURE_C = 25.0


def _is_current(spec):
    return (
        spec["kind"] == "eligible_recorded_current"
        and spec["control"] == "current"
        and spec["temperature_C"] in (None, TEMPERATURE_C)
        and spec["measurement"] is None
        and not spec["termination"]
    )


def _is_transition(spec):
    return (
        spec["kind"] == "assumed_exposure"
        and spec["raw_kind"] in ("Ns_transition_interval", "controller_transition_interval")
        and spec["control"] == "rest"
        and spec["value"] is None
        and spec["temperature_C"] == TEMPERATURE_C
        and 0 < spec["duration_s"] <= MAX_TRANSITION_S
    )


def _join(specs):
    durations = np.array([spec["duration_s"] for spec in specs], dtype=float)
    if not np.isfinite(durations).all() or not (durations > 0).all():
        raise ValueError("Grouped controls need finite positive durations")
    offsets = np.r_[0.0, np.cumsum(durations)]
    if not (np.diff(offsets) > 0).all():
        raise ValueError("Grouped phase offsets are not strictly increasing")

    phases, original_charges = [], []
    for spec, (start, end) in zip(specs, zip(offsets[:-1], offsets[1:]), strict=True):
        if _is_transition(spec):
            phases.append(np.array([[start, 0.0], [end, 0.0]]))
            original_charges.append(0.0)
            continue
        table = spec["value"]
        if (
            table.ndim != 2 or table.shape[1] != 2 or len(table) < 2
            or not np.isfinite(table).all() or not (np.diff(table[:, 0]) > 0).all()
            or table[0, 0] != 0 or table[-1, 0] != spec["duration_s"]
        ):
            raise ValueError(f"Invalid waveform for grouped control {spec['source_ref']}")
        original_charges.append(float(np.trapezoid(table[:, 1], table[:, 0]) / 3600))
        phase = table.copy()
        phase[:, 0] += start
        phase[-1, 0] = end
        phases.append(phase)

    charge_bounds, ramps, grouped_charges, parts = [], [], [], []
    for index, phase in enumerate(phases):
        charge_bound = 0.0
        if index + 1 < len(phases) and phase[-1, 1] != phases[index + 1][0, 1]:
            # Interpolate the unmodified source at the start of the ramp. Only
            # this final fraction of the phase is altered by an endpoint jump.
            last_time, last_value = phase[-1]
            next_value = phases[index + 1][0, 1]
            available_s = last_time - phase[-2, 0]
            ramp_start = last_time - min(MAX_RAMP_S, available_s * 0.5)
            width = float(last_time - ramp_start)
            if not phase[-2, 0] < ramp_start < last_time or width > MAX_RAMP_S + np.spacing(last_time):
                raise ValueError("Cannot represent an audited current transition")
            source_value = float(np.interp(ramp_start, phase[:, 0], phase[:, 1]))
            phase = np.vstack((phase[:-1], (ramp_start, source_value), (last_time, next_value)))
            charge_bound = width * abs(next_value - last_value) / 7200
            ramps.append(width)
        grouped_charges.append(float(np.trapezoid(phase[:, 1], phase[:, 0]) / 3600))
        charge_bounds.append(charge_bound)
        parts.append(phase if index == 0 else phase[1:])

    table = np.vstack(parts)
    if not (np.diff(table[:, 0]) > 0).all():
        raise ValueError("Grouped waveform knots are not strictly increasing")
    original, grouped = sum(original_charges), sum(grouped_charges)
    residual = grouped - original
    tolerance = 1e-12 * max(1.0, sum(map(abs, original_charges)))
    if any(abs(new - old) > bound + tolerance for old, new, bound in zip(
        original_charges, grouped_charges, charge_bounds, strict=True
    )) or abs(residual) > sum(charge_bounds) + tolerance:
        raise ValueError("Grouped waveform exceeded the charge error audit")
    table.setflags(write=False)
    spec = dict(specs[0])
    # A compressed individual phase's receipt does not describe the joined waveform.
    spec.pop("compression", None)
    spec.update(
        kind="grouped_current", value=table, temperature_C=TEMPERATURE_C,
        source_ref=f"{specs[0]['source_ref']} through {specs[-1]['source_ref']}",
        duration_s=float(offsets[-1]),
        observed_duration_s=sum(phase["observed_duration_s"] for phase in specs),
        canonical_duration_us=sum(phase["canonical_duration_us"] for phase in specs),
        absolute_end_naive=specs[-1]["absolute_end_naive"],
        grouped_original_count=len(specs),
        grouped_offsets_s=offsets.tolist(),
        grouped_original_phase_charge_Ah=original_charges,
        grouped_phase_charge_Ah=grouped_charges,
        grouped_original_charge_Ah=original,
        grouped_charge_Ah=grouped,
        grouped_charge_residual_Ah=residual,
        grouped_charge_error_bound_Ah=sum(charge_bounds) + tolerance,
        grouped_transition_ramp_count=len(ramps),
        grouped_max_transition_ramp_s=max(ramps, default=0.0),
    )
    return spec


def group_current_specs(specs):
    """Return solver controls paired with the original phases for grouped controls."""
    specs = list(specs)
    groups = []
    start = 0
    while start < len(specs):
        first = specs[start]
        last = start
        if _is_current(first):
            for next_index in range(start + 1, min(len(specs), start + MAX_PHASES)):
                prior, candidate = specs[next_index - 1], specs[next_index]
                if (prior["index"] + 1 != candidate["index"]
                        or prior["absolute_end_naive"] != candidate["absolute_start_naive"]):
                    break
                if _is_current(candidate):
                    last = next_index
                elif not _is_transition(candidate):
                    break
        if last == start:
            groups.append((first, None))
        else:
            originals = specs[start:last + 1]
            groups.append((_join(originals), originals))
        start = last + 1
    return groups
