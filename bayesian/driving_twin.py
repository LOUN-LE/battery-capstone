"""Evaluate matched future current waveforms from one virtual SPMe state."""

from __future__ import annotations

import numpy as np

from simulation.rebuilt_lifetime import Engine, fingerprint, pybamm


def _simulate(engine, initial, profile):
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2:
        raise ValueError("Expected a waveform with elapsed seconds and cell amperes")
    t, current = profile.T
    if not np.isfinite(profile).all() or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Waveform must be finite and start at zero with increasing time")
    if np.max(abs(current)) > 5.0:
        raise ValueError("Waveform exceeds the exploratory 5 A cell current limit")

    duration_s = float(t[-1])
    step = pybamm.step.current(profile, duration=duration_s, period=1,
                               skip_ok=False)
    key = "bayesian_drive"
    try:
        solution = engine.sim(key, [step]).solve(
            starting_solution=initial.copy(), calc_esoh=False)
    finally:
        engine.sims.pop(key, None)
    drive = solution.cycles[-1].steps[-1]
    elapsed = np.asarray(drive["Time [s]"].entries)
    elapsed = elapsed - elapsed[0]
    amps = np.asarray(drive["Current [A]"].entries)
    volts = np.asarray(drive["Voltage [V]"].entries)
    capacity = np.asarray(drive["Discharge capacity [A.h]"].entries)
    if (len(elapsed) < 2 or np.any(np.diff(elapsed) <= 0)
            or not all(np.isfinite(x).all() for x in (elapsed, amps, volts, capacity))
            or abs(elapsed[-1] - duration_s) > 1e-6
            or drive.termination != "final time"):
        raise ValueError("SPMe did not finish the complete trip")
    if not np.allclose(amps, np.interp(elapsed, t, current), atol=1e-6, rtol=0):
        raise ValueError("SPMe current differs from the cell current waveform")
    signed_ah = float(np.trapezoid(current, t) / 3600)
    residual_ah = float(capacity[-1] - capacity[0] - signed_ah)
    if abs(residual_ah) > 2e-6 + 1e-4 * float(np.trapezoid(abs(current), t) / 3600):
        raise ValueError("Integrated cell current differs from model capacity change")
    return solution.last_state.copy(), {
        "duration_s": duration_s,
        "signed_discharge_Ah": signed_ah,
        "charge_balance_residual_Ah": residual_ah,
        "net_cell_energy_Wh": float(np.trapezoid(amps * volts, elapsed) / 3600),
        "voltage_min_V": float(np.min(volts)),
        "voltage_max_V": float(np.max(volts)),
        "end_operating_soc": float(engine.soc(solution.last_state)),
    }


def evaluate_prediction(prediction, *, temperature_C=25.0):
    """Run the reference, GP mean and every joint GP draw through the same SPMe."""
    if prediction.context["cell_basis_id"] not in {
        "synthetic-m50t-single-cell", "okane2022-spme-single-cell",
    }:
        raise ValueError("The configured SPMe does not match the supplied cell basis")
    engine = Engine(family="SPMe", temperature_K=float(temperature_C) + 273.15)
    initial = engine.start()
    initial_id = fingerprint(initial)
    initial_rpt, _ = engine.reference(initial)
    q0 = initial_rpt["Qref_Ah"]
    profiles = {
        "matched_reference": [prediction.reference_profile],
        "gp_mean": [prediction.expected_profile],
        "joint_draws": prediction.sample_profiles,
    }
    results = {}
    for group, waveforms in profiles.items():
        cases = []
        for profile in waveforms:
            try:
                end, metrics = _simulate(engine, initial, profile)
                final_rpt, _ = engine.reference(end)
                capacity = final_rpt["Qref_Ah"]
                cases.append({
                    "status": "complete",
                    **metrics,
                    "post_rpt_capacity_Ah": capacity,
                    "one_trip_rpt_delta_pp": 100 * (q0 - capacity) / q0,
                })
            except (ValueError, pybamm.SolverError) as error:
                cases.append({"status": "failed", "reason": str(error)})
        results[group] = cases
    if fingerprint(initial) != initial_id:
        raise RuntimeError("A scenario modified the shared starting state")

    draws = results["joint_draws"]
    # A dropped draw would condition the estimate on solver success.
    energy = None
    if draws and all(case["status"] == "complete" for case in draws):
        values = [case["net_cell_energy_Wh"] for case in draws]
        energy = {
            "mean_Wh": float(np.mean(values)),
            "sample_min_Wh": min(values),
            "sample_max_Wh": max(values),
        }
    return {
        "scope": "exploratory virtual cell, one trip, no lifetime ranking",
        "model": "legacy OKane2022 SPMe with isothermal SEI and plating",
        "status": "complete" if all(
            case["status"] == "complete" for cases in results.values() for case in cases
        ) else "incomplete",
        "temperature_C": temperature_C,
        "initial_operating_soc": float(engine.soc(initial)),
        "initial_state_sha256": initial_id,
        "initial_rpt_capacity_Ah": q0,
        "rpt_interpretation": (
            "Diagnostic capacities are recorded from the post-trip states. "
            "RPT charging exposure can differ with the end SOC, so their differences "
            "are not used to compare trip-induced degradation."
        ),
        "cases": results,
        "next_trip_cell_energy_from_draws": energy,
    }
