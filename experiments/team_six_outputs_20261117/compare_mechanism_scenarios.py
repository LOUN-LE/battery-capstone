"""Check four degradation switches on one measured window from a shared state."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import coupled_run as coupled
import numpy as np

twin = coupled.twin

SCENARIOS = (
    ("both_active", True, True),
    ("sei_growth_paused", False, True),
    ("plating_reactions_paused", True, False),
    ("both_paused", False, False),
)
SEI_RATE = "SEI solvent diffusivity [m2.s-1]"
PLATING_RATE = "Lithium plating kinetic rate constant [m.s-1]"
DEAD_LITHIUM_RATE = "Dead lithium decay constant [s-1]"
SEI_CURRENT = "X-averaged negative electrode SEI interfacial current density [A.m-2]"


def replay_scenario(parameters, config, state, observed, table, sei, plating):
    rates = parameters.copy()
    if not sei:
        rates[SEI_RATE] = 0.0
    if not plating:
        rates[PLATING_RATE] = 0.0
        rates[DEAD_LITHIUM_RATE] = 0.0
    _, trace, audit = twin.observed_step(rates, config, state, table)
    values = coupled.align(trace, table[:, 0])
    channels = ("sei_thickness_nm", "sei_loss_ah", "plated_li_mol_m3", "dead_li_mol_m3")
    changes = {key: float(values[key][-1] - values[key][0]) for key in channels}
    # Thickness is concentration divided by evolving electrode area.
    # Check SEI reaction current and consumed lithium, even if thickness moves.
    sei_current = float(np.max(np.abs(trace["sei_current_density_A_m2"])))
    drifts = {key: float(np.max(np.abs(trace[variable] - trace[variable][0])))
              for key, (variable, _, _) in coupled.EXPORT.items() if key in channels[1:]}
    if not sei and (sei_current > 1e-12 or drifts["sei_loss_ah"] > 1e-10):
        raise ValueError(f"Paused SEI reaction remained active: {sei_current}, {drifts['sei_loss_ah']}")
    if not plating and any(drifts[key] > 1e-9 for key in channels[2:]):
        raise ValueError(f"Paused plating inventory changed: {drifts}")
    return {
        "sei_growth_enabled": sei,
        "plating_reactions_enabled": plating,
        "changed_parameters": {key: rates[key] for key in (SEI_RATE, PLATING_RATE, DEAD_LITHIUM_RATE)
                               if rates[key] != parameters[key]},
        "voltage_rmse_mV": float(np.sqrt(np.mean((values["model_voltage_V"] - observed[:, 1]) ** 2)) * 1000),
        "model_changes": changes,
        "maximum_abs_sei_current_density_A_m2": sei_current,
        "maximum_state_drift": drifts,
        "numerics": audit,
    }


def run(input_path, out, names):
    started = time.perf_counter()
    preparation = coupled.DEFAULT_PREP
    flags = coupled.BASE / "configs/frozen_model.json"
    calibration = coupled.BASE / "configs/cell_calibration.json"
    rows, observed = twin.read_input(input_path)
    prep_rows, _ = twin.read_input(preparation)
    if rows[0]["cell_id"] != prep_rows[0]["cell_id"] or float(prep_rows[-1]["time_s"]) >= observed[0, 0]:
        raise ValueError("Expected measured observations after preparation for the same cell")
    relative_t = observed[:, 0] - observed[0, 0]
    if len(rows) != 591 or not 0 < relative_t[-1] <= 600.001:
        raise ValueError("Expected the selected 591-row initial performance-test window")
    table = np.column_stack((relative_t, observed[:, 2]))
    config = twin.read_json(flags)
    config["options"].update({
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "loss of active material": "stress-driven",
        "SEI on cracks": "false",
    })
    target, _ = twin.rest_anchor(prep_rows)
    parameters, _ = twin.make_parameters(config, twin.read_json(calibration), target)
    state, prep = twin.prepare_state(parameters, config, prep_rows)
    state_hash = twin.fingerprint(state)
    records = []
    with patch.dict(twin.VARIABLES, {**coupled.EXTRA, "sei_current_density_A_m2": SEI_CURRENT}):
        for name, sei, plating in SCENARIOS:
            if name not in names:
                continue
            began = time.perf_counter()
            record = {"scenario": name, "status": "failed"}
            try:
                result = replay_scenario(parameters, config, state, observed, table, sei, plating)
                if result["numerics"]["input_fingerprint"] != state_hash:
                    raise ValueError("Scenario did not start from the shared checkpoint")
                record.update(result, status="completed")
            except (ValueError, RuntimeError, coupled.pybamm.SolverError) as error:
                record["error"] = str(error)
            record["wall_s"] = time.perf_counter() - began
            records.append(record)
            print(f'{name} {record["status"]} {record["wall_s"]:.3f}s', flush=True)
    completed = sum(record["status"] == "completed" for record in records)
    twin.write_json(out, {
        "scope": "scenario switch verification on the same initial RPT0 window",
        "cell_id": rows[0]["cell_id"], "observed_rows": len(rows), "duration_s": float(relative_t[-1]),
        "ageing_cycles": 0, "physical_cause_validated": False,
        "scenarios_attempted": len(records), "scenarios_completed": completed,
        "source_sha256": {path.name: twin.digest(path) for path in (input_path, preparation, flags, calibration, Path(__file__))},
        "shared_state_fingerprint": state_hash, "preparation_wall_s": prep["wall_s"],
        "total_wall_s": time.perf_counter() - started, "scenarios": records,
    })
    return all(record["status"] == "completed" for record in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=coupled.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    scenario_names = [name for name, _, _ in SCENARIOS]
    parser.add_argument("--only", nargs="+", choices=scenario_names, default=scenario_names)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    raise SystemExit(0 if run(args.input, args.output, args.only) else 1)
