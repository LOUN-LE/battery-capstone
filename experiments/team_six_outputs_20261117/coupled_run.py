"""Isolated coupled-damage replay of the measured C/10 window for cell D."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "bms_pipeline"
WORKSPACE = HERE.parents[1]
sys.path.insert(0, str(BASE))
import twin_runner as twin

import numpy as np
import pybamm

EXTRA = {
    "sei_thickness_m": "X-averaged negative SEI thickness [m]",
    "plated_li_mol_m3": "X-averaged negative lithium plating concentration [mol.m-3]",
    "negative_crack_length_m": "X-averaged negative particle crack length [m]",
    "lam_ne_pct": "Loss of active material in negative electrode [%]",
    "lam_pe_pct": "Loss of active material in positive electrode [%]",
    "negative_porosity": "X-averaged negative electrode porosity",
}
EXPORT = {
    "model_voltage_V": ("voltage_V", 1., 0.),
    "model_current_A": ("current_A", 1., 0.),
    "model_temperature_C": ("model_temperature_K", 1., -273.15),
    "sei_thickness_nm": ("sei_thickness_m", 1e9, 0.),
    "sei_loss_ah": ("SEI_lithium_loss_Ah", 1., 0.),
    "plated_li_mol_m3": ("plated_li_mol_m3", 1., 0.),
    "dead_li_mol_m3": ("dead_lithium_mol_m3", 1., 0.),
    "negative_crack_length_m": ("negative_crack_length_m", 1., 0.),
    "lam_ne_pct": ("lam_ne_pct", 1., 0.),
    "lam_pe_pct": ("lam_pe_pct", 1., 0.),
    "negative_porosity": ("negative_porosity", 1., 0.),
}
DEFAULT_INPUT = (BASE / "evidence"
                 / "02_머신러닝_후_디지털트윈_입력.csv")
DEFAULT_PREP = BASE / "data/cell_D_preparation.csv"
MAIN_MESH = dict(twin.MESH)
REFINED_MESH = {key: value * 2 for key, value in MAIN_MESH.items()}


def solve_at_mesh(parameters, config, preparation_rows, table, mesh, check_split=False):
    with patch.object(twin, "MESH", mesh), patch.dict(twin.VARIABLES, EXTRA):
        starting, prep_report = twin.prepare_state(parameters, config, preparation_rows)
        terminal, trace, replay_report = twin.observed_step(parameters, config, starting, table)
        audit = {"mesh": mesh, "preparation": prep_report, "observed_replay": replay_report}
        if check_split:
            split_at = float(table[len(table) // 2, 0])
            first, left_trace, left = twin.observed_step(
                parameters, config, starting, twin.slice_table(table, 0., split_at)
            )
            second, right_trace, right = twin.observed_step(
                parameters, config, first, twin.slice_table(table, split_at, table[-1, 0])
            )
            state = twin.compare_states(terminal, second)
            time_s = np.r_[left_trace["time_s"], right_trace["time_s"][1:] + split_at]
            voltage = np.r_[left_trace["voltage_V"], right_trace["voltage_V"][1:]]
            voltage_diff = float(np.max(abs(
                np.interp(time_s, trace["time_s"], trace["voltage_V"]) - voltage
            )))
            # Near the split, output grids differ. Compare on the common observed times.
            comparison_grid = table[:, 0]
            aligned_voltage_diff = float(np.max(abs(
                np.interp(comparison_grid, trace["time_s"], trace["voltage_V"])
                - np.interp(comparison_grid, time_s, voltage)
            )))
            fields = {}
            for column, (name, scale, offset) in EXPORT.items():
                divided = np.r_[left_trace[name], right_trace[name][1:]]
                whole = np.interp(comparison_grid, trace["time_s"], trace[name])
                segmented = np.interp(comparison_grid, time_s, divided)
                fields[column] = float(np.max(abs(whole - segmented)) * scale)
            audit["split_comparison"] = {
                "split_time_s": split_at, "left": left, "right": right,
                "terminal_state": state,
                "maximum_voltage_difference_V_on_split_grid": voltage_diff,
                "maximum_voltage_difference_V_on_source_grid": aligned_voltage_diff,
                "maximum_trajectory_differences": fields,
                "state_tolerance": twin.STATE_RELATIVE_TOL,
                "voltage_tolerance_V": twin.VOLTAGE_TOL_V,
                "passed": (
                    state["max_scaled_state_difference"] <= twin.STATE_RELATIVE_TOL
                    and aligned_voltage_diff <= twin.VOLTAGE_TOL_V
                ),
            }
            if not audit["split_comparison"]["passed"]:
                raise RuntimeError(f"Numerical split check failed: {audit['split_comparison']}")
    return trace, audit


def align(trace, t):
    return {
        column: np.interp(t, trace["time_s"], trace[name]) * scale + offset
        for column, (name, scale, offset) in EXPORT.items()
    }


def run(input_csv, preparation_csv, outdir):
    rows, observed = twin.read_input(input_csv)
    prep_rows, _ = twin.read_input(preparation_csv)
    if rows[0]["cell_id"] != prep_rows[0]["cell_id"]:
        raise ValueError("Preparation and observed replay must be for the same cell")
    if float(prep_rows[-1]["time_s"]) >= observed[0, 0]:
        raise ValueError("Preparation must precede evaluated measurements")
    t = observed[:, 0] - observed[0, 0]
    if len(rows) != 591 or not 0 < t[-1] <= 600.001:
        raise ValueError("Expected the saved, bounded 591-row selected-cell window")
    table = np.column_stack((t, observed[:, 2]))
    flags = BASE / "configs/frozen_model.json"
    cal = BASE / "configs/cell_calibration.json"
    config = deepcopy(twin.read_json(flags))
    extended_options = {
        "particle mechanics": ["swelling and cracking", "swelling only"],
        "loss of active material": "stress-driven",
        "SEI on cracks": "false",
    }
    config["options"].update(extended_options)
    config["options"]["particle mechanics"] = tuple(config["options"]["particle mechanics"])
    target, anchor = twin.rest_anchor(prep_rows)
    parameters, init = twin.make_parameters(config, twin.read_json(cal), target)
    identities = {
        path.name: twin.digest(path) for path in
        (input_csv, preparation_csv, flags, cal, Path(__file__), BASE / "twin_runner.py")
    }
    outdir.mkdir(parents=True, exist_ok=False)
    twin.write_json(outdir / "coupled_model.json", config)
    trace, audit = solve_at_mesh(parameters, config, prep_rows, table, MAIN_MESH, check_split=True)
    values = align(trace, t)
    if not np.allclose(values["model_current_A"], observed[:, 2], rtol=0, atol=1e-8):
        raise ValueError("Aligned model current disagrees with observed current")
    checks = {
        "all_finite": all(np.isfinite(column).all() for column in values.values()),
        "positive_crack_length": bool((values["negative_crack_length_m"] > 0).all()),
        "positive_sei_thickness": bool((values["sei_thickness_nm"] > 0).all()),
        "positive_porosity_below_one": bool(((0 < values["negative_porosity"]) & (values["negative_porosity"] < 1)).all()),
        "isothermal_temperature_fixed_at_25C": bool(np.allclose(values["model_temperature_C"], 25, rtol=0, atol=1e-9)),
        "source_rows_unique": len({r["source_row"] for r in rows}) == len(rows),
    }
    for column in ["sei_loss_ah", "plated_li_mol_m3", "dead_li_mol_m3", "lam_ne_pct", "lam_pe_pct"]:
        checks[f"{column}_nonnegative_with_roundoff_tolerance"] = bool((values[column] >= -1e-10).all())
    if not all(checks.values()):
        raise ValueError(f"Replay output checks failed: {checks}")
    output = outdir / "coupled_model_rows.csv"
    output_rows = [{
        "cell_id": source["cell_id"], "session_id": source["session_id"],
        "source_row": source["source_row"], "source_time_s": source["time_s"],
        **{value: float(series[index]) for value, series in values.items()},
    } for index, source in enumerate(rows)]
    twin.write_csv(output, output_rows, list(output_rows[0]))
    # Persist successful primary checks before independently checking mesh sensitivity.
    report = {
        "status": "primary_numerics_passed",
        "scope": "experimental coupled SPMe; the added degradation channels have no paired experimental validation",
        "versions": {"pybamm": pybamm.__version__},
        "cell_id": rows[0]["cell_id"], "session_id": rows[0]["session_id"],
        "input_rows": len(rows), "duration_s": float(t[-1]),
        "historical_source_roles": {"voltage": "evaluation only in this 591-row window",
                                    "current": "imposed control", "surface_temperature": "recorded only"},
        "initialization": {**init, "rest_anchor": anchor},
        "preparation": "same initial anchor and declared controls as the frozen baseline; complete re-solve with added mechanisms",
        "temperature": "fixed isothermal 25 degrees C; measured surface temperature is not imposed",
        "baseline_config_changes": extended_options,
        "parameter_provenance": "PyBaMM OKane2022 coefficients; previously frozen M50T material overrides and initial lithium inventory; no new fitting",
        "variables": {column: twin.VARIABLES.get(name, EXTRA.get(name)) for column, (name, _, _) in EXPORT.items()},
        "observed_voltage_rmse_mV": float(np.sqrt(np.mean((values["model_voltage_V"] - observed[:, 1]) ** 2)) * 1000),
        "observed_voltage_max_abs_error_mV": float(np.max(abs(values["model_voltage_V"] - observed[:, 1])) * 1000),
        "physical_accuracy_gate_passed": False,
        "degradation_outputs_validated_against_measurements": False,
        "lifetime_risk_validated": False,
        "checks": checks,
        "primary_mesh": audit,
        "ranges": {
            key: {"minimum": float(np.min(series)), "maximum": float(np.max(series)),
                  "start": float(series[0]), "end": float(series[-1]),
                  "end_minus_start": float(series[-1] - series[0])}
            for key, series in values.items()
        },
        "source_sha256": identities,
        "output_sha256": twin.digest(output),
    }
    twin.write_json(outdir / "report.json", report)
    finer_trace, finer_audit = solve_at_mesh(parameters, config, prep_rows, table, REFINED_MESH)
    finer_values = align(finer_trace, t)
    report["refined_mesh"] = finer_audit
    report["mesh_sensitivity_max_abs_by_column"] = {
        column: float(np.max(abs(values[column] - finer_values[column])))
        for column in values
    }
    report["mesh_sensitivity_scope"] = "numeric sensitivity only; no predeclared acceptance limit for the newly added degradation channels"
    if any(twin.digest(path) != identities[path.name] for path in
           (input_csv, preparation_csv, flags, cal, Path(__file__), BASE / "twin_runner.py")):
        raise RuntimeError("Input, code or configuration changed during replay")
    report["status"] = "completed"
    twin.write_json(outdir / "report.json", report)
    print(outdir / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.preparation, args.output)
