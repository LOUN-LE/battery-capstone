"""Replay D's first ageing set and compare mechanisms from the same RPT0 state."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import shutil
import time

from compare_mechanism_scenarios import (
    DEAD_LITHIUM_RATE, PLATING_RATE, SCENARIOS, SEI_RATE, SEI_CURRENT,
)
import coupled_run as coupled
import numpy as np
import pybamm

from d_long_route import get_route

twin = coupled.twin
T_AMBIENT_K = 298.15
MESH = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 20, "r_p": 20}
RTOL, ATOL = 1e-7, 1e-9
REFERENCE_END_V = 2.5
CHANNELS = {
    "sei_li_consumed_Ah": "Loss of capacity to negative SEI [A.h]",
    "plated_li_mol_m3": coupled.EXTRA["plated_li_mol_m3"],
    "dead_li_mol_m3": twin.VARIABLES["dead_lithium_mol_m3"],
    "lam_ne_pct": coupled.EXTRA["lam_ne_pct"],
    "lam_pe_pct": coupled.EXTRA["lam_pe_pct"],
    "lli_pct": "Loss of lithium inventory [%]",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def array(solution, variable):
    return np.asarray(solution[variable].entries, dtype=float)


def persist_json(path, data):
    temporary = path.with_suffix(".tmp")
    twin.write_json(temporary, data)
    os.replace(temporary, path)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    # Only consumed from the output directory of this local run.
    with temporary.open("wb") as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def prepare_parameters(route):
    files = [coupled.BASE / "configs" / name for name in
             ("frozen_model.json", "cell_calibration.json")]
    config = twin.read_json(files[0])
    config["options"].update({
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "loss of active material": "stress-driven",
        "SEI on cracks": "false",
    })
    prep_rows, _ = twin.read_input(coupled.DEFAULT_PREP)
    voltage, anchor = twin.rest_anchor(prep_rows)
    if abs(voltage - route.initial_anchor["target_voltage_V"]) > 1e-7:
        raise ValueError("Selected-cell rest anchor differs from the native D archive")
    parameters, initial = twin.make_parameters(config, twin.read_json(files[1]), voltage)
    return parameters, config["options"], initial, anchor, [*files, coupled.DEFAULT_PREP]


def step_control(spec):
    operation = spec["control"]
    duration = float(spec["duration_s"])
    temp = spec["temperature_C"]
    termination = list(spec["termination"] or ())
    period = 4.0 if spec["measurement"] is not None else 50.0
    options = {
        "duration": duration, "termination": termination,
        "skip_ok": (spec["kind"] == "native_RPT_control" and spec["measurement"] is None
                    and bool(termination)),
        "period": min(period, duration),
        "temperature": T_AMBIENT_K if temp is None else 273.15 + float(temp),
    }
    factory = getattr(pybamm.step, operation)
    return factory(**options) if operation == "rest" else factory(spec["value"], **options)


def simulate(parameters, options, specs, state):
    simulation = pybamm.Simulation(
        pybamm.lithium_ion.SPMe(options=options),
        parameter_values=parameters.copy(),
        var_pts=MESH,
        experiment=pybamm.Experiment([tuple(map(step_control, specs))], temperature=T_AMBIENT_K),
        solver=pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL),
    )
    solution = simulation.solve(calc_esoh=False, starting_solution=state)
    pieces = solution.cycles[-1].steps
    if len(pieces) != len(specs):
        raise ValueError(f"Solved {len(pieces)} of {len(specs)} steps: {solution.termination}")
    return solution, pieces


def review_step(spec, solution, output):
    record = {key: spec[key] for key in ("index", "kind", "source_ref", "observed_duration_s",
                                        "temperature_C")}
    if "compression" in spec:
        record["compression"] = spec["compression"]
    if isinstance(solution, pybamm.EmptySolution):
        if spec["measurement"] is not None:
            raise ValueError("Capacity measurement was skipped")
        if spec["kind"] != "native_RPT_control" or not spec["termination"]:
            raise ValueError("Non-event history step was skipped")
        record.update(status="native_event_already_met_at_start")
        return record
    t = array(solution, "Time [s]")
    current = array(solution, "Current [A]")
    voltage = array(solution, "Voltage [V]")
    q = array(solution, "Discharge capacity [A.h]")
    if not all(np.isfinite(series).all() for series in (t, current, voltage, q)):
        raise ValueError(f"Nonfinite output at step {spec['index']}")
    duration = float(t[-1] - t[0])
    charge = float(q[-1] - q[0])
    record.update(status="solved", duration_s=duration, throughput_net_Ah=charge,
                  start_model_s=float(t[0]), end_model_s=float(t[-1]),
                  termination=str(solution.termination), voltage_min_V=float(voltage.min()),
                  voltage_max_V=float(voltage.max()))
    if spec["kind"] != "native_RPT_control":
        if solution.termination != "final time" or abs(duration - spec["duration_s"]) > 1e-5:
            raise ValueError(f"History step ended early at {spec['index']}: {record}")
        if isinstance(spec["value"], np.ndarray):
            source_t, source_current = spec["value"].T
            source_q = float(np.trapezoid(source_current, source_t) / 3600.0)
            applied = np.interp(t - t[0], source_t, source_current)
            record.update(input_charge_Ah=source_q, charge_error_Ah=charge - source_q,
                          current_error_max_A=float(np.max(np.abs(current - applied))))
    if spec["measurement"] is not None:
        title = str(spec["measurement"]["rpt"])
        rpt_number = int(title.lower().replace("rpt", "").strip())
        if "event:" not in record["termination"] or abs(float(voltage[-1]) - REFERENCE_END_V) > 1e-5:
            raise ValueError(f"RPT{rpt_number} did not finish at its independent 2.5 V endpoint")
        path = output / f"RPT{rpt_number}.npz"
        np.savez_compressed(path, time_s=t - t[0], current_A=current, voltage_V=voltage,
                            discharge_capacity_Ah=q - q[0], termination=record["termination"],
                            independent_cutoff=True)
        record.update(rpt_id=rpt_number, capacity_Ah=charge, curve_file=path.name,
                      end_channels={key: float(array(solution, variable)[-1])
                                    for key, variable in CHANNELS.items()})
    return record


def review_group(spec, members, solution):
    t = array(solution, "Time [s]")
    q = array(solution, "Discharge capacity [A.h]")
    voltage = array(solution, "Voltage [V]")
    current = array(solution, "Current [A]")
    if not all(np.isfinite(series).all() for series in (t, q, voltage, current)):
        raise ValueError("Grouped solve produced nonfinite samples")
    duration = float(t[-1] - t[0])
    if solution.termination != "final time" or abs(duration - spec["duration_s"]) > 1e-5:
        raise ValueError(f"Group at {members[0]['index']} ended early at {duration}: {solution.termination}")
    raw_t, raw_i = spec["value"].T
    integrated = float(np.trapezoid(raw_i, raw_t) / 3600)
    if abs(float(q[-1] - q[0]) - integrated) > twin.CHARGE_TOL_AH:
        raise ValueError(f"Total group charge differs from its input at {members[0]['index']}")
    table_i = np.interp(t - t[0], raw_t, raw_i)
    # Absolute time subtraction loses precision on microsecond transitions.
    # Every phase's integrated charge is checked separately below.
    offsets = np.asarray(spec["grouped_offsets_s"]) + t[0]
    observed_q = np.interp(offsets, t, q)
    observed_v = np.interp(offsets, t, voltage)
    records = []
    for i, phase in enumerate(members):
        member = {key: phase[key] for key in ("index", "kind", "source_ref",
                                               "observed_duration_s", "temperature_C")}
        a, b = offsets[i:i+2]
        j, k = np.searchsorted(t, [a, b])
        v = np.r_[observed_v[i:i+2], voltage[j:k]]
        source = phase["value"]
        source_q = float(np.trapezoid(source[:, 1], source[:, 0]) / 3600) if isinstance(source, np.ndarray) else 0.0
        charge = float(observed_q[i+1] - observed_q[i])
        if abs(charge - source_q) > twin.CHARGE_TOL_AH:
            raise ValueError(f"Grouped member charge mismatch at {phase['index']}: {charge-source_q}")
        member.update(status="solved", duration_s=float(b-a), start_model_s=float(a),
                      end_model_s=float(b), termination="final time", grouped_with=len(members),
                      voltage_min_V=float(v.min()), voltage_max_V=float(v.max()),
                      throughput_net_Ah=charge, input_charge_Ah=source_q, charge_error_Ah=charge-source_q)
        if "compression" in phase:
            member["compression"] = phase["compression"]
        records.append(member)
    records[0]["group_charge_residual_Ah"] = float(q[-1] - q[0]) - integrated
    records[0]["group_current_grid_max_error_A"] = float(np.max(np.abs(current-table_i)))
    records[0]["group_voltage_span_V"] = [float(voltage.min()), float(voltage.max())]
    return records


def run(args, start, started_utc):
    route = get_route(args.root, args.gap_temperature)
    p, options, initial, anchor, parameter_files = prepare_parameters(route)
    report_path = args.output / "progress.json"
    checkpoint_path = args.output / "checkpoint.pkl"
    file_paths = [*parameter_files, Path(__file__), Path(coupled.__file__), Path(twin.__file__)]
    sources = {str(path.resolve()): twin.digest(path) for path in file_paths}
    common = {"scenario": args.scenario, "gap_temperature_C": args.gap_temperature,
                "route_sources": route.fingerprints,
                "mesh": MESH, "rtol": RTOL, "atol": ATOL, "pybamm": pybamm.__version__}
    identity = {**common, "model_sources": sources, "compressed_verified_cc": args.compress_cc}
    if args.compress_cc:
        from d_long_compression import compressed_steps
        import d_long_compression as compression
        identity["compression_source"] = twin.digest(Path(compression.__file__))
        identity["compression_sources"] = compression.compression_fingerprints()
    if args.group_cc:
        from d_long_grouped import group_current_specs
        import d_long_grouped as grouped
        identity["grouping_source"] = twin.digest(Path(grouped.__file__))
    if args.seed_checkpoint:
        prior = load_checkpoint(args.seed_checkpoint)
        if not all(prior["identity"].get(key) == value for key, value in common.items()):
            raise ValueError("The source checkpoint has different physical settings or input provenance")
        archived = args.seed_checkpoint.parent / "run_d_long_scenarios_raw.py"
        original = {**sources, str(Path(__file__).resolve()): twin.digest(archived)}
        if prior["identity"]["model_sources"] != original:
            raise ValueError("The original raw runner and model sources differ from the checkpoint")
        shutil.copyfile(args.seed_checkpoint.parent / "RPT0.npz", args.output / "RPT0.npz")
        identity["seed_sha256"] = twin.digest(args.seed_checkpoint)
        identity["seed_step_count"] = len(prior["journal"])
    journal, state, previous_time = [], None, 0.0
    if args.resume or args.seed_checkpoint:
        if args.resume:
            prior = load_checkpoint(checkpoint_path)
            for key in ("seed_sha256", "seed_step_count"):
                if key in prior["identity"]:
                    identity[key] = prior["identity"][key]
            if prior["identity"] != identity:
                raise ValueError("Resume requires the same input, model, solver, and scenario")
        journal, state = prior["journal"], prior["state"]
        if any(record["index"] != i or record["source_ref"] != route.plan[i]["source_ref"]
               for i, record in enumerate(journal)):
            raise ValueError("Saved journal is not a contiguous prefix of this route")
        previous_time = prior["elapsed_s"]
        start_utc = prior["started_utc"]
    else:
        start_utc = started_utc
    scenario_parameters = p.copy()
    _, sei, plating = next(row for row in SCENARIOS if row[0] == args.scenario)
    if not sei:
        scenario_parameters[SEI_RATE] = 0.0
    if not plating:
        scenario_parameters[PLATING_RATE] = 0.0
        scenario_parameters[DEAD_LITHIUM_RATE] = 0.0
    target_steps = len(route.plan)
    rpt0_end = next(i + 1 for i, phase in enumerate(route.plan) if phase.get("measurement") is not None)
    if args.initial_only:
        target_steps = rpt0_end
    if args.stop_after_step is not None:
        target_steps = min(target_steps, args.stop_after_step)
    report = {
        "started_utc": start_utc, "status": "running", "identity": identity,
        "cell_id": "D", "scope": "conditional RPT0 through RPT1 replay",
        "route_summary": route.summary, "initial_state": initial, "initial_anchor": anchor,
        "planned_steps": target_steps, "full_route_steps": len(route.plan), "attempt_count": 1,
        "experimental_conditions": {
            "recorded_phase_model_temperature_C": 25,
            "gap_assumption": "0 A rest at declared gap temperature; not observed",
            "mechanism_switch_step": rpt0_end,
            "ageing_rates_paused_after_RPT0": {"sei_growth": not sei, "plating_reactions": not plating},
        },
    }
    consumed = len(journal)
    spec_iterator = iter(compressed_steps(route, start_index=consumed)) if args.compress_cc else iter(route.steps())
    # Consume validated descriptors to the resume checkpoint without executing them.
    from itertools import islice
    if consumed and not args.compress_cc:
        for _ in islice(spec_iterator, consumed):
            pass
    try:
        while consumed < target_steps:
            limit = rpt0_end if consumed < rpt0_end else target_steps
            batch = list(islice(spec_iterator, min(args.batch_size, limit - consumed)))
            if not batch:
                raise ValueError("The validated route ended early")
            start_batch = time.perf_counter()
            params = p if consumed < rpt0_end else scenario_parameters
            grouped_batch = group_current_specs(batch) if args.group_cc else [(spec, None) for spec in batch]
            solution, pieces = simulate(params, options, [spec for spec, _ in grouped_batch], state)
            entries = []
            for (spec, members), piece in zip(grouped_batch, pieces, strict=True):
                if members is None:
                    entries.append(review_step(spec, piece, args.output))
                else:
                    entries.extend(review_group(spec, members, piece))
            state = solution.last_state.copy()
            journal.extend(entries)
            consumed += len(batch)
            elapsed = previous_time + time.perf_counter() - start
            checkpoint = {"identity": identity, "journal": journal, "state": state,
                          "elapsed_s": elapsed, "started_utc": start_utc}
            save_checkpoint(checkpoint_path, checkpoint)
            if consumed == rpt0_end:
                save_checkpoint(args.output / "RPT0_checkpoint.pkl", checkpoint)
            report.update(completed_steps=consumed, elapsed_s=elapsed,
                          measurements=[entry for entry in journal if "rpt_id" in entry],
                          last_batch_s=time.perf_counter() - start_batch,
                          last_step=entries[-1], step_status_counts=dict(Counter(e["status"] for e in journal)))
            persist_json(report_path, report)
            print(f"{args.scenario} {consumed}/{target_steps} steps, {elapsed:.2f}s", flush=True)
        report["status"] = "replay_completed" if consumed == len(route.plan) else "partial_checkpoint_saved"
        persist_json(args.output / "steps.json", journal)
        if consumed == len(route.plan):
            from d_long_scoring import score_rpts
            report["scores"] = score_rpts(
                {n: args.output / f"RPT{n}.npz" for n in (0, 1)},
                args.output,
                raw_reference_dir=args.root / "work/external_m50_benchmark",
                criteria_path=coupled.WORKSPACE / "configs/empirical_acceptance.json",
            )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      completed_steps=consumed, elapsed_s=previous_time + time.perf_counter() - start)
        persist_json(report_path, report)
        raise
    report.update(ended_utc=stamp(), elapsed_s=previous_time + time.perf_counter() - start)
    persist_json(report_path, report)
    return report


def main():
    start, started_utc = time.perf_counter(), stamp()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Restored capstone root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-temperature", type=int, default=25, choices=(10, 25, 40))
    parser.add_argument("--scenario", choices=[row[0] for row in SCENARIOS], default="both_active")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compress-cc", action="store_true")
    parser.add_argument("--group-cc", action="store_true")
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int, help="Stop after this many verified steps")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.seed_checkpoint and (args.resume or not args.compress_cc):
        parser.error("Seeding requires a new output directory and explicit CC compression")
    if args.group_cc and not args.compress_cc:
        parser.error("Phase grouping uses verified compressed CC waveforms")
    if args.resume:
        if not (args.output / "checkpoint.pkl").is_file():
            parser.error("No checkpoint for resume")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    run(args, start, started_utc)


if __name__ == "__main__":
    main()
