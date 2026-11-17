"""Bounded, fresh-state SPMe/DFN comparison against C native RPT0 observations.

This isolated diagnostic changes model family only. It never resumes an old
state, fits parameters, grants lifetime admission, or starts cloud resources.
Primary voltage/current errors use the ORIGINAL observed elapsed clock. Model
phase timings are not shifted or warped to force agreement with observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"

import numpy as np

from run_exposure_scenario_smoke import read, write, sha


def select_initial_route(plan, include_recovery=True):
    """Keep the complete chronological prefix through RPT0, optionally its rest."""
    index = next((i for i, item in enumerate(plan)
                  if item.get("measurement") is not None
                  and item["measurement"]["rpt"] == 0), None)
    if index is None:
        raise ValueError("RPT0 measurement absent")
    end = index + 1
    if include_recovery:
        # Include only explicit transition rests and the first native rest.
        # Never accidentally include the next ageing or recharge current.
        for j in range(end, len(plan)):
            item = plan[j]
            if item["kind"] == "assumed_exposure":
                source = item["source_step"]
                if source.get("assumed_operation") != "rest":
                    break
                continue
            if (item["kind"] == "native_RPT_control"
                    and item["spec"]["operation"] == "rest"):
                end = j + 1
            break
    result = plan[:end]
    if sum(item.get("measurement") is not None for item in result) != 1:
        raise ValueError("Only the first RPT0 capacity measurement may enter")
    return result


def error_metrics(prediction, observed):
    prediction, observed = np.asarray(prediction), np.asarray(observed)
    if prediction.shape != observed.shape or not prediction.size:
        raise ValueError("Nonempty paired samples required")
    if not np.isfinite(prediction).all() or not np.isfinite(observed).all():
        raise ValueError("Nonfinite score samples")
    error = prediction - observed
    return dict(n=int(error.size), bias=float(error.mean()),
                rmse=float(np.sqrt(np.mean(error ** 2))),
                max_abs=float(np.max(np.abs(error))))


def strictly_increasing_trace(t, *values):
    """At a controller boundary choose its right-hand sample, without time shift."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all():
        raise ValueError("Finite one-dimensional model clock required")
    if np.any(np.diff(t) < 0):
        raise ValueError("Model clock moved backwards")
    keep = np.r_[np.diff(t) > 0, True]
    if np.count_nonzero(keep) < 2:
        raise ValueError("At least two distinct clock samples required")
    arrays = [np.asarray(value, dtype=float) for value in values]
    if any(value.shape != t.shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Trace shape/finite check failed")
    return (t[keep], *(value[keep] for value in arrays))


def score_observed_window(model_t, model_v, model_i, observed):
    """Interpolate only inside model support, retaining uncovered raw counts."""
    t = observed["t_s"]
    valid = observed["valid"] & np.isfinite(t)
    covered = valid & (t >= model_t[0]) & (t <= model_t[-1])
    result = dict(valid_raw_rows=int(valid.sum()), scored_raw_rows=int(covered.sum()),
                  uncovered_raw_rows=int((valid & ~covered).sum()),
                  alignment="original observed absolute elapsed seconds; no phase shift or time warp")
    if covered.any():
        result["voltage_V"] = error_metrics(np.interp(t[covered], model_t, model_v), observed["V"][covered])
        result["current_A"] = error_metrics(np.interp(t[covered], model_t, model_i), observed["I_A"][covered])
    return result


def observed_window(item, rawsteps, arrays, origin):
    if item["kind"] != "native_RPT_control":
        return None
    source = item["native_source"]
    arrays.get(source)  # verify SHA256 through the existing source registry
    path = (arrays.directory / source["arrays_file"]).resolve()
    rawstep = rawsteps[item["source_step"]["raw_step_index"]]
    lo, hi = rawstep["row_start"], rawstep["row_end_exclusive"]
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["t_s"][lo:hi], dtype=float)
        voltage = np.asarray(z["V_V"][lo:hi], dtype=float)
        current = -np.asarray(z["I_mA"][lo:hi], dtype=float) / 1000
        valid = np.asarray(z["analysis_valid_row"][lo:hi], dtype=bool)
    offset = (datetime.fromisoformat(source["timestamp_naive"]) - origin).total_seconds()
    t = t + offset
    valid &= np.isfinite(t) & np.isfinite(voltage) & np.isfinite(current)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Raw phase clock must be strictly increasing")
    return dict(t_s=t, V=voltage, I_A=current, valid=valid,
                source_array_path=str(path), source_array_sha256=source["arrays_sha256"],
                row_start=lo, row_end_exclusive=hi)


def memory_mb():
    try:
        import psutil
        return dict(current_rss_MB=psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        if sys.platform == "win32":
            # Existing implementation already handles Windows process counters.
            from battery import memory_mb as original_memory
            return original_memory()
        import resource
        scale = 2**20 if sys.platform == "darwin" else 1024
        return dict(peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale)


def prepare(args):
    from interstitial_pilot_support import validate_contract, prepare_parameters
    from run_corrected_history_trial import compile_route
    contract = validate_contract(args.contract, args.dint_multiplier)
    scenario, raw, plan, zero, arrays, programs = compile_route(args.root, args.scenario, 1 if args.include_recovery else 0)
    if scenario["cell_id"] != "C":
        raise ValueError("This diagnostic supports only the declared C source")
    plan = select_initial_route(plan, args.include_recovery)
    parameters, options, metadata, identity = prepare_parameters(args.candidate, args.anchor, contract, args.dint_multiplier)
    mesh = dict(x_n=args.x_points, x_s=args.x_points, x_p=args.x_points,
                r_n=args.r_points, r_n_prim=args.r_points, r_n_sec=args.r_points, r_p=args.r_points)
    paths = [Path(__file__), args.contract, args.scenario, args.candidate, args.anchor, raw, programs]
    paths.extend(Path(__file__).with_name(name) for name in (
        "diagnostic_model_family.py", "interstitial_pilot_support.py", "run_corrected_history_trial.py",
        "run_exposure_scenario_smoke.py", "native_program_contracts.py", "conditional_history_inputs.py"))
    guards = {str(path.resolve()): sha(path) for path in paths}
    guards.update(identity["source_sha256"])
    guards.update(arrays.guards)
    return contract, raw, plan, arrays, parameters, options, metadata, identity, mesh, guards


def make_controls(plan, period_s):
    import pybamm
    controls = []
    for item in plan:
        if item["kind"] == "native_RPT_control":
            spec = item["spec"]
            kw = dict(duration=spec["duration_s"], termination=spec["termination"],
                      skip_ok=item["measurement"] is None and bool(spec["termination"]),
                      period=min(period_s, spec["duration_s"]))
            make = getattr(pybamm.step, spec["operation"])
            control = make(**kw) if spec["operation"] == "rest" else make(spec["value"], **kw)
        elif item["kind"] == "assumed_exposure":
            source = item["source_step"]
            if (source.get("assumed_operation") != "rest" or source.get("assumed_current_A") != 0
                    or source.get("assumed_temperature_C") != 25):
                raise ValueError("Only explicit existing 0A/25C transition assumptions supported")
            duration = source["duration_us"] / 1e6
            control = pybamm.step.rest(duration, period=min(period_s, duration), skip_ok=False)
        else:
            raise ValueError("Unexpected nonnative source before initial RPT0 recovery")
        controls.append(control)
    return controls


def check_local_bounds(extrema, parameters):
    """Check resolved local fields rather than an electrode average."""
    failures = []
    if extrema["electrolyte"]["min"] <= 0:
        failures.append("Electrolyte concentration is nonpositive")
    maxima = {
        "negative_primary": float(parameters["Primary: Maximum concentration in negative electrode [mol.m-3]"]),
        "negative_secondary": float(parameters["Secondary: Maximum concentration in negative electrode [mol.m-3]"]),
        "positive": float(parameters["Maximum concentration in positive electrode [mol.m-3]"]),
    }
    for key, maximum in maxima.items():
        epsilon = 1e-9 * maximum
        if extrema[key]["min"] < -epsilon or extrema[key]["max"] > maximum+epsilon:
            failures.append(key+" concentration outside0..cmax")
    for key in ("negative_porosity", "positive_porosity"):
        if extrema[key]["min"] < -1e-12 or extrema[key]["max"] > 1+1e-12:
            failures.append(key+" outside0..1")
    return failures


def worker(args):
    import pybamm
    from diagnostic_model_family import make_model
    from interstitial_pilot_support import native_control_evidence, validate_charge_residuals
    began = time.monotonic()
    out = args.output / args.worker_family
    out.mkdir()
    report = dict(status="started", model_family=args.worker_family, fresh_initial_state=True,
                  source_checkpoint_loaded=False, parameter_fitting_performed=False,
                  physical_validation_complete=False, lifetime_restart_admitted=False,
                  charge_optimization_admitted=False, AWS_started=False)
    guards, phases = {}, []
    try:
        c, raw, plan, arrays, p, options, meta, identity, mesh, guards = prepare(args)
        write(out / "runtime_identity.json", identity)
        write(out / "route.json", plan)
        report.update(options=options, mesh=mesh, rtol=args.rtol, atol=args.atol, period_s=args.period_s,
                      Dint_multiplier=args.dint_multiplier, candidate_metadata=meta,
                      runtime_versions=identity["runtime_versions"], runtime_identity_sha256=identity["sha256"],
                      planned_steps=len(plan), completed_steps=0, source_sha256=guards,
                      scope="C initial RPT0 and optional first recovery; no ageing-curve/lifetime validation",
                      clock_scope="Original naive instrument clock; no UTC/DST inference; model clock not corrected",
                      inherited_transition_assumptions=[item for item in plan if item["kind"] == "assumed_exposure"])
        write(out / "report.json", report)
        controls = make_controls(plan, args.period_s)
        sim = pybamm.Simulation(make_model(args.worker_family, options), parameter_values=p.copy(), var_pts=mesh,
                                experiment=pybamm.Experiment([tuple(controls)], temperature=298.15),
                                solver=pybamm.IDAKLUSolver(rtol=args.rtol, atol=args.atol))
        before_build = time.monotonic()
        sim.build_for_experiment()
        built = time.monotonic()
        solution = sim.solve(calc_esoh=False)
        solved = time.monotonic()
        pieces = solution.cycles[-1].steps
        report.update(build_s=built-before_build, solve_s=solved-built, memory_after_solve=memory_mb())
        rawsteps = read(raw)["cell"]["steps"]
        origin = datetime.fromisoformat(plan[0]["source_step"]["absolute_start_naive"])
        trace_parts, observed_parts = [], []
        baseline_li = None
        previous_end = None
        maximum_time_jump, maximum_charge_jump = 0., 0.
        for idx, (item, piece) in enumerate(zip(plan, pieces)):
            observed = observed_window(item, rawsteps, arrays, origin)
            phase = dict(index=idx, source_ref=item["source_ref"], kind=item["kind"])
            if observed is not None:
                np.savez_compressed(out / f"observed_{idx:03d}.npz", **{key:observed[key] for key in ("t_s", "V", "I_A", "valid")})
                phase.update(observed_start_s=float(observed["t_s"][0]), observed_end_s=float(observed["t_s"][-1]),
                             observed_duration_s=float(observed["t_s"][-1]-observed["t_s"][0]),
                             source_array_path=observed["source_array_path"], source_array_sha256=observed["source_array_sha256"],
                             raw_row_window=[observed["row_start"], observed["row_end_exclusive"]], spec=item["spec"])
                observed_parts.append((phase, observed))
            phases.append(phase)
            if isinstance(piece, pybamm.EmptySolution):
                phase.update(status="initial_event_skip", actual_exposure_mismatch=True)
                if item.get("measurement") is not None:
                    raise ValueError("RPT capacity measurement was skipped")
                continue
            names = ("Time [s]", "Current [A]", "Voltage [V]", "Discharge capacity [A.h]",
                     "Throughput capacity [A.h]", "Total lithium in particles [mol]", "Loss of capacity to negative SEI [A.h]")
            t, current, voltage, charge, throughput, lithium, sei = [np.asarray(piece[name].entries, dtype=float).reshape(-1) for name in names]
            if not all(np.isfinite(value).all() for value in (t,current,voltage,charge,throughput,lithium,sei)):
                raise ValueError("Nonfinite primary output")
            conserved = lithium * 96485.33212331001 / 3600 + sei
            if baseline_li is None:
                baseline_li = float(conserved[0])
            residual = float(np.max(np.abs(conserved-baseline_li)))
            phase.update(status="simulated", model_start_s=float(t[0]), model_end_s=float(t[-1]),
                         model_duration_s=float(t[-1]-t[0]), termination=str(piece.termination),
                         signed_Ah=float(charge[-1]-charge[0]), absolute_Ah=float(throughput[-1]-throughput[0]),
                         voltage_min_V=float(voltage.min()), voltage_max_V=float(voltage.max()),
                         Li_plus_SEI_conservation_error_Ah=residual)
            if previous_end is not None:
                time_jump = abs(float(t[0])-previous_end[0])
                charge_jump = abs(float(charge[0])-previous_end[1])
                maximum_time_jump = max(maximum_time_jump, time_jump)
                maximum_charge_jump = max(maximum_charge_jump, charge_jump)
                phase.update(boundary_time_jump_s=time_jump, boundary_charge_jump_Ah=charge_jump)
            previous_end = (float(t[-1]), float(charge[-1]))
            report.update(maximum_boundary_time_jump_s=maximum_time_jump,
                          maximum_boundary_charge_jump_Ah=maximum_charge_jump)
            if observed is not None:
                phase.update(start_drift_s=float(t[0]-observed["t_s"][0]),
                             end_drift_s=float(t[-1]-observed["t_s"][-1]),
                             duration_error_s=phase["model_duration_s"]-phase["observed_duration_s"])
            artifact = dict(t_s=t, I_A=current, V=voltage, Q_Ah=charge, throughput_Ah=throughput, Li_mol=lithium, SEI_Ah=sei,
                            terminal_y=np.asarray(piece.last_state.all_ys[-1], dtype=float).reshape(-1))
            local = {}
            for key, name in (("electrolyte", "Electrolyte concentration [mol.m-3]"),
                              ("negative_primary", "Negative primary particle concentration [mol.m-3]"),
                              ("negative_secondary", "Negative secondary particle concentration [mol.m-3]"),
                              ("positive", "Positive particle concentration [mol.m-3]"),
                              ("negative_porosity", "Negative electrode porosity"),
                              ("positive_porosity", "Positive electrode porosity")):
                field = np.asarray(piece[name].entries, dtype=float)
                if not np.isfinite(field).all():
                    raise ValueError("Nonfinite local field: " + name)
                local[key] = dict(min=float(field.min()), max=float(field.max()))
                spatial = field.reshape(-1, len(t))
                artifact[key+"_local_min"] = spatial.min(axis=0)
                artifact[key+"_local_max"] = spatial.max(axis=0)
            phase["local_field_extrema"] = local
            phase["local_field_bounds_failures"] = check_local_bounds(local, p)
            np.savez_compressed(out / f"model_{idx:03d}.npz", **artifact)
            trace_parts.append((t, voltage, current))
            if maximum_time_jump > 1e-5 or maximum_charge_jump > 1e-8:
                raise ValueError("Main clock or accumulated charge discontinuity")
            if phase["local_field_bounds_failures"]:
                raise ValueError("; ".join(phase["local_field_bounds_failures"]))
            if item["kind"] == "native_RPT_control":
                spec = item["spec"]
                native_control_evidence(spec, current, voltage, phase["model_duration_s"], str(piece.termination), c, evidence=phase)
                if spec["operation"] in ("current", "rest"):
                    expected = spec["value"] if spec["operation"] == "current" else 0.
                    applied_error = float(np.max(np.abs(current-expected)))
                    phase["applied_current_max_error_A"] = applied_error
                    if applied_error > c["solver_diagnostic_guards"]["applied_current_max_error_A"]:
                        raise ValueError("Native CC/rest applied current mismatch")
                    signed_residual = phase["signed_Ah"]-expected*phase["model_duration_s"]/3600
                    absolute_residual = phase["absolute_Ah"]-abs(expected)*phase["model_duration_s"]/3600
                    phase.update(signed_charge_residual_Ah=signed_residual, absolute_charge_residual_Ah=absolute_residual)
                    validate_charge_residuals(signed_residual, absolute_residual, c)
                if item["measurement"] is not None:
                    target = item["measurement"]["performance_summary_C10_capacity_Ah"]
                    report["RPT0"] = dict(model_capacity_Ah=phase["signed_Ah"], observed_capacity_Ah=target,
                                           capacity_bias_Ah=phase["signed_Ah"]-target,
                                           capacity_relative_error_percent=100*(phase["signed_Ah"]-target)/target)
            if residual > c["solver_diagnostic_guards"]["Li_plus_SEI_conservation_max_error_Ah"]:
                raise ValueError("Li+SEI conservation guard failed")
            report["completed_steps"] = idx+1
            write(out / "phase_journal.json", phases)
        if len(pieces) != len(plan):
            raise ValueError("Incomplete experiment phase count")
        model_t, model_v, model_i = strictly_increasing_trace(*(np.concatenate([part[k] for part in trace_parts]) for k in range(3)))
        np.savez_compressed(out / "model_trace.npz", t_s=model_t, V=model_v, I_A=model_i)
        scored_v, scored_i, obs_v, obs_i = [], [], [], []
        for phase, observed in observed_parts:
            phase["absolute_time_score"] = score_observed_window(model_t, model_v, model_i, observed)
            mask = observed["valid"] & (observed["t_s"] >= model_t[0]) & (observed["t_s"] <= model_t[-1])
            scored_v.extend(np.interp(observed["t_s"][mask], model_t, model_v))
            scored_i.extend(np.interp(observed["t_s"][mask], model_t, model_i))
            obs_v.extend(observed["V"][mask]); obs_i.extend(observed["I_A"][mask])
        report.update(status="diagnostic_complete", initial_total_Li_plus_SEI_Ah=baseline_li,
                      voltage_absolute_time_score_V=error_metrics(scored_v, obs_v),
                      current_absolute_time_score_A=error_metrics(scored_i, obs_i),
                      uncovered_raw_rows=sum(phase["absolute_time_score"]["uncovered_raw_rows"] for phase, _ in observed_parts),
                      scoring_scope="sample-weighted raw rows on original absolute elapsed clock; output interpolation only; no time warp",
                      all_phases_simulated=all(phase["status"] == "simulated" for phase in phases))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    report.update(wall_s=time.monotonic()-began, memory_final=memory_mb(), source_sha256=guards,
                  source_hashes_unchanged=bool(guards) and all(sha(path)==value for path,value in guards.items()))
    if guards and not report["source_hashes_unchanged"]:
        report["status"] = "failed_source_integrity"
    write(out / "phase_journal.json", phases)
    write(out / "report.json", report)
    return 0 if report["status"] == "diagnostic_complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "candidate", "anchor", "contract", "scenario", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=("SPMe", "DFN"), default=["SPMe", "DFN"])
    parser.add_argument("--x-points", type=int, default=10)
    parser.add_argument("--r-points", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--period-s", type=float, default=10.)
    parser.add_argument("--dint-multiplier", type=float, default=1000.)
    parser.add_argument("--max-wall-s", type=float, default=600., help="Total watchdog budget across all requested families")
    parser.add_argument("--no-recovery", dest="include_recovery", action="store_false")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-family", choices=("SPMe", "DFN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if not 30 <= args.max_wall_s <= 1200:
        parser.error("Total max-wall-s must be 30..1200")
    if len(set(args.families)) != len(args.families):
        parser.error("Families must be unique")
    if not 3 <= args.x_points <= 100 or not 3 <= args.r_points <= 100:
        parser.error("Mesh points must be 3..100")
    if not all(math.isfinite(value) and value > 0 for value in (args.rtol,args.atol,args.period_s)):
        parser.error("Positive finite solver settings required")
    if args.period_s > 10:
        parser.error("Output period may not exceed10s")
    args.output.relative_to(args.root)
    if args.worker_family:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    began, results = time.monotonic(), []
    if args.prepare_only:
        _, _, plan, _, _, options, meta, identity, mesh, guards = prepare(args)
        write(args.output / "preparation.json", dict(status="compiled_no_PDE", steps=len(plan), mesh=mesh,
              options=options, candidate_metadata=meta, runtime_identity_sha256=identity["sha256"],
              source_sha256=guards, families=args.families, physical_validation_complete=False))
        return 0
    for family in args.families:
        remaining = args.max_wall_s-(time.monotonic()-began)
        if remaining <= 0:
            results.append(dict(family=family, status="not_started_total_budget")); continue
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-family", family]
        with (args.output / (family+"_stdout.log")).open("w",encoding="utf-8") as stdout, (args.output / (family+"_stderr.log")).open("w",encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=remaining)
                status = "worker_completed" if code == 0 else "worker_failed"
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                code, status = process.returncode, "terminated_total_wall_budget"
            report_path = args.output / family / "report.json"
            results.append(dict(family=family,status=status,returncode=code,pid=process.pid,
                                report=read(report_path) if report_path.exists() else None))
    complete = all(result["status"] == "worker_completed" for result in results)
    write(args.output / "comparison.json", dict(status="diagnostic_complete" if complete else "incomplete_or_failed",
          results=results,wall_s=time.monotonic()-began,max_wall_s=args.max_wall_s,
          physical_validation_complete=False,lifetime_restart_admitted=False,BO_admitted=False,AWS_started=False))
    print(json.dumps(dict(status="diagnostic_complete" if complete else "incomplete_or_failed", output=str(args.output)),ensure_ascii=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
