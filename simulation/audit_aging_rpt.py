"""Read-only audit of aging accumulation, lithium budgets and independent RPTs.

Run only with trusted checkpoints in the identical original dependency environment.
No production source, model option, tolerance, checkpoint or operational ledger is
modified. Each state is audited in a separate bounded child process. RPTs branch
from copies, with a fresh Cell/REF simulation for every repetition.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import argparse
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import battery as B
import run_100efc as R
import lifetime_policy as P

PACKAGES = ("pybamm", "pybammsolvers", "casadi", "numpy", "scipy", "pyarrow")
SCALAR_VARS = {
    "total_lithium_mol": "Total lithium [mol]",
    "particle_lithium_mol": "Total lithium in particles [mol]",
    "electrolyte_lithium_mol": "Total lithium in electrolyte [mol]",
    "total_lithium_lost_mol": "Total lithium lost [mol]",
    "particle_lithium_lost_mol": "Total lithium lost from particles [mol]",
    "electrolyte_lithium_lost_mol": "Total lithium lost from electrolyte [mol]",
    "side_reaction_lithium_lost_mol": "Total lithium lost to side reactions [mol]",
    "side_reaction_capacity_lost_Ah": "Total capacity lost to side reactions [A.h]",
    "LLI_particles_percent": "Loss of lithium inventory [%]",
    "LLI_with_electrolyte_percent": "Loss of lithium inventory, including electrolyte [%]",
    "negative_SEI_lithium_lost_mol": "Loss of lithium to negative SEI [mol]",
    "positive_SEI_lithium_lost_mol": "Loss of lithium to positive SEI [mol]",
    "negative_plating_lithium_lost_mol": "Loss of lithium to negative lithium plating [mol]",
    "positive_plating_lithium_lost_mol": "Loss of lithium to positive lithium plating [mol]",
    "negative_crack_SEI_lithium_lost_mol": "Loss of lithium to negative SEI on cracks [mol]",
    "positive_crack_SEI_lithium_lost_mol": "Loss of lithium to positive SEI on cracks [mol]",
    "negative_capacity_Ah": "Negative electrode capacity [A.h]",
    "positive_capacity_Ah": "Positive electrode capacity [A.h]",
    "negative_LAM_percent": "Loss of active material in negative electrode [%]",
    "positive_LAM_percent": "Loss of active material in positive electrode [%]",
    "current_A": "Current [A]",
    "voltage_V": "Voltage [V]",
    "time_s": "Time [s]",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def manifest(args):
    paths = [B.ROOT / p for p in P.LEGACY_SHA256]
    for directory in (Path(args.legacy_dir).resolve(), Path(args.final_dir).resolve()):
        paths.extend(directory / name for name in ("checkpoint.pkl", "summary.json", "diagnostics.json", "FAILED.json", "outcome.json") if (directory / name).is_file())
    return {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}


def validate_identity(identity, args):
    actual = {p: importlib.metadata.version(p) for p in PACKAGES}
    expected = {"cell": args.cell, "current_A": args.current, "mesh_scale": args.mesh_scale,
                "versions": actual, "python": platform.python_version(),
                "tolerance": {"rtol": B.DEFAULT_RTOL, "atol": B.DEFAULT_ATOL}}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError("Original environment/config identity mismatch: " + key)
    for rel, expected_hash in identity["source_sha256"].items():
        source = B.ROOT / rel.replace("\\", "/")
        if digest(source) != expected_hash:
            raise RuntimeError("Original source mismatch: " + rel)
    for rel, expected_hash in P.LEGACY_SHA256.items():
        if digest(B.ROOT / rel) != expected_hash:
            raise RuntimeError("Frozen physics baseline mismatch: " + rel)


def make_cell(args):
    specs = read(B.ROOT / "configs/cells_pilot9.json")["cells"]
    spec = next(s for s in specs if s["id"] == args.cell)
    return B.Cell({"Negative particle radius [m]": spec["negative_particle_radius_m"],
                   "SEI solvent diffusivity [m2.s-1]": spec["sei_solvent_diffusivity_m2s"]},
                  mesh_scale=args.mesh_scale)


def finite_scalar(step, name):
    x = B.scalar(step, name)
    if not math.isfinite(x):
        raise ValueError("Nonfinite scalar variable: " + name)
    return x


def array_stats(array):
    a = np.asarray(array, dtype=float)
    f = a[np.isfinite(a)]
    return {"shape": list(a.shape), "min": float(f.min()) if f.size else None,
            "max": float(f.max()) if f.size else None,
            "mean": float(f.mean()) if f.size else None,
            "nonfinite_count": int(a.size-f.size)}


def constants(cell):
    p = B.make_model().param
    return {"initial_total_lithium_mol": float(cell.param.evaluate(p.n_Li_init)),
            "initial_particle_lithium_mol": float(cell.param.evaluate(p.n_Li_particles_init)),
            "initial_electrolyte_lithium_mol": float(cell.param.evaluate(p.n_Li_e_init)),
            "F_C_per_mol": B.F,
            "negative_electrode_volume_m3": float(cell.neg_volume_m3)}


def physical_readout(cell, sol, const):
    step = B.last_step(sol)
    values, missing = {}, {}
    for key, variable in SCALAR_VARS.items():
        try:
            values[key] = finite_scalar(step, variable)
        except Exception as exc:
            missing[key] = {"variable": variable, "error": type(exc).__name__ + ": " + str(exc)}
    spatial = {}
    for key, variable in B.STATE_VARS.items():
        if key != "discharge_capacity":
            try:
                spatial[key] = array_stats(B.val(step, variable))
            except Exception as exc:
                missing[key] = {"variable": variable, "error": str(exc)}
    try:
        aging = {key: float(value) for key, value in cell.degradation_Ah(step).items()}
    except Exception as exc:
        aging = {}; missing["degradation_Ah"] = {"error": str(exc)}
    budgets = {}
    def calc(key, keys, expression):
        if all(k in values for k in keys):
            budgets[key] = float(expression(values))
    calc("total_equals_particles_plus_electrolyte_residual_mol",
         ["total_lithium_mol", "particle_lithium_mol", "electrolyte_lithium_mol"],
         lambda v: v["total_lithium_mol"]-v["particle_lithium_mol"]-v["electrolyte_lithium_mol"])
    calc("active_plus_side_reactions_minus_initial_total_residual_mol",
         ["total_lithium_mol", "side_reaction_lithium_lost_mol"],
         lambda v: v["total_lithium_mol"]+v["side_reaction_lithium_lost_mol"]-const["initial_total_lithium_mol"])
    calc("particles_LLI_formula_residual_percent", ["particle_lithium_mol", "LLI_particles_percent"],
         lambda v: v["LLI_particles_percent"]-100*(1-v["particle_lithium_mol"]/const["initial_particle_lithium_mol"]))
    calc("total_LLI_formula_residual_percent", ["total_lithium_mol", "LLI_with_electrolyte_percent"],
         lambda v: v["LLI_with_electrolyte_percent"]-100*(1-v["total_lithium_mol"]/const["initial_total_lithium_mol"]))
    calc("side_reaction_Ah_conversion_residual_Ah", ["side_reaction_lithium_lost_mol", "side_reaction_capacity_lost_Ah"],
         lambda v: v["side_reaction_capacity_lost_Ah"]-v["side_reaction_lithium_lost_mol"]*B.F/3600)
    component_keys = [key for key in values if key.endswith("_lithium_lost_mol") and key.startswith(("negative_", "positive_"))]
    if len(component_keys) == 6 and "side_reaction_lithium_lost_mol" in values:
        budgets["side_reaction_component_sum_residual_mol"] = values["side_reaction_lithium_lost_mol"]-sum(values[k] for k in component_keys)
    if "active_plus_side_reactions_minus_initial_total_residual_mol" in budgets:
        budgets["inventory_residual_fraction_of_initial"] = budgets["active_plus_side_reactions_minus_initial_total_residual_mol"]/const["initial_total_lithium_mol"]
    required = ("total_lithium_mol", "particle_lithium_mol", "electrolyte_lithium_mol",
                "side_reaction_lithium_lost_mol", "LLI_particles_percent", "LLI_with_electrolyte_percent")
    missing_required = [key for key in required if key not in values]
    missing_required += ["degradation_Ah."+key for key in ("sei_Ah", "plated_Ah", "dead_Ah") if key not in aging]
    return {"values": values, "spatial_state": spatial, "degradation_Ah": aging,
            "budgets": budgets, "missing_variables": missing,
            "missing_required_readouts": missing_required,
            "required_readouts_available": not bool(missing_required)}


def source_fingerprint(sol, ledger):
    h = hashlib.sha256()
    for y in sol.all_ys:
        a = np.asarray(y, dtype=float)
        h.update(str(a.shape).encode());h.update(a.tobytes())
    for t in sol.all_ts:
        h.update(np.asarray(t, dtype=float).tobytes())
    h.update(json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def stored_diagnostic(directory, efc):
    points = read(Path(directory) / "diagnostics.json")
    exact = [d for d in points if abs(float(d["efc"])-efc) < 1e-8]
    return {"same_EFC_reference": exact[-1] if exact else None,
            "last_reference_in_source": {k: points[-1].get(k) for k in ("efc", "target_efc", "Qref_Ah", "SOH_percent")}}


def worker(args):
    state_name = args._worker_state
    legacy_summary = read(Path(args.legacy_dir) / "summary.json")
    final_summary = read(Path(args.final_dir) / "summary.json")
    validate_identity(legacy_summary["identity"], args)
    validate_identity(final_summary["identity"], args)
    cell = make_cell(args)
    q0 = float(legacy_summary["ledger"]["q0_Ah"])
    if state_name == "initial":
        sol = cell.initial_state(soc=B.SOC_LO, rest_s=1.)
        ledger = B.ledger(B.RunState(args.cell, "audit_initial", q0_Ah=q0, t_abs_s=float(sol.t[-1])))
        source_dir = args.legacy_dir
    else:
        source_dir = args.legacy_dir if state_name == "legacy100" else args.final_dir
        source = R.load_checkpoint(Path(source_dir) / "checkpoint.pkl")
        summary = read(Path(source_dir) / "summary.json")
        if source["identity"] != summary["identity"] or source["ledger"] != summary["ledger"]:
            raise RuntimeError("Source checkpoint/summary identity or ledger mismatch")
        sol = source["solution"].last_state
        ledger = copy.deepcopy(source["ledger"])
    fingerprint_before = source_fingerprint(sol, ledger)
    y_before = B.y_end(sol)
    const = constants(cell)
    result = {"state": state_name, "source_EFC": float(ledger["efc"]), "source_ledger": ledger,
              "Q0_Ah": q0, "parameter_constants": const,
              "readout": physical_readout(cell, sol, const),
              "reference_history": stored_diagnostic(source_dir, float(ledger["efc"])),
              "reference_repeats": [], "started_utc": utc()}
    destination = Path(args.output_dir) / (state_name + ".json")
    save(destination, result)
    if not args.skip_rpt:
        for repeat in range(args.repeats):
            # New Cell and newly built REF model on every repetition.
            ref_cell = make_cell(args)
            before = source_fingerprint(sol, ledger)
            rpt = ref_cell.reference_test(sol.last_state.copy())
            branch = ref_cell._sims["REF"].solution
            after = source_fingerprint(sol, ledger)
            direct_error = float(abs(rpt["Q_trapz_Ah"]-rpt["Q_cum_Ah"]))
            record = {"repeat": repeat+1, "reference": rpt,
                      "SOH_percent": 100*float(rpt["Q_cum_Ah"])/q0,
                      "direct_integral_absolute_difference_Ah": direct_error,
                      "direct_integral_relative_difference": direct_error/abs(rpt["Q_cum_Ah"]),
                      "existing_integral_check_pass": bool(rpt["ok"] and direct_error/abs(rpt["Q_cum_Ah"]) <= 1e-4),
                      "source_fingerprint_before": before, "source_fingerprint_after": after,
                      "source_state_and_ledger_unchanged": before == after,
                      "source_y_max_abs_change": float(np.max(np.abs(B.y_end(sol)-y_before))),
                      "diagnostic_branch_aging_end": physical_readout(ref_cell, branch, const),
                      "memory_MB": B.memory_mb()}
            result["reference_repeats"].append(record)
            save(destination, result)
            if before != after:
                raise RuntimeError("Independent RPT mutated original state/ledger")
            del branch, ref_cell
    reps = result["reference_repeats"]
    if reps:
        qs = [r["reference"]["Q_cum_Ah"] for r in reps]
        result["repeat_comparison"] = {"count": len(qs), "max_minus_min_Ah": max(qs)-min(qs),
            "relative_spread": (max(qs)-min(qs))/abs(qs[0]),
            "SOH_spread_percent_points": 100*(max(qs)-min(qs))/q0,
            "exactly_equal": max(qs) == min(qs),
            "acceptance_note": "Raw spread reported; no new repeat tolerance was invented."}
        match = result["reference_history"]["same_EFC_reference"]
        if match:
            result["stored_same_EFC_RPT_difference_Ah"] = qs[0]-float(match["Qref_Ah"])
    result.update(finished_utc=utc(), source_fingerprint_before=fingerprint_before,
                  source_fingerprint_after=source_fingerprint(sol, ledger),
                  source_state_and_ledger_unchanged=source_fingerprint(sol, ledger) == fingerprint_before,
                  memory_MB=B.memory_mb(), status="audit_complete")
    save(destination, result)


def cross_state_report(results):
    initial = results.get("initial")
    if not initial:
        return {"available": False, "reason": "initial state readout unavailable"}
    iv = initial.get("readout", {}).get("values", {})
    ia = initial.get("readout", {}).get("degradation_Ah", {})
    out = {}
    for name, data in results.items():
        v = data.get("readout", {}).get("values", {})
        a = data.get("readout", {}).get("degradation_Ah", {})
        row = {"source_EFC": data.get("source_EFC"), "aging_change_from_initial_Ah":
               {k: a[k]-ia[k] for k in ("sei_Ah", "plated_Ah", "dead_Ah") if k in a and k in ia}}
        required = ("total_lithium_mol", "side_reaction_lithium_lost_mol")
        if all(k in v and k in iv for k in required):
            active_loss = iv["total_lithium_mol"]-v["total_lithium_mol"]
            side_gain = v["side_reaction_lithium_lost_mol"]-iv["side_reaction_lithium_lost_mol"]
            row.update(active_lithium_loss_mol=active_loss, side_reaction_gain_mol=side_gain,
                       differential_inventory_balance_residual_mol=active_loss-side_gain,
                       active_lithium_loss_Ah=active_loss*B.F/3600)
        if data.get("reference_repeats"):
            q = data["reference_repeats"][0]["reference"]["Q_cum_Ah"]
            row["RPT_capacity_loss_relative_to_Q0_Ah"] = data["Q0_Ah"]-q
            row["capacity_loss_vs_lithium_loss_note"] = "Delivered capacity and lithium inventory loss are distinct quantities; no equality requirement is imposed. RPT has finite current/voltage cutoffs and a separate diagnostic aging branch."
        out[name] = row
    out["interpretation_limits"] = [
        "Instantaneous reversible plated Li can fall during stripping; do not require monotonic plated Li at different SOC/phases.",
        "SEI/dead lithium, loss to side reactions, total-inventory conservation and electrode-only LLI must be distinguished.",
        "Independent copied RPT branches age during their own protocol; they do not contribute to the main trajectory EFC or aging.",
        "This audit does not transfer a checkpoint across spatial grids and does not establish mesh convergence.",
        "Only exact matching EFC is compared to a stored RPT; the last 1610-EFC RPT is not a 1614-EFC measurement.",
        "No experimental-cell validation or parameter calibration is performed by this script."]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-dir", required=True)
    p.add_argument("--final-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cell", type=int, default=4)
    p.add_argument("--current", type=float, default=3.2)
    p.add_argument("--mesh-scale", type=int, default=4)
    p.add_argument("--states", nargs="+", choices=("initial", "legacy100", "final"), default=["initial", "legacy100", "final"])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--skip-rpt", action="store_true")
    p.add_argument("--state-timeout-s", type=float, default=600.)
    p.add_argument("--_worker-state", choices=("initial", "legacy100", "final"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.repeats < 2 or not (0 < args.state_timeout_s <= 1800):
        p.error("At least two independent RPTs and a positive state timeout <=1800s are required")
    if args._worker_state:
        worker(args); return
    out = Path(args.output_dir).resolve()
    sources = [Path(args.legacy_dir).resolve(), Path(args.final_dir).resolve()]
    if any(out == source or out in source.parents or source in out.parents for source in sources):
        p.error("Audit output must be disjoint from both original source folders")
    if out.exists() and any(out.iterdir()):
        p.error("Use a new empty audit output folder")
    out.mkdir(parents=True, exist_ok=True)
    before = manifest(args)
    report = {"started_utc": utc(), "audit_sha256": digest(__file__), "source_manifest_before": before,
              "original_models_and_tolerances_unchanged": True,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: importlib.metadata.version(name) for name in PACKAGES}},
              "states": {}, "execution": {}, "limitations": ["Read-only trusted checkpoint audit; no parameter fitting or production state mutation."]}
    save(out / "audit_report.json", report)
    try:
        for name in args.states:
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker-state", name]
            started = time.perf_counter()
            with (out/(name+".log")).open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.state_timeout_s, check=False)
                    execution = {"returncode": completed.returncode, "wall_s": time.perf_counter()-started}
                except subprocess.TimeoutExpired:
                    execution = {"status": "timeout", "wall_s": time.perf_counter()-started}
            report["execution"][name] = execution
            result_file = out/(name+".json")
            if result_file.is_file():
                report["states"][name] = read(result_file)
            save(out/"audit_report.json", report)
    finally:
        after = manifest(args)
        report.update(source_manifest_after=after, original_source_files_unchanged=before == after,
                      cross_state=cross_state_report(report["states"]), finished_utc=utc(),
                      status="complete" if len(report["states"]) == len(args.states) and all(v.get("status") == "audit_complete" for v in report["states"].values()) else "partial_or_failed")
        if before != after:
            report["status"] = "original_source_modified"
        save(out/"audit_report.json", report)
    report["execution_complete"] = report["status"] == "complete"
    report["required_readouts_available"] = (bool(report["states"]) and
        all(state.get("readout", {}).get("required_readouts_available", False)
            for state in report["states"].values()))
    report["validation_status"] = ("readouts_available_for_review" if report["required_readouts_available"]
                                   else "incomplete_required_readouts")
    report["chemistry_verified"] = False
    report["validation_note"] = "Complete refers only to audit execution. Raw conservation residuals and repeated RPT differences require separate numerical and experimental review; no chemistry-validation claim is made."
    save(out/"audit_report.json", report)
    print(json.dumps({"status": report["status"], "report": str(out/"audit_report.json"),
                      "original_source_files_unchanged": report["original_source_files_unchanged"]}))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
