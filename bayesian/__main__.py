"""Prepare team input or run a Bayesian workflow from the repository root."""

import argparse
import json
from pathlib import Path


def _write_new_report(path, result):
    data = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as report:
        report.write(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="workflow", required=True)
    road = commands.add_parser("evaluate-road-gp", help="Evaluate functional GP on public road telemetry by vehicle, not by driver/route")
    road.add_argument("--archive", type=Path, required=True)
    road.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare-driving", help="Run harmonized SOC/time/RPT comparison using explicitly synthetic GP demo profiles")
    compare.add_argument("--output", type=Path, required=True)
    scenario = commands.add_parser("compare-scenario-driving", help="Test a scenario-conditioned residual GP and compare virtual-cell outcomes")
    scenario.add_argument("--output", type=Path, required=True, help="New result directory")
    scenario.add_argument("--gp-only", action="store_true", help="Evaluate held-out current prediction without running SPMe")
    model_selection = commands.add_parser("compare-scenario-models", help="Select a virtual-driving forecaster on validation and verify it on fresh episodes")
    model_selection.add_argument("--output", type=Path, required=True, help="New result directory")
    model_selection.add_argument("--predictions-only", action="store_true", help="Compare predictors without running SPMe")
    habits = commands.add_parser("compare-habit-stress", help="Compare virtual driving habits and test GP prediction of paired stress differences")
    habits.add_argument("--output", type=Path, required=True, help="New result directory")
    habits.add_argument("--metrics-only", action="store_true", help="Run the controlled scenario and GP benchmark without SPMe")
    habit_prediction = commands.add_parser("predict-habit-burden", help="Predict virtual driving burden changes using the selected quadratic model")
    habit_prediction.add_argument("--launch-ramp-s", type=float, required=True)
    habit_prediction.add_argument("--brake-ramp-s", type=float, required=True)
    habit_prediction.add_argument("--output", type=Path, required=True, help="New JSON file")
    habit_life = commands.add_parser("compare-habit-life", help="Project virtual driving habits to the capacity threshold with BLAST-Lite")
    habit_life.add_argument("--habit-report", type=Path, required=True)
    habit_life.add_argument("--output", type=Path, required=True, help="New result directory")
    life_models = commands.add_parser("compare-habit-life-models", help="Tune and compare GP and quadratic waveform models against virtual lifetime")
    life_models.add_argument("--output", type=Path, required=True, help="New result directory")
    stability = commands.add_parser("charging-stability", help="Extend preserved BO observations and independently repeat the best feasible candidate")
    stability.add_argument("--prior-state", type=Path, required=True)
    stability.add_argument("--prior-report", type=Path, required=True)
    stability.add_argument("--budget", type=int, default=12)
    stability.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-input", help="Validate six-column pack telemetry and convert to equivalent cell units")
    prepare.add_argument("--csv", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--pack-config", type=Path, required=True, help="Explicit topology and mapping provenance")
    prepare.add_argument("--output", type=Path, required=True, help="New output directory")
    replay = commands.add_parser("replay-input", help="Replay a bounded source segment through SPMe and write result.json")
    replay.add_argument("--csv", type=Path, required=True)
    replay.add_argument("--metadata", type=Path, required=True)
    replay.add_argument("--pack-config", type=Path, required=True)
    replay.add_argument("--cycle-id", type=int, default=1)
    replay.add_argument("--duration-s", type=float, default=300.0)
    replay.add_argument("--cell-calibration", choices=("m50-c-joint-v3",),
                        help="Use the frozen C-cell material calibration with input BMS SOC")
    replay.add_argument("--thermal-mode", choices=("isothermal", "lumped"), default="isothermal",
                        help="Use isothermal SPMe or SPMe with lumped thermal dynamics")
    replay.add_argument("--ambient-temperature-c", type=float,
                        help="Constant ambient boundary temperature for lumped replay")
    replay.add_argument("--output", type=Path, required=True, help="New result directory")
    driving = commands.add_parser("driving", help="Forecast and compare future driving")
    driving.add_argument("--output", type=Path, required=True)
    driving.add_argument("--csv", type=Path, help="Normalized trip history. Omit for synthetic demo.")
    driving.add_argument("--target", help="Target vehicle ID")
    driving.add_argument("--reference", help="Matched reference vehicle ID")
    driving.add_argument("--target-driver", help="Target driver ID for shared vehicles")
    driving.add_argument("--reference-driver", help="Reference driver ID for shared vehicles")
    driving.add_argument("--route")
    driving.add_argument("--direction")
    driving.add_argument("--traffic")
    driving.add_argument("--distance-m", type=float)
    driving.add_argument("--duration-s", type=float)
    driving.add_argument("--as-of", help="Forecast origin with UTC offset")
    driving.add_argument("--samples", type=int, default=8)
    charging = commands.add_parser("charging", help="Evaluate or search charging currents")
    charging.add_argument("--output", type=Path, required=True)
    modes = charging.add_mutually_exclusive_group()
    modes.add_argument("--current-a", type=float, help="Evaluate a single current (default 3 A)")
    modes.add_argument("--budget", type=int, help="Run a bounded Bayesian search")
    args = parser.parse_args(argv)

    if args.workflow == "charging-stability":
        from .charging_stability import run_stability

        try:
            run_stability(args.prior_state, args.prior_report, args.output,
                          budget=args.budget, progress=print)
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        print(args.output.resolve())
        return
    if args.workflow == "evaluate-road-gp":
        from .road_gp import evaluate_archive

        try:
            evaluate_archive(args.archive, args.output)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(args.output.resolve())
        return
    if args.workflow == "compare-driving":
        from .driving import demo_trips, predict_next
        from .driving_comparison import compare_profiles

        context = dict(reference_vehicle="reference", target_vehicle="personal",
                       route_id="demo-city-loop", direction="clockwise", traffic="moderate",
                       distance_m=3000.0, duration_s=320.0)
        prediction = predict_next(demo_trips(), **context)
        task = {key: context[key] for key in ("route_id", "direction", "traffic", "distance_m")}
        task.update(cell_basis_id=prediction.context["cell_basis_id"], source_kind="synthetic_demo")
        compare_profiles({"matched_reference": prediction.reference_profile, "gp_mean": prediction.expected_profile},
                         task, args.output, progress=print)
        print((args.output / "driving_comparison.json").resolve())
        return
    if args.workflow == "compare-scenario-driving":
        from .scenario_gp import run_scenario_gp

        try:
            result = run_scenario_gp(args.output)
            print((args.output / "scenario_gp.json").resolve(), flush=True)
            if not args.gp_only:
                from .scenario_comparison import run_scenario_comparison

                run_scenario_comparison(result, args.output, progress=print)
                print((args.output / "scenario_comparison.json").resolve())
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        return
    if args.workflow == "compare-scenario-models":
        from .scenario_model_selection import run_model_selection

        try:
            result = run_model_selection(args.output)
            print((args.output / "scenario_model_selection.json").resolve(), flush=True)
            if not args.predictions_only:
                from .scenario_comparison import run_scenario_comparison

                run_scenario_comparison(result, args.output, prediction_arm="selected_prediction",
                                        prediction_filename="scenario_model_selection.json", progress=print)
                print((args.output / "scenario_comparison.json").resolve())
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        return
    if args.workflow == "compare-habit-stress":
        from .habit_stress import run_habit_stress

        try:
            result = run_habit_stress(args.output)
            print((args.output / "habit_stress.json").resolve(), flush=True)
            if not args.metrics_only:
                from .habit_stress_twin import replay_habits

                replay_habits(result, args.output, progress=print)
                print((args.output / "habit_stress_spme.json").resolve())
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        return
    if args.workflow == "predict-habit-burden":
        from .habit_predictor import predict_habit_burden

        try:
            result = predict_habit_burden(args.launch_ramp_s, args.brake_ramp_s)
            _write_new_report(args.output, result)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(args.output.resolve())
        return
    if args.workflow == "compare-habit-life":
        from .habit_life import run_habit_life

        try:
            run_habit_life(args.habit_report, args.output, progress=print)
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        print((args.output / "habit_life.json").resolve())
        return
    if args.workflow == "compare-habit-life-models":
        from .habit_life_benchmark import run_life_model_benchmark

        try:
            run_life_model_benchmark(args.output, progress=print)
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        print((args.output / "habit_life_model_benchmark.json").resolve())
        return
    if args.workflow == "replay-input":
        from .pack_replay import run_replay

        try:
            run_replay(args.csv, args.metadata, args.pack_config, args.output,
                       cycle_id=args.cycle_id, duration_s=args.duration_s,
                       cell_calibration=args.cell_calibration, thermal_mode=args.thermal_mode,
                       ambient_temperature_c=args.ambient_temperature_c)
        except (ValueError, OSError, RuntimeError) as error:
            parser.error(str(error))
        print((args.output / "result.json").resolve())
        return
    if args.workflow == "prepare-input":
        from .pack_input import prepare_input

        try:
            prepare_input(args.csv, args.metadata, args.pack_config, args.output)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(args.output.resolve())
        return
    if args.workflow == "charging":
        from .charging import run_baseline, run_search

        if args.budget is not None:
            run_search(args.output, budget=args.budget)
            print(args.output.resolve())
            return
        if args.output.exists():
            parser.error("Baseline report already exists")
        current = 3.0 if args.current_a is None else args.current_a
        report = run_baseline(current)
        evidence = report["result"]
        reason = None
        if not evidence["physical_constraints_valid"]:
            reason = "physical_checks_failed"
        elif not evidence["numerically_resolved"]:
            reason = "numerical_resolution_failed"
        elif report["objective_pp"] < -evidence["numerical_error_pp"]:
            reason = "apparent_capacity_gain_exceeds_mesh_difference"
        report["status"] = "invalid" if reason else "complete"
        report["feasible"] = reason is None and report["meets_deadline"]
        if reason:
            report["objective_pp"] = None
        report["reason"] = reason or (None if report["meets_deadline"] else "charge_deadline_exceeded")
        _write_new_report(args.output, report)
    else:
        from .driving import demo_trips, predict_next, read_trips
        from .driving_twin import evaluate_prediction

        if args.output.exists():
            parser.error("Driving report already exists")
        inputs = {
            "target_vehicle": args.target, "reference_vehicle": args.reference,
            "route_id": args.route, "direction": args.direction, "traffic": args.traffic,
            "distance_m": args.distance_m, "duration_s": args.duration_s,
        }
        if args.csv:
            if any(value is None for value in inputs.values()) or args.as_of is None:
                parser.error("--csv requires both IDs, route, direction, traffic, distance, duration and --as-of")
            trips = read_trips(args.csv)
            source = "user_supplied_normalized_cell_current_csv"
        else:
            if (any(value is not None for value in inputs.values()) or args.as_of is not None
                    or args.target_driver is not None or args.reference_driver is not None):
                parser.error("Trip context options require --csv")
            trips = demo_trips()
            source = "synthetic_demo"
            inputs = dict(target_vehicle="personal", reference_vehicle="reference",
                          route_id="demo-city-loop", direction="clockwise", traffic="moderate",
                          distance_m=3000.0, duration_s=320.0)
        if args.csv:
            inputs.update(target_driver=args.target_driver, reference_driver=args.reference_driver)
        prediction = predict_next(trips, **inputs, as_of=args.as_of, samples=args.samples)
        _write_new_report(args.output, {
            "source": source, "forecast": prediction.to_dict(),
            "spme": evaluate_prediction(prediction),
        })
    print(args.output.resolve())


if __name__ == "__main__":
    main()
