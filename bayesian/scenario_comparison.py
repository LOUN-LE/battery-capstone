"""Evaluate a prespecified synthetic trip in the existing virtual-cell protocol."""
from pathlib import Path

from .driving_comparison import compare_profiles
from .scenario_gp import demo_comparison
from simulation.rebuilt_lifetime import sha, write_json


def run_scenario_comparison(prediction_report, output, *, prediction_arm="gp_mean",
                            prediction_filename="scenario_gp.json", progress=None):
    output = Path(output)
    profiles, context = demo_comparison(prediction_report)
    if set(profiles) != {"matched_reference", "heldout_target", prediction_arm}:
        raise ValueError("Scenario comparison requires its reference, selected prediction and held-out profile")
    twin = compare_profiles(profiles, context, output / "spme", progress=progress)

    # Anchor both differences to the same reference driving policy on each mesh.
    predicted_key = f"{prediction_arm}_vs_reference_pp"
    effect_error_key = ("gp_minus_heldout_capacity_effect_pp" if prediction_arm == "gp_mean"
                        else "selected_minus_heldout_capacity_effect_pp")
    mesh_rows = []
    for mesh in twin["mesh_results"]:
        arms = mesh["arms"]
        reference = arms["matched_reference"]["rpt_capacity_Ah"]
        observed = 100 * (reference - arms["heldout_target"]["rpt_capacity_Ah"]) / reference
        predicted = 100 * (reference - arms[prediction_arm]["rpt_capacity_Ah"]) / reference
        mesh_rows.append({
            "mesh": mesh["mesh"],
            "reference_rpt_capacity_Ah": reference,
            "heldout_target_vs_reference_pp": observed,
            predicted_key: predicted,
            effect_error_key: predicted - observed,
        })
    coarse, fine = mesh_rows
    keys = ("heldout_target_vs_reference_pp", predicted_key)
    shifts = {key: abs(fine[key] - coarse[key]) for key in keys}
    report = {
        "schema": "virtual_scenario_capacity_comparison_v1",
        "status": "complete",
        "scope": "Synthetic drive, recovery and rest followed by RPT in a virtual cell",
        "method": "Positive values denote lower RPT capacity than the scenario reference. Mesh shifts are sensitivity checks, not certified error bounds.",
        "duty": twin["duty"],
        "all_arms_pass_model_checks": all(arm["physical"]["physical_constraints_valid"]
                                          for row in twin["mesh_results"] for arm in row["arms"].values()),
        "mesh_results": mesh_rows,
        "absolute_change_between_tested_meshes_pp": shifts,
        "duty_matched_ranking": twin["summary"]["duty_matched_ranking"],
        "lifetime_extension_days": None,
        "prediction_report_filename": prediction_filename,
        "prediction_arm": prediction_arm,
        "prediction_report_sha256": sha(output / prediction_filename),
        "spme_report_sha256": sha(output / "spme/driving_comparison.json"),
    }
    write_json(output / "scenario_comparison.json", report)
    return report
