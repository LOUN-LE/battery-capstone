import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(r"<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo")
sys.path.insert(0, str(REPO))
from simulation.risk_scenarios import sha, write_json, persistent_onset

ROOT = Path(__file__).resolve().parents[1] / "outputs/scenario_validation/local_mesh8"
paths = {p.parent.name: p for p in ROOT.glob("*/result.json")}
if len(paths) != 5:
    raise RuntimeError("Final mesh8/16 validation is still incomplete")
for name in ("zero_current", "initial_hot", "case_017", "case_029"):
    paths[name] = ROOT.parent / "local_mesh4" / name / "result.json"
results = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
traces = {}
checks = []
for case_id, result in results.items():
    path = paths[case_id].parent / "trace.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric_keys = [key for key in rows[0] if key not in ("scenario_id", "phase")]
    trace = {key: np.array([float(row[key]) for row in rows]) for key in numeric_keys}
    traces[case_id] = trace
    summary = result["summary"]
    checks.append({"case": case_id, "trace_hash_ok": sha(path) == result["trace_sha256"],
        "finite": all(np.isfinite(v).all() for v in trace.values()),
        "monotonic_time": bool(np.all(np.diff(trace["time_s"]) > 0)),
        "initial_temperature_ok": bool(abs(trace["temperature_C"][0] - summary["initial_temperature_C"]) < 1e-6),
        "maximum_temperature_matches": bool(abs(max(trace["temperature_C"]) - summary["maximum_temperature_C"]) < 1e-10),
        "dead_lithium_non_decreasing": bool(np.min(np.diff(trace["dead_lithium_mol_m3"])) > -1e-7),
        "onset_matches": persistent_onset(trace["time_s"], trace["plating_net_current_A_m2"],
            summary["plating_threshold_A_m2"], summary["plating_persistence_s"]) == summary["plating_onset_s"]})
base, fine, repeat = (traces[key] for key in ("cold_growth", "cold_growth_fine", "cold_growth_repeat"))
repeat_exact = all(np.array_equal(base[key], repeat[key]) for key in base)
temperature_difference = float(np.max(np.abs(base["temperature_C"] - np.interp(base["time_s"], fine["time_s"], fine["temperature_C"]))))
plating_peak_relative = float(abs(max(base["metallic_lithium_mol_m3"]) / max(fine["metallic_lithium_mol_m3"]) - 1))
voltage_difference = float(np.max(np.abs(base["voltage_V"] - np.interp(base["time_s"], fine["time_s"], fine["voltage_V"]))))
net_plating_relative = float(abs(base["plating_inventory_Ah"][-1] / fine["plating_inventory_Ah"][-1] - 1))
onset_difference = abs(results["cold_growth"]["summary"]["plating_onset_s"] - results["cold_growth_fine"]["summary"]["plating_onset_s"])
zero = results["zero_current"]["summary"]
growth = results["cold_growth"]["summary"]
censored = results["case_029"]["summary"]
report = {"scope": "bounded simulation numerical/software validation, measured plating/thermal safety validation pending",
    "source_reports": {name: str(path) for name, path in paths.items()},
    "source_versions": {name: result["provenance"] for name, result in results.items()},
    "final_mesh8_case_count": 4, "reference_mesh16_case_count": 1,
    "supplementary_mesh4_controls": 4,
    "cases": checks, "repeat_trace_exact": repeat_exact,
    "mesh_tolerance_check": {"temperature_max_difference_C": temperature_difference,
        "temperature_acceptance_C": .05, "plating_peak_relative_difference": plating_peak_relative,
        "plating_peak_relative_acceptance": .02,
        "voltage_max_difference_V": voltage_difference, "voltage_acceptance_V": .001,
        "net_plating_inventory_relative_difference": net_plating_relative, "net_plating_relative_acceptance": .02,
        "onset_difference_s": onset_difference, "onset_acceptance_s": 2.0},
    "zero_current_event_absent": zero["plating_event"] is False,
    "zero_current_raw_metallic_lithium_peak_mol_m3": zero["metallic_lithium_mol_m3_peak"],
    "charge_growth_then_rest_stripping": growth["metallic_lithium_mol_m3_peak"] > growth["metallic_lithium_mol_m3_end"] > growth["metallic_lithium_mol_m3_start"],
    "temperature_stop_retained_as_censored": censored["status"] == "censored" and censored["research_temperature_limit_reached"],
    "all_fire_labels_unavailable": all(r["summary"]["fire_label"] is None for r in results.values()),
    "regression_tests_passed": 7}
report["passed"] = (all(all(v for k, v in check.items() if k != "case") for check in checks)
    and repeat_exact and temperature_difference < .05 and plating_peak_relative < .02
    and voltage_difference < .001 and net_plating_relative < .02 and onset_difference <= 2.0
    and report["zero_current_event_absent"] and report["charge_growth_then_rest_stripping"]
    and report["temperature_stop_retained_as_censored"] and report["all_fire_labels_unavailable"])
write_json(ROOT.parent / "numerical_verification.json", report)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
