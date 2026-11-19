import itertools
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "outputs/risk_dataset1000"
out.mkdir(parents=True, exist_ok=True)
values = []
axes = ([0., 10., 25., 35., 40.], [.5, 1., 1.5, 2., 2.5], [.2, .3, .4, .5, .6],
        [round(5 + 5*i/7, 9) for i in range(8)])
for i, (temp, current, soc, cooling) in enumerate(itertools.product(*axes), 1):
    values.append(dict(scenario_id=f"risk1000_{i:04d}", initial_soc=soc,
                       initial_temperature_C=temp, ambient_temperature_C=temp,
                       charge_current_A=current, heat_transfer_W_m2K=cooling,
                       charge_horizon_s=1800., rest_s=300., voltage_limit_V=4.2,
                       temperature_stop_C=60., sample_period_s=1.,
                       plating_threshold_A_m2=.001, plating_persistence_s=5.,
                       mesh_scale=8, rtol=1e-7, atol=1e-9))
(out / "design_scenarios.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
print(len(values))
for i in [193, 993, 500, 745, 61, 412]:
    print(values[i-1])
