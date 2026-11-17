"""Select a virtual driving forecast on validation trips before testing it.

All point forecasts receive the same route/traffic labels, elapsed grid and
training records. In particular, nominal-phase pooling sees the declared
schedule for a condition; realized timing of a future trip is withheld.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import scenario_gp as scenario


VALIDATION_SEEDS = tuple(range(701, 765))
INDEPENDENT_TEST_SEEDS = tuple(range(901, 965))
DEMO_EPISODE = ('urban_grid', 'moderate', 901)
CONTEXTS = tuple((route, traffic) for route in scenario.ROUTES for traffic in scenario.TRAFFIC)
METHODS = {
    'all_conditions_reference_mean': 'Reference mean across all routes and traffic',
    'all_conditions_personal_mean': 'Personal mean across all routes and traffic',
    'condition_reference_mean': 'Reference mean within the route and traffic condition',
    'condition_personal_mean': 'Personal mean within the route and traffic condition',
    'condition_personal_median': 'Pointwise personal median within the condition',
    'condition_personal_trimmed_mean': 'Personal mean after trimming 12.5% at each tail',
    'condition_personal_smoothed_mean': 'Personal mean smoothed with a fixed 1 s Gaussian scale',
    'condition_residual_gp': 'Original conditional reference plus functional GP residual',
    'phase_residual_pool_20': 'Nominal-phase residual pooling, 20% from other traffic on the same route',
    'phase_residual_pool_40': 'Nominal-phase residual pooling, 40% from other traffic on the same route',
}


def nominal_boundaries(traffic: str) -> np.ndarray:
    """Get fixed start/stop landmarks from the declared scenario timetable."""
    movement, waits = scenario.TRAFFIC[traffic]
    starts = 4 + np.r_[0, np.cumsum(np.array(movement[:-1]) + waits)]
    ends = starts + movement
    return np.r_[0., np.column_stack([starts, ends]).ravel(), scenario.ELAPSED_S[-1]]


def _phase_pool(residual: np.ndarray, peer_residual: np.ndarray,
                traffic: str, peer_traffic: str, weight: float) -> np.ndarray:
    peer_time = np.interp(scenario.ELAPSED_S, nominal_boundaries(traffic), nominal_boundaries(peer_traffic))
    aligned_peer = np.interp(peer_time, scenario.ELAPSED_S, peer_residual)
    return (1. - weight) * residual + weight * aligned_peer


def _fit_candidates() -> dict:
    training = {}
    for context in CONTEXTS:
        pairs = [scenario.generate_episode(*context, seed) for seed in scenario.TRAIN_SEEDS]
        training[context] = (np.stack([ep.reference_A for ep in pairs]), np.stack([ep.target_A for ep in pairs]))
    reference_mean = np.mean(np.concatenate([ref for ref, _ in training.values()]), axis=0)
    personal_mean = np.mean(np.concatenate([target for _, target in training.values()]), axis=0)
    residuals = {context: (target - ref).mean(axis=0) for context, (ref, target) in training.items()}
    # Preserve the original scenario GP definition, including the fixed kernel.
    time = scenario.ELAPSED_S
    distance = np.sqrt(3.) * abs(time[:, None] - time[None, :]) / scenario.LENGTH_SCALE_S
    kernel = (1. + distance) * np.exp(-distance)
    forecasts = {}
    for (route, traffic), (ref, target) in training.items():
        baseline, gp, _ = scenario._fit(target, ref, kernel)
        own_mean = target.mean(axis=0)
        trim = len(target) // 8
        peer_traffic, = (value for value in scenario.TRAFFIC if value != traffic)
        residual = residuals[(route, traffic)]
        peer = residuals[(route, peer_traffic)]
        forecasts[(route, traffic)] = {
            'all_conditions_reference_mean': reference_mean,
            'all_conditions_personal_mean': personal_mean,
            'condition_reference_mean': baseline,
            'condition_personal_mean': own_mean,
            'condition_personal_median': np.median(target, axis=0),
            'condition_personal_trimmed_mean': np.sort(target, axis=0)[trim:len(target)-trim].mean(axis=0),
            'condition_personal_smoothed_mean': gaussian_filter1d(own_mean, sigma=1., mode='nearest'),
            'condition_residual_gp': gp,
            'phase_residual_pool_20': baseline + _phase_pool(residual, peer, traffic, peer_traffic, .2),
            'phase_residual_pool_40': baseline + _phase_pool(residual, peer, traffic, peer_traffic, .4),
        }
    return forecasts


def _evaluate_split(forecasts: dict, seeds: tuple[int, ...], *, demo_method: str | None = None):
    conditions = {}
    outcomes = []
    repeated_forecasts = {name: [] for name in METHODS}
    demo = None
    for (route, traffic), models in forecasts.items():
        truth = []
        for seed in seeds:
            episode = scenario.generate_episode(route, traffic, seed)
            truth.append(episode.target_A)
            if demo_method is not None and (route, traffic, seed) == DEMO_EPISODE:
                demo = _comparison_demo(episode, route, traffic, seed, demo_method, models)
        observed = np.stack(truth)
        conditions[f'{route}/{traffic}'] = {
            'metrics': {name: scenario._metrics(observed, curve) for name, curve in models.items()},
            'target_current_A': observed.tolist(),
        }
        outcomes.append(observed)
        for name, curve in models.items():
            repeated_forecasts[name].append(np.broadcast_to(curve, observed.shape))
    observed = np.concatenate(outcomes)
    metrics = {name: scenario._metrics(observed, np.concatenate(curves)) for name, curves in repeated_forecasts.items()}
    return {'metrics': metrics, 'conditions': conditions,
            'episode_count': len(observed), 'sample_count': int(observed.size),
            'rank_by_rmse': sorted(metrics, key=lambda name: (metrics[name]['rmse_A'], name))}, demo


def _comparison_demo(episode, route, traffic, seed, selected, models):
    time = scenario.ELAPSED_S
    return {
        'episode_seed': seed, 'schedule': episode.schedule, 'selected_method': selected,
        'context': {'route_id': route, 'direction': 'forward', 'traffic': traffic,
                    'distance_m': sum(scenario.ROUTES[route][0]), 'cell_basis_id': scenario.CELL_BASIS_ID,
                    'source_kind': 'synthetic_selected_current_forecast'},
        'profiles': {'matched_reference': scenario._curve(episode.reference_A),
                     'selected_prediction': scenario._curve(models[selected]),
                     'heldout_target': scenario._curve(episode.target_A)},
        'baseline_profiles': {name: scenario._curve(models[name]) for name in
                              ('condition_reference_mean', 'condition_personal_mean', 'condition_residual_gp')},
        'reference_speed_m_s': episode.reference_speed_m_s.tolist(),
        'target_speed_m_s': episode.target_speed_m_s.tolist(),
        'integrated_reference_distance_m': float(np.trapezoid(episode.reference_speed_m_s, time)),
        'integrated_target_distance_m': float(np.trapezoid(episode.target_speed_m_s, time)),
        'metrics': {name: scenario._metrics(episode.target_A, curve) for name, curve in models.items()},
    }


def evaluate_model_selection() -> dict:
    split_seeds = (scenario.TRAIN_SEEDS, VALIDATION_SEEDS, INDEPENDENT_TEST_SEEDS)
    if sum(map(len, split_seeds)) != len(set().union(*split_seeds)):
        raise ValueError('Train, validation and test episode seeds must be disjoint')
    forecasts = _fit_candidates()
    validation, _ = _evaluate_split(forecasts, VALIDATION_SEEDS)
    selected = validation['rank_by_rmse'][0]
    # This is the first generation of independent test outcomes. The winner
    # and all forecasts are already fixed; validation data did not refit them.
    test, demo = _evaluate_split(forecasts, INDEPENDENT_TEST_SEEDS, demo_method=selected)
    if demo is None:
        raise RuntimeError('Predesignated comparison episode is missing')
    previous_gp_rmse = test['metrics']['condition_residual_gp']['rmse_A']
    selected_rmse = test['metrics'][selected]['rmse_A']
    root = Path(__file__).resolve().parent
    return {
        'schema': 'synthetic_driving_model_selection_v1', 'status': 'complete',
        'scope': 'forecasting fresh synthetic episodes in four known route/traffic conditions',
        'real_road_validation': False, 'unseen_condition_validation': False,
        'selected_method': selected,
        'selection_metric': 'pooled validation RMSE in cell A, ascending; method ID breaks exact ties',
        'selected_method_test_rank': 1 + test['rank_by_rmse'].index(selected),
        'selected_test_rmse_reduction_vs_gp_pct': 100. * (1. - selected_rmse / previous_gp_rmse),
        'method_descriptions': METHODS, 'elapsed_s': scenario.ELAPSED_S.tolist(),
        'split': {'train_seeds': list(scenario.TRAIN_SEEDS), 'validation_seeds': list(VALIDATION_SEEDS),
                  'test_seeds': list(INDEPENDENT_TEST_SEEDS),
                  'train_episode_count': len(scenario.TRAIN_SEEDS) * len(CONTEXTS),
                  'demo_episode': list(DEMO_EPISODE),
                  'forecast_inputs': ['route ID', 'traffic label', 'elapsed time', 'fixed training history'],
                  'reused_realized_test_timing_as_feature': False, 'refit_after_selection': False},
        'nominal_phase_boundaries_s': {name: nominal_boundaries(name).tolist() for name in scenario.TRAFFIC},
        'forecasts_A': {f'{route}/{traffic}': {name: curve.tolist() for name, curve in models.items()}
                        for (route, traffic), models in forecasts.items()},
        'validation': validation, 'test': test, 'demo': demo,
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                          ('scenario_model_selection.py', 'scenario_gp.py', 'driving.py')},
    }


def run_model_selection(output) -> dict:
    """Write selection, evaluation and unmodified comparison curves to a new folder."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    report = evaluate_model_selection()
    for name, content in (('scenario_model_selection.json', report), ('demo_profiles.json', report['demo'])):
        (path / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
