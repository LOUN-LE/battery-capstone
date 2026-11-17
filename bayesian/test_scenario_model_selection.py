"""Check that the previous GP and independent model selection are preserved."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from . import scenario_gp, scenario_model_selection as selection


class ScenarioModelSelectionTests(unittest.TestCase):
    def test_gp_candidate_preserves_original_forecasts(self):
        old = scenario_gp.evaluate_scenarios()['conditions']
        new = selection._fit_candidates()
        for (route, traffic), candidates in new.items():
            original = old[f'{route}/{traffic}']['forecasts_A']['condition_residual_gp']
            self.assertEqual(original, candidates['condition_residual_gp'].tolist())

    def test_independent_test_outcomes_cannot_change_selection_or_predictions(self):
        generator = scenario_gp.generate_episode
        first = selection.evaluate_model_selection()

        def changed_test_episode(route, traffic, seed):
            episode = generator(route, traffic, seed)
            if seed in selection.INDEPENDENT_TEST_SEEDS:
                return replace(episode, target_A=episode.target_A + .15)
            return episode

        with patch.object(scenario_gp, 'generate_episode', side_effect=changed_test_episode):
            changed = selection.evaluate_model_selection()

        self.assertEqual(first['selected_method'], changed['selected_method'])
        self.assertEqual(first['validation'], changed['validation'])
        self.assertEqual(first['forecasts_A'], changed['forecasts_A'])
        self.assertEqual(first['demo']['profiles']['selected_prediction'],
                         changed['demo']['profiles']['selected_prediction'])
        self.assertNotEqual(first['test']['metrics'], changed['test']['metrics'])


if __name__ == '__main__':
    unittest.main()
