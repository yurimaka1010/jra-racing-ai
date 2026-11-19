"""Synthetic rankings and mocked counters only; no training or database calls."""
import copy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from jra_racing_ai import tuner_rank_ablation as ablation
from jra_racing_ai.ml.safety import MLFailure


class GainRankAblationTests(unittest.TestCase):
    def setUp(self):
        self.original = list(ablation.core.features.ALL_FEATURES)
        self.ranking = [{'feature': name, 'gain_rank': 39 - i}
                        for i, name in enumerate(self.original)]

    def test_remove_only_three_gain_ranks_and_preserve_native_order(self):
        removed = {row['feature'] for row in self.ranking if row['gain_rank'] in (24, 28, 30)}
        actual = ablation.remove_gain_ranks(self.ranking, self.original)
        self.assertEqual(len(actual), 36)
        self.assertEqual(actual, [name for name in self.original if name not in removed])
        self.assertNotEqual(removed, {self.original[i - 1] for i in (24, 28, 30)})

    def test_shuffled_table_does_not_change_selected_features(self):
        expected = ablation.remove_gain_ranks(self.ranking, self.original)
        self.assertEqual(expected, ablation.remove_gain_ranks(self.ranking[::-1], self.original))
        self.assertEqual(ablation.COUNTS, (36,))
        self.assertEqual(ablation.MAX_FITS, 8)

    def test_duplicate_missing_or_non_integer_ranks_are_rejected(self):
        invalid = []
        for value in (True, 0, 40, 2.5):
            rows = copy.deepcopy(self.ranking)
            rows[0]['gain_rank'] = value
            invalid.append(rows)
        invalid.append(self.ranking[:-1])
        rows = copy.deepcopy(self.ranking)
        rows[-1]['feature'] = rows[0]['feature']
        invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=len(rows)), self.assertRaises(MLFailure):
                ablation.remove_gain_ranks(rows, self.original)

    def test_changed_original_feature_order_is_rejected(self):
        with self.assertRaises(MLFailure):
            ablation.remove_gain_ranks(self.ranking, self.original[::-1])

    def test_ninth_fit_and_random_forest_are_blocked_before_reservation(self):
        with patch.object(ablation.core.CPU, 'sample', return_value=(1, 2, 3)), \
                patch.object(ablation.core.Budget, 'reserve') as reserve:
            budget = ablation.FixedBudget(Path('synthetic'), time.monotonic() + 10)
            budget.calls = 8
            with self.assertRaises(MLFailure):
                budget.reserve('LGBM', 'gain_drop_36')
            budget.calls = 0
            with self.assertRaises(MLFailure):
                budget.reserve('RF', 'forbidden')
            reserve.assert_not_called()

    def test_approved_plan_binds_the_exact_gain_subset(self):
        expected = ablation.remove_gain_ranks(self.ranking, self.original)
        plan = {'version': ablation.VERSION, 'ranking': self.ranking,
                'base_feature_spec': {'features': self.original},
                'removed_gain_ranks': [24, 28, 30],
                'configurations': [{'name': 'gain_drop_36', 'feature_count': 36, 'features': expected}]}
        approval = {'source': 'human_user', 'allow_execution': True,
                    'actual_fits_max': 8, 'plan_sha256': 'synthetic_hash'}
        with patch.object(ablation.core, 'read', side_effect=[plan, approval]), \
                patch.object(ablation.safety, 'digest', return_value='synthetic_hash'):
            self.assertEqual(ablation.approved(Path('synthetic')), plan)
        plan['configurations'][0]['features'] = self.original[:36]
        with patch.object(ablation.core, 'read', side_effect=[plan, approval]), \
                patch.object(ablation.safety, 'digest', return_value='synthetic_hash'), \
                patch.object(ablation.data, 'load_part') as load:
            with self.assertRaises(MLFailure):
                ablation.approved(Path('synthetic'))
            load.assert_not_called()

    def test_tolerance_uses_only_selection_metrics(self):
        rows = [{'name': 'A39', 'feature_count': 39, 'selection_metrics': {'race_log_loss': 2.0},
                 'evaluation_LL': 9.0},
                {'name': 'gain_drop_36', 'feature_count': 36, 'selection_metrics': {'race_log_loss': 2.0019},
                 'evaluation_LL': 10.0}]
        self.assertEqual(ablation.core.choose(rows, .002)['name'], 'gain_drop_36')
        self.assertEqual(ablation.core.choose(rows, 0.)['name'], 'A39')
        rows[1]['selection_metrics']['race_log_loss'] = 2.0021
        self.assertEqual(ablation.core.choose(rows, .002)['name'], 'A39')


if __name__ == '__main__':
    unittest.main()
