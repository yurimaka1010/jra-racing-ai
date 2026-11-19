"""Synthetic data only for the finite four-model study."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from jra_racing_ai import tuner_four_models as study
from jra_racing_ai.ml.safety import MLFailure


class FourModelStudyTests(unittest.TestCase):
    def test_Gain_top10_and_top20_preserve_original_native_order(self):
        original = list(study.core.features.ALL_FEATURES)
        ranks = [{'feature': name, 'gain_rank': 39-i} for i, name in enumerate(original)]
        for name, count in study.NEW_COUNTS.items():
            selected = study.top_features(ranks, count, original)
            self.assertEqual(len(selected), count)
            self.assertEqual(selected, original[-count:])
        self.assertEqual(study.MAX_FITS, 16)

    def test_strict_selection_does_not_apply_old_tolerance_or_evaluation_results(self):
        rows = [{'name': 'D39', 'feature_count': 39, 'selection_metrics': {'race_log_loss': 2.0}, 'evaluation_LL': 9.},
                {'name': 'B20', 'feature_count': 20, 'selection_metrics': {'race_log_loss': 2.0001}, 'evaluation_LL': 1.}]
        self.assertEqual(study.select_strict(rows)['name'], 'D39')
        rows[1]['selection_metrics']['race_log_loss'] = 1.999999
        self.assertEqual(study.select_strict(rows)['name'], 'B20')

    def test_nonfinite_selection_loss_stops(self):
        for value in (np.nan, np.inf):
            with self.assertRaises(MLFailure):
                study.select_strict([{'name': 'mock', 'selection_metrics': {'race_log_loss': value}}])

    def test_saved_models_Boruta_and_fit17_never_consume_an_attempt(self):
        with patch.object(study.core.CPU, 'sample', return_value=(1,2,3)), \
                patch.object(study.core.Budget, 'reserve') as reserve:
            budget = study.FourModelBudget(Path('synthetic'), time.monotonic()+10)
            for kind, name in (('LGBM','C19'), ('LGBM','D39'), ('RF','Boruta')):
                with self.assertRaises(MLFailure):
                    budget.reserve(kind, name)
            budget.calls = 16
            with self.assertRaises(MLFailure):
                budget.reserve('LGBM', 'A10')
            reserve.assert_not_called()

    def test_official_track_groups_keep_sand_jump_and_unknown_separate(self):
        for code in ('10','11','17','22'):
            self.assertEqual(study.surface_group(code), 'turf')
        for code in ('23','24','25','26','29'):
            self.assertEqual(study.surface_group(code), 'dirt')
        self.assertEqual(study.surface_group('27'), 'sand')
        self.assertEqual(study.surface_group('52'), 'jump')
        for value in (None, pd.NA, '00', '11.0', '99'):
            self.assertEqual(study.surface_group(value), 'unknown')

    def test_distance_bins_cover_boundaries_and_keep_missing(self):
        for number, group in ((1400,'up_to_1400m'),(1401,'1401_1800m'),(1800,'1401_1800m'),
                              (1801,'1801_2200m'),(2200,'1801_2200m'),(2201,'2201m_and_above')):
            self.assertEqual(study.distance_group(number), group)
        for value in (None, np.nan, np.inf, pd.NA, -1, 1600.5, True):
            self.assertEqual(study.distance_group(value), 'unknown')

    def synthetic_frame(self):
        return pd.DataFrame({'entry_key':['202501050601010101','202501050601010102','202501050601010201','202501050601010202'],
                             'date':['2025-01-05']*4, 'winner_label':[1,0,0,1],
                             'distance':[1200,1200,1600,1600], 'surface':['11','11','23','23']})

    def test_subgroups_have_the_same_rosters_and_do_not_drop_inconsistent_races(self):
        frame = self.synthetic_frame()
        frame.loc[1,'distance'] = np.nan
        predictions = frame[['entry_key','date','winner_label']].copy()
        for name in study.NAMES:
            predictions[name] = [.6,.4,.2,.8]
        rows = study.grouped_metrics(frame, predictions)
        for axis in ('distance','surface'):
            for name in study.NAMES:
                chosen = [row for row in rows if row['axis']==axis and row['configuration']==name]
                self.assertEqual(sum(row['races'] for row in chosen), 2)
                self.assertEqual(sum(row['runners'] for row in chosen), 4)
        self.assertIn('inconsistent', {row['group'] for row in rows})

    def test_saved_prediction_join_is_by_key_and_rejects_label_changes(self):
        frame = self.synthetic_frame()
        saved = frame[['entry_key','date','winner_label']].copy()
        saved['Boruta_C'] = [.6,.4,.2,.8]
        saved['A39'] = [.7,.3,.1,.9]
        with patch.object(pd,'read_csv', return_value=saved.iloc[::-1]):
            result = study.frozen_probabilities(frame)
            np.testing.assert_array_equal(result['C19'], [.6,.4,.2,.8])
            np.testing.assert_array_equal(result['D39'], [.7,.3,.1,.9])
        saved.loc[0,'winner_label'] = 0
        with patch.object(pd,'read_csv', return_value=saved), self.assertRaises(MLFailure):
            study.frozen_probabilities(frame)

    def test_calibration_summary_uses_all_occupied_bands(self):
        result = study.calibration_summary([{'runners':2,'mean_prediction':.2,'observed_win_rate':.1},
                                            {'runners':1,'mean_prediction':.4,'observed_win_rate':.7},
                                            {'runners':0,'mean_prediction':None,'observed_win_rate':None}])
        self.assertAlmostEqual(result['fixed_bin_ECE'], .5/3)


if __name__ == '__main__':
    unittest.main()
