"""Synthetic rankings and mocked counters, never real model fits."""
import time
from pathlib import Path
import unittest
from unittest.mock import patch
from jra_racing_ai import tuner_fixed_subsets as f
from jra_racing_ai.ml.safety import MLFailure


class FixedSubsetTests(unittest.TestCase):
    def test_all_three_counts_use_the_same_origin_ranking_and_native_order(self):
        original=list(f.core.features.ALL_FEATURES)
        ranking=[{'feature':n,'gain_rank':len(original)-i} for i,n in enumerate(original)]
        for k in f.COUNTS:
            keep=f.top_features(ranking,k,original)
            self.assertEqual(len(keep),k);self.assertEqual(keep,original[-k:])
        self.assertTrue(set(f.top_features(ranking,24,original)) < set(f.top_features(ranking,28,original)))

    def test_invalid_counts_duplicate_names_or_bad_ranks_stop(self):
        original=list(f.core.features.ALL_FEATURES)
        ranking=[{'feature':n,'gain_rank':i+1} for i,n in enumerate(original)]
        for k in (0,40,True):
            with self.assertRaises(MLFailure):f.top_features(ranking,k,original)
        ranking[-1]['gain_rank']=1
        with self.assertRaises(MLFailure):f.top_features(ranking,24,original)

    def test_actual_fit_25_and_any_RF_are_rejected(self):
        with patch.object(f.core.CPU,'sample',return_value=(1,2,3)),patch.object(f.core.Budget,'reserve') as reserve:
            budget=f.FixedBudget(Path('synthetic'),time.monotonic()+10);budget.calls=24
            with self.assertRaises(MLFailure):budget.reserve('LGBM','fixed_24')
            budget.calls=0
            with self.assertRaises(MLFailure):budget.reserve('RF','not_allowed')
            reserve.assert_not_called()

    def test_latest_tolerance_prefers_smaller_only_inside_band(self):
        rows=[{'name':'A39','feature_count':39,'selection_metrics':{'race_log_loss':2.0}},
              {'name':'fixed_24','feature_count':24,'selection_metrics':{'race_log_loss':2.003}},
              {'name':'fixed_28','feature_count':28,'selection_metrics':{'race_log_loss':2.001}}]
        self.assertEqual(f.core.choose(rows,.002)['name'],'fixed_28')
        self.assertEqual(f.core.choose(rows,0.)['name'],'A39')

    def test_missing_authorization_blocks_before_worker_data(self):
        with patch.object(f.core,'read',side_effect=[{'version':f.VERSION},{'source':'not_human'}]), \
             patch.object(f.data,'load_part') as load:
            with self.assertRaises(MLFailure):f.approved(Path('synthetic'))
            load.assert_not_called()


if __name__=='__main__':unittest.main()
