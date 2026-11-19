"""Synthetic records and mock budgets only. No new real-data learning."""
from copy import deepcopy
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from jra_racing_ai import tuner_reduction as r
from jra_racing_ai.ml.safety import MLFailure


class ReductionTests(unittest.TestCase):
    def test_initial_32_and_finite_fit_budget(self):
        names = [n for n in r.features.ALL_FEATURES if n not in r.REMOVE]
        self.assertEqual(len(r.REMOVE), 7); self.assertEqual(len(names), 32)
        self.assertEqual(r.COUNTS, tuple(range(32, 23, -1)))
        self.assertEqual(len(r.COUNTS) * 8 + 20 + 8, r.MAX_FITS)

    def test_subset_retains_original_order_without_mutating_contract(self):
        before = deepcopy(r.features.spec(('A','B','C','D')))
        names = list(r.features.ALL_FEATURES)[:3]
        self.assertEqual(r.subset(names), names)
        self.assertEqual(r.features.spec(('A','B','C','D')), before)
        for bad in ([], list(reversed(names)), [names[0]]*2, ['winner_label'], ['final_odds']):
            with self.assertRaises(MLFailure): r.subset(bad)

    def test_min_gain_uses_current_model_and_stable_original_order(self):
        names = list(r.features.ALL_FEATURES)[:3]
        self.assertEqual(r.next_remove(names, dict(zip(names,[2.,1.,1.]))), names[1])
        self.assertEqual(r.next_remove(names, dict(zip(names,[.1,3.,1.]))), names[0])

    def test_bad_gain_values_or_missing_features_do_not_delete_arbitrarily(self):
        names = list(r.features.ALL_FEATURES)[:2]
        for gains in ({names[0]:0}, dict(zip(names,[np.nan,1.])), dict(zip(names,[-1.,1.]))):
            with self.assertRaises(MLFailure): r.next_remove(names,gains)

    def test_feature_count_selection_uses_2024_only_and_prefers_near_tie_simplicity(self):
        records=[{'name':'A39','feature_count':39,'selection_metrics':{'race_log_loss':2.0},'evaluation_LL':1.},
                 {'name':'B32','feature_count':32,'selection_metrics':{'race_log_loss':2.001},'evaluation_LL':9.},
                 {'name':'B24','feature_count':24,'selection_metrics':{'race_log_loss':2.004},'evaluation_LL':0.}]
        self.assertEqual(r.choose(records)['name'],'B32')
        records[2]['evaluation_LL']=99.
        self.assertEqual(r.choose(records)['name'],'B32')

    def test_validation_2026_cannot_be_scored(self):
        f=pd.DataFrame({'date':['2026-01-01'],'entry_key':['not_used'],'winner_label':['not_parsed']})
        with self.assertRaisesRegex(MLFailure,'invalid_split'):r.point(f,[.5])

    def test_metric_definition_keeps_tie_credit_and_probabilities(self):
        f=pd.DataFrame({'date':['2024-11-09']*3,'entry_key':['202411090501010101','202411090501010102','202411090501010103'],
                        'winner_label':[1,0,0]})
        before=f.copy(deep=True);p=np.asarray([.4,.4,.2]);out=r.point(f,p)
        self.assertAlmostEqual(out['race_log_loss'],-np.log(.4))
        self.assertEqual(out['top1_accuracy_percent'],50.)
        pd.testing.assert_frame_equal(before,f);np.testing.assert_array_equal(p,[.4,.4,.2])

    def test_global_actual_attempt_cap_stops_before_writing_next_fit(self):
        with patch.object(r.CPU,'sample',return_value=(1,2,3)), patch.object(r.CPU,'percent',return_value=1):
            budget=r.Budget(Path('synthetic'),time.monotonic()+10);budget.calls=100
            with patch.object(r,'write') as write, self.assertRaises(MLFailure):budget.reserve('LGBM','mock')
            write.assert_not_called()

    def test_RF_cap_and_failed_attempt_consume_budget(self):
        with patch.object(r.CPU,'sample',return_value=(1,2,3)),patch.object(r.CPU,'percent',return_value=1),patch.object(r,'write'):
            budget=r.Budget(Path('synthetic'),time.monotonic()+10)
            number=budget.reserve('RF','Boruta');budget.finished(number,'RF','failed',.01)
            self.assertEqual(budget.calls,1);self.assertEqual(budget.kind_counts['RF'],1)
            budget.kind_counts['RF']=20
            with self.assertRaises(MLFailure):budget.reserve('RF','Boruta')

    def test_expired_budget_does_not_attempt_fit(self):
        with patch.object(r.CPU,'sample',return_value=(1,2,3)):
            budget=r.Budget(Path('synthetic'),time.monotonic()-1)
            with patch.object(r,'write') as write,self.assertRaisesRegex(MLFailure,'query_timeout'):budget.reserve('LGBM','mock')
            write.assert_not_called()


if __name__=='__main__':unittest.main()
