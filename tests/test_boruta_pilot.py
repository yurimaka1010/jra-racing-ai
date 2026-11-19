"""Fabricated data and mock estimators only; no RF/LightGBM fit."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch

import numpy as np
import pandas as pd

from jra_racing_ai import boruta_pilot as b
from jra_racing_ai.ml import safety
from jra_racing_ai.research import contracts as features


def frame():
    rows=[]
    for i in range(4):
        row={name:('A' if i<2 else 'B') if name in features.CATEGORIES else float(i) for name in features.ALL_FEATURES}
        row.update(date=f'2024-01-{6+i:02d}',going=None,previous_time_seconds=None)
        rows.append(row)
    return pd.DataFrame(rows)


class BorutaPreparationTests(unittest.TestCase):
    def test_compatibility_and_proposal_do_not_fit(self):
        from sklearn.ensemble import RandomForestClassifier
        with patch.object(RandomForestClassifier,'fit',side_effect=AssertionError('no_fit')):
            value=b.compatibility()
        self.assertEqual(value['actual_forest_fits'],0);self.assertEqual(value['LightGBM_fits'],0)
        self.assertEqual(value['plan']['actual_forest_fit_max'],3)
        self.assertEqual(value['plan']['BorutaPy']['max_iter'],4)

    def test_preprocessor_drops_only_all_missing_without_rows(self):
        f=frame();p=b.make_preprocessor(f);x=b.transform(f,p)
        self.assertEqual(set(p['excluded_all_missing']),{'going','previous_time_seconds'})
        self.assertEqual(len(x),len(f));self.assertTrue(np.isfinite(x.to_numpy()).all())
        self.assertEqual(len(x.columns),len(set(x.columns)))

    def test_numeric_imputation_uses_training_median(self):
        f=frame();f['distance']=[1.,np.nan,3.,9.]
        p=b.make_preprocessor(f)
        self.assertEqual(p['numeric_training_medians']['distance'],3.)
        column=next(v['name'] for v in p['encoded_columns'] if v['source']=='distance')
        future=frame();future['date']='2025-01-05';future['distance']=np.nan
        self.assertEqual(b.transform(future,p)[column].tolist(),[3.]*4)

    def test_new_category_maps_to_saved_unknown_dictionary(self):
        f=frame();p=b.make_preprocessor(f);future=f.copy();future['venue']='UNSEEN';future['date']='2025-01-05'
        column=next(v['name'] for v in p['encoded_columns'] if v['source']=='venue' and v['category']=='__UNKNOWN__')
        self.assertEqual(b.transform(future,p)[column].tolist(),[1.]*4)
        self.assertNotIn('v:UNSEEN',p['categories']['venue'])

    def test_json_roundtrip_preserves_transform_and_order(self):
        f=frame();p=b.make_preprocessor(f);again=json.loads(json.dumps(p))
        pd.testing.assert_frame_equal(b.transform(f,p),b.transform(f,again))

    def test_future_and_evaluation_dates_cannot_select_preprocessing(self):
        for day in ('2024-09-08','2025-01-01','2026-01-01','2026-10-24'):
            f=frame();f['date']=day
            with self.subTest(day=day),self.assertRaises(safety.MLFailure):b.make_preprocessor(f)

    def test_transform_cannot_open_sealed_or_stage7_rows(self):
        p=b.make_preprocessor(frame())
        for day in ('2026-01-01','2026-10-24'):
            f=frame();f['date']=day
            with self.assertRaises(safety.MLFailure):b.transform(f,p)

    def test_current_results_and_final_odds_do_not_change_matrix(self):
        f=frame();p=b.make_preprocessor(f);before=b.transform(f,p)
        f['winner_label']=[1,0,0,0];f['finish_position']=[1,2,3,4];f['final_odds']=900.
        pd.testing.assert_frame_equal(before,b.transform(f,p))

    def test_infinite_or_invalid_numerical_values_stop_with_fixed_code(self):
        for v in (np.inf,'PRIVATE_RECORD'):
            f=frame();f['distance']=v
            with self.assertRaises(safety.MLFailure) as failure:b.make_preprocessor(f)
            self.assertNotIn('PRIVATE',str(failure.exception))

    def test_tentative_never_promoted_even_if_library_support_overlaps(self):
        p=b.make_preprocessor(frame());n=len(p['encoded_columns']);selector=MagicMock()
        selector.support_=np.ones(n,bool);selector.support_weak_=np.zeros(n,bool);selector.support_weak_[0]=True
        result=b.result_masks(selector,p)
        self.assertFalse(result['confirmed_mask'][0]);self.assertTrue(result['tentative_mask'][0])
        self.assertNotIn(result['encoded_columns'][0],result['confirmed_columns'])

    def test_empty_confirmation_stops_without_fallback(self):
        f=frame();p=b.make_preprocessor(f);n=len(p['encoded_columns']);selector=MagicMock()
        selector.support_=np.zeros(n,bool);selector.support_weak_=np.ones(n,bool)
        result=b.result_masks(selector,p)
        with self.assertRaises(safety.MLFailure):b.selected_matrix(f,p,result)
        self.assertEqual(len(b.selected_matrix(f,p,result,include_tentative=True).columns),n)

    def test_select_without_explicit_authorization_stops_before_preparation_or_fit(self):
        with patch.object(b,'make_preprocessor') as prep,patch.object(b.safety,'read_json') as read:
            with self.assertRaises(safety.MLFailure):b.select_training(frame(),[1,0,0,0],directory=Path('synthetic'),authorization=None)
        prep.assert_not_called();read.assert_not_called()

    def test_selection_requires_owned_research_lock(self):
        authorization={'source':'human_user','proposal_sha256':b.proposal_hash(),'allow_selection':True,'actual_forest_fit_max':3}
        with patch.object(b.safety,'local_path',return_value=Path('synthetic')), \
                patch.object(b.safety,'read_json',return_value={'pid':os.getpid()+1,'run_id':'synthetic','purpose':b.VERSION}), \
                patch.object(b,'make_preprocessor') as prep:
            with self.assertRaises(safety.MLFailure):b.select_training(frame(),[1,0,0,0],directory=Path('synthetic'),authorization=authorization)
        prep.assert_not_called()


if __name__=='__main__':unittest.main()
