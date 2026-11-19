from datetime import date
from copy import deepcopy
import unittest
from unittest.mock import MagicMock,patch
from pathlib import Path
import tempfile
from datetime import datetime

from jra_racing_ai.ml import data, features, safety, v03
from jra_racing_ai.ml.encodings import jv_time_seconds
from jra_racing_ai import post_step2
from jra_racing_ai.win_sim import core
from test_ml_v03 import race_rows,catalog

class PostFeatureTests(unittest.TestCase):
    def mapping(self):
        m=v03.feature_mapping(catalog()); m['reviewed']=True
        m.update(format_version=2,formats={'finish_time_seconds':'jv_mss_tenths_4'})
        m['columns']['finish_time_seconds']='soha_time'
        return m

    def test_explicit_minute_second_decoder(self):
        self.assertEqual(jv_time_seconds('1345'),94.5)
        self.assertEqual(jv_time_seconds('0599'),59.9)
        self.assertEqual(jv_time_seconds('1000'),60.)
        self.assertEqual(jv_time_seconds('3599'),239.9)

    def test_invalid_time_is_missing_not_guessed_or_zero(self):
        for value in (None,'','    ','0000','1600','1999','134','01345','１３４５',1345,'94.5'):
            self.assertIsNone(jv_time_seconds(value))

    def test_time_encoding_cannot_be_combined_with_numeric_scale(self):
        m=self.mapping(); self.assertEqual(data.validate_mapping(m),m)
        m['scales']['finish_time_seconds']=.1
        with self.assertRaises(safety.MLFailure):data.validate_mapping(m)

    def test_history_scope_uses_bound_parameters(self):
        sql,params=data.export_query(self.mapping(),start=date(2024,1,1),end=date(2024,2,1),horse_ids={'SYNTHETIC'})
        self.assertNotIn('SYNTHETIC',sql)
        self.assertIn('= ANY(%s)',sql)
        self.assertEqual(params[-2],['SYNTHETIC'])

    def test_conditioned_time_uses_same_distance_track_and_seven_day_lag(self):
        target=race_rows(date(2025,2,15))
        past=race_rows(date(2025,2,1))
        other=[dict(r,distance='1200',finish_time_seconds='70') for r in race_rows(date(2025,2,8))]
        too_recent=[dict(r,finish_time_seconds='1') for r in race_rows(date(2025,2,9))]
        for r in past:r['finish_time_seconds']='94.5'
        actual=features.build_features(target,history=past+other+too_recent,feature_profile='conditioned_previous_time_v1')
        self.assertTrue(all(r['previous_time_seconds']==94.5 for r in actual))
        legacy=features.build_features(target,history=past+other+too_recent)
        self.assertTrue(all(r['previous_time_seconds']==70 for r in legacy))

    def test_current_result_and_time_do_not_change_current_features(self):
        rows=race_rows(date(2025,2,15))
        changed=[dict(r,finish_position='28',finish_time_seconds='1') for r in rows]
        a=features.build_features(rows,feature_profile='conditioned_previous_time_v1')
        self.assertEqual(a,features.build_features(changed,feature_profile='conditioned_previous_time_v1'))

    def test_unknown_profile_does_not_silently_use_legacy_definition(self):
        with self.assertRaises(safety.MLFailure):
            features.build_features(race_rows(date(2025,2,15)),feature_profile='unknown')

    def test_inference_loads_saved_profile_and_returns_uncalibrated_scores(self):
        past=[dict(r,finish_time_seconds='94.5') for r in race_rows(date(2025,2,1))]
        bundle={'metadata':{'history_lag_days':7,'categories':{},'feature_profile':'conditioned_previous_time_v1'},
            'boosters':{'win':MagicMock()}}
        bundle['boosters']['win'].predict.return_value=[.1]*8
        with patch.object(v03.model,'frame',side_effect=lambda values,schema:values) as frame:
            predictions,_=v03.predict_win(bundle,race_rows(date(2025,2,15)),past)
        self.assertTrue(all(r['previous_time_seconds']==94.5 for r in frame.call_args.args[0]))
        self.assertTrue(all(p==.1 for p in predictions[0].probabilities.values()))

class ResultCohortTests(unittest.TestCase):
    def setUp(self):
        rows=race_rows(date(2026,10,18)); self.race=features.entry_identity(rows[0])[0][:-1]
        self.predictions=[core.PredictionRace(self.race,{r['horse_number']:1/8 for r in rows})]
        self.identities={(*self.race,r['horse_number']):r['horse_id'] for r in rows}
        self.se=[dict(race_key=self.race,status='6',horse_number=r['horse_number'],horse_id=r['horse_id'],
            rank=r['finish_position'],anomaly='0') for r in rows]
        self.outcomes={self.race:core.Outcome(self.race,'normal_final',{'01':200},True,True)}

    def evaluate(self):
        return post_step2.result_cohort(self.se,self.outcomes,self.predictions,self.identities)

    def test_full_roster_and_official_winner_match(self):
        winners,ranks,valid,reasons=self.evaluate()
        self.assertEqual(winners[self.race],'01');self.assertEqual(len(ranks[self.race]),8)
        self.assertFalse(reasons)

    def test_missing_or_wrong_horse_is_not_a_negative_label(self):
        self.se.pop()
        self.assertFalse(self.evaluate()[0])
        self.assertEqual(self.evaluate()[3],{'result_roster_missing':1})

    def test_cancelled_runner_excludes_whole_race_without_renormalizing(self):
        self.se[-1]['anomaly']='1'
        self.assertFalse(self.evaluate()[0])

    def test_did_not_finish_is_started_nonwinner(self):
        self.se[-1].update(anomaly='4',rank='00')
        self.assertEqual(self.evaluate()[0][self.race],'01')
        self.assertEqual(self.evaluate()[1][self.race]['08'],0)

    def test_pending_result_and_special_payout_remain_unsettled(self):
        self.outcomes[self.race]=core.Outcome(self.race,'pending',reason='special_pending')
        self.assertFalse(self.evaluate()[0])

    def test_hr_se_winner_disagreement_is_not_scored(self):
        self.outcomes[self.race]=core.Outcome(self.race,'normal_final',{'02':200},True,True)
        self.assertEqual(self.evaluate()[3],{'multiple_winners_or_hr_se_mismatch':1})

    def test_same_model_race_decision_is_saved_once_and_not_reselected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);work=base/'work';work.mkdir()
            clock=datetime(2026,10,18,12,tzinfo=core.JST)
            document={'data_read_completed_at_jst':clock.replace(hour=9).isoformat(),
                'first_prediction_at_jst':clock.replace(hour=9).isoformat()}
            with patch.object(post_step2,'ROOT',base),patch.object(post_step2,'now',return_value=clock), \
                    patch.object(post_step2,'frozen_predictions',return_value=(document,self.predictions,{self.race:clock})), \
                    patch.object(post_step2,'make_observed_snapshots',return_value={self.race:[]}):
                first=post_step2.save_due_decisions(work)
                second=post_step2.save_due_decisions(work)
            self.assertEqual(first['new_decisions'],1)
            self.assertEqual(second['new_decisions'],0)
            self.assertEqual(second['existing_decisions_preserved'],1)
            self.assertEqual(len(list((work/'decisions').glob('*.json'))),1)
            for path in base.rglob('*'):
                if path.is_file():path.chmod(0o666)
