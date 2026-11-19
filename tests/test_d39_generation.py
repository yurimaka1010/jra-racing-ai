"""Synthetic current-D39 compatibility, causality and finite-fit checks."""
from copy import deepcopy
from datetime import date,datetime
from pathlib import Path
import inspect
import json
import hashlib
from contextlib import ExitStack
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from jra_racing_ai.d39_generation import contracts as c,data,learning
from jra_racing_ai.d39_generation import forward,outcomes
from jra_racing_ai.research import features as rf,contracts as old,model as matrices
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim import core
from test_research_stage2 import source


class GenerationTests(unittest.TestCase):
    def plan(self):
        return {'version':c.VERSION,'2026_training_authorized':True,'latest_final_date':'2026-10-18'}

    def test_sealed_old_entry_point_stays_closed(self):
        with self.assertRaises(MLFailure):old.guard_development(date(2026,1,1),date(2026,9,30))
        c.guard_day(date(2026,9,30),self.plan())

    def test_new_permission_is_versioned_and_date_bounded(self):
        for plan in (dict(self.plan(),version='old'),dict(self.plan(),**{'2026_training_authorized':False})):
            with self.assertRaises(MLFailure):c.guard_day(date(2026,1,1),plan)
        with self.assertRaises(MLFailure):c.guard_day(date(2026,10,19),self.plan())

    def test_latest_split_has_independent_recent_calibration_and_gaps(self):
        spans=c.split(date(2026,10,18))
        self.assertEqual(spans['train']['end'],'2026-08-16')
        self.assertEqual(spans['early_stopping'],{'start':'2026-08-24','end':'2026-09-20'})
        self.assertEqual(spans['calibration'],{'start':'2026-09-28','end':'2026-10-18'})
        self.assertEqual(spans['final_train']['end'],'2026-09-20')

    def test_month_chunks_keep_existing_row_limit(self):
        chunks=list(data.months(date(2025,12,1),date(2026,10,18)))
        self.assertEqual(len(chunks),11)
        cols={alias:{value[1]:{} for value in old.RAW_FIELDS.values() if value[0]==alias} for alias in ('ra','se')}
        sql,args,_=data.source_query(cols,*chunks[-1],self.plan())
        self.assertEqual(args[-1],20001);self.assertIn("='7'",sql)
        self.assertNotIn('UPDATE',sql)

    def test_live_query_excludes_all_current_results_and_final_odds(self):
        cols={alias:{value[1]:{} for value in old.RAW_FIELDS.values() if value[0]==alias} for alias in ('ra','se')}
        with patch.object(c,'clock',return_value=datetime(2026,10,24,8,tzinfo=core.JST)):
            sql,args,fields=data.source_query(cols,date(2026,10,24),date(2026,10,24),self.plan(),live=True)
        for field in ('finish_position','late3f_raw','corner4','actual_runners','margin1'):
            self.assertNotIn(field,fields)
        self.assertNotIn('odds',sql);self.assertNotIn('kakutei_chakujun',sql)

    def test_history_replay_matches_generator_for_multiple_races(self):
        past=source(date(2026,10,1));second=deepcopy(past)
        for row in second:
            row['race_bango']='02';row['horse_id']='OTHER'+row['horse_number'];row['late3f_raw']='440'
        full=rf.Builder();fast=rf.Builder();full.transform_day(past+second);data.replay(fast,past+second)
        target=source(date(2026,10,10))
        self.assertEqual(full.transform_day(target),fast.transform_day(target))
        self.assertEqual(fast.global_counts,[6,2,0])

    def test_39_features_formula_and_unknown_categories_unchanged(self):
        self.assertEqual(c.FEATURE_SPEC,old.spec(('A','B','C','D')))
        self.assertEqual(len(c.FEATURE_SPEC['features']),39)
        frame=pd.DataFrame(rf.build(source(date(2026,10,10))))
        categories=matrices.schema(frame,c.FEATURE_SPEC)
        frame.loc[0,'surface']='UNKNOWN'
        self.assertEqual(matrices.matrix(frame,c.FEATURE_SPEC,categories).loc[0,'surface'],'__UNKNOWN__')

    def test_tenth_fit_and_wrong_phase_rejected_without_training(self):
        with patch.object(learning.prior,'CPU'):
            budget=learning.Budget(Path('synthetic'),time.monotonic()+10);budget.calls=9
            with self.assertRaises(MLFailure):budget.reserve('final_refit')
            budget.calls=8
            with self.assertRaises(MLFailure):budget.reserve('tuner')

    def test_metric_definition_and_race_probability_sum(self):
        frame=pd.DataFrame({'entry_key':['202610240501010101','202610240501010102'],
                            'date':['2026-10-24']*2,'winner_label':[1,0]})
        point=learning.point(frame,[.75,.25])
        self.assertAlmostEqual(point['race_log_loss'],-np.log(.75));self.assertAlmostEqual(point['brier_score'],.0625)
        with self.assertRaises(MLFailure):learning.point(frame,[.75,.75])

    def test_temperature_is_applied_once_after_native_prediction(self):
        class Booster:
            def predict(self,*a,**k):return np.array([.2,.1])
        frame=pd.DataFrame({'entry_key':['202610240501010101','202610240501010102']})
        meta={'feature_spec':c.FEATURE_SPEC,'categories':{},'temperature':.75}
        with patch.object(matrices,'matrix',return_value=frame):
            actual=learning.predict((Booster(),meta),frame)
        expected=np.array([.2,.1])**(1/.75);expected/=expected.sum()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)

    def test_live_inference_is_same_39_formula_with_current_results_stripped(self):
        history=rf.Builder();data.replay(history,source(date(2026,10,10)));history._available(date(2026,10,24))
        entries=source(date(2026,10,24));changed=deepcopy(entries)
        for row in changed:row.update(finish_position='28',actual_runners='28',late3f_raw='100',corner4='01',margin1='A  ')
        a=forward.live_features(history,entries);b=forward.live_features(history,changed)
        pd.testing.assert_frame_equal(a,b)
        expected=pd.DataFrame(rf.build(entries,history=source(date(2026,10,10))))
        pd.testing.assert_frame_equal(a.loc[:,old.ALL_FEATURES],expected.loc[:,old.ALL_FEATURES])
        self.assertEqual(history.last_day,date(2026,10,10))

    def test_missing_horse_identity_stops_only_inference(self):
        rows=source(date(2026,10,24));rows[0]['horse_id']=''
        with self.assertRaises(MLFailure):forward.live_features(rf.Builder(),rows)

    def test_frozen_forecast_policy_uses_already_calibrated_p_and_existing_one_point_rule(self):
        race=('2026','1024','05','01','01','01')
        registered={'roster':['01','02'],'scheduled_start':'2026-10-24T10:00:00+09:00','registered_at':'2026-10-24T09:00:00+09:00'}
        quote={'published_at':'2026-10-24T09:49:00+09:00','first_seen_at':'2026-10-24T09:49:20+09:00','odds':{'01':4.,'02':1.5}}
        result=forward.paper_decision(race,{'01':.4,'02':.6},quote,registered,{'start_date':'2026-10-24','end_date':'2026-10-25'})
        self.assertEqual(result['ai_horse'],'01');self.assertEqual(result['ai']['stake_yen'],100)
        self.assertEqual(result['ai']['status'],'pending')
        self.assertEqual(result['runners'][0]['probability'],.4)

    def test_unchanged_entries_get_new_observation_but_quotes_keep_first_seen(self):
        stored={};root=Path('synthetic')
        with patch.object(forward,'mkdir',side_effect=lambda p:p),patch.object(Path,'exists',side_effect=lambda:False), \
             patch.object(c,'write',side_effect=lambda p,v:stored.setdefault(str(p),v)):
            for minute in (1,2):
                now=datetime(2026,10,24,9,minute,tzinfo=core.JST)
                forward.capture(root,'entries',[{'horse_number':'01'}],now)
                forward.capture(root,'odds',{'payload':'fake'},now)
        self.assertEqual(sum('entries' in p for p in stored),2)
        quotes=[v for p,v in stored.items() if 'odds' in p]
        self.assertEqual(len(quotes),1);self.assertEqual(quotes[0]['first_seen_at'],'2026-10-24T09:01:00+09:00')

    def test_future_record_loader_has_no_results_or_final_odds_query(self):
        text=inspect.getsource(forward)
        self.assertNotIn('kakutei_chakujun',text);self.assertNotIn('jvd_hr',text)
        self.assertNotIn('jvd_o1',text);self.assertNotIn('lgb.train',text)

    def test_evaluation_cannot_open_DB_on_race_day_or_calibration_dates(self):
        plan={'start_date':'2026-10-24','end_date':'2026-10-25'}
        session=__import__('unittest.mock',fromlist=['MagicMock']).MagicMock()
        with patch.object(c,'approved'),patch.object(c,'read',return_value=plan), \
             patch.object(c,'clock',return_value=datetime(2026,10,24,19,tzinfo=core.JST)):
            for day in ('2026-10-24','2026-10-18','2026-09-30'):
                with self.assertRaises(MLFailure):outcomes.evaluate(Path('synthetic'),session,day)
        session.assert_not_called()

    def test_result_join_requires_exact_saved_horse_identity_and_unique_winner(self):
        record={'registered':{'roster':['01','02']},'race_key':['2026','1024','05','01','01','01'],
                'entry_snapshot':{'path':'synthetic'}}
        saved={'payload':[{'horse_number':'01','horse_id':'FAKE1'},{'horse_number':'02','horse_id':'FAKE2'}]}
        rows=[{'horse':'01','horse_id':'FAKE1','status':'7','anomaly':'0','rank':'01'},
              {'horse':'02','horse_id':'FAKE2','status':'7','anomaly':'0','rank':'02'}]
        with patch.object(c,'read',return_value=saved):
            self.assertEqual(outcomes.result_labels(record,rows),({'01':1,'02':0},None))
            changed=deepcopy(rows);changed[0]['horse_id']='OTHER'
            self.assertIsNotNone(outcomes.result_labels(record,changed)[1])
            changed=deepcopy(rows);changed[1]['rank']='01'
            self.assertEqual(outcomes.result_labels(record,changed)[1],'multiple_or_unconfirmed_winners')
            changed=deepcopy(rows);changed[1]['rank']='AA'
            self.assertEqual(outcomes.result_labels(record,changed)[1],'result_rank_unconfirmed')

    def test_before_first_race_day_only_counts_cards_without_generating_forecasts(self):
        plan={'start_date':'2026-10-24','end_date':'2026-10-25'}
        from unittest.mock import MagicMock
        session=MagicMock()
        with patch.object(c,'read',return_value=plan),patch.object(c,'write') as write, \
             patch.object(c,'clock',return_value=datetime(2026,10,23,12,tzinfo=core.JST)), \
             patch.object(forward,'preflight',return_value={'status':'ready'}),patch.object(forward,'mkdir',side_effect=lambda p:p), \
             patch.object(forward.existing,'live_cards',return_value={}),patch.object(forward,'warm_history') as warm:
            result=forward.run(Path('synthetic'),session,once=True)
        self.assertEqual(result['card_readiness']['race_count'],0)
        self.assertEqual(result['card_readiness']['forecasts_generated'],0);warm.assert_not_called()

    def test_late_entries_cannot_replace_input_snapshot_before_cutoff(self):
        cutoff=datetime(2026,10,24,9,50,tzinfo=core.JST)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            folder=Path(temp);(folder/'entries').mkdir()
            a=folder/'entries/a.json';b=folder/'entries/b.json'
            a.write_text(json.dumps({'first_seen_at':'2026-10-24T09:49:45+09:00','payload':[{'horse_number':'01','age':'04'},{'horse_number':'02'}]}),encoding='utf-8')
            b.write_text(json.dumps({'first_seen_at':'2026-10-24T09:50:01+09:00','payload':[{'horse_number':'01','age':'99'},{'horse_number':'02'}]}),encoding='utf-8')
            stack.enter_context(patch.object(c,'read',side_effect=lambda p:json.loads(p.read_text(encoding='utf-8'))))
            stack.enter_context(patch.object(c,'sha',return_value='synthetic_hash'))
            selected,reason=forward.on_time_entries(folder,['01','02'],cutoff)
            self.assertIsNone(reason);self.assertEqual(selected['payload'][0]['age'],'04')

    def test_complete_synthetic_T10_prediction_freezes_all_three_before_answers(self):
        race=('2026','1024','05','01','01','01');now=datetime(2026,10,24,9,50,5,tzinfo=core.JST)
        reg={'race_key':list(race),'roster':['01','02'],'scheduled_start':'2026-10-24T10:00:00+09:00',
             'registered_at':'2026-10-24T09:00:00+09:00','reason':None}
        quote={'published_at':'2026-10-24T09:49:00+09:00','first_seen_at':'2026-10-24T09:49:20+09:00',
               'odds':{'01':4.,'02':1.5}}
        frame=pd.DataFrame({'entry_key':[''.join(race)+'01',''.join(race)+'02'],'date':['2026-10-24']*2,
                            'distance':[1600.,1600.],'surface':['11','11']})
        entries=[{'horse_number':'01','horse_id':'FAKE1'},{'horse_number':'02','horse_id':'FAKE2'}]
        plan={'model_ids':dict(zip(forward.NAMES,('fake_old','fake_new','fake_H'))),
              'temperature':dict(zip(forward.NAMES,(.75,1.,1.))),'references':{},'start_date':'2026-10-24','end_date':'2026-10-25'}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            folder=Path(temp)
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')))
            stack.enter_context(patch.object(c,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()))
            stack.enter_context(patch.object(forward.inputs,'select',return_value=(quote,None)))
            stack.enter_context(patch.object(forward,'on_time_entries',return_value=({'payload':entries,'path':'synthetic','sha256':'mock','first_seen_at':'2026-10-24T09:49:40+09:00'},None)))
            stack.enter_context(patch.object(forward,'live_features',return_value=frame))
            stack.enter_context(patch.object(forward.tuner,'predict',return_value=np.array([.4,.6])))
            stack.enter_context(patch.object(learning,'predict',return_value=np.array([.5,.5])))
            stack.enter_context(patch.object(forward.inputs,'predict',return_value=({'H':{'01':.6,'02':.4}},None)))
            result=forward.decide(folder,reg,reg,plan,{'old_D39':(None,None),'new_D39':None,'H':None},None,clock=lambda:now)
            self.assertEqual(result['status'],'saved')
            saved=json.loads((folder/'prediction.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['result_accessed']);self.assertEqual(set(saved['probabilities']),set(forward.NAMES))
            csv=pd.read_csv(folder/'predictions.csv');self.assertEqual(len(csv),6)
            self.assertEqual(set(csv.generated_at),{now.isoformat()})


if __name__=='__main__':unittest.main()
