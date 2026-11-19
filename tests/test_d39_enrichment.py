"""Synthetic inputs only: causal time connection, verified going, and finite learning."""
from copy import deepcopy
from datetime import date,timedelta,datetime
import unittest
from unittest.mock import patch
from jra_racing_ai.d39_enrichment import contracts as c,features as f,weather as w,sources,learning
from jra_racing_ai.d39_next import features as old
from jra_racing_ai.ml import safety
from jra_racing_ai.ml.encodings import jv_time_seconds
from jra_racing_ai.win_sim.core import JST
from test_research_stage2 import source
from pathlib import Path
import json
import numpy as np
from types import SimpleNamespace
import test_d39_next_forward as prior_tests
from test_keiba_lab_live import fixture
from jra_racing_ai.d39_next import forward as old_forward
from jra_racing_ai.d39_enrichment import forward as new_forward
from jra_racing_ai import keiba_lab_live as lab


def time_index(rows,seconds=93.4):
    return {''.join(old.legacy.legacy.entry_identity(r)[0]):{'horse_id':r['horse_id'],
        'finish_position':r['finish_position'],'time_seconds':seconds} for r in rows}


def build(history,target,index,resolver=None):
    b=f.Builder(index,resolver)
    for rows in history:b.transform_day(rows)
    return b.transform_day(target,prediction=True)[0]


class TimeTests(unittest.TestCase):
    def setUp(self):self.day=date(2024,1,1);self.rows=source(self.day)

    def test_official_existing_decoder_seconds_not_decimal_magnitude(self):
        self.assertEqual(jv_time_seconds('1345'),94.5)
        for invalid in ('0000','1600','１２３４','123','1.23',None):self.assertIsNone(jv_time_seconds(invalid))

    def test_time_becomes_available_only_after_seven_days(self):
        for lag in (6,7):
            out=build([self.rows],source(self.day+timedelta(days=lag)),time_index(self.rows))
            self.assertEqual(out[0]['previous_time_seconds'],None if lag==6 else 93.4)

    def test_current_race_time_and_final_RA_going_are_not_inputs(self):
        target=source(self.day+timedelta(days=8));index=time_index(self.rows)
        before=build([self.rows],target,index)
        for r in target:r.update(finish_time_seconds='5000',soha_time='5000',going='4',finish_position='00',final_odds='9999')
        self.assertEqual(before,build([self.rows],target,{**index,**time_index(target,200.)}))
        self.assertIsNone(before[0]['going'])

    def test_whole_abnormal_race_never_supplies_times(self):
        self.rows[0]['anomaly']='4'
        result=build([self.rows],source(self.day+timedelta(days=8)),time_index(self.rows))
        self.assertTrue(all(r['previous_time_seconds'] is None for r in result))

    def test_identity_and_finish_disagreement_with_archived_source_hold_missing(self):
        index=time_index(self.rows);first=next(iter(index));index[first]['horse_id']='unrelated'
        result=build([self.rows],source(self.day+timedelta(days=8)),index)
        self.assertIsNone(result[0]['previous_time_seconds'])
        index=time_index(self.rows);index[first]['finish_position']='28'
        result=build([self.rows],source(self.day+timedelta(days=8)),index)
        self.assertIsNone(result[0]['previous_time_seconds'])

    def test_distance_or_exact_track_does_not_mix_times(self):
        for field,value in [('distance','2500'),('surface','22')]:
            target=source(self.day+timedelta(days=8))
            for r in target:r[field]=value
            self.assertTrue(all(r['previous_time_seconds'] is None for r in build([self.rows],target,time_index(self.rows))))

    def test_later_invalid_time_keeps_latest_valid_conditioned_time(self):
        middle=source(self.day+timedelta(days=8));index={**time_index(self.rows),**time_index(middle,None)}
        self.assertEqual(build([self.rows,middle],source(self.day+timedelta(days=16)),index)[0]['previous_time_seconds'],93.4)

    def test_future_addition_cannot_change_earlier_prediction(self):
        target=source(self.day+timedelta(days=8));future=source(self.day+timedelta(days=100))
        index=time_index(self.rows)
        self.assertEqual(build([self.rows],target,index),build([self.rows],target,{**index,**time_index(future,300.)}))

    def test_other_45_features_and_missing_cohort_stay_unchanged(self):
        target=source(self.day+timedelta(days=8));b=old.Builder();b.transform_day(self.rows)
        expected,meta=b.transform_day(target,prediction=True)
        actual=build([self.rows],target,time_index(self.rows))
        for a,e in zip(actual,expected):
            self.assertEqual({k:v for k,v in a.items() if k not in ('going','previous_time_seconds')},
                             {k:v for k,v in e.items() if k not in ('going','previous_time_seconds')})
        self.assertEqual(len(actual),len(expected))

    def test_duplicate_conflict_is_missing_and_leading_zero_finish_matches(self):
        index={};a={'horse_id':'a','finish_position':'01','time_seconds':90.}
        sources.put(index,'key',a);sources.put(index,'key',dict(a,finish_position='1'))
        self.assertEqual(index['key']['time_seconds'],90.)
        sources.put(index,'key',dict(a,time_seconds=91.))
        self.assertIsNone(index['key']['time_seconds'])


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(source(date(2024,1,1))[0],surface='11',start_time='1000',previous_start='0000')
        self.event={k:self.row[k] for k in w.SCOPE}
        self.event.update(data_kubun='1',happyo_tsukihi_jifun='01010900',henko_shikibetsu='1',
            babajotai_code_shiba='1',babajotai_code_dirt='2',babajotai_code_shiba_henkomae='0',babajotai_code_dirt_henkomae='0')

    def test_initial_state_and_unbroken_change(self):
        changed=dict(self.event,henko_shikibetsu='3',happyo_tsukihi_jifun='01010930',babajotai_code_shiba='3',babajotai_code_shiba_henkomae='1')
        self.assertEqual(w.Resolver([self.event,changed]).resolve(self.row),'3')

    def test_future_and_equal_minute_publications_excluded_conservatively(self):
        change=dict(self.event,henko_shikibetsu='3',happyo_tsukihi_jifun='01010950',babajotai_code_shiba='4',babajotai_code_shiba_henkomae='1')
        self.assertEqual(w.Resolver([self.event,change]).resolve(self.row),'1')
        change['happyo_tsukihi_jifun']='01010949'
        self.assertEqual(w.Resolver([self.event,change]).resolve(self.row),'4')

    def test_change_without_initial_state_or_broken_chain_is_missing(self):
        changed=dict(self.event,henko_shikibetsu='3',happyo_tsukihi_jifun='01010930',babajotai_code_shiba_henkomae='2')
        self.assertIsNone(w.Resolver([changed]).resolve(self.row))
        self.assertIsNone(w.Resolver([self.event,changed]).resolve(self.row))

    def test_dirt_uses_dirt_state_not_turf(self):
        self.assertEqual(w.Resolver([self.event]).resolve(dict(self.row,surface='24')),'2')
        for track in ('27','28','52','99'):self.assertIsNone(w.Resolver([self.event]).resolve(dict(self.row,surface=track)))

    def test_weather_only_update_does_not_replace_going_with_zero(self):
        event=dict(self.event,henko_shikibetsu='2',happyo_tsukihi_jifun='01010920',babajotai_code_shiba='0')
        self.assertEqual(w.Resolver([self.event,event]).resolve(self.row),'1')

    def test_schedule_revision_or_unknown_flag_is_missing(self):
        self.assertIsNone(w.Resolver([self.event]).resolve(dict(self.row,previous_start='0950')))
        for field,value in [('data_kubun','0'),('henko_shikibetsu','9'),('babajotai_code_shiba','0')]:
            self.assertIsNone(w.Resolver([dict(self.event,**{field:value})]).resolve(self.row))

    def test_complete_meeting_key_and_conflicting_duplicate(self):
        self.assertIsNone(w.Resolver([self.event]).resolve(dict(self.row,kaisai_kai='99')))
        self.assertIsNone(w.Resolver([self.event,dict(self.event,babajotai_code_shiba='4')]).resolve(self.row))

    def test_unparseable_change_time_never_silently_keeps_old_state(self):
        self.assertIsNone(w.Resolver([self.event,dict(self.event,happyo_tsukihi_jifun='unknown')]).resolve(self.row))


class BudgetTests(unittest.TestCase):
    def test_old_spec_unchanged_new_spec_explicit(self):
        self.assertEqual(c.prior.spec()['format_version'],7)
        self.assertEqual(c.spec()['format_version'],8)
        self.assertEqual(c.spec()['features'],c.prior.spec()['features'])

    def test_failed_fit_consumes_single_attempt(self):
        counter=learning.budget.FitCounter(1,float('inf'),lambda *args:None)
        attempt=counter.wrap(lambda **kwargs:(_ for _ in ()).throw(ValueError()))
        with self.assertRaises(ValueError):attempt()
        with self.assertRaisesRegex(safety.MLFailure,'input_limit'):attempt()
        self.assertEqual(counter.calls,1)

    def test_exact_one_fit_approval_required(self):
        with patch.object(c,'read',side_effect=[{'version':c.VERSION},safety.MLFailure('io_failed')]):
            with self.assertRaisesRegex(safety.MLFailure,'explicit_execution_required'):c.approved(c.ROOT/'fake')


class EnrichedForwardTests(unittest.TestCase):
    setUp=prior_tests.NewForwardTests.setUp

    def test_new_store_model_name_and_temperature_use_original_cutoff_contract(self):
        folder,_=fixture(self.parent)
        reg=json.loads((folder/'registered.json').read_text());reg['registered_at']='2026-10-24T08:00:00+09:00'
        (folder/'registered.json').write_text(json.dumps(reg))
        original=json.loads((folder/'prediction.json').read_text())
        entry=Path(original['entry_snapshot']['path']);target=source(date(2026,10,24))
        entry.write_text(json.dumps({'payload':target}))
        original['entry_snapshot']['sha256']=old_forward.c.sha(entry)
        original['quote_T10']['odds']={'01':2.,'02':5.,'03':10.}
        (folder/'prediction.json').write_text(json.dumps(original))
        receipt=json.loads((folder/'receipt.json').read_text());receipt['prediction_sha256']=old_forward.c.sha(folder/'prediction.json')
        (folder/'receipt.json').write_text(json.dumps(receipt))
        before={p.name:old_forward.c.sha(p) for p in folder.iterdir() if p.is_file()}
        builder=f.Builder({},None);builder.transform_day(source(date(2026,1,1)));builder._available(date(2026,10,24))
        store=SimpleNamespace(VERSION=c.VERSION,write=old_forward.c.write)
        output=self.root/'new';seen=[]
        def predictor(bundle,frame,*,raw):
            self.assertTrue(raw);seen.append(frame.copy());return np.array([.6,.3,.1])
        result=old_forward.decide(folder,output,{'activated_at':'2026-10-24T08:00:00+09:00'},
            (None,{'model_id':'improved-fake','temperature':1.25}),builder,
            clock=lambda:datetime(2026,10,24,9,35,40,tzinfo=JST),store=store,predictor=predictor,model_name=new_forward.NAME)
        record=json.loads((output/'prediction.json').read_text())
        self.assertEqual(result['status'],'saved');self.assertEqual(record['version'],c.VERSION)
        self.assertEqual({r['model'] for r in record['rows']},{new_forward.NAME})
        expected=np.array([.6,.3,.1])**.8;expected/=expected.sum()
        np.testing.assert_allclose([r['probability'] for r in record['rows']],expected,rtol=0,atol=1e-15)
        self.assertEqual(len(seen),1);self.assertNotIn('winner_label',seen[0])
        self.assertFalse(record['result_accessed'])
        self.assertEqual(before,{p.name:old_forward.c.sha(p) for p in folder.iterdir() if p.is_file()})

    def test_clone_does_not_mutate_enriched_history(self):
        past=source(date(2024,1,1));target=source(date(2024,1,9))
        builder=f.Builder(time_index(past),None);builder.transform_day(past)
        before_pending=list(builder.time_pending);before_times=dict(builder.past_times)
        out=old_forward.live_frame(builder,target)
        self.assertEqual(list(builder.time_pending),before_pending);self.assertEqual(builder.past_times,before_times)
        self.assertEqual(out.previous_time_seconds.tolist(),[93.4]*3)

    def test_optional_enriched_result_projection_still_forbids_H(self):
        with self.assertRaisesRegex(safety.MLFailure,'invalid_input'):
            lab.result_projection({},[],None,names=(*new_forward.MODELS,'H'))
        with self.assertRaisesRegex(safety.MLFailure,'invalid_input'):
            lab.fetch_results([],None,names=('H',new_forward.NAME))

    def test_result_request_has_no_H_and_only_saved_candidates(self):
        folder,plan=fixture(self.parent);clock=datetime(2026,10,24,10,0,tzinfo=JST)
        base=lab.load_forecast(folder,plan,clock)
        for name in ('next_D39',new_forward.NAME):
            base['models'][name]=deepcopy(base['models']['new_D39'])
            base['d39_tickets'][name]={'horse':'01','reason':'ev_selected'}
        def answers(requests,session,*,names):
            self.assertNotIn('H',names);self.assertNotIn('H',requests[0]['models'])
            self.assertNotIn('H',requests[0]['d39_tickets']);return {},[]
        with patch.object(new_forward,'saved',return_value=[base]),patch.object(c,'clock',return_value=clock), \
             patch.object(c,'write',side_effect=old_forward.c.write),patch.object(lab,'fetch_results',side_effect=answers):
            new_forward.evaluate(self.root)
        self.assertFalse(json.loads((self.root/'latest_results.json').read_text())['H_results_included'])

    def test_going_capture_selects_no_results_and_saves_only_before_T10(self):
        from contextlib import nullcontext
        from unittest.mock import MagicMock
        fields=(*sources.RACE_KEY,'data_kubun','track_code','hasso_jikoku','babajotai_code_shiba','babajotai_code_dirt')
        schema={'jvd_ra':[{'name':k,'type_is_builtin':True} for k in fields]}
        base={k:source(date(2026,10,25))[0][k] for k in sources.RACE_KEY}
        rows=[dict(base,data_kubun='2',track_code='11',hasso_jikoku='1210',babajotai_code_shiba='1',babajotai_code_dirt='2'),
              dict(base,race_bango='02',data_kubun='2',track_code='11',hasso_jikoku='1200',babajotai_code_shiba='1',babajotai_code_dirt='2')]
        db=MagicMock();db.aggregate.return_value=rows
        with patch.object(c,'read',return_value=schema),patch.object(c,'write',side_effect=old_forward.c.write), \
             patch.object(c,'clock',return_value=datetime(2026,10,25,11,55,tzinfo=JST)),patch.object(sources,'session',return_value=nullcontext(db)):
            result=sources.capture_going(self.root)
        self.assertEqual(result['pre_cutoff_races'],1);self.assertTrue(result['new_snapshot'])
        sql=db.aggregate.call_args.args[1]
        for banned in ('kakutei_chakujun','soha_time','odds','haraimodoshi'):self.assertNotIn(banned,sql)
        value=json.loads(next((self.root/'going-observed'/'2026-10-25').glob('*.json')).read_text())
        self.assertFalse(value['results_read']);self.assertFalse(value['used_by_current_candidate'])
        self.assertTrue(value['all_observed_before_T10'])


if __name__=='__main__':unittest.main()
