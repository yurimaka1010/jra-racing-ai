"""Synthetic four-model watching; no formal-study files or real DB connections."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from io import BytesIO
from unittest.mock import Mock, patch
import json
import unittest

from jra_racing_ai import keiba_lab_live as live, keiba_lab_watch as w
from jra_racing_ai.win_sim import core


def forecast():
    return {'race_id':'2026103105040501','race_key':['2026','1031','05','04','05','01'],
        'date':'2026-10-31','venue':'架空会場','race_no':1,'scheduled_start':'2026-10-31T09:50:00+09:00',
        'roster':['01','02','03'],'horse_ids':{'01':'id-1','02':'id-2','03':'id-3'},
        'source_prediction_sha256':'a'*64,'valid_for_results':True,
        'models':{n:{'status':'saved','model_id':'synthetic-'+n,'predicted_at':'2026-10-31T09:40:01+09:00',
            'rows':[{'horse_no':h,'rank':i,'probability':p} for i,(h,p) in enumerate(zip(['01','02','03'],[.6,.3,.1]),1)]} for n in w.NAMES}}


def official(winner='02',category='7'):
    ranks={'01':'02','02':'01','03':'03'} if winner=='02' else {'01':'01','02':'02','03':'03'}
    return [{'horse':h,'horse_id':'id-'+str(int(h)),'rank':ranks[h],'status':category,'anomaly':'0'} for h in ranks]


def outcome(winner='02'):
    return core.Outcome(tuple(forecast()['race_key']),'normal_final',{winner:450},True,True)


class WatchProjectionTests(unittest.TestCase):
    def project(self,f=None,rows=None,payout=None):
        return w.project(f or forecast(),official() if rows is None else rows,outcome() if payout is None else payout,checked_at='2026-11-01T12:00:00+09:00')

    def test_four_models_choose_saved_rank_one_without_EV_or_odds(self):
        f=forecast();before=deepcopy(f);p=self.project(f)
        for model in p['models'].values():
            self.assertEqual(model['horse_no'],'01');self.assertEqual(model['probability'],.6)
            self.assertEqual(model['top_finish'],2);self.assertFalse(model['top_hit'])
            self.assertEqual(model['ticket']['stake_yen'],100);self.assertEqual(model['ticket']['payout_yen'],0)
        self.assertEqual(f,before)

    def test_win_uses_official_yen_amount(self):
        p=self.project(rows=official('01'),payout=outcome('01'))
        self.assertEqual(p['models']['H']['ticket']['payout_yen'],450)
        self.assertEqual(p['models']['H']['ticket']['net_profit_yen'],350)

    def test_veto_is_visible_but_never_scored(self):
        f=forecast();f.update(valid_for_results=False,reason='schedule_or_roster_changed_after_registration')
        p=self.project(f)
        self.assertEqual(p['result_status'],'confirmed')
        for m in p['models'].values():
            self.assertEqual(m['selection_state'],'held');self.assertIsNone(m['top_hit'])
            self.assertEqual(m['ticket']['stake_yen'],0)

    def test_missing_prediction_never_gains_an_outcome_pick(self):
        f=forecast();f['models']['H']={'status':'missing'}
        p=self.project(f);m=p['models']['H']
        self.assertEqual(m['selection_state'],'missing');self.assertIsNone(m['horse_no'])
        self.assertEqual(m['ticket']['stake_yen'],0)

    def test_result_pending_does_not_become_loss(self):
        p=self.project(rows=[])
        self.assertEqual(p['models']['H']['selection_state'],'pending')
        self.assertIsNone(p['models']['H']['ticket']['payout_yen'])

    def test_known_finish_without_payout_remains_unsettled(self):
        p=self.project(payout=core.Outcome(tuple(forecast()['race_key']),'pending',reason='result_unavailable'))
        self.assertEqual(p['result_status'],'confirmed')
        self.assertEqual(p['models']['H']['selection_state'],'loss')
        self.assertIsNone(p['models']['H']['ticket']['payout_yen'])

    def test_flash_without_confirmed_payout_stays_pending(self):
        p=self.project(rows=official(category='5'),payout=core.Outcome(tuple(forecast()['race_key']),'pending',reason='refund_pending'))
        self.assertEqual(p['result_status'],'provisional')
        self.assertIsNone(p['models']['H']['top_hit'])

    def test_confirmed_HR_settles_R1_without_fabricating_missing_full_order(self):
        rows=official(category='2')
        for row in rows:row['rank']='00'
        p=self.project(rows=rows)
        self.assertEqual(p['result_status'],'winner_confirmed')
        self.assertEqual(p['finishes'],{'02':1})
        self.assertIsNone(p['models']['H']['top_finish'])
        self.assertFalse(p['models']['H']['top_hit'])
        self.assertEqual(p['models']['H']['ticket']['payout_yen'],0)

    def test_official_refund_is_separate_from_loss_and_hit_denominator(self):
        rows=official();rows[0].update(anomaly='1',rank='00')
        p=self.project(rows=rows,payout=core.Outcome(tuple(forecast()['race_key']),'pending',reason='refund_pending'))
        m=p['models']['H'];self.assertEqual(m['selection_state'],'refunded')
        self.assertEqual(m['ticket']['payout_yen'],100);self.assertIsNone(m['top_hit'])
        s=w.scoreboard([{'watch':p}])['H']
        self.assertEqual(s['refunded'],1);self.assertEqual(s['investment_yen'],100)
        self.assertEqual(s['net_profit_yen'],0);self.assertIsNone(s['hit_rate'])

    def test_dead_heat_uses_both_official_payouts(self):
        rows=official();rows[0]['rank']='01'
        p=self.project(rows=rows,payout=core.Outcome(tuple(forecast()['race_key']),'normal_final',{'01':220,'02':330},True,True))
        self.assertTrue(p['models']['H']['top_hit']);self.assertEqual(p['models']['H']['ticket']['payout_yen'],220)
        self.assertEqual(len([x for x in p['podium'] if x['finish']==1]),2)

    def test_identity_duplicate_and_winner_mismatch_never_settle(self):
        for rows,payout in [(official()+official()[:1],outcome()),
                            ([dict(r,horse_id='wrong') for r in official()],outcome()),
                            (official(),outcome('01'))]:
            p=self.project(rows=rows,payout=payout)
            self.assertEqual(p['result_status'],'pending');self.assertIsNone(p['models']['H']['top_hit'])

    def test_retracted_result_does_not_keep_a_confirmed_hit(self):
        p=self.project(rows=official('01','3'),payout=outcome('01'))
        self.assertEqual(p['result_status'],'pending')

    def test_scoreboard_equals_details_and_excludes_missing_and_held(self):
        yes=self.project(rows=official('01'),payout=outcome('01'))
        pending=self.project(rows=[])
        held=forecast();held['valid_for_results']=False
        missing=forecast();missing['models']={}
        s=w.scoreboard([{'watch':p} for p in (yes,pending,self.project(held),self.project(missing))])['H']
        self.assertEqual((s['target_races'],s['prediction_saved'],s['held'],s['missing']),(4,3,1,1))
        self.assertEqual((s['hits'],s['result_confirmed_races'],s['hit_rate']),(1,1,1))
        self.assertEqual((s['settled_count'],s['unsettled_count'],s['investment_yen'],s['payout_yen']),(1,1,100,450))
        self.assertIsNone(s['whole_day_payout_yen'])

    def test_no_bet_has_no_ROI(self):
        f=forecast();f['models']={}
        self.assertIsNone(w.scoreboard([{'watch':self.project(f)}])['H']['return_rate'])


class WatchViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.parent=self.base/'parent';self.parent.mkdir()
        self.root=self.base/'viewer';self.root.mkdir()
        (self.parent/'four_addendum_plan.json').write_text('{}')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.clock=datetime.fromisoformat('2026-11-01T12:00:00+09:00');self.tick=0
        self.stack.enter_context(patch.object(live,'BASE',self.base));self.stack.enter_context(patch.object(live,'ROOT',self.root))
        self.stack.enter_context(patch.object(live,'now',side_effect=lambda:self.clock))
        self.stack.enter_context(patch.object(live,'read',side_effect=lambda p:json.loads(p.read_text())))
        self.stack.enter_context(patch.object(live.safety,'local_path',side_effect=lambda p,**kw:p))
        def write(p,v,**kw):
            with p.open('x',encoding='utf-8') as f:json.dump(v,f)
        self.stack.enter_context(patch.object(live.safety,'write_json',side_effect=write))
        self.f=forecast();self.v=live.Viewer(parent=self.parent,directory=self.root/'run',session=Mock(),clock=lambda:self.clock,monotonic=lambda:self.tick)
        def forecasts(day):
            self.v.results[day]=[deepcopy(self.f)]
            return {'date':day,'available_dates':['2026-10-31'],'coverage':{'registered_races':1},'version':live.VERSION}
        self.v.forecasts=Mock(side_effect=forecasts)
        self.obs={self.f['race_id']:(official(),outcome())}

    def test_watch_authorization_and_disclosure_are_explicit_and_separate(self):
        with patch.object(live,'fetch_watch_results',return_value=(self.obs,[])),patch.object(live,'fetch_results') as legacy:
            body=self.v.watch('2026-10-31')
        legacy.assert_not_called();self.assertTrue(body['H_results_included'])
        self.assertFalse(body['Stage7_study_access']);self.assertFalse(body['pedigree_21_week_study_access'])
        log=json.loads(next((self.root/'watch-disclosures').glob('*.json')).read_text())
        self.assertEqual(log['date'],'2026-10-31');self.assertIn('H',log['models'])
        self.assertFalse(body['disclosure']['is_unviewed']);self.assertEqual(self.f,forecast())

    def test_no_result_fetch_before_start(self):
        self.clock=datetime.fromisoformat('2026-10-31T09:45:00+09:00')
        with patch.object(live,'fetch_watch_results',return_value=({},[])) as fetch:self.v.watch('2026-10-31')
        self.assertEqual(fetch.call_args.args[0],[])

    def test_poll_and_correction_change_results_only(self):
        with patch.object(live,'fetch_watch_results',return_value=(self.obs,[])) as fetch:
            self.v.watch('2026-10-31');self.v.watch('2026-10-31');self.assertEqual(fetch.call_count,1)
        self.tick=16
        revised={self.f['race_id']:(official('01'),outcome('01'))}
        with patch.object(live,'fetch_watch_results',return_value=(revised,[])):body=self.v.watch('2026-10-31')
        self.assertEqual(body['scoreboard']['H']['hits'],1);self.assertEqual(self.f,forecast())
        self.assertEqual(len(list((self.v.directory/'watch-versions').glob('*.json'))),2)

    def test_DB_failure_keeps_explicitly_stale_previous_results(self):
        with patch.object(live,'fetch_watch_results',return_value=(self.obs,[])):self.v.watch('2026-10-31')
        self.tick=16
        with patch.object(live,'fetch_watch_results',side_effect=Exception('secret')):body=self.v.watch('2026-10-31')
        self.assertTrue(body['stale']);self.assertNotIn('secret',json.dumps(body))

    def test_unchanged_poll_does_not_duplicate_result_or_disclosure_versions(self):
        with patch.object(live,'fetch_watch_results',return_value=(self.obs,[])):
            self.v.watch('2026-10-31');self.tick=16;self.clock+=timedelta(seconds=16)
            self.v.watch('2026-10-31')
        self.assertEqual(len(list((self.v.directory/'watch-versions').glob('*.json'))),1)
        self.assertEqual(len(list((self.root/'watch-disclosures').glob('*.json'))),1)

    def test_holdout_and_unapproved_past_or_future_days_fail_before_fetch(self):
        with patch.object(live,'fetch_watch_results') as fetch:
            for day in ('2026-09-30','2026-10-25','2026-11-02','2027-03-15'):
                with self.assertRaises(live.safety.MLFailure):self.v.watch(day)
        fetch.assert_not_called()

    def test_watch_provider_rejects_sealed_keys_before_connection(self):
        session=Mock();f=forecast();f['race_key']=['2026','0930','05','04','05','01']
        with self.assertRaises(live.safety.MLFailure):live.fetch_watch_results([f],session)
        session.assert_not_called()

    def test_watch_endpoint_rejects_other_models_and_cross_origin_before_read(self):
        viewer=Mock();handler=live.make_handler(viewer,8767)
        h=object.__new__(handler);h.headers={'Host':'127.0.0.1:8767'}
        h.path='/api/watch?date=2026-10-31&model=pedigree45'
        h.send_response=Mock();h.send_header=Mock();h.end_headers=Mock();h.wfile=BytesIO()
        h.do_GET();h.send_response.assert_called_once_with(400);viewer.watch.assert_not_called()
        h.path='/api/watch?date=2026-10-31';h.headers={'Host':'127.0.0.1:8767','Origin':'https://other.example'}
        h.send_error=Mock();h.do_GET();h.send_error.assert_called_once_with(403);viewer.watch.assert_not_called()


if __name__=='__main__':unittest.main()
