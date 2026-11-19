"""Fictional two-date fixtures only; no real predictions, database or network."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import csv,hashlib,json,unittest
from unittest.mock import patch
from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.ml import safety
from jra_racing_ai.duel import replay as r,scoring as s,render
from jra_racing_ai.db_connection import validate_audit_sql


def fixture(number='01',day='2026-10-17',a=None,b=None):
    race=('2026',day[5:7]+day[8:10],'05','01','01',number);horses=('01','02','03')
    item={'race_key':list(race),'date':day,'venue':'架空競馬場','race_number':int(number),'comparison_id':r.SAVED_MODELS[day]['id'],'start':day+'T15:00:00+09:00',
        'roster':{h:{'horse_id':'FICTION_'+h,'snapshot_at':day+'T07:00:00+09:00'} for h in horses}}
    for name,p in (('saved',a or {'01':.2,'02':.5,'03':.3}),('H',b or {'01':.6,'02':.3,'03':.1})):
        item[name]={'model_id':r.SAVED_MODELS[day]['model_id'] if name=='saved' else 'FICTIONAL_H','status':'ready','reason':None,'probabilities':p,'predicted_at':day+'T07:00:00+09:00','decision_at':day+'T14:50:00+09:00'}
    rows=[{'horse_number':h,'horse_id':'FICTION_'+h,'name':'架空馬'+h,'rank':h,'anomaly':'0','status':'7'} for h in horses]
    outcome={'status':'normal_final','payouts_yen_per_100':{'01':350},'verified':True,'complete_winner_set':True,'reason':None}
    return item,rows,outcome


def view(**kwargs):return s.race_view(*fixture(**kwargs))


def demo_payload():
    races=[view(),view(number='02',a={'01':.1,'02':.2,'03':.7}),view(number='03',a={'01':.7,'02':.2,'03':.1})]
    item,rows,outcome=fixture(day='2026-10-18')
    races.append(s.race_view(item,rows,outcome))
    days={d:s.summarize([x for x in races if x['date']==d]) for d in r.DAYS}
    return {'races':races,'summary':{'coverage':{'target_races':4,'saved_forecast_races':4,'H_replayed_races':4,'common_races':4},'days':days},
        'saved_sources':[],'saved_models':r.SAVED_MODELS,'H_source':{'model_id':'FICTIONAL_H','split':{'train_end':'2024-08-31'}},
        'freeze_time':'2026-10-20T12:00:00+09:00','results_collected_at':'2026-10-20T12:01:00+09:00',
        'results_joined_at':'2026-10-20T12:02:00+09:00','forecast_sha256':'fictional_hash',
        'dates':list(r.DAYS),'venues':{'05':'架空競馬場'}}



class DuelScoringTests(unittest.TestCase):
    def test_second_and_third_places_are_separate(self):
        payload=demo_payload();x=payload['summary']['days']['2026-10-17']['common']['saved']
        self.assertEqual(x['rank1_count'],1);self.assertEqual(x['rank2_count'],1);self.assertEqual(x['rank3_count'],1)
        self.assertAlmostEqual(x['rank2_rate'],100/3);self.assertAlmostEqual(x['rank3_rate'],100/3);self.assertEqual(x['top3_rate'],100.)

    def test_different_models_have_separate_denominators(self):
        d=demo_payload()['summary'];self.assertEqual(d['coverage']['target_races'],4)
        self.assertEqual(d['days']['2026-10-17']['common_races'],3)
        self.assertEqual(d['days']['2026-10-18']['common_races'],1)
        self.assertNotIn('all',d)

    def test_missing_day_not_zero_accuracy(self):
        item,rows,outcome=fixture(day='2026-10-18');item['saved'].update(status='missing',probabilities={})
        d=s.summarize([s.race_view(item,rows,outcome)])
        self.assertIsNone(d['available']['saved']['rank1_rate']);self.assertEqual(d['common_races'],0)

    def test_cannot_pool_saved_models_across_days(self):
        with self.assertRaises(safety.MLFailure):s.summarize(demo_payload()['races'])

    def test_cannot_pool_model_ids_even_if_same_cohort_label(self):
        first=view();second=view(number='02');second['forecast_metadata']['saved']['model_id']='OTHER_FICTIONAL_MODEL'
        with self.assertRaises(safety.MLFailure):s.summarize([first,second])

    def test_saved_answers_must_match_committed_forecasts(self):
        with self.assertRaises(safety.MLFailure):s.build({'committed_sha256':'new'},{'forecast_sha256':'old'})

    def test_existing_within_auc_and_winner_ranking(self):
        x=view();self.assertEqual(x['models']['saved']['within_race_auc'],0.);self.assertEqual(x['models']['H']['within_race_auc'],1.)
        self.assertEqual(x['models']['saved']['winner_predicted_rank'],3)

    def test_disagreement_hits_and_finish_order(self):
        v=s.summarize([view()]);self.assertEqual(v['disagreement']['races'],1)
        self.assertEqual(v['disagreement']['saved_top1_hits'],0);self.assertEqual(v['disagreement']['H_top1_hits'],1)
        self.assertEqual(v['disagreement']['finish_comparison'],{'H_ahead':1})

    def test_agreement_has_identical_outcomes(self):
        v=s.summarize([view(a={'01':.6,'02':.3,'03':.1})]);self.assertEqual(v['agreement']['races'],1);self.assertEqual(v['agreement']['both_top1_hits'],1)

    def test_dead_heat_excludes_single_winner_auc_not_top_hit(self):
        item,rows,outcome=fixture();rows[1]['rank']='01';outcome['payouts_yen_per_100']['02']=200
        x=s.race_view(item,rows,outcome);self.assertTrue(x['dead_heat']);self.assertIsNone(x['models']['H']['within_race_auc'])
        self.assertEqual(s.summarize([x])['common']['saved']['rank1_rate'],100.)

    def test_cancelled_top_not_counted_as_loss_or_reselected(self):
        item,rows,outcome=fixture();rows[1].update(rank='00',anomaly='1');x=s.race_view(item,rows,outcome)
        self.assertEqual(x['models']['saved']['top'],'02');self.assertFalse(x['common']);self.assertIsNone(x['models']['H']['within_race_auc'])
        self.assertIsNone(s.summarize([x])['available']['saved']['rank1_rate'])

    def test_nonfinish_counts_as_started_failure_and_separate_category(self):
        item,rows,outcome=fixture();rows[1].update(rank='00',anomaly='4');x=s.race_view(item,rows,outcome)
        v=s.summarize([x]);self.assertEqual(v['common']['saved']['top_evaluable_races'],1)
        self.assertEqual(v['common']['saved']['rank1_rate'],0.);self.assertEqual(v['common']['saved']['distribution']['競走中止'],1)
        self.assertEqual(v['disagreement']['finish_comparison'],{'rank_unavailable':1})

    def test_duplicate_identity_and_unconfirmed_results_are_excluded(self):
        for mode in ('duplicate','identity','status'):
            item,rows,outcome=fixture()
            if mode=='duplicate':rows.append(deepcopy(rows[0]))
            if mode=='identity':rows[0]['horse_id']='WRONG_FICTIONAL_ID'
            if mode=='status':rows[0]['status']='2'
            v=s.race_view(item,rows,outcome);self.assertFalse(v['result_ok']);self.assertFalse(v['common'])
            self.assertEqual(v['models']['H']['paper']['status'],'pending')

    def test_mismatched_prediction_rosters_do_not_enter_common_comparison(self):
        item,rows,outcome=fixture();item['H']['probabilities']={'01':.6,'02':.4};x=s.race_view(item,rows,outcome)
        self.assertFalse(x['common']);self.assertEqual(x['common_reason'],'prediction_roster_mismatch')

    def test_flat_100_settlement_uses_official_payout_not_odds(self):
        summary=s.summarize([view()]);a,h=summary['paper']['saved'],summary['paper']['H']
        self.assertEqual(a['payout_yen'],0);self.assertEqual(a['net_profit_yen'],-100)
        self.assertEqual(h['payout_yen'],350);self.assertEqual(h['return_rate_percent'],350.)

    def test_unknown_HR_never_assumed_zero_and_finance_cohort_shared(self):
        item,rows,outcome=fixture();outcome['verified']=False;x=s.race_view(item,rows,outcome);v=s.summarize([x])
        for name in ('saved','H'):
            self.assertEqual(v['paper'][name]['settled_common_races'],0);self.assertEqual(v['paper'][name]['unsettled_common_races'],1)
            self.assertIsNone(v['paper'][name]['return_rate_percent'])

    def test_ties_select_lowest_horse_number_without_probability_change(self):
        item,rows,outcome=fixture(a={'01':.4,'02':.4,'03':.2});before=deepcopy(item);x=s.race_view(item,rows,outcome)
        self.assertEqual(x['models']['saved']['top'],'01');self.assertEqual(item,before)


class DuelReplayTests(unittest.TestCase):
    def test_cached_H_reuse_commits_before_opening_saved_results(self):
        item,_,_=fixture();item['saved']['probabilities']={'01':.2,'02':.5,'03':.3}
        parent=Path('parent');target=Path('target');race=tuple(item['race_key'])
        old={'races':[deepcopy(item)],'references':{},'H_source':{'model_id':'FICTIONAL_H'},'results_accessed':False}
        reads=[];writes={}
        def read(path):
            reads.append(path.name)
            if path.name=='forecasts.local.json':return old
            if path.name=='H_inputs.local.json':return {'records':[]}
            raise AssertionError('answer_key_must_remain_closed')
        with patch.object(r,'saved_cards',return_value=({race:deepcopy(item)},[],{})), \
                patch.object(r,'verify_commit',return_value={'forecast_sha256':'parent_hash'}), \
                patch.object(r,'read',side_effect=read),patch.object(r,'sha',return_value='fictional_digest'), \
                patch.object(r,'write',side_effect=lambda d,n,v:writes.update({n:deepcopy(v)})), \
                patch.object(r.old_store,'task_configuration',return_value={'status':'unavailable'}), \
                patch.object(r,'session') as db,patch.object(r.h_model,'predict') as predict:
            r.freeze(target,parent)
        db.assert_not_called();predict.assert_not_called()
        self.assertNotIn('results.local.json',reads)
        self.assertEqual(writes['forecasts.local.json']['races'][0]['H'],item['H'])
        self.assertEqual(writes['forecasts.local.json']['races'][0]['saved']['probabilities'],item['saved']['probabilities'])
        self.assertTrue(writes['forecast_commit.json']['results_not_opened_before_commit'])

    def test_cached_H_reuse_rejects_changed_roster_before_commit(self):
        item,_,_=fixture();race=tuple(item['race_key']);previous=deepcopy(item);previous['roster']['01']['horse_id']='WRONG_FICTION'
        with patch.object(r,'saved_cards',return_value=({race:item},[],{})), \
                patch.object(r,'verify_commit',return_value={'forecast_sha256':'parent_hash'}), \
                patch.object(r,'read',return_value={'races':[previous],'results_accessed':False}),patch.object(r,'write') as write:
            with self.assertRaises(safety.MLFailure):r.freeze(Path('target'),Path('parent'))
            write.assert_not_called()

    def test_cached_results_reject_parent_hash_mismatch(self):
        item,_,_=fixture()
        def read(path):
            if path.name=='forecasts.local.json':return {'races':[item],'reuse_parent_id':'a'*32,'parent_forecast_sha256':'expected'}
            if path.name=='dashboard_manifest.json':return {'result_sha256':'hash'}
            raise AssertionError('do_not_open_unverified_answers')
        with patch.object(r,'verify_commit',return_value={'forecast_sha256':'wrong'}),patch.object(r,'read',side_effect=read), \
                patch.object(r,'parent_directory',return_value=Path('parent')),patch.object(r,'session') as db:
            with self.assertRaises(safety.MLFailure):r.results(Path('target'))
            db.assert_not_called()

    def test_old_finished_HTML_is_returned_without_rewriting_or_reaggregation(self):
        from jra_racing_ai.duel import cli
        with TemporaryDirectory() as td:
            directory=Path(td);(directory/'予想対決.html').touch()
            summary={'kind':'old_unchanged_artifact'}
            with patch.object(r,'verify_commit'),patch.object(r,'sha',return_value='hash'), \
                    patch.object(r,'read',side_effect=[{'HTML_sha256':'hash','summary_sha256':'hash'},summary]), \
                    patch.object(r,'write') as write,patch.object(r,'freeze') as freeze:
                self.assertEqual(cli.build(directory),summary)
                write.assert_not_called();freeze.assert_not_called()

    def test_results_cannot_be_opened_before_valid_forecast_commit(self):
        with patch.object(r,'verify_commit',side_effect=safety.MLFailure('artifact_invalid')),patch.object(r,'session') as db,patch.object(r,'read') as read:
            with self.assertRaises(safety.MLFailure):r.results(Path('unused'))
            db.assert_not_called();read.assert_not_called()

    def test_only_two_approved_dates_and_intermediate_O1_query(self):
        item,_,_=fixture();race=tuple(item['race_key']);sql,args=r.query([race],{race:item});validate_audit_sql(sql)
        self.assertNotIn('jvd_se',sql);self.assertNotIn('jvd_hr',sql);self.assertIn("s.data_kubun::text='1'",sql)
        self.assertIn('10171429',args);self.assertIn('10171449',args)
        for day in ('0101','0930','1016','1019'):
            with self.assertRaises(safety.MLFailure):r.race_day(('2026',day,'05','01','01','01'))

    def test_input_construction_ignores_results_and_rejects_future_quote(self):
        item,_,_=fixture();race=tuple(item['race_key']);item['roster']={h:v for h,v in item['roster'].items() if h!='03'}
        row={'race_key':list(race),'record_type':'O1','status':'1','sale':'7','payload':'0100200102004002'+' '*8*26,
            'published':'10171449','oversized':False,'unsafe_version':False,'source_table':'apd_sokuho_o1'}
        prices={(race,10):[row],(race,30):[dict(row,published='10171429')]}
        a,_=r.select_inputs({race:deepcopy(item)},prices)
        changed=deepcopy(item);changed.update(finish_position=1,final_odds=1.1,payout=999999)
        b,_=r.select_inputs({race:changed},prices)
        self.assertTrue(a.equals(b));self.assertEqual(set(a.columns),{'entry_key','date',*r.H_FIELDS})
        cards={race:deepcopy(item)};r.select_inputs(cards,{(race,10):[dict(row,published='10171450')]})
        self.assertEqual(cards[race]['H']['status'],'missing')

    def test_correct_two_saved_models_are_used_without_recalibration(self):
        with TemporaryDirectory() as td,patch.object(r,'BASE',Path(td)):
            paths={}
            for day in r.DAYS:
                folder=Path(td)/day;folder.mkdir();paths[day]=folder;entries=[]
                for i in range(36):
                    race=('2026',day[5:7]+day[8:10],('04','05','08')[i//12],'01','01',f'{i%12+1:02d}')
                    for h,p in (('01',.6),('02',.4)):
                        entries.append(dict(zip(RACE_KEY,race),horse_number=h,horse_id='FICTION_'+h,scheduled_start_jst=day+'T15:00:00+09:00',
                            predicted_at_jst=day+'T07:00:00+09:00',model_id=r.SAVED_MODELS[day]['model_id'],
                            temperature='2.0' if day.endswith('17') else '0.75',win_probability=p))
                with (folder/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
                run={'status':'ok','model_id':entries[0]['model_id'],'temperature':float(entries[0]['temperature']),
                    'predictions_sha256':hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest()}
                (folder/'run.json').write_text(json.dumps(run),encoding='utf-8')
            # Synthetic fixtures may live under OneDrive; real artifact I/O still
            # uses the unchanged local-only safety layer in production.
            with patch.object(r,'SAVED',paths),patch.object(r,'read',side_effect=lambda p:json.loads(p.read_text(encoding='utf-8'))), \
                    patch.object(r,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()):
                cards,sources,refs=r.saved_cards()
            self.assertEqual(sum(v['saved']['status']=='ready' for v in cards.values()),72)
            for v in cards.values():
                self.assertEqual(v['saved']['probabilities'],{'01':.6,'02':.4})
                self.assertEqual(v['saved']['model_id'],r.SAVED_MODELS[v['date']]['model_id'])
                self.assertEqual(v['saved']['temperature'],r.SAVED_MODELS[v['date']]['temperature'])


class DuelHTMLTests(unittest.TestCase):
    def test_best242_default_and_separate_legacy_model_view(self):
        text=render.html(demo_payload())
        self.assertIn("day:'2026-10-18'",text)
        for token in ('10/18 カノープス','10/17 旧V0.3','両日・モデル別一覧','best242','09:26:31','T=0.75','T=2.0'):
            self.assertIn(token,text)
        self.assertNotIn('10/18のカノープス予測は欠損',text)
        self.assertNotIn('D.summary.all',text)

    def test_self_contained_HTML_has_controls_and_no_remote_assets(self):
        text=render.html(demo_payload())
        for token in ('結果を表示','finish-chart','winner-chart','id="different"','id="venue"','id="race-number"','connect-src \'none\''):
            self.assertIn(token,text)
        self.assertNotIn('__DUEL_DATA__',text);self.assertNotIn('src="http',text);self.assertNotIn('href="http',text)
        self.assertNotIn('fetch(',text);self.assertIn('result-hidden',text)

    def test_embedded_names_cannot_end_JSON_script_or_inject_HTML(self):
        d=demo_payload();name='</script><img src="https://invalid.example/">';d['races'][0]['horses'][0]['name']=name
        text=render.html(d);self.assertNotIn(name,text);self.assertEqual(text.count('<script'),2)
        encoded=text.split('<script id="duel-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(encoded)['races'][0]['horses'][0]['name'],name)


if __name__=='__main__':unittest.main()
