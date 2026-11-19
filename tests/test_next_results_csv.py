"""Synthetic saved results only; no real records, authentication, fitting, or sockets."""
from contextlib import ExitStack
from copy import deepcopy
from io import StringIO,BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock,patch
import csv
import json
import unittest
from jra_racing_ai.d39_next import results_csv as m
from jra_racing_ai import keiba_lab_live as lab


def packet(choice='improved'):
    names=m.CHOICES[choice];race='2026102505010101'
    models={name:{'status':'saved','model_id':'fake-'+name,'predicted_at':'2026-10-25T09:30:10+09:00',
        'rows':[{'horse_no':'01','rank':1,'probability':.7},{'horse_no':'02','rank':2,'probability':.3}]} for name in names}
    forecast={'race_id':race,'source_prediction_sha256':'a'*64,'models':deepcopy(models),'next_surface':'11','next_distance':1600}
    for model in models.values():
        model.update(paper_horse='01',paper_ticket={'status':'settled','stake_yen':100,'payout_yen':0,'net_profit_yen':-100,'hit':False})
        for row in model['rows']:row.update(finish_position=2 if row['horse_no']=='01' else 1,win_payout_100=0 if row['horse_no']=='01' else 450)
    result={'race_id':race,'venue':'テスト','race_no':1,'scheduled_start':'2026-10-25T09:40:00+09:00',
        'input_cutoff':'2026-10-25T09:30:00+09:00','source_prediction_sha256':'a'*64,
        'result_status':'confirmed','payout_status':'confirmed','models':models,'horse_names':{'01':'架空,テスト"馬','02':'架空馬二'}}
    metrics={name:{'races':1,'runners':2,'race_log_loss':1.2039728043259361,'horse_log_loss':1.2,
        'brier_score':.49,'roc_auc':0.,'within_race_auc':0.,'top1_accuracy_percent':0.,'top2_percent':100.,'top3_percent':0.} for name in names}
    financial={name:{'bet_count':1,'skip_count':0,'investment_yen':100,'hit_count':0,'hit_rate_percent':0.,
        'payout_yen':0,'net_profit_yen':-100,'return_rate_percent':0.,'unsettled_count':0,'settled_bet_count':1} for name in names}
    return {'date':'2026-10-25','activation':{'run_id':'b'*32},'races':[forecast],
        'results':{'date':'2026-10-25','checked_at':'2026-10-25T10:00:00+09:00','races':[result],
            'metrics':metrics,'financial':financial,'common_result_races':1,'pending_result_races':0,
            'kind':'prospective_exploratory','H_results_included':False,'Stage7_results_read':False}}


def rows(value):return list(csv.DictReader(StringIO(value.lstrip('\ufeff'))))


class CsvTests(unittest.TestCase):
    def test_exact_saved_values_official_results_and_model_ids(self):
        p=packet();detail,summary,info=m.build(p,'improved');export=rows(detail)
        self.assertTrue(detail.startswith('\ufeff'));self.assertEqual(len(export),10)
        r=next(r for r in export if r['model_id']=='enriched_D39' and r['horse_no']=='1')
        self.assertEqual(float(r['win_prob']),.7);self.assertEqual(r['pred_rank'],'1');self.assertEqual(r['finish_position'],'2')
        self.assertEqual(r['win_payout_100'],'0');self.assertEqual(r['virtual_net_profit_yen'],'-100')
        self.assertEqual(r['model_artifact_id'],'fake-enriched_D39');self.assertEqual(r['horse_name'],'架空,テスト"馬')
        self.assertEqual(float(rows(summary)[0]['race_log_loss']),p['results']['metrics']['old_D39']['race_log_loss'])
        self.assertEqual(info['races'],1)

    def test_pending_outcomes_and_payouts_are_empty_not_losses(self):
        p=packet();r=p['results']['races'][0];r.update(result_status='pending',payout_status='pending')
        for model in r['models'].values():
            model['paper_ticket'].update(status='pending',payout_yen=None,net_profit_yen=None,hit=None)
            for v in model['rows']:v.update(finish_position=None,win_payout_100=None)
        for money in p['results']['financial'].values():money.update(payout_yen=None,net_profit_yen=None,return_rate_percent=None,unsettled_count=1)
        detail,summary,_=m.build(p,'improved')
        first=rows(detail)[0];self.assertEqual(first['finish_position'],'');self.assertEqual(first['win_payout_100'],'')
        self.assertEqual(first['virtual_net_profit_yen'],'');self.assertEqual(first['virtual_status'],'pending')
        self.assertEqual(rows(summary)[0]['return_rate_percent'],'')

    def test_result_and_stage7_seals_rejected_even_with_valid_predictions(self):
        for flag in ('H_results_included','Stage7_results_read'):
            p=packet();p['results'][flag]=True
            with self.assertRaises(m.safety.MLFailure):m.build(p,'improved')
        p=packet();p['results']['races'][0]['models']['H']={}
        with self.assertRaises(m.safety.MLFailure):m.build(p,'improved')
        p=packet();p['results']['metrics']['H']={}
        with self.assertRaises(m.safety.MLFailure):m.build(p,'improved')

    def test_reference_predictions_never_substitute_missing_formal_record(self):
        p=packet();p['reference']={'races':p.pop('races')}
        with self.assertRaises(m.safety.MLFailure):m.build(p,'improved')

    def test_model_or_probability_mismatch_cannot_be_exported(self):
        for field,value in [('model_id','other'),('rows',[{'horse_no':'01','probability':.71,'rank':1}])]:
            p=packet();p['results']['races'][0]['models']['enriched_D39'][field]=value
            with self.assertRaises(m.safety.MLFailure):m.build(p,'improved')

    def test_missing_optional_model_is_not_fabricated(self):
        p=packet();p['results']['races'][0]['models']['control_D39']={'status':'missing'}
        detail,_,_=m.build(p,'improved');export=rows(detail)
        self.assertEqual(len(export),8);self.assertNotIn('control_D39',{r['model_id'] for r in export})
        self.assertEqual({r['common_probability_cohort'] for r in export},{'False'})

    def test_original_and_improved_candidates_are_separate(self):
        old=m.build(packet('original'),'original');new=m.build(packet(),'improved')
        self.assertEqual(len(rows(old[0])),8);self.assertEqual(len(rows(new[0])),10)
        self.assertNotIn('enriched_D39',{r['model_id'] for r in rows(old[0])})
        self.assertEqual({r['source_stage'] for r in rows(new[0])},{'d39_enrichment'})

    def test_empty_results_produce_header_only(self):
        p=packet();p['results']=None
        detail,summary,info=m.build(p,'improved');self.assertEqual(rows(detail),[]);self.assertEqual(rows(summary),[])
        self.assertEqual(info['status'],'pending')

    def test_csv_formula_text_is_escaped_but_numeric_losses_are_not(self):
        p=packet();p['results']['races'][0]['horse_names']['01']='=1+1'
        detail,_,_=m.build(p,'improved');r=rows(detail)[0]
        self.assertEqual(r['horse_name'],"'=1+1");self.assertEqual(r['virtual_net_profit_yen'],'-100')

    def test_download_handler_has_attachment_and_csv_headers(self):
        handler=lab.make_handler(Mock(),8767).__new__(lab.make_handler(Mock(),8767))
        handler.path='/api/next-results.csv?date=2026-10-25&candidate=improved&kind=detail'
        handler.headers={'Host':'127.0.0.1:8767'};handler.wfile=BytesIO()
        handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
        with patch.object(m,'download',return_value=m.build(packet(),'improved')[0]):handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        headers=dict(c.args for c in handler.send_header.call_args_list)
        self.assertIn('text/csv',headers['Content-Type']);self.assertIn('attachment;',headers['Content-Disposition'])
        self.assertEqual(len(rows(handler.wfile.getvalue().decode('utf-8'))),10)

    def test_download_rejects_sealed_date_and_unknown_candidate(self):
        for query in ('date=2026-01-01&candidate=improved&kind=detail','date=2026-10-25&candidate=H&kind=summary'):
            cls=lab.make_handler(Mock(),8767);handler=cls.__new__(cls)
            handler.path='/api/next-results.csv?'+query;handler.headers={'Host':'127.0.0.1:8767'};handler.wfile=BytesIO()
            handler.send_response=Mock();handler.send_header=Mock();handler.end_headers=Mock()
            with patch.object(lab,'next_packet') as api:handler.do_GET()
            api.assert_not_called();handler.send_response.assert_called_once_with(400)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'csv'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m,'ROOT',self.root))
        self.stack.enter_context(patch.object(m.safety,'local_path',side_effect=lambda p,**kwargs:p))

    def test_semantically_identical_results_do_not_duplicate_versions(self):
        p=packet();first=m.persist(p,'improved');p['results']['checked_at']='2026-10-25T11:00:00+09:00'
        second=m.persist(p,'improved')
        self.assertEqual(first['revision'],second['revision'])
        self.assertEqual(len(list((self.root/p['date']/'improved'/'revisions').iterdir())),1)
        self.assertEqual(len(rows((self.root/p['date']/'improved'/'results.csv').read_text(encoding='utf-8'))),10)

    def test_changed_results_get_new_version_and_keep_old_csv(self):
        p=packet();first=m.persist(p,'improved');old=self.root/p['date']/'improved'/'revisions'/first['revision']/'results.csv'
        before=old.read_bytes();p['results']['races'][0]['horse_names']['01']='別の表示名'
        second=m.persist(p,'improved');self.assertNotEqual(first['revision'],second['revision'])
        self.assertEqual(old.read_bytes(),before)

    def test_original_and_improved_files_have_independent_directories(self):
        a=m.persist(packet('original'),'original');b=m.persist(packet(),'improved')
        self.assertNotEqual(a['directory'],b['directory'])

    def test_optional_csv_failure_does_not_break_prediction_packet(self):
        p=packet('original');p['improved']=packet();before=deepcopy(p['races'])
        with patch.object(m,'export_status',side_effect=OSError('synthetic')):m.attach_exports(p)
        self.assertEqual(p['races'],before);self.assertEqual(p['csv_exports']['improved']['status'],'pending')

    def test_download_uses_verified_saved_csv_without_prediction_or_DB(self):
        p=packet();info=m.persist(p,'improved')
        self.assertEqual(len(rows(m.download(p['date'],'improved','detail'))),10)
        path=self.root/p['date']/'improved'/'revisions'/info['revision']/'results.csv'
        path.write_text('synthetic corruption')
        with self.assertRaises(m.safety.MLFailure):m.download(p['date'],'improved','detail')

    def test_unchanged_saved_results_need_no_model_load_or_forecast_replay(self):
        from jra_racing_ai.d39_next import forward
        p=packet('original');m.persist(p,'original')
        source=self.root/'source';source.mkdir();(source/'current.json').write_text(json.dumps(p['activation']))
        body=deepcopy(p['results']);body['checked_at']='2026-10-25T11:00:00+09:00'
        body['races'][0].pop('horse_names')
        (source/'latest_results.json').write_text(json.dumps(body))
        with patch.object(m.c,'ROOT',source),patch.object(m.c,'folder',return_value=source), \
             patch.object(m.c,'read',side_effect=lambda path:json.loads(path.read_text())),patch.object(forward,'saved') as replay:
            result=m.refresh_saved(p['date'],'original')
        replay.assert_not_called();self.assertEqual(result['status'],'saved')


if __name__=='__main__':unittest.main()
