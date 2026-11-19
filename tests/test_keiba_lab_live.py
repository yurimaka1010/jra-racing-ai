"""Synthetic saved receipts and mocked SELECT only. No H outcome evaluation."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import csv
import hashlib
import json
import unittest

from jra_racing_ai import keiba_lab_live as m


def write(path, value, **kwargs):
    with path.open('x', encoding='utf-8') as f: json.dump(value, f)


def fixture(parent):
    race = ('2026','1024','05','04','05','01'); roster = ['01','02','03']
    folder = parent/'forward-four-sealed/2026-10-24/races'/''.join(race)
    folder.mkdir(parents=True)
    model_ids = {n:'synthetic-'+n for n in m.MODELS}
    plan = {'model_ids':model_ids,'temperatures':dict(zip(m.MODELS,(.75,1,.75,1)))}
    write(parent/'four_addendum_plan.json', plan)
    reg = {'race_key':list(race),'roster':roster,'scheduled_start':'2026-10-24T09:45:00+09:00'}
    write(folder/'registered.json',reg)
    entries = folder/'entries.json'; write(entries,{'payload':[{'horse_number':h,'horse_id':'horse-'+h} for h in roster]})
    p = {n:dict(zip(roster,[.6,.3,.1])) for n in m.MODELS}
    tickets = {n:{'ai_horse':'01','reason':'ev_selected'} for n in m.MODELS}
    base = {'race_key':list(race),'registered':reg,'input_cutoff':'2026-10-24T09:35:00+09:00',
        'generated_at':'2026-10-24T09:35:10+09:00','probabilities':{n:p[n] for n in ('old_D39','new_D39','H')},
        'model_ids':{n:model_ids[n] for n in ('old_D39','new_D39','H')},'result_accessed':False,
        'ledgers':{n:tickets[n] for n in ('old_D39','new_D39','H')},
        'entry_snapshot':{'path':str(entries),'sha256':m.sha(entries),'first_seen_at':'2026-10-24T09:34:50+09:00'},
        'quote_T10':{'first_seen_at':'2026-10-24T09:34:50+09:00','published_at':'2026-10-24T09:34:00+09:00'}}
    write(folder/'prediction.json',base)
    fields=['entry_key','model','model_id','probability','rank','model_status']
    def csv_file(filename,names):
        with (folder/filename).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for n in names:
                for i,h in enumerate(roster,1):w.writerow(dict(entry_key=''.join(race)+h,model=n,model_id=model_ids[n],probability=p[n][h],rank=i,model_status='saved'))
    csv_file('predictions.csv',('old_D39','new_D39','H'))
    receipt={'status':'saved','written_at':'2026-10-24T09:35:20+09:00','deadline':'2026-10-24T09:36:00+09:00',
        'prediction_sha256':m.sha(folder/'prediction.json'),'CSV_sha256':m.sha(folder/'predictions.csv')}
    write(folder/'receipt.json',receipt)
    side={'race_key':list(race),'probabilities':p,'model_ids':model_ids,'ledgers':tickets,
        'generated_at':'2026-10-24T09:35:25+09:00','input_cutoff':base['input_cutoff'],'result_accessed':False,
        'base_three_prediction_sha256':receipt['prediction_sha256']}
    write(folder/'four_prediction.json',side);csv_file('four_predictions.csv',m.MODELS)
    write(folder/'four_receipt.json',{'status':'saved','written_at':'2026-10-24T09:35:30+09:00',
        'prediction_sha256':m.sha(folder/'four_prediction.json'),'CSV_sha256':m.sha(folder/'four_predictions.csv')})
    return folder,plan


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.parent=self.base/'parent';self.parent.mkdir();self.root=self.base/'viewer';self.root.mkdir()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m,'BASE',self.base));self.stack.enter_context(patch.object(m,'ROOT',self.root))
        self.stack.enter_context(patch.object(m.safety,'local_path',side_effect=lambda p,**kw:p))
        self.stack.enter_context(patch.object(m.safety,'write_json',side_effect=write))
        self.stack.enter_context(patch.object(m,'read',side_effect=lambda p:json.loads(p.read_text(encoding='utf-8'))))
        self.stack.enter_context(patch.object(m,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()))
        self.folder,self.plan=fixture(self.parent)
        self.when=datetime.fromisoformat('2026-10-24T10:00:00+09:00');self.tick=0

    def record(self):return m.load_forecast(self.folder,self.plan,self.when)
    def viewer(self,session=None):return m.Viewer(parent=self.parent,directory=self.root/'run',session=session,clock=lambda:self.when,monotonic=lambda:self.tick)
    def results(self,category='7'):
        return [{'horse':h,'horse_id':'horse-'+h,'status':category,'anomaly':'0','rank':r}
            for h,r in zip(['01','02','03'],['02','01','03'])]
    def payout(self):return m.core.Outcome(tuple(self.record()['race_key']),'normal_final',{'02':450},True,True)

    def test_all_four_saved_values_and_ids_are_read_without_prediction(self):
        r=self.record();self.assertEqual(set(r['models']),set(m.MODELS))
        for n in m.MODELS:
            self.assertEqual([v['probability'] for v in r['models'][n]['rows']],[.6,.3,.1])
            self.assertEqual(r['models'][n]['model_id'],self.plan['model_ids'][n])

    def test_no_receipt_means_wait_not_fabricated_prediction(self):
        (self.folder/'receipt.json').unlink();r=self.record()
        self.assertFalse(r['models']);self.assertEqual(r['reason'],'forecast_not_saved')

    def test_frozen_D39_tickets_remain_pending_when_no_result_rows(self):
        r=m.result_projection(self.record(),[],None)
        self.assertEqual(r['result_status'],'pending');self.assertNotIn('H',r['models'])
        self.assertEqual(r['models']['new_D39']['paper_ticket']['stake_yen'],100)
        self.assertIsNone(r['models']['new_D39']['paper_ticket']['payout_yen'])
        self.assertIsNone(r['models']['new_D39']['top_hit'])

    def test_missing_fourth_keeps_original_three(self):
        (self.folder/'four_receipt.json').unlink();r=self.record()
        self.assertEqual(r['models']['control_D39']['status'],'missing')
        self.assertTrue(all(r['models'][n]['status']=='saved' for n in ('old_D39','new_D39','H')))

    def test_checksum_mismatch_blocks_untrusted_predictions(self):
        with (self.folder/'predictions.csv').open('a') as f:f.write('changed')
        with self.assertRaises(m.safety.MLFailure):self.record()

    def test_synthetic_or_post_result_records_are_not_accepted(self):
        for key in ('synthetic_verification','result_accessed'):
            path=self.folder/'prediction.json';b=m.read(path);b[key]=True;path.write_text(json.dumps(b))
            receipt=m.read(self.folder/'receipt.json');receipt['prediction_sha256']=m.sha(path)
            (self.folder/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(m.safety.MLFailure):self.record()

    def test_after_start_or_late_save_is_not_prerace_record(self):
        r=m.read(self.folder/'receipt.json');r['written_at']='2026-10-24T09:45:00+09:00'
        (self.folder/'receipt.json').write_text(json.dumps(r))
        with self.assertRaises(m.safety.MLFailure):self.record()

    def test_source_rank_must_match_saved_probability(self):
        path=self.folder/'predictions.csv';text=path.read_text().replace(',0.6,1,saved',',0.6,2,saved');path.write_text(text)
        sr=m.read(self.folder/'receipt.json');sr['CSV_sha256']=m.sha(path);(self.folder/'receipt.json').write_text(json.dumps(sr))
        with self.assertRaises(m.safety.MLFailure):self.record()

    def test_invalid_fourth_sidecar_keeps_authoritative_three(self):
        path=self.folder/'four_prediction.json';path.write_text('{}')
        r=self.record()
        self.assertEqual(r['models']['control_D39']['status'],'missing')
        self.assertTrue(all(r['models'][n]['status']=='saved' for n in ('old_D39','new_D39','H')))

    def test_display_name_adapter_selects_no_result_column(self):
        r=self.record();columns={p:{'type_is_builtin':True,'type_name':'varchar'} for p in ('umaban','ketto_toroku_bango','bamei')}
        session=Mock();session.return_value.__enter__=Mock(return_value=Mock());session.return_value.__exit__=Mock(return_value=False)
        rows=[{'race_key':r['race_key'],'horse':'01','horse_id':'horse-01','name':'架空テスト馬'}]
        with patch.object(m.local,'table_info',return_value=columns),patch.object(m.local,'_rows',return_value=rows) as selected:
            names=m.fetch_names([r],session)
            self.assertEqual(set(selected.call_args.args[3]),{'horse','horse_id','name'})
        self.assertEqual(names[r['race_id']]['01']['name'],'架空テスト馬')

    def test_prediction_response_has_no_results_and_no_H_tickets(self):
        r=m.public_forecast(self.record());text=json.dumps(r)
        self.assertNotIn('finish_position',text);self.assertNotIn('ledgers',text);self.assertNotIn('d39_tickets',text)

    def test_result_projection_excludes_H_even_when_source_contains_H(self):
        r=m.result_projection(self.record(),self.results(),self.payout())
        self.assertEqual(set(r['models']),set(m.D39));self.assertNotIn('synthetic-H',json.dumps(r))
        self.assertFalse(r['models']['old_D39']['top_hit']);self.assertEqual(r['models']['old_D39']['top_finish'],2)

    def test_rank_and_payout_are_from_outcomes_not_decision_odds(self):
        r=m.result_projection(self.record(),self.results(),self.payout())
        self.assertEqual(r['finishes']['02'],1)
        self.assertEqual(r['models']['new_D39']['rows'][1]['win_payout_100'],450)
        self.assertEqual(r['models']['new_D39']['paper_ticket']['payout_yen'],0)

    def test_SE7_can_show_rank_while_payout_remains_pending(self):
        r=m.result_projection(self.record(),self.results(),None)
        self.assertEqual(r['result_status'],'confirmed');self.assertEqual(r['payout_status'],'pending')
        self.assertIsNone(r['models']['new_D39']['paper_ticket']['payout_yen'])

    def test_full_order_flash_requires_confirmed_payout(self):
        self.assertEqual(m.result_projection(self.record(),self.results('5'),None)['result_status'],'pending')
        self.assertEqual(m.result_projection(self.record(),self.results('6'),self.payout())['result_status'],'confirmed')

    def test_partial_order_flash_is_never_final(self):
        self.assertEqual(m.result_projection(self.record(),self.results('3'),self.payout())['result_status'],'pending')

    def test_winner_mismatch_is_pending(self):
        outcome=m.core.Outcome(tuple(self.record()['race_key']),'normal_final',{'01':450},True,True)
        self.assertEqual(m.result_projection(self.record(),self.results(),outcome)['reason'],'hr_se_winner_mismatch')

    def test_horse_identity_and_duplicate_results_are_checked(self):
        rows=self.results();rows[0]['horse_id']='other'
        self.assertEqual(m.result_projection(self.record(),rows,self.payout())['result_status'],'pending')
        rows=self.results();rows[1]['horse']='01'
        self.assertEqual(m.result_projection(self.record(),rows,self.payout())['result_status'],'pending')

    def test_dead_heat_is_not_arbitrarily_selected(self):
        rows=self.results();rows[0]['rank']='01'
        self.assertEqual(m.result_projection(self.record(),rows,self.payout())['reason'],'dead_heat_or_unconfirmed_winner')

    def test_cancelled_or_vetoed_race_is_not_scored(self):
        r=self.record();r['valid_for_results']=False
        self.assertFalse(m.result_projection(r,self.results(),self.payout())['models'])

    def test_sealed_and_other_dates_fail_before_fetch(self):
        v=self.viewer()
        for day in ('2026-01-01','2026-09-30','2026-10-17','2027-03-14'):
            with self.assertRaises(m.safety.MLFailure):v.d39_results(day)

    def test_current_plan_day_is_visible_without_broadening_result_access(self):
        v=self.viewer()
        with patch.object(m,'fetch_results') as fetch:
            body=v.forecasts('2026-10-31')
            self.assertEqual(body['date'],'2026-10-31')
            self.assertEqual(body['coverage']['registered_races'],0)
            self.assertFalse(body['H_results_included'])
            with self.assertRaises(m.safety.MLFailure):v.d39_results('2026-10-31')
        fetch.assert_not_called()

    def test_prediction_window_ends_at_existing_plan_boundary(self):
        self.assertEqual(m.day_guard('2027-03-14'),'2027-03-14')
        for day in ('2026-09-30','2026-10-23','2027-03-15'):
            with self.assertRaises(m.safety.MLFailure):m.day_guard(day)

    def test_available_prediction_days_come_from_saved_directories(self):
        (self.parent/'forward-four-sealed/2026-10-31').mkdir()
        (self.parent/'forward-four-sealed/2026-01-01').mkdir()
        body=self.viewer().forecasts('2026-10-31')
        self.assertEqual(body['available_dates'],['2026-10-31','2026-10-24'])

    def test_pre_start_forecasts_do_not_trigger_result_SQL(self):
        self.when=datetime.fromisoformat('2026-10-24T09:36:00+09:00');v=self.viewer(session=Mock())
        with patch.object(m,'fetch_results',return_value=({},[])) as fetch:
            v.d39_results('2026-10-24');self.assertEqual(fetch.call_args.args[0],[])

    def test_provider_receives_D39_only_and_history_is_recorded(self):
        v=self.viewer(session=Mock());expected=m.result_projection(self.record(),self.results(),self.payout())
        with patch.object(m,'fetch_results',return_value=({expected['race_id']:expected},[])) as fetch:
            first=v.d39_results('2026-10-24');v.d39_results('2026-10-24')
            self.assertEqual(fetch.call_count,1)
            self.assertNotIn('H',fetch.call_args.args[0][0]['models'])
        self.assertEqual(len(list((v.directory/'view-history').glob('*.json'))),2)
        self.assertFalse(first['H_results_included']);self.assertNotIn('synthetic-H',json.dumps(first))

    def test_DB_failure_is_fixed_reason_no_secret_echo(self):
        v=self.viewer(session=Mock())
        with patch.object(m,'fetch_results',side_effect=Exception('postgres://private-secret actual-row')):
            r=v.d39_results('2026-10-24')
        self.assertEqual(r['reason'],'result_source_unavailable');self.assertNotIn('private-secret',json.dumps(r))

    def test_DB_failure_retains_last_confirmed_result_as_explicitly_stale(self):
        v=self.viewer(session=Mock());item=m.result_projection(self.record(),self.results(),self.payout())
        with patch.object(m,'fetch_results',return_value=({item['race_id']:item},[])):v.d39_results('2026-10-24')
        self.tick=31
        with patch.object(m,'fetch_results',side_effect=Exception('synthetic connection interruption')):body=v.d39_results('2026-10-24')
        self.assertTrue(body['stale']);self.assertEqual(len(body['races']),1)
        self.assertEqual(body['races'][0]['models']['new_D39']['top_finish'],2)
        self.assertNotIn('H',body['races'][0]['models'])

    def test_incremental_result_polling_keeps_new_versions_not_rewrite(self):
        v=self.viewer(session=Mock())
        with patch.object(m,'fetch_results',return_value=({},[])):
            v.d39_results('2026-10-24')
        self.tick=31;item=m.result_projection(self.record(),self.results(),self.payout())
        with patch.object(m,'fetch_results',return_value=({item['race_id']:item},[])):
            self.assertEqual(len(v.d39_results('2026-10-24')['races']),1)
        self.assertEqual(len(list((v.directory/'result-versions').glob('*.json'))),2)

    def test_frozen_prediction_is_not_overwritten_on_later_tampering(self):
        v=self.viewer();v.forecasts('2026-10-24');f=next((v.directory/'fixed-predictions').glob('*.json'));before=f.read_bytes()
        with patch.object(m,'load_forecast',return_value=deepcopy(self.record())) as load:
            changed=load.return_value;changed['models']['old_D39']['rows'][0]['probability']=.8;self.tick=11
            r=v.forecasts('2026-10-24')
        self.assertEqual(f.read_bytes(),before)
        self.assertIn('ledger_tampered',r['coverage']['missing_reasons'])
        self.assertEqual(r['races'][0]['models']['old_D39']['rows'][0]['probability'],.6)

    def test_CSV_reuses_25_columns_and_excludes_H(self):
        r=m.result_projection(self.record(),self.results(),self.payout())
        text=m.result_csv({'date':'2026-10-24','races':[r]})
        self.assertEqual(text.splitlines()[0].lstrip('\ufeff').split(','),list(m.COLUMNS))
        self.assertNotIn('synthetic-H',text);self.assertEqual(len(text.splitlines()),10)

    def test_cross_origin_and_unexpected_host_are_denied_before_data(self):
        v=Mock();H=m.make_handler(v,8767)
        for headers in ({'Host':'evil.example'},{'Host':'127.0.0.1:8767','Origin':'https://other.example'},
                {'Host':'127.0.0.1:8767','Sec-Fetch-Site':'cross-site'}):
            h=object.__new__(H);h.headers=headers;h.send_error=Mock();h.path='/api/d39-results?date=2026-10-24'
            h.do_GET();h.send_error.assert_called_once_with(403)
        v.d39_results.assert_not_called()

    def test_HTTP_date_guard_cannot_be_overridden_with_model_parameter(self):
        v=Mock();H=m.make_handler(v,8767);h=object.__new__(H)
        h.headers={'Host':'127.0.0.1:8767'};h.path='/api/d39-results?date=2026-01-01&model=H'
        h.send_response=Mock();h.send_header=Mock();h.end_headers=Mock();h.wfile=BytesIO()
        h.do_GET();h.send_response.assert_called_once_with(400);v.d39_results.assert_not_called()

    def test_cross_site_top_level_HTML_is_not_an_API_permission(self):
        v=Mock();H=m.make_handler(v,8767);h=object.__new__(H)
        h.headers={'Host':'127.0.0.1:8767','Sec-Fetch-Site':'cross-site','Sec-Fetch-Mode':'navigate','Sec-Fetch-Dest':'document'}
        h.path='/';h.send_response=Mock();h.send_header=Mock();h.end_headers=Mock();h.wfile=BytesIO();h.send_error=Mock()
        h.do_GET();h.send_response.assert_called_once_with(200);v.d39_results.assert_not_called()
        h.path='/api/d39-results?date=2026-10-24';h.send_error.reset_mock();h.do_GET();h.send_error.assert_called_once_with(403)

    def test_HR1_is_explicitly_scoped_and_existing_decoder_is_reused(self):
        session=Mock();session.return_value.__enter__=Mock(return_value=Mock());session.return_value.__exit__=Mock(return_value=False)
        columns={p:{'type_is_builtin':True,'type_name':'varchar'} for p in ('umaban','ketto_toroku_bango','kakutei_chakujun','data_kubun','ijo_kubun_code')}
        request=self.record();request['models']={n:value for n,value in request['models'].items() if n in m.D39}
        with patch.object(m.local,'table_info',return_value=columns),patch.object(m.local,'_rows',return_value=[dict(r,race_key=request['race_key']) for r in self.results('5')]),patch('jra_racing_ai.d39_generation.outcomes.reviewed_profile',return_value={'allowed_final_categories':['2']}),patch.object(m.settle_saved,'fetch_hr',return_value=([{'synthetic':'HR1'}],{'synthetic':True},{'rows':1})) as fetch,patch.object(m.settle_saved,'decode_existing_hr',return_value=[self.payout()]) as decode:
            results,_=m.fetch_results([request],session)
            self.assertEqual(fetch.call_args.kwargs['reviewed_source']['allowed_final_categories'],['1','2'])
            decode.assert_called_once();self.assertEqual(results[request['race_id']]['result_status'],'confirmed')

    def test_saved_ticket_changes_are_not_accepted_even_if_probabilities_are_same(self):
        v=self.viewer();v.forecasts('2026-10-24');r=self.record();r['source_prediction_sha256']='f'*64;self.tick=11
        with patch.object(m,'load_forecast',return_value=r):body=v.forecasts('2026-10-24')
        self.assertIn('ledger_tampered',body['coverage']['missing_reasons'])

    def reference_fixture(self):
        folder=self.parent/'dry-four/synthetic-reference';folder.mkdir(parents=True)
        write(folder/'summary.json',{'date':'2026-10-24','kind':'declared_entry_D39_dry_run',
            'results_read':False,'real_forward_record':False,'new_fits':0,'checked_at':'2026-10-24T09:35:30+09:00'})
        with (self.folder/'predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if r['model'] in m.D39]
        rows.extend([dict(r,model='control_D39',model_id=self.plan['model_ids']['control_D39']) for r in rows if r['model']=='old_D39'])
        for r in rows:r.update(temperature=self.plan['temperatures'][r['model']],dry_run_only=True,
            generated_at='2026-10-24T09:35:10+09:00',source_observed_at='2026-10-24T09:34:50+09:00')
        with (folder/'predictions.dry.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return folder

    def test_reference_displays_saved_D39_values_without_H_or_results(self):
        self.reference_fixture();v=self.viewer();body=v.references('2026-10-24')
        self.assertEqual(len(body['races']),1);self.assertEqual(body['coverage']['three_model_races'],1)
        self.assertFalse(body['real_forward_record']);self.assertFalse(body['results_accessed'])
        self.assertNotIn('H',body['races'][0]['models']);self.assertNotIn('finish_position',json.dumps(body))
        self.assertEqual(body['races'][0]['models']['new_D39']['rows'][0]['probability'],.6)

    def test_reference_never_enters_result_cohort(self):
        self.reference_fixture();(self.folder/'receipt.json').unlink();v=self.viewer(session=Mock())
        self.assertEqual(len(v.references('2026-10-24')['races']),1)
        with patch.object(m,'fetch_results',return_value=({},[])) as fetch:
            self.assertFalse(v.d39_results('2026-10-24')['races']);self.assertFalse(fetch.call_args.args[0])

    def test_reference_source_cannot_be_rewritten_under_same_identifier(self):
        folder=self.reference_fixture();v=self.viewer();v.references('2026-10-24')
        with (folder/'predictions.dry.csv').open('a') as f:f.write('\n')
        with self.assertRaises(m.safety.MLFailure):v.references('2026-10-24')


if __name__=='__main__':unittest.main()
