"""Only fictional prices, horses and HR packets; no DB/network/training."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest
from unittest.mock import patch

from test_pedigree_replay_review import record,official
from jra_racing_ai import pedigree_replay_review as r,pedigree_market_favorite as m
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim.core import Outcome


def cache(rec,o,*,refund='0',winner='02'):
    key=tuple(rec['race_key'])
    hr={'race_key':list(key),'record_type':'HR','status':'1','unestablished':'0','special':'0','refund':refund}
    for i in range(1,4):
        for part,width in zip('abc',(2,9,2)):hr[f'slot_{i}{part}']=' '*width
    hr.update(slot_1a=winner,slot_1b='000000280',slot_1c='01')
    profile={'table':'jvd_hr','reviewed_normal_conditions':True,'currency':'JPY','base_stake_yen':100,
        'final_category':'2','allowed_final_categories':['1','2'],'revision_policy':'single_reviewed_final',
        'semantics_reviewed':True,'official_reference':'synthetic_only','local_reconciliation_reference':'synthetic_only',
        'years':[2026],'venues':['05'],'numerator':1,'denominator':1,
        'columns':{'status':'data_kubun','unestablished':'fuseiritsu_flag_tansho','special':'tokubarai_flag_tansho','refund':'henkan_flag_tansho'}}
    return {'only_target_dates':list(r.DAYS),'SE':[o['SE'][key]],'RA':[o['RA'][key]],
        'HR':[{'date':rec['date'],'HR_rows':[hr],'reviewed_source':profile}]}


def fixture():
    rec=record();rec['quote_T10']['odds']={'01':4.,'02':2.,'03':8.};o=official(rec)
    return rec,r.analyze([rec],o),cache(rec,o)


class MarketFavoriteTests(unittest.TestCase):
    def test_lowest_saved_T10_price_selected(self):
        rec,_,_=fixture();pick=m.selection(rec)
        self.assertEqual(pick['horse_number'],'02');self.assertEqual(pick['T10_win_odds'],2.)
        self.assertEqual(pick['T10_popularity_rank'],1)

    def test_equal_lowest_uses_smallest_number_not_winner(self):
        rec=record();rec['roster']=['03','02','01'];rec['quote_T10']['odds']={'03':8,'02':'2.00','01':2.}
        pick=m.selection(rec);self.assertEqual(pick['horse_number'],'01')
        self.assertTrue(pick['T10_popularity_tied']);self.assertEqual(pick['tied_lowest_horse_numbers'],['01','02'])

    def test_final_prices_and_results_do_not_select_horse(self):
        rec,_,_=fixture();a=m.selection(rec)
        rec.update(final_odds={'03':1.},actual_winner='03',current_finish='01')
        self.assertEqual(m.selection(rec),a)

    def test_missing_stale_and_future_prices_not_used(self):
        for kind in ('missing','stale','future'):
            rec=record()
            if kind=='missing':rec['quote_T10']=None
            elif kind=='stale':rec['quote_T10']['published_at']='2026-10-24T09:40:00+09:00'
            else:rec['quote_T10']['first_seen_at']='2026-10-24T10:00:01+09:00'
            self.assertEqual(m.selection(rec)['status'],'unavailable')

    def test_official_HR_not_T10_price_used_for_payout(self):
        rec,report,data=fixture();value=m.add(report,[rec],data)
        cash=value[m.NAME]['scopes']['all|all']['finance_on_three_common'][m.NAME]
        self.assertEqual(cash['bet_count'],1);self.assertEqual(cash['payout_yen'],280)
        self.assertEqual(cash['net_profit_yen'],180);self.assertEqual(cash['return_rate_percent'],280.)
        self.assertEqual(cash['hit_rate_percent'],100.)

    def test_original_three_models_and_metrics_not_changed(self):
        rec,report,data=fixture();before=deepcopy(report);source=deepcopy(rec)
        with patch('lightgbm.train',side_effect=AssertionError('no_fit')):
            value=m.add(report,[rec],data)
        self.assertTrue(m.unchanged_original(report,value));self.assertEqual(report,before);self.assertEqual(rec,source)

    def test_market_only_missing_does_not_shrink_old_cohort(self):
        rec,report,data=fixture();data['HR']=[];value=m.add(report,[rec],data)
        scope=value[m.NAME]['scopes']['all|all']
        self.assertEqual(scope['old_three_common_races'],1)
        self.assertEqual(scope['market_only_unsettled_on_three_common'],1)
        self.assertEqual(scope['four_common_races'],0)
        self.assertIsNone(scope['finance_on_three_common'][m.NAME]['return_rate_percent'])
        self.assertEqual(scope['finance_on_three_common']['pedigree45'],report['finance_common_settled']['pedigree45'])

    def test_market_name_missing_does_not_block_valid_settlement(self):
        rec,report,data=fixture();data['SE'][0][1]['horse_name']=None
        value=m.add(report,[rec],data);scope=value[m.NAME]['scopes']['all|all']
        self.assertEqual(scope['market_missing_name_races'],1)
        self.assertEqual(scope['market_only_unsettled_on_three_common'],0)

    def test_identity_mismatch_stays_unsettled(self):
        rec,report,data=fixture();data['SE'][0][1]['horse_id']='9999999999'
        value=m.add(report,[rec],data)
        self.assertEqual(value['details'][0][m.NAME]['ticket']['status'],'unsettled')

    def test_irregular_finish_remains_in_P3_with_loss(self):
        for anomaly in ('4','5'):
            rec=record();rec['quote_T10']['odds']={'01':4.,'02':2.,'03':8.};o=official(rec);key=tuple(rec['race_key'])
            o['SE'][key][0]['finish_position']='01';o['SE'][key][1].update(finish_position='00',anomaly=anomaly)
            o['payouts'][key]=Outcome(key,'normal_final',{'01':280},True,True)
            report=r.analyze([rec],o);value=m.add(report,[rec],cache(rec,o,winner='01'))
            self.assertEqual(report['P1_eligible_races'],0);self.assertEqual(value[m.NAME]['scopes']['all|all']['P3_races'],1)
            self.assertEqual(value['details'][0][m.NAME]['ticket']['payout_yen'],0)

    def test_refund_uses_existing_P3_rule(self):
        rec=record();rec['quote_T10']['odds']={'01':4.,'02':2.,'03':8.};o=official(rec);key=tuple(rec['race_key'])
        o['SE'][key][1].update(anomaly='1',finish_position='00')
        o['payouts'][key]=Outcome(key,'pending',reason='refund_pending')
        report=r.analyze([rec],o);value=m.add(report,[rec],cache(rec,o,refund='1'))
        t=value['details'][0][m.NAME]['ticket']
        self.assertEqual(t['status'],'refunded');self.assertEqual(t['payout_yen'],100);self.assertFalse(t['hit'])

    def test_P3_cohort_mismatch_stops(self):
        rec,report,data=fixture();report['details'][0]['P3']=False
        with self.assertRaises(MLFailure):m.add(report,[rec],data)

    def test_cached_data_outside_two_dates_rejected(self):
        rec,report,data=fixture();data['SE'][0][0]['race_key']=['2026','0101','05','01','01','01']
        with self.assertRaises(MLFailure):m.add(report,[rec],data)

    def test_precommitted_choice_cannot_change_on_join(self):
        rec,report,data=fixture();choice={tuple(rec['race_key']):dict(m.selection(rec),horse_number='03')}
        with self.assertRaises(MLFailure):m.add(report,[rec],data,choice)

    def test_local_report_has_four_money_bars_and_official_market_fields(self):
        from jra_racing_ai.pedigree_replay_report import save
        rec,report,data=fixture();report.update(model_ids=r.frozen.MODEL_IDS,temperatures=r.frozen.TEMPERATURE,
            caveats=['開催後再生'],created_at='2026-10-26T15:00:00+09:00',artifact_id='synthetic')
        value=m.add(report,[rec],data);project=Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=project) as temp:
            path=Path(temp);save(path,value)
            text=(path/'比較レポート.html').read_text(encoding='utf-8')
            self.assertIn('市場1番人気',text);self.assertIn('R1払戻（100円）',text)
            self.assertIn('市場1番人気',(path/'本命単勝の仮想損益.svg').read_text(encoding='utf-8'))
            with (path/'市場1番人気の仮想購入.local.csv').open(encoding='utf-8-sig',newline='') as s:rows=list(csv.DictReader(s))
            self.assertEqual(rows[0]['市場本命馬番'],'02');self.assertEqual(rows[0]['払戻額_円'],'280')


if __name__=='__main__':unittest.main()
