"""Fictional replay review only. No database, fitting or protected forward inputs."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from jra_racing_ai import pedigree_replay_review as r
from jra_racing_ai.win_sim.core import Outcome
from jra_racing_ai.ml.safety import MLFailure


def record():
    models={}
    for name,values in zip(r.NAMES,((.6,.3,.1),(.3,.6,.1),(.5,.4,.1))):
        p=dict(zip(('01','02','03'),values))
        models[name]={'status':'saved','model_id':r.frozen.MODEL_IDS[name],'temperature':r.frozen.TEMPERATURE[name],
            'probability':p,'rank':{h:i for i,h in enumerate(sorted(p,key=lambda h:(-p[h],h)),1)}}
    return {'race_key':['2026','1024','05','01','01','01'],'date':'2026-10-24','roster':['01','02','03'],
        'horse_registration_ids':{f'{i:02d}':f'{i:010d}' for i in range(1,4)},'models':models,
        'replay_only':True,'H_results_accessed':False,'input_cutoff':'2026-10-24T10:00:00+09:00',
        'prediction_at':'2026-10-26T06:00:00+09:00','quote_T10':{'odds':{'01':2.,'02':3.,'03':6.},
        'published_at':'2026-10-24T09:58:00+09:00','first_seen_at':'2026-10-24T09:59:00+09:00'}}


def official(rec):
    key=tuple(rec['race_key'])
    rows=[{'race_key':key,'horse_number':f'{h:02d}','horse_id':f'{h:010d}',
           'finish_position':f'{pos:02d}','status':'7','anomaly':'0','horse_name':f'架空馬{h}'}
           for h,pos in ((1,2),(2,1),(3,3))]
    return {'SE':{key:rows},'RA':{key:[{'race_key':key,'status':'7','actual_runners':'03'}]},
        'payouts':{key:Outcome(key,'normal_final',{'02':280},True,True)},'sources':[]}


class ReplayReviewTests(unittest.TestCase):
    def test_two_dates_only(self):
        for text in ('2026-01-01','2026-09-30','2026-10-31','2027-03-21','2025-01-01'):
            with self.assertRaises(MLFailure):r.check_day(text)
        self.assertEqual(str(r.check_day('2026-10-25')),'2026-10-25')

    def test_formal_record_rejected(self):
        x=record();x['replay_only']=False
        with self.assertRaises(MLFailure):r.validate_prediction(x)

    def test_H_record_rejected(self):
        x=record();x['models']['H']=x['models']['control39']
        with self.assertRaises(MLFailure):r.validate_prediction(x)

    def test_saved_ranks_and_probabilities_validated(self):
        x=record();r.validate_prediction(x)
        x['models']['pedigree45']['rank']['02']=3
        with self.assertRaises(MLFailure):r.validate_prediction(x)

    def test_unmatched_horse_ID_is_not_joined(self):
        x=record();o=official(x);o['SE'][tuple(x['race_key'])][0]['horse_id']='9999999999'
        value=r.analyze([x],o)
        self.assertEqual(value['P1_eligible_races'],0)
        self.assertIn('official_identity_mismatch',value['exclusion_reasons'])

    def test_metric_values_match_definitions_and_saved_probabilities(self):
        x=record();o=official(x);before=deepcopy(x)
        with patch('lightgbm.train',side_effect=AssertionError('no_fit')):
            v=r.analyze([x],o)
        a=v['periods']['all']['P1']['control39'];b=v['periods']['all']['P1']['pedigree45']
        self.assertEqual(a['within_race_auc'],.5);self.assertEqual(b['within_race_auc'],1.)
        self.assertAlmostEqual(a['race_log_loss'],-math.log(.3))
        self.assertAlmostEqual(a['brier_score'],(.6**2+(.3-1)**2+.1**2)/3)
        self.assertEqual(a['top1_hit_count'],0);self.assertEqual(b['top1_hit_count'],1)
        self.assertAlmostEqual(v['pedigree_minus_control']['within_race_auc'],.5)
        self.assertEqual(x,before)

    def test_equal_top_probability_uses_saved_horse_number_tie_rule(self):
        x=record();x['models']['control39']['probability']={'01':.45,'02':.45,'03':.1}
        v=r.analyze([x],official(x))['periods']['all']['P1']['control39']
        self.assertEqual(v['top1_accuracy_percent'],0)
        self.assertEqual(v['top_probability_tie_races'],1)

    def test_same_cohort_market_probability(self):
        x=record();v=r.analyze([x],official(x))
        self.assertEqual(v['P1_eligible_races'],v['P2_eligible_races'])
        self.assertAlmostEqual(v['periods']['all']['P2']['market']['race_log_loss'],-math.log(1/3))

    def test_missing_odds_does_not_remove_accuracy_race(self):
        x=record();x['quote_T10']=None;v=r.analyze([x],official(x))
        self.assertEqual(v['P1_eligible_races'],1);self.assertEqual(v['P2_eligible_races'],0)
        self.assertEqual(v['P3_eligible_races'],0)
        self.assertIsNone(v['finance_P3']['control39']['return_rate_percent'])

    def test_future_or_stale_odds_not_used(self):
        for field,value in (('first_seen_at','2026-10-24T10:00:01+09:00'),('published_at','2026-10-24T09:50:00+09:00')):
            x=record();x['quote_T10'][field]=value;v=r.analyze([x],official(x))
            self.assertEqual(v['P2_eligible_races'],0);self.assertEqual(v['P3_eligible_races'],0)

    def test_irregular_race_excluded_whole_but_P3_retained(self):
        x=record();o=official(x);row=o['SE'][tuple(x['race_key'])][2]
        row.update(anomaly='4',finish_position='00')
        v=r.analyze([x],o)
        self.assertEqual(v['P1_eligible_races'],0);self.assertEqual(v['P3_eligible_races'],1)
        self.assertEqual(v['finance_P3']['pedigree45']['payout_yen'],280)

    def test_flash_results_require_normal_confirmed_HR(self):
        x=record();o=official(x)
        for row in o['SE'][tuple(x['race_key'])]:row['status']='6'
        self.assertEqual(r.analyze([x],o)['P1_eligible_races'],1)
        o['payouts']={}
        self.assertEqual(r.analyze([x],o)['P1_eligible_races'],0)

    def test_monday_final_metrics_can_continue_without_HR_profit(self):
        x=record();o=official(x);o['payouts']={};v=r.analyze([x],o)
        self.assertEqual(v['P1_eligible_races'],1)
        self.assertIsNone(v['finance_P3']['control39']['net_profit_yen'])
        self.assertEqual(v['finance_P3']['control39']['unsettled_count'],1)

    def test_R1_uses_official_payout_not_T10_price_and_no_EV_filter(self):
        x=record();x['quote_T10']['odds']={h:1. for h in x['roster']};v=r.analyze([x],official(x))
        cash=v['finance_P3']['pedigree45']
        self.assertEqual(cash['investment_yen'],100);self.assertEqual(cash['payout_yen'],280)
        self.assertEqual(cash['net_profit_yen'],180);self.assertEqual(cash['return_rate_percent'],280.)

    def test_disagreements_finishes_and_Japanese_labels(self):
        x=record();v=r.analyze([x],official(x));pair=v['pair_disagreements']['pedigree45_vs_control39']
        self.assertEqual(pair['first_finishes_ahead'],1)
        self.assertEqual(pair['first_top_win_count'],1);self.assertEqual(pair['second_top_win_count'],0)
        self.assertEqual(v['details'][0]['models']['control39']['finish'],2)
        self.assertEqual(r.LABELS['pedigree45'],'血統45列')

    def test_no_report_adoption_or_formal_evaluation(self):
        x=record();v=r.analyze([x],official(x))
        self.assertIs(v['formal_21_week_included'],False)
        self.assertIs(v['H_Stage7_accessed'],False)
        self.assertEqual(v['new_fits'],0)
        self.assertEqual(v['scopes']['2026-10-24|05']['P1_races'],1)
        self.assertEqual(v['scopes']['2026-10-25|05']['P1_races'],0)

    def test_result_digit_width_not_guessed(self):
        x=record();o=official(x);o['SE'][tuple(x['race_key'])][1]['finish_position']='1'
        self.assertEqual(r.analyze([x],o)['P1_eligible_races'],0)

    def test_Japanese_HTML_offline_and_escaped_data(self):
        from jra_racing_ai.pedigree_replay_report import save
        x=record();o=official(x);o['SE'][tuple(x['race_key'])][1]['horse_name']='</script><script>unexpected()</script>'
        v=r.analyze([x],o);v.update(model_ids=r.frozen.MODEL_IDS,temperatures=r.frozen.TEMPERATURE,
            caveats=['開催後再生'],created_at='2026-10-26T06:00:00+09:00',artifact_id='synthetic')
        project=Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=project) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(project))
            path=Path(temp);save(path,v);text=(path/'比較レポート.html').read_text(encoding='utf-8')
            self.assertIn('血統45列',text);self.assertIn('connect-src \'none\'',text)
            self.assertNotIn('</script><script>unexpected()',text)
            self.assertNotIn('https://',text);self.assertNotIn('__REVIEW_DATA__',text)
            self.assertIn('公式着順',(path/'レース別本命比較.local.csv').read_text(encoding='utf-8-sig'))
            self.assertIn('280円',(path/'比較レポート.md').read_text(encoding='utf-8'))

    def test_pending_profit_is_not_zero_in_chart_or_summary(self):
        from jra_racing_ai.pedigree_replay_report import money_svg
        x=record();o=official(x);o['payouts']={};v=r.analyze([x],o)
        self.assertIn('未精算',money_svg(v['periods']['all']))
        for n in r.NAMES:self.assertIsNone(v['finance_P3'][n]['return_rate_percent'])


if __name__=='__main__':unittest.main()
