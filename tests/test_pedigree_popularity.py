"""Fictional odds and results only; no fitting, DB access, or protected inputs."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from test_pedigree_replay_review import record,official
from jra_racing_ai import pedigree_replay_review as review, pedigree_popularity as pop
from jra_racing_ai.ml.safety import MLFailure


def fixture():
    rec=record();rec['quote_T10']['odds']={'01':4.,'02':2.,'03':8.}
    return rec,review.analyze([rec],official(rec))


class PopularityTests(unittest.TestCase):
    def test_competition_ties_skip_rank(self):
        ranks,ties=pop.competition_ranks({'01':2.,'02':'2.00','03':4.,'04':8.},['01','02','03','04'])
        self.assertEqual(ranks,{'01':1,'02':1,'03':3,'04':4})
        self.assertTrue(ties['01']);self.assertTrue(ties['02']);self.assertFalse(ties['03'])

    def test_multiple_tie_groups_use_strictly_lower_prices(self):
        x={'01':2,'02':2,'03':3,'04':3,'05':8}
        self.assertEqual(pop.competition_ranks(x,list(x))[0],{'01':1,'02':1,'03':3,'04':3,'05':5})

    def test_bad_or_incomplete_odds_not_imputed(self):
        for price in (None,0,-2,float('nan'),float('inf'),'---',True):
            self.assertIsNone(pop.competition_ranks({'01':price,'02':2},['01','02']))
        self.assertIsNone(pop.competition_ranks({'01':2},['01','02']))
        self.assertIsNone(pop.competition_ranks({'01':2,'02':3},['01','01']))

    def test_popularity_band_boundaries(self):
        for k,value in ((1,'fav1'),(2,'fav2_3'),(3,'fav2_3'),(4,'fav4_6'),(6,'fav4_6'),(7,'fav7plus'),(28,'fav7plus'),(None,'unknown'),(0,'unknown')):
            self.assertEqual(pop.band(k),value)

    def test_model_specific_top_horse_uses_its_own_price(self):
        rec,base=fixture();updated=pop.add(base,[rec]);d=updated['details'][0]
        self.assertEqual(d['models']['control39']['T10_popularity_rank'],2)
        self.assertEqual(d['models']['pedigree45']['T10_popularity_rank'],1)
        self.assertEqual(d['models']['control39']['T10_win_odds'],4.)
        self.assertEqual(d['models']['pedigree45']['T10_win_odds'],2.)

    def test_old_metrics_predictions_cohorts_and_source_unchanged(self):
        rec,base=fixture();a=deepcopy(base);b=deepcopy(rec)
        with patch('lightgbm.train',side_effect=AssertionError('no_fit')):
            updated=pop.add(base,[rec])
        self.assertEqual(base,a);self.assertEqual(rec,b)
        self.assertTrue(pop.unchanged_original(base,updated))
        self.assertEqual(updated['periods'],base['periods'])

    def test_distribution_mean_and_share(self):
        rec,base=fixture();v=pop.add(base,[rec])['popularity']['scopes']['all|all']['models']
        self.assertEqual(v['control39']['mean_popularity_rank'],2.)
        self.assertEqual(v['pedigree45']['mean_popularity_rank'],1.)
        self.assertEqual(sum(b['count'] for b in v['control39']['bands']),1)
        self.assertEqual(sum(b['share_percent'] for b in v['control39']['bands']),100.)

    def test_P3_distribution_includes_abnormal_race_not_P2_only(self):
        rec=record();o=official(rec);o['SE'][tuple(rec['race_key'])][2].update(anomaly='4',finish_position='00')
        base=review.analyze([rec],o);v=pop.add(base,[rec])['popularity']['scopes']['all|all']
        self.assertEqual(base['P2_eligible_races'],0);self.assertEqual(v['P3_races'],1)
        self.assertEqual(v['models']['pedigree45']['known_popularity_count'],1)

    def test_missing_or_stale_quotes_show_unknown(self):
        for change in ('missing','stale'):
            rec=record()
            if change=='missing':rec['quote_T10']=None
            else:rec['quote_T10']['published_at']='2026-10-24T09:40:00+09:00'
            base=review.analyze([rec],official(rec));v=pop.add(base,[rec])
            for m in v['details'][0]['models'].values():
                self.assertIsNone(m['T10_popularity_rank']);self.assertIsNone(m['T10_win_odds'])
                self.assertEqual(m['T10_popularity_band'],'unknown')
            self.assertEqual(v['popularity']['scopes']['all|all']['P3_races'],0)

    def test_P3_mismatch_stops_instead_of_changing_cohort(self):
        rec,base=fixture();base['details'][0]['P3']=False
        with self.assertRaises(MLFailure):pop.add(base,[rec])

    def test_saved_top_probability_mismatch_stops(self):
        rec,base=fixture();base['details'][0]['models']['control39']['probability']=.9
        with self.assertRaises(MLFailure):pop.add(base,[rec])

    def test_official_payout_not_display_price_for_ROI(self):
        rec,base=fixture();m=pop.add(base,[rec])['popularity']['scopes']['all|all']['models']['pedigree45']
        b=m['bands'][0];self.assertEqual(b['win_count'],1);self.assertEqual(b['hit_rate_percent'],100.)
        self.assertEqual(b['settled_subset']['payout_yen'],280)
        self.assertEqual(b['settled_subset']['return_rate_percent'],280.)

    def test_unknown_flash_result_not_counted_as_loss(self):
        rec=record();o=official(rec);o['payouts']={}
        for row in o['SE'][tuple(rec['race_key'])]:row['status']='6'
        base=review.analyze([rec],o);m=pop.add(base,[rec])['popularity']['scopes']['all|all']['models']['pedigree45']
        b=next(x for x in m['bands'] if x['count'])
        self.assertEqual(b['known_result_count'],0);self.assertIsNone(b['hit_rate_percent'])
        self.assertEqual(b['finance']['unsettled_count'],1)
        self.assertIsNone(b['finance']['return_rate_percent'])

    def test_known_finish_without_payout_and_refund_denominators(self):
        rec,base=fixture();d=base['details'][0];d['models']['pedigree45']['ticket']={'status':'unsettled'}
        self.assertTrue(pop.known_hit(d,d['models']['pedigree45']))
        d['models']['pedigree45']['anomaly']='1'
        self.assertIsNone(pop.known_hit(d,d['models']['pedigree45']))

    def test_small_bands_and_empty_scopes_are_explicit(self):
        rec,base=fixture();v=pop.add(base,[rec])['popularity']['scopes']
        active=next(b for b in v['all|all']['models']['control39']['bands'] if b['count'])
        self.assertTrue(active['small_sample']);self.assertIn('少数例',active['note'])
        empty=v['2026-10-25|05']['models']['control39']
        self.assertIsNone(empty['mean_popularity_rank'])
        self.assertTrue(all(b['share_percent'] is None for b in empty['bands']))

    def test_Japanese_popularity_graph_and_CSV_preserve_legacy_tables(self):
        from jra_racing_ai.pedigree_replay_report import save
        rec,base=fixture();base.update(model_ids=review.frozen.MODEL_IDS,temperatures=review.frozen.TEMPERATURE,
            caveats=['開催後再生'],created_at='2026-10-26T14:00:00+09:00',artifact_id='synthetic')
        updated=pop.add(base,[rec]);project=Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=project) as temp:
            first=Path(temp)/'old';second=Path(temp)/'new';first.mkdir();second.mkdir()
            save(first,deepcopy(base));save(second,updated)
            for name in ('総合性能比較.csv','血統追加の差.csv','本命単勝100円の仮想収益.csv','総合精度比較.svg','市場との共通比較.svg'):
                self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
            self.assertIn('平均参考人気',(second/'本命のT10参考人気分布.svg').read_text(encoding='utf-8'))
            self.assertIn('T10参考人気順位',(second/'レース別本命比較.local.csv').read_text(encoding='utf-8-sig'))
            self.assertTrue((second/'T10人気帯別成績.csv').is_file())


if __name__=='__main__':unittest.main()
