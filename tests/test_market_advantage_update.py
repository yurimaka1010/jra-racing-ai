"""Amendment-2 selection, official settlement and weekly ratio intervals."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from unittest.mock import patch
import unittest
import numpy as np

from jra_racing_ai import market_advantage as old
from jra_racing_ai import market_advantage_update as m
from jra_racing_ai.stage5.metrics import WeekBootstrap
from jra_racing_ai.win_sim import core

RACE=('2023','0107','06','01','02','01')


def decision():
    return dict(race_key=list(RACE),favorite='01',selected_horses=['01','02'],
                runners=[dict(horse_number='01',probability=.6,market_probability=.5,decision_odds=3.,reference_popularity=1),
                         dict(horse_number='02',probability=.4,market_probability=.5,decision_odds=3.,reference_popularity=1)])


def ticket(race,payout,status='settled'):
    return dict(race_key=list(race),settlement=dict(status=status,reason='test',stake_yen=100,
                payout_yen=payout,hit=bool(payout) if payout is not None else None,net_profit_yen=payout-100 if payout is not None else None))


class MarketAmendmentTests(unittest.TestCase):
    def test_buys_every_value_horse_not_only_max_ev(self):
        self.assertEqual(m.all_value_horses(decision()),['01','02'])

    def test_threshold_exactly_one_is_inclusive(self):
        row=decision();row['runners'][0].update(probability=.5,decision_odds=2.)
        row['runners'][1].update(probability=.5,decision_odds=1.9)
        self.assertEqual(m.all_value_horses(row),['01'])

    def test_postrace_information_cannot_change_purchase(self):
        row=decision();before=m.all_value_horses(row)
        row.update(winner='02',payout={'02':9999999},confirmed_odds={'01':1000,'02':1})
        self.assertEqual(m.all_value_horses(row),before)

    def test_favorite_one_ticket_per_condition_race(self):
        row=decision();outcome=core.Outcome(RACE,'normal_final',{'02':270},True,True)
        ai,favorite=m.settle_tickets([row],{RACE:outcome})
        self.assertEqual(len(ai),2);self.assertEqual(len(favorite),1)
        self.assertEqual(sum(t['settlement']['payout_yen'] for t in ai),270)
        self.assertEqual(favorite[0]['horse_number'],'01')

    def test_favorite_retained_even_if_ai_does_not_buy(self):
        row=decision();row['selected_horses']=[]
        ai,favorite=m.settle_tickets([row],{})
        self.assertEqual(ai,[]);self.assertEqual(len(favorite),1)

    def test_weekly_totals_ratio_not_mean_of_weekly_roi(self):
        second=('2023','0114','06','01','02','01')
        boot=WeekBootstrap([m.week(RACE),m.week(second)],sample_indices=[[0,0],[1,1],[0,1]])
        v=m.financial([ticket(RACE,200),ticket(RACE,200),ticket(second,0)],boot)
        self.assertAlmostEqual(v['return_rate_percent'],400/300*100)
        np.testing.assert_allclose(v['ROI95CI'],np.quantile([200.,0.,400/300*100],[.025,.975]))

    def test_pending_means_roi_and_ci_unavailable(self):
        boot=WeekBootstrap([m.week(RACE)],repetitions=20)
        v=m.financial([ticket(RACE,None,'pending')],boot)
        self.assertEqual(v['unsettled_count'],1)
        self.assertIsNone(v['return_rate_percent']);self.assertIsNone(v['ROI95CI'])

    def test_no_purchases_not_zero_return(self):
        boot=WeekBootstrap([m.week(RACE)],repetitions=20)
        v=m.financial([],boot)
        self.assertIsNone(v['return_rate_percent']);self.assertIsNone(v['ROI95CI'])
        self.assertEqual(v['zero_investment_replica_fraction'],1.)

    def test_closing_ratio_direction_and_bands(self):
        row=decision();row['runners'][1]['decision_odds']=20
        ai,_=m.settle_tickets([row],{})
        finals={RACE:{'odds':{'01':1.5,'02':26.},'category':'5'}}
        v=m.closing_drift(ai,finals)
        self.assertAlmostEqual(v['bands']['3〜10倍未満']['quantiles']['median'],.5)
        self.assertAlmostEqual(v['bands']['10〜30倍未満']['quantiles']['median'],1.3)
        self.assertEqual(m.all_value_horses(row),['01','02'])

    def test_missing_closing_odds_never_replace_pre_race_odds(self):
        row=decision();before=deepcopy(row);ai,_=m.settle_tickets([row],{})
        v=m.closing_drift(ai,{})
        self.assertEqual(v['bands']['3〜10倍未満']['missing'],2)
        self.assertIsNone(v['bands']['3〜10倍未満']['quantiles'])
        self.assertEqual(row,before)

    def test_quality_matches_existing_binary_ll(self):
        row=decision();v=m.quality([row],{RACE:'01'})
        self.assertAlmostEqual(v['AI']['log_loss'],-np.log(.6))
        self.assertAlmostEqual(v['market']['log_loss'],-np.log(.5))

    def test_final_collector_reuses_decoder_and_rejects_category_three(self):
        for category in ('3','5'):
            with self.subTest(category=category),TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'purchase_commit.json').write_text('{}',encoding='utf-8')
                payload='01002001'+'02003002'+' '*208
                raw=[dict(race_key=RACE,record_type='O1',status=category,sale='7',payload=payload,oversized=False)]
                @contextmanager
                def session():yield object()
                with patch.object(m,'folder',return_value=root),patch.object(m,'decisions',return_value=({},[decision()])),\
                     patch.object(m.publication,'_columns',return_value={}),patch.object(m.local,'_rows',return_value=raw):
                    result=m.collect_final('synthetic',session)
                v=old.read(root/'final-odds/0000.local.json')['records'][0]
                self.assertEqual(result['races'],1)
                if category=='3':
                    self.assertIsNone(v['odds']);self.assertEqual(v['reason'],'confirmed_category4_5_unavailable')
                else:self.assertEqual(v['odds'],{'01':2.,'02':3.})


if __name__=='__main__':unittest.main()
