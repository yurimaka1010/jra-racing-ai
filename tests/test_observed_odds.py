from dataclasses import replace
from datetime import timedelta
import unittest
from jra_racing_ai.win_sim import core,observed
from test_win_sim_v02 import example,policy

class ObservedOddsTests(unittest.TestCase):
    def test_future_observation_is_never_used(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        snap=replace(o,observed_at=cutoff+timedelta(seconds=1))
        value,reason=observed.fresh_snapshot(p.race,[snap],cutoff,s.horses)
        self.assertIsNone(value); self.assertEqual(reason,'odds_after_decision')

    def test_stale_latest_snapshot_is_not_replaced_with_old_one(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        value,reason=observed.fresh_snapshot(p.race,[replace(o,published_at=cutoff-timedelta(minutes=5))],cutoff,s.horses)
        self.assertIsNone(value); self.assertEqual(reason,'odds_stale')

    def test_minute_boundary_does_not_invent_publication_seconds(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        snap=replace(o,published_at=cutoff,observed_at=cutoff,publication_precision='minute')
        self.assertEqual(observed.fresh_snapshot(p.race,[snap],cutoff,s.horses)[1],'odds_time_precision')

    def test_decision_distinguishes_top1_and_max_ev_and_never_places_bet(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        row,reason=observed.decision(p,s,[o],policy(),recorded_at=cutoff)
        self.assertIsNone(reason); self.assertEqual(row['prediction_top_horse'],'01')
        self.assertEqual(row['ai_horse'],'03'); self.assertFalse(row['actual_bet_placed'])
        self.assertEqual(row['decision_class'],'virtual_purchase')

    def test_not_due_is_not_saved_as_a_skip(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        self.assertEqual(observed.decision(p,s,[o],policy(),recorded_at=cutoff-timedelta(seconds=1)),(None,'decision_not_due'))

    def test_unusable_latest_event_blocks_fallback_to_older_odds(self):
        p,s,o,_=example();cutoff=s.scheduled_start-timedelta(minutes=10)
        invalid=replace(o,observed_at=cutoff-timedelta(seconds=1),reason='unsupported_status')
        self.assertEqual(observed.fresh_snapshot(p.race,[o,invalid],cutoff,s.horses)[1],'unsupported_status')

    def test_data_failure_and_ev_skip_are_different(self):
        p,s,o,_=example(); cutoff=s.scheduled_start-timedelta(minutes=10)
        row,_=observed.decision(p,s,[],policy(),recorded_at=cutoff)
        self.assertEqual(row['decision_class'],'data_unavailable')
        row,_=observed.decision(p,s,[o],policy(threshold='5'),recorded_at=cutoff)
        self.assertEqual(row['decision_class'],'conditions_not_met')

    def test_drawdown_and_large_hit_dependency(self):
        rows=[{'scheduled_start':str(i),'ai':{'status':'settled','payout_yen':p}} for i,p in enumerate((1000,0,0))]
        result=observed.risk_metrics(rows)
        self.assertEqual(result['max_drawdown_yen'],200)
        self.assertEqual(result['largest_payout_share_percent'],100)
        self.assertEqual(result['net_profit_without_largest_win_yen'],-200)
        rows.append({'ai':{'status':'pending','payout_yen':None}})
        self.assertIsNone(observed.risk_metrics(rows)['max_drawdown_yen'])
