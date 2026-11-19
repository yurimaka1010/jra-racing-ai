"""Fictional-only checks for the independent, zero-fit market exploration."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import math
import tempfile
import unittest

from jra_racing_ai import market_advantage as m
from jra_racing_ai.win_sim import core

RACE = ('2023', '0107', '06', '01', '02', '01')


def fixture():
    features = {h: dict(start_time='1200', previous_start='', past_starts='4') for h in ('01', '02')}
    def quote(clock):
        return dict(race_key=list(RACE), odds={'01': 3., '02': 3.}, source='apd_sokuho_o1',
                    category='1', basis='pre_race_publication_time',
                    published_at=f'2023-01-07T{clock}:00+09:00')
    packet = dict(scheduled_start='2023-01-07T12:00:00+09:00',
                  quotes={'10': quote('11:49'), '30': quote('11:29')}, failures={})
    return {'01': .6, '02': .4}, features, packet


class MarketAdvantageTests(unittest.TestCase):
    def decision(self):
        p, features, packet = fixture()
        result, reason = m.decide(RACE, p, features, packet)
        self.assertIsNone(reason)
        return result

    def test_ev_and_fixed_candidate(self):
        row = self.decision()
        self.assertEqual(row['selected'], '01')
        self.assertAlmostEqual(row['candidate_ev'], .8)
        self.assertEqual(row['conditions']['C1'], True)
        self.assertEqual(row['conditions']['C2'], False)

    def test_future_minute_boundary_not_accepted(self):
        p, f, packet = fixture()
        packet['quotes']['10']['published_at'] = '2023-01-07T11:50:00+09:00'
        row, reason = m.decide(RACE, p, f, packet)
        self.assertIsNone(row)
        self.assertEqual(reason, 'odds_time_precision')

    def test_stale_quote_no_substitution(self):
        p, f, packet = fixture()
        packet['quotes']['10']['published_at'] = '2023-01-07T11:44:00+09:00'
        self.assertEqual(m.decide(RACE, p, f, packet)[1], 'odds_stale')
        packet['quotes']['10']['published_at'] = '2023-01-07T11:45:00+09:00'
        self.assertIsNone(m.decide(RACE, p, f, packet)[1])

    def test_final_quote_never_used(self):
        p, f, packet = fixture()
        packet['quotes']['10']['category'] = '4'
        self.assertEqual(m.decide(RACE, p, f, packet)[1], 'odds_unverified')
        packet['quotes']['10'] = None
        self.assertEqual(m.decide(RACE, p, f, packet)[1], 'odds_unavailable')

    def test_incomplete_roster_never_used(self):
        p, f, packet = fixture()
        del packet['quotes']['10']['odds']['02']
        self.assertEqual(m.decide(RACE, p, f, packet)[1], 'odds_incomplete')

    def test_changed_schedule_excluded(self):
        p, f, packet = fixture()
        for row in f.values():
            row['previous_start'] = '1155'
        self.assertEqual(m.decide(RACE, p, f, packet)[1], 'schedule_changed_asof_unavailable')

    def test_missing_optional_features_keep_reference_population(self):
        p, f, packet = fixture()
        for row in f.values():
            row['past_starts'] = ''
        packet['quotes']['30'] = None
        row, reason = m.decide(RACE, p, f, packet)
        self.assertIsNone(reason)
        self.assertIsNone(row['conditions']['C4'])
        self.assertIsNone(row['conditions']['C5'])
        answered = m.answer([row], {RACE: '02'}, {})
        values = m.aggregate(answered)
        self.assertEqual(values['ALL']['target_races'], 1)
        self.assertEqual(values['C5']['target_races'], 0)
        self.assertEqual(values['C5']['unassessable_races'], 1)

    def test_popularity_competition_tie_and_tie_break(self):
        row = self.decision()
        self.assertEqual([h['reference_popularity'] for h in row['runners']], [1, 1])
        self.assertEqual(row['favorite'], '01')
        p, f, packet = fixture()
        p = {'01': .5, '02': .5}
        self.assertEqual(m.decide(RACE, p, f, packet)[0]['candidate'], '01')

    def test_all_horses_choose_before_condition(self):
        p, f, packet = fixture()
        f['01']['past_starts'] = '0'
        row, _ = m.decide(RACE, p, f, packet)
        self.assertEqual(row['candidate'], '01')
        self.assertIs(row['conditions']['C4'], False)
        self.assertEqual(row['selected'], '01')  # No switching to eligible 02.

    def test_threshold_inclusive_and_no_bet(self):
        p, f, packet = fixture()
        packet['quotes']['10']['odds'] = {'01': 3., '02': 1.}
        row, _ = m.decide(RACE, {'01': .35, '02': .65}, f, packet)
        self.assertEqual(row['selected'], '01')
        row, _ = m.decide(RACE, {'01': .34, '02': .66}, f, packet)
        self.assertIsNone(row['selected'])
        answered = m.answer([row], {RACE: '01'}, {RACE: core.Outcome(RACE, 'normal_final', {'01': 220}, True, True)})
        summary = m.describe(answered)
        self.assertIsNone(summary['financial']['return_rate_percent'])
        self.assertEqual(summary['favorite_same_purchases']['bet_count'], 0)

    def test_outcome_cannot_change_choice(self):
        p, f, packet = fixture()
        a, _ = m.decide(RACE, p, f, packet)
        packet['outcome'] = dict(payouts_yen_per_100={'02': 9999900}, winner='02')
        b, _ = m.decide(RACE, p, f, packet)
        self.assertEqual(a, b)
        self.assertNotIn('winner', a)

    def test_hr_amount_not_displayed_price(self):
        row = self.decision()
        outcome = core.Outcome(RACE, 'normal_final', {'01': 280}, True, True)
        answered = m.answer([row], {RACE: '01'}, {RACE: outcome})
        summary = m.describe(answered)
        self.assertEqual(summary['financial']['payout_yen'], 280)
        self.assertEqual(summary['financial']['net_profit_yen'], 180)
        self.assertEqual(summary['financial']['return_rate_percent'], 280)

    def test_pending_not_a_loss_or_zero_roi(self):
        values = m.describe(m.answer([self.decision()], {RACE: '01'}, {}))
        self.assertEqual(values['financial']['unsettled_count'], 1)
        self.assertIsNone(values['financial']['payout_yen'])
        self.assertIsNone(values['financial']['return_rate_percent'])

    def test_metrics_same_horse_set_and_original_probabilities(self):
        values = m.describe(m.answer([self.decision()], {RACE: '01'}, {}))
        self.assertEqual(values['runners'], 2)
        self.assertAlmostEqual(values['scores']['AI']['log_loss'], -math.log(.6))
        self.assertAlmostEqual(values['scores']['market']['log_loss'], -math.log(.5))
        self.assertAlmostEqual(values['mean_probability_gap'], .1)

    def test_favorite_counterfactual_same_purchased_races(self):
        row = self.decision()
        no_bet = deepcopy(row); no_bet['selected'] = None
        rows = m.answer([row, no_bet], {RACE: '01'}, {RACE: core.Outcome(RACE, 'normal_final', {'01': 200}, True, True)})
        result = m.describe(rows)
        self.assertEqual(result['financial']['investment_yen'], 100)
        self.assertEqual(result['favorite_same_purchases']['investment_yen'], 100)

    def test_early_support_uses_normalized_probabilities(self):
        p, f, packet = fixture()
        packet['quotes']['30']['odds'] = {'01': 4., '02': 2.}
        row, _ = m.decide(RACE, p, f, packet)
        self.assertIs(row['conditions']['C5'], True)
        self.assertAlmostEqual(row['history_log_change'], math.log(.5/(1/3)))

    def test_scope_only_declared_development_periods(self):
        self.assertTrue(m.in_scope('2023-09-30'))
        self.assertFalse(m.in_scope('2023-10-01'))
        self.assertFalse(m.in_scope('2026-01-01'))
        self.assertFalse(m.in_scope('2026-10-24'))
        self.assertRaises(ValueError, m.race_key, 'incorrect')
        self.assertRaises(ValueError, m.run, '../protected')

    def test_csv_and_report_write_only_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.csv'
            m.csv_output(path, ['購入件数'], [{'購入件数': 1}])
            self.assertIn('購入件数', path.read_text(encoding='utf-8-sig'))
            self.assertRaises(FileExistsError, m.csv_output, path, ['購入件数'], [])


if __name__ == '__main__':
    unittest.main()
