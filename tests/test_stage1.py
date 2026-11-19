"""Fictional Stage 1 inputs; the suite blocks sockets and real DB drivers."""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai import stage1, stage1_analysis as a
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.ml import market
from jra_racing_ai.win_sim import core


RACE = ('2025', '0301', '05', '01', '01', '01')
ROSTER = ('01', '02')


def row(clock='03011448', *, prices=('0020', '0040'), table='apd_sokuho_o1', category='1'):
    return {'race_key': RACE, 'source_table': table, 'record_type': 'O1', 'status': category,
            'published': clock, 'sale': '7', 'payload':
            ''.join(f'{i:02d}'+p+f'{i:02d}' for i, p in enumerate(prices, 1))+' '*8*(28-len(prices)),
            'oversized': False}


def schedule():
    return core.Schedule(RACE, datetime(2025, 3, 1, 15, tzinfo=core.JST), None,
                         ROSTER, True, 'reviewed_historical_schedule')


def ra(**changes):
    return dict({'record_type': 'RA', 'status': '7', 'start': '1500', 'previous_start': '0000',
                 'oversized': False}, **changes)


def example():
    quote = market.MarketRace(RACE, {'01': core.decimal('2'), '02': core.decimal('4')},
                             'apd_sokuho_o1', '1', 'pre_race_publication_time',
                             datetime(2025, 3, 1, 14, 48, tzinfo=core.JST), 12)
    return [core.PredictionRace(RACE, {'01': .1, '02': .9})], [quote], {RACE: '01'}


class ScheduleTests(unittest.TestCase):
    def test_final_RA_is_explicitly_a_reference_without_observation_proof(self):
        value, reason = a.reference_schedule(RACE, [ra()], ROSTER)
        self.assertIsNone(reason)
        self.assertIsNone(value.observed_at)
        self.assertEqual(value.availability_basis, 'reviewed_historical_schedule')

    def test_changed_start_is_not_backfilled(self):
        self.assertEqual(a.reference_schedule(RACE, [ra(previous_start='1455')], ROSTER)[1],
                         'schedule_changed_asof_unavailable')

    def test_missing_schedule_is_excluded(self):
        self.assertEqual(a.reference_schedule(RACE, [], ROSTER)[1], 'schedule_unavailable')

    def test_multiple_versions_are_not_arbitrarily_selected(self):
        self.assertEqual(a.reference_schedule(RACE, [ra(), ra()], ROSTER)[1], 'schedule_conflict')

    def test_zero_or_invalid_time_is_rejected(self):
        for start in ('0000', '2561', '', None):
            self.assertIsNone(a.reference_schedule(RACE, [ra(start=start)], ROSTER)[0])

    def test_cancelled_unknown_or_oversized_RA_is_rejected(self):
        for value in (ra(status='9'), ra(status='X'), ra(oversized=True)):
            self.assertIsNone(a.reference_schedule(RACE, [value], ROSTER)[0])


class SnapshotTests(unittest.TestCase):
    def select(self, rows, **kw):
        return a.select_quote(RACE, rows, ROSTER, schedule(), **kw)

    def test_correct_latest_pre_cutoff_quote_and_scale(self):
        q, why = self.select([row(), row('03011449')])
        self.assertIsNone(why)
        self.assertEqual(q.published_at.minute, 49)
        self.assertEqual(q.odds['01'], core.decimal('2.0'))

    def test_future_quote_never_changes_choice(self):
        q, _ = self.select([row(), row('03011451', prices=('0010', '9998'))])
        self.assertEqual(q.published_at.minute, 48)
        self.assertEqual(q.odds['01'], core.decimal('2'))

    def test_same_minute_as_cutoff_is_not_assumed_available(self):
        q, _ = self.select([row(), row('03011450')])
        self.assertEqual(q.published_at.minute, 48)

    def test_only_boundary_minute_reports_precision_reason(self):
        q, why = self.select([row('03011450')])
        self.assertIsNone(q)
        self.assertEqual(why, 'odds_time_precision')

    def test_final_price_never_replaces_pre_race_price(self):
        final = row('00000000', category='5', prices=('0010', '9998'), table='jvd_o1')
        q, _ = self.select([row(), final])
        self.assertEqual(q.odds['01'], core.decimal('2'))

    def test_final_only_is_unavailable(self):
        self.assertIsNone(self.select([row(category='5')])[0])

    def test_latest_invalid_snapshot_does_not_fall_back(self):
        q, why = self.select([row(), dict(row('03011449'), payload='bad')])
        self.assertIsNone(q)
        self.assertEqual(why, 'odds_invalid')

    def test_duplicates_in_one_table_are_rejected(self):
        self.assertEqual(self.select([row(), row()])[1], 'odds_conflict')

    def test_identical_cross_table_copies_are_deduplicated(self):
        q, why = self.select([row(), row(table='jvd_o1')])
        self.assertIsNone(why)
        self.assertEqual(q.source, 'apd_sokuho_o1')

    def test_cancelled_horse_inside_roster_excludes_race(self):
        self.assertIsNone(self.select([row(prices=('0020', '----'))])[0])

    def test_cancelled_horse_outside_roster_is_masked(self):
        value = row()
        value['payload'] = value['payload'][:16]+'03------'+value['payload'][24:]
        q, why = self.select([value])
        self.assertIsNone(why)
        self.assertEqual(q.cancelled_outside_roster, 1)

    def test_missing_horse_is_not_filled(self):
        self.assertIsNone(self.select([row(prices=('0020',))])[0])

    def test_stale_price_excluded_without_fallback_or_future(self):
        self.assertEqual(self.select([row('03011444')])[1], 'odds_stale')

    def test_unparseable_time_excludes_race(self):
        self.assertIsNone(self.select([row('not_time')])[0])

    def test_delete_or_unknown_status_does_not_silently_disappear(self):
        for status in ('0', '9', 'X'):
            self.assertEqual(self.select([row(), row(category=status)])[1], 'odds_unverified')

    def test_all_four_fixed_cutoffs_select_only_earlier_minutes(self):
        for offset in a.OFFSETS:
            minute = 60-offset-1
            q, why = self.select([row(f'030114{minute:02d}')], minutes=offset)
            self.assertIsNone(why)
            self.assertEqual(q.minutes_before_start, offset+1)


class ProbabilityTests(unittest.TestCase):
    def test_AI_T075_is_applied_once(self):
        value = a.normalized_ai([core.PredictionRace(RACE, {'01': .2, '02': .1})])[0]
        expected = .2**(1/.75)/(.2**(1/.75)+.1**(1/.75))
        self.assertAlmostEqual(value.probabilities['01'], expected)
        variants = a.variants([value], example()[1], {'market_temperature': 1.,
            'ai_weights': dict(linear_raw=0., linear_recalibrated=0., geometric_recalibrated=0.)})
        self.assertEqual(variants['best242'][0].probabilities, value.probabilities)

    def test_inverse_odds_are_normalized(self):
        p = a.market_predictions(example()[1])[0].probabilities
        self.assertAlmostEqual(p['01'], 2/3)
        self.assertAlmostEqual(sum(p.values()), 1.)

    def test_final_odds_cannot_be_passed_into_market_diagnostic(self):
        final = market.MarketRace(RACE, {'01': 2., '02': 4.}, 'jvd_o1', '5', 'final_category_5')
        with self.assertRaises(core.SimulationError):
            a.market_predictions([final])

    def test_linear_endpoints_match_market_and_AI(self):
        ai, q, _ = example()
        m = a.market_predictions(q)
        self.assertEqual(a.blend(ai, m, 0)[0].probabilities, m[0].probabilities)
        self.assertEqual(a.blend(ai, m, 1)[0].probabilities, ai[0].probabilities)

    def test_geometric_blend_positive_and_normalized(self):
        ai, q, _ = example()
        p = a.blend(ai, a.market_predictions(q), .5, geometric=True)[0].probabilities
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertTrue(all(v > 0 for v in p.values()))

    def test_missing_or_different_runner_set_fails(self):
        ai, _, _ = example()
        with self.assertRaises(core.SimulationError):
            a.blend(ai, [core.PredictionRace(RACE, {'01': .5, '03': .5})], .5)

    def test_weight_outside_declared_grid_is_rejected(self):
        ai, q, _ = example()
        for value in (-1, .123, 2, True):
            with self.assertRaises(core.SimulationError):
                a.blend(ai, a.market_predictions(q), value)

    def test_selection_uses_only_development_labels(self):
        ai, q, winners = example()
        selected = a.fit_development(ai, q, winners, development_end='2025-05-31', evaluation_start='2025-06-01')
        altered = dict(winners, **{})
        altered[('2025', '0601', '05', '01', '01', '01')] = '02'
        self.assertEqual(selected, a.fit_development(ai, q, altered,
            development_end='2025-05-31', evaluation_start='2025-06-01'))
        self.assertEqual(selected['ai_weights']['linear_recalibrated'], 0.)

    def test_selection_rejects_evaluation_period(self):
        ai, q, winners = example()
        with self.assertRaises(core.SimulationError):
            a.fit_development(ai, q, winners, development_end='2025-05-31', evaluation_start='2025-05-01')

    def test_selection_rejects_earlier_training_data(self):
        ai, q, winners = example()
        with self.assertRaises(core.SimulationError):
            a.fit_development(ai, q, winners, development_end='2025-05-31',
                              development_start='2025-03-02', evaluation_start='2025-06-01')

    def test_disagreement_uses_lowest_horse_number_tie_rule(self):
        ai = [core.PredictionRace(RACE, {'01': .5, '02': .5})]
        markets = [core.PredictionRace(RACE, {'01': .4, '02': .6})]
        result = a.disagreement(ai, markets, {RACE: '01'})
        self.assertEqual(result['disagreement']['ai_hits'], 1)
        self.assertEqual(result['disagreement']['market_hits'], 0)

    def test_agreement_hits_always_match(self):
        ai, _, winners = example()
        result = a.disagreement(ai, ai, winners)
        self.assertEqual(result['agreement']['ai_hits'], result['agreement']['market_hits'])

    def test_paired_bootstrap_identical_predictors_zero_difference(self):
        ai, _, winners = example()
        result = a.paired(ai, ai, winners)
        self.assertEqual(result['log_loss']['ci95_low'], 0.)
        self.assertEqual(result['race_log_loss']['ci95_high'], 0.)


class FinanceAndSafetyTests(unittest.TestCase):
    def test_HR_SE_disagreement_is_held_without_changing_labels(self):
        original = core.Outcome(RACE, 'normal_final', {'02': 230}, True, True)
        labels = {RACE: '01'}
        checked, count = a.match_outcomes({RACE: original}, labels)
        self.assertEqual(count, 1)
        self.assertEqual(checked[RACE].status, 'pending')
        self.assertEqual(labels[RACE], '01')
        self.assertEqual(original.payouts_yen_per_100, {'02': 230})

    def test_consistent_HR_is_reused(self):
        original = core.Outcome(RACE, 'normal_final', {'01': 230}, True, True)
        checked, count = a.match_outcomes({RACE: original}, {RACE: '01'})
        self.assertEqual(count, 0)
        self.assertIs(checked[RACE], original)

    def simulate(self, outcome=None, p=.7, prices=None):
        quote = example()[1][0]
        return core.run_simulation([core.PredictionRace(RACE, {'01': p, '02': 1-p})], [schedule()],
            [core.Snapshot(RACE, quote.published_at, None, prices or quote.odds, verified=True, publication_precision='minute')],
            [outcome] if outcome else [], policy=core.WinPolicy(probability_method='identity',
                timing_basis='jra_publication_time', evaluation_start='2025-03-01', evaluation_end='2025-05-31',
                model_fit_end='2024-12-28'))

    def test_settlement_uses_HR_not_decision_odds(self):
        ledger = self.simulate(core.Outcome(RACE, 'normal_final', {'01': 230}, True, True))
        self.assertEqual(ledger['summary']['ai']['payout_yen'], 230)
        self.assertEqual(ledger['summary']['ai']['net_profit_yen'], 130)

    def test_missing_payout_is_pending_not_zero(self):
        result = self.simulate()['summary']['ai']
        self.assertEqual(result['unsettled_count'], 1)
        self.assertIsNone(result['payout_yen'])
        self.assertIsNone(result['return_rate_percent'])

    def test_zero_purchases_have_no_ROI(self):
        # A normal overround market: inverse prices sum to 7/6, so all raw-market EVs are negative.
        f = self.simulate(p=4/7, prices={'01': 1.5, '02': 2.})['summary']['ai']
        self.assertEqual(f['bet_count'], 0)
        self.assertIsNone(f['return_rate_percent'])

    def test_selected_bands_cover_each_purchase_once(self):
        ledger = self.simulate(core.Outcome(RACE, 'normal_final', {'01': 230}, True, True))
        for values in stage1.selected_bands(ledger['rows']).values():
            self.assertEqual(sum(v['bet_count'] for v in values), 1)

    def test_future_final_odds_drift_does_not_modify_prediction(self):
        ai, q, winners = example()
        before = dict(ai[0].probabilities)
        final = market.MarketRace(RACE, {'01': core.decimal('1'), '02': core.decimal('8')},
                                  'jvd_o1', '5', 'final_category_5')
        self.assertEqual(a.price_drift(q, {RACE: final})['horse_quote_count'], 2)
        self.assertEqual(ai[0].probabilities, before)

    def test_writing_SQL_is_rejected(self):
        for sql in ('DELETE FROM public.jvd_o1', 'SELECT 1; DROP TABLE x', 'UPDATE public.jvd_ra SET x=1'):
            with self.assertRaises(SafeAuditError):
                validate_audit_sql(sql)

    def test_errors_never_echo_exception_payload(self):
        output = io.StringIO()
        with patch.object(stage1.safety, 'new_artifact', side_effect=ValueError('SYNTHETIC_SECRET')), redirect_stdout(output):
            self.assertEqual(stage1.main(['--stage', 'inventory']), 2)
        self.assertNotIn('SYNTHETIC_SECRET', output.getvalue())
        self.assertEqual(json.loads(output.getvalue())['reason'], 'internal_error')

    def test_unsafe_work_id_rejected_without_DB(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(stage1.main(['--work-id', '../bad']), 2)
        self.assertEqual(json.loads(output.getvalue())['reason'], 'invalid_input')

    def test_sync_output_path_rejected(self):
        with self.assertRaises(stage1.safety.MLFailure):
            stage1.safety.local_path(Path(r'C:\Users\acbd5\OneDrive\unsafe.json'), root=stage1.ROOT)


if __name__ == '__main__':
    unittest.main()
