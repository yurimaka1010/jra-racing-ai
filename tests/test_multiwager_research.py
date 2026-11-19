"""No real data, DB, fits, or network. Outcome and feature contracts only."""
from copy import deepcopy
import csv
from dataclasses import replace
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.multiwager import contracts as c, outcomes as o, models, runner, training
from jra_racing_ai.multiwager.scoring import Scores
from jra_racing_ai.multiwager.probabilities import plackett_luce


def record():
    return {'race_id': '2025030105010101', 'date': '2025-03-01', 'paid_positions': 2,
            'finishes': {'01': 1, '02': 2, '03': 3, '04': 4}}


def official(wager='place', payouts=None):
    return o.OfficialOutcome(record()['race_id'], record()['date'], wager, 'normal',
            {'01': 150, '02': 200} if payouts is None else payouts, 'a' * 64, True, True, 'JPY_PER_100', True)


def feature_frame():
    r = record(); rows = []
    for horse, finish in r['finishes'].items():
        d = dict.fromkeys(models.base_spec()['features'], None)
        d.update(entry_key=r['race_id'] + horse, date=r['date'], finish_position=finish,
                 actual_starters=4, place_paid_positions=2, official_label_status='verified_ordinary_HR',
                 official_place_label=int(finish <= 2))
        rows.append(d)
    return pd.DataFrame(rows)


class RankAndOfficialOutcomeTests(unittest.TestCase):
    def test_pay_regime_changes_place_only(self):
        r = record(); a, _ = o.rank_reference(r['finishes'], 2); b, _ = o.rank_reference(r['finishes'], 3)
        self.assertEqual(sum(a['place'].values()), 2); self.assertEqual(sum(b['place'].values()), 3)
        self.assertEqual(a['umaren'], b['umaren']); self.assertEqual(a['wide'], b['wide'])
        self.assertEqual(sum(a['umaren'].values()), 1); self.assertEqual(sum(a['wide'].values()), 3)

    def test_same_rank_two_or_three_is_not_all_top3_pairs(self):
        for finishes in ({'01': 1, '02': 2, '03': 2, '04': 4}, {'01': 1, '02': 2, '03': 3, '04': 3}):
            labels, why = o.rank_reference(finishes, 3)
            self.assertIsNone(labels); self.assertIn('needs_official', why)

    def test_no_pay_key_inference_from_headcount(self):
        for key in (None, '3', 0, 4, True):
            with self.assertRaises(MLFailure): o.rank_reference(record()['finishes'], key)

    def test_reference_lower_finish_tie_does_not_change_podium(self):
        finishes = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 4}
        labels, why = o.rank_reference(finishes, 3)
        self.assertIsNone(why); self.assertEqual(sum(labels['wide'].values()), 3)

    def test_official_pairs_can_represent_complex_tie_targets_without_rank_guess(self):
        event = official('wide', {'01-02': 150, '01-03': 200, '01-04': 300, '02-03': 400, '02-04': 500})
        labels, why = o.official_labels(event, list(record()['finishes']))
        self.assertIsNone(why); self.assertEqual(sum(labels.values()), 5)
        self.assertEqual(labels['03-04'], 0)

    def test_unverified_units_flags_or_revisions_never_produce_labels(self):
        for kwargs in ({'units': 'UNKNOWN'}, {'flags_verified': False}, {'revision_policy_verified': False}, {'complete_winning_set': False}):
            labels, why = o.official_labels(replace(official(), **kwargs), list(record()['finishes']))
            self.assertIsNone(labels); self.assertIn('unconfirmed', why)

    def test_specials_not_normal_losses_or_zero_payouts(self):
        for state in ('refund', 'not_offered', 'not_established', 'special_payment', 'cancelled', 'unconfirmed'):
            result = o.settle_ticket(replace(official(), state=state), list(record()['finishes']), '01')
            self.assertEqual(result['status'], 'pending'); self.assertIsNone(result['payout_yen'])

    def test_normal_single_and_pair_settlement_use_official_money(self):
        r = list(record()['finishes'])
        win = o.settle_ticket(official(), r, '01', 200)
        self.assertEqual((win['payout_yen'], win['net_profit_yen']), (300, 100))
        event = official('umaren', {'01-02': 850})
        self.assertEqual(o.settle_ticket(event, r, '01-02')['payout_yen'], 850)
        self.assertEqual(o.settle_ticket(event, r, '01-03')['net_profit_yen'], -100)

    def test_noncanonical_duplicate_or_wrong_kind_target_rejected(self):
        for target in ('02-01', '01-01', '00-01', '01'):
            with self.assertRaises(MLFailure): o.settle_ticket(official('wide', {'01-02': 150}), list(record()['finishes']), target)

    def test_invalid_money_roster_and_stake_rejected(self):
        for amount in (0, -1, True, '150', None):
            with self.assertRaises(MLFailure): o.validate_official(official(payouts={'01': amount}), list(record()['finishes']))
        for stake in (0, 50, -100, 100.0, True):
            with self.assertRaises(MLFailure): o.settle_ticket(official(), list(record()['finishes']), '01', stake)

    def test_official_details_not_in_repr_or_errors(self):
        self.assertEqual(repr(official()), 'OfficialOutcome(details=local_only)')
        with self.assertRaises(MLFailure) as caught: o.target_key(['SECRET_PASSWORD'])
        self.assertNotIn('SECRET', str(caught.exception))

    def test_sealed_and_stage7_periods_rejected(self):
        for d in ('2026-01-01', '2026-09-30', '2026-10-24'):
            with self.assertRaises(MLFailure): c.day(d)

    def test_fixed_ticket_profit_aggregates_by_wager_and_month(self):
        result = o.settle_ticket(official(), list(record()['finishes']), '01')
        summary = o.summarize_settlements([dict(result, date='2025-03-01', wager='place')])
        self.assertEqual(summary['groups']['place']['net_profit_yen'], 50)
        self.assertEqual(summary['groups']['2025-03/place']['return_rate_percent'], 150)

    def test_pending_ticket_keeps_total_profit_unknown(self):
        result = o.settle_ticket(replace(official(), flags_verified=False), list(record()['finishes']), '01')
        group = o.summarize_settlements([dict(result, date='2025-03-01', wager='place')])['groups']['place']
        self.assertEqual(group['unsettled'], 1)
        self.assertIsNone(group['payout_yen']); self.assertIsNone(group['net_profit_yen'])
        self.assertIsNone(group['return_rate_percent'])

    def test_no_tickets_does_not_mean_zero_percent_ROI(self):
        result = o.summarize_settlements([])
        self.assertEqual(result['ticket_count'], 0); self.assertIsNone(result['return_rate_percent_when_no_tickets'])


class ModelAndScoreContractTests(unittest.TestCase):
    def test_place_direct_is_marginal_not_forced_sum_three(self):
        raw = np.array([.8, .6, .3, .1]); p = models.place_probability(raw)
        np.testing.assert_array_equal(p, raw); self.assertAlmostEqual(sum(p), 1.8)
        self.assertNotAlmostEqual(sum(models.place_probability(raw, 2)), 3)

    def test_place_temperature_applied_once_on_logit(self):
        np.testing.assert_allclose(models.place_probability([.8, .2], 2), [2/3, 1/3])

    def test_invalid_calibration_parameters_cannot_produce_fake_half_probabilities(self):
        for t in (0, -1, float('inf'), float('nan'), True):
            for f in (models.place_probability, models.strengths):
                with self.subTest(temperature=t, function=f.__name__), self.assertRaises(MLFailure):
                    f([.3, .7], t)

    def test_same_base_features_plus_confirmed_pay_key_only(self):
        f = feature_frame(); categories = models.existing.schema(f, models.base_spec())
        a = models.matrix(f, categories, c.FAMILIES[0]); b = models.matrix(f, categories, c.FAMILIES[1])
        self.assertEqual(len(a.columns), 40); self.assertEqual(len(b.columns), 39)
        self.assertEqual(a['place_paid_positions'].cat.categories.tolist(), ['2', '3'])

    def test_results_and_payouts_excluded_from_feature_matrix(self):
        f = feature_frame(); categories = models.existing.schema(f, models.base_spec())
        before = models.matrix(f, categories, c.FAMILIES[0])
        f.finish_position = 9; f.official_place_label = 0; f['final_odds'] = 500; f['payout'] = 15000
        pd.testing.assert_frame_equal(before, models.matrix(f, categories, c.FAMILIES[0]))

    def test_direct_training_blocks_rank_proxy_label(self):
        f = feature_frame(); models.validate_frame(f, c.FAMILIES[0])
        f.official_label_status = 'rank_reference_only'
        with self.assertRaises(MLFailure): models.validate_frame(f, c.FAMILIES[0])

    def test_missing_optional_features_do_not_remove_rows(self):
        f = feature_frame(); models.validate_frame(f, c.FAMILIES[1])
        x = models.matrix(f, models.existing.schema(f, models.base_spec()), c.FAMILIES[1])
        self.assertEqual(len(x), 4)

    def test_rank_temperature_is_shared_distribution_not_pair_product(self):
        weights = models.strengths(np.array([3., 2., 1., 0.]), .5)
        result = plackett_luce(dict(zip(('01', '02', '03', '04'), weights)), 2)
        self.assertNotAlmostEqual(result['umaren']['01-02'], result['win']['01'] * result['win']['02'])

    def test_top3_order_loss_prefers_correct_order(self):
        f = feature_frame()
        self.assertLess(models.top3_loss(f, np.array([3., 2., 1., 0.]), 1), models.top3_loss(f, np.array([0., 1., 2., 3.]), 1))

    def test_streaming_metrics_match_sklearn(self):
        scores = Scores(); p = {'01': .8, '02': .5, '03': .1}; y = {'01': 1, '02': 1, '03': 0}
        scores.add(p, y); got = scores.result()
        self.assertAlmostEqual(got['binary_log_loss'], log_loss(list(y.values()), list(p.values())))
        self.assertAlmostEqual(got['brier_score'], brier_score_loss(list(y.values()), list(p.values())))
        self.assertAlmostEqual(got['roc_auc'], roc_auc_score(list(y.values()), list(p.values())))
        self.assertEqual(sum(b['targets'] for b in got['calibration']), 3); self.assertIsNone(got['ROI'])

    def test_scores_require_exact_target_set(self):
        with self.assertRaises(MLFailure): Scores().add({'01': .3}, {'02': 1})

    def test_keiba_lab_extended_schema_keeps_official_missing_and_real_model_id(self):
        r = record(); labels, _ = o.rank_reference(r['finishes'], 2)
        p = plackett_luce(dict.fromkeys(r['finishes'], 1.), 2)
        rows = list(runner.sample_rows(r, 'H_PL', p, labels, '2026-10-22T00:00:00+00:00', 'a' * 64, 'actual_saved_model_id'))
        self.assertEqual(len(rows), 4 + 6 + 6)
        self.assertTrue(all(v['official_hit'] == '' and v['official_payout_yen_per_100'] == '' and v['predicted_at'] == '' for v in rows))
        self.assertTrue(all(v['model_id'] == 'actual_saved_model_id' for v in rows))
        stream = StringIO(); w = csv.DictWriter(stream, fieldnames=c.COLS); w.writeheader(); w.writerows(rows)
        stream.seek(0); self.assertEqual(len(list(csv.DictReader(stream))), 16)

    def test_two_family_fit_cap_preregistered_without_real_execution(self):
        p = c.training_plan(); self.assertEqual(p['real_data_fits_max'], 2)
        self.assertEqual(len(p['families']), 2); self.assertFalse(p['sealed_2026_read'])

    def test_direct_fit_api_cannot_skip_human_authorization(self):
        with patch.object(training, 'approved', side_effect=MLFailure('explicit_execution_required')), \
                patch('lightgbm.train') as fit:
            with self.assertRaises(MLFailure):
                models.fit_family(MagicMock(), c.FAMILIES[0], None, None, None,
                                  approved_plan_sha256='a' * 64, reserve_fit=MagicMock())
        fit.assert_not_called()

    def test_cli_unapproved_train_stops_before_data_or_fit(self):
        with patch.object(c, 'folder', return_value=MagicMock()), \
                patch.object(training, 'approved', side_effect=MLFailure('explicit_execution_required')), \
                patch('builtins.print') as emit, patch('lightgbm.train') as fit:
            result = runner.main(['train', '--run-id', 'a' * 32])
        self.assertEqual(result, 1); self.assertIn('explicit_execution_required', emit.call_args.args[0])
        fit.assert_not_called()

    def test_multibet_budget_not_shared_with_optuna_or_tuner_fits(self):
        directory = MagicMock(); directory.glob.return_value = [1, 2]
        with patch.object(training, 'approved', return_value='a' * 64), \
                patch.object(c, 'write') as write, self.assertRaises(MLFailure):
            training.reserve(directory, c.FAMILIES[1])
        write.assert_not_called()


if __name__ == '__main__': unittest.main()
