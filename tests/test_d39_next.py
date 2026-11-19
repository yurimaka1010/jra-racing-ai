"""Synthetic-only regressions for the new clean-race candidate and finite budget."""
from copy import deepcopy
from datetime import date, timedelta, datetime
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from jra_racing_ai.d39_next import contracts as c, features as f, data, learning
from jra_racing_ai.research import contracts as legacy_c, features as legacy_f
from jra_racing_ai.ml import safety
from test_research_stage2 import source


def build(history, target):
    b = f.Builder()
    for rows in history:
        b.transform_day(rows)
    x, _ = b.transform_day(target, prediction=True)
    return x


class CleanRaceTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2024, 1, 1)
        self.raw = source(self.day)

    def test_all_seven_official_anomalies_veto_entire_race(self):
        for code in f.ANOMALIES:
            rows = deepcopy(self.raw); rows[1]['anomaly'] = code
            b = f.Builder(); _, meta = b.transform_day(rows)
            self.assertTrue(all(not m['eligible'] for m in meta))
            self.assertEqual(b.excluded_races, 1)
            self.assertEqual(b.pending[-1][1], [])

    def test_missing_unknown_anomaly_never_normal(self):
        for value in (None, '', ' ', '8', '00'):
            rows = deepcopy(self.raw); rows[0]['anomaly'] = value
            self.assertIn('anomaly_unknown', f.race_reasons(rows))

    def test_normal_race_and_39_base_features_retained(self):
        self.assertEqual(f.race_reasons(self.raw), [])
        expected = legacy_f.Builder().transform_day(self.raw)[0]
        actual = f.Builder().transform_day(self.raw)[0]
        self.assertEqual(len(c.spec()['features']), 47)
        for x, y in zip(actual, expected):
            self.assertEqual({k: x[k] for k in legacy_c.ALL_FEATURES}, y)

    def test_gap_tie_duplicate_and_incomplete_roster_excluded(self):
        for field, value, reason in [('finish_position', '02', 'tied_or_inconsistent_finish'),
                                      ('finish_position', '00', 'invalid_finish'),
                                      ('actual_runners', '04', 'incomplete_starter_set')]:
            rows = deepcopy(self.raw); rows[0][field] = value
            self.assertIn(reason, f.race_reasons(rows))
        self.assertIn('duplicate_entry', f.race_reasons([*self.raw, self.raw[0]]))

    def test_all_horses_rank_history_removed_not_just_anomalous_horse(self):
        rows = deepcopy(self.raw); rows[0]['anomaly'] = '7'
        target = source(self.day + timedelta(days=8))
        result = build([rows], target)
        self.assertTrue(all(x['past_starts'] == 0 for x in result))
        self.assertTrue(all(x['previous_finish'] is None for x in result))
        self.assertTrue(all(x['a_previous_finish_percentile'] is None for x in result))

    def test_abnormal_start_date_can_supply_gap_but_no_rank(self):
        rows = deepcopy(self.raw); rows[0].update(anomaly='4', finish_position='00')
        result = build([rows], source(self.day + timedelta(days=8)))
        self.assertEqual(result[0]['e_days_since_actual_start'], 8)
        self.assertEqual(result[0]['e_last_start_irregular'], 1.)
        self.assertIsNone(result[0]['previous_finish'])

    def test_cancelled_horse_does_not_supply_actual_start_date(self):
        rows = deepcopy(self.raw); rows[0].update(anomaly='1', finish_position='00')
        result = build([rows], source(self.day + timedelta(days=8)))
        self.assertIsNone(result[0]['e_days_since_actual_start'])
        self.assertEqual(result[1]['e_days_since_actual_start'], 8)

    def test_current_results_and_final_odds_do_not_change_predictions(self):
        target = source(self.day + timedelta(days=100)); changed = deepcopy(target)
        for r in changed:
            r.update(finish_position='00', anomaly='7', late3f_raw='100', corner4='01',
                     actual_runners='28', final_odds='9999', payout='100000')
        self.assertEqual(build([self.raw], target), build([self.raw], changed))

    def test_six_day_results_unavailable_and_seven_day_results_available(self):
        six = build([self.raw], source(self.day + timedelta(days=6)))
        seven = build([self.raw], source(self.day + timedelta(days=7)))
        self.assertIsNone(six[0]['e_days_since_actual_start'])
        self.assertEqual(seven[0]['e_days_since_actual_start'], 7)

    def test_rest_first_second_third_and_performance_past_only(self):
        after = source(self.day + timedelta(days=100))
        second = source(self.day + timedelta(days=120))
        third = source(self.day + timedelta(days=140))
        self.assertEqual(build([self.raw], after)[0]['e_rest_stage'], 'first')
        self.assertEqual(build([self.raw, after], second)[0]['e_rest_stage'], 'second')
        out = build([self.raw, after, second], third)[0]
        self.assertEqual(out['e_rest_stage'], 'third')
        self.assertEqual(out['e_rest_first_past_starts'], 1)
        self.assertEqual(out['e_rest_first_smoothed_finish_percentile'], 2.5 / 4)
        self.assertEqual(out['e_previous_start_gap_days'], 20)

    def test_irregular_rest_first_performance_never_counted(self):
        after = source(self.day + timedelta(days=100)); after[0]['anomaly'] = '7'
        out = build([self.raw, after], source(self.day + timedelta(days=120)))
        self.assertEqual(out[0]['e_rest_first_past_starts'], 0)
        self.assertEqual(out[1]['e_rest_first_past_starts'], 0)

    def test_optional_history_missing_does_not_remove_target_population(self):
        x, meta = f.Builder().transform_day(self.raw)
        self.assertTrue(all(m['eligible'] for m in meta))
        self.assertTrue(all(v['e_rest_stage'] is None for v in x))

    def test_unordered_input_still_joins_extras_by_identity(self):
        b = f.Builder(); b.transform_day(self.raw)
        target = list(reversed(source(self.day + timedelta(days=8))))
        values, meta = b.transform_day(target, prediction=True)
        self.assertEqual([m['key'][-1] for m in meta], ['03', '02', '01'])
        self.assertTrue(all(x['e_days_since_actual_start'] == 8 for x in values))

    def test_sql_uses_full_key_antijoin_with_unknown_flag_rejection(self):
        fields = {v[1] for v in legacy_c.RAW_FIELDS.values()}
        schema = {'se': fields, 'ra': fields}
        p = {'version': c.old.VERSION, '2026_training_authorized': True, 'latest_final_date': '2026-10-18'}
        sql, args, _ = data.target_query(schema, date(2024, 1, 1), date(2024, 1, 31), p)
        self.assertIn('NOT EXISTS', sql)
        self.assertIn("bad.ijo_kubun_code::text IS DISTINCT FROM '0'", sql)
        self.assertEqual(sql.count('bad."'), 6)
        self.assertNotIn("se.ijo_kubun_code::text='0'", sql)

    def test_new_category_schema_is_stable_and_unknown_explicit(self):
        frame = pd.DataFrame(f.Builder().transform_day(self.raw)[0])
        frame['entry_key'] = ['2024010105010101' + f'{i:02}' for i in range(1, 4)]
        cats = learning.matrices.schema(frame, c.spec())
        other = frame.copy(); other['e_rest_stage'] = 'unexpected'
        out = data.matrix(other, c.spec(), cats)
        self.assertEqual(out.columns.tolist(), c.spec()['features'])
        self.assertTrue((out.e_rest_stage == '__UNKNOWN__').all())
        self.assertEqual(out.e_rest_stage.dtype.categories.tolist(), cats['e_rest_stage'])

    def test_future_added_rows_never_change_previously_generated_features(self):
        builder = f.Builder(); builder.transform_day(self.raw)
        target = source(self.day + timedelta(days=100))
        original = deepcopy(builder.transform_day(target)[0])
        builder.transform_day(source(self.day + timedelta(days=200)))
        self.assertEqual(original, build([self.raw], target))


class EightParameterTests(unittest.TestCase):
    def test_all_eight_suggestions_and_no_conflicting_round_or_leaf_alias(self):
        trial = MagicMock()
        trial.suggest_float.side_effect = lambda name, lo, hi, **kw: lo
        trial.suggest_int.side_effect = lambda name, lo, hi, **kw: lo
        plan = {'parameters': {'min_data_in_leaf': 20, 'num_iterations': 999}, 'search': {
            'lambda_l1': [1e-8, 10.], 'lambda_l2': [1e-8, 10.], 'num_leaves': [7, 63],
            'feature_fraction': [.6, 1.], 'bagging_fraction': [.6, 1.], 'bagging_freq': [1, 7],
            'min_child_samples': [10, 100], 'num_iterations': [100, 600, 100]}}
        params, rounds = learning.trial_parameters(trial, plan)
        names = {x.args[0] for x in trial.suggest_float.call_args_list + trial.suggest_int.call_args_list}
        self.assertEqual(names, set(plan['search']))
        self.assertEqual(rounds, 100)
        self.assertNotIn('num_iterations', params)
        self.assertNotIn('min_data_in_leaf', params)
        self.assertEqual(params['min_child_samples'], 10)

    def test_no_training_without_new_explicit_approval(self):
        with patch.object(c, 'read', side_effect=[{'version': c.VERSION}, safety.MLFailure('invalid_input')]):
            with self.assertRaisesRegex(safety.MLFailure, 'explicit_execution_required'):
                c.approved(c.ROOT / 'fake')

    def test_actual_train_counter_refuses_ninth_search_attempt(self):
        events = []
        counter = learning.prior_tuner.FitCounter(8, float('inf'), lambda *x: events.append(x))
        counted = counter.wrap(lambda **kw: 'fake')
        for i in range(8):
            self.assertEqual(counted(), 'fake')
        with self.assertRaisesRegex(safety.MLFailure, 'input_limit'):
            counted()
        self.assertEqual(counter.calls, 8)

    def test_failed_train_consumes_one_attempt(self):
        counter = learning.prior_tuner.FitCounter(1, float('inf'), lambda *x: None)
        with self.assertRaises(ValueError):
            counter.wrap(lambda **kw: (_ for _ in ()).throw(ValueError('fictional')))()
        self.assertEqual(counter.calls, 1)


if __name__ == '__main__':
    unittest.main()
