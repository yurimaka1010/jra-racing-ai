"""Fabricated rosters, mocks, and pure math only: no real fits or private inputs."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from jra_racing_ai.ml import safety
from jra_racing_ai.research import contracts as features, model as existing
from jra_racing_ai.objective_pilot import contracts as c, data, model, runner


def frame(day='2024-11-09', races=2, n=6):
    rows = []
    for race in range(1, races + 1):
        for horse in range(1, n + 1):
            row = dict.fromkeys(features.ALL_FEATURES, None)
            row.update(entry_key=day.replace('-', '') + f'050501{race:02d}{horse:02d}', date=day,
                       finish_position=horse, actual_starters=n, winner_label=int(horse == 1))
            rows.append(row)
    return pd.DataFrame(rows)


class TargetContractTests(unittest.TestCase):
    def test_binary_labels_are_finish_one(self):
        np.testing.assert_array_equal(model.targets(frame(), 'binary'), [1, 0, 0, 0, 0, 0] * 2)

    def test_four_classes_and_ranking_share_relevance(self):
        for objective in ('multiclass', 'lambdarank'):
            np.testing.assert_array_equal(model.targets(frame(), objective), [3, 2, 2, 1, 1, 0] * 2)

    def test_regression_uses_actual_starters_and_direction(self):
        f = frame(); f['declared_runners'] = 28
        np.testing.assert_allclose(model.targets(f, 'regression'), [1, .8, .6, .4, .2, 0] * 2)

    def test_nonwinner_dead_heat_kept_without_fake_rank_repair(self):
        f = frame(races=1); f.loc[2, 'finish_position'] = 2
        data.validate_roster(f)
        np.testing.assert_array_equal(model.targets(f, 'multiclass'), [3, 2, 2, 1, 1, 0])

    def test_invalid_target_is_rejected(self):
        for field, value in [('finish_position', 0), ('finish_position', np.nan), ('finish_position', 2.5),
                             ('finish_position', 7), ('actual_starters', 1), ('actual_starters', 29)]:
            f = frame(); f[field] = f[field].astype(float); f.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(safety.MLFailure):
                model.targets(f, 'regression')

    def test_actual_starter_roster_mismatch_rejected(self):
        f = frame(); f.loc[0, 'actual_starters'] = 7
        with self.assertRaises(safety.MLFailure): data.validate_roster(f)

    def test_duplicate_entry_and_missing_runner_rejected(self):
        for f in (pd.concat([frame(), frame().iloc[:1]]), frame().iloc[1:]):
            with self.assertRaises(safety.MLFailure): data.validate_roster(f)

    def test_label_key_and_result_agreement_required(self):
        f = frame(); f.loc[0, 'winner_label'] = 0; f.loc[1, 'winner_label'] = 1
        with self.assertRaises(safety.MLFailure): data.validate_roster(f)
        f = frame(); f.loc[0, 'date'] = '2024-11-10'
        with self.assertRaises(safety.MLFailure): data.validate_roster(f)

    def test_sealed_and_stage7_dates_forbidden_before_fit(self):
        for d in ('2026-01-01', '2026-09-30', '2026-10-24'):
            with self.subTest(day=d), self.assertRaises(safety.MLFailure): data.validate_roster(frame(d))

    def test_lambda_groups_are_full_sorted_races(self):
        self.assertEqual(model.groups(frame()), [6, 6])
        with self.assertRaises(safety.MLFailure): model.groups(frame().iloc[::-1])
        with self.assertRaises(safety.MLFailure): model.groups(frame().iloc[:1])


class ProbabilityAdapterTests(unittest.TestCase):
    def test_binary_uses_existing_power_normalization(self):
        f = frame(); scores = np.linspace(.01, .9, len(f))
        np.testing.assert_array_equal(model.probability(f, scores, 'binary', .75), existing.normalize(f, scores, .75))

    def test_multiclass_extracts_winner_class_three(self):
        f = frame(); x = np.tile([.2, .1, .3, .4], (len(f), 1))
        np.testing.assert_allclose(model.probability(f, x, 'multiclass', 1), np.full(len(f), 1 / 6))

    def test_multiclass_invalid_rows_and_wrong_shape_rejected(self):
        f = frame()
        for v in (np.ones((len(f), 4)), np.zeros((len(f), 3)), np.full((len(f), 4), np.nan)):
            with self.assertRaises(safety.MLFailure): model.probability(f, v, 'multiclass', 1)

    def test_softmax_stable_for_large_scores_and_shift_invariant(self):
        f = frame(); raw = np.arange(len(f), dtype=float)
        for objective in ('regression', 'lambdarank'):
            p = model.probability(f, raw, objective, .25)
            np.testing.assert_allclose(p, model.probability(f, raw + 100000, objective, .25))
            np.testing.assert_allclose(p.reshape(2, 6).sum(axis=1), 1)

    def test_softmax_preserves_rank_not_raw_values_as_probabilities(self):
        f = frame(); scores = np.arange(len(f), dtype=float) - 5
        p = model.probability(f, scores, 'regression', 1)
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertEqual(np.argmax(p[:6]), 5)

    def test_bad_temperature_or_score_rejected(self):
        for t in (0, -1, np.nan, np.inf):
            with self.assertRaises(safety.MLFailure): model.probability(frame(), np.ones(12), 'binary', t)
        for scores in (np.ones(11), np.full(12, np.nan), np.full(12, 2.)):
            with self.assertRaises(safety.MLFailure): model.probability(frame(), scores, 'binary', 1)

    def test_temperature_selected_by_calibration_race_LL_only(self):
        f = frame(); raw = np.tile([3., 2., 1., 0., -1., -2.], 2)
        config = {'objective': 'lambdarank', 'temperature_candidates': [.25, .5, 1., 2.]}
        temperature, grid = model.calibrate(f, raw, config)
        self.assertEqual(temperature, .25)
        self.assertEqual(min(r['race_log_loss'] for r in grid), grid[0]['race_log_loss'])

    def test_temperature_ties_follow_fixed_grid_order(self):
        config = {'objective': 'regression', 'temperature_candidates': [.25, .5, 1., 2.]}
        self.assertEqual(model.calibrate(frame(), np.zeros(12), config)[0], .25)

    def test_outcomes_final_odds_and_target_do_not_enter_model_matrix(self):
        spec = features.spec(('A', 'B', 'C', 'D')); f = frame()
        categories = existing.schema(f, spec)
        before = existing.matrix(f, spec, categories)
        f['finish_position'] = 9; f['winner_label'] = 0; f['actual_starters'] = 28
        f['final_odds'] = 999; f['payout_yen'] = 90000
        pd.testing.assert_frame_equal(before, existing.matrix(f, spec, categories))
        self.assertEqual(len(before.columns), 39)

    def test_optional_feature_missing_keeps_roster(self):
        f = frame(); spec = features.spec(('A', 'B', 'C', 'D'))
        x = existing.matrix(f, spec, existing.schema(f, spec))
        self.assertEqual(len(x), len(f)); data.validate_roster(f)

    def test_prediction_round_trip_adapter_uses_saved_temperature_once(self):
        f = frame(); spec = features.spec(('A', 'B', 'C', 'D')); booster = MagicMock()
        raw = np.tile([.6, .2, .1, .05, .03, .02], 2); booster.predict.return_value = raw
        bundle = {'booster': booster, 'metadata': {'feature_spec': spec, 'categories': existing.schema(f, spec),
                  'configuration': {'objective': 'binary'}, 'calibration': {'temperature': 2.}}}
        np.testing.assert_allclose(model.predict(bundle, f), existing.normalize(f, raw, 2.))


class BudgetAndReportTests(unittest.TestCase):
    def test_parameters_do_not_inherit_market_H_offset_or_temperature(self):
        p = {'parameters': {'num_threads': 4, 'seed': 42}}
        got = model.parameters(p, {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 4})
        self.assertEqual(got['num_class'], 4); self.assertNotIn('init_score', got)
        self.assertNotIn('temperature', got); self.assertEqual(p['parameters'], {'num_threads': 4, 'seed': 42})

    def test_failed_reserved_fit_still_consumes_budget_and_no_retry(self):
        directory = MagicMock(); directory.glob.return_value = [1, 2, 3, 4]
        with patch.object(c, 'write') as write, self.assertRaises(safety.MLFailure):
            c.reserve(directory, c.NAMES[0])
        write.assert_not_called()

    def test_same_fit_reservation_is_exclusive(self):
        directory = Path('synthetic'); written = set()
        def write(_, name, value):
            if name in written: raise safety.MLFailure('artifact_exists')
            written.add(name)
        with patch.object(Path, 'glob', return_value=[]), patch.object(c, 'write', side_effect=write):
            c.reserve(directory, c.NAMES[0])
            with self.assertRaises(safety.MLFailure): c.reserve(directory, c.NAMES[0])

    def test_invalid_extra_configuration_is_not_reserved(self):
        with self.assertRaises(safety.MLFailure): c.reserve(Path('synthetic'), 'optuna')

    def test_worker_failure_outputs_fixed_reason_not_private_exception(self):
        with patch.object(runner.c, 'folder', side_effect=RuntimeError('SECRET_RECORD_AND_PASSWORD')), \
                patch('builtins.print') as emit:
            code = runner.main(['_worker', '--run-id', 'a' * 32, '--phase', 'fit'])
        self.assertEqual(code, 1); message = emit.call_args.args[0]
        self.assertNotIn('SECRET', message); self.assertIn('internal_error', message)

    def test_band_counts_and_binary_counts_keep_all_rows(self):
        y = np.array([1, 0, 0]); p = np.array([.6, .3, .1])
        table = runner.calibration_table(y, p)
        self.assertEqual(sum(r['runners'] for r in table), 3)
        self.assertEqual(next(r for r in table if r['lower_inclusive'] == .5)['observed_win_rate'], 1)

    def test_identifier_rejects_paths(self):
        for value in ('..', 'abc', 'a' * 32 + '/x'):
            with self.assertRaises(safety.MLFailure): c.folder(value)

    def test_worker_memory_includes_venv_interpreter_descendants(self):
        parents = {10: 1, 20: 10, 30: 20, 99: 1}
        self.assertEqual(runner.descendants(10, parents), {10, 20, 30})
        with patch.object(runner, 'process_parents', return_value=parents), \
                patch.object(runner, 'memory_bytes', side_effect=lambda pid: {10: 2, 20: 100, 30: 50}[pid]):
            self.assertEqual(runner.tree_memory(10), 152)

    def test_exited_descendant_not_counted_as_zero_measured_tree(self):
        with patch.object(runner, 'process_parents', return_value={10: 1, 20: 10}), \
                patch.object(runner, 'memory_bytes', side_effect=lambda pid: None if pid == 20 else 2):
            self.assertEqual(runner.tree_memory(10), 2)
        with patch.object(runner, 'process_parents', return_value={}), \
                patch.object(runner, 'memory_bytes', return_value=None):
            self.assertIsNone(runner.tree_memory(10))

    def test_budget_termination_stops_process_tree_not_only_launcher(self):
        child = MagicMock(pid=123)
        with patch.object(runner.subprocess, 'run') as terminate:
            runner.terminate_worker(child)
        self.assertEqual(terminate.call_args.args[0][-4:], ['/PID', '123', '/T', '/F'])
        child.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
