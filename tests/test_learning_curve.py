"""Controlled training and uncertainty tests use fictional rows only."""
from copy import deepcopy
from datetime import date
import unittest

from jra_racing_ai.ml import features, learning_curve as curve, model, safety
from jra_racing_ai.ml.demo import synthetic_rows


class LearningCurveTests(unittest.TestCase):
    def test_nested_windows_keep_complete_race_keys_and_same_end(self):
        races = [(str(i).zfill(4),) for i in range(20)]
        windows = curve.nested_windows(races, requested=(3, 7, 12, 100))
        self.assertEqual([n for n, _ in windows], [3, 7, 12, 20])
        for (_, first), (_, second) in zip(windows, windows[1:]):
            self.assertTrue(set(first).issubset(second))
            self.assertEqual(first[-1], second[-1])

    def test_cluster_bootstrap_preserves_horse_weighted_loss(self):
        baseline = {"a": (4., 2., 2, 0.), "b": (40., 20., 20, 1.)}
        new = {"a": (3.8, 1.8, 2, 1.), "b": (40., 20., 20, 1.)}
        result = curve.paired_bootstrap(new, baseline)
        self.assertAlmostEqual(result["log_loss"]["difference_new_minus_baseline"], -.2 / 22)
        self.assertAlmostEqual(result["log_loss"]["ci95_low"], -.1)
        self.assertAlmostEqual(result["log_loss"]["ci95_high"], 0.)
        self.assertEqual(result["top1_accuracy_percentage_points"]["difference_new_minus_baseline"], 50.)
        self.assertEqual(result, curve.paired_bootstrap(new, baseline))

    def test_identical_paired_predictions_have_exact_zero_differences_and_intervals(self):
        scores = {str(i): (float(i+1), .2, i+2, .5) for i in range(10)}
        result = curve.paired_bootstrap(scores, scores)
        for key in ("log_loss", "brier_score", "top1_accuracy_percentage_points"):
            self.assertEqual(set(result[key].values()), {0.})

    def test_different_race_or_runner_sets_are_not_paired(self):
        for other in ({"b": (1, 1, 8, 1)}, {"a": (1, 1, 9, 1)}):
            with self.assertRaises(safety.MLFailure):
                curve.paired_bootstrap({"a": (1, 1, 8, 1)}, other)

    def test_burnin_lag_boundaries_and_predeclared_temperatures_enforced(self):
        plan = dict(history_start="2020-01-01", train_start="2021-01-01", train_end="2021-06-30",
            validation_start="2021-07-08", validation_end="2021-07-31", calibration_start="2021-08-08",
            calibration_end="2021-08-31", test_start="2021-09-08", test_end="2021-09-30",
            history_lag_days=7, max_rounds=100, temperature_series={"recalibrated_candidates":[.75,1.,1.25,2.]})
        self.assertIn("train_end", curve.validate_plan(plan))
        for key, value in (("history_start", "2020-02-01"), ("validation_start", "2021-07-01"),
                           ("calibration_start", "2021-08-01"), ("test_start", "2021-09-01")):
            bad = dict(plan, **{key:value})
            with self.assertRaises(safety.MLFailure):
                curve.validate_plan(bad)


class PreparedFeatureTests(unittest.TestCase):
    def test_prepared_features_match_default_training_and_record_executed_rounds(self):
        rows = synthetic_rows(days=16)
        dates = sorted({features.entry_identity(row)[1] for row in rows})
        matrix = features.build_features(rows)
        before = deepcopy(matrix)
        kwargs = dict(rounds=5, targets=("win",), train_end=dates[9], valid_end=dates[12], test_end=dates[-1])
        default = model.fit_models(rows, **kwargs)
        cached = model.fit_models(rows, _prepared_features=matrix, **kwargs)
        self.assertEqual(default["predictions"], cached["predictions"])
        self.assertEqual(matrix, before)
        rounds = cached["metadata"]["training_rounds"]["win"]
        self.assertEqual(rounds["rounds_executed"], 5)
        self.assertTrue(rounds["reached_round_limit"])
        self.assertFalse(rounds["early_stopped"])

    def test_wrong_cache_size_or_column_set_rejected(self):
        rows = synthetic_rows(days=8)
        for matrix in ([], [{} for _ in rows]):
            with self.assertRaises(safety.MLFailure):
                model.fit_models(rows, rounds=5, targets=("win",), _prepared_features=matrix)
