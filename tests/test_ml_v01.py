"""Fictional-only tests for the independent Version 0.1 pipeline."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import ml_cli
from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.db_connection import QueryRunner
from jra_racing_ai.ml import data, features, model, paper, safety
from jra_racing_ai.ml.demo import synthetic_rows
from jra_racing_ai.ml.safety import MLFailure


def mapping():
    value = {"format_version": 1, "reviewed": True, "columns": {key: None for key in data.SOURCES},
             "pre_race_fields": [], "scales": {}, "history_lag_days": 7}
    value["columns"].update(race_status="fake_status", horse_status="fake_status",
                            horse_number="fake_number", finish_position="fake_finish", horse_id="fake_horse")
    return value


def catalog_for(value):
    return {alias: [{"name": column, "type_is_builtin": True, "type_name": "varchar"}
                    for column in set(RACE_KEY) | {column for field, column in value["columns"].items()
                                                   if column and data.SOURCES[field] == alias}]
            for alias in ("ra", "se")}


def odds_document(rows, *, wager="win", price=5., kind="scalar"):
    records = []
    for row in rows:
        cutoff = features.race_cutoff(row)
        item = {key: row[key] for key in (*RACE_KEY, "horse_number")}
        item.update(wager=wager, captured_at=(cutoff - timedelta(minutes=10)).isoformat(),
                    decision_at=(cutoff - timedelta(minutes=5)).isoformat(), price=price, price_kind=kind)
        records.append(item)
    return {"format_version": 1, "source": "pre_race_snapshot", "records": records}


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows(12)

    def test_current_result_never_feature(self):
        original = features.build_features(self.rows[-8:], history=self.rows[:-8])
        altered = [dict(row, finish_position="28", finish_time_seconds="800") for row in self.rows[-8:]]
        self.assertEqual(original, features.build_features(altered, history=self.rows[:-8]))
        self.assertNotIn("finish_position", original[0])
        self.assertNotIn("horse_id", original[0])

    def test_future_results_cannot_change_earlier_features(self):
        original = features.build_features(self.rows[:16], history=self.rows[16:])
        changed = [dict(row, finish_position="28") for row in self.rows[16:]]
        self.assertEqual(original, features.build_features(self.rows[:16], history=changed))

    def test_past_features_and_course(self):
        result = features.build_features(self.rows)
        self.assertEqual(result[8]["past_starts"], 1)
        self.assertEqual(result[8]["days_since_previous"], 8)
        self.assertEqual(result[8]["previous_finish"], 1)
        self.assertEqual(result[8]["same_course_mean_finish"], 1)
        self.assertEqual(result[8]["previous_time_seconds"], 95)

    def test_lag_prevents_recent_result_use(self):
        result = features.build_features(self.rows[:16], history_lag_days=9)
        self.assertEqual(result[8]["past_starts"], 0)

    def test_same_day_results_are_excluded(self):
        old = self.rows[0]
        new = dict(old, race_bango="02", finish_position="2")
        result = features.build_features([old, new], history_lag_days=1)
        self.assertEqual([row["past_starts"] for row in result], [0, 0])

    def test_missing_horse_history(self):
        result = features.build_features([dict(row, horse_id="") for row in self.rows])
        self.assertEqual(result[-1]["past_starts"], 0)
        self.assertIsNone(result[-1]["previous_finish"])

    def test_duplicate_runner_rejected(self):
        with self.assertRaisesRegex(MLFailure, "duplicate_entry"):
            features.validate_rows([self.rows[0], self.rows[0]])

    def test_invalid_key_is_fixed_error(self):
        with self.assertRaises(MLFailure) as failure:
            features.entry_identity(dict(self.rows[0], horse_number="SYNTHETIC_SECRET"))
        self.assertNotIn("SECRET", str(failure.exception))

    def test_full_race_key_keeps_meetings_separate(self):
        row = dict(self.rows[0], kaisai_kai="02")
        self.assertNotEqual(features.entry_identity(row)[0], features.entry_identity(self.rows[0])[0])

    def test_invalid_calendar_date_rejected(self):
        with self.assertRaisesRegex(MLFailure, "invalid_date"):
            features.entry_identity(dict(self.rows[0], kaisai_tsukihi="0230"))

    def test_race_level_inconsistency_rejected(self):
        rows = deepcopy(self.rows[:8])
        rows[0]["distance"] = "2000"
        with self.assertRaises(MLFailure):
            features.validate_rows(rows)

    def test_optional_bad_numbers_become_missing(self):
        result = features.build_features([dict(self.rows[0], weight_kg="NaN", age="-1")])
        self.assertIsNone(result[0]["weight_kg"])
        self.assertIsNone(result[0]["age"])

    def test_proxy_win_and_place_labels(self):
        labels, reasons = features.rank_labels(self.rows[:8])
        self.assertEqual(sum(row["win"] for row in labels), 1)
        self.assertEqual(sum(row["place"] for row in labels), 3)
        self.assertEqual(reasons, {})

    def test_seven_runner_proxy_is_two_places(self):
        labels, _ = features.rank_labels(self.rows[:7])
        self.assertEqual(sum(row["place"] for row in labels), 2)

    def test_small_field_place_is_unknown(self):
        labels, reasons = features.rank_labels(self.rows[:4])
        self.assertTrue(all(row["place"] is None for row in labels))
        self.assertEqual(reasons["place_proxy_field_too_small"], 4)

    def test_invalid_rank_not_negative_label(self):
        labels, _ = features.rank_labels([dict(self.rows[0], finish_position="00")])
        self.assertEqual(labels, [{"win": None, "place": None}])

    def test_dead_heat_proxy_keeps_multiple_winners(self):
        rows = deepcopy(self.rows[:8])
        rows[1]["finish_position"] = "1"
        labels, _ = features.rank_labels(rows)
        self.assertEqual(sum(row["win"] for row in labels), 2)

    def test_chronological_split_never_mixes_race(self):
        split = features.chronological_split(self.rows)
        dates = {name: {features.entry_identity(self.rows[i])[1] for i in indices} for name, indices in split.items()}
        self.assertLess(max(dates["train"]), min(dates["validation"]))
        self.assertLess(max(dates["validation"]), min(dates["test"]))

    def test_explicit_split_and_test_limit(self):
        split = features.chronological_split(self.rows, train_end=date(2020, 2, 1),
                                              valid_end=date(2020, 3, 1), test_end=date(2020, 3, 25))
        self.assertTrue(all(features.entry_identity(self.rows[i])[1] <= date(2020, 3, 25) for i in split["test"]))

    def test_empty_split_rejected(self):
        with self.assertRaises(MLFailure):
            features.chronological_split(self.rows[:8])


class MappingAndQueryTests(unittest.TestCase):
    def test_mapping_needs_human_review(self):
        value = mapping()
        value["reviewed"] = False
        with self.assertRaisesRegex(MLFailure, "mapping_not_reviewed"):
            data.validate_mapping(value)

    def test_weight_and_time_scale_explicit(self):
        for field in ("weight_kg", "finish_time_seconds"):
            value = mapping()
            value["columns"][field] = "fake_units"
            with self.assertRaisesRegex(MLFailure, "invalid_mapping"):
                data.validate_mapping(value)
            value["scales"][field] = .1
            data.validate_mapping(value)

    def test_sql_select_only_and_full_join(self):
        from jra_racing_ai.db_connection import validate_audit_sql
        sql, params = data.export_query(mapping(), start=date(2025, 1, 1), end=date(2025, 1, 31))
        validate_audit_sql(sql)
        for key in RACE_KEY:
            self.assertIn(f'ra."{key}" = se."{key}"', sql)
        self.assertEqual(params[:4], ("7", "7", "20250101", "20250131"))
        self.assertEqual(params[-1], 100001)

    def test_optional_features_require_asof_review(self):
        value = mapping()
        value["columns"]["distance"] = "fake_distance"
        sql, _ = data.export_query(value, start=date(2025, 1, 1), end=date(2025, 1, 31))
        self.assertNotIn('ra."fake_distance"', sql)
        value["pre_race_fields"] = ["distance"]
        sql, _ = data.export_query(value, start=date(2025, 1, 1), end=date(2025, 1, 31))
        self.assertIn('ra."fake_distance"', sql)

    def test_sql_injection_column_rejected(self):
        value = mapping()
        value["columns"]["finish_position"] = 'fake; DROP TABLE x'
        with self.assertRaises(MLFailure):
            data.validate_mapping(value)

    def test_custom_type_rejected(self):
        value = mapping()
        schema = catalog_for(value)
        next(row for row in schema["se"] if row["name"] == "fake_finish")["type_is_builtin"] = False
        with self.assertRaises(MLFailure):
            data.check_mapping_schema(value, schema)

    def test_export_limit_validation(self):
        for cap in (0, 500001):
            with self.assertRaises(MLFailure):
                data.export_query(mapping(), start=date(2025, 1, 1), end=date(2025, 1, 31), max_rows=cap)

    def test_stream_rejects_write(self):
        runner = QueryRunner(MagicMock())
        with self.assertRaises(SafeAuditError) as failure:
            list(runner.iter_batches("ml.test", "DELETE FROM public.fake"))
        self.assertEqual(failure.exception.code, "unsafe_sql")

    def test_stream_uses_named_cursor_fetchmany_and_cap(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchmany.side_effect = [[{"x": 1}, {"x": 2}], [{"x": 3}]]
        runner = QueryRunner(connection)
        with self.assertRaises(SafeAuditError) as failure:
            list(runner.iter_batches("ml.test", "SELECT 1", max_rows=2))
        self.assertEqual(failure.exception.code, "result_limit")
        connection.cursor.assert_called_once_with(name="jra_ml_export")
        cursor.fetchall.assert_not_called()

    def test_stream_deadline(self):
        runner = QueryRunner(MagicMock())
        with patch("jra_racing_ai.db_connection.time.monotonic", side_effect=[0, 11]):
            with self.assertRaises(SafeAuditError) as failure:
                list(runner.iter_batches("ml.test", "SELECT 1", max_seconds=10))
            self.assertEqual(failure.exception.code, "query_timeout")


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows(1)
        self.predictions = {"win": [.4] * 8, "place": [.6] * 8}

    def test_missing_odds_all_skip_and_no_finances(self):
        summary = paper.summarize(paper.paper_decisions(self.rows, self.predictions))
        self.assertEqual(summary["paper_bet_count"], 0)
        self.assertEqual(summary["investment_yen"], 0)
        self.assertIsNone(summary["net_profit_yen"])
        self.assertIsNone(summary["payout_yen"])

    def test_ev_and_stake_and_race_cap(self):
        odds = paper.parse_odds(odds_document(self.rows))
        decisions = paper.paper_decisions(self.rows, self.predictions, odds=odds)
        buys = [item for item in decisions if item["decision"] == "paper_buy"]
        self.assertEqual(len(buys), 2)
        self.assertEqual(buys[0]["expected_return"], 1.)
        self.assertEqual(buys[0]["stake_yen"], 100)
        summary = paper.summarize(decisions)
        self.assertEqual(summary["investment_yen"], 200)
        self.assertIsNone(summary["return_rate"])

    def test_place_ranges_are_not_scalar_prices(self):
        odds = paper.parse_odds(odds_document(self.rows, wager="place", price=[1.5, 2.8], kind="range"))
        decisions = paper.paper_decisions(self.rows, self.predictions, odds=odds)
        self.assertTrue(all(item["decision"] == "skip" for item in decisions))
        self.assertEqual(sum(item["reason"] == "place_range_unpriced" for item in decisions), 8)

    def test_final_odds_source_rejected(self):
        value = odds_document(self.rows)
        value["source"] = "final_odds"
        with self.assertRaises(MLFailure):
            paper.parse_odds(value)

    def test_duplicate_odds_rejected(self):
        value = odds_document(self.rows)
        value["records"].append(value["records"][0])
        with self.assertRaises(MLFailure):
            paper.parse_odds(value)

    def test_odds_after_decision_rejected(self):
        value = odds_document(self.rows)
        value["records"][0]["captured_at"] = features.race_cutoff(self.rows[0]).isoformat()
        with self.assertRaises(MLFailure):
            paper.parse_odds(value)

    def test_odds_after_race_hold(self):
        value = odds_document(self.rows)
        for item in value["records"]:
            item["decision_at"] = features.race_cutoff(self.rows[0]).isoformat()
        decisions = paper.paper_decisions(self.rows, self.predictions, odds=paper.parse_odds(value))
        self.assertTrue(all(item["decision"] == "skip" for item in decisions))

    def test_invalid_stake_rejected(self):
        for stake in (0, 150, True):
            with self.assertRaises(MLFailure):
                paper.PurchasePolicy(stake_yen=stake).validate()

    def test_nan_threshold_rejected(self):
        with self.assertRaises(MLFailure):
            paper.PurchasePolicy(min_expected_return=float("nan")).validate()

    def test_policy_roundtrip_frozen(self):
        policy = paper.PurchasePolicy(.2, 200, 3)
        self.assertEqual(policy, paper.PurchasePolicy.from_dict(policy.as_dict()))
        tampered = dict(policy.as_dict(), production_settlement_enabled=True)
        with self.assertRaises(MLFailure):
            paper.PurchasePolicy.from_dict(tampered)

    def test_forward_requires_future_and_no_results(self):
        now = features.race_cutoff(self.rows[0]) - timedelta(hours=1)
        with self.assertRaisesRegex(MLFailure, "forward_cutoff"):
            paper.paper_decisions(self.rows, self.predictions, mode="forward", now=now)
        rows = [dict(row, finish_position="", finish_time_seconds="") for row in self.rows]
        decisions = paper.paper_decisions(rows, self.predictions, mode="forward", now=now)
        self.assertTrue(all(item["decision_at"] == now.isoformat() for item in decisions))
        with self.assertRaises(MLFailure):
            paper.paper_decisions(rows, self.predictions, mode="forward", now=now + timedelta(hours=1))

    def test_missing_start_time_uses_midnight_cutoff(self):
        cutoff = features.race_cutoff(dict(self.rows[0], start_time=""))
        self.assertEqual(cutoff.hour, 0)

    def test_finances_never_derived_from_odds(self):
        decisions = paper.paper_decisions(self.rows, self.predictions,
                                          odds=paper.parse_odds(odds_document(self.rows)))
        label_map = {(*features.entry_identity(row)[0], "win"): 1 for row in self.rows}
        result = paper.summarize(decisions, labels=label_map)
        self.assertEqual(result["rank_proxy_hit_count"], 2)
        self.assertIsNone(result["payout_yen"])
        self.assertIsNone(result["net_profit_yen"])

    def test_summary_rejects_arbitrary_private_reason(self):
        decisions = paper.paper_decisions(self.rows, self.predictions)
        decisions[0]["reason"] = "SYNTHETIC_PRIVATE_REASON"
        with self.assertRaises(MLFailure) as failure:
            paper.summarize(decisions)
        self.assertNotIn("PRIVATE", str(failure.exception))

    def test_forward_rejects_current_race_time(self):
        rows = [dict(row, finish_position="") for row in self.rows]
        with self.assertRaisesRegex(MLFailure, "forward_cutoff"):
            paper.paper_decisions(rows, self.predictions, mode="forward",
                                  now=features.race_cutoff(rows[0]) - timedelta(hours=1))


class ModelAndStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows(24)
        try:
            model.libraries()
        except MLFailure as exc:
            if exc.code == "missing_library":
                raise unittest.SkipTest("optional_ml_libraries_not_installed") from None
            raise
        cls.bundle = model.fit_models(cls.rows, rounds=20)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def test_actual_lightgbm_fit_and_probability_metrics(self):
        self.assertEqual(set(self.bundle["boosters"]), {"win", "place"})
        for target in ("win", "place"):
            self.assertTrue(all(0 <= p <= 1 for p in self.bundle["predictions"][target]))
            metrics = self.bundle["metadata"]["metrics"][target]["test"]
            self.assertIn("train_prior_baseline", metrics)
            self.assertEqual(sum(item["count"] for item in metrics["model"]["calibration_bins"]), metrics["model"]["count"])

    def test_category_vocabulary_train_only_and_unknown_at_predict(self):
        values = features.build_features(self.rows[:8])
        schema = model.category_schema(values)
        changed = [dict(values[0], venue="SYNTHETIC_UNSEEN")]
        frame = model.frame(changed, schema)
        self.assertEqual(str(frame["venue"].dtype), "category")
        self.assertEqual(frame.iloc[0]["venue"], "__UNKNOWN__")
        self.assertEqual(list(frame["venue"].cat.categories), schema["venue"])

    def test_missing_category_has_separate_sentinel(self):
        values = features.build_features(self.rows[:8])
        schema = model.category_schema(values)
        frame = model.frame([dict(values[0], sex=None)], schema)
        self.assertEqual(frame.iloc[0]["sex"], "__MISSING__")

    def test_probability_metric_known_values(self):
        result = model.probability_metrics([0, 1], [.5, .5])
        self.assertAlmostEqual(result["log_loss"], .6931471805599453)
        self.assertEqual(result["brier_score"], .25)
        self.assertEqual(result["calibration_ece"], 0.)

    def test_saved_model_reload_predict_roundtrip(self):
        directory = safety.new_artifact("models", root=self.root)
        model.save_models(self.bundle, directory, root=self.root)
        loaded = model.load_models(directory, root=self.root)
        rows = self.rows[-8:]
        original = model.predict(self.bundle, rows, history=self.rows[:-8])
        restored = model.predict(loaded, rows, history=self.rows[:-8])
        self.assertEqual(original, restored)
        self.assertIs(loaded["metadata"]["production_hr_settlement_enabled"], False)

    def test_saved_model_hash_tamper_rejected(self):
        directory = safety.new_artifact("models", root=self.root)
        model.save_models(self.bundle, directory, root=self.root)
        with (directory / "win.lgb.txt").open("a", encoding="utf-8") as stream:
            stream.write("SYNTHETIC_TAMPER")
        with self.assertRaises(MLFailure):
            model.load_models(directory, root=self.root)

    def test_dataset_roundtrip_and_hash(self):
        directory = safety.new_artifact("datasets", root=self.root)
        path = data.write_synthetic_dataset(self.rows, directory, root=self.root)
        rows, metadata = data.read_dataset(path, root=self.root)
        self.assertEqual(rows, self.rows)
        self.assertEqual(metadata["source"], "synthetic")
        with path.open("a") as stream:
            stream.write("SYNTHETIC_TAMPER")
        with self.assertRaises(MLFailure):
            data.read_dataset(path, root=self.root)

    def test_dataset_untrusted_source_is_never_printed(self):
        directory = safety.new_artifact("datasets", root=self.root)
        path = data.write_synthetic_dataset(self.rows, directory, root=self.root)
        metadata = safety.read_json(directory / "dataset.json", root=self.root)
        metadata["source"] = "SYNTHETIC_PRIVATE_SOURCE"
        (directory / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(MLFailure, "artifact_invalid"):
            data.read_dataset(path, root=self.root)

    def test_existing_file_never_overwritten(self):
        path = self.root / "test.json"
        safety.write_json(path, {"a": 1}, root=self.root)
        with self.assertRaisesRegex(MLFailure, "artifact_exists"):
            safety.write_json(path, {"a": 2}, root=self.root)

    def test_path_outside_local_root_rejected(self):
        with self.assertRaisesRegex(MLFailure, "unsafe_path"):
            safety.local_path(self.root.parent / "outside.json", root=self.root)

    def test_onedrive_root_rejected(self):
        sync_root = self.root / "OneDrive" / "ml"
        with self.assertRaises(MLFailure):
            safety.local_path(sync_root / "test.json", root=sync_root)

    def test_cloud_placeholder_rejected(self):
        with patch("jra_racing_ai.ml.safety._cloud_or_reparse", return_value=True):
            with self.assertRaises(MLFailure):
                safety.local_path(self.root / "test.json", root=self.root)

    def test_ledger_write_separate_evaluation_and_integrity(self):
        rows = self.rows[-8:]
        predicted = model.predict(self.bundle, rows, history=self.rows[:-8])
        decisions = paper.paper_decisions(rows, predicted)
        directory = safety.new_artifact("predictions", root=self.root)
        paper.save_ledger(directory, decisions=decisions, mode="backtest", model_hash="synthetic",
                          policy=paper.PurchasePolicy(), root=self.root)
        before = safety.digest(directory / "predictions.json", root=self.root)
        result = paper.evaluate_ledger(directory, rows, root=self.root)
        self.assertEqual(result["missing_outcome_predictions"], 0)
        self.assertIsNotNone(result["probability_metrics"]["win"])
        self.assertIsNone(result["roi"])
        self.assertEqual(before, safety.digest(directory / "predictions.json", root=self.root))
        (directory / "predictions.json").chmod(0o666)
        with (directory / "predictions.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(MLFailure, "ledger_tampered"):
            paper.evaluate_ledger(directory, rows, root=self.root)

    def test_cli_masks_exception_details(self):
        output = io.StringIO()
        with patch.object(ml_cli, "run", side_effect=RuntimeError("SYNTHETIC_PASSWORD")), redirect_stdout(output):
            code = ml_cli.main(["demo"])
        self.assertEqual(code, 2)
        self.assertNotIn("PASSWORD", output.getvalue())

    def test_cli_rejects_db_without_explicit_flag(self):
        output = io.StringIO()
        with patch.object(ml_cli, "read_only_session") as connect, redirect_stdout(output):
            self.assertEqual(ml_cli.main(["inspect"]), 2)
        connect.assert_not_called()

    def test_cli_help_has_no_network_or_driver_use(self):
        self.assertIn("first-run", ml_cli.parser().format_help())

    def test_hidden_prompt_failure_has_no_plaintext_fallback(self):
        import getpass
        with patch.dict("os.environ", {}, clear=True), patch.object(getpass, "getpass", side_effect=getpass.GetPassWarning):
            with self.assertRaisesRegex(MLFailure, "needs_authentication"):
                ml_cli._password()

    def test_minimal_mapping_confirmation(self):
        draft = mapping()
        with patch("builtins.input", return_value="CONFIRM"), redirect_stderr(io.StringIO()):
            result = ml_cli._minimal_mapping(draft)
        self.assertTrue(result["reviewed"])
        self.assertEqual(result["pre_race_fields"], [])

    def test_minimal_mapping_rejection_stops(self):
        with patch("builtins.input", return_value="no"), redirect_stderr(io.StringIO()):
            with self.assertRaises(MLFailure):
                ml_cli._minimal_mapping(mapping())

    def test_export_mock_stream_and_completion_manifest(self):
        value = mapping()
        runner = MagicMock()
        runner.iter_batches.return_value = [self.rows[:8], self.rows[8:16]]
        directory = safety.new_artifact("datasets", root=self.root)
        with patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(value)}, value)):
            metadata = data.export_dataset(runner, value, directory, start=date(2020, 1, 1),
                                            end=date(2021, 1, 1), max_rows=100, root=self.root)
        rows, _ = data.read_dataset(directory / "runners.csv", root=self.root)
        self.assertEqual(metadata["rows"], 16)
        self.assertEqual(len(rows), 16)
        runner.iter_batches.assert_called_once()

    def test_failed_export_preserves_existing_file(self):
        value = mapping()
        directory = safety.new_artifact("datasets", root=self.root)
        path = directory / "runners.csv"
        path.write_text("SYNTHETIC_EXISTING", encoding="utf-8")
        with patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(value)}, value)):
            with self.assertRaises(FileExistsError):
                data.export_dataset(MagicMock(), value, directory, start=date(2020, 1, 1),
                                     end=date(2021, 1, 1), root=self.root)
        self.assertEqual(path.read_text(encoding="utf-8"), "SYNTHETIC_EXISTING")

    def test_stream_failure_does_not_leave_trainable_export(self):
        value = mapping()
        runner = MagicMock()
        runner.iter_batches.side_effect = MLFailure("query_timeout")
        directory = safety.new_artifact("datasets", root=self.root)
        with patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(value)}, value)):
            with self.assertRaises(MLFailure):
                data.export_dataset(runner, value, directory, start=date(2020, 1, 1),
                                     end=date(2021, 1, 1), root=self.root)
        self.assertFalse((directory / "dataset.json").exists())
        self.assertFalse((directory / "runners.csv").exists())

    def test_cli_first_run_mock_db_to_real_lightgbm(self):
        value = mapping()
        runner = MagicMock()
        runner.iter_batches.return_value = [self.rows]
        connect = MagicMock()
        connect.return_value.__enter__.return_value = runner
        output = io.StringIO()
        with patch.object(safety, "ML_ROOT", self.root), patch.object(ml_cli, "read_only_session", connect), \
                patch.object(ml_cli, "_password", return_value="SYNTHETIC_PRIVATE_PASSWORD"), \
                patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(value)}, value)), \
                patch("builtins.input", return_value="CONFIRM"), patch.dict("os.environ", {}, clear=True), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = ml_cli.main(["first-run", "--execute-local", "--start", "2020-01-01",
                                "--end", "2021-01-01", "--rounds", "10"])
        self.assertEqual(code, 0, output.getvalue())
        result = json.loads(output.getvalue())
        self.assertEqual(result["models"], 2)
        self.assertGreater(result["split_counts"]["test"]["rows"], 0)
        self.assertNotIn("PRIVATE_PASSWORD", output.getvalue())
        self.assertNotIn("SYNTHETIC_HORSE", output.getvalue())
        self.assertEqual(connect.call_count, 2)

    def test_cli_train_predict_evaluate_roundtrip(self):
        directory = safety.new_artifact("datasets", root=self.root)
        dataset = data.write_synthetic_dataset(self.rows, directory, root=self.root)
        with patch.object(safety, "ML_ROOT", self.root):
            args = ml_cli.parser().parse_args(["train", "--dataset", str(dataset), "--rounds", "10"])
            result = ml_cli.run(args)
            model_dir = self.root / "models" / result["artifact_id"]
            target_dir = safety.new_artifact("datasets", root=self.root)
            targets = data.write_synthetic_dataset(self.rows[-8:], target_dir, root=self.root)
            history_dir = safety.new_artifact("datasets", root=self.root)
            history = data.write_synthetic_dataset(self.rows[:-8], history_dir, root=self.root)
            prediction_args = ml_cli.parser().parse_args(["predict", "--model", str(model_dir),
                                                           "--entries", str(targets), "--history", str(history),
                                                           "--mode", "backtest"])
            predicted = ml_cli.run(prediction_args)
            self.assertEqual(predicted["summary"]["race_count"], 1)
            eval_args = ml_cli.parser().parse_args(["evaluate", "--ledger",
                                                   str(self.root / "predictions" / predicted["artifact_id"]),
                                                   "--outcomes", str(targets)])
            evaluated = ml_cli.run(eval_args)
        self.assertEqual(evaluated["summary"]["missing_outcome_predictions"], 0)
        self.assertIsNone(evaluated["summary"]["net_profit_yen"])

    def test_cli_predictions_on_training_period_rejected(self):
        directory = safety.new_artifact("models", root=self.root)
        model.save_models(self.bundle, directory, root=self.root)
        inputs = safety.new_artifact("datasets", root=self.root)
        path = data.write_synthetic_dataset(self.rows[:8], inputs, root=self.root)
        with patch.object(safety, "ML_ROOT", self.root):
            args = ml_cli.parser().parse_args(["predict", "--model", str(directory), "--entries", str(path), "--mode", "backtest"])
            with self.assertRaisesRegex(MLFailure, "invalid_split"):
                ml_cli.run(args)

    def test_late_forward_ledger_cannot_be_completed(self):
        decisions = paper.paper_decisions(self.rows[:8], {"win": [.2] * 8, "place": [.4] * 8})
        directory = safety.new_artifact("predictions", root=self.root)
        with self.assertRaisesRegex(MLFailure, "forward_cutoff"):
            paper.save_ledger(directory, decisions=decisions, mode="forward", model_hash="synthetic",
                              policy=paper.PurchasePolicy(), root=self.root)
        self.assertFalse((directory / "manifest.json").exists())
