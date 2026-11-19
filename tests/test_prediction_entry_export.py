"""Prediction-only RA/SE export: no current outcomes and no training fallback."""
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import ml_cli, win_sim_cli
from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.ml import data, model, safety
from jra_racing_ai.ml.demo import synthetic_rows
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim import core, local
from test_ml_v01 import mapping, catalog_for


class PredictionEntryExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "v01"
        self.current = self.base / "v02"
        self.root.mkdir()
        self.current.mkdir()
        self.patches = [patch.object(safety, "ML_ROOT", self.root), patch.object(local, "ROOT", self.current)]
        for item in self.patches:
            item.start()
        self.mapping = mapping()
        self.rows = [dict(row, _ra_status="2", _se_status="2") for row in synthetic_rows(20)[-8:]]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temporary.cleanup()

    def export(self, rows=None):
        directory = safety.new_artifact("datasets")
        runner = MagicMock()
        runner.iter_batches.return_value = [self.rows if rows is None else rows]
        with patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(self.mapping)}, {})):
            metadata = data.export_dataset(runner, self.mapping, directory, start=date(2020, 1, 1),
                                           end=date(2021, 1, 1), prediction_only=True)
        return directory, metadata

    def test_prediction_sql_never_selects_current_finish_or_time(self):
        self.mapping["columns"]["finish_time_seconds"] = "fake_time"
        self.mapping["scales"]["finish_time_seconds"] = 1
        sql, params = data.export_query(self.mapping, start=date(2030, 1, 1), end=date(2030, 1, 1), prediction_only=True)
        validate_audit_sql(sql)
        self.assertNotIn('se."fake_finish"', sql)
        self.assertNotIn('se."fake_time"', sql)
        self.assertIn('NULL::text AS "finish_position"', sql)
        self.assertIn('NULL::text AS "finish_time_seconds"', sql)
        self.assertEqual(params[:14], tuple("1234567") * 2)
        for field in RACE_KEY:
            self.assertIn(f'ra."{field}" = se."{field}"', sql)

    def test_entry_export_erases_adapter_outcomes_and_records_real_category_counts(self):
        directory, metadata = self.export()
        rows, _ = data.read_dataset(directory / "runners.csv", allow_prediction_snapshot=True)
        self.assertTrue(all(not row["finish_position"] and not row["finish_time_seconds"] for row in rows))
        self.assertEqual(metadata["source_status_counts"], {"ra_races": {"2": 1}, "se_runners": {"2": 8}})
        self.assertEqual(metadata["purpose"], "prediction_only")
        self.assertTrue(metadata["current_results_excluded"])
        self.assertFalse(metadata["production_hr_settlement_enabled"])

    def test_prediction_entries_are_rejected_by_training_before_model_fit(self):
        directory, _ = self.export()
        with patch.object(model, "fit_models") as fit:
            with self.assertRaises(MLFailure):
                ml_cli.run(ml_cli.parser().parse_args(["train", "--dataset", str(directory / "runners.csv")]))
        fit.assert_not_called()

    def test_prediction_manifest_cannot_hide_reintroduced_current_outcome(self):
        directory, metadata = self.export()
        path = directory / "runners.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["finish_position"] = "1"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data.DATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        metadata["csv_sha256"] = safety.digest(path)
        (directory / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(MLFailure):
            data.read_dataset(path, allow_prediction_snapshot=True)

    def test_deleted_canceled_and_unknown_categories_cannot_be_promoted_to_entries(self):
        for status in ("0", "9", "A", "B", None):
            with self.subTest(status=status), self.assertRaises(MLFailure):
                self.export([dict(self.rows[0], _se_status=status)])
        self.assertFalse(list(self.root.rglob("dataset.json")))
        self.assertFalse(list(self.root.rglob("runners.csv")))

    def test_new_cli_needs_explicit_local_execution(self):
        with patch.object(ml_cli, "read_only_session") as connect, self.assertRaises(MLFailure):
            ml_cli.run(ml_cli.parser().parse_args(["export-entries", "--mapping", "SYNTHETIC",
                                                "--start", "2030-01-01", "--end", "2030-01-01"]))
        connect.assert_not_called()

    def test_entry_cli_prints_counts_without_current_results_or_individual_records(self):
        mapping_path = self.root / "mapping.json"
        safety.write_json(mapping_path, self.mapping)
        runner = MagicMock()
        runner.iter_batches.return_value = [self.rows]
        connection = MagicMock()
        connection.return_value.__enter__.return_value = runner
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), patch.object(ml_cli, "_password", return_value="SYNTHETIC_PASSWORD"), \
                patch.object(ml_cli, "read_only_session", connection), \
                patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(self.mapping)}, {})), redirect_stdout(output):
            code = ml_cli.main(["export-entries", "--execute-local", "--mapping", str(mapping_path),
                                "--start", "2020-01-01", "--end", "2021-01-01"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["source_status_counts"]["se_runners"], {"2": 8})
        self.assertTrue(result["current_results_excluded"])
        self.assertNotIn("SYNTHETIC_PASSWORD", output.getvalue())
        self.assertNotIn("SYNTHETIC_HORSE", output.getvalue())

    def test_non_final_entries_use_existing_win_model_and_reach_simulation(self):
        try:
            model.libraries()
        except MLFailure:
            self.skipTest("optional_ml_libraries_not_installed")
        all_rows = synthetic_rows(20)
        bundle = model.fit_models(all_rows, rounds=10)
        model_dir = safety.new_artifact("models")
        model.save_models(bundle, model_dir)
        directory, _ = self.export()
        history_dir = safety.new_artifact("datasets")
        history = data.write_synthetic_dataset(all_rows[:-8], history_dir)
        place_hash = safety.digest(model_dir / "place.lgb.txt")
        loaded = model.load_models(model_dir)
        with patch.object(model, "load_models", return_value=loaded), \
                patch.object(loaded["boosters"]["place"], "predict", side_effect=AssertionError("place_must_not_run")):
            args = win_sim_cli.parser().parse_args(["predict-win", "--model", str(model_dir),
                "--entries", str(directory / "runners.csv"), "--history", str(history)])
            output = win_sim_cli.run(args)
        self.assertEqual(output["runner_count"], 8)
        predictions_dir = self.current / "predictions" / output["artifact_id"]
        predictions, _ = local.load_predictions(predictions_dir)
        self.assertEqual(local.read(predictions_dir / "manifest.json")["entries_source"], data.ENTRY_SOURCE)
        self.assertEqual(place_hash, safety.digest(model_dir / "place.lgb.txt"))
        prediction = predictions[0]
        day = date(int(prediction.race[0]), int(prediction.race[1][:2]), int(prediction.race[1][2:]))
        start = datetime(day.year, day.month, day.day, 15, tzinfo=core.JST)
        schedule = core.Schedule(prediction.race, start, start.replace(hour=9), tuple(prediction.probabilities), True)
        snapshot = core.Snapshot(prediction.race, start - timedelta(minutes=12), start - timedelta(minutes=11),
                                 {horse: "10" for horse in prediction.probabilities}, verified=True)
        chosen = min(prediction.probabilities, key=lambda horse: (-prediction.probabilities[horse], horse))
        outcome = core.Outcome(prediction.race, "normal_final", {chosen: 620}, True, True)
        policy = core.WinPolicy(evaluation_start=day.isoformat(), evaluation_end=day.isoformat(),
                                model_fit_end=bundle["metadata"]["split"]["validation"]["end"])
        result = core.run_simulation(predictions, [schedule], [snapshot], [outcome], policy=policy)
        self.assertEqual(result["summary"]["ai"]["net_profit_yen"], 520)
        self.assertEqual(result["summary"]["comparisons"]["all_common_eligible_races"]["race_count"], 1)
