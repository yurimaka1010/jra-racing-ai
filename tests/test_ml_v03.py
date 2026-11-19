"""Version 0.3 uses synthetic races, mocked DB reads and local temporary files only."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import ml_v03_cli as cli
from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.ml import data, features, model, safety, v03
from jra_racing_ai.ml.demo import synthetic_rows
from jra_racing_ai.win_sim import core, local, settle_saved
from test_saved_hr_settlement import native_hr, source as hr_source
from test_win_sim_v02 import example, policy, simulate


def catalog():
    values = {alias: set(RACE_KEY) for alias in ("ra", "se")}
    for field in (*data.REQUIRED, *v03.BASIC_FIELDS):
        column = "toroku_tosu" if field == "declared_runners" else data.CANDIDATES[field][0]
        values[data.SOURCES[field]].add(column)
    return {"tables": {alias: [{"name": column, "type_is_builtin": True, "type_name": "varchar"}
                              for column in sorted(columns)] for alias, columns in values.items()}}


def race_rows(day, *, count=8):
    rows = []
    for i in range(count):
        row = {name: "" for name in features.DATA_FIELDS}
        row.update(kaisai_nen=str(day.year), kaisai_tsukihi=day.strftime("%m%d"), keibajo_code="05",
                   kaisai_kai="01", kaisai_nichime="01", race_bango="01", horse_number=f"{i+1:02d}",
                   horse_id=f"SYNTHETIC_HORSE_{i+1:02d}", finish_position=str(i+1),
                   surface="SYNTHETIC_TURF", distance="1600", declared_runners=str(count),
                   frame=str(i % 8 + 1), age="4", sex="SYNTHETIC_F", weight_kg="55")
        rows.append(row)
    return rows


MAIN_DAYS = tuple(date(2025, 1, d) for d in (5, 6, 11, 12, 18, 19)) + tuple(date(2025, 2, d) for d in (1, 2, 15, 16))


class V03PureTests(unittest.TestCase):
    def test_mapping_enables_priority_fields_and_uses_declared_not_final_runner_count(self):
        candidate = v03.feature_mapping(catalog())
        self.assertFalse(candidate["reviewed"])
        self.assertEqual(candidate["columns"]["declared_runners"], "toroku_tosu")
        self.assertEqual(candidate["scales"]["weight_kg"], .1)
        for field in ("surface", "distance", "age", "weight_kg", "horse_id"):
            self.assertIsNotNone(candidate["columns"][field])
        self.assertIsNone(candidate["columns"]["going"])
        self.assertIsNone(candidate["columns"]["finish_time_seconds"])

    def test_missing_or_custom_priority_column_is_not_guessed(self):
        for remove in (True, False):
            value = catalog()
            item = next(c for c in value["tables"]["se"] if c["name"] == "futan_juryo")
            if remove:
                value["tables"]["se"].remove(item)
            else:
                item["type_is_builtin"] = False
            with self.assertRaises(safety.MLFailure):
                v03.feature_mapping(value)

    def test_mapping_review_is_required_and_does_not_echo_private_input(self):
        value = v03.feature_mapping(catalog())
        with patch("builtins.input", return_value="CONFIRM_V03_FEATURES"), patch("sys.stderr", io.StringIO()):
            self.assertTrue(v03.confirm_mapping(value)["reviewed"])
        with patch("builtins.input", return_value="SYNTHETIC_PRIVATE"), patch("sys.stderr", io.StringIO()) as stderr:
            with self.assertRaises(safety.MLFailure):
                v03.confirm_mapping(value)
        self.assertNotIn("SYNTHETIC_PRIVATE", stderr.getvalue())

    def test_warmup_restores_previous_race_without_current_or_future_outcomes(self):
        target = race_rows(date(2025, 1, 5))
        past = race_rows(date(2024, 12, 20))
        future = race_rows(date(2025, 1, 6))
        first = features.build_features(target, history=past+future)
        altered = [dict(row, finish_position="8", finish_time_seconds="999") for row in target]
        self.assertEqual(first, features.build_features(altered, history=past+future))
        self.assertEqual(first[0]["previous_finish"], 1)
        self.assertEqual(first[0]["days_since_previous"], 16)
        self.assertEqual(first[0]["past_starts"], 1)
        self.assertEqual(v03.coverage(first)["missing_rate"]["previous_finish"], 0)

    def test_priority_all_missing_or_no_past_history_stops_before_fit(self):
        rows = race_rows(date(2025, 1, 5))
        with self.assertRaises(safety.MLFailure):
            v03.check_training_coverage(v03.coverage(features.build_features(rows)))
        built = features.build_features(rows, history=race_rows(date(2024, 12, 20)))
        for row in built:
            row["weight_kg"] = None
        with self.assertRaises(safety.MLFailure):
            v03.check_training_coverage(v03.coverage(built))

    def test_ev_verification_reuses_formula_and_keeps_original_ledger(self):
        ledger = simulate()
        previous = deepcopy(ledger)
        result = v03.ev_check(ledger)
        self.assertEqual(result["formula_mismatch_count"], 0)
        self.assertEqual(ledger, previous)
        ledger["rows"][0]["runners"][2]["expected_return"] = 99
        with self.assertRaises(safety.MLFailure):
            v03.ev_check(ledger)

    def test_final_probability_evaluation_and_temperature_reuse(self):
        p, _, _, _ = example()
        winners = {p.race: "01"}
        before = v03.probability_evaluation([p], winners, policy())
        fitted = core.fit_temperature([p], winners, model_fit_end="2029-12-31", test_start="2030-02-01")
        after = v03.probability_evaluation([p], winners, policy(temperature=fitted["temperature"]))
        self.assertLessEqual(after["model"]["race_log_loss"], before["model"]["race_log_loss"])
        self.assertEqual(sum(item["count"] for item in after["low_probability_calibration"]), 3)

    def test_labels_require_complete_single_winner(self):
        rows = race_rows(date(2025, 1, 5))
        self.assertEqual(len(v03.single_winners(rows)), 1)
        rows[1]["finish_position"] = "1"
        self.assertFalse(v03.single_winners(rows))
        rows[1]["finish_position"] = ""
        self.assertFalse(v03.single_winners(rows))

    def test_uniform_tied_predictions_do_not_score_as_perfect_discrimination(self):
        p, _, _, _ = example()
        p = core.PredictionRace(p.race, {horse: 1/3 for horse in p.probabilities})
        result = v03.probability_evaluation([p], {p.race: "01"}, policy())
        self.assertAlmostEqual(result["model"]["top1_accuracy_percent"], 100/3)
        self.assertEqual(result["model"]["winner_mean_rank"], 2)

    def test_disjoint_periods_and_embargo_checked_before_auth(self):
        args = cli.parser().parse_args(["first-run", "--execute-local"])
        _, days = cli.validate_args(args)
        self.assertGreater(days["test_start"], days["calibration_end"]+timedelta(days=7))
        args.calibration_start = args.valid_end
        with self.assertRaises(safety.MLFailure):
            cli.validate_args(args)

    def test_current_target_is_removed_from_cached_final_history(self):
        rows = race_rows(date(2025, 1, 5))
        bundle = {"metadata": {"history_lag_days": 7, "categories": {}}, "boosters": {"win": MagicMock()}}
        bundle["boosters"]["win"].predict.return_value = [.125]*8
        with patch.object(model, "frame", return_value=None):
            predictions, result = v03.predict_win(bundle, rows, rows+race_rows(date(2024, 12, 20)))
        self.assertEqual(result["missing_count"]["previous_finish"], 0)
        self.assertEqual(len(predictions), 1)

    def test_reference_comparison_uses_same_eligible_races_for_both_versions(self):
        old = simulate([example(), example(number="02")])
        new = deepcopy(old)
        new["rows"][1]["common_settlement_eligible"] = False
        value = v03.comparison(old, new)
        self.assertEqual(value["common_race_count"], 1)
        self.assertEqual(value["v02_ai"]["investment_yen"], value["v03_ai"]["investment_yen"])


class V03LocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_rows = synthetic_rows(days=24)
        cls.old_bundle = model.fit_models(cls.old_rows, rounds=10)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.v01, self.v02, self.v03 = (self.base/name for name in ("v01", "v02", "v03"))
        for path in (self.v01, self.v02, self.v03):
            path.mkdir()
        self.patches = [patch.object(safety, "ML_ROOT", self.v01), patch.object(local, "ROOT", self.v02), patch.object(cli, "ROOT", self.v03)]
        for item in self.patches:
            item.start()
        self.args = cli.parser().parse_args(["first-run", "--execute-local", "--rounds", "10"])
        old_directory = safety.new_artifact("datasets")
        data.write_synthetic_dataset(self.old_rows, old_directory)
        metadata = safety.read_json(old_directory / "dataset.json")
        mapping = dict(v03.feature_mapping(catalog()), reviewed=True)
        metadata["mapping"] = mapping
        (old_directory / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        bundle = {**self.old_bundle, "metadata": dict(self.old_bundle["metadata"], dataset_sha256=metadata["csv_sha256"])}
        model_directory = self.v01 / "models" / cli.OLD_MODEL
        model_directory.mkdir(parents=True)
        model.save_models(bundle, model_directory)
        self.old_hash = safety.digest(model_directory / "model.json")
        self.place_hash = safety.digest(model_directory / "place.lgb.txt")
        p, s, o, outcome = example()
        self.reference = {"ledger": simulate(), "policy_document": local.policy_document(policy(), model_hash=self.old_hash),
                          "outcomes": [outcome], "schedules": [s], "snapshots": [o],
                          "ledger_sha256": "SYNTHETIC_FROZEN_LEDGER", "manifest_sha256": "SYNTHETIC_FROZEN_MANIFEST"}
        old_predictions = self.v02 / "predictions" / cli.OLD_PREDICTIONS
        old_predictions.mkdir(parents=True)
        local.write(old_predictions / "predictions.json", {"format_version": 2, "mode": "hypothetical_backtest",
            "records": [{"entry_key": list(p.race)+[h], "wager": "win", "probability": q} for h,q in p.probabilities.items()]})
        local.write(old_predictions / "manifest.json", {"format_version": 2, "mode": "hypothetical_backtest",
                    "model_manifest_sha256": self.old_hash, "predictions_sha256": local.sha(old_predictions / "predictions.json")})
        self.main_rows = [row for day in MAIN_DAYS for row in race_rows(day)]
        self.history_rows = race_rows(date(2024, 12, 20)) + self.main_rows
        self.reference_rows = race_rows(date(2030, 1, 5), count=3)
        self.progress = {}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def run_first(self):
        runner = MagicMock()
        sets = []
        for index, rows in enumerate((self.history_rows, self.main_rows, self.reference_rows)):
            raw = [dict(row, weight_kg="550") for row in rows]
            if index == 2:
                raw = [dict(row, _ra_status="2", _se_status="2") for row in raw]
            sets.append(iter((raw,)))
        runner.iter_batches.side_effect = sets
        with patch.object(cli, "password", return_value="SYNTHETIC_PASSWORD"), \
                patch.object(cli, "read_only_session") as session, \
                patch.object(data, "inspect_sources", return_value=(catalog(), {})), \
                patch.object(v03, "confirm_mapping", side_effect=lambda value: dict(value, reviewed=True)), \
                patch.object(v03, "reference_inputs", return_value=self.reference), patch("sys.stderr", io.StringIO()):
            session.return_value.__enter__.return_value = runner
            result = cli.first_run(self.args, self.progress)
        return result

    def test_full_fake_db_train_calibrate_predict_settle_and_preserve_v02(self):
        result = self.run_first()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["v02_preserved"])
        self.assertNotEqual(result["model_id"], cli.OLD_MODEL)
        for name in v03.PRIORITY_FEATURES+("previous_finish",):
            self.assertEqual(result["feature_coverage"]["v03_train"]["missing_count"][name], 0)
        self.assertEqual(safety.digest(self.v01/"models"/cli.OLD_MODEL/"model.json"), self.old_hash)
        self.assertEqual(safety.digest(self.v01/"models"/cli.OLD_MODEL/"place.lgb.txt"), self.place_hash)
        self.assertEqual(result["reference_summary"]["ai"]["unsettled_count"], 0)
        self.assertGreater(result["heldout_test"]["v03_calibrated"]["races"], 0)
        self.assertLessEqual(result["calibration_metrics"]["after"]["model"]["race_log_loss"],
                             result["calibration_metrics"]["before"]["model"]["race_log_loss"])
        path = self.v01/"models"/result["model_id"]
        self.assertFalse((path/"place.lgb.txt").exists())
        model.load_models(path, required_targets=("win",))
        with self.assertRaises(safety.MLFailure):
            model.load_models(path)
        saved = safety.read_json(Path(result["run_directory"])/"run.json", root=self.v03)
        self.assertEqual(saved["artifact_id"], result["artifact_id"])
        self.assertTrue((Path(result["run_directory"])/"report.md").is_file())
        dataset, _ = data.read_dataset(self.v01/"datasets"/result["dataset_id"]/"runners.csv")
        self.assertEqual(float(dataset[0]["weight_kg"]), 55)
        text = json.dumps(result)
        self.assertNotIn("SYNTHETIC_HORSE_", text)
        self.assertNotIn("SYNTHETIC_PASSWORD", text)
        self.assertNotIn('"race_key"', text)

    def test_single_win_model_has_four_nonoverlapping_periods(self):
        rows = self.main_rows
        bundle = model.fit_models(rows, history=race_rows(date(2024,12,20)), targets=("win",),
            train_end=date(2025,1,18), valid_end=date(2025,1,19), calibration_start=date(2025,2,1),
            calibration_end=date(2025,2,2), test_start=date(2025,2,15), test_end=date(2025,2,16), rounds=5)
        self.assertEqual(set(bundle["boosters"]), {"win"})
        groups = list(bundle["split"].values())
        for i, group in enumerate(groups):
            for other in groups[i+1:]:
                self.assertFalse(set(group)&set(other))
        self.assertTrue(all(bundle["metadata"]["missing_features"]["train"][field] == 0 for field in v03.PRIORITY_FEATURES))

    def test_bad_periods_and_no_execution_do_not_authenticate(self):
        for changes in ({"execute_local":False}, {"test_start":"2025-01-01"}, {"max_history_rows":500001}):
            args=deepcopy(self.args)
            for key,value in changes.items():
                setattr(args,key,value)
            with patch.object(cli,"password") as secret:
                with self.assertRaises(safety.MLFailure):
                    cli.first_run(args,{})
            secret.assert_not_called()

    def test_cli_failure_is_fixed_code_and_has_no_private_record(self):
        output=io.StringIO()
        with patch.object(cli,"first_run",side_effect=RuntimeError("SYNTHETIC_PASSWORD")), redirect_stdout(output):
            result=cli.main(["first-run","--execute-local"])
        self.assertEqual(result,2)
        self.assertEqual(json.loads(output.getvalue())["reason"],"internal_error")
        self.assertNotIn("SYNTHETIC_PASSWORD",output.getvalue())

    def test_existing_reference_inputs_require_frozen_hashes(self):
        with patch.object(v03.settle_saved,"load_saved",return_value=(self.v02,self.reference["ledger"],{},(),
                {"provenance":{"hr_review_sha256":"BAD","hr_source_sha256":"BAD"}})), \
                patch.object(local,"sha",return_value="ACTUAL"):
            with self.assertRaises(safety.MLFailure):
                v03.reference_inputs("synthetic")

    def test_additional_date_reuses_model_prediction_and_existing_hr_settlement(self):
        first = self.run_first()
        schedule = self.reference["schedules"][0]
        snapshot = self.reference["snapshots"][0]
        schedule_path, odds_path = self.v02/"schedules.json", self.v02/"odds.json"
        local.write(schedule_path, {"format_version":2,"source":"synthetic","records":[{
            "race_key":list(schedule.race),"scheduled_start":schedule.scheduled_start.isoformat(),
            "observed_at":schedule.observed_at.isoformat(),"horses":list(schedule.horses),"availability_verified":True}]})
        local.write(odds_path, {"format_version":2,"source":"synthetic","records":[{
            "race_key":list(snapshot.race),"published_at":snapshot.published_at.isoformat(),
            "observed_at":snapshot.observed_at.isoformat(),"availability_verified":True,"unit_verified":True,
            "odds":[{"horse_number":horse,"value":value} for horse,value in snapshot.odds.items()]}]})
        args=cli.parser().parse_args(["evaluate","--execute-local","--run",str(Path(first["run_directory"])/"run.json"),
            "--race-date","2030-01-05","--schedules",str(schedule_path),"--odds",str(odds_path)])
        runner=MagicMock()
        runner.iter_batches.side_effect=[iter(([dict(row,weight_kg="550") for row in self.history_rows],)),
            iter(([dict(row,weight_kg="550",_ra_status="2",_se_status="2") for row in self.reference_rows],))]
        with patch.object(cli,"password",return_value="SYNTHETIC_PASSWORD"), \
                patch.object(cli,"read_only_session") as session, \
                patch.object(data,"inspect_sources",return_value=(catalog(),{})), \
                patch.object(settle_saved,"fetch_hr",return_value=([native_hr()],hr_source(),{"table":"jvd_hr","rows":1})), \
                patch.object(settle_saved,"confirm_source"), patch("sys.stderr",io.StringIO()):
            session.return_value.__enter__.return_value=runner
            result=cli.evaluate(args,{})
        self.assertEqual(result["kind"],"v03_additional_evaluation")
        self.assertEqual(result["summary"]["ai"]["unsettled_count"],0)
        folder=self.v02/"simulations"/result["artifact_id"]
        self.assertIn("Version 0.3",(folder/"report.md").read_text(encoding="utf-8"))

    def test_changed_feature_mapping_cannot_be_used_with_trained_model(self):
        first=self.run_first()
        path=self.v01/"schema"/first["mapping_id"]/"mapping.reviewed.json"
        mapping=safety.read_json(path)
        mapping["scales"]["weight_kg"]=1
        path.chmod(0o666)
        path.write_text(json.dumps(mapping),encoding="utf-8")
        args=cli.parser().parse_args(["evaluate","--execute-local","--run",str(Path(first["run_directory"])/"run.json"),
                                      "--race-date","2030-01-05"])
        with patch.object(cli,"password") as secret:
            with self.assertRaises(safety.MLFailure):
                cli.evaluate(args,{})
        secret.assert_not_called()
