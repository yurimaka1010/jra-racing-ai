"""One-day trial uses mock data only, and never registers tasks in tests."""
from contextlib import redirect_stdout
from datetime import date, datetime
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import morning_trial as trial
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.ml import safety
from jra_racing_ai.win_sim import core
from test_ml_v03 import race_rows

DAY = date(2026, 10, 17)
CLOCK = datetime(2026, 10, 17, 7, tzinfo=core.JST)


def policy():
    return core.WinPolicy(temperature=2., model_fit_end="2025-01-19", calibration_end="2025-02-02",
        calibration_source="independent_validation", evaluation_start=DAY.isoformat(), evaluation_end=DAY.isoformat(),
        timing_basis="jra_publication_time")


class MorningPureTests(unittest.TestCase):
    def setUp(self):
        self.rows=race_rows(DAY)
        self.race=tuple(self.rows[0][key] for key in trial.RACE_KEY)
        self.schedule=core.Schedule(self.race, CLOCK.replace(hour=12), None, tuple(row["horse_number"] for row in self.rows),True,"ra_advance_card")

    def test_complete_pre_race_card_eligible_without_imputing_features(self):
        rows,starts,skipped=trial.eligible_rows(self.rows,[self.schedule],policy(),CLOCK)
        self.assertEqual(rows,self.rows)
        self.assertEqual(starts[self.race].hour,12)
        self.assertFalse(skipped)

    def test_missing_schedule_declared_runner_mismatch_or_basic_field_skips_whole_race(self):
        for rows,schedules in ((self.rows,[]),([dict(r,declared_runners="9") for r in self.rows],[self.schedule]),
                               ([dict(r,weight_kg="") for r in self.rows],[self.schedule])):
            valid,_,skipped=trial.eligible_rows(rows,schedules,policy(),CLOCK)
            self.assertFalse(valid)
            self.assertEqual(sum(skipped.values()),1)

    def test_late_prediction_never_backdated_or_generated_after_deadline(self):
        valid,_,skipped=trial.eligible_rows(self.rows,[self.schedule],policy(),CLOCK.replace(hour=11,minute=50))
        self.assertFalse(valid)
        self.assertEqual(skipped["prediction_deadline_passed"],1)

    def test_probe_is_bounded_readonly_and_no_real_rows_returned(self):
        runner=MagicMock()
        runner.aggregate.return_value=[{"runners":8,"races":1,"non_card_rows":0}]
        with patch.object(trial.data,"check_mapping_schema"),patch.object(trial.data,"export_query",return_value=("SELECT 1",())):
            self.assertEqual(trial.probe(runner,{},DAY,{"tables":{}})["runners"],8)
        self.assertEqual(runner.aggregate.call_args.kwargs["max_rows"],1)
        self.assertIn("count(DISTINCT",runner.aggregate.call_args.args[1])

    def test_probe_empty_or_postrace_categories_stops(self):
        for value in ({"runners":0,"races":0,"non_card_rows":0},{"runners":8,"races":1,"non_card_rows":1}):
            runner=MagicMock()
            runner.aggregate.return_value=[value]
            with patch.object(trial.data,"check_mapping_schema"),patch.object(trial.data,"export_query",return_value=("SELECT 1",())),self.assertRaises(safety.MLFailure):
                trial.probe(runner,{},DAY,{"tables":{}})

    def test_missing_cards_can_be_pending_only_in_setup_probe_and_log_counts(self):
        for allow_missing in (False, True):
            runner=MagicMock()
            runner.aggregate.side_effect=[{"runners":0,"races":0,"non_card_rows":0}], [{"ra_rows":0,"se_rows":0}]
            report={}
            with patch.object(trial.data,"check_mapping_schema"),patch.object(trial.data,"export_query",return_value=("SELECT 1",())):
                if allow_missing:
                    counts=trial.probe(runner,{},DAY,{"tables":{}},report=report,allow_missing=True)
                    self.assertEqual(counts["runners"],0)
                else:
                    with self.assertRaises(safety.MLFailure):
                        trial.probe(runner,{},DAY,{"tables":{}},report=report)
            self.assertEqual(report["data_status"],"ra_se_target_date_missing")
            self.assertEqual(report["source_date_counts"],{"ra_rows":0,"se_rows":0})

    def test_setup_probe_does_not_relax_missing_half_or_unmatched_or_postrace_data(self):
        for source_counts in ({"ra_rows":0,"se_rows":8},{"ra_rows":1,"se_rows":0},
                              {"ra_rows":1,"se_rows":8},{"ra_rows":None,"se_rows":8}):
            runner=MagicMock()
            runner.aggregate.side_effect=[{"runners":0,"races":0,"non_card_rows":0}], [source_counts]
            with patch.object(trial.data,"check_mapping_schema"),patch.object(trial.data,"export_query",return_value=("SELECT 1",())),self.assertRaises(safety.MLFailure):
                trial.probe(runner,{},DAY,{"tables":{}},report={},allow_missing=True)
        runner=MagicMock()
        runner.aggregate.return_value=[{"runners":8,"races":1,"non_card_rows":1}]
        with patch.object(trial.data,"check_mapping_schema"),patch.object(trial.data,"export_query",return_value=("SELECT 1",())),self.assertRaises(safety.MLFailure):
            trial.probe(runner,{},DAY,{"tables":{}},report={},allow_missing=True)


class MorningRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.root=self.base/"trial"
        self.v01=self.base/"v01"
        self.root.mkdir()
        self.v01.mkdir()
        self.model_id="a"*32
        model_path=self.v01/"models"/self.model_id
        model_path.mkdir(parents=True)
        (model_path/"model.json").write_text("{}",encoding="utf-8")
        self.rows=race_rows(DAY)
        self.race=tuple(self.rows[0][key] for key in trial.RACE_KEY)
        self.raw_predictions=[core.PredictionRace(self.race,{row["horse_number"]:(.81 if i==0 else .01) for i,row in enumerate(self.rows)})]
        self.bundle={"metadata":{"history_lag_days":7},"boosters":{"win":MagicMock()}}
        source=({"model_id":self.model_id},{"history_lag_days":7},self.bundle,policy(),[])
        self.patches=[patch.object(trial,"ROOT",self.root),patch.object(safety,"ML_ROOT",self.v01),
            patch.object(trial,"now",return_value=CLOCK),patch.object(trial,"load_source",return_value=source),
            patch.dict(os.environ,{"PGPASSWORD":"SYNTHETIC_PASSWORD"})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def call(self,arguments):
        out=io.StringIO()
        with redirect_stdout(out):
            code=trial.main(arguments)
        return code,json.loads(out.getvalue())

    def test_offline_preflight_never_connects_or_requests_password(self):
        with patch.object(trial,"read_only_session") as db,patch.dict(os.environ,{},clear=True):
            code,result=self.call(["--preflight"])
        self.assertEqual(code,0)
        self.assertFalse(result["database_accessed"])
        db.assert_not_called()
        self.assertTrue((Path(result["run_directory"])/"run.json").is_file())

    def test_unattended_missing_secret_logs_fixed_failure_without_prompt(self):
        with patch.dict(os.environ,{},clear=True),patch.object(trial,"read_only_session") as db:
            code,result=self.call([])
        self.assertEqual(code,2)
        self.assertEqual(result["reason"],"needs_authentication")
        db.assert_not_called()

    def test_database_error_has_no_secret_or_exception_text_and_no_prediction_csv(self):
        with patch.object(trial,"read_only_session",side_effect=SafeAuditError("connection_failed")):
            code,result=self.call([])
        self.assertEqual(code,2)
        self.assertEqual(result["reason"],"connection_failed")
        self.assertFalse(list(self.root.rglob("predictions.csv")))
        self.assertNotIn("SYNTHETIC_PASSWORD",json.dumps(result))

    def test_count_only_probe_pending_card_does_not_export_or_predict(self):
        def pending(db, mapping, day, catalog, *, report, allow_missing):
            self.assertTrue(allow_missing)
            report.update(data_status="ra_se_target_date_missing",source_date_counts={"ra_rows":0,"se_rows":0})
            return {"runners":0,"races":0,"non_card_rows":0}
        with patch.object(trial,"read_only_session"),patch.object(trial.data,"inspect_sources",return_value=({},{})), \
             patch.object(trial,"probe",side_effect=pending),patch.object(trial,"export") as export, \
             patch.object(trial.v03,"predict_win") as predict:
            code,result=self.call(["--probe"])
        self.assertEqual(code,0)
        self.assertFalse(result["ready_for_prediction"])
        self.assertEqual(result["data_status"],"ra_se_target_date_missing")
        export.assert_not_called()
        predict.assert_not_called()

    def test_actual_run_with_missing_card_stops_before_export_and_writes_failure_log(self):
        def missing(db, mapping, day, catalog, *, report, allow_missing):
            self.assertFalse(allow_missing)
            report.update(data_status="ra_se_target_date_missing",source_date_counts={"ra_rows":0,"se_rows":0})
            raise safety.MLFailure("insufficient_history")
        with patch.object(trial,"read_only_session"),patch.object(trial.data,"inspect_sources",return_value=({},{})), \
             patch.object(trial,"probe",side_effect=missing),patch.object(trial,"export") as export:
            code,result=self.call([])
        self.assertEqual(code,2)
        self.assertEqual(result["data_status"],"ra_se_target_date_missing")
        self.assertTrue((Path(result["run_directory"])/"run.json").exists())
        self.assertFalse(list(self.root.rglob("predictions.csv")))
        export.assert_not_called()

    def test_success_exports_calibrated_t2_csv_and_aggregate_log_with_new_run_ids(self):
        ra=[{"race_key":self.race,"record_type":"RA","status":"2","created":"20261016","start":"1200","previous_start":"0000","oversized":False}]
        entry_meta={"source_status_counts":{"ra_races":{"2":1},"se_runners":{"2":8}},"races":1}
        with patch.object(trial,"read_only_session"),patch.object(trial.data,"inspect_sources",return_value=({},{})), \
            patch.object(trial,"probe",return_value={"runners":8,"races":1,"non_card_rows":0}), \
            patch.object(trial,"export",side_effect=[(race_rows(date(2026,10,1)),{}),(self.rows,entry_meta)]), \
            patch.object(trial.publication,"_columns",return_value={}),patch.object(trial.local,"_rows",return_value=ra), \
            patch.object(trial.v03,"predict_win",return_value=(self.raw_predictions,{"rows":8})):
            code,result=self.call([])
        self.assertEqual(code,0)
        self.assertEqual(result["race_count"],1)
        self.assertEqual(result["temperature"],2)
        with Path(result["predictions_csv"]).open(encoding="utf-8-sig",newline="") as stream:
            rows=list(csv.DictReader(stream))
        self.assertAlmostEqual(sum(float(row["win_probability"]) for row in rows),1.)
        self.assertAlmostEqual(float(rows[0]["win_probability"]),.5625)
        self.assertEqual(rows[0]["predicted_at_jst"],CLOCK.isoformat())
        self.assertNotIn("SYNTHETIC_HORSE_",json.dumps(result))
        self.assertFalse(result["live_purchase_enabled"])

    def test_task_entrypoints_do_not_pipe_confirmation_tokens_or_contain_password_literals(self):
        project=Path(__file__).resolve().parents[1]
        runner=(project/"scripts/run_20261017_trial.ps1").read_text(encoding="utf-8")
        setup=(project/"scripts/prepare_20261017_trial.ps1").read_text(encoding="utf-8")
        self.assertIn("ZeroFreeBSTR",runner)
        self.assertIn("ConvertFrom-SecureString",setup)
        self.assertIn("-LogonType Interactive",setup)
        self.assertNotIn("CONFIRM_RAW_O1",runner+setup)
