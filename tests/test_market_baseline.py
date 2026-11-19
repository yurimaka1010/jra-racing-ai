"""Synthetic measurement tests. No model training, network or real DB access."""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import date
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import market_baseline_cli as cli
from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.ml import data, features, market, safety, v03
from jra_racing_ai.win_sim import core, local, publication
from test_ml_v03 import race_rows

RACE = ("2030", "0105", "05", "01", "01", "01")


def prediction(race=RACE):
    return core.PredictionRace(race, {"01": .4, "02": .3, "03": .3})


def raw(*, race=RACE, category="5", source="jvd_o1", stamp="01051149", amounts=("0020", "0040", "0040")):
    payload = "".join(f"{i+1:02d}{amount}{i+1:02d}" for i, amount in enumerate(amounts)) + " " * (224 - 8 * len(amounts))
    return {"race_key": race, "record_type": "O1", "status": category, "sale": "7", "payload": payload,
            "oversized": False, "source_table": source, "published": stamp}


def ra(race=RACE):
    return {"race_key": race, "record_type": "RA", "status": "2", "start": "1200", "previous_start": "0000",
            "created": str(int(race[0])-1) + "1231", "oversized": False}


class MarketPureTests(unittest.TestCase):
    def test_inverse_odds_normalization_not_temperature_or_odds_as_probability(self):
        p, total = market.implied_probabilities({"01": 2, "02": 4, "03": 4})
        self.assertEqual(float(total), 1)
        self.assertEqual([float(p[h]) for h in p], [.5, .25, .25])
        p, total = market.implied_probabilities({"01": 2, "02": 3, "03": 3})
        self.assertGreater(float(total), 1)
        self.assertAlmostEqual(float(sum(p.values())), 1)

    def test_invalid_capped_or_missing_numeric_prices_are_never_imputed(self):
        for value in (0, -1, .9, "999.9", None, "----", "nan", True):
            with self.subTest(value=value), self.assertRaises(core.SimulationError):
                market.implied_probabilities({"01": value, "02": 4, "03": 4})

    def test_binary_log_loss_brier_and_rank_match_manual_calculation(self):
        q, _ = market.select_final(RACE, [raw()], prediction().probabilities)
        result = market.measure([prediction()], {RACE: "01"}, [q], core.WinPolicy())
        metric = result["metrics"]["market"]
        self.assertAlmostEqual(metric["log_loss"], (-math.log(.5)-2*math.log(.75))/3)
        self.assertAlmostEqual(metric["brier_score"], .125)
        self.assertEqual(metric["top1_accuracy_percent"], 100)
        self.assertEqual(metric["winner_mean_rank"], 1)
        self.assertAlmostEqual(result["comparison_with_unmeasured_estimate"]["difference"], metric["log_loss"]-.21)

    def test_spearman_uses_average_ties_and_handles_constant_vectors(self):
        a = {"01": 3, "02": 2, "03": 1}
        self.assertAlmostEqual(market.spearman(a, a), 1)
        self.assertAlmostEqual(market.spearman(a, {h: -x for h,x in a.items()}), -1)
        self.assertAlmostEqual(market.spearman(a, {"01": 2, "02": 1, "03": 1}), math.sqrt(3)/2)
        self.assertIsNone(market.spearman(a, {h: 1 for h in a}))

    def test_pre_race_minute_boundary_and_future_odds_excluded(self):
        rows = [raw(category="1", stamp=s, amounts=a) for s,a in
                (("01051149", ("0020","0040","0040")), ("01051150", ("0030","0030","0030")),
                 ("01051201", ("0010","0990","0990")))]
        cohorts, reasons = market.select_cohorts([prediction()], rows, [ra()], core.WinPolicy(timing_basis="jra_publication_time"))
        chosen = cohorts["pre_race_publication_time"][0]
        self.assertEqual(float(chosen.odds["01"]), 2)
        self.assertEqual(chosen.minutes_before_start, 11)
        self.assertEqual(reasons["pre_race"]["excluded_race_count"], 0)

    def test_latest_incomplete_snapshot_does_not_fall_back_to_older_valid_prices(self):
        rows = [raw(category="1", stamp="01051140"), raw(category="1", amounts=("0020","0040"))]
        cohorts, reasons = market.select_cohorts([prediction()], rows, [ra()], core.WinPolicy(timing_basis="jra_publication_time"))
        self.assertFalse(cohorts["pre_race_publication_time"])
        self.assertEqual(reasons["pre_race"]["reasons"], {"roster_mismatch": 1})

    def test_final_record_does_not_acquire_invented_publication_clock(self):
        q, reason = market.select_final(RACE, [raw(stamp="invalid")], prediction().probabilities)
        self.assertIsNone(reason)
        self.assertIsNone(q.published_at)
        self.assertEqual(q.basis, "final_category_5")

    def test_final_categories_are_prioritized_and_never_mixed_in_measurement(self):
        q, _ = market.select_final(RACE, [raw(category="3"), raw(category="5")], prediction().probabilities)
        self.assertEqual(q.category, "5")
        with self.assertRaises(core.SimulationError):
            market.measure([prediction()], {RACE: "01"}, [q, replace(q,basis="final_category_4")], core.WinPolicy())

    def test_duplicate_final_rows_conflicting_sources_deletion_and_no_sale_excluded(self):
        for rows, reason in (([raw(),raw()], "final_odds_conflict"),
            ([raw(),raw(source="apd_sokuho_o1",amounts=("0030","0040","0040"))],"final_odds_conflict"),
            ([raw(),raw(category="0")],"final_status_unconfirmed"),
            ([dict(raw(),sale="0")],"odds_invalid")):
            with self.subTest(reason=reason):
                self.assertEqual(market.select_final(RACE,rows,prediction().probabilities)[1],reason)

    def test_identical_cross_table_copies_deduplicated(self):
        q, reason = market.select_final(RACE,[raw(),raw(source="apd_sokuho_o1")],prediction().probabilities)
        self.assertIsNone(reason)
        self.assertEqual(q.source,"jvd_o1")

    def test_explicit_cancelled_outside_roster_removed_but_target_roster_kept(self):
        row=raw(amounts=("0020","0040","0040","----","****"))
        q,reason=market.select_final(RACE,[row],prediction().probabilities)
        self.assertIsNone(reason)
        self.assertEqual(q.cancelled_outside_roster,2)
        self.assertEqual(set(q.odds),set(prediction().probabilities))
        self.assertEqual(market.select_final(RACE,[raw(amounts=("0020","----","0040"))],prediction().probabilities)[1],"odds_invalid")

    def test_non_cancelled_extra_runner_or_bad_block_is_not_silently_removed(self):
        for row in (raw(amounts=("0020","0040","0040","0050")), dict(raw(),payload="0"*223),
                    dict(raw(),payload="０"*224), dict(raw(),payload=None)):
            self.assertIsNotNone(market.select_final(RACE,[row],prediction().probabilities)[1])

    def test_missing_odds_excludes_whole_race_and_reports_runner_count(self):
        cohorts, reasons = market.select_cohorts([prediction()], [], [], core.WinPolicy())
        self.assertFalse(any(cohorts.values()))
        self.assertEqual(reasons["final"]["excluded_runner_count"],3)
        self.assertEqual(reasons["pre_race"]["reasons"],{"odds_history_unavailable":1})

    def test_calibration_ratio_zero_wins_empty_bins_and_same_existing_boundaries(self):
        q,_=market.select_final(RACE,[raw()],prediction().probabilities)
        result=market.measure([prediction()],{RACE:"01"},[q],core.WinPolicy())
        bins=result["calibration"]["market"]
        self.assertEqual([b["probability_band"] for b in bins],list(market.BIN_NAMES))
        self.assertEqual(bins[0]["ratio_status"],"empty_bin")
        self.assertIsNone(bins[0]["predicted_to_observed_ratio"])
        self.assertAlmostEqual(bins[5]["predicted_to_observed_ratio"],1)

    def test_calibration_zero_observations_does_not_become_infinite_ratio(self):
        result={"low_probability_calibration":[{"bin":0,"count":5,"wins":0,"mean_probability":.005,"observed_rate":0}]}
        self.assertEqual(market.calibration_table(result)[0]["ratio_status"],"no_observed_wins")
        self.assertIsNone(market.calibration_table(result)[0]["predicted_to_observed_ratio"])

    def test_comparison_recomputed_on_exact_market_subset(self):
        second=(*RACE[:-1],"02")
        q,_=market.select_final(RACE,[raw()],prediction().probabilities)
        result=market.measure([prediction(),prediction(second)],{RACE:"01",second:"03"},[q],core.WinPolicy())
        self.assertEqual(result["race_count"],1)
        self.assertEqual(result["runner_count"],3)
        self.assertEqual({x["count"] for x in result["metrics"].values()},{3})


class MarketLocalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.v01,self.v02,self.v03=(self.base/name for name in ("v01","v02","v03"))
        for path in (self.v01,self.v02,self.v03):
            path.mkdir()
        self.patches=[patch.object(safety,"ML_ROOT",self.v01),patch.object(local,"ROOT",self.v02),patch.object(cli,"ROOT",self.v03)]
        for item in self.patches:
            item.start()
        rows=race_rows(date(2025,2,15),count=3)+race_rows(date(2025,2,16),count=3)
        extra=[dict(row,race_bango="02",finish_position="") for row in rows[-3:]]
        rows+=extra
        dataset=safety.new_artifact("datasets")
        data.write_synthetic_dataset(rows,dataset)
        model_dir=safety.new_artifact("models")
        (model_dir/"win.lgb.txt").write_text("synthetic model not loaded",encoding="utf-8")
        self.train={"start":"2025-01-05","end":"2025-01-18","races":144,"rows":2237}
        safety.write_json(model_dir/"model.json",{"format_version":1,"model_kind":"lightgbm_rank_proxy_v01",
            "production_hr_settlement_enabled":False,"features":list(features.FEATURES),
            "artifacts":{"win.lgb.txt":safety.digest(model_dir/"win.lgb.txt")},
            "split":{"train":self.train,"validation":{"end":"2025-01-19"}},"dataset_sha256":safety.digest(dataset/"runners.csv")})
        model_hash=safety.digest(model_dir/"model.json")
        predictions=[]
        for day,number in ((15,"01"),(16,"01"),(16,"02")):
            predictions.append(prediction(("2025",f"02{day:02d}","05","01","01",number)))
        directory=local.artifact("predictions")
        local.write(directory/"predictions.json",{"format_version":2,"mode":"hypothetical_backtest","records":[
            {"entry_key":list(p.race)+[h],"wager":"win","probability":v} for p in predictions for h,v in p.probabilities.items()]})
        local.write(directory/"manifest.json",{"format_version":2,"mode":"hypothetical_backtest","pipeline_version":"0.3",
            "purpose":"heldout_probability_test","model_manifest_sha256":model_hash,
            "entries_sha256":safety.digest(dataset/"runners.csv"),"predictions_sha256":local.sha(directory/"predictions.json")})
        calibration={"temperature":2.,"calibration_start":"2025-02-01","calibration_end":"2025-02-02"}
        policy=core.WinPolicy(temperature=2.,model_fit_end="2025-01-19",calibration_end="2025-02-02",
            calibration_source="independent_validation",evaluation_start="2030-01-05",evaluation_end="2030-01-05",timing_basis="jra_publication_time")
        policy_dir=local.artifact("policies")
        local.write(policy_dir/"policy.json",local.policy_document(policy,model_hash=model_hash,calibration=calibration))
        winners=v03.single_winners(rows)
        evaluated=v03.probability_evaluation(predictions,winners,policy)
        run_dir=safety.new_artifact("evaluations",root=self.v03)
        self.path=run_dir/"run.json"
        safety.write_json(self.path,{"kind":"v03_first_run","status":"ok","model_id":model_dir.name,
            "dataset_id":dataset.name,"heldout_predictions_id":directory.name,"policy_id":policy_dir.name,
            "split":{"test":{"start":"2025-02-15","end":"2025-02-16","rows":9,"races":3}},
            "calibration":calibration,"heldout_test":{"v03_calibrated":evaluated}},root=self.v03)
        self.predictions=predictions[:2]
        self.source_paths=[p for root in (self.v01,self.v02,self.v03) for p in root.rglob("*") if p.is_file()]
        self.before={p:p.read_bytes() for p in self.source_paths}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def test_saved_predictions_labels_and_temperature_reproduce_exact_original_metrics(self):
        frozen=cli.load_frozen(self.path)
        self.assertEqual(len(frozen["predictions"]),2)
        self.assertEqual(frozen["full"]["model"]["count"],6)
        self.assertEqual(frozen["policy"].temperature,2.)

    def test_fake_db_measurement_saves_aggregate_report_and_keeps_source_artifacts_unchanged(self):
        args=cli.parser().parse_args(["--execute-local","--run",str(self.path)])
        rows=[raw(race=p.race) for p in self.predictions]
        output=io.StringIO()
        with patch.object(cli,"password",return_value="SYNTHETIC_PASSWORD"),patch.object(cli,"read_only_session"), \
                patch.object(cli,"fetch",return_value=(rows,[],[{"table":"jvd_o1","rows":2}])), \
                patch.object(cli,"review_format"),redirect_stdout(output):
            result=cli.run(args,{})
        self.assertEqual(result["measurement_status"],"complete")
        cohort=result["cohorts"]["final_category_5"]
        self.assertEqual(cohort["runner_count"],6)
        self.assertEqual(result["existing_label_selection"]["previously_excluded_races"],1)
        self.assertEqual({p:p.read_bytes() for p in self.source_paths},self.before)
        folder=Path(result["report_directory"])
        self.assertTrue((folder/"report.md").is_file())
        self.assertTrue((folder/"selected_inputs.local.json").is_file())
        summary=(folder/"summary.json").read_text(encoding="utf-8")
        self.assertNotIn('"race_key"',summary)
        self.assertNotIn("SYNTHETIC_PASSWORD",summary)
        self.assertNotIn("SYNTHETIC_HORSE_",summary)

    def test_no_odds_returns_unavailable_not_fabricated_point_two_one(self):
        args=cli.parser().parse_args(["--execute-local","--run",str(self.path)])
        with patch.object(cli,"password",return_value="SYNTHETIC_PASSWORD"),patch.object(cli,"read_only_session"), \
                patch.object(cli,"fetch",return_value=([],[],[])):
            result=cli.run(args,{})
        self.assertEqual(result["measurement_status"],"odds_unavailable")
        self.assertTrue(all(v["status"]=="unavailable" for v in result["cohorts"].values()))

    def test_tampered_prediction_file_stops_before_authentication(self):
        path=next((self.v02/"predictions").glob("*/predictions.json"))
        path.write_text("{}",encoding="utf-8")
        with patch.object(cli,"password") as auth,self.assertRaises(SafeAuditError):
            cli.load_frozen(self.path)
        auth.assert_not_called()

    def test_false_original_metric_stops(self):
        saved=safety.read_json(self.path,root=self.v03)
        saved["heldout_test"]["v03_calibrated"]["model"]["log_loss"]+=.01
        self.path.write_text(json.dumps(saved),encoding="utf-8")
        with self.assertRaises(safety.MLFailure):
            cli.load_frozen(self.path)

    def test_execution_and_limits_checked_before_auth(self):
        for changes in ({"execute_local":False},{"max_rows":100001},{"statement_timeout_ms":999}):
            args=cli.parser().parse_args(["--execute-local","--run",str(self.path)])
            for field,value in changes.items():
                setattr(args,field,value)
            with patch.object(cli,"password") as auth,self.assertRaises(safety.MLFailure):
                cli.run(args,{})
            auth.assert_not_called()

    def test_cli_error_has_no_private_exception_or_record(self):
        out=io.StringIO()
        with patch.object(cli,"run",side_effect=RuntimeError("SYNTHETIC_PASSWORD")),redirect_stdout(out):
            self.assertEqual(cli.main(["--execute-local"]),2)
        self.assertEqual(json.loads(out.getvalue()),{"status":"stopped","reason":"internal_error","stage":"preflight"})

    def test_existing_query_builder_uses_complete_keys_parameters_and_only_select(self):
        fields=publication.O1_FIELDS
        columns={name:{"type_is_builtin":True,"type_name":"varchar"} for name in (*RACE_KEY,*(p[0] for p in fields.values()))}
        sql,params=local.build_rows_query("jvd_o1",fields,columns,tuple(p.race for p in self.predictions),500,join_targets=True)
        validate_audit_sql(sql)
        self.assertNotIn("2025",sql)
        self.assertEqual(params[-1],501)
        for key in RACE_KEY:
            self.assertIn('requested."'+key+'"',sql)
        self.assertIn("SELECT ",sql)
        self.assertNotIn("UPDATE",sql)

    def test_db_fetch_reuses_bounded_adapter_and_only_reads_ra_for_intermediate_odds(self):
        races=tuple(p.race for p in self.predictions)
        fields=publication.O1_FIELDS
        columns={name:{"type_is_builtin":True,"type_name":"varchar"} for name in (*RACE_KEY,*(p[0] for p in fields.values()))}
        row=raw(race=races[0])
        row.pop("race_key")
        row.pop("source_table")
        row.update(dict(zip(RACE_KEY,races[0])))
        db=MagicMock()
        db.iter_batches.side_effect=[iter(()),iter(([row],))]
        with patch.object(publication,"_columns",return_value=columns) as structure:
            rows,ra_rows,sources=cli.fetch(db,races,max_rows=100,max_seconds=30)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["race_key"],races[0])
        self.assertFalse(ra_rows)
        self.assertEqual(structure.call_count,2)
        self.assertEqual([s["rows"] for s in sources],[0,1])
        for call in db.iter_batches.call_args_list:
            validate_audit_sql(call.args[1])
            self.assertIn("JOIN (VALUES",call.args[1])
            self.assertLessEqual(call.kwargs["max_rows"],100)
            self.assertLessEqual(call.kwargs["max_seconds"],30)

    def test_measurement_output_refuses_project_and_synchronised_paths(self):
        args=cli.parser().parse_args(["--execute-local","--run",str(Path(__file__).resolve().parent/"run.json")])
        with patch.object(cli,"password") as auth,self.assertRaises(safety.MLFailure):
            cli.run(args,{})
        auth.assert_not_called()
