"""Version 0.2 local-only win simulation CLI; stdout contains aggregates only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import getpass
import json
import os
from pathlib import Path
import warnings

from .core import SafeAuditError
from .db_connection import ConnectionConfig, read_only_session
from .ml.safety import MLFailure
from .win_sim import core, formats, local


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise core.SimulationError("invalid_input")


def parser():
    result = Parser(description="Version 0.2 local win-only hypothetical simulation")
    commands = result.add_subparsers(dest="command", required=True, parser_class=Parser)
    demo = commands.add_parser("demo")
    demo.add_argument("--save", action="store_true")
    availability = commands.add_parser("availability")
    availability.add_argument("--execute-local", action="store_true")
    availability.add_argument("--sample-rows", type=int, default=100)
    availability.add_argument("--query-timeout-ms", type=int, default=5000)
    policy = commands.add_parser("make-policy")
    policy.add_argument("--model", type=Path, required=True)
    policy.add_argument("--test-start", required=True)
    policy.add_argument("--test-end", required=True)
    policy.add_argument("--threshold", default="0.05")
    policy.add_argument("--minutes-before-start", type=int, default=10)
    policy.add_argument("--probability-method", choices=("race_normalize", "identity"), default="race_normalize")
    policy.add_argument("--timing-basis", choices=("observed", "jra_publication_time"), default="observed")
    policy.add_argument("--validation-predictions", type=Path)
    policy.add_argument("--validation-payouts", type=Path)
    predict = commands.add_parser("predict-win")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--entries", type=Path, required=True)
    predict.add_argument("--history", type=Path)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("--policy", type=Path, required=True)
    simulate.add_argument("--predictions", type=Path, required=True)
    simulate.add_argument("--schedules", type=Path, required=True)
    simulate.add_argument("--odds", type=Path)
    simulate.add_argument("--payouts", type=Path)
    simulate.add_argument("--db-config", type=Path)
    simulate.add_argument("--execute-local", action="store_true")
    simulate.add_argument("--max-races", type=int, default=200)
    simulate.add_argument("--max-rows", type=int, default=50000)
    simulate.add_argument("--max-seconds", type=int, default=120)
    simulate.add_argument("--query-timeout-ms", type=int, default=30000)
    publication = commands.add_parser("publication-run")
    publication.add_argument("--policy", type=Path, required=True)
    publication.add_argument("--predictions", type=Path, required=True)
    publication.add_argument("--entries", type=Path, required=True)
    publication.add_argument("--execute-local", action="store_true")
    publication.add_argument("--schedules", type=Path)
    publication.add_argument("--payouts", type=Path)
    publication.add_argument("--odds-table", choices=formats.ODDS_TABLES, default="apd_sokuho_o1")
    publication.add_argument("--max-races", type=int, default=50)
    publication.add_argument("--max-rows", type=int, default=20000)
    publication.add_argument("--max-seconds", type=int, default=60)
    publication.add_argument("--query-timeout-ms", type=int, default=15000)
    settlement = commands.add_parser("settle-hr")
    settlement.add_argument("--simulation-id", required=True)
    settlement.add_argument("--execute-local", action="store_true")
    settlement.add_argument("--payout-source", type=Path)
    settlement.add_argument("--max-races", type=int, default=50)
    settlement.add_argument("--max-rows", type=int, default=200)
    settlement.add_argument("--max-seconds", type=int, default=30)
    settlement.add_argument("--query-timeout-ms", type=int, default=10000)
    return result


def password():
    if os.environ.get("PGPASSWORD"):
        return os.environ["PGPASSWORD"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("Local read-only DB password (hidden): ")
        if not value:
            raise core.SimulationError("needs_authentication")
        return value
    except (getpass.GetPassWarning, EOFError):
        raise core.SimulationError("needs_authentication") from None


def connection_config(timeout):
    config = ConnectionConfig.from_env(statement_timeout_ms=timeout)
    if config.host != "127.0.0.1" or config.port != 5432 or config.database != "pckeiba" or config.user != "jra_audit_reader":
        raise core.SimulationError("invalid_input")
    return config


def _freeze_files(directory, names):
    for name in names:
        (directory / name).chmod(0o444)


def run_publication(args):
    from dataclasses import replace
    from .win_sim import publication
    if not args.execute_local:
        raise core.SimulationError("explicit_execution_required")
    if not 1 <= args.max_races <= 50 or not 1 <= args.max_rows <= 50000 or not 1 <= args.max_seconds <= 120:
        raise core.SimulationError("input_limit")
    policy, original_document = local.load_policy(args.policy)
    policy = replace(policy, timing_basis=publication.TIMING_BASIS).validate()
    predictions, prediction_manifest = local.load_predictions(args.predictions,
        expected_model_hash=original_document["model_manifest_sha256"])
    if not 1 <= len(predictions) <= args.max_races:
        raise core.SimulationError("input_limit")
    for prediction in predictions:
        race = core.key(prediction.race)
        day = date(int(race[0]), int(race[1][:2]), int(race[1][2:]))
        if not date.fromisoformat(policy.evaluation_start) <= day <= date.fromisoformat(policy.evaluation_end):
            raise core.SimulationError("invalid_period")
    publication.validate_entries(args.entries, predictions, prediction_manifest)
    schedules = formats.parse_schedules(local.read(args.schedules)) if args.schedules else None
    policy_document = local.policy_document(policy, model_hash=original_document["model_manifest_sha256"],
                                            calibration=original_document.get("calibration"))
    policy_document["original_policy_sha256"] = local.sha(args.policy)
    policy_document["timing_evidence"] = "publication_only_actual_observation_and_purchase_availability_unverified"
    policy_directory = local.artifact("policies")
    local.write(policy_directory / "policy.json", policy_document)
    _freeze_files(policy_directory, ("policy.json",))
    config = connection_config(min(args.query_timeout_ms, args.max_seconds * 1000))
    secret = password()
    try:
        with read_only_session(config, secret) as db:
            ra_rows, odds_rows, db_status = publication.fetch(db, tuple(p.race for p in predictions),
                table=args.odds_table, max_rows=args.max_rows, max_seconds=args.max_seconds,
                include_ra=schedules is None)
    finally:
        secret = None
    if schedules is None:
        schedules, schedule_counts = publication.schedules_from_ra(ra_rows, predictions)
    else:
        schedule_counts = {"provided_schedule_records": len(schedules)}
    snapshots, failures, format_counts, examples = publication.decode_odds(odds_rows, source=args.odds_table)
    directory = local.artifact("simulations")
    review_path = directory / "raw_format_review.json"
    local.write(review_path, {"format_version": 2, "data_source": "local_jra_van_do_not_share",
                "timing_basis": publication.TIMING_BASIS, "source_table": args.odds_table,
                "field_mapping": {"odds_tansho": "examples[].raw_win_block"},
                "official_reference": str(local.SPEC_PDF), "expected_raw_format": formats.JV_O1_WIN,
                "unit_review_status": "pending_human_confirmation", "examples": examples})
    _freeze_files(directory, ("raw_format_review.json",))
    reviewed = any(item.reason is None and item.odds for item in snapshots)
    if reviewed:
        publication.confirm_raw_format(review_path)
        snapshots = [replace(item, verified=True) for item in snapshots]
    publication.save_replay_inputs(directory, schedules, snapshots, failures)
    outcomes = formats.parse_outcomes(local.read(args.payouts)) if args.payouts else []
    simulation = core.run_simulation(predictions, schedules, snapshots, outcomes, policy=policy, odds_failures=failures)
    local.write_report(directory, simulation, policy_payload=policy_document, provenance={
        "source": "local_inputs", "timing_basis": publication.TIMING_BASIS,
        "actual_observation_time_verified": False, "purchase_availability_verified": False,
        "odds_raw_format_and_scale_human_confirmed": bool(reviewed),
        "raw_format_review_sha256": local.sha(review_path), "schedule_counts": schedule_counts,
        "schedule_rule": ("provided_schedule_evidence" if args.schedules else
                          "advance_card_category2_created_before_race_day_no_postrace_start_replacement"),
        "publication_clock_precision": "minute_interval_upper_bound_before_decision",
        "replay_input_sha256": {name: local.sha(directory / name) for name in ("input-schedules.json", "input-odds.json")},
        "prediction_manifest_sha256": local.sha(args.predictions / "manifest.json"),
        "entries_sha256": local.sha(args.entries, root=local.old_io.ML_ROOT),
        "db_sources": db_status, "format_counts": format_counts,
        "history_scope": "original_training_history_recent_runs_not_refreshed",
        "payouts": "local_reference_checked_per_record" if args.payouts else "unavailable_no_hr_unit_assumption",
        "payout_reference_sha256": local.sha(args.payouts) if args.payouts else None})
    return {"status": "ok", "artifact_id": directory.name, "kind": "simulations", "wager": "win",
            "policy_id": policy_directory.name, "timing_basis": publication.TIMING_BASIS,
            "summary": simulation["summary"], "db_sources": db_status, "format_counts": format_counts,
            "schedule_counts": schedule_counts, "notice": simulation["notice"]}


def run(args):
    if args.command == "settle-hr":
        from .win_sim import settle_saved
        return settle_saved.run(args, connection_config=connection_config, password=password,
                                read_only_session=read_only_session)
    if args.command == "publication-run":
        return run_publication(args)
    if args.command == "demo":
        from .win_sim.demo import run_demo
        simulation, policy = run_demo()
        result = {"status": "ok", "source": "synthetic", "summary": simulation["summary"], "notice": simulation["notice"]}
        if args.save:
            directory = local.artifact("demo")
            local.write_report(directory, simulation, policy_payload=local.policy_document(policy, model_hash="synthetic"),
                               provenance={"source": "synthetic"})
            result["artifact_id"] = directory.name
        return result
    if args.command == "availability":
        if not args.execute_local:
            raise core.SimulationError("explicit_execution_required")
        if not 1 <= args.sample_rows <= 1000:
            raise core.SimulationError("input_limit")
        local.old_io.local_path(local.ROOT, root=local.ROOT)
        config = connection_config(args.query_timeout_ms)
        secret = password()
        try:
            with read_only_session(config, secret) as db:
                report = local.inspect_availability(db, sample_limit=args.sample_rows)
        finally:
            secret = None
        directory = local.artifact("inspection")
        local.write(directory / "availability.json", report)
        local.write(directory / "db-config.draft.json", local.draft_config())
        return {"status": "ok", "artifact_id": directory.name, "kind": "inspection",
                "tables": [{"table": item["table"], "status": item["status"], "column_count": len(item["columns"]),
                            "recent_sample": item.get("recent_sample")} for item in report["tables"]],
                "format_and_observed_time": "needs_local_confirmation"}
    if args.command == "make-policy":
        info, model_hash = local.model_info(args.model)
        fit_end = info["split"]["validation"]["end"]
        temperature, calibration_end, calibration = 1., None, None
        if bool(args.validation_predictions) != bool(args.validation_payouts):
            raise core.SimulationError("invalid_input")
        if args.validation_predictions:
            if args.probability_method != "race_normalize":
                raise core.SimulationError("invalid_policy")
            predictions, _ = local.load_predictions(args.validation_predictions, expected_model_hash=model_hash)
            outcomes = formats.parse_outcomes(local.read(args.validation_payouts))
            by_race = {}
            for outcome in outcomes:
                by_race.setdefault(outcome.race, []).append(outcome)
            winners = {}
            for prediction in predictions:
                matches = by_race.get(prediction.race, [])
                if (len(matches) == 1 and core.outcome_reason(matches[0], tuple(prediction.probabilities)) is None
                        and len(matches[0].payouts_yen_per_100) == 1):
                    winners[prediction.race] = next(iter(matches[0].payouts_yen_per_100))
            calibration = core.fit_temperature(predictions, winners, model_fit_end=fit_end, test_start=args.test_start)
            temperature, calibration_end = calibration["temperature"], calibration["calibration_end"]
        policy = core.WinPolicy(threshold=args.threshold, minutes_before_start=args.minutes_before_start,
                                probability_method=args.probability_method, temperature=temperature,
                                evaluation_start=args.test_start, evaluation_end=args.test_end, model_fit_end=fit_end,
                                calibration_end=calibration_end,
                                timing_basis=args.timing_basis,
                                calibration_source="independent_validation" if calibration else "fixed_not_fitted").validate()
        document = local.policy_document(policy, model_hash=model_hash, calibration=calibration)
        directory = local.artifact("policies")
        local.write(directory / "policy.json", document)
        _freeze_files(directory, ("policy.json",))
        return {"status": "ok", "artifact_id": directory.name, "kind": "policies", "wager": "win",
                "threshold": float(core.decimal(args.threshold)), "minutes_before_start": args.minutes_before_start,
                "probability_method": policy.probability_method, "calibration_source": policy.calibration_source,
                "timing_basis": policy.timing_basis}
    if args.command == "predict-win":
        from .ml import data, model
        from .ml.features import build_features, entry_identity
        bundle = model.load_models(args.model, required_targets=("win",))
        rows, entry_metadata = data.read_dataset(args.entries, allow_prediction_snapshot=True)
        history = data.read_dataset(args.history)[0] if args.history else []
        fit_end = date.fromisoformat(bundle["metadata"]["split"]["validation"]["end"])
        if any(entry_identity(row)[1] <= fit_end for row in rows):
            raise core.SimulationError("invalid_period")
        features = build_features(rows, history=history, history_lag_days=bundle["metadata"]["history_lag_days"])
        matrix = model.frame(features, bundle["metadata"]["categories"])
        probabilities = bundle["boosters"]["win"].predict(matrix, num_threads=4)
        records = [{"entry_key": list(entry_identity(row)[0]), "wager": "win", "probability": float(p)}
                   for row, p in zip(rows, probabilities)]
        directory = local.artifact("predictions")
        local.write(directory / "predictions.json", {"format_version": 2, "mode": "hypothetical_backtest", "records": records})
        local.write(directory / "manifest.json", {"format_version": 2, "mode": "hypothetical_backtest",
                    "model_manifest_sha256": local.sha(args.model / "model.json", root=local.old_io.ML_ROOT),
                    "predictions_sha256": local.sha(directory / "predictions.json"), "wager": "win",
                    "entries_source": entry_metadata["source"],
                    "entries_sha256": local.sha(args.entries, root=local.old_io.ML_ROOT),
                    "current_results_excluded": entry_metadata.get("current_results_excluded", False)})
        _freeze_files(directory, ("predictions.json", "manifest.json"))
        return {"status": "ok", "artifact_id": directory.name, "kind": "predictions", "wager": "win",
                "runner_count": len(records), "race_count": len({tuple(row["entry_key"][:6]) for row in records})}
    if args.command == "simulate":
        if not 1 <= args.max_races <= 5000 or not 1 <= args.max_rows <= 200000 or not 1 <= args.max_seconds <= 600:
            raise core.SimulationError("input_limit")
        policy, policy_document = local.load_policy(args.policy)
        predictions, manifest = local.load_predictions(args.predictions, expected_model_hash=policy_document["model_manifest_sha256"])
        if len(predictions) > args.max_races:
            raise core.SimulationError("input_limit")
        for prediction in predictions:
            race = core.key(prediction.race)
            day = date(int(race[0]), int(race[1][:2]), int(race[1][2:]))
            if not date.fromisoformat(policy.evaluation_start) <= day <= date.fromisoformat(policy.evaluation_end):
                raise core.SimulationError("invalid_period")
        schedules = formats.parse_schedules(local.read(args.schedules))
        snapshots = formats.parse_snapshots(local.read(args.odds)) if args.odds else []
        outcomes = formats.parse_outcomes(local.read(args.payouts)) if args.payouts else []
        failures, db_status = {}, []
        if args.db_config:
            if not args.execute_local:
                raise core.SimulationError("explicit_execution_required")
            if args.odds:
                raise core.SimulationError("invalid_input")
            config_document = local.validate_config(local.read(args.db_config))
            if args.payouts:
                config_document = dict(config_document, payout_source=None)
            config = connection_config(min(args.query_timeout_ms, args.max_seconds*1000))
            secret = password()
            try:
                with read_only_session(config, secret) as db:
                    snapshots, db_outcomes, failures, db_status = local.fetch_inputs(db, config_document,
                        [prediction.race for prediction in predictions], max_rows=args.max_rows, max_seconds=args.max_seconds)
            finally:
                secret = None
            if not args.payouts:
                outcomes = db_outcomes
        simulation = core.run_simulation(predictions, schedules, snapshots, outcomes, policy=policy, odds_failures=failures)
        directory = local.artifact("simulations")
        local.write_report(directory, simulation, policy_payload=policy_document,
                           provenance={"prediction_manifest_sha256": local.sha(args.predictions / "manifest.json",
                             root=local.old_io.ML_ROOT if local.old_io.ML_ROOT in args.predictions.parents else local.ROOT),
                             "policy_sha256": local.sha(args.policy), "db_sources": db_status,
                             "model_features": "v01_retrospective_snapshot_asof_features_not_independently_verified",
                             "test_reuse": "descriptive_backtest_no_new_untouched_test_claim"})
        return {"status": "ok", "artifact_id": directory.name, "kind": "simulations", "wager": "win",
                "summary": simulation["summary"], "db_sources": db_status, "notice": simulation["notice"]}
    raise core.SimulationError("invalid_input")


def main(argv=None):
    try:
        result = run(parser().parse_args(argv))
        # ASCII JSON survives both Windows PowerShell legacy code pages and
        # PowerShell 7 UTF-8 native pipelines. Local Markdown remains UTF-8.
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (core.SimulationError, SafeAuditError, MLFailure) as exc:
        print(json.dumps({"status": "stopped", "reason": exc.code}))
        return 2
    except KeyboardInterrupt:
        print('{"status":"stopped","reason":"interrupted"}')
        return 130
    except Exception:
        print('{"status":"stopped","reason":"internal_error"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
