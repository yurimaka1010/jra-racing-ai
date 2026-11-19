"""User-operated Version 0.3: causal features, one win model and existing settlement."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time

from .core import SafeAuditError
from .db_connection import read_only_session
from .ml import data, features, model, safety, v03
from .win_sim import core, formats, local, publication, settle_saved
from .win_sim_cli import connection_config, password

ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v03")
OLD_MODEL = "f1cce38718a4473ca1967b2252cb7997"
OLD_PREDICTIONS = "0bc0baa4431748308e94c416a515e3ed"
REFERENCE = "b465a4023292420ba075385876d8d71c"


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise safety.MLFailure("invalid_input")


def parser():
    result = Parser(description="Local Version 0.3 win-only feature improvement")
    sub = result.add_subparsers(dest="command", required=True, parser_class=Parser)
    first = sub.add_parser("first-run")
    first.add_argument("--train-start", default="2025-01-05")
    first.add_argument("--train-end", default="2025-01-18")
    first.add_argument("--valid-end", default="2025-01-19")
    first.add_argument("--calibration-start", default="2025-02-01")
    first.add_argument("--calibration-end", default="2025-02-02")
    first.add_argument("--test-start", default="2025-02-15")
    first.add_argument("--test-end", default="2025-02-16")
    first.add_argument("--reference-simulation", default=REFERENCE)
    first.add_argument("--old-model", default=OLD_MODEL)
    first.add_argument("--old-predictions", default=OLD_PREDICTIONS)
    first.add_argument("--rounds", type=int, default=100)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True, help="OneDrive外の今回のrun.json")
    evaluate.add_argument("--race-date", required=True)
    evaluate.add_argument("--schedules", type=Path)
    evaluate.add_argument("--odds", type=Path)
    evaluate.add_argument("--payout-source", type=Path)
    evaluate.add_argument("--odds-table", choices=formats.ODDS_TABLES, default="apd_sokuho_o1")
    for cmd in (first, evaluate):
        cmd.add_argument("--execute-local", action="store_true")
        cmd.add_argument("--history-start", default="2024-01-01")
        cmd.add_argument("--max-history-rows", type=int, default=250000)
        cmd.add_argument("--max-rows", type=int, default=20000)
        cmd.add_argument("--max-seconds", type=int, default=300)
        cmd.add_argument("--statement-timeout-ms", type=int, default=60000)
    return result


def stage(progress, name):
    progress["stage"] = name
    print("v03: " + name, file=sys.stderr)


def validate_args(args):
    if not args.execute_local:
        raise safety.MLFailure("explicit_execution_required")
    if (not 1 <= args.max_history_rows <= 400000 or not 1 <= args.max_rows <= 50000
            or not 5 <= args.max_seconds <= 600 or not 1000 <= args.statement_timeout_ms <= 60000):
        raise safety.MLFailure("input_limit")
    try:
        history_start = date.fromisoformat(args.history_start)
        if args.command == "first-run":
            days = {field: date.fromisoformat(getattr(args, field)) for field in
                    ("train_start", "train_end", "valid_end", "calibration_start", "calibration_end", "test_start", "test_end")}
            if (not history_start < days["train_start"] <= days["train_end"] < days["valid_end"]
                    or not days["valid_end"] + timedelta(days=7) < days["calibration_start"] <= days["calibration_end"]
                    or not days["calibration_end"] + timedelta(days=7) < days["test_start"] <= days["test_end"]
                    or not 5 <= args.rounds <= 1000):
                raise ValueError
            return history_start, days
        day = date.fromisoformat(args.race_date)
        if not history_start < day - timedelta(days=7):
            raise ValueError
        return history_start, day
    except (TypeError, ValueError):
        raise safety.MLFailure("invalid_split") from None


def freeze(directory):
    for name in ("runners.csv", "dataset.json", "mapping.reviewed.json", "model.json", "win.lgb.txt"):
        if (directory / name).is_file():
            (directory / name).chmod(0o444)


def export(runner, mapping, catalog, start, end, args, budget, *, prediction_only=False):
    budget.check()
    directory = safety.new_artifact("datasets")
    data.export_dataset(runner, mapping, directory, start=start, end=end,
        max_rows=args.max_rows if prediction_only else args.max_history_rows,
        max_seconds=max(1, int(budget.deadline - time.monotonic())), catalog=catalog,
        prediction_only=prediction_only)
    budget.check()
    freeze(directory)
    rows, metadata = data.read_dataset(directory / "runners.csv", allow_prediction_snapshot=prediction_only)
    return rows, metadata, directory


def save_predictions(predictions, model_hash, entries_path, *, purpose):
    directory = local.artifact("predictions")
    local.write(directory / "predictions.json", {"format_version": 2, "mode": "hypothetical_backtest",
        "records": [{"entry_key": list(item.race) + [horse], "wager": "win", "probability": float(p)}
                    for item in predictions for horse, p in item.probabilities.items()]})
    local.write(directory / "manifest.json", {"format_version": 2, "mode": "hypothetical_backtest",
        "pipeline_version": "0.3", "purpose": purpose, "model_manifest_sha256": model_hash,
        "predictions_sha256": local.sha(directory / "predictions.json"), "wager": "win",
        "entries_sha256": safety.digest(entries_path), "current_results_excluded_from_features": True})
    for name in ("predictions.json", "manifest.json"):
        (directory / name).chmod(0o444)
    return directory


def save_policy(policy, model_hash, calibration):
    document = local.policy_document(policy, model_hash=model_hash, calibration=calibration)
    document["pipeline_version"] = "0.3"
    directory = local.artifact("policies")
    local.write(directory / "policy.json", document)
    (directory / "policy.json").chmod(0o444)
    return directory, document


def old_history(bundle):
    matches = []
    for directory in (safety.ML_ROOT / "datasets").iterdir():
        if not directory.is_dir() or not (directory / "dataset.json").is_file():
            continue
        value = safety.read_json(directory / "dataset.json")
        if value.get("complete") is True and value.get("csv_sha256") == bundle["metadata"]["dataset_sha256"]:
            matches.append(directory / "runners.csv")
    if len(matches) != 1:
        raise safety.MLFailure("artifact_invalid")
    return data.read_dataset(matches[0])


def old_entries(rows, mapping):
    output = []
    for row in rows:
        entry = dict(row, finish_position="", finish_time_seconds="")
        for field in features.OPTIONAL_FIELDS:
            if not mapping["columns"].get(field) or (field in data.PRE_RACE and field not in mapping["pre_race_fields"]):
                entry[field] = ""
        output.append(entry)
    return output


def write_run_report(directory, result):
    def show(value):
        return "未算出" if value is None else str(round(value, 6) if isinstance(value, float) else value)
    lines = ["# Version 0.3 単勝モデルと参考バックテスト", "",
             "既知の参考日と未使用の確率評価期間を分離しています。期待利益率の低下だけでは改善と判定しません。", "",
             "モデルID: " + result["model_id"], "", "## 特徴量欠損", "",
             "| 特徴量 | 旧学習欠損率 | 新学習欠損率 | 参考日欠損率 |", "|---|---:|---:|---:|"]
    coverage = result["feature_coverage"]
    for name in features.FEATURES:
        old = coverage["v02_train_missing_count"][name] / coverage["v02_train_rows"]
        lines.append("| " + name + " | " + " | ".join(show(value) for value in
            (old, coverage["v03_train"]["missing_rate"][name], coverage["v03_reference"]["missing_rate"][name])) + " |")
    lines += ["", "## 温度校正と未使用期間", "", "選択温度: " + show(result["calibration"]["temperature"]), "",
              "| 区分 | 開始 | 終了 | レース数 |", "|---|---|---|---:|"]
    for name, part in result["split"].items():
        lines.append(f"| {name} | {part['start']} | {part['end']} | {part['races']} |")
    lines += ["", "| モデル | 馬単位Log Loss | Brier | レースLog Loss | 予測1位的中率% | 勝馬平均予測順位 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, part in result["heldout_test"].items():
        metric = part.get("model", {})
        lines.append("| " + name + " | " + " | ".join(show(metric.get(field)) for field in
            ("log_loss", "brier_score", "race_log_loss", "top1_accuracy_percent", "winner_mean_rank")) + " |")
    lines += ["", "## 既知の参考日：同じ価格・精算条件の共通レース", "",
              "共通レース数: " + str(result["reference_comparison"]["common_race_count"]), "",
              "| 戦略 | 購入 | 的中 | 的中率% | 投資円 | 払戻円 | 純利益円 | 回収率% |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("v02_ai", "v03_ai", "favorite"):
        value = result["reference_comparison"][name]
        lines.append("| " + name + " | " + " | ".join(show(value.get(field)) for field in
            ("bet_count", "hit_count", "hit_rate_percent", "investment_yen", "payout_yen", "net_profit_yen", "return_rate_percent")) + " |")
    lines += ["", "EV分布・低確率帯の校正・欠損件数等は同じフォルダーのrun.jsonを参照してください。",
              "発表時刻基準の暫定評価。実取得時刻・購入可能性と特徴量の全訂正履歴は未検証。", ""]
    path = safety.local_path(directory / "report.md", root=ROOT)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        path.chmod(0o444)
    except Exception:
        raise safety.MLFailure("io_failed") from None


def first_run(args, progress):
    history_start, days = validate_args(args)
    stage(progress, "preflight")
    reference = v03.reference_inputs(args.reference_simulation)
    ev = v03.ev_check(reference["ledger"])
    reference_days = {date(int(row["race_key"][0]), int(row["race_key"][1][:2]), int(row["race_key"][1][2:]))
                      for row in reference["ledger"]["rows"]}
    if len(reference_days) != 1:
        raise safety.MLFailure("invalid_split")
    reference_day = next(iter(reference_days))
    if reference_day <= days["test_end"]:
        raise safety.MLFailure("invalid_split")
    old_bundle = model.load_models(safety.ML_ROOT / "models" / args.old_model, required_targets=("win",))
    legacy_history, legacy_metadata = old_history(old_bundle)
    original_hash = local.sha(safety.ML_ROOT / "models" / args.old_model / "model.json", root=safety.ML_ROOT)
    old_predictions, _ = local.load_predictions(local.ROOT / "predictions" / args.old_predictions,
                                               expected_model_hash=original_hash)
    original_policy = core.WinPolicy(**reference["policy_document"]["policy"]).validate()
    if (original_policy.threshold != "0.05" or original_policy.minutes_before_start != 10):
        raise safety.MLFailure("invalid_input")
    secret = password()
    config = connection_config(args.statement_timeout_ms)
    try:
        stage(progress, "feature_mapping")
        with read_only_session(config, secret) as runner:
            catalog, _ = data.inspect_sources(runner, include_years=False)
        mapping = v03.confirm_mapping(v03.feature_mapping(catalog))
        mapping_directory = safety.new_artifact("schema")
        safety.write_json(mapping_directory / "mapping.reviewed.json", mapping)
        freeze(mapping_directory)
        budget = local.Budget(args.max_seconds)
        with read_only_session(config, secret) as runner:
            catalog, _ = data.inspect_sources(runner, include_years=False)
            stage(progress, "history_export")
            history, history_metadata, history_directory = export(runner, mapping, catalog, history_start,
                reference_day - timedelta(days=mapping["history_lag_days"]), args, budget)
            stage(progress, "training_export")
            training_args = argparse.Namespace(**vars(args))
            training_args.max_history_rows = args.max_rows
            rows, metadata, dataset_directory = export(runner, mapping, catalog,
                days["train_start"], days["test_end"], training_args, budget)
            stage(progress, "reference_export")
            entries, entry_metadata, entry_directory = export(runner, mapping, catalog,
                reference_day, reference_day, args, budget, prediction_only=True)
    finally:
        secret = None
    stage(progress, "feature_coverage")
    built = features.build_features(rows, history=history, history_lag_days=mapping["history_lag_days"])
    train_features = [row for row, raw in zip(built, rows) if features.entry_identity(raw)[1] <= days["train_end"]]
    training_coverage = v03.coverage(train_features)
    progress["feature_coverage"] = training_coverage
    v03.check_training_coverage(training_coverage)
    stage(progress, "fit_win_model")
    bundle = model.fit_models(rows, history=history, targets=("win",), history_lag_days=mapping["history_lag_days"],
        train_end=days["train_end"], valid_end=days["valid_end"], test_end=days["test_end"], rounds=args.rounds,
        calibration_start=days["calibration_start"], calibration_end=days["calibration_end"], test_start=days["test_start"])
    bundle["metadata"].update(pipeline_version="0.3", dataset_sha256=metadata["csv_sha256"],
        history_dataset_sha256=history_metadata["csv_sha256"], mapping_sha256=safety.digest(mapping_directory / "mapping.reviewed.json"),
        feature_definition="shared_ml.features.build_features", feature_coverage=training_coverage,
        availability="retrospective_declared_fields_no_current_results_history_date_plus_7_days")
    model_directory = safety.new_artifact("models")
    model.save_models(bundle, model_directory)
    freeze(model_directory)
    model_hash = safety.digest(model_directory / "model.json")
    stage(progress, "temperature_calibration")
    def partition(name):
        indices = bundle["split"][name]
        selected = [rows[i] for i in indices]
        predictions = v03.prediction_races(selected, [bundle["predictions"]["win"][i] for i in indices])
        return selected, predictions, v03.single_winners(selected)
    _, calibration_predictions, calibration_winners = partition("calibration")
    calibration = core.fit_temperature(calibration_predictions, calibration_winners,
                                      model_fit_end=args.valid_end, test_start=args.test_start)
    calibrated = replace(original_policy, temperature=calibration["temperature"], calibration_source="independent_validation",
                         calibration_end=calibration["calibration_end"], model_fit_end=args.valid_end)
    calibration_metrics = {"before": v03.probability_evaluation(calibration_predictions, calibration_winners, original_policy),
                           "after": v03.probability_evaluation(calibration_predictions, calibration_winners, calibrated)}
    stage(progress, "heldout_probability_evaluation")
    test_rows, test_predictions, test_winners = partition("test")
    legacy_test_predictions, _ = v03.predict_win(old_bundle, old_entries(test_rows, legacy_metadata["mapping"]), legacy_history)
    test_metrics = {"v02": v03.probability_evaluation(legacy_test_predictions, test_winners, original_policy),
                    "v03_uncalibrated": v03.probability_evaluation(test_predictions, test_winners, original_policy),
                    "v03_calibrated": v03.probability_evaluation(test_predictions, test_winners, calibrated)}
    test_prediction_directory = save_predictions(test_predictions, model_hash, dataset_directory / "runners.csv", purpose="heldout_probability_test")
    stage(progress, "reference_simulation")
    reference_predictions, reference_coverage = v03.predict_win(bundle, entries, history)
    new_reference_keys = {item.race for item in reference_predictions}
    if new_reference_keys != {core.key(row["race_key"]) for row in reference["ledger"]["rows"]}:
        raise safety.MLFailure("invalid_identity")
    reference_policy = replace(calibrated, evaluation_start=reference_day.isoformat(), evaluation_end=reference_day.isoformat()).validate()
    policy_directory, document = save_policy(reference_policy, model_hash, calibration)
    prediction_directory = save_predictions(reference_predictions, model_hash, entry_directory / "runners.csv", purpose="known_reference_date")
    simulation = core.run_simulation(reference_predictions, reference["schedules"], reference["snapshots"],
                                     reference["outcomes"], policy=reference_policy)
    simulation_directory = local.artifact("simulations")
    local.write_report(simulation_directory, simulation, policy_payload=document, provenance={
        "pipeline_version": "0.3", "source": "local_inputs", "evaluation_role": "known_reference_not_new_holdout",
        "reference_simulation": args.reference_simulation, "reference_ledger_sha256": reference["ledger_sha256"],
        "reference_manifest_sha256": reference["manifest_sha256"], "prediction_manifest_sha256": local.sha(prediction_directory / "manifest.json"),
        "model_manifest_sha256": model_hash, "history_dataset_sha256": history_metadata["csv_sha256"],
        "actual_observation_time_verified": False, "purchase_availability_verified": False})
    winners = v03.outcome_winners(reference["outcomes"], reference_predictions)
    result = {"status": "ok", "kind": "v03_first_run", "model_id": model_directory.name,
        "dataset_id": dataset_directory.name, "history_dataset_id": history_directory.name, "mapping_id": mapping_directory.name,
        "entry_dataset_id": entry_directory.name, "predictions_id": prediction_directory.name, "policy_id": policy_directory.name,
        "simulation_id": simulation_directory.name, "heldout_predictions_id": test_prediction_directory.name,
        "ev_formula_check": ev, "split": bundle["metadata"]["split"], "feature_coverage": {
            "v02_train_missing_count": old_bundle["metadata"]["missing_features"]["train"],
            "v02_train_rows": old_bundle["metadata"]["split"]["train"]["rows"],
            "v03_train": training_coverage, "v03_reference": reference_coverage},
        "calibration": calibration, "calibration_metrics": calibration_metrics, "heldout_test": test_metrics,
        "reference_probability_metrics": {"v02": v03.probability_evaluation(old_predictions, winners, original_policy),
            "v03": v03.probability_evaluation(reference_predictions, winners, reference_policy)},
        "reference_comparison": v03.comparison(reference["ledger"], simulation),
        "reference_summary": simulation["summary"], "v02_preserved": True,
        "evaluation_notice": "reference date is known; heldout probabilities use a separate later-than-calibration period; no profitability guarantee"}
    directory = safety.new_artifact("evaluations", root=ROOT)
    result["artifact_id"] = directory.name
    result["run_directory"] = str(directory)
    safety.write_json(directory / "run.json", result, root=ROOT)
    (directory / "run.json").chmod(0o444)
    write_run_report(directory, result)
    progress.pop("feature_coverage", None)
    return result


def evaluate(args, progress):
    history_start, day = validate_args(args)
    saved = safety.read_json(args.run, root=ROOT)
    if saved.get("kind") != "v03_first_run" or saved.get("status") != "ok":
        raise safety.MLFailure("invalid_input")
    mapping_path = safety.ML_ROOT / "schema" / saved["mapping_id"] / "mapping.reviewed.json"
    mapping = data.validate_mapping(safety.read_json(mapping_path))
    bundle = model.load_models(safety.ML_ROOT / "models" / saved["model_id"], required_targets=("win",))
    if safety.digest(mapping_path) != bundle["metadata"].get("mapping_sha256"):
        raise safety.MLFailure("artifact_invalid")
    calibration = saved["calibration"]
    if day <= date.fromisoformat(calibration["calibration_end"]) + timedelta(days=mapping["history_lag_days"]):
        raise safety.MLFailure("invalid_split")
    old_policy, policy_document = local.load_policy(local.ROOT / "policies" / saved["policy_id"] / "policy.json")
    if policy_document["model_manifest_sha256"] != safety.digest(safety.ML_ROOT / "models" / saved["model_id"] / "model.json"):
        raise safety.MLFailure("artifact_invalid")
    policy = replace(old_policy, evaluation_start=day.isoformat(), evaluation_end=day.isoformat()).validate()
    if bool(args.schedules) != bool(args.odds):
        raise safety.MLFailure("invalid_input")
    schedules = formats.parse_schedules(local.read(args.schedules)) if args.schedules else None
    snapshots = formats.parse_snapshots(local.read(args.odds)) if args.odds else None
    budget, config, secret = local.Budget(args.max_seconds), connection_config(args.statement_timeout_ms), password()
    try:
        with read_only_session(config, secret) as runner:
            catalog, _ = data.inspect_sources(runner, include_years=False)
            stage(progress, "history_export")
            history, _, _ = export(runner, mapping, catalog, history_start,
                day - timedelta(days=mapping["history_lag_days"]), args, budget)
            stage(progress, "entries_export")
            rows, _, entry_directory = export(runner, mapping, catalog, day, day, args, budget, prediction_only=True)
        predictions, feature_coverage = v03.predict_win(bundle, rows, history)
        failures = {}
        if schedules is None:
            with read_only_session(config, secret) as runner:
                ra, odds, _ = publication.fetch(runner, tuple(item.race for item in predictions),
                    table=args.odds_table, max_rows=20000, max_seconds=min(120, args.max_seconds))
            schedules, _ = publication.schedules_from_ra(ra, predictions)
            snapshots, failures, _, examples = publication.decode_odds(odds, source=args.odds_table)
            review_directory = local.artifact("simulations")
            local.write(review_directory / "raw_format_review.json", {"source_table": args.odds_table,
                "field_mapping": {"odds_tansho": "examples[].raw_win_block"}, "examples": examples})
            if any(item.reason is None and item.odds for item in snapshots):
                publication.confirm_raw_format(review_directory / "raw_format_review.json")
                snapshots = [replace(item, verified=True) for item in snapshots]
        model_hash = safety.digest(safety.ML_ROOT / "models" / saved["model_id"] / "model.json")
        predictions_directory = save_predictions(predictions, model_hash, entry_directory / "runners.csv", purpose="additional_period")
        _, document = save_policy(policy, model_hash, calibration)
        pending = core.run_simulation(predictions, schedules, snapshots, policy=policy, odds_failures=failures)
        pending_directory = local.artifact("simulations")
        local.write_report(pending_directory, pending, policy_payload=document, provenance={"pipeline_version": "0.3",
            "source": "local_inputs", "evaluation_role": "additional_period_not_automatically_new_holdout",
            "prediction_manifest_sha256": local.sha(predictions_directory / "manifest.json")})
        settlement_args = argparse.Namespace(simulation_id=pending_directory.name, execute_local=True,
            payout_source=args.payout_source, max_races=50, max_rows=200, max_seconds=30, query_timeout_ms=10000)
        result = settle_saved.run(settlement_args, connection_config=connection_config,
                                  password=lambda: secret, read_only_session=read_only_session)
        result.update(kind="v03_additional_evaluation", pipeline_version="0.3", feature_coverage=feature_coverage,
                      pending_simulation_id=pending_directory.name, predictions_id=predictions_directory.name)
        return result
    finally:
        secret = None


def main(argv=None):
    progress = {"stage": "preflight"}
    try:
        args = parser().parse_args(argv)
        result = first_run(args, progress) if args.command == "first-run" else evaluate(args, progress)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (safety.MLFailure, core.SimulationError, SafeAuditError) as exc:
        print(json.dumps(dict(status="stopped", reason=exc.code, **progress), ensure_ascii=True))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped", "reason": "interrupted", "stage": progress["stage"]}))
        return 130
    except Exception:
        print(json.dumps({"status": "stopped", "reason": "internal_error", "stage": progress["stage"]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
