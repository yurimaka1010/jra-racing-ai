"""Explicit local Version 0.1 entry point; stdout never contains runner records."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import warnings

from .core import SafeAuditError
from .db_connection import ConnectionConfig, read_only_session
from .ml import data, model, paper, safety
from .ml.features import entry_identity, rank_labels
from .ml.safety import MLFailure


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise MLFailure("invalid_input")


def parser():
    result = Parser(description="Local chronological LightGBM Version 0.1")
    commands = result.add_subparsers(dest="command", required=True, parser_class=Parser)
    demo = commands.add_parser("demo", help="Fictional data only, fit and predict in memory")
    demo.add_argument("--rounds", type=int, default=40)
    for name in ("inspect", "export", "export-entries"):
        command = commands.add_parser(name)
        command.add_argument("--execute-local", action="store_true")
        command.add_argument("--statement-timeout-ms", type=int, default=60000)
        if name in ("export", "export-entries"):
            command.add_argument("--mapping", type=Path, required=True)
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
            command.add_argument("--max-rows", type=int, default=100000)
            command.add_argument("--max-seconds", type=int, default=180)
    first = commands.add_parser("first-run", help="Local interactive minimal mapping, small export, fit")
    first.add_argument("--execute-local", action="store_true")
    first.add_argument("--start", default="2025-01-01")
    first.add_argument("--end", default="2025-01-31")
    first.add_argument("--max-rows", type=int, default=10000)
    first.add_argument("--max-seconds", type=int, default=180)
    first.add_argument("--statement-timeout-ms", type=int, default=60000)
    first.add_argument("--rounds", type=int, default=100)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--train-end")
    train.add_argument("--valid-end")
    train.add_argument("--test-end")
    train.add_argument("--rounds", type=int, default=180)
    train.add_argument("--odds", type=Path)
    train.add_argument("--min-ev", type=float, default=.10)
    train.add_argument("--stake-yen", type=int, default=100)
    train.add_argument("--max-bets-per-race", type=int, default=2)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--entries", type=Path, required=True)
    predict.add_argument("--history", type=Path)
    predict.add_argument("--odds", type=Path)
    predict.add_argument("--mode", choices=("backtest", "forward"), required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--outcomes", type=Path, required=True)
    return result


def _date(value):
    try:
        return None if value is None else date.fromisoformat(value)
    except (ValueError, TypeError):
        raise MLFailure("invalid_date") from None


def _password():
    if os.environ.get("PGPASSWORD"):
        return os.environ["PGPASSWORD"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("Local read-only DB password (hidden): ")
            if not value:
                raise MLFailure("needs_authentication")
            return value
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        raise MLFailure("needs_authentication") from None


def _brief(bundle, source):
    return {"status": "ok", "source": source, "models": 2,
            "split_counts": {name: {k: v for k, v in summary.items() if k in ("rows", "races")}
                             for name, summary in bundle["metadata"]["split"].items()},
            "test_metrics": {target: {key: value for key, value in metric["test"]["model"].items()
                                      if key in ("log_loss", "brier_score", "calibration_ece", "count")}
                             for target, metric in bundle["metadata"]["metrics"].items()},
            "production_hr_settlement_enabled": False}


def _minimal_mapping(draft):
    """A human on this PC confirms the few meanings needed to get training started."""
    columns = draft["columns"]
    if any(not columns[field] for field in data.REQUIRED):
        raise MLFailure("invalid_mapping")
    keep = set(data.REQUIRED) | ({"horse_id"} if columns.get("horse_id") else set())
    print("列名の候補だけを表示します。実レコードは表示しません。", file=sys.stderr)
    meanings = {"race_status": "RA: 7=確定成績の区分列", "horse_status": "SE: 7=確定成績の区分列",
                "horse_number": "SE: 2桁の馬番", "finish_position": "SE: 確定着順（1=1着）",
                "horse_id": "SE: 同一馬を過去走まで追跡する識別列"}
    for field in sorted(keep):
        print(meanings[field] + " -> " + columns[field], file=sys.stderr)
    print("上記の対応をPC上の列定義で確認済みの場合のみ CONFIRM と入力してください。", file=sys.stderr)
    try:
        if input() != "CONFIRM":
            raise MLFailure("mapping_not_reviewed")
    except EOFError:
        raise MLFailure("mapping_not_reviewed") from None
    return data.validate_mapping(dict(draft, reviewed=True,
                                     columns={field: column if field in keep else None for field, column in columns.items()}))


def _train_local(args, rows, metadata, dataset_path):
    policy = paper.PurchasePolicy(args.min_ev, args.stake_yen, args.max_bets_per_race)
    lag = metadata.get("mapping", {}).get("history_lag_days", metadata.get("history_lag_days", 7))
    odds = paper.parse_odds(safety.read_json(args.odds)) if args.odds else {}
    bundle = model.fit_models(rows, history_lag_days=lag, train_end=_date(args.train_end),
                              valid_end=_date(args.valid_end), test_end=_date(args.test_end),
                              rounds=args.rounds, policy=policy)
    bundle["metadata"]["dataset_sha256"] = safety.digest(dataset_path)
    bundle["metadata"]["dataset_source"] = metadata["source"]
    directory = safety.new_artifact("models")
    model.save_models(bundle, directory)
    paper_report = {}
    for part in ("validation", "test"):
        indices = bundle["split"][part]
        selected = [rows[i] for i in indices]
        predictions = {target: [values[i] for i in indices] for target, values in bundle["predictions"].items()}
        decisions = paper.paper_decisions(selected, predictions, policy=policy, odds=odds)
        label_map = {(*entry_identity(rows[i])[0], target): bundle["labels"][i][target]
                     for i in indices for target in ("win", "place")}
        paper_report[part] = paper.summarize(decisions, labels=label_map)
        if part == "test":
            ledger = safety.new_artifact("predictions")
            paper.save_ledger(ledger, decisions=decisions, mode="backtest",
                              model_hash=safety.digest(directory / "model.json"), policy=policy)
    safety.write_json(directory / "paper_evaluation.json", paper_report)
    result = _brief(bundle, metadata["source"])
    result.update(artifact_id=directory.name, kind="models", test_ledger_id=ledger.name,
                  paper_test=paper_report["test"])
    return result


def run(args):
    if args.command == "demo":
        from .ml.demo import synthetic_rows
        rows = synthetic_rows()
        bundle = model.fit_models(rows, rounds=args.rounds)
        # Predict rows with all current outcomes removed, including both models.
        entries = [dict(rows[i], finish_position="", finish_time_seconds="") for i in bundle["split"]["test"]]
        history = [rows[i] for part in ("train", "validation") for i in bundle["split"][part]]
        predicted = model.predict(bundle, entries, history=history)
        result = _brief(bundle, "synthetic")
        result["prediction_count"] = len(predicted["win"])
        result["library_versions"] = bundle["metadata"]["library_versions"]
        return result
    if args.command in ("inspect", "export", "export-entries", "first-run"):
        if not args.execute_local:
            raise MLFailure("explicit_execution_required")
        # All validation and destination checks precede authentication/connect.
        safety.local_path(safety.ML_ROOT)
        mapping = None
        if args.command in ("export", "export-entries", "first-run"):
            if args.command in ("export", "export-entries"):
                mapping = data.validate_mapping(safety.read_json(args.mapping))
                data.export_query(mapping, start=_date(args.start), end=_date(args.end), max_rows=args.max_rows,
                                  prediction_only=args.command == "export-entries")
            elif (_date(args.start) > _date(args.end) or not 1 <= args.max_rows <= 50000
                  or not 5 <= args.rounds <= 1000):
                raise MLFailure("invalid_input")
            if not 1 <= args.max_seconds <= 600:
                raise MLFailure("invalid_input")
        config = ConnectionConfig.from_env(statement_timeout_ms=args.statement_timeout_ms)
        if config.user != "jra_audit_reader" or config.host != "127.0.0.1" or config.port != 5432 or config.database != "pckeiba":
            raise MLFailure("invalid_input")
        password = _password()
        try:
            with read_only_session(config, password) as runner:
                if args.command == "first-run":
                    catalog, draft = data.inspect_sources(runner)
                    # End the read transaction before waiting for a human.
                    first_catalog, first_draft = catalog, draft
                elif args.command == "inspect":
                    catalog, draft = data.inspect_sources(runner)
                    directory = safety.new_artifact("schema")
                    safety.write_json(directory / "schema.json", catalog)
                    safety.write_json(directory / "mapping.draft.json", draft)
                    return {"status": "ok", "artifact_id": directory.name, "kind": "schema",
                            "tables": 2, "column_counts": {k: len(v) for k, v in catalog["tables"].items()},
                            "next": "review_mapping_locally"}
                else:
                    directory = safety.new_artifact("datasets")
                    metadata = data.export_dataset(runner, mapping, directory, start=_date(args.start), end=_date(args.end),
                                                   max_rows=args.max_rows, max_seconds=args.max_seconds,
                                                   prediction_only=args.command == "export-entries")
                    result = {"status": "ok", "artifact_id": directory.name, "kind": "datasets",
                              "rows": metadata["rows"], "races": metadata["races"]}
                    if args.command == "export-entries":
                        result.update(purpose=metadata["purpose"], current_results_excluded=True,
                                      source_status_counts=metadata["source_status_counts"])
                    return result
            if args.command == "first-run":
                schema_directory = safety.new_artifact("schema")
                safety.write_json(schema_directory / "schema.json", first_catalog)
                safety.write_json(schema_directory / "mapping.draft.json", first_draft)
                mapping = _minimal_mapping(first_draft)
                safety.write_json(schema_directory / "mapping.reviewed.json", mapping)
                directory = safety.new_artifact("datasets")
                with read_only_session(config, password) as runner:
                    metadata = data.export_dataset(runner, mapping, directory, start=_date(args.start), end=_date(args.end),
                                                   max_rows=args.max_rows, max_seconds=args.max_seconds)
                # Database transaction has closed before CPU training starts.
                rows, _ = data.read_dataset(directory / "runners.csv")
                train_args = argparse.Namespace(min_ev=.10, stake_yen=100, max_bets_per_race=2,
                                                odds=None, train_end=None, valid_end=None, test_end=None,
                                                rounds=args.rounds)
                result = _train_local(train_args, rows, metadata, directory / "runners.csv")
                result.update(dataset_id=directory.name, mapping_id=schema_directory.name)
                return result
        finally:
            password = None
    if args.command == "train":
        rows, metadata = data.read_dataset(args.dataset)
        return _train_local(args, rows, metadata, args.dataset)
    if args.command == "predict":
        bundle = model.load_models(args.model)
        rows, _ = data.read_dataset(args.entries, require_manifest=args.mode != "forward")
        history = data.read_dataset(args.history)[0] if args.history else []
        last_fit = _date(bundle["metadata"]["split"]["validation"]["end"])
        # Validation was used for early stopping: only later races are untouched.
        if any(entry_identity(row)[1] <= last_fit for row in rows):
            raise MLFailure("invalid_split")
        odds = paper.parse_odds(safety.read_json(args.odds)) if args.odds else {}
        policy = paper.PurchasePolicy.from_dict(bundle["metadata"]["purchase_policy"])
        # Forward records are validated before model inference or file creation.
        now = datetime.now(timezone.utc)
        if args.mode == "forward":
            from .ml.features import race_cutoff
            if any(now >= race_cutoff(row) or any(row.get(field) not in (None, "")
                                                  for field in ("finish_position", "finish_time_seconds")) for row in rows):
                raise MLFailure("forward_cutoff")
        predictions = model.predict(bundle, rows, history=history)
        now = datetime.now(timezone.utc)
        decisions = paper.paper_decisions(rows, predictions, policy=policy, odds=odds, mode=args.mode, now=now)
        directory = safety.new_artifact("predictions")
        paper.save_ledger(directory, decisions=decisions, mode=args.mode,
                          model_hash=safety.digest(args.model / "model.json"), policy=policy)
        return {"status": "ok", "artifact_id": directory.name, "kind": "predictions",
                "mode": args.mode, "summary": paper.summarize(decisions)}
    if args.command == "evaluate":
        outcomes, _ = data.read_dataset(args.outcomes)
        summary = paper.evaluate_ledger(args.ledger, outcomes)
        directory = safety.new_artifact("evaluations")
        safety.write_json(directory / "evaluation.json", summary)
        return {"status": "ok", "artifact_id": directory.name, "kind": "evaluations", "summary": summary}
    raise MLFailure("invalid_input")


def main(argv=None):
    try:
        result = run(parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (MLFailure, SafeAuditError) as exc:
        result = {"status": "stopped", "reason": exc.code}
        diagnostics = getattr(exc, "diagnostics", None)
        if diagnostics:
            result["diagnostics"] = diagnostics
        print(json.dumps(result, ensure_ascii=False))
        return 2
    except KeyboardInterrupt:
        print('{"status":"stopped","reason":"interrupted"}')
        return 130
    except Exception:
        print('{"status":"stopped","reason":"internal_error"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
