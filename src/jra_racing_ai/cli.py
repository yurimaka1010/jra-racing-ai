"""Commands are inert until called; real audits require an explicit local flag."""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

from .core import AuditContext, DOMESTIC_VENUES, SafeAuditError, Scope
from .db_connection import ConnectionConfig, read_only_session
from .generate_report import PROJECT_ROOT, REAL_REPORT_DIR, build_report, validate_output_dir, write_report
from .synthetic import demonstration_report


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default errors echo arguments, possibly pasted secrets.
        raise SafeAuditError("invalid_configuration")


def parser() -> argparse.ArgumentParser:
    root = SafeParser(description="JRA Phase 1 ローカル読み取り専用監査")
    sub = root.add_subparsers(dest="command", required=True, parser_class=SafeParser)
    demo = sub.add_parser("demo", help="実DBを使わない架空データのレポート例")
    demo.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports")
    for name in ("schema", "audit"):
        cmd = sub.add_parser(name, help="利用者自身が実行するローカルDB監査")
        cmd.add_argument("--execute-local", action="store_true", help="実DBのローカル参照を明示")
        cmd.add_argument("--output", type=Path, default=REAL_REPORT_DIR)
        cmd.add_argument("--statement-timeout-ms", type=int, default=60000)
        if name == "audit":
            cmd.add_argument("--year", type=int, required=True)
            cmd.add_argument("--venues", default=",".join(DOMESTIC_VENUES))
            cmd.add_argument("--sections", nargs="+", choices=("races", "horses", "odds", "payouts"),
                             default=["races", "horses", "odds"])
            cmd.add_argument("--include-global-summary", action="store_true",
                             help="全期間集計を追加（全表走査の可能性）")
            cmd.add_argument("--race-status-column")
            cmd.add_argument("--horse-status-column")
            cmd.add_argument("--odds-status-column")
            cmd.add_argument("--payout-status-column", help="仕様で確認したjvd_hr区分列を明示")
            cmd.add_argument("--horse-identity-columns", default="",
                             help="仕様を確認した馬識別列。複数列はカンマ区切り")
    from .payout_reconciliation_cli import add_parser
    add_parser(sub)
    return root


def run(args) -> dict:
    if args.command == "reconcile-payouts":
        from .payout_reconciliation_cli import execute
        return execute(args)
    if args.command == "demo":
        return write_report(demonstration_report(), args.output)
    if not args.execute_local:
        raise SafeAuditError("invalid_configuration")
    # Output and endpoint validation happen before reading secrets or connecting.
    validate_output_dir(args.output, real=True)
    config = ConnectionConfig.from_env(statement_timeout_ms=args.statement_timeout_ms)
    scope = None
    if args.command == "audit":
        scope = Scope(args.year, tuple(args.venues.split(",")), args.include_global_summary)
    if importlib.util.find_spec("psycopg") is None:
        raise SafeAuditError("missing_driver")
    password = os.environ.get("PGPASSWORD")
    if not password:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                password = getpass.getpass("読み取り専用ユーザーのパスワード（保存しません）: ")
            except getpass.GetPassWarning:
                raise SafeAuditError("invalid_configuration") from None
    from .inspect_schema import inspect_schema
    with read_only_session(config, password) as db:
        schema = inspect_schema(db)
        sections = {}
        if scope is not None:
            mappings = {table: column for table, column in (
                ("jvd_ra", args.race_status_column), ("jvd_se", args.horse_status_column),
                ("jvd_o1", args.odds_status_column), ("jvd_hr", getattr(args, "payout_status_column", None))) if column}
            ctx = AuditContext(db, scope, {t["name"]: t for t in schema["tables"]}, mappings,
                               tuple(filter(None, args.horse_identity_columns.split(","))))
            if "races" in args.sections:
                from .audit_races import audit_races
                sections["races"] = audit_races(ctx)
            if "horses" in args.sections:
                from .audit_horses import audit_horses
                sections["horses"] = audit_horses(ctx)
            if "odds" in args.sections:
                from .audit_odds import audit_odds
                sections["odds"] = audit_odds(ctx)
            if "payouts" in args.sections:
                from .audit_payouts import audit_payouts
                sections["payouts"] = audit_payouts(ctx)
        report = build_report(schema, sections, source="local_database", scope=scope)
    password = None
    return write_report(report, args.output)


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.command == "reconcile-payouts":
            # The new command prints typed aggregates only, never input paths.
            print(json.dumps(run(args), ensure_ascii=False))
            return 0
        outputs = run(args)
        # Only destination paths and a fixed completion message are displayed.
        print("監査レポートをローカル保存しました。")
        print(outputs["audit.md"])
        return 0
    except SafeAuditError as exc:
        from .payout_reconciliation import ReconciliationError
        if type(exc) is ReconciliationError:
            print(json.dumps(exc.diagnostic()), file=sys.stderr)
            return 2
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("処理を中止しました。", file=sys.stderr)
        return 130
    except Exception:
        print(str(SafeAuditError("internal_error")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
