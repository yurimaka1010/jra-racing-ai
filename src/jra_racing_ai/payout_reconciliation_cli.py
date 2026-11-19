"""User-operated reconciliation command. This module never runs on import."""
from __future__ import annotations

import getpass
import importlib.util
import os
from pathlib import Path
import warnings

from .core import SafeAuditError
from .db_connection import ConnectionConfig, read_only_session
from .generate_report import REAL_REPORT_DIR, validate_output_dir
from .payout_reconciliation import (
    ALL_HYPOTHESES, AmountHypothesis, ReconciliationError, ReconciliationLimits, reconcile_payouts,
)
from .reconciliation_local import (
    ExecutionBudget, check_local_spec_pdf, fetch_local_hr, read_reference_file, write_reconciliation_summary,
)


def add_parser(sub):
    cmd = sub.add_parser("reconcile-payouts", help="利用者本人による少数HRとローカル公式参照の仮説照合")
    mode = cmd.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-input-only", action="store_true", help="参照ファイルだけを検証。DB接続なし")
    mode.add_argument("--execute-local", action="store_true", help="本人によるローカルDB参照を明示")
    cmd.add_argument("--confirm-local-unsynced", action="store_true", help="保存先が外部同期・バックアップ対象外と本人が確認")
    cmd.add_argument("--reference", type=Path, required=True)
    cmd.add_argument("--spec-pdf", type=Path, help="ローカル公式PDF。存在確認のみ、内容は本人がオフラインで確認")
    cmd.add_argument("--h1-config", type=Path, help="H1/HR列対応とH1版レビューのローカルJSON。指定時はH1照合必須")
    cmd.add_argument("--max-h1-rows", type=int, default=40)
    cmd.add_argument("--output", type=Path, default=REAL_REPORT_DIR)
    cmd.add_argument("--year", type=int, default=2025)
    cmd.add_argument("--max-races", type=int, default=5)
    cmd.add_argument("--max-hr-rows", type=int, default=40)
    cmd.add_argument("--statement-timeout-ms", type=int, default=5000)
    cmd.add_argument("--max-run-seconds", type=int, default=30)
    cmd.add_argument("--hypotheses", nargs="+", choices=[h.value for h in ALL_HYPOTHESES],
                     default=[h.value for h in ALL_HYPOTHESES])


def execute(args):
    if not args.confirm_local_unsynced:
        raise ReconciliationError("unsynced_confirmation_required")
    validate_output_dir(args.output, real=True)
    limits = ReconciliationLimits(args.year, args.max_races, args.max_hr_rows,
                                  args.statement_timeout_ms, args.max_run_seconds)
    hypotheses = tuple(AmountHypothesis(h) for h in args.hypotheses)
    if len(set(hypotheses)) != len(hypotheses):
        raise ReconciliationError("invalid_hypothesis")
    document = read_reference_file(args.reference)
    limits.check_document(document)
    h1_config = None
    if args.h1_config is not None:
        from .h1_local import read_h1_config_file
        from .h1_reconciliation import check_h1_budget
        check_h1_budget(limits, args.max_h1_rows)
        h1_config = read_h1_config_file(args.h1_config)
        h1_config.check_document(document)
    pdf_supplied = check_local_spec_pdf(args.spec_pdf) if args.spec_pdf else False
    if args.check_input_only:
        result = {"status": "input_valid_needs_confirmation", "requested_races": len(document.races),
                "reference_cases": len(document.cases), "reference_targets": sum(len(c.targets) for c in document.cases),
                "source_reviewed_locally": document.source_reviewed_locally,
                "abc_mapping_reviewed_locally": document.abc_mapping_reviewed_locally,
                "reviewed_flag_rules": sum(r.reviewed_locally for r in document.flag_rules),
                "reviewed_case_snapshots": sum(c.revision_review.value == "snapshot_reviewed_locally" for c in document.cases),
                "local_spec_pdf_supplied": pdf_supplied, "pdf_content_verified_by_program": False,
                "source_is_synthetic": document.source == "synthetic", "database_accessed": False,
                "production_settlement_enabled": False}
        if h1_config is not None:
            result["h1_validation"] = {"mode": "h1_required", "mapping_reviewed_locally": h1_config.column_meanings_reviewed_locally,
                "reviewed_snapshots": sum(r.revision.value == "snapshot_reviewed_locally" for r in h1_config.reviews),
                "missing_review_races": len(document.races) - len(h1_config.reviews), "database_accessed": False}
        return result
    if not args.execute_local:
        raise ReconciliationError("local_execution_required")
    if document.source != "jra_official_manual":
        raise ReconciliationError("synthetic_reference_forbidden")
    if not document.source_reviewed_locally:
        raise ReconciliationError("reference_source_unreviewed")
    config = ConnectionConfig.from_env(statement_timeout_ms=limits.statement_timeout_ms)
    if (config.host, config.port, config.database, config.user) != ("127.0.0.1", 5432, "pckeiba", "jra_audit_reader"):
        raise ReconciliationError("endpoint_mismatch")
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
    budget = ExecutionBudget(limits)
    try:
        with read_only_session(config, password) as db:
            if h1_config is None:
                records = fetch_local_hr(db, document, limits, budget)
                summary = reconcile_payouts(document, records, limits=limits, hypotheses=hypotheses)
            else:
                from .h1_local import fetch_local_h1
                from .h1_reconciliation import reconcile_h1_payouts
                records = fetch_local_hr(db, document, limits, budget, flag_columns=h1_config.hr_columns)
                h1_records = fetch_local_h1(db, document, h1_config, limits, budget, max_rows=args.max_h1_rows)
                summary = reconcile_h1_payouts(document, records, h1_records, h1_config, limits=limits,
                                               max_h1_rows=args.max_h1_rows, hypotheses=hypotheses)
            budget.check()
    finally:
        password = None
    # Close the read-only transaction before saving aggregate output.
    return write_reconciliation_summary(summary, args.output, local_spec_pdf_supplied=pdf_supplied)
