"""Mocked local files, DB, timers and CLI. Never opens a real HR/reference file."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import warnings

from jra_racing_ai import cli, payout_reconciliation_cli as command, reconciliation_local as local
from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.db_connection import ConnectionConfig, QueryRunner, validate_audit_sql
from jra_racing_ai.generate_report import PROJECT_ROOT
from jra_racing_ai.payout_reconciliation import (
    ReconciliationError, ReconciliationLimits, ReferenceDocument, reconcile_payouts,
)
from test_payout_reconciliation import document, json_reference, race, record, reference_case, rules


SECRET = "SYNTHETIC_PRIVATE_INPUT_DO_NOT_PRINT"
ROOT = Path(os.environ.get("TEMP") or "/tmp") / "jra_phase15_synthetic_no_write"
INPUT = ROOT / local.INPUT_SUBDIR / "fake-reference.json"


def row_value(value=None):
    value = value or record()
    row = dict(zip(RACE_KEY, value.race.parts)) | {
        "hr_category": value.category, "refund_bitmap": value.refund_bitmap,
        "refund_tansho": value.refund_tansho, "refund_fukusho": value.refund_fukusho,
        "oversized_fields": value.oversized, "unselected_secret": SECRET,
    }
    for wager in ("tansho", "fukusho"):
        for index, slot in enumerate(getattr(value, wager), 1):
            for part in "abc":
                row[f"haraimodoshi_{wager}_{index}{part}"] = getattr(slot, part)
    row.update({f"flag_{i}": v for i, v in enumerate(value.flags)})
    return row


def catalog(doc=None):
    doc = doc or document()
    names = set(RACE_KEY) | {doc.status_column, "henkan_umaban_joho", "henkan_flag_tansho", "henkan_flag_fukusho"}
    names.update(r.column for r in doc.flag_rules)
    names.update(f"haraimodoshi_{w}_{n}{f}" for w, count in (("tansho", 3), ("fukusho", 5))
                 for n in range(1, count+1) for f in "abc")
    return [{"name": name, "type_name": "varchar", "type_is_builtin": True} for name in sorted(names)]


class FakeDB:
    def __init__(self, *, doc=None, rows=None, table=None, columns=None, error=None):
        self.calls = []
        self.error = error
        self.responses = {
            "reconciliation.hr_table": [{"kind": "r", "method": "heap", "row_security": False, "force_row_security": False}]
                                      if table is None else table,
            "reconciliation.hr_columns": catalog(doc) if columns is None else columns,
            "reconciliation.hr_rows": [row_value()] if rows is None else rows,
        }

    def aggregate(self, query_id, sql, params=(), max_rows=1000):
        validate_audit_sql(sql)
        self.calls.append((query_id, sql, params, max_rows))
        if self.error:
            raise self.error
        return self.responses[query_id]


def fetch(db=None, doc=None, limits=None, clock=lambda: 0):
    limits = limits or ReconciliationLimits()
    return local.fetch_local_hr(db or FakeDB(), doc or document(), limits, local.ExecutionBudget(limits, clock=clock))


class LocalQuerySafetyTests(unittest.TestCase):
    def test_only_three_bounded_queries_are_sent_after_session_role_check(self):
        db = FakeDB()
        rows = fetch(db)
        self.assertEqual(len(rows), 1)
        self.assertEqual([c[0] for c in db.calls], ["reconciliation.hr_table", "reconciliation.hr_columns", "reconciliation.hr_rows"])
        self.assertEqual([c[3] for c in db.calls], [1, 1600, 40])

    def test_request_keys_and_limit_are_bound_parameters_not_sql_text(self):
        db = FakeDB()
        fetch(db)
        _, sql, params, _ = db.calls[-1]
        self.assertEqual(params, race().parts + (41,))
        self.assertNotIn("2025", sql)
        self.assertIn('FROM ONLY "public"."jvd_hr"', sql)
        self.assertIn("LIMIT %s", sql)

    def test_no_payload_cast_trim_or_odds_query_is_used(self):
        db = FakeDB()
        fetch(db)
        sql = db.calls[-1][1]
        for forbidden in ("::text", "btrim", "trim(", "jvd_o1", "SELECT *"):
            self.assertNotIn(forbidden, sql)
        self.assertIn("pg_catalog.octet_length", sql)
        self.assertIn("oversized_fields", sql)
        self.assertIn("ELSE NULL", sql)

    def test_full_slot_counts_and_original_nine_characters_are_kept(self):
        rows = fetch()
        self.assertEqual((len(rows[0].tansho), len(rows[0].fukusho)), (3, 5))
        self.assertEqual(rows[0].tansho[0].b, "000000350")
        self.assertEqual(rows[0].tansho[1].b, " " * 9)
        self.assertNotIn(SECRET, repr(rows[0]))

    def test_non_heap_foreign_view_and_rls_relations_are_rejected(self):
        baseline = {"kind": "r", "method": "heap", "row_security": False, "force_row_security": False}
        for change in ({"kind": "f"}, {"kind": "v"}, {"kind": "p"}, {"method": "custom"},
                       {"row_security": True}, {"force_row_security": True}):
            db = FakeDB(table=[baseline | change])
            with self.assertRaises(ReconciliationError) as caught:
                fetch(db)
            self.assertEqual(caught.exception.reason, "hr_table_unavailable")
            self.assertEqual(len(db.calls), 1)

    def test_missing_hr_table_is_not_queried_for_records(self):
        db = FakeDB(table=[])
        with self.assertRaises(ReconciliationError):
            fetch(db)
        self.assertEqual(len(db.calls), 1)

    def test_missing_or_non_builtin_or_numeric_columns_stop_before_payload(self):
        for change in ({"type_name": "numeric"}, {"type_is_builtin": False}, {"type_name": "bytea"}):
            columns = catalog()
            columns[0] = columns[0] | change
            db = FakeDB(columns=columns)
            with self.assertRaises(ReconciliationError) as caught:
                fetch(db)
            self.assertEqual(caught.exception.reason, "hr_column_mapping_unconfirmed")
            self.assertEqual(len(db.calls), 2)
        with self.assertRaises(ReconciliationError):
            fetch(FakeDB(columns=[]))

    def test_text_varchar_and_bpchar_are_explicitly_supported(self):
        for name in ("text", "varchar", "bpchar"):
            columns = [c | {"type_name": name} for c in catalog()]
            self.assertEqual(len(fetch(FakeDB(columns=columns))), 1)

    def test_shared_code_column_is_selected_once_and_reused_without_loss(self):
        values = list(rules())
        values[2] = replace(values[2], column=values[0].column, active_value="2")
        doc = document(flag_rules=tuple(values))
        db = FakeDB(doc=doc)
        rows = fetch(db, doc)
        self.assertEqual(rows[0].flags[0], rows[0].flags[2])
        self.assertNotIn('AS "flag_2"', db.calls[-1][1])

    def test_overflow_aborts_before_any_partial_summary(self):
        db = FakeDB(rows=[row_value()] * 2)
        with self.assertRaises(ReconciliationError) as caught:
            fetch(db, limits=ReconciliationLimits(max_hr_rows=1))
        self.assertEqual(caught.exception.reason, "row_limit")

    def test_unrequested_key_in_result_is_rejected(self):
        with self.assertRaises(ReconciliationError) as caught:
            fetch(FakeDB(rows=[row_value(record(race=race(number="10")))]))
        self.assertEqual(caught.exception.reason, "unexpected_hr_race")

    def test_deadline_prevents_starting_another_query(self):
        clock = iter((0, 26))
        db = FakeDB()
        with self.assertRaises(ReconciliationError) as caught:
            fetch(db, clock=lambda: next(clock))
        self.assertEqual(caught.exception.reason, "time_limit")
        self.assertEqual(db.calls, [])

    def test_elapsed_deadline_after_query_stops_without_returning_rows(self):
        clock = iter((0, 0, 31))
        with self.assertRaises(ReconciliationError) as caught:
            fetch(clock=lambda: next(clock))
        self.assertEqual(caught.exception.reason, "time_limit")

    def test_query_timeout_and_permission_failure_stay_redacted(self):
        for code in ("query_timeout", "permission_denied"):
            with self.assertRaises(SafeAuditError) as caught:
                fetch(FakeDB(error=SafeAuditError(code)))
            self.assertEqual(caught.exception.code, code)

    def test_existing_runner_rejects_writes_before_cursor_execute(self):
        connection = MagicMock()
        for sql in ("DELETE FROM public.jvd_hr", "UPDATE public.jvd_hr SET x=1", "DROP TABLE public.jvd_hr",
                    "WITH x AS (INSERT INTO public.jvd_hr VALUES (1) RETURNING *) SELECT * FROM x"):
            with self.assertRaises(SafeAuditError):
                QueryRunner(connection).aggregate("reconciliation.test", sql)
        connection.cursor.assert_not_called()


class ReferenceFileSafetyTests(unittest.TestCase):
    def test_allowed_local_input_path_without_real_file_access(self):
        with patch.object(local, "validate_output_dir", return_value=ROOT), \
                patch.object(local, "_local_disk", return_value=True), patch.object(local, "_has_link", return_value=False), \
                patch.object(local, "_cloud_or_reparse", return_value=False), \
                patch.object(Path, "is_file", return_value=True), patch.object(Path, "stat", return_value=SimpleNamespace(st_nlink=1)):
            self.assertEqual(local.validate_local_input(INPUT, suffix=".json"), INPUT.resolve())

    def test_relative_unc_and_wrong_parent_are_rejected_before_leaf_read(self):
        for path in (Path("fake-reference.json"), ROOT / "fake-reference.json", PROJECT_ROOT / "private.json",
                     Path(r"\\synthetic-server\share\private.json")):
            with patch.object(local, "validate_output_dir", return_value=ROOT), patch.object(Path, "is_file") as probe:
                with self.assertRaises(ReconciliationError):
                    local.validate_local_input(path, suffix=".json")
                probe.assert_not_called()

    def test_link_junction_and_cloud_placeholder_inputs_are_rejected(self):
        for link, cloud in ((True, False), (False, True)):
            with patch.object(local, "validate_output_dir", return_value=ROOT), \
                    patch.object(local, "_local_disk", return_value=True), patch.object(local, "_has_link", return_value=link), \
                    patch.object(local, "_cloud_or_reparse", return_value=cloud):
                with self.assertRaises(ReconciliationError):
                    local.validate_local_input(INPUT, suffix=".json")

    def test_hard_link_input_is_rejected(self):
        with patch.object(local, "validate_output_dir", return_value=ROOT), \
                patch.object(local, "_local_disk", return_value=True), patch.object(local, "_has_link", return_value=False), \
                patch.object(local, "_cloud_or_reparse", return_value=False), patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_nlink=2)):
            with self.assertRaises(ReconciliationError):
                local.validate_local_input(INPUT, suffix=".json")

    def test_onedrive_name_and_configured_sync_root_are_refused(self):
        for root, env in ((ROOT / "OneDrive" / "reports", {}),
                          (ROOT / "sync" / "reports", {"OneDrive": str(ROOT / "sync")})):
            with self.assertRaises(SafeAuditError):
                local.validate_local_input(root / local.INPUT_SUBDIR / "fake.json", suffix=".json",
                                           approved_root=root, project_root=PROJECT_ROOT, env=env)

    def test_windows_reparse_offline_and_recall_attributes_are_detected(self):
        for attribute in (0x400, 0x1000, 0x40000, 0x400000):
            with patch.object(Path, "lstat", return_value=SimpleNamespace(st_file_attributes=attribute)):
                self.assertTrue(local._cloud_or_reparse(INPUT))

    def read_bytes(self, data, size=None):
        with patch.object(local, "validate_local_input", return_value=INPUT), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=len(data) if size is None else size)), \
                patch.object(Path, "open", return_value=io.BytesIO(data)):
            return local.read_reference_file(INPUT)

    def test_valid_utf8_and_bom_json_are_read_from_bounded_stream(self):
        data = json.dumps(json_reference()).encode("utf-8")
        self.assertEqual(self.read_bytes(data), document())
        self.assertEqual(self.read_bytes(b"\xef\xbb\xbf" + data), document())

    def test_duplicate_json_keys_are_not_overwritten(self):
        with self.assertRaises(ReconciliationError) as caught:
            self.read_bytes(b'{"source":"synthetic","source":"private"}')
        self.assertEqual(caught.exception.reason, "duplicate_reference_key")

    def test_bad_json_unicode_and_nonfinite_numbers_never_echo_input(self):
        for data in (b'{"' + SECRET.encode() + b'":', b'\xff', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ReconciliationError) as caught:
                self.read_bytes(data)
            self.assertNotIn(SECRET, str(caught.exception) + repr(caught.exception) + json.dumps(caught.exception.diagnostic()))

    def test_oversized_reference_is_rejected_before_open(self):
        with patch.object(local, "validate_local_input", return_value=INPUT), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=local.MAX_REFERENCE_BYTES+1)), \
                patch.object(Path, "open") as opened:
            with self.assertRaises(ReconciliationError) as caught:
                local.read_reference_file(INPUT)
            self.assertEqual(caught.exception.reason, "input_size_limit")
            opened.assert_not_called()

    def test_growing_file_cannot_bypass_read_size_cap(self):
        with self.assertRaises(ReconciliationError) as caught:
            self.read_bytes(b" " * (local.MAX_REFERENCE_BYTES + 1), size=1)
        self.assertEqual(caught.exception.reason, "input_size_limit")

    def test_pdf_check_reads_header_only_without_claiming_semantics(self):
        handle = MagicMock()
        handle.__enter__.return_value = handle
        handle.read.return_value = b"%PDF-"
        with patch.object(local, "validate_local_input", return_value=INPUT.with_suffix(".pdf")), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=80)), \
                patch.object(Path, "open", return_value=handle):
            self.assertTrue(local.check_local_spec_pdf(INPUT.with_suffix(".pdf")))
        handle.read.assert_called_once_with(5)

    def test_non_pdf_header_is_rejected_with_fixed_reason(self):
        with patch.object(local, "validate_local_input", return_value=INPUT.with_suffix(".pdf")), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=80)), \
                patch.object(Path, "open", return_value=io.BytesIO(SECRET.encode())):
            with self.assertRaises(ReconciliationError) as caught:
                local.check_local_spec_pdf(INPUT.with_suffix(".pdf"))
            self.assertEqual(caught.exception.reason, "invalid_pdf")


class ReconciliationOutputSafetyTests(unittest.TestCase):
    def test_only_typed_aggregates_can_be_saved(self):
        with patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(ReconciliationError):
                local.write_reconciliation_summary({"private_row": SECRET}, ROOT)
            mkdir.assert_not_called()

    def test_project_and_onedrive_output_are_refused_before_mkdir(self):
        summary = reconcile_payouts(document(), (record(),))
        with patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(SafeAuditError):
                local.write_reconciliation_summary(summary, PROJECT_ROOT / "reports")
            mkdir.assert_not_called()

    def test_json_and_markdown_contain_no_identity_amount_or_input_path(self):
        summary = reconcile_payouts(document(), (record(),))
        captures = []

        class Capture:
            def __init__(self, path, mode):
                self.path, self.mode, self.stream = path, mode, io.StringIO()
            def __enter__(self):
                return self.stream
            def __exit__(self, *args):
                captures.append((self.path.name, self.mode, self.stream.getvalue()))
                self.stream.close()

        with patch.object(local, "validate_output_dir", return_value=ROOT), \
                patch.object(local, "_cloud_or_reparse", return_value=False), \
                patch.object(local, "_has_link", return_value=False), patch.object(Path, "mkdir"), \
                patch.object(Path, "open", autospec=True, side_effect=lambda path, mode, **kw: Capture(path, mode)):
            payload = local.write_reconciliation_summary(summary, ROOT, local_spec_pdf_supplied=True)
        self.assertEqual({name for name, _, _ in captures}, {"reconciliation.json", "reconciliation.md"})
        for _, mode, text in captures:
            self.assertEqual(mode, "x")
            for private in (SECRET, "000000350", "0106", "horse_number", str(INPUT), "payout_yen_per_100"):
                self.assertNotIn(private, text)
        self.assertFalse(payload["production_settlement_enabled"])
        self.assertFalse(payload["pdf_content_verified_by_program"])

    def test_cloud_output_root_is_rejected_before_writing(self):
        with patch.object(local, "validate_output_dir", return_value=ROOT), \
                patch.object(local, "_cloud_or_reparse", return_value=True), patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(ReconciliationError):
                local.write_reconciliation_summary(reconcile_payouts(document(), (record(),)), ROOT)
            mkdir.assert_not_called()


class ReconciliationCliTests(unittest.TestCase):
    def args(self, *extra, execute=False):
        return cli.parser().parse_args(["reconcile-payouts", "--execute-local" if execute else "--check-input-only",
                                         "--confirm-local-unsynced", "--reference", str(INPUT), *extra])

    def test_no_mode_or_both_modes_do_not_implicitly_connect(self):
        for extra in ([], ["--check-input-only", "--execute-local"]):
            with self.assertRaises(SafeAuditError):
                cli.parser().parse_args(["reconcile-payouts", "--reference", str(INPUT), *extra])

    def test_direct_cli_run_dispatches_to_the_dedicated_workflow(self):
        expected = {"status": "input_valid_needs_confirmation", "database_accessed": False}
        with patch.object(command, "execute", return_value=expected) as execute, patch.object(cli, "read_only_session") as legacy_session:
            self.assertEqual(cli.run(self.args()), expected)
            execute.assert_called_once()
            legacy_session.assert_not_called()

    def test_unsynced_confirmation_is_required_before_reading_files(self):
        args = self.args()
        args.confirm_local_unsynced = False
        with patch.object(command, "read_reference_file") as reader:
            with self.assertRaises(ReconciliationError) as caught:
                command.execute(args)
            self.assertEqual(caught.exception.reason, "unsynced_confirmation_required")
            reader.assert_not_called()

    def test_input_only_does_not_probe_driver_prompt_connect_or_write(self):
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=document()), \
                patch.object(command.importlib.util, "find_spec") as driver, patch.object(command.getpass, "getpass") as password, \
                patch.object(command, "read_only_session") as session, patch.object(command, "write_reconciliation_summary") as writer:
            result = command.execute(self.args())
            self.assertFalse(result["database_accessed"])
            self.assertTrue(result["source_is_synthetic"])
            self.assertEqual(result["reviewed_flag_rules"], 6)
            self.assertEqual(result["reviewed_case_snapshots"], 1)
            self.assertTrue(result["abc_mapping_reviewed_locally"])
            driver.assert_not_called()
            password.assert_not_called()
            session.assert_not_called()
            writer.assert_not_called()

    def test_synthetic_template_cannot_trigger_real_db_access(self):
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=document()), \
                patch.object(command, "read_only_session") as session:
            with self.assertRaises(ReconciliationError) as caught:
                command.execute(self.args(execute=True))
            self.assertEqual(caught.exception.reason, "synthetic_reference_forbidden")
            session.assert_not_called()

    def test_exact_local_host_port_database_reader_are_required(self):
        for config in (ConnectionConfig(host="::1"), ConnectionConfig(port=5433),
                       ConnectionConfig(database="other"), ConnectionConfig(user="other_reader")):
            with patch.object(command, "validate_output_dir"), \
                    patch.object(command, "read_reference_file", return_value=replace(document(), source="jra_official_manual")), \
                    patch.object(command.ConnectionConfig, "from_env", return_value=config), \
                    patch.object(command, "read_only_session") as session:
                with self.assertRaises(ReconciliationError) as caught:
                    command.execute(self.args(execute=True))
                self.assertEqual(caught.exception.reason, "endpoint_mismatch")
                session.assert_not_called()

    def test_invalid_limits_and_repeated_hypotheses_stop_before_file_read(self):
        for extra in (("--max-races", "21"), ("--max-hr-rows", "201"), ("--statement-timeout-ms", "10001"),
                      ("--max-run-seconds", "10"), ("--hypotheses", "raw_integer_as_yen_per_100", "raw_integer_as_yen_per_100")):
            with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file") as reader:
                with self.assertRaises(ReconciliationError):
                    command.execute(self.args(*extra))
                reader.assert_not_called()

    def test_missing_driver_does_not_prompt_or_connect(self):
        with patch.object(command, "validate_output_dir"), \
                patch.object(command, "read_reference_file", return_value=replace(document(), source="jra_official_manual")), \
                patch.object(command.ConnectionConfig, "from_env", return_value=ConnectionConfig()), \
                patch.object(command.importlib.util, "find_spec", return_value=None), \
                patch.object(command.getpass, "getpass") as password, patch.object(command, "read_only_session") as session:
            with self.assertRaises(SafeAuditError) as caught:
                command.execute(self.args(execute=True))
            self.assertEqual(caught.exception.code, "missing_driver")
            password.assert_not_called()
            session.assert_not_called()

    def test_full_user_execution_uses_readonly_session_and_returns_aggregate_only(self):
        with patch.object(command, "validate_output_dir"), \
                patch.object(command, "read_reference_file", return_value=replace(document(), source="jra_official_manual")), \
                patch.object(command.ConnectionConfig, "from_env", return_value=ConnectionConfig(statement_timeout_ms=5000)), \
                patch.object(command.importlib.util, "find_spec", return_value=object()), \
                patch.dict(os.environ, {}, clear=True), patch.object(command.getpass, "getpass", return_value=SECRET), \
                patch.object(command, "read_only_session") as session, patch.object(command, "fetch_local_hr", return_value=(record(),)), \
                patch.object(command, "write_reconciliation_summary", side_effect=lambda summary, *a, **k: summary.to_safe_dict()) as writer:
            result = command.execute(self.args(execute=True))
            self.assertFalse(result["production_settlement_enabled"])
            self.assertNotIn(SECRET, json.dumps(result))
            session.assert_called_once()
            self.assertEqual(session.call_args.args[0].statement_timeout_ms, 5000)
            self.assertEqual(session.call_args.args[1], SECRET)
            writer.assert_called_once()
            session.return_value.__exit__.assert_called_once()

    def test_hidden_password_fallback_is_rejected_without_echo(self):
        def unsafe(*a, **k):
            warnings.warn(SECRET, command.getpass.GetPassWarning)
        with patch.object(command, "validate_output_dir"), \
                patch.object(command, "read_reference_file", return_value=replace(document(), source="jra_official_manual")), \
                patch.object(command.ConnectionConfig, "from_env", return_value=ConnectionConfig()), \
                patch.object(command.importlib.util, "find_spec", return_value=object()), \
                patch.dict(os.environ, {}, clear=True), patch.object(command.getpass, "getpass", side_effect=unsafe), \
                patch.object(command, "read_only_session") as session, redirect_stderr(io.StringIO()) as stderr:
            code = cli.main(["reconcile-payouts", "--execute-local", "--confirm-local-unsynced", "--reference", str(INPUT)])
            self.assertEqual(code, 2)
            self.assertNotIn(SECRET, stderr.getvalue())
            session.assert_not_called()

    def test_cli_stdout_contains_only_aggregate_payload(self):
        with patch.object(command, "execute", return_value=reconcile_payouts(document(), (record(),)).to_safe_dict()), \
                redirect_stdout(io.StringIO()) as stdout:
            code = cli.main(["reconcile-payouts", "--check-input-only", "--reference", str(INPUT)])
            self.assertEqual(code, 0)
            self.assertNotIn(str(INPUT), stdout.getvalue())
            self.assertFalse(json.loads(stdout.getvalue())["production_settlement_enabled"])

    def test_fixed_reason_diagnostics_and_unknown_args_do_not_echo_input(self):
        with patch.object(command, "execute", side_effect=ReconciliationError("invalid_json")), \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main(["reconcile-payouts", "--check-input-only", "--reference", SECRET]), 2)
            self.assertNotIn(SECRET, stderr.getvalue())
            self.assertEqual(json.loads(stderr.getvalue())["reason"], "invalid_json")
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main(["reconcile-payouts", "--password", SECRET]), 2)
            self.assertNotIn(SECRET, stderr.getvalue())

    def test_unexpected_exception_does_not_print_traceback_or_sensitive_context(self):
        with patch.object(command, "execute", side_effect=RuntimeError(SECRET)), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main(["reconcile-payouts", "--check-input-only", "--reference", str(INPUT)]), 2)
            self.assertNotIn(SECRET, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
