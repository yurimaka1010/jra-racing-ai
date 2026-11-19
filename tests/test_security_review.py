"""Independent safety tests using only mocks and invented aggregate values."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import MagicMock, patch

from jra_racing_ai import cli
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.db_connection import (
    ConnectionConfig, QueryRunner, read_only_session, validate_audit_sql,
)
from jra_racing_ai import generate_report as reports


SYNTHETIC_SECRET = "SYNTHETIC_PASSWORD_SENTINEL_DO_NOT_PRINT"


def fake_driver(checks=None):
    """No PostgreSQL driver is imported or invoked by this test helper."""
    connection = MagicMock(name="invented_connection")
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchmany.return_value = [{
        "read_only": "on", "isolation": "repeatable read",
        "privileged": False, "writable": False, "schema_writable": False,
    }] if checks is None else checks
    driver = SimpleNamespace(
        connect=MagicMock(return_value=connection),
        rows=SimpleNamespace(dict_row=object()),
        IsolationLevel=SimpleNamespace(REPEATABLE_READ="invented_repeatable_read"),
    )
    return driver, connection, cursor


class ConnectionConfigurationSafetyTests(unittest.TestCase):
    def test_remote_and_ambiguous_hosts_are_rejected(self):
        for host in ("localhost", "192.0.2.1", "example.invalid", "127.0.0.1,192.0.2.1", ""):
            with self.subTest(host=host), self.assertRaises(SafeAuditError) as caught:
                ConnectionConfig.from_env({"PGHOST": host})
            self.assertEqual(caught.exception.code, "unsafe_connection")

    def test_libpq_override_environment_is_rejected(self):
        for key in ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR", "PGOPTIONS"):
            with self.subTest(key=key), self.assertRaises(SafeAuditError) as caught:
                ConnectionConfig.from_env({key: SYNTHETIC_SECRET})
            self.assertEqual(caught.exception.code, "unsafe_connection")
            self.assertNotIn(SYNTHETIC_SECRET, str(caught.exception))

    def test_database_connection_string_cannot_replace_endpoint(self):
        for database in ("postgresql://example.invalid/db", "host=example.invalid", "db name"):
            with self.subTest(database=database), self.assertRaises(SafeAuditError):
                ConnectionConfig.from_env({"PGDATABASE": database})

    def test_admin_username_and_unbounded_timeout_are_rejected(self):
        with self.assertRaises(SafeAuditError) as caught:
            ConnectionConfig.from_env({"PGUSER": "postgres"})
        self.assertEqual(caught.exception.code, "unsafe_role")
        for timeout in (0, 999, 600001):
            with self.subTest(timeout=timeout), self.assertRaises(SafeAuditError):
                ConnectionConfig.from_env({}, statement_timeout_ms=timeout)

    def test_direct_session_cannot_bypass_environment_validation(self):
        for key in ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR", "PGOPTIONS"):
            driver, _, _ = fake_driver()
            with self.subTest(key=key), patch.dict(os.environ, {key: "invented"}, clear=True):
                with self.assertRaises(SafeAuditError) as caught:
                    with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver):
                        self.fail("unsafe environment yielded a runner")
                self.assertEqual(caught.exception.code, "unsafe_connection")
                driver.connect.assert_not_called()


class MockReadOnlySessionTests(unittest.TestCase):
    def test_explicit_readonly_snapshot_and_endpoint_are_checked(self):
        driver, connection, cursor = fake_driver()
        with patch.dict(os.environ, {}, clear=True):
            with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver) as runner:
                self.assertIsInstance(runner, QueryRunner)
                self.assertTrue(connection.read_only)
                self.assertEqual(connection.isolation_level, driver.IsolationLevel.REPEATABLE_READ)
        keywords = driver.connect.call_args.kwargs
        self.assertEqual(keywords["host"], "127.0.0.1")
        self.assertEqual(keywords["hostaddr"], "127.0.0.1")
        self.assertEqual(keywords["dbname"], "pckeiba")
        self.assertEqual(keywords["passfile"], os.devnull)
        self.assertEqual(keywords["sslmode"], "disable")
        self.assertEqual(keywords["gssencmode"], "disable")
        self.assertEqual(keywords["require_auth"], "scram-sha-256,md5,password")
        self.assertFalse(keywords["autocommit"])
        self.assertIn("default_transaction_read_only=on", keywords["options"])
        self.assertIn("statement_timeout=60000", keywords["options"])
        self.assertIn("lock_timeout=2000", keywords["options"])
        self.assertIn("search_path=pg_catalog", keywords["options"])
        cursor.fetchmany.assert_called_once_with(2)
        cursor.fetchall.assert_not_called()
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()

    def test_server_readonly_and_isolation_fail_closed(self):
        for override in ({"read_only": "off"}, {"isolation": "read committed"}):
            row = {"read_only": "on", "isolation": "repeatable read", "privileged": False,
                   "writable": False, "schema_writable": False, **override}
            driver, connection, _ = fake_driver([row])
            with self.subTest(override=override), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SafeAuditError) as caught:
                    with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver):
                        self.fail("unverified read-only session was allowed")
                self.assertEqual(caught.exception.code, "not_read_only")
                connection.close.assert_called_once()

    def test_empty_server_safety_result_is_rejected(self):
        driver, _, _ = fake_driver([])
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SafeAuditError) as caught:
            with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver):
                self.fail("missing server safety result was allowed")
        self.assertEqual(caught.exception.code, "not_read_only")

    def test_privilege_ownership_and_missing_checks_are_rejected(self):
        baseline = {"read_only": "on", "isolation": "repeatable read", "privileged": False,
                    "writable": False, "schema_writable": False}
        for key in ("privileged", "writable", "schema_writable"):
            for missing in (False, True):
                row = dict(baseline)
                if missing:
                    row.pop(key)
                else:
                    row[key] = True
                driver, _, _ = fake_driver([row])
                with self.subTest(key=key, missing=missing), patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(SafeAuditError) as caught:
                        with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver):
                            self.fail("privileged or unverified role was allowed")
                    self.assertEqual(caught.exception.code, "unsafe_role")

    def test_connection_exception_does_not_expose_secret_or_original_traceback(self):
        driver, _, _ = fake_driver()
        driver.connect.side_effect = RuntimeError(SYNTHETIC_SECRET)
        with patch.dict(os.environ, {}, clear=True):
            try:
                with read_only_session(ConnectionConfig(), SYNTHETIC_SECRET, driver=driver):
                    self.fail("failing synthetic driver yielded a runner")
            except SafeAuditError as exception:
                self.assertEqual(exception.code, "connection_failed")
                self.assertNotIn(SYNTHETIC_SECRET, str(exception))
                rendered = "".join(traceback.format_exception(exception))
                self.assertNotIn(SYNTHETIC_SECRET, rendered)
            else:
                self.fail("a safe connection failure was expected")

    def test_empty_password_prevents_driver_call(self):
        driver, _, _ = fake_driver()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SafeAuditError):
            with read_only_session(ConnectionConfig(), "", driver=driver):
                self.fail("empty password was allowed")
        driver.connect.assert_not_called()


class SQLAndBoundedResultSafetyTests(unittest.TestCase):
    def test_write_and_side_effect_sql_is_rejected_before_execution(self):
        driver, connection, cursor = fake_driver()
        runner = QueryRunner(connection)
        queries = (
            "DELETE FROM public.invented", "INSERT INTO public.invented VALUES (1)",
            "WITH changed AS (UPDATE public.invented SET n = 1 RETURNING n) SELECT n FROM changed",
            "SELECT pg_sleep(1)", "SELECT nextval('invented')", "SELECT dblink('x', 'y')",
            "SELECT 1; SELECT 2", "SELECT 1 -- hidden", "SELECT 1 /* hidden */",
            "SELECT $$invented$$", "COPY public.invented TO STDOUT",
        )
        for query in queries:
            with self.subTest(query=query), self.assertRaises(SafeAuditError) as caught:
                runner.aggregate("invented.test", query)
            self.assertEqual(caught.exception.code, "unsafe_sql")
        cursor.execute.assert_not_called()
        driver.connect.assert_not_called()

    def test_catalog_status_literals_and_quoted_identifiers_are_allowed(self):
        validate_audit_sql("SELECT 'INSERT,UPDATE,DELETE' AS permissions, \"update\" FROM pg_catalog.pg_class")
        validate_audit_sql("WITH aggregates AS (SELECT count(*) AS n FROM pg_catalog.pg_class) SELECT n FROM aggregates")

    def test_overflow_uses_fetchmany_and_returns_no_partial_data(self):
        _, connection, cursor = fake_driver([{"records": 1}] * 3)
        with self.assertRaises(SafeAuditError) as caught:
            QueryRunner(connection).aggregate("invented.count", "SELECT count(*) AS records", max_rows=2)
        self.assertEqual(caught.exception.code, "result_limit")
        cursor.fetchmany.assert_called_once_with(3)
        cursor.fetchall.assert_not_called()

    def test_invalid_result_limit_is_rejected_before_sql(self):
        _, connection, cursor = fake_driver()
        for max_rows in (0, -1, 12001):
            with self.subTest(max_rows=max_rows), self.assertRaises(SafeAuditError):
                QueryRunner(connection).aggregate("invented.count", "SELECT count(*) AS records", max_rows=max_rows)
        cursor.execute.assert_not_called()

    def test_sqlstate_maps_to_safe_diagnosis_without_exception_text(self):
        for state, expected in (("57014", "query_timeout"), ("42501", "permission_denied"),
                                ("XX000", "database_error")):
            _, connection, cursor = fake_driver()
            error = RuntimeError(SYNTHETIC_SECRET)
            error.sqlstate = state
            cursor.execute.side_effect = error
            with self.subTest(state=state), self.assertRaises(SafeAuditError) as caught:
                QueryRunner(connection).aggregate("invented.count", "SELECT count(*) AS records")
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn(SYNTHETIC_SECRET, str(caught.exception))


class ReportLocationSafetyTests(unittest.TestCase):
    def setUp(self):
        # These are invented paths only. Do not create or remove directories.
        self.root = Path(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp") / "jra_synthetic_path_no_write"

    def test_synthetic_destination_is_restricted_to_project_reports(self):
        project = self.root / "invented_project"
        permitted = project / "reports" / "synthetic"
        self.assertEqual(reports.validate_output_dir(permitted, real=False, project_root=project), permitted.resolve())
        with self.assertRaises(SafeAuditError):
            reports.validate_output_dir(self.root / "elsewhere", real=False, project_root=project)

    def test_real_destination_cannot_use_project_reports(self):
        project = self.root / "invented_project"
        destination = project / "reports"
        with self.assertRaises(SafeAuditError):
            reports.validate_output_dir(destination, real=True, approved_root=destination, project_root=project, env={})

    def test_only_explicit_real_root_is_allowed(self):
        project = self.root / "invented_project"
        approved = self.root / "invented_local_reports"
        with patch.object(reports, "_local_disk", return_value=True):
            self.assertEqual(reports.validate_output_dir(approved, real=True, approved_root=approved,
                                                        project_root=project, env={}), approved.resolve())
            for destination in (self.root / "elsewhere", approved / "child"):
                with self.subTest(destination=destination), self.assertRaises(SafeAuditError):
                    reports.validate_output_dir(destination, real=True, approved_root=approved,
                                                project_root=project, env={})

    def test_onedrive_name_and_configured_sync_root_are_rejected(self):
        project = self.root / "invented_project"
        destinations = ((self.root / "OneDrive" / "reports", {}),
                        (self.root / "invented_sync" / "reports", {"OneDrive": str(self.root / "invented_sync")}))
        for destination, env in destinations:
            with self.subTest(destination=destination), self.assertRaises(SafeAuditError):
                reports.validate_output_dir(destination, real=True, approved_root=destination,
                                            project_root=project, env=env)

    def test_links_are_rejected_without_resolving_destination(self):
        destination = self.root / "invented_local_reports"
        with patch.object(reports, "_has_link", return_value=True), self.assertRaises(SafeAuditError):
            reports.validate_output_dir(destination, real=True, approved_root=destination,
                                        project_root=self.root / "invented_project", env={})

    def test_parent_junction_is_detected_without_creating_windows_links(self):
        def unsafe_child_probe():
            self.fail("a descendant was probed before its parent junction was rejected")

        parent = SimpleNamespace(is_symlink=lambda: False, is_junction=lambda: True)
        path = SimpleNamespace(is_symlink=unsafe_child_probe, is_junction=lambda: False, parents=(parent,))
        self.assertTrue(reports._has_link(path))

    def test_sync_root_link_is_rejected_before_resolve(self):
        destination, sync = self.root / "invented_local_reports", self.root / "invented_sync_link"
        original_resolve = Path.resolve

        def safe_resolve(path, *arguments, **keywords):
            if path == sync:
                self.fail("a sync-root junction was resolved before being rejected")
            return original_resolve(path, *arguments, **keywords)

        with patch.object(reports, "_has_link", side_effect=lambda path: path == sync), \
                patch.object(Path, "resolve", autospec=True, side_effect=safe_resolve), \
                self.assertRaises(SafeAuditError):
            reports.validate_output_dir(destination, real=True, approved_root=destination,
                                        project_root=self.root / "invented_project", env={"OneDrive": str(sync)})

    def test_network_output_is_rejected_before_any_filesystem_probe(self):
        destination = self.root / "invented_destination"
        with patch.object(reports, "_local_disk", return_value=False), \
                patch.object(reports, "_has_link", side_effect=AssertionError("unsafe filesystem probe")):
            for real in (False, True):
                with self.subTest(real=real), self.assertRaises(SafeAuditError):
                    reports.validate_output_dir(destination, real=real, approved_root=destination,
                                                project_root=self.root / "invented_project", env={})

    def test_unc_detection_does_not_access_a_share(self):
        self.assertFalse(reports._local_disk(Path(r"\\invented-server\invented-share\reports")))


class CLINoConnectionSafetyTests(unittest.TestCase):
    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_demo_never_discovers_driver_prompts_or_connects(self):
        output = Path("invented_synthetic_report.md")
        with patch.object(cli, "read_only_session") as session, \
                patch.object(cli.importlib.util, "find_spec") as discovery, \
                patch.object(cli.getpass, "getpass") as prompt, \
                patch.object(cli, "write_report", return_value={"audit.md": output}) as writer:
            status, _, _ = self.invoke(["demo"])
        self.assertEqual(status, 0)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.args[0]["source"], "synthetic")
        session.assert_not_called()
        discovery.assert_not_called()
        prompt.assert_not_called()

    def test_schema_and_audit_without_explicit_flag_never_connect(self):
        for arguments in (["schema"], ["audit", "--year", "2025"]):
            with self.subTest(arguments=arguments), patch.object(cli, "read_only_session") as session, \
                    patch.object(cli.importlib.util, "find_spec") as discovery, \
                    patch.object(cli.getpass, "getpass") as prompt:
                status, stdout, stderr = self.invoke(arguments)
                self.assertEqual(status, 2)
                self.assertEqual(stdout, "")
                self.assertIn("invalid_configuration", stderr)
                session.assert_not_called()
                discovery.assert_not_called()
                prompt.assert_not_called()

    def test_argument_errors_do_not_echo_accidentally_pasted_secret(self):
        with patch.object(cli, "read_only_session") as session:
            status, stdout, stderr = self.invoke(["schema", "--password", SYNTHETIC_SECRET])
        self.assertEqual(status, 2)
        self.assertNotIn(SYNTHETIC_SECRET, stdout + stderr)
        self.assertNotIn("--password", stdout + stderr)
        session.assert_not_called()

    def test_missing_driver_is_reported_before_secret_prompt_or_connection(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(cli, "validate_output_dir") as location, \
                patch.object(cli.importlib.util, "find_spec", return_value=None), \
                patch.object(cli.getpass, "getpass") as prompt, \
                patch.object(cli, "read_only_session") as session:
            status, stdout, stderr = self.invoke(["schema", "--execute-local"])
        self.assertEqual(status, 2)
        self.assertIn("missing_driver", stderr)
        self.assertEqual(stdout, "")
        location.assert_called_once()
        prompt.assert_not_called()
        session.assert_not_called()

    def test_unexpected_exception_is_redacted(self):
        with patch.object(cli, "run", side_effect=RuntimeError(SYNTHETIC_SECRET)):
            status, stdout, stderr = self.invoke(["demo"])
        self.assertEqual(status, 2)
        self.assertNotIn(SYNTHETIC_SECRET, stdout + stderr)

    def test_unmasked_password_fallback_is_blocked_before_input(self):
        fallback_input = []

        def console_unavailable(*arguments, **keywords):
            warnings.warn(SYNTHETIC_SECRET, cli.getpass.GetPassWarning)
            fallback_input.append("unmasked_input_was_reached")
            return SYNTHETIC_SECRET

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(cli, "validate_output_dir"), \
                patch.object(cli.importlib.util, "find_spec", return_value=object()), \
                patch.object(cli.getpass, "getpass", side_effect=console_unavailable), \
                patch.object(cli, "read_only_session") as session:
            status, stdout, stderr = self.invoke(["schema", "--execute-local"])
        self.assertEqual(status, 2)
        self.assertEqual(fallback_input, [])
        self.assertNotIn(SYNTHETIC_SECRET, stdout + stderr)
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
