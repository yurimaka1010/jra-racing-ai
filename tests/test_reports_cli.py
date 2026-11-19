import importlib
import io
import json
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jra_racing_ai import cli
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.generate_report import PROJECT_ROOT, build_report, write_report
from jra_racing_ai.synthetic import demonstration_report


class ReportCliTests(unittest.TestCase):
    def test_synthetic_set_difference_is_anonymous(self):
        report = demonstration_report()
        self.assertEqual(report["source"], "synthetic")
        self.assertEqual(report["audits"]["horses"]["only_ra_races"], 1)
        self.assertEqual(report["audits"]["horses"]["only_se_races"], 1)
        self.assertEqual(report["audits"]["horses"]["record_count"], 4)
        payload = json.dumps(report)
        self.assertNotIn("0103", payload)
        self.assertNotIn("password", payload.lower())

    def test_imports_have_no_connection_or_output_side_effects(self):
        with patch("jra_racing_ai.db_connection.read_only_session") as connection, \
                patch.object(Path, "mkdir") as mkdir:
            for name in ("audit_races", "audit_horses", "audit_odds", "inspect_schema"):
                importlib.reload(importlib.import_module("jra_racing_ai." + name))
            connection.assert_not_called()
            mkdir.assert_not_called()

    def test_real_source_cannot_save_to_synthetic_directory(self):
        report = build_report({"tables": []}, {}, source="local_database")
        with patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(SafeAuditError) as caught:
                write_report(report, PROJECT_ROOT / "reports")
            self.assertEqual(caught.exception.code, "unsafe_output")
            mkdir.assert_not_called()

    def test_unknown_arguments_are_not_echoed(self):
        out = io.StringIO()
        with redirect_stderr(out):
            result = cli.main(["schema", "--password", "FAKE_SECRET_SENTINEL"])
        self.assertEqual(result, 2)
        self.assertNotIn("FAKE_SECRET_SENTINEL", out.getvalue())

    def test_missing_driver_blocks_before_password_and_connection(self):
        with patch("jra_racing_ai.cli.validate_output_dir"), \
                patch("jra_racing_ai.cli.ConnectionConfig.from_env"), \
                patch("jra_racing_ai.cli.importlib.util.find_spec", return_value=None), \
                patch("jra_racing_ai.cli.getpass.getpass") as password, \
                patch("jra_racing_ai.cli.read_only_session") as session:
            with self.assertRaises(SafeAuditError) as caught:
                cli.run(cli.parser().parse_args(["schema", "--execute-local"]))
            self.assertEqual(caught.exception.code, "missing_driver")
            password.assert_not_called()
            session.assert_not_called()

    def test_hidden_password_fallback_is_rejected(self):
        def unsafe_prompt(*args, **kwargs):
            warnings.warn("FAKE_SECRET_SENTINEL", cli.getpass.GetPassWarning)
        with patch("jra_racing_ai.cli.validate_output_dir"), \
                patch("jra_racing_ai.cli.ConnectionConfig.from_env"), \
                patch("jra_racing_ai.cli.importlib.util.find_spec", return_value=object()), \
                patch.dict("os.environ", {}, clear=True), \
                patch("jra_racing_ai.cli.getpass.getpass", side_effect=unsafe_prompt), \
                patch("jra_racing_ai.cli.read_only_session") as session, \
                redirect_stderr(io.StringIO()) as out:
            code = cli.main(["schema", "--execute-local"])
            self.assertEqual(code, 2)
            self.assertNotIn("FAKE_SECRET_SENTINEL", out.getvalue())
            session.assert_not_called()

    def test_demo_never_imports_driver_or_connects(self):
        paths = {"audit.md": Path("synthetic-audit.md")}
        with patch("jra_racing_ai.cli.write_report", return_value=paths) as writer, \
                patch("jra_racing_ai.cli.read_only_session") as session, \
                patch("jra_racing_ai.cli.importlib.util.find_spec") as driver, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["demo"]), 0)
            self.assertEqual(writer.call_args.args[0]["source"], "synthetic")
            session.assert_not_called()
            driver.assert_not_called()

    def test_unhandled_exception_does_not_leak(self):
        with patch("jra_racing_ai.cli.run", side_effect=RuntimeError("FAKE_SECRET_SENTINEL")), \
                redirect_stderr(io.StringIO()) as out:
            self.assertEqual(cli.main(["demo"]), 2)
            self.assertNotIn("FAKE_SECRET_SENTINEL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
