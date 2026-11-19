"""Synthetic-only checks for the single-date empty-export diagnosis."""
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.db_connection import validate_audit_sql
from test_ml_v01 import mapping

spec = importlib.util.spec_from_file_location("v02_source_check", Path(__file__).resolve().parents[1] / "scripts" / "check_v02_source.py")
source_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_check)


class V02SourceCheckTests(unittest.TestCase):
    def test_missing_race_horse_data_and_ready_scope_are_distinguished_without_rows(self):
        fields = ("ra_rows", "ra_final_rows", "se_rows", "se_final_rows", "matched_final_runner_rows")
        cases = (((0, 0, 0, 0, 0), "ra_se_missing"), ((0, 0, 8, 8, 0), "ra_missing"),
                 ((1, 1, 0, 0, 0), "se_missing"), ((1, 0, 8, 8, 0), "final_category_missing"),
                 ((1, 1, 8, 8, 0), "race_keys_unmatched"), ((1, 1, 8, 8, 8), "source_ready"))
        for values, expected in cases:
            db = MagicMock()
            db.aggregate.return_value = [dict(zip(fields, values), private_record="SYNTHETIC_PRIVATE")]
            @contextmanager
            def session(config, secret):
                self.assertEqual(config.user, "jra_audit_reader")
                self.assertEqual(config.statement_timeout_ms, 10000)
                yield db
            with self.subTest(reason=expected), patch.object(source_check.safety, "read_json", return_value=mapping()), \
                    patch.object(source_check, "_password", return_value="SYNTHETIC_PASSWORD"), \
                    patch.object(source_check, "read_only_session", session):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(source_check.main(["--race-date", "2030-01-05"]), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["reason"], expected)
                self.assertEqual(set(result["counts"]), set(fields))
                self.assertNotIn("PRIVATE", output.getvalue())
                sql, params = db.aggregate.call_args.args[1:3]
                validate_audit_sql(sql)
                self.assertIn("JOIN se USING (" + ", ".join('"' + name + '"' for name in RACE_KEY) + ")", sql)
                self.assertEqual(params[:2], ("2030", "0105"))
                self.assertEqual(db.aggregate.call_count, 1)

    def test_invalid_date_and_unexpected_failure_never_print_inputs_or_exception(self):
        for argv in (["--race-date", "SYNTHETIC_PRIVATE"], ["--unexpected", "SYNTHETIC_PRIVATE"]):
            output = io.StringIO()
            with patch.object(source_check, "read_only_session") as connect, redirect_stdout(output):
                self.assertEqual(source_check.main(argv), 2)
            self.assertNotIn("PRIVATE", output.getvalue())
            connect.assert_not_called()
