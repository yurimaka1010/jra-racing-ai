"""Regression: application export failures must retain their safe reason codes."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jra_racing_ai import ml_cli
from jra_racing_ai.db_connection import read_only_session
from jra_racing_ai.ml import data, safety
from jra_racing_ai.ml.demo import synthetic_rows
from jra_racing_ai.ml.safety import MLFailure
from test_ml_v01 import mapping, catalog_for
from test_security_review import fake_driver, SYNTHETIC_SECRET


class ExportErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(safety, "ML_ROOT", self.root).start()
        patch.dict("os.environ", {}, clear=True).start()
        patch.object(ml_cli, "_password", return_value="SYNTHETIC_PASSWORD").start()
        self.mapping = mapping()
        self.mapping_path = self.root / "mapping.json"
        safety.write_json(self.mapping_path, self.mapping)
        self.driver, self.connection, self.cursor = fake_driver()
        self.checks = self.cursor.fetchmany.return_value
        patch.object(data, "inspect_sources", return_value=({"tables": catalog_for(self.mapping)}, {})).start()
        patch.object(ml_cli, "read_only_session", side_effect=lambda config, secret:
                     read_only_session(config, secret, driver=self.driver)).start()
        self.rows = synthetic_rows(12)[:8]

    def invoke(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = ml_cli.main(["export", "--execute-local", "--mapping", str(self.mapping_path),
                                  "--start", "2020-01-01", "--end", "2020-12-31", "--max-rows", "100"])
        self.assertNotIn(SYNTHETIC_SECRET, output.getvalue())
        self.assertNotIn("SYNTHETIC_PASSWORD", output.getvalue())
        self.connection.rollback.assert_called_once()
        self.connection.close.assert_called_once()
        return result, json.loads(output.getvalue())

    def test_empty_export_reports_data_absence_not_database_failure(self):
        self.cursor.fetchmany.side_effect = [self.checks, []]
        code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output, {"status": "stopped", "reason": "insufficient_history"})
        self.assertFalse(list(self.root.rglob("runners.csv")))
        self.assertFalse(list(self.root.rglob("dataset.json")))

    def test_duplicate_export_keeps_reason_and_leaves_no_partial_dataset(self):
        self.cursor.fetchmany.side_effect = [self.checks, [self.rows[0], self.rows[0]]]
        code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output["reason"], "duplicate_entry")
        self.assertFalse(list(self.root.rglob("runners.csv")))

    def test_schema_failure_keeps_fixed_application_reason(self):
        self.cursor.fetchmany.side_effect = [self.checks]
        with patch.object(data, "inspect_sources", side_effect=MLFailure("invalid_schema")):
            code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output["reason"], "invalid_schema")

    def test_unknown_application_code_is_still_sanitized(self):
        self.cursor.fetchmany.side_effect = [self.checks]
        with patch.object(data, "inspect_sources", side_effect=MLFailure(SYNTHETIC_SECRET)):
            code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output["reason"], "internal_error")

    def test_real_query_error_is_still_sanitized_as_database_error(self):
        error = RuntimeError(SYNTHETIC_SECRET)
        error.sqlstate = "42703"
        self.cursor.execute.side_effect = [None, error]
        self.cursor.fetchmany.side_effect = [self.checks]
        code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output["reason"], "database_error")
        self.assertEqual(output["diagnostics"], {"query_id": "ml.export", "sqlstate": "42703"})

    def test_unexpected_exception_in_export_does_not_leak(self):
        self.cursor.fetchmany.side_effect = [self.checks]
        with patch.object(data, "inspect_sources", side_effect=RuntimeError(SYNTHETIC_SECRET)):
            code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output, {"status": "stopped", "reason": "database_error"})

    def test_successful_export_still_completes_with_aggregate_only_output(self):
        self.cursor.fetchmany.side_effect = [self.checks, self.rows, []]
        code, output = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["rows"], 8)
        self.assertEqual(output["races"], 1)
        self.assertNotIn("records", output)
        self.assertTrue(list(self.root.rglob("dataset.json")))
