"""Synthetic catalog/aggregate tests. No PostgreSQL client or network required."""

from __future__ import annotations

import json
import re
import unittest
from types import SimpleNamespace

from jra_racing_ai.audit_odds import RACE_KEY, audit_odds
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.inspect_schema import inspect_schema


class FakeDB:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def aggregate(self, query_id, sql, params=(), max_rows=1000):
        validate_audit_sql(sql)
        self.calls.append((query_id, sql, params, max_rows))
        value = self.responses.get(query_id, [])
        try:
            value = self.responses.get((query_id, params), value)
        except TypeError:
            # Scope params contain an array for PostgreSQL ANY, intentionally.
            pass
        if query_id == "odds.summary":
            defaults = {"records": 0, "incomplete_key_records": 0, "blank_key_records": 0}
            defaults.update({name + "_null_records": 0 for name in RACE_KEY})
            value = [{**defaults, **row} for row in value] if value else [defaults]
        elif query_id == "odds.grain" and not value:
            value = [{"complete_race_key_groups": 0, "records_with_complete_keys": 0,
                      "single_record_groups": 0, "multiple_record_groups": 0,
                      "minimum_records_per_race": None, "maximum_records_per_race": None}]
        return [dict(row) for row in value]


def odds_context(db=None, mapped_status=False, global_summary=False):
    columns = [
        {"name": name, "data_type": "character varying(4)", "type_name": "varchar", "type_is_builtin": True, "nullable": False}
        for name in RACE_KEY
    ]
    columns += [
        {"name": "data_kubun", "data_type": "text", "type_name": "text", "type_is_builtin": True, "nullable": True},
        {"name": "hypothetical_happyo_jifun", "data_type": "text", "type_name": "text", "type_is_builtin": True, "nullable": True},
        {"name": "hypothetical_tansho_odds", "data_type": "text", "type_name": "text", "type_is_builtin": True, "nullable": True},
    ]
    return SimpleNamespace(
        db=db or FakeDB(),
        scope=SimpleNamespace(year=2025, venues=("01", "02"), global_summary=global_summary),
        status_columns={"jvd_o1": "data_kubun"} if mapped_status else {},
        schema={"jvd_o1": {
            "name": "jvd_o1", "kind": "table", "auditable": True,
            "table_access_method": "heap",
            "row_security": False, "force_row_security": False,
            "columns": columns, "primary_key": ["hypothetical_sequence"],
        }},
    )


class SchemaTests(unittest.TestCase):
    def test_catalog_allowlists_and_relation_exclusion(self):
        inventory = [
            {"name": "jvd_o1", "kind": "table", "estimated_rows": 42,
             "table_access_method": "heap",
             "size_bytes": 8192, "index_size_bytes": 8192, "total_size_bytes": 16384,
             "row_security": False, "force_row_security": False, "secret_comment": "SYNTHETIC_SECRET"},
            {"name": "hypothetical_remote", "kind": "foreign_table", "estimated_rows": None,
             "row_security": False, "force_row_security": False},
            {"name": "hypothetical_partitioned", "kind": "partitioned_table", "estimated_rows": 20,
             "row_security": False, "force_row_security": False},
            {"name": "hypothetical_rls", "kind": "table", "estimated_rows": 10,
             "table_access_method": "heap",
             "row_security": True, "force_row_security": False},
        ]
        db = FakeDB({
            "schema.tables": inventory,
            "schema.columns": [{"name": "fake_column", "data_type": "text", "type_name": "text",
                                "type_is_builtin": True, "nullable": False,
                                "ordinal_position": 1, "default": "SYNTHETIC_SECRET"}],
            "schema.primary_key": [{"name": "fake_pk", "columns": ["fake_column"]}],
            "schema.foreign_keys": [{"name": "fake_fk", "columns": ["fake_column"],
                                     "referenced_schema": "public", "referenced_table": "fake_parent",
                                     "referenced_columns": ["fake_parent_key"], "validated": True,
                                     "definition": "SYNTHETIC_SECRET"}],
            "schema.indexes": [{"name": "fake_index", "columns": ["fake_column"], "unique": True,
                                "valid": True, "has_expressions": True, "has_predicate": True,
                                "expression": "SYNTHETIC_SECRET", "predicate": "SYNTHETIC_SECRET"}],
        })
        result = inspect_schema(db)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["catalog_counts"]["listed_relations"], 4)
        self.assertEqual(result["catalog_counts"]["auditable_heap_tables"], 1)
        self.assertEqual([t["auditable"] for t in result["tables"]], [True, False, False, False])
        self.assertEqual(result["tables"][0]["primary_key"], ["fake_column"])
        self.assertIsNone(result["tables"][1]["estimated_rows"])
        self.assertNotIn("SYNTHETIC_SECRET", json.dumps(result))
        self.assertTrue(result["tables"][0]["indexes"][0]["has_expressions"])
        self.assertTrue(all("pg_catalog." in sql for _, sql, _, _ in db.calls))
        self.assertFalse(any(re.search(r"\bFROM\s+(?:ONLY\s+)?(?:\"?public\"?)\.", sql, re.I) for _, sql, _, _ in db.calls))
        self.assertFalse(any("count(*)" in sql.lower() for _, sql, _, _ in db.calls))

    def test_unknown_security_state_is_excluded(self):
        db = FakeDB({"schema.tables": [{"name": "fake_table", "kind": "table", "estimated_rows": None}]})
        result = inspect_schema(db)
        self.assertFalse(result["tables"][0]["auditable"])
        self.assertTrue(result["tables"][0]["warnings"])


class OddsTests(unittest.TestCase):
    def test_missing_table_is_no_query(self):
        ctx = odds_context()
        ctx.schema = {}
        result = audit_odds(ctx)
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])

    def test_missing_key_is_no_query(self):
        ctx = odds_context()
        ctx.schema["jvd_o1"]["columns"] = [c for c in ctx.schema["jvd_o1"]["columns"] if c["name"] != "race_bango"]
        result = audit_odds(ctx)
        self.assertEqual(result["missing_race_key_columns"], ["race_bango"])
        self.assertEqual(ctx.db.calls, [])

    def test_unconfirmed_type_is_no_query(self):
        ctx = odds_context()
        ctx.schema["jvd_o1"]["columns"][0]["type_is_builtin"] = False
        result = audit_odds(ctx)
        self.assertEqual(result["unconfirmed_column_types"], ["kaisai_nen"])
        self.assertEqual(ctx.db.calls, [])

    def test_foreign_partition_and_rls_are_no_query(self):
        for kind, rls in [("foreign_table", False), ("partitioned_table", False), ("table", True)]:
            with self.subTest(kind=kind, rls=rls):
                ctx = odds_context()
                ctx.schema["jvd_o1"].update(kind=kind, row_security=rls)
                result = audit_odds(ctx)
                self.assertEqual(result["status"], "needs_confirmation")
                self.assertEqual(ctx.db.calls, [])

    def test_custom_table_access_method_is_no_query(self):
        ctx = odds_context()
        ctx.schema["jvd_o1"]["table_access_method"] = "hypothetical_extension_am"
        result = audit_odds(ctx)
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])

    def test_default_scope_and_candidates_do_not_fetch_values(self):
        db = FakeDB({
            "odds.summary": [{"records": 5, "incomplete_key_records": 1, "blank_key_records": 1}],
            "odds.grain": [{"complete_race_key_groups": 2, "records_with_complete_keys": 4,
                            "single_record_groups": 1, "multiple_record_groups": 1,
                            "minimum_records_per_race": 1, "maximum_records_per_race": 3}],
            "odds.grain_distribution": [{"records_per_race_band": "1", "race_key_groups": 1},
                                         {"records_per_race_band": "2-5", "race_key_groups": 1}],
        })
        result = audit_odds(odds_context(db))
        self.assertEqual(result["record_grain"], "要確認")
        self.assertEqual(result["global_year_coverage"]["status"], "not_run")
        self.assertEqual(result["status_summary"]["status"], "not_run")
        self.assertEqual(result["race_key_grain_summary"][0]["multiple_record_groups"], 1)
        self.assertEqual(result["time_column_candidates"][0]["meaning"], "要確認")
        self.assertEqual(result["odds_column_candidates"][0]["name"], "hypothetical_tansho_odds")
        for query_id, sql, params, max_rows in db.calls:
            self.assertNotIn("hypothetical_happyo_jifun", sql)
            self.assertNotIn("hypothetical_tansho_odds", sql)
            self.assertNotIn("data_kubun", sql)
            self.assertIn('FROM ONLY "public"."jvd_o1"', sql)
            self.assertTrue(params)
            self.assertIn("2025", str(params))
            self.assertLessEqual(max_rows, 7)
            self.assertRegex(query_id, r"^[a-z_\.]+$")

    def test_mapped_status_is_validated_and_global_range_is_opt_in(self):
        db = FakeDB({
            "odds.status": [{"status_code": "7", "records": 4}, {"status_code": "INVALID", "records": 1}],
            "odds.global_year_coverage": [{"minimum_year": 2020, "maximum_year": 2025,
                                           "null_year_records": 0, "invalid_year_records": 1}],
        })
        result = audit_odds(odds_context(db, mapped_status=True, global_summary=True))
        self.assertEqual(result["status_summary"]["column"], "data_kubun")
        self.assertEqual(result["status_summary"]["meaning"], "要確認")
        self.assertEqual(result["global_year_coverage"]["summary"][0]["minimum_year"], 2020)
        status_sql = next(sql for query_id, sql, _, _ in db.calls if query_id == "odds.status")
        self.assertIn("~ '^[0-9]$'", status_sql)
        self.assertIn("'INVALID'", status_sql)
        coverage_params = next(params for query_id, _, params, _ in db.calls if query_id == "odds.global_year_coverage")
        self.assertEqual(coverage_params, ())

    def test_unexpected_aggregate_fields_are_discarded(self):
        db = FakeDB({"odds.summary": [{"records": 1, "horse_name": "SYNTHETIC_SECRET"}]})
        result = audit_odds(odds_context(db))
        self.assertNotIn("SYNTHETIC_SECRET", json.dumps(result))

    def test_unvalidated_bucket_is_rejected(self):
        db = FakeDB({"odds.status": [{"status_code": "SYNTHETIC_SECRET", "records": 1}]})
        with self.assertRaises(SafeAuditError) as raised:
            audit_odds(odds_context(db, mapped_status=True))
        self.assertNotIn("SYNTHETIC_SECRET", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
