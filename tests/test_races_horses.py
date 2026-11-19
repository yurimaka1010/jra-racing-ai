"""Fabricated records interpreted by a mock; no PostgreSQL or credentials used."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
from types import SimpleNamespace
import unittest

from jra_racing_ai.core import RACE_KEY, Scope
from jra_racing_ai.audit_races import audit_races, valid_key_format
from jra_racing_ai.audit_horses import audit_horses


def fake_race(number: str, category: str = "7", **changes) -> dict:
    values = dict(zip(RACE_KEY, ("2025", "0105", "01", "01", "01", number)))
    values.update(data_kubun=category, horse_fake_id="FICTIONAL_SECRET_HORSE")
    values.update(changes)
    return values


def key(row: dict) -> tuple:
    return tuple(None if row.get(name) is None else str(row[name]).strip() for name in RACE_KEY)


def valid(row: dict) -> bool:
    values = key(row)
    if any(value is None for value in values):
        return False
    if not all(value.isascii() and value.isdigit() for value in values):
        return False
    if len(values[0]) != 4 or len(values[1]) != 4 or any(not 1 <= len(value) <= 2 for value in values[2:]):
        return False
    try:
        date(int(values[0]), int(values[1][:2]), int(values[1][2:]))
    except ValueError:
        return False
    return all(1 <= int(value) <= 99 for value in values[2:])


def category(row: dict) -> str:
    value = row.get("data_kubun")
    if value is None:
        return "null"
    value = str(value).strip()
    return value if value in set("0123456789") else "other"


class SyntheticAggregate:
    """SQL is recorded, never executed; return fabricated grouped aggregates."""

    def __init__(self, ra: list[dict], se: list[dict], scope):
        self.tables = {"ra": ra, "se": se}
        self.scope = scope
        self.calls = []

    def scoped(self, prefix: str, global_scan: bool = False) -> list[dict]:
        return [row for row in self.tables[prefix] if global_scan or (
            str(row.get("kaisai_nen", "")).strip() == str(self.scope.year)
            and str(row.get("keibajo_code", "")).strip() in self.scope.venues
        )]

    def aggregate(self, query_id, sql, params=(), max_rows=1000):
        self.calls.append((query_id, sql, params, max_rows))
        prefix = query_id.split(".")[0]
        rows = self.scoped(prefix, ".global" in query_id)
        complete = [row for row in rows if all(value is not None for value in key(row))]
        keys = Counter(key(row) for row in complete)
        action = query_id.split(".")[-1]
        if action == "summary":
            counts = {
                "records": len(rows), "null_key_records": len(rows) - len(complete),
                "empty_key_records": sum(any(value == "" for value in key(row)) for row in complete),
                "invalid_format_records": sum(not valid(row) for row in complete),
                "distinct_complete_races": len(keys),
            }
            repeated = sum(count > 1 for count in keys.values())
            excess = sum(count - 1 for count in keys.values())
            if prefix == "ra":
                counts.update(duplicate_key_groups=repeated, duplicate_excess_records=excess)
            else:
                counts.update(races_with_multiple_records=repeated, records_beyond_one_per_race=excess)
            result = [counts]
        elif action == "key_quality":
            result = [{"field": name, "null_records": sum(row.get(name) is None for row in rows), "empty_records": sum(row.get(name) == "" for row in rows)} for name in RACE_KEY]
        elif action == "venues":
            result = [{"venue": venue, "records": count} for venue, count in Counter(row["keibajo_code"] for row in rows).items()]
        elif action == "period":
            dates = []
            for row in rows:
                try:
                    dates.append(date(int(row["kaisai_nen"]), int(row["kaisai_tsukihi"][:2]), int(row["kaisai_tsukihi"][2:])).isoformat())
                except (ValueError, TypeError):
                    pass
            result = [{"earliest_date": min(dates, default=None), "latest_date": max(dates, default=None), "valid_date_records": len(dates), "invalid_or_null_date_records": len(rows) - len(dates)}]
        elif action == "categories":
            result = [{"category": cat, "records": sum(category(row) == cat for row in rows), "distinct_races": len({key(row) for row in complete if category(row) == cat})} for cat in sorted({category(row) for row in rows})]
        elif action == "horse_count_distribution":
            def bucket(count):
                return str(count) if count <= 40 else "41-100" if count <= 100 else "101-1000" if count <= 1000 else "1001+"
            result = [{"records_per_race": size, "races": count} for size, count in Counter(bucket(n) for n in keys.values()).items()]
        elif action == "race_sets":
            ra = {key(row) for row in self.scoped("ra") if None not in key(row)}
            se = {key(row) for row in self.scoped("se") if None not in key(row)}
            rav = {key(row) for row in self.scoped("ra") if valid(row)}
            sev = {key(row) for row in self.scoped("se") if valid(row)}
            result = [{
                "ra_complete_distinct_races": len(ra), "se_complete_distinct_races": len(se),
                "shared_complete_races": len(ra & se), "races_only_in_ra": len(ra - se), "races_only_in_se": len(se - ra),
                "ra_valid_format_distinct_races": len(rav), "se_valid_format_distinct_races": len(sev),
                "valid_format_races_only_in_ra": len(rav - sev), "valid_format_races_only_in_se": len(sev - rav),
            }]
        elif action == "category_race_sets":
            ra = {(category(row), key(row)) for row in self.scoped("ra") if None not in key(row)}
            se = {(category(row), key(row)) for row in self.scoped("se") if None not in key(row)}
            result = [{"category": cat, "ra_distinct_races": sum(c == cat for c, _ in ra), "se_distinct_races": sum(c == cat for c, _ in se), "races_only_in_ra": sum(c == cat for c, _ in ra - se), "races_only_in_se": sum(c == cat for c, _ in se - ra)} for cat in sorted({c for c, _ in ra | se})]
        elif action == "horse_duplicates":
            identities = Counter(key(row) + (row["horse_fake_id"],) for row in complete if row["horse_fake_id"] is not None)
            result = [{"duplicate_identity_groups": sum(n > 1 for n in identities.values()), "duplicate_excess_records": sum(n - 1 for n in identities.values()), "null_identity_records": sum(None in key(row) or row["horse_fake_id"] is None for row in rows)}]
        elif action == "global_years":
            counts = Counter("null" if row["kaisai_nen"] is None else row["kaisai_nen"] if isinstance(row["kaisai_nen"], str) and len(row["kaisai_nen"]) == 4 and row["kaisai_nen"].isascii() and row["kaisai_nen"].isdigit() and row["kaisai_nen"] != "0000" else "other" for row in rows)
            result = [{"year": year, "records": count} for year, count in sorted(counts.items())]
        elif action == "global_key_quality":
            quality = {"records": len(rows), "null_key_records": len(rows) - len(complete), "empty_key_records": sum("" in key(row) for row in complete), "invalid_format_records": sum(not valid(row) for row in complete)}
            for name in RACE_KEY:
                quality[name + "_null_records"] = sum(row.get(name) is None for row in rows)
                quality[name + "_empty_records"] = sum(row.get(name) == "" for row in rows)
            result = [quality]
        else:
            raise AssertionError("Unexpected fabricated query: " + query_id)
        if len(result) > max_rows:
            raise AssertionError("Fabricated aggregate exceeded requested row bound")
        return result


def context(ra=None, se=None, mapping=True, horse_identity=False, global_summary=False):
    scope = Scope(year=2025, venues=tuple(f"{n:02}" for n in range(1, 11)), global_summary=global_summary)
    schema = {}
    for table in ("jvd_ra", "jvd_se"):
        schema[table] = {
            "name": table, "auditable": True, "kind": "table", "table_access_method": "heap",
            "row_security": False, "force_row_security": False,
            "columns": [{"name": name, "data_type": "character varying", "type_is_builtin": True, "nullable": True} for name in RACE_KEY + ("data_kubun", "horse_fake_id")],
            "primary_key": ["horse_fake_id"] if table == "jvd_se" else [],
        }
    if ra is None:
        ra = [fake_race("01"), fake_race("01"), fake_race("02", "9"), fake_race("03"), fake_race("04", kaisai_kai=None), fake_race("05", "PRIVATE_FREE_TEXT", kaisai_tsukihi="0231"), fake_race("", "7"), fake_race("08", kaisai_nen="2024"), fake_race("09", keibajo_code="11")]
    if se is None:
        se = [fake_race("01", horse_fake_id="FICTIONAL_SECRET_HORSE_A"), fake_race("01", horse_fake_id="FICTIONAL_SECRET_HORSE_B"), fake_race("01", horse_fake_id="FICTIONAL_SECRET_HORSE_B"), fake_race("02", "9"), fake_race("07"), fake_race("04", kaisai_kai=None)]
    return SimpleNamespace(
        db=SyntheticAggregate(ra, se, scope), scope=scope, schema=schema,
        status_columns={"jvd_ra": "data_kubun", "jvd_se": "data_kubun"} if mapping else {},
        horse_identity_columns=("horse_fake_id",) if horse_identity else (),
    )


class RacesHorsesTests(unittest.TestCase):
    def test_scope_duplicates_null_and_format_are_separate(self):
        ctx = context()
        result = audit_races(ctx)
        self.assertEqual(result["summary"]["records"], 7)
        self.assertEqual(result["summary"]["null_key_records"], 1)
        self.assertEqual(result["summary"]["invalid_format_records"], 2)
        self.assertEqual(result["summary"]["empty_key_records"], 1)
        self.assertEqual(result["summary"]["duplicate_key_groups"], 1)
        self.assertEqual(result["summary"]["duplicate_excess_records"], 1)
        self.assertEqual(result["period"]["invalid_or_null_date_records"], 1)
        self.assertEqual({r["category"] for r in result["category_counts"]}, {"7", "9", "other"})
        self.assertEqual(result["reference_2025"]["difference"]["7"]["records"], 5 - 3455)
        self.assertFalse(any("global" in query for query, *_ in ctx.db.calls))

    def test_equal_counts_do_not_imply_matching_race_sets(self):
        ctx = context(ra=[fake_race("01"), fake_race("02")], se=[fake_race("01"), fake_race("03")])
        result = audit_horses(ctx)
        counts = result["race_set_comparison"]["counts"]
        self.assertEqual(counts["ra_complete_distinct_races"], counts["se_complete_distinct_races"])
        self.assertEqual(counts["races_only_in_ra"], 1)
        self.assertEqual(counts["races_only_in_se"], 1)
        query = next(call for call in ctx.db.calls if call[0] == "se.race_sets")
        self.assertIn("EXCEPT", query[1])
        self.assertIn("INTERSECT", query[1])
        self.assertEqual(query[2][:len(query[2]) // 2], query[2][len(query[2]) // 2:])

    def test_race_repetition_is_distinguished_from_horse_identity_duplicates(self):
        ctx = context(horse_identity=True)
        result = audit_horses(ctx)
        self.assertEqual(result["summary"]["races_with_multiple_records"], 1)
        self.assertEqual(result["summary"]["records_beyond_one_per_race"], 2)
        self.assertEqual(result["horse_identity_duplicates"]["counts"]["duplicate_excess_records"], 1)
        self.assertIn({"records_per_race": "3", "races": 1}, result["horse_records_per_race_distribution"])
        self.assertEqual(result["race_set_comparison"]["counts"]["races_only_in_ra"], 3)
        self.assertEqual(result["race_set_comparison"]["counts"]["races_only_in_se"], 1)

    def test_unconfirmed_mappings_do_not_guess_from_names_or_primary_key(self):
        ctx = context(mapping=False)
        result = audit_horses(ctx)
        self.assertEqual(result["category_counts"]["status"], "needs_confirmation")
        self.assertEqual(result["horse_identity_duplicates"]["status"], "needs_confirmation")
        self.assertFalse(any(q.endswith("categories") or q.endswith("horse_duplicates") for q, *_ in ctx.db.calls))

    def test_outputs_never_contain_source_names_ids_or_free_text(self):
        ctx = context(horse_identity=True)
        output = json.dumps({"ra": audit_races(ctx), "se": audit_horses(ctx)}, ensure_ascii=False)
        self.assertNotIn("FICTIONAL_SECRET_HORSE", output)
        self.assertNotIn("PRIVATE_FREE_TEXT", output)
        self.assertNotIn("0231", output)

    def test_missing_keys_and_nonlocal_relations_skip_before_queries(self):
        ctx = context()
        ctx.schema["jvd_ra"]["columns"] = ctx.schema["jvd_ra"]["columns"][1:]
        self.assertEqual(audit_races(ctx)["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])
        ctx.schema["jvd_se"]["auditable"] = False
        self.assertEqual(audit_horses(ctx)["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])

    def test_custom_key_type_skips_before_queries(self):
        ctx = context()
        ctx.schema["jvd_ra"]["columns"][0]["data_type"] = "untrusted_custom_type"
        self.assertEqual(audit_races(ctx)["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])

    def test_custom_status_type_is_not_cast_or_aggregated(self):
        ctx = context()
        for col in ctx.schema["jvd_ra"]["columns"]:
            if col["name"] == "data_kubun":
                col["data_type"] = "custom_status_type"
        self.assertEqual(audit_races(ctx)["category_counts"]["status"], "needs_confirmation")
        self.assertFalse(any(q == "ra.categories" for q, *_ in ctx.db.calls))

    def test_unspecified_relation_safety_skips_before_queries(self):
        ctx = context()
        ctx.schema["jvd_ra"].pop("auditable")
        self.assertEqual(audit_races(ctx)["status"], "needs_confirmation")
        self.assertEqual(ctx.db.calls, [])

    def test_missing_heap_or_rls_metadata_skips_before_queries(self):
        for field in ("table_access_method", "row_security", "force_row_security"):
            ctx = context()
            ctx.schema["jvd_ra"].pop(field)
            self.assertEqual(audit_races(ctx)["status"], "needs_confirmation")
            self.assertEqual(ctx.db.calls, [])

    def test_race_columns_alone_cannot_be_horse_identity(self):
        ctx = context()
        ctx.horse_identity_columns = RACE_KEY
        result = audit_horses(ctx)
        self.assertEqual(result["horse_identity_duplicates"]["reason"], "horse_identity_needs_at_least_one_nonrace_column")
        self.assertFalse(any(q == "se.horse_duplicates" for q, *_ in ctx.db.calls))

    def test_global_scan_requires_explicit_flag(self):
        ctx = context(global_summary=True)
        result = audit_races(ctx)
        self.assertEqual(result["global_summary"]["status"], "completed_explicit_full_table_scan")
        self.assertEqual({r["year"] for r in result["global_summary"]["year_counts"]}, {"2024", "2025"})
        self.assertEqual(result["global_summary"]["period"]["earliest_date"], "2024-01-05")

    def test_global_quality_includes_null_or_malformed_years_excluded_from_scope(self):
        ctx = context(ra=[fake_race("01"), fake_race("02", kaisai_nen=None), fake_race("03", kaisai_nen="FICTIONAL_PRIVATE_YEAR"), fake_race("04", keibajo_code=None)], global_summary=True)
        result = audit_races(ctx)
        self.assertEqual(result["summary"]["records"], 1)
        quality = result["global_summary"]["key_quality"]
        self.assertEqual(quality["null_key_records"], 2)
        self.assertEqual(quality["invalid_format_records"], 1)
        self.assertNotIn("FICTIONAL_PRIVATE_YEAR", json.dumps(result))

    def test_sql_only_uses_aggregates_and_local_table_scans(self):
        ctx = context(horse_identity=True, global_summary=True)
        audit_races(ctx)
        audit_horses(ctx)
        for query_id, sql, _, max_rows in ctx.db.calls:
            self.assertTrue(sql.strip().startswith(("SELECT", "WITH")), query_id)
            self.assertIn("FROM ONLY", sql, query_id)
            self.assertGreater(max_rows, 0)
            self.assertNotIn("LIMIT 1", sql)
            self.assertNotIn("INSERT", sql.upper())
            self.assertNotIn("DELETE", sql.upper())
            self.assertNotIn("UPDATE", sql.upper())
            self.assertNotIn("DROP", sql.upper())

    def test_casts_are_guarded_and_use_bounded_syntax(self):
        sql = valid_key_format()
        self.assertTrue(sql.startswith("CASE WHEN"))
        self.assertIn("^[0-9]{4}$", sql)
        self.assertIn("mod(", sql)
        self.assertIn("ELSE FALSE END", sql)

    def test_zero_padding_is_preserved_in_key_comparison(self):
        ctx = context(ra=[fake_race("01")], se=[fake_race("1")])
        counts = audit_horses(ctx)["race_set_comparison"]["counts"]
        self.assertEqual(counts["races_only_in_ra"], 1)
        self.assertEqual(counts["races_only_in_se"], 1)
        sql = next(call[1] for call in ctx.db.calls if call[0] == "se.race_sets")
        self.assertNotIn("lpad", sql)

    def test_anomalous_histogram_is_bounded(self):
        ctx = context(ra=[fake_race("01")], se=[fake_race("01") for _ in range(1001)])
        self.assertEqual(audit_horses(ctx)["horse_records_per_race_distribution"], [{"records_per_race": "1001+", "races": 1}])

    def test_bad_aggregate_bucket_is_rejected_without_echoing_value(self):
        ctx = context()
        original = ctx.db.aggregate
        def incorrect(query_id, *args, **kwargs):
            if query_id == "ra.venues":
                return [{"venue": "FICTIONAL_PRIVATE_VALUE", "records": 1}]
            return original(query_id, *args, **kwargs)
        ctx.db.aggregate = incorrect
        with self.assertRaisesRegex(ValueError, "unexpected_aggregate_bucket") as caught:
            audit_races(ctx)
        self.assertNotIn("FICTIONAL_PRIVATE_VALUE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
