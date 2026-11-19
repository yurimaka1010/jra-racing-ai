"""Fictitious HR records and aggregate mocks; no payload decoding or real SQL."""
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr
from datetime import date
import io
import json
import re
import unittest
from unittest.mock import patch

from jra_racing_ai import cli
from jra_racing_ai.core import AuditContext, RACE_KEY, SafeAuditError, Scope
from jra_racing_ai.audit_payouts import audit_payouts
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.generate_report import report_markdown, schema_markdown
from jra_racing_ai.payout_policy import PAYOUT_CASES, describe_case


def row(number="01", status="1", **changes):
    result = dict(zip(RACE_KEY, ("2025", "0105", "01", "01", "01", number)))
    result.update(data_kubun=status, fake_case="normal_win_hit",
                  fake_payload="FICTIONAL_PAYOUT_PRIVATE_PAYLOAD", fake_horse="FICTIONAL_HORSE_ID",
                  fake_amount=876543210)
    result.update(changes)
    return result


def race_key(record):
    return tuple(None if record.get(name) is None else str(record[name]).strip() for name in RACE_KEY)


def valid(record):
    key = race_key(record)
    if any(value is None or not re.fullmatch(r"[0-9]+", value) for value in key):
        return False
    if len(key[0]) != 4 or len(key[1]) != 4 or any(not 1 <= len(value) <= 2 for value in key[2:]):
        return False
    try:
        date(int(key[0]), int(key[1][:2]), int(key[1][2:]))
    except ValueError:
        return False
    return all(1 <= int(value) <= 99 for value in key[2:])


def category(record):
    value = record.get("data_kubun")
    if value is None:
        return "null"
    value = str(value).strip()
    return value if re.fullmatch(r"[0-9]", value) else "other"


def table(name):
    names = (*RACE_KEY, "data_kubun", "tansho_payload", "fukusho_payload", "henkan_flag",
             "tokubarai_flag", "fuseiritsu_flag", "toroku_fake_count")
    return {"name": name, "kind": "table", "auditable": True, "table_access_method": "heap",
            "row_security": False, "force_row_security": False,
            "columns": [{"name": n, "data_type": "text", "type_is_builtin": True, "nullable": True} for n in names],
            "primary_key": list(RACE_KEY), "indexes": [{"name": "fake_index", "columns": list(RACE_KEY),
                "has_expressions": True, "expression": "FICTIONAL_PRIVATE_EXPRESSION"}]}


class PayoutMock:
    """An independent Python oracle for invented grouped counts, not a SQL engine."""
    def __init__(self, hr, ra, scope):
        self.hr, self.ra, self.scope = hr, ra, scope
        self.calls = []

    def scoped(self, records, all_years=False):
        return [r for r in records if all_years or (
            str(r.get("kaisai_nen", "")).strip() == str(self.scope.year)
            and str(r.get("keibajo_code", "")).strip() in self.scope.venues)]

    def aggregate(self, query_id, sql, params=(), max_rows=1000):
        validate_audit_sql(sql)
        # Ensure arithmetic % hasn't silently become a psycopg placeholder.
        if re.search(r"%(?!s)", sql):
            raise AssertionError("invalid_placeholder")
        self.calls.append((query_id, sql, params, max_rows))
        records = self.scoped(self.hr, ".global" in query_id)
        complete = [r for r in records if None not in race_key(r)]
        keys = Counter(race_key(r) for r in complete)
        if query_id == "hr.summary":
            result = [{"records": len(records), "null_key_records": len(records)-len(complete),
                       "empty_key_records": sum("" in race_key(r) for r in complete),
                       "invalid_format_records": sum(not valid(r) for r in complete),
                       "distinct_complete_races": len(keys), "duplicate_key_groups": sum(n > 1 for n in keys.values()),
                       "duplicate_excess_records": sum(n - 1 for n in keys.values())}]
        elif query_id == "hr.key_quality":
            result = [{"field": n, "null_records": sum(r.get(n) is None for r in records),
                       "empty_records": sum(r.get(n) is not None and str(r[n]).strip() == "" for r in records)} for n in RACE_KEY]
        elif query_id == "hr.venues":
            result = [{"venue": n, "records": count} for n, count in Counter(str(r["keibajo_code"]).strip() for r in records).items()]
        elif query_id == "hr.records_per_race":
            bands = Counter(str(n) if n <= 10 else "11-100" if n <= 100 else "101+" for n in keys.values())
            result = [{"records_per_race": n, "races": count} for n, count in bands.items()]
        elif query_id == "hr.categories":
            result = [{"category": c, "records": sum(category(r) == c for r in records),
                       "distinct_races": len({race_key(r) for r in complete if category(r) == c})}
                      for c in sorted({category(r) for r in records})]
        elif query_id == "hr.race_sets":
            ra_records = self.scoped(self.ra)
            ra_keys = {race_key(r) for r in ra_records if None not in race_key(r)}
            hr_keys = set(keys)
            ra_valid = {race_key(r) for r in ra_records if valid(r)}
            hr_valid = {race_key(r) for r in records if valid(r)}
            result = [{"ra_complete_distinct_races": len(ra_keys), "hr_complete_distinct_races": len(hr_keys),
                       "shared_complete_races": len(ra_keys & hr_keys), "races_only_in_ra": len(ra_keys - hr_keys),
                       "races_only_in_hr": len(hr_keys - ra_keys),
                       "valid_format_races_only_in_ra": len(ra_valid - hr_valid),
                       "valid_format_races_only_in_hr": len(hr_valid - ra_valid)}]
        elif query_id == "hr.coverage":
            result = [self.coverage(records)]
        elif query_id == "hr.global_years":
            years = Counter("null" if r.get("kaisai_nen") is None else
                            str(r["kaisai_nen"]).strip() if re.fullmatch(r"[0-9]{4}", str(r["kaisai_nen"]).strip()) and str(r["kaisai_nen"]).strip() != "0000" else "other"
                            for r in records)
            result = [{"year": y, "records": n} for y, n in years.items()]
        elif query_id == "hr.global_key_quality":
            result = [{"records": len(records), "null_key_records": len(records)-len(complete),
                       "empty_key_records": sum("" in race_key(r) for r in complete),
                       "invalid_format_records": sum(not valid(r) for r in complete),
                       **{n+"_null_records": sum(r.get(n) is None for r in records) for n in RACE_KEY},
                       **{n+"_empty_records": sum(r.get(n) is not None and str(r[n]).strip() == "" for r in records) for n in RACE_KEY}}]
        elif query_id == "hr.global.period":
            dates = []
            for r in records:
                try:
                    dates.append(date(int(r["kaisai_nen"]), int(r["kaisai_tsukihi"][:2]), int(r["kaisai_tsukihi"][2:])).isoformat())
                except (TypeError, ValueError):
                    pass
            result = [{"earliest_date": min(dates, default=None), "latest_date": max(dates, default=None),
                       "valid_date_records": len(dates), "invalid_or_null_date_records": len(records)-len(dates)}]
        else:
            raise AssertionError("unexpected_query_id")
        if len(result) > max_rows:
            raise SafeAuditError("result_limit")
        return result

    def coverage(self, records):
        ra = defaultdict(set)
        hr = defaultdict(set)
        for r in self.scoped(self.ra):
            if valid(r):
                ra[race_key(r)].add(category(r))
        for r in records:
            if valid(r):
                hr[race_key(r)].add(category(r))
        executed = {k for k, s in ra.items() if s == {"7"}}
        cancelled = {k for k, s in ra.items() if s == {"9"}}
        final = {k for k, s in hr.items() if s <= {"1", "2"}}
        review = {k for k, s in hr.items() if (not s <= {"0", "1", "2", "9"}
                  or ("0" in s and len(s) > 1) or ("9" in s and bool(s & {"1", "2"})))}
        deleted = {k for k, s in hr.items() if s == {"0"}}
        return {"executed_races": len(executed), "cancelled_races": len(cancelled),
                "ra_conflicting_status_races": sum(len(s) > 1 for s in ra.values()),
                "hr_review_required_races": len(review),
                "executed_races_with_payout_record": len(executed & final),
                "executed_races_without_payout_record": len(executed - final),
                "executed_races_without_any_hr_record": len(executed - hr.keys()),
                "executed_races_with_review_required_hr": len(executed & review),
                "executed_races_with_only_deleted_hr": len(executed & deleted),
                "cancelled_races_with_any_hr_record": len(cancelled & hr.keys()),
                "cancelled_races_with_hr_cancelled_status": sum("9" in hr.get(k, ()) for k in cancelled),
                "cancelled_races_with_hr_final_status": len(cancelled & final),
                "cancelled_hr_nondeleted_known_status_records": sum(valid(r) and race_key(r) in cancelled and category(r) in {"1", "2", "9"} for r in records),
                "cancelled_hr_cancelled_status_records": sum(valid(r) and race_key(r) in cancelled and category(r) == "9" for r in records)}


def context(hr=None, ra=None, mapped=True, global_summary=False):
    scope = Scope(2025, global_summary=global_summary)
    hr = [row("01"), row("02", "2"), row("03", "9")] if hr is None else hr
    ra = [row("01", "7"), row("02", "7"), row("03", "9")] if ra is None else ra
    db = PayoutMock(hr, ra, scope)
    return AuditContext(db, scope, {n: table(n) for n in ("jvd_hr", "jvd_ra")},
                        {"jvd_hr": "data_kubun", "jvd_ra": "data_kubun"} if mapped else {})


class PayoutAuditTests(unittest.TestCase):
    def test_normal_scope_and_category_counts(self):
        ctx = context()
        result = audit_payouts(ctx)
        self.assertEqual(result["summary"]["records"], 3)
        self.assertEqual(result["year_counts"], [{"year": 2025, "records": 3}])
        self.assertEqual(result["venue_counts"], [{"venue": "01", "records": 3}])
        self.assertEqual({r["category"]: r["records"] for r in result["category_counts"]}, {"1": 1, "2": 1, "9": 1})
        counts = result["settlement_record_coverage"]["counts"]
        self.assertEqual(counts["executed_races_with_payout_record"], 2)
        self.assertEqual(counts["cancelled_hr_cancelled_status_records"], 1)

    def test_same_counts_can_have_two_set_differences(self):
        result = audit_payouts(context(hr=[row("01"), row("04")], ra=[row("01", "7"), row("02", "7")]))
        counts = result["race_set_comparison"]["counts"]
        self.assertEqual(counts["races_only_in_ra"], 1)
        self.assertEqual(counts["races_only_in_hr"], 1)
        self.assertEqual(counts["ra_complete_distinct_races"], counts["hr_complete_distinct_races"])

    def test_missing_empty_invalid_and_duplicate_keys_are_separate(self):
        records = [row("01"), row("01"), row(None), row(""), row("02", kaisai_tsukihi="0231"),
                   row("03", kaisai_nen="2024"), row("04", keibajo_code="11")]
        result = audit_payouts(context(hr=records))
        self.assertEqual(result["summary"]["records"], 5)
        self.assertEqual(result["summary"]["null_key_records"], 1)
        self.assertEqual(result["summary"]["empty_key_records"], 1)
        self.assertEqual(result["summary"]["invalid_format_records"], 2)
        self.assertEqual(result["summary"]["duplicate_excess_records"], 1)

    def test_padding_not_silently_normalized(self):
        result = audit_payouts(context(hr=[row("1")], ra=[row("01", "7")]))
        self.assertEqual(result["race_set_comparison"]["counts"]["races_only_in_ra"], 1)

    def test_versions_are_counted_without_assuming_record_grain(self):
        result = audit_payouts(context(hr=[row("01", "1"), row("01", "2")], ra=[row("01", "7")]))
        self.assertEqual(result["summary"]["duplicate_key_groups"], 1)
        self.assertEqual(result["records_per_race_distribution"], [{"records_per_race": "2", "races": 1}])
        self.assertEqual(result["settlement_record_coverage"]["counts"]["executed_races_with_payout_record"], 1)
        self.assertIn("needs_confirmation", result["record_grain_status"])

    def test_deleted_cancelled_and_unknown_hr_are_not_final_payouts(self):
        ctx = context(hr=[row("01", "0"), row("02", "9"), row("03", "5"), row("04", None),
                          row("05", "FICTIONAL_BAD_STATUS")], ra=[row(f"{n:02}", "7") for n in range(1, 7)])
        result = audit_payouts(ctx)
        counts = result["settlement_record_coverage"]["counts"]
        self.assertEqual(counts["executed_races_with_payout_record"], 0)
        self.assertEqual(counts["executed_races_without_payout_record"], 6)
        self.assertEqual(counts["executed_races_without_any_hr_record"], 1)
        self.assertEqual(counts["executed_races_with_only_deleted_hr"], 1)
        self.assertEqual(counts["executed_races_with_review_required_hr"], 3)
        self.assertNotIn("FICTIONAL_BAD_STATUS", json.dumps(result))

    def test_conflicting_ra_or_hr_states_require_review(self):
        result = audit_payouts(context(hr=[row("01"), row("01", "0"), row("02"), row("02", "9"), row("03")],
                                      ra=[row("01", "7"), row("02", "7"), row("03", "7"), row("03", "9")]))
        counts = result["settlement_record_coverage"]["counts"]
        self.assertEqual(counts["ra_conflicting_status_races"], 1)
        self.assertEqual(counts["hr_review_required_races"], 2)
        self.assertEqual(counts["executed_races"], 2)
        self.assertEqual(counts["executed_races_with_payout_record"], 0)

    def test_ra_duplicates_do_not_multiply_cancelled_hr_record_counts(self):
        result = audit_payouts(context(hr=[row("01", "9"), row("01", "9")], ra=[row("01", "9"), row("01", "9")]))
        counts = result["settlement_record_coverage"]["counts"]
        self.assertEqual(counts["cancelled_races"], 1)
        self.assertEqual(counts["cancelled_hr_cancelled_status_records"], 2)

    def test_mapping_not_guessed_from_names(self):
        ctx = context(mapped=False)
        result = audit_payouts(ctx)
        self.assertEqual(result["category_counts"]["status"], "needs_confirmation")
        self.assertEqual(result["settlement_record_coverage"]["status"], "needs_confirmation")
        self.assertFalse(any(q in ("hr.categories", "hr.coverage") for q, *_ in ctx.db.calls))

    def test_only_hr_mapping_is_not_enough_for_execution_coverage(self):
        ctx = context()
        ctx.status_columns.pop("jvd_ra")
        self.assertEqual(audit_payouts(ctx)["settlement_record_coverage"]["status"], "needs_confirmation")
        self.assertTrue(any(q == "hr.categories" for q, *_ in ctx.db.calls))

    def test_missing_table_and_missing_key_do_not_query(self):
        for missing in ("table", "key"):
            ctx = context()
            if missing == "table":
                ctx.schema.pop("jvd_hr")
            else:
                ctx.schema["jvd_hr"]["columns"] = ctx.schema["jvd_hr"]["columns"][1:]
            result = audit_payouts(ctx)
            self.assertEqual(result["status"], "needs_confirmation")
            self.assertEqual(ctx.db.calls, [])

    def test_unsafe_table_or_unconfirmed_type_do_not_query(self):
        for changes in ({"kind": "foreign_table"}, {"row_security": True},
                        {"kind": "partitioned_table"}, {"table_access_method": "custom"}, {"auditable": False}):
            ctx = context()
            ctx.schema["jvd_hr"].update(changes)
            self.assertEqual(audit_payouts(ctx)["status"], "needs_confirmation")
            self.assertEqual(ctx.db.calls, [])
        for field in ("kaisai_nen", "data_kubun"):
            ctx = context()
            col = next(c for c in ctx.schema["jvd_hr"]["columns"] if c["name"] == field)
            col.update(data_type="custom", type_is_builtin=False)
            self.assertEqual(audit_payouts(ctx)["status"], "needs_confirmation")
            self.assertEqual(ctx.db.calls, [])

    def test_wrong_or_injection_like_status_column_stops_without_echo(self):
        for table_name in ("jvd_hr", "jvd_ra"):
            ctx = context()
            ctx.status_columns[table_name] = "FAKE_PRIVATE_COLUMN\"; DROP TABLE x"
            with self.assertRaises(SafeAuditError) as caught:
                audit_payouts(ctx)
            self.assertEqual(ctx.db.calls, [])
            self.assertNotIn("FAKE_PRIVATE_COLUMN", str(caught.exception))

    def test_unusable_ra_still_allows_hr_structure_and_counts(self):
        ctx = context()
        ctx.schema.pop("jvd_ra")
        result = audit_payouts(ctx)
        self.assertEqual(result["summary"]["records"], 3)
        self.assertEqual(result["race_set_comparison"]["reason"], "table_not_found")
        self.assertEqual(result["settlement_record_coverage"]["reason"], "table_not_found")

    def test_ra_status_custom_type_is_not_evaluated(self):
        ctx = context()
        col = next(c for c in ctx.schema["jvd_ra"]["columns"] if c["name"] == "data_kubun")
        col.update(data_type="custom", type_is_builtin=False)
        result = audit_payouts(ctx)
        self.assertEqual(result["settlement_record_coverage"]["reason"], "ra_status_column_type_requires_review")
        self.assertFalse(any(q == "hr.coverage" for q, *_ in ctx.db.calls))

    def test_empty_scope_is_zero_counts_not_missing_table(self):
        result = audit_payouts(context(hr=[], ra=[]))
        self.assertEqual(result["summary"]["records"], 0)
        self.assertEqual(result["category_counts"], [])
        self.assertTrue(result["structure"]["exists"])

    def test_global_scan_is_explicit_and_includes_excluded_invalid_keys(self):
        records = [row("01"), row("02", kaisai_nen=None), row("03", keibajo_code=None), row("04", kaisai_nen="2024")]
        ctx = context(hr=records)
        self.assertEqual(audit_payouts(ctx)["global_summary"]["status"], "not_requested")
        self.assertFalse(any("global" in q for q, *_ in ctx.db.calls))
        ctx = context(hr=records, global_summary=True)
        global_result = audit_payouts(ctx)["global_summary"]
        self.assertEqual(global_result["key_quality"]["records"], 4)
        self.assertEqual(global_result["key_quality"]["null_key_records"], 2)

    def test_no_raw_values_and_no_payload_columns_in_queries(self):
        ctx = context(hr=[row("01", fake_case=c) for c in PAYOUT_CASES])
        result = audit_payouts(ctx)
        encoded = json.dumps(result)
        for secret in ("FICTIONAL_PAYOUT_PRIVATE_PAYLOAD", "FICTIONAL_HORSE_ID", "876543210", "FICTIONAL_PRIVATE_EXPRESSION"):
            self.assertNotIn(secret, encoded)
        for q, sql, params, limit in ctx.db.calls:
            self.assertIn("FROM ONLY", sql)
            self.assertGreater(limit, 0)
            self.assertNotIn("tansho_payload", sql)
            self.assertNotIn("henkan_flag", sql)
            self.assertNotIn("fake_amount", sql)
            self.assertEqual(len(params), sql.count("%s"))
        self.assertEqual(result["payload_decoding"], "not_implemented")

    def test_future_query_extra_fields_cannot_leak(self):
        ctx = context()
        original = ctx.db.aggregate
        def extra(*args, **kwargs):
            return [{**r, "raw_key": "FICTIONAL_PRIVATE_ROW"} for r in original(*args, **kwargs)]
        ctx.db.aggregate = extra
        self.assertNotIn("FICTIONAL_PRIVATE_ROW", json.dumps(audit_payouts(ctx)))

    def test_invalid_count_bucket_and_result_shape_are_safely_rejected(self):
        for replacement in ([], [{"records": "FICTIONAL_PRIVATE_ROW"}], [{"records": -1}]):
            ctx = context()
            original = ctx.db.aggregate
            ctx.db.aggregate = lambda q, *a, **kw: replacement if q == "hr.summary" else original(q, *a, **kw)
            with self.assertRaises(SafeAuditError) as caught:
                audit_payouts(ctx)
            self.assertNotIn("FICTIONAL_PRIVATE_ROW", str(caught.exception))

    def test_permission_timeout_errors_propagate_without_partial_report(self):
        for code in ("permission_denied", "query_timeout", "result_limit"):
            ctx = context()
            ctx.db.aggregate = lambda *a, **kw: (_ for _ in ()).throw(SafeAuditError(code))
            with self.assertRaises(SafeAuditError) as caught:
                audit_payouts(ctx)
            self.assertEqual(caught.exception.code, code)

    def test_special_case_payloads_and_multiple_targets_are_never_decoded(self):
        for case in PAYOUT_CASES:
            with self.subTest(case=case):
                fixture = row("01", fake_case=case,
                              tansho_payload=[("01", 765432101), ("02", 765432102), ("03", 765432103)],
                              fukusho_payload=[("01", 765432111), ("02", 765432112), ("03", 765432113),
                                               ("04", 765432114), ("05", 765432115)],
                              henkan_flag="FICTIONAL_REFUND_FLAG", tokubarai_flag="FICTIONAL_SPECIAL_FLAG",
                              fuseiritsu_flag="FICTIONAL_NOT_ESTABLISHED_FLAG")
                ctx = context(hr=[fixture])
                report = audit_payouts(ctx)
                self.assertEqual(report["special_case_policy"]["record_classification"], "not_implemented")
                self.assertEqual(report["summary"]["records"], 1)
                self.assertNotIn("7654321", json.dumps(report))
                self.assertNotIn("FICTIONAL_REFUND_FLAG", json.dumps(report))

    def test_initial_or_absent_payload_is_not_treated_as_zero_payout(self):
        for payload in (None, "", "000000", "FICTIONAL_MALFORMED_PAYLOAD"):
            report = audit_payouts(context(hr=[row("01", tansho_payload=payload)]))
            self.assertEqual(report["payload_decoding"], "not_implemented")
            self.assertEqual(report["profit_calculation"], "not_implemented")
            self.assertNotIn("FICTIONAL_MALFORMED_PAYLOAD", json.dumps(report))

    def test_histogram_for_large_groups_remains_bounded(self):
        result = audit_payouts(context(hr=[row("01") for _ in range(101)]))
        self.assertEqual(result["records_per_race_distribution"], [{"records_per_race": "101+", "races": 1}])

    def test_bad_histogram_and_venue_buckets_do_not_echo_raw_values(self):
        for target, replacement in (
            ("hr.records_per_race", [{"records_per_race": "FICTIONAL_PRIVATE_BUCKET", "races": 1}]),
            ("hr.venues", [{"venue": "FICTIONAL_PRIVATE_BUCKET", "records": 1}]),
        ):
            ctx = context()
            original = ctx.db.aggregate
            ctx.db.aggregate = lambda q, *a, **kw: replacement if q == target else original(q, *a, **kw)
            with self.assertRaises((SafeAuditError, ValueError)) as caught:
                audit_payouts(ctx)
            self.assertNotIn("FICTIONAL_PRIVATE_BUCKET", str(caught.exception))


class PayoutCasePolicyTests(unittest.TestCase):
    def test_normal_and_dead_heat_cases_use_official_target_sets(self):
        for case in ("normal_win_hit", "dead_heat_win_hit", "normal_place_hit", "dead_heat_place_hit"):
            policy = describe_case(case)
            self.assertTrue(policy["standard_hit_label_rule"].startswith("official_"))
            self.assertEqual(policy["source_mapping_status"], "needs_confirmation")
            self.assertEqual(policy["profit_calculation"], "not_implemented")

    def test_refund_special_not_established_cancelled_are_separate(self):
        kinds = {describe_case(c)["ledger_kind"] for c in ("refund", "special_payment", "not_established", "race_cancelled")}
        self.assertEqual(len(kinds), 4)
        for case in ("refund", "special_payment", "not_established", "race_cancelled"):
            self.assertEqual(describe_case(case)["standard_hit_label_rule"], "no_standard_hit_label")

    def test_not_offered_and_missing_or_initial_are_not_zero_loss_labels(self):
        self.assertEqual(describe_case("not_offered")["score_rule"], "no_purchase_allowed")
        for case in ("missing", "initial_value", "invalid", "unmapped"):
            self.assertEqual(describe_case(case)["standard_hit_label_rule"], "unknown_never_zero")

    def test_unknown_case_rejected_without_exposing_record(self):
        for value in ("FICTIONAL_PRIVATE_CASE", {"amount": 876543210}):
            with self.assertRaises(SafeAuditError) as caught:
                describe_case(value)
            self.assertNotIn("876543210", str(caught.exception))
            self.assertNotIn("FICTIONAL_PRIVATE_CASE", str(caught.exception))


class PayoutCLITests(unittest.TestCase):
    def test_new_arguments_and_old_defaults_are_compatible(self):
        args = cli.parser().parse_args(["audit", "--year", "2025", "--sections", "payouts", "--payout-status-column", "data_kubun"])
        self.assertEqual(args.sections, ["payouts"])
        self.assertEqual(args.payout_status_column, "data_kubun")
        old = cli.parser().parse_args(["audit", "--year", "2025"])
        self.assertEqual(old.sections, ["races", "horses", "odds"])
        self.assertIsNone(old.payout_status_column)

    def test_payouts_without_execute_flag_never_connects(self):
        with patch("jra_racing_ai.cli.read_only_session") as session, redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["audit", "--year", "2025", "--sections", "payouts"]), 2)
            session.assert_not_called()

    def test_full_cli_uses_only_fake_session_and_integrates_existing_report_format(self):
        ctx = context()
        @contextmanager
        def fake_session(*args, **kwargs):
            yield ctx.db
        inventory = {"status": "synthetic", "tables": list(ctx.schema.values())}
        with patch("jra_racing_ai.cli.validate_output_dir"), \
                patch("jra_racing_ai.cli.ConnectionConfig.from_env"), \
                patch("jra_racing_ai.cli.importlib.util.find_spec", return_value=object()), \
                patch("jra_racing_ai.cli.getpass.getpass", return_value="FICTIONAL_SECRET"), \
                patch.dict("os.environ", {}, clear=True), \
                patch("jra_racing_ai.cli.read_only_session", side_effect=fake_session), \
                patch("jra_racing_ai.inspect_schema.inspect_schema", return_value=inventory), \
                patch("jra_racing_ai.cli.write_report", return_value={}) as writer:
            cli.run(cli.parser().parse_args(["audit", "--execute-local", "--year", "2025", "--sections", "payouts",
                                           "--race-status-column", "data_kubun", "--payout-status-column", "data_kubun"]))
            report = writer.call_args.args[0]
            self.assertEqual(report["format_version"], 1)
            self.assertEqual(set(report["audits"]), {"payouts"})
            self.assertIn("payouts", report_markdown(report))
            self.assertIn("public.jvd_hr", schema_markdown(report["schema"]))
            self.assertEqual(report["ml_readiness"]["tables"][0]["feature_status"], "prohibited_feature")
            self.assertNotIn("FICTIONAL_SECRET", json.dumps(report))

    def test_payout_import_has_no_connection_or_output_side_effect(self):
        import importlib
        from pathlib import Path
        with patch("jra_racing_ai.db_connection.read_only_session") as session, patch.object(Path, "mkdir") as mkdir:
            importlib.reload(importlib.import_module("jra_racing_ai.audit_payouts"))
            session.assert_not_called()
            mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
