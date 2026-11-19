"""Invented H1/HR data only; never uses the user's observed race."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai import cli, h1_local, payout_reconciliation_cli as command
from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.h1_reconciliation import (
    H1Config, H1Review, LocalH1Record, check_h1_budget, parse_h1_config, reconcile_h1_payouts,
)
from jra_racing_ai.payout_reconciliation import (
    RaceKey, ReconciliationError, ReconciliationLimits, RevisionReview, reconcile_payouts,
)
from jra_racing_ai.reconciliation_local import ExecutionBudget, fetch_local_hr
from jra_racing_ai import reconciliation_local as local
from test_payout_reconciliation import document, layout, observation, race, record, reference_case, target
from test_reconciliation_local import FakeDB, INPUT, SECRET, catalog, row_value


LIMITS = ReconciliationLimits(max_run_seconds=45)


def config(**changes):
    values = dict(h1_columns=("fake_h1_status", "fake_h1_win_sale", "fake_h1_place_sale", "fake_h1_terms"),
                  hr_columns=("fake_win_not_established", "fake_win_special", "fake_place_not_established", "fake_place_special"),
                  column_meanings_reviewed_locally=True, reviews=(H1Review(race(), RevisionReview.SNAPSHOT_REVIEWED),))
    return H1Config(**(values | changes))


def hr(**changes):
    return record(**({"flags": ("0",) * 4} | changes))


def h1(**changes):
    return LocalH1Record(**(dict(race=race(), category="5", tansho_sale="7", fukusho_sale="7", place_paid_positions="3") | changes))


def compare(doc=None, hrs=None, h1s=None, cfg=None, **changes):
    return reconcile_h1_payouts(doc or document(reference_case(), reference_case("fukusho"), flag_rules=()),
                                 (hr(),) if hrs is None else hrs, (h1(),) if h1s is None else h1s,
                                 cfg or config(), **({"limits": LIMITS} | changes))


def config_json():
    from jra_racing_ai.h1_reconciliation import H1_ROLES, HR_ROLES
    cfg = config()
    return {"format_version": 1, "h1_columns": dict(zip(H1_ROLES, cfg.h1_columns)),
            "hr_columns": dict(zip(HR_ROLES, cfg.hr_columns)), "column_meanings_reviewed_locally": True,
            "h1_revision_reviews": [{"race_key": dict(zip(RACE_KEY, race().parts)),
                                     "revision_review": "snapshot_reviewed_locally"}]}


class H1ConfigTests(unittest.TestCase):
    def test_sidecar_round_trip_without_altering_reference_json(self):
        self.assertEqual(parse_h1_config(config_json()), config())

    def test_shipped_sidecar_is_unreviewed_and_synthetic(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "h1-config.synthetic.json"
        cfg = parse_h1_config(json.loads(path.read_text(encoding="utf-8")))
        self.assertFalse(cfg.column_meanings_reviewed_locally)
        self.assertTrue(all(r.revision is RevisionReview.UNCONFIRMED for r in cfg.reviews))

    def test_unknown_fields_and_coerced_review_values_are_rejected(self):
        for field, value in (("assume_confirmed", True), ("column_meanings_reviewed_locally", "true"), ("format_version", True)):
            data = config_json()
            data[field] = value
            with self.assertRaises(ReconciliationError):
                parse_h1_config(data)

    def test_duplicate_race_reviews_are_rejected(self):
        with self.assertRaises(ReconciliationError):
            config(reviews=config().reviews * 2)

    def test_unknown_race_review_is_rejected_before_queries(self):
        cfg = config(reviews=(H1Review(race(number="10"), RevisionReview.SNAPSHOT_REVIEWED),))
        with self.assertRaises(ReconciliationError):
            cfg.check_document(document())

    def test_sql_injection_reserved_and_duplicate_columns_are_rejected(self):
        for columns in (("x; DROP TABLE jvd_h1", "a", "b", "c"), ("kaisai_nen", "a", "b", "c"), ("a", "a", "b", "c")):
            with self.assertRaises(ReconciliationError):
                config(h1_columns=columns)
        for column in ("henkan_flag_tansho", "haraimodoshi_tansho_1b"):
            with self.assertRaises(ReconciliationError):
                config(hr_columns=(column,) + config().hr_columns[1:])

    def test_seven_query_budget_and_row_limit_are_bounded(self):
        check_h1_budget(LIMITS, 40)
        for limits, rows in ((ReconciliationLimits(), 40), (LIMITS, 201), (LIMITS, True), (LIMITS, 0)):
            with self.assertRaises(ReconciliationError):
                check_h1_budget(limits, rows)


class H1NormalContextTests(unittest.TestCase):
    def assert_blocked(self, reason, **changes):
        result = compare(**changes)
        self.assertEqual(dict(result.h1_evidence.reasons).get(reason), 1)
        for wager in result.wagers:
            self.assertEqual(wager.eligible_cases, 0)
            self.assertEqual(observation(wager).matched + observation(wager).mismatched, 0)
        return result

    def test_hr_two_and_h1_five_are_valid_without_numeric_equality(self):
        result = compare()
        self.assertEqual(result.h1_evidence.compatible_races, 1)
        self.assertEqual(dict(result.categories), {"2": 1})
        self.assertEqual(dict(result.h1_evidence.categories), {"5": 1})
        self.assertTrue(all(observation(w).matched == 1 for w in result.wagers))

    def test_hr_five_is_not_accepted_just_because_h1_is_five(self):
        self.assert_blocked("hr_category_mismatch_for_h1", hrs=(hr(category="5"),))

    def test_h1_two_is_not_accepted_just_because_hr_is_two(self):
        self.assert_blocked("h1_category_mismatch", h1s=(h1(category="2"),))

    def test_missing_h1_never_falls_back_to_legacy_flags(self):
        self.assert_blocked("missing_h1", h1s=())

    def test_multiple_identical_h1_records_still_block(self):
        self.assert_blocked("multiple_h1_records", h1s=(h1(), h1()))

    def test_multiple_h1_versions_are_not_selected_by_max_category(self):
        self.assert_blocked("multiple_h1_records", h1s=(h1(category="1"), h1()))

    def test_missing_and_multiple_hr_are_not_hidden_by_h1_presence(self):
        self.assert_blocked("missing_hr", hrs=())
        self.assert_blocked("multiple_hr_versions", hrs=(hr(), hr()))

    def test_h1_unknown_deleted_candidate_and_null_statuses_block(self):
        for value in ("0", "1", "9", " ", "55", 5, None):
            self.assert_blocked("h1_category_mismatch", h1s=(h1(category=value),))

    def test_both_sale_flags_must_be_exact_string_seven(self):
        for field in ("tansho_sale", "fukusho_sale"):
            for value in ("0", "1", " ", "07", "７", 7, None):
                self.assert_blocked("h1_sale_flag_unconfirmed", h1s=(h1(**{field: value}),))

    def test_all_four_hr_normal_flags_must_be_exact_string_zero(self):
        for index in range(4):
            for value in ("1", "7", " ", 0, None):
                values = list(hr().flags)
                values[index] = value
                self.assert_blocked("hr_normal_flags_unconfirmed", hrs=(hr(flags=tuple(values)),))

    def test_hr_refund_flags_must_both_be_zero(self):
        for field in ("refund_tansho", "refund_fukusho"):
            for value in ("1", 0, None):
                self.assert_blocked("hr_refund_flags_not_zero", hrs=(hr(**{field: value}),))

    def test_unknown_place_key_is_not_assumed_three(self):
        for value in ("0", "1", "4", " ", "03", 3, None):
            self.assert_blocked("h1_place_key_unconfirmed", h1s=(h1(place_paid_positions=value),))

    def test_two_paid_can_compare_reviewed_numeric_targets_without_zero_horse(self):
        targets = (target("01", 350), target("02", 140))
        doc = document(reference_case("fukusho", targets, place_paid_positions=2), flag_rules=())
        rows = (hr(fukusho=layout(("01", "000000350", "02"), ("02", "000000140", "01"), wager="fukusho")),)
        result = compare(doc=doc, hrs=rows, h1s=(h1(place_paid_positions="2"),))
        self.assertEqual(observation(result.wagers[1]).matched, 2)
        self.assertEqual(result.h1_evidence.compatible_races, 1)

    def test_place_key_must_agree_with_reference_terms(self):
        self.assert_blocked("h1_place_terms_mismatch", h1s=(h1(place_paid_positions="2"),))

    def test_zero_horse_meaning_is_not_released_by_h1_two_paid(self):
        doc = document(reference_case("fukusho", place_paid_positions=2), flag_rules=())
        row = hr(fukusho=layout(("01", "000000350", "02"), ("00", " " * 9, "  "), wager="fukusho"))
        result = compare(doc=doc, hrs=(row,), h1s=(h1(place_paid_positions="2"),))
        self.assertEqual(dict(result.wagers[1].reasons)["zero_horse_unconfirmed"], 1)
        self.assertEqual(observation(result.wagers[1]).matched, 0)

    def test_h1_column_meanings_require_review(self):
        self.assert_blocked("h1_mapping_unreviewed", cfg=config(column_meanings_reviewed_locally=False))

    def test_h1_revision_is_reviewed_independently_from_hr(self):
        self.assert_blocked("h1_revision_unreviewed", cfg=config(reviews=()))

    def test_h1_unconfirmed_correction_deletion_and_preliminary_are_separate(self):
        for revision, reason in ((RevisionReview.UNCONFIRMED, "h1_revision_unreviewed"),
                                 (RevisionReview.CORRECTION, "h1_correction_unreviewed"),
                                 (RevisionReview.DELETED, "h1_deletion_unreviewed"),
                                 (RevisionReview.PRELIMINARY, "h1_preliminary_unreviewed")):
            self.assert_blocked(reason, cfg=config(reviews=(H1Review(race(), revision),)))

    def test_hr_revision_review_is_still_required(self):
        doc = document(reference_case(revision_review=RevisionReview.UNCONFIRMED), flag_rules=())
        result = compare(doc=doc)
        self.assertEqual(dict(result.wagers[0].reasons)["revision_unreviewed"], 1)
        self.assertEqual(observation(result.wagers[0]).matched, 0)

    def test_every_component_of_the_race_key_is_used(self):
        for index, value in enumerate(("2024", "0107", "05", "02", "03", "10")):
            parts = list(race().parts)
            parts[index] = value
            with self.assertRaises(ReconciliationError) as caught:
                compare(h1s=(h1(race=RaceKey(tuple(parts))),))
            self.assertEqual(caught.exception.reason, "unexpected_h1_race")

    def test_partial_h1_coverage_keeps_hr_only_races_as_blocked(self):
        other = race(number="10")
        doc = document(reference_case(), reference_case(race=other), flag_rules=())
        cfg = config(reviews=config().reviews + (H1Review(other, RevisionReview.SNAPSHOT_REVIEWED),))
        result = compare(doc=doc, hrs=(hr(), hr(race=other)), h1s=(h1(),), cfg=cfg)
        self.assertEqual((result.h1_evidence.requested_races, result.h1_evidence.compatible_races, result.h1_evidence.blocked_races), (2, 1, 1))
        self.assertEqual((observation(result.wagers[0]).matched, observation(result.wagers[0]).uncomparable), (1, 1))

    def test_h1_row_overflow_and_oversized_fields_stop_comparisons(self):
        with self.assertRaises(ReconciliationError):
            compare(h1s=(h1(), h1()), max_h1_rows=1)
        self.assert_blocked("oversized_h1_field", h1s=(h1(oversized=True),))

    def test_bad_refund_bitmap_is_not_ignored_even_with_zero_flags(self):
        result = compare(hrs=(hr(refund_bitmap="1" + "0" * 27),))
        self.assertEqual(dict(result.wagers[0].reasons)["refund_context_unconfirmed"], 1)
        self.assertEqual(observation(result.wagers[0]).matched, 0)

    def test_amount_mismatch_and_all_hypotheses_are_still_reported(self):
        doc = document(reference_case(targets=(target(amount=360),)), flag_rules=())
        result = compare(doc=doc)
        self.assertEqual(len(result.wagers[0].observations), 3)
        self.assertEqual(observation(result.wagers[0]).mismatched, 1)
        self.assertFalse(result.to_safe_dict()["production_settlement_enabled"])

    def test_legacy_comparison_does_not_claim_h1_validation(self):
        result = reconcile_payouts(document(), (record(),)).to_safe_dict()
        self.assertNotIn("h1_validation", result)
        self.assertFalse(result["production_settlement_enabled"])

    def test_redacted_repr_and_aggregate_only_outputs(self):
        row = h1(category=SECRET)
        result = compare(h1s=(row,)).to_safe_dict()
        for value in (row, config(), config().reviews[0]):
            self.assertNotIn(SECRET, repr(value))
        text = json.dumps(result)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("0106", text)
        self.assertNotIn("horse_number", text)

    def test_pure_h1_comparison_has_no_file_database_or_print_side_effects(self):
        with patch.object(command, "read_only_session") as db, patch.object(Path, "open") as opened, \
                patch.object(Path, "mkdir") as mkdir, redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            compare()
            db.assert_not_called()
            opened.assert_not_called()
            mkdir.assert_not_called()
            self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")


class H1LocalAndCliTests(unittest.TestCase):
    def db(self, *, rows=None):
        doc, cfg = document(flag_rules=()), config()
        columns = catalog(doc) + [{"name": c, "type_name": "text", "type_is_builtin": True} for c in cfg.hr_columns]
        db = FakeDB(doc=doc, rows=[row_value(hr())], columns=columns)
        db.responses.update({
            "reconciliation.h1_table": [{"kind": "r", "method": "heap", "row_security": False, "force_row_security": False}],
            "reconciliation.h1_columns": [{"name": c, "type_name": "varchar", "type_is_builtin": True}
                                           for c in (*RACE_KEY, *cfg.h1_columns)],
            "reconciliation.h1_rows": [dict(zip(RACE_KEY, race().parts)) | {
                "h1_category": "5", "h1_win_sale": "7", "h1_place_sale": "7", "h1_paid_positions": "3",
                "oversized_fields": False, "unselected_secret": SECRET}] if rows is None else rows,
        })
        return db

    def fetch(self, db):
        return h1_local.fetch_local_h1(db, document(flag_rules=()), config(), LIMITS, ExecutionBudget(LIMITS, clock=lambda: 0))

    def args(self, mode="--check-input-only", extra=()):
        return cli.parser().parse_args(["reconcile-payouts", mode, "--confirm-local-unsynced", "--reference", str(INPUT),
                                         "--h1-config", str(INPUT.with_name("h1-config.json")), "--max-run-seconds", "45", *extra])

    def test_h1_reads_use_bound_keys_limit_and_no_status_filter(self):
        db = self.db()
        result = self.fetch(db)
        self.assertEqual(len(db.calls), 3)
        _, sql, params, max_rows = db.calls[-1]
        self.assertIn('FROM ONLY "public"."jvd_h1"', sql)
        self.assertEqual(params, race().parts + (41,))
        self.assertEqual(max_rows, 40)
        self.assertNotIn("::text", sql)
        self.assertNotIn("btrim", sql)
        self.assertNotIn("2025", sql)
        self.assertNotIn("= '5'", sql)
        self.assertEqual(result, (h1(),))

    def test_hr_and_h1_share_one_execution_budget_with_six_additional_queries(self):
        db, doc, cfg = self.db(), document(flag_rules=()), config()
        budget = ExecutionBudget(LIMITS, clock=lambda: 0)
        hrs = fetch_local_hr(db, doc, LIMITS, budget, flag_columns=cfg.hr_columns)
        h1s = h1_local.fetch_local_h1(db, doc, cfg, LIMITS, budget)
        self.assertEqual(len(db.calls), 6)
        self.assertEqual(len(hrs[0].flags), 4)
        self.assertEqual(compare(doc=doc, hrs=hrs, h1s=h1s).h1_evidence.compatible_races, 1)

    def test_h1_table_missing_or_not_plain_heap_stops(self):
        for table in ([], [{"kind": "f"}], [{"kind": "r", "method": "heap", "row_security": True}]):
            db = self.db()
            db.responses["reconciliation.h1_table"] = table
            with self.assertRaises(ReconciliationError) as caught:
                self.fetch(db)
            self.assertEqual(caught.exception.reason, "h1_table_unavailable")
            self.assertEqual(len(db.calls), 1)

    def test_h1_missing_custom_or_numeric_columns_stop_before_records(self):
        for changed in ([], [{"name": c, "type_name": "integer", "type_is_builtin": True}
                             for c in (*RACE_KEY, *config().h1_columns)]):
            db = self.db()
            db.responses["reconciliation.h1_columns"] = changed
            with self.assertRaises(ReconciliationError) as caught:
                self.fetch(db)
            self.assertEqual(caught.exception.reason, "h1_column_mapping_unconfirmed")
            self.assertEqual(len(db.calls), 2)

    def test_h1_row_limit_and_unexpected_key_are_rejected(self):
        db = self.db()
        db.responses["reconciliation.h1_rows"] *= 41
        with self.assertRaises(ReconciliationError) as caught:
            self.fetch(db)
        self.assertEqual(caught.exception.reason, "row_limit")
        db = self.db()
        db.responses["reconciliation.h1_rows"][0]["kaisai_kai"] = "02"
        with self.assertRaises(ReconciliationError) as caught:
            self.fetch(db)
        self.assertEqual(caught.exception.reason, "unexpected_h1_race")

    def test_h1_deadline_blocks_query_start(self):
        clock = iter((0, 41))
        db = self.db()
        with self.assertRaises(ReconciliationError):
            h1_local.fetch_local_h1(db, document(), config(), LIMITS, ExecutionBudget(LIMITS, clock=lambda: next(clock)))
        self.assertEqual(db.calls, [])

    def test_sidecar_uses_existing_local_json_guard(self):
        with patch.object(h1_local, "_read_json_file", return_value=config()) as reader:
            self.assertEqual(h1_local.read_h1_config_file(INPUT), config())
            self.assertIs(reader.call_args.args[1], parse_h1_config)

    def test_input_only_h1_mode_never_imports_driver_connects_or_saves(self):
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=document()), \
                patch.object(h1_local, "read_h1_config_file", return_value=config()), \
                patch.object(command.importlib.util, "find_spec") as driver, patch.object(command.getpass, "getpass") as password, \
                patch.object(command, "read_only_session") as session, patch.object(command, "write_reconciliation_summary") as writer:
            result = command.execute(self.args())
            self.assertEqual(result["h1_validation"]["reviewed_snapshots"], 1)
            self.assertFalse(result["h1_validation"]["database_accessed"])
            driver.assert_not_called()
            password.assert_not_called()
            session.assert_not_called()
            writer.assert_not_called()

    def test_insufficient_h1_time_budget_stops_before_secret_or_connection(self):
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=document()), \
                patch.object(h1_local, "read_h1_config_file") as reader, patch.object(command, "read_only_session") as session:
            with self.assertRaises(ReconciliationError) as caught:
                command.execute(self.args(extra=("--max-run-seconds", "30")))
            self.assertEqual(caught.exception.reason, "h1_time_budget")
            reader.assert_not_called()
            session.assert_not_called()

    def test_user_execution_fetches_both_in_same_readonly_session(self):
        doc, cfg, db = replace(document(flag_rules=()), source="jra_official_manual"), config(), self.db()
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=doc), \
                patch.object(h1_local, "read_h1_config_file", return_value=cfg), \
                patch.object(command.importlib.util, "find_spec", return_value=object()), \
                patch.dict(os.environ, {"PGPASSWORD": SECRET}, clear=True), \
                patch.object(command, "read_only_session") as session, \
                patch.object(command, "write_reconciliation_summary", side_effect=lambda s, *a, **k: s.to_safe_dict()):
            session.return_value.__enter__.return_value = db
            result = command.execute(self.args("--execute-local"))
            session.assert_called_once()
            session.return_value.__exit__.assert_called_once()
            self.assertEqual(len(db.calls), 6)
            self.assertEqual(result["h1_validation"]["compatible_races"], 1)
            self.assertFalse(result["production_settlement_enabled"])
            self.assertNotIn(SECRET, json.dumps(result))

    def test_cli_reason_and_aggregate_output_do_not_echo_h1_values_or_paths(self):
        with patch.object(command, "execute", return_value=compare().to_safe_dict()), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cli.main(["reconcile-payouts", "--check-input-only", "--reference", str(INPUT)]), 0)
            self.assertNotIn(str(INPUT), stdout.getvalue())
        with patch.object(command, "execute", side_effect=ReconciliationError("missing_h1")), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main(["reconcile-payouts", "--check-input-only", "--reference", SECRET]), 2)
            self.assertEqual(json.loads(stderr.getvalue())["reason"], "missing_h1")
            self.assertNotIn(SECRET, stderr.getvalue())

    def test_failed_h1_fetch_never_falls_back_or_writes_a_success_report(self):
        doc = replace(document(flag_rules=()), source="jra_official_manual")
        with patch.object(command, "validate_output_dir"), patch.object(command, "read_reference_file", return_value=doc), \
                patch.object(h1_local, "read_h1_config_file", return_value=config()), \
                patch.object(command.importlib.util, "find_spec", return_value=object()), \
                patch.dict(os.environ, {"PGPASSWORD": SECRET}, clear=True), patch.object(command, "read_only_session"), \
                patch.object(command, "fetch_local_hr", return_value=(hr(),)), \
                patch.object(h1_local, "fetch_local_h1", side_effect=ReconciliationError("h1_table_unavailable")), \
                patch.object(command, "reconcile_payouts") as legacy, patch.object(command, "write_reconciliation_summary") as writer:
            with self.assertRaises(ReconciliationError):
                command.execute(self.args("--execute-local"))
            legacy.assert_not_called()
            writer.assert_not_called()

    def test_h1_summary_writer_persists_only_typed_aggregate_fields(self):
        captures = []
        class Capture:
            def __enter__(self):
                self.stream = io.StringIO()
                return self.stream
            def __exit__(self, *a):
                captures.append(self.stream.getvalue())
                self.stream.close()
        root = INPUT.parent.parent
        with patch.object(local, "validate_output_dir", return_value=root), patch.object(local, "_has_link", return_value=False), \
                patch.object(local, "_cloud_or_reparse", return_value=False), patch.object(Path, "mkdir"), \
                patch.object(Path, "open", side_effect=lambda *a, **k: Capture()):
            payload = local.write_reconciliation_summary(compare(), root)
        self.assertEqual(payload["h1_validation"]["mode"], "h1_required")
        self.assertEqual(len(captures), 2)
        for text in captures:
            for private in (SECRET, "0106", "horse_number", "000000350", "fake_h1_status"):
                self.assertNotIn(private, text)


if __name__ == "__main__":
    unittest.main()
