"""All identities, amounts, flags and records in this file are invented."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai.payout_decoder import RawPayoutSlot, decode_hr_payouts, parse_payout_layout, PayoutDecodingError
from jra_racing_ai.payout_reconciliation import (
    ALL_HYPOTHESES, FLAG_KINDS, WAGERS, AmountHypothesis, CaseState, FlagRule, LocalHRRecord,
    RaceKey, ReconciliationError, ReconciliationLimits, ReferenceCase, ReferenceDocument,
    ReferenceTarget, RevisionReview, parse_reference_document, reconcile_payouts,
)


def race(day="0106", venue="04", number="09", year="2025"):
    return RaceKey((year, day, venue, "01", "02", number))


def target(horse="01", amount=350, plus10=False):
    return ReferenceTarget(horse, amount, plus10)


def reference_case(wager="tansho", targets=None, **changes):
    values = dict(race=race(), wager_type=wager, state=CaseState.NORMAL,
                  revision_review=RevisionReview.SNAPSHOT_REVIEWED, dead_heat=False,
                  place_paid_positions=3 if wager == "fukusho" else None,
                  targets=(target(),) if targets is None else targets)
    return ReferenceCase(**(values | changes))


def rules():
    return tuple(FlagRule(w, k, f"fake_{w}_{k}", "0", "1", True) for w in WAGERS for k in FLAG_KINDS)


def document(*cases, **changes):
    values = dict(source="synthetic", source_reviewed_locally=True, abc_mapping_reviewed_locally=True,
                  status_column="fake_category", flag_rules=rules(), cases=cases or (reference_case(),))
    return ReferenceDocument(**(values | changes))


def layout(*values, wager="tansho"):
    count = 3 if wager == "tansho" else 5
    return tuple(RawPayoutSlot(a, b, c) for a, b, c in values) + tuple(
        RawPayoutSlot("  ", " " * 9, "  ") for _ in range(count - len(values)))


def record(**changes):
    values = dict(race=race(), category="2", tansho=layout(("01", "000000350", "02")),
                  fukusho=layout(("01", "000000350", "02"), wager="fukusho"),
                  refund_bitmap="0" * 28, refund_tansho="0", refund_fukusho="0", flags=("0",) * 6)
    return LocalHRRecord(**(values | changes))


def evidence(doc=None, rows=None, wager="tansho", **kwargs):
    result = reconcile_payouts(doc or document(), (record(),) if rows is None else rows, **kwargs)
    return next(w for w in result.wagers if w.wager_type == wager)


def observation(value, hypothesis=AmountHypothesis.RAW_YEN):
    return next(o for o in value.observations if o.hypothesis is hypothesis)


def json_reference():
    return {
        "format_version": 1, "source": "synthetic", "official_base_yen": 100,
        "source_reviewed_locally": True, "abc_mapping_reviewed_locally": True,
        "status_column": "fake_category",
        "flag_rules": [{name: getattr(r, name) for name in (
            "wager_type", "kind", "column", "normal_value", "active_value", "reviewed_locally")} for r in rules()],
        "cases": [{"race_key": dict(zip(("kaisai_nen", "kaisai_tsukihi", "keibajo_code", "kaisai_kai", "kaisai_nichime", "race_bango"), race().parts)),
                   "wager_type": "tansho", "state": "normal", "revision_review": "snapshot_reviewed_locally",
                   "dead_heat": False, "place_paid_positions": None,
                   "targets": [{"horse_number": "01", "payout_yen_per_100": 350, "plus10": False}]}],
    }


class ReconciliationContractTests(unittest.TestCase):
    def test_shipped_template_is_synthetic_and_has_no_review_approvals(self):
        value = json.loads((Path(__file__).resolve().parents[1] / "examples" / "reconciliation.synthetic.json").read_text(encoding="utf-8"))
        parsed = parse_reference_document(value)
        self.assertEqual(parsed.source, "synthetic")
        self.assertEqual({c.wager_type for c in parsed.cases}, set(WAGERS))
        self.assertFalse(parsed.source_reviewed_locally)
        self.assertFalse(parsed.abc_mapping_reviewed_locally)
        self.assertTrue(all(not r.reviewed_locally for r in parsed.flag_rules))
        self.assertTrue(all(c.revision_review is RevisionReview.UNCONFIRMED for c in parsed.cases))

    def test_reference_round_trip_with_explicit_local_reviews(self):
        value = parse_reference_document(json.loads(json.dumps(json_reference())))
        self.assertEqual(value, document())

    def test_exact_six_part_race_key_and_calendar_are_required(self):
        invalid = [("2025", "0230", "04", "01", "02", "09"),
                   ("2025", "0106", "11", "01", "02", "09"),
                   ("2025", "0106", "04", "00", "02", "09"),
                   ("2025", "106", "04", "01", "02", "09"),
                   ("２０２５", "0106", "04", "01", "02", "09")]
        for parts in invalid:
            with self.assertRaises(ReconciliationError):
                RaceKey(parts)

    def test_target_horse_is_not_zero_or_out_of_range(self):
        for value in ("00", "29", "1", " 1", 1, "０１"):
            with self.assertRaises(ReconciliationError):
                target(value)

    def test_official_reference_amount_is_a_positive_exact_integer(self):
        for amount in (0, -1, True, 350.0, "350", None, 10**12 + 1):
            with self.assertRaises(ReconciliationError):
                target(amount=amount)

    def test_unknown_or_mistyped_source_and_base_are_rejected(self):
        for field, bad in (("source", "final_odds"), ("official_base_yen", 1),
                           ("official_base_yen", True), ("format_version", True),
                           ("source_reviewed_locally", "yes")):
            value = json_reference()
            value[field] = bad
            with self.assertRaises(ReconciliationError):
                parse_reference_document(value)

    def test_unknown_json_fields_cannot_supply_assume_confirmed(self):
        value = json_reference()
        value["assume_confirmed"] = True
        with self.assertRaises(ReconciliationError):
            parse_reference_document(value)

    def test_duplicate_case_is_not_silently_deduplicated(self):
        with self.assertRaises(ReconciliationError) as caught:
            document(reference_case(), reference_case())
        self.assertEqual(caught.exception.reason, "duplicate_reference_case")

    def test_duplicate_target_is_rejected_even_with_same_amount(self):
        with self.assertRaises(ReconciliationError) as caught:
            reference_case("fukusho", (target(), target()))
        self.assertEqual(caught.exception.reason, "duplicate_reference_target")

    def test_conflicting_amount_for_same_horse_is_rejected(self):
        with self.assertRaises(ReconciliationError):
            reference_case("fukusho", (target(), target(amount=360)))

    def test_revision_reviews_for_same_race_must_agree(self):
        with self.assertRaises(ReconciliationError) as caught:
            document(reference_case(), reference_case("fukusho", revision_review=RevisionReview.CORRECTION))
        self.assertEqual(caught.exception.reason, "conflicting_reference")

    def test_win_multiple_targets_require_explicit_dead_heat(self):
        with self.assertRaises(ReconciliationError):
            reference_case(targets=(target(), target("02")))
        self.assertEqual(len(reference_case(targets=(target(), target("02")), dead_heat=True).targets), 2)

    def test_win_three_and_place_five_target_bounds_are_enforced(self):
        for wager, count in (("tansho", 4), ("fukusho", 6)):
            with self.assertRaises(ReconciliationError):
                reference_case(wager, tuple(target(f"{n:02d}") for n in range(1, count+1)), dead_heat=True)

    def test_special_case_cannot_embed_normal_winning_amounts(self):
        for state in CaseState:
            if state is not CaseState.NORMAL:
                with self.assertRaises(ReconciliationError):
                    reference_case(state=state)

    def test_flag_meanings_are_explicit_not_inferred(self):
        with self.assertRaises(ReconciliationError):
            FlagRule("tansho", "special_payment", "fake_flag", "0", "0", True)
        with self.assertRaises(ReconciliationError):
            FlagRule("tansho", "special_payment", "fake_flag; DELETE", "0", "1", True)

    def test_mapping_cannot_reuse_reserved_columns_or_conflicting_flag_codes(self):
        for column in ("haraimodoshi_tansho_1b", "henkan_flag_tansho", "fake_category", rules()[1].column):
            changed = (replace(rules()[0], column=column),) + rules()[1:]
            with self.assertRaises(ReconciliationError):
                document(flag_rules=changed)

    def test_shared_code_column_supports_distinct_special_states_without_guessing(self):
        shared_rules = list(rules())
        shared_rules[2] = replace(shared_rules[2], column=shared_rules[0].column, active_value="2")
        doc = document(flag_rules=tuple(shared_rules))
        self.assertEqual(observation(evidence(doc)).matched, 1)
        value = evidence(doc, (record(flags=("2", "0", "2", "0", "0", "0")),))
        self.assertEqual(dict(value.reasons)["hr_not_offered"], 1)
        self.assertEqual(observation(value).matched, 0)

    def test_more_than_requested_races_and_wrong_year_are_rejected(self):
        doc = document(reference_case(), reference_case(race=race(number="10")))
        with self.assertRaises(ReconciliationError) as caught:
            reconcile_payouts(doc, (), limits=ReconciliationLimits(max_races=1))
        self.assertEqual(caught.exception.reason, "race_limit")
        with self.assertRaises(ReconciliationError):
            reconcile_payouts(document(reference_case(race=race(year="2024"))), ())

    def test_limits_reject_bool_unbounded_counts_and_inconsistent_time_budget(self):
        for changes in ({"max_races": 21}, {"max_hr_rows": 201}, {"max_races": True},
                        {"statement_timeout_ms": 999}, {"statement_timeout_ms": 10001},
                        {"max_run_seconds": 9}, {"max_run_seconds": 121},
                        {"statement_timeout_ms": 10000, "max_run_seconds": 30}):
            with self.assertRaises(ReconciliationError):
                ReconciliationLimits(**changes)


class NormalHypothesisComparisonTests(unittest.TestCase):
    def test_same_race_wager_horse_matches_for_win_and_place(self):
        result = reconcile_payouts(document(reference_case(), reference_case("fukusho")), (record(),))
        for value in result.wagers:
            obs = observation(value)
            self.assertEqual((obs.matched, obs.mismatched, obs.uncomparable), (1, 0, 0))
            self.assertEqual((value.eligible_cases, value.blocked_cases), (1, 0))

    def test_official_amount_mismatch_is_not_a_unit_confirmation(self):
        value = evidence(document(reference_case(targets=(target(amount=360),))))
        self.assertEqual(observation(value).mismatched, 1)
        self.assertEqual(observation(value).to_dict()["observation_status"], "mismatch_observed")
        self.assertFalse(observation(value).to_dict()["unit_confirmed"])

    def test_all_three_ratios_are_named_hypotheses_not_overrides(self):
        value = evidence()
        self.assertEqual(observation(value).matched, 1)
        self.assertEqual(observation(value, AmountHypothesis.DIVIDE_10).mismatched, 1)
        self.assertEqual(observation(value, AmountHypothesis.MULTIPLY_10).mismatched, 1)

    def test_divide_ten_hypothesis_can_match_without_affecting_original(self):
        row = record(tansho=layout(("01", "000003500", "02")))
        value = evidence(rows=(row,))
        self.assertEqual(observation(value, AmountHypothesis.DIVIDE_10).matched, 1)
        self.assertEqual(row.tansho[0].b, "000003500")
        self.assertIs(type(row.tansho[0].b), str)

    def test_multiply_ten_hypothesis_can_match(self):
        value = evidence(rows=(record(tansho=layout(("01", "000000035", "02"))),))
        self.assertEqual(observation(value, AmountHypothesis.MULTIPLY_10).matched, 1)

    def test_fractional_yen_hypothesis_is_uncomparable_without_rounding(self):
        value = evidence(document(reference_case(targets=(target(amount=351),))),
                         (record(tansho=layout(("01", "000000351", "02"))),))
        self.assertEqual(observation(value).matched, 1)
        obs = observation(value, AmountHypothesis.DIVIDE_10)
        self.assertEqual((obs.matched, obs.mismatched, obs.uncomparable), (0, 0, 1))
        self.assertEqual(dict(value.reasons)["fractional_yen_hypothesis"], 1)

    def test_same_race_different_wagers_are_compared_to_their_own_amounts(self):
        doc = document(reference_case(), reference_case("fukusho", (target(amount=140),)))
        result = reconcile_payouts(doc, (record(fukusho=layout(("01", "000000140", "01"), wager="fukusho")),))
        self.assertTrue(all(observation(w).matched == 1 for w in result.wagers))

    def test_each_of_three_win_dead_heat_targets_is_compared(self):
        targets = (target("01", 350), target("05", 480), target("08", 610))
        doc = document(reference_case(targets=targets, dead_heat=True))
        row = record(tansho=layout(("01", "000000350", "01"), ("05", "000000480", "02"), ("08", "000000610", "03")))
        value = evidence(doc, (row,))
        self.assertEqual((observation(value).matched, value.dead_heat_cases), (3, 1))

    def test_five_place_targets_keep_one_individual_mismatch(self):
        targets = tuple(target(f"{n:02d}", 150+n*10) for n in range(1, 6))
        row_values = [(t.horse_number, f"{t.payout_yen_per_100:09d}", "01") for t in targets]
        row_values[-1] = ("05", "000000210", "01")
        doc = document(reference_case("fukusho", targets, dead_heat=True))
        value = evidence(doc, (record(fukusho=layout(*row_values, wager="fukusho")),), wager="fukusho")
        self.assertEqual((observation(value).matched, observation(value).mismatched), (4, 1))

    def test_plus10_official_110_is_compared_directly_without_odds(self):
        doc = document(reference_case(targets=(target(amount=110, plus10=True),)))
        value = evidence(doc, (record(tansho=layout(("01", "000000110", "01"))),))
        self.assertEqual((observation(value).matched, value.plus10_targets), (1, 1))

    def test_plus10_mismatch_stays_a_mismatch_without_adding_ten(self):
        doc = document(reference_case(targets=(target(amount=110, plus10=True),)))
        value = evidence(doc, (record(tansho=layout(("01", "000000100", "01"))),))
        self.assertEqual(observation(value).mismatched, 1)

    def test_unknown_plus10_annotation_is_not_inferred_from_amount(self):
        value = evidence(document(reference_case(targets=(target(plus10=None),))))
        self.assertEqual((value.plus10_targets, value.plus10_unknown_targets), (0, 1))
        self.assertEqual(observation(value).matched, 1)

    def test_match_rate_excludes_uncomparable_but_keeps_its_count(self):
        doc = document(reference_case(), reference_case(race=race(number="10")), reference_case(race=race(number="11")))
        value = evidence(doc, (record(), record(race=race(number="10"), tansho=layout(("01", "000000360", "02")))))
        obs = observation(value)
        self.assertEqual((obs.matched, obs.mismatched, obs.uncomparable), (1, 1, 1))
        self.assertEqual(obs.to_dict()["match_rate_among_comparable"], 0.5)

    def test_zero_comparable_targets_has_no_percentage(self):
        self.assertIsNone(observation(evidence(rows=())).to_dict()["match_rate_among_comparable"])

    def test_only_one_wager_sample_does_not_approve_the_other(self):
        result = reconcile_payouts(document(), (record(),)).to_safe_dict()
        self.assertEqual(result["wagers"][1]["cases"], 0)
        self.assertFalse(result["production_settlement_enabled"])
        self.assertEqual(result["amount_interpretation_status"], "needs_confirmation")


class BlockedHRCaseTests(unittest.TestCase):
    def assert_block(self, reason, *, doc=None, rows=None, wager="tansho"):
        value = evidence(doc, rows, wager=wager)
        self.assertEqual(dict(value.reasons).get(reason), 1)
        self.assertEqual((value.eligible_cases, value.blocked_cases), (0, 1))
        self.assertEqual(observation(value).matched + observation(value).mismatched, 0)
        return value

    def test_missing_hr_is_not_a_loss(self):
        self.assert_block("missing_hr", rows=())

    def test_multiple_hr_versions_are_not_resolved_by_max_status(self):
        self.assert_block("multiple_hr_versions", rows=(record(category="1"), record(category="2")))

    def test_duplicate_identical_hr_records_still_block(self):
        self.assert_block("multiple_hr_versions", rows=(record(), record()))

    def test_preliminary_deleted_cancelled_and_unknown_are_separate(self):
        for category, reason in (("0", "deleted_hr"), ("1", "preliminary_hr"), ("9", "cancelled_hr"),
                                 ("3", "unknown_hr_status"), (None, "unknown_hr_status")):
            self.assert_block(reason, rows=(record(category=category),))

    def test_category_two_does_not_verify_revision(self):
        self.assert_block("revision_unreviewed", doc=document(reference_case(revision_review=RevisionReview.UNCONFIRMED)))

    def test_correction_is_not_automatically_adopted(self):
        self.assert_block("correction_unreviewed", doc=document(reference_case(revision_review=RevisionReview.CORRECTION)))

    def test_official_special_states_have_no_normal_targets(self):
        for state in CaseState:
            if state is CaseState.NORMAL:
                continue
            value = self.assert_block("reference_" + state.value,
                                     doc=document(reference_case(state=state, targets=())))
            self.assertEqual(observation(value).uncomparable, 0)

    def test_refund_context_can_coexist_with_a_different_normal_winner(self):
        value = evidence(rows=(record(refund_bitmap="01" + "0" * 26, refund_tansho="1", refund_fukusho="1"),))
        self.assertEqual((observation(value).matched, value.refund_present_cases), (1, 1))

    def test_refund_target_and_normal_payout_overlap_stops(self):
        self.assert_block("refund_payout_overlap", rows=(record(
            refund_bitmap="1" + "0" * 27, refund_tansho="1", refund_fukusho="1"),))

    def test_mixed_or_invalid_refund_flags_still_block(self):
        for changes in ({"refund_tansho": "1"}, {"refund_bitmap": None}, {"refund_bitmap": "0" * 27}):
            self.assert_block("refund_context_unconfirmed", rows=(record(**changes),))

    def test_zero_horse_is_not_removed_to_force_a_target_match(self):
        self.assert_block("zero_horse_unconfirmed", rows=(record(tansho=layout(
            ("01", "000000350", "02"), ("00", " " * 9, "  "))),))

    def test_place_two_paid_with_zero_horse_stays_unconfirmed(self):
        doc = document(reference_case("fukusho", (target(), target("02", 140)), place_paid_positions=2))
        row = record(fukusho=layout(("01", "000000350", "02"), ("02", "000000140", "01"),
                                   ("00", " " * 9, "  "), wager="fukusho"))
        value = self.assert_block("place_two_paid_unconfirmed", doc=doc, rows=(row,), wager="fukusho")
        self.assertEqual(value.place_two_paid_cases, 1)

    def test_unspecified_place_terms_are_not_assumed_three_paid(self):
        self.assert_block("place_terms_unreviewed", doc=document(reference_case("fukusho", place_paid_positions=None)), wager="fukusho")

    def test_zero_amount_or_rank_stops_before_integer_comparison(self):
        for b, c in (("000000000", "01"), ("000000350", "00")):
            self.assert_block("zero_value_unconfirmed", rows=(record(tansho=layout(("01", b, c))),))

    def test_null_invalid_width_characters_and_combinations_stay_uncomparable(self):
        for b, c, reason in ((None, "01", "missing_field"), ("00000350", "01", "invalid_field_width"),
                              ("000003.50", "01", "unsupported_field_characters"), (" " * 9, "01", "inconsistent_slot_fields")):
            self.assert_block(reason, rows=(record(tansho=layout(("01", b, c))),))

    def test_duplicate_target_and_incomplete_layout_stop(self):
        self.assert_block("duplicate_raw_slot", rows=(record(tansho=layout(
            ("01", "000000350", "02"), ("01", "000000350", "02"))),))
        self.assert_block("incomplete_layout", rows=(record(tansho=(RawPayoutSlot("01", "000000350", "02"),)),))

    def test_missing_flag_rules_or_unreviewed_meanings_stop(self):
        self.assert_block("flag_mapping_unreviewed", doc=document(flag_rules=()))
        self.assert_block("flag_mapping_unreviewed", doc=document(flag_rules=(replace(rules()[0], reviewed_locally=False),) + rules()[1:]))

    def test_explicit_active_flags_are_not_treated_as_normal(self):
        for i, kind in enumerate(FLAG_KINDS):
            flags = list(record().flags)
            flags[i] = "1"
            self.assert_block("hr_" + kind, rows=(record(flags=tuple(flags)),))

    def test_unknown_flag_code_or_missing_value_is_not_defaulted_to_normal(self):
        for value in (None, " ", "9", 0):
            self.assert_block("flag_value_unconfirmed", rows=(record(flags=(value,) + ("0",) * 5),))

    def test_source_and_abc_meanings_need_local_review(self):
        self.assert_block("reference_source_unreviewed", doc=document(source_reviewed_locally=False))
        self.assert_block("abc_mapping_unreviewed", doc=document(abc_mapping_reviewed_locally=False))

    def test_target_set_missing_or_extra_horse_is_not_a_monetary_mismatch(self):
        self.assert_block("target_set_mismatch", rows=(record(tansho=layout(("02", "000000350", "02"))),))
        value = self.assert_block("target_set_mismatch", rows=(record(tansho=layout(
            ("01", "000000350", "02"), ("02", "000000490", "03"))),))
        self.assertEqual(value.unexpected_hr_targets, 1)

    def test_empty_unused_layout_is_not_a_zero_return(self):
        self.assert_block("target_set_mismatch", rows=(record(tansho=layout()),))

    def test_unrequested_hr_race_and_row_overflow_abort_without_partial_summary(self):
        for rows, limits, reason in (((record(race=race(number="10")),), ReconciliationLimits(), "unexpected_hr_race"),
                                    ((record(), record()), ReconciliationLimits(max_hr_rows=1), "row_limit")):
            with self.assertRaises(ReconciliationError) as caught:
                reconcile_payouts(document(), rows, limits=limits)
            self.assertEqual(caught.exception.reason, reason)

    def test_oversized_field_marker_stays_a_rejection(self):
        self.assert_block("oversized_hr_field", rows=(record(oversized=True),))


class ReconciliationBoundaryTests(unittest.TestCase):
    def test_original_b_and_private_containers_are_not_printed(self):
        row = record(tansho=layout(("01", "987654321", "02")))
        doc = document(reference_case(targets=(target(amount=987654321),)))
        summary = reconcile_payouts(doc, (row,)).to_safe_dict()
        text = json.dumps(summary)
        for item in (row, doc, doc.cases[0], doc.cases[0].targets[0], doc.races[0], *doc.flag_rules):
            self.assertNotIn("987654321", repr(item))
        self.assertNotIn("987654321", text)
        self.assertNotIn("0106", text)
        self.assertNotIn("horse_number", text)

    def test_errors_contain_only_fixed_reason_codes(self):
        error = ReconciliationError("PRIVATE_RECORD_VALUE")
        self.assertNotIn("PRIVATE_RECORD_VALUE", str(error) + repr(error) + json.dumps(error.diagnostic()) + json.dumps(vars(error)))

    def test_function_has_no_io_or_connection_side_effects(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("jra_racing_ai.db_connection.read_only_session") as db, patch("builtins.open") as opened, \
                patch.object(Path, "mkdir") as mkdir, redirect_stdout(stdout), redirect_stderr(stderr):
            reconcile_payouts(document(), (record(),))
            db.assert_not_called()
            opened.assert_not_called()
            mkdir.assert_not_called()
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")

    def test_inputs_remain_immutable(self):
        row = record()
        with self.assertRaises(FrozenInstanceError):
            row.category = "0"
        reconcile_payouts(document(), (row,))
        self.assertEqual(row.tansho[0].b, "000000350")

    def test_successful_sample_cannot_open_existing_decoder(self):
        self.assertFalse(reconcile_payouts(document(), (record(),)).to_safe_dict()["production_settlement_enabled"])
        with self.assertRaises(PayoutDecodingError):
            decode_hr_payouts(parse_payout_layout(wager_type="tansho", slots=record().tansho))

    def test_odds_probability_and_normalized_settlement_are_not_inputs(self):
        for extras in ({"final_odds": 3.5}, {"probability": 0.4}, {"enable_settlement": True}):
            with self.assertRaises(TypeError):
                reconcile_payouts(document(), (record(),), **extras)

    def test_hypotheses_must_be_known_distinct_enum_values(self):
        for values in ((), ("raw_integer_as_yen_per_100",), (ALL_HYPOTHESES[0],) * 2, [ALL_HYPOTHESES[0]]):
            with self.assertRaises(ReconciliationError):
                reconcile_payouts(document(), (record(),), hypotheses=values)

    def test_future_year_evidence_is_separate_and_never_approved(self):
        key = race(year="2024")
        summary = reconcile_payouts(document(reference_case(race=key)), (record(race=key),),
                                     limits=ReconciliationLimits(year=2024)).to_safe_dict()
        self.assertEqual(summary["year"], 2024)
        self.assertFalse(summary["production_settlement_enabled"])


if __name__ == "__main__":
    unittest.main()
