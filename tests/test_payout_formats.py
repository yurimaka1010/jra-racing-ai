"""Phase 1.4 format tests: artificial strings only, never official HR rows."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai.payout_decoder import (
    FieldFormat, SlotKind, PayoutDecodingError, RawPayoutSlot,
    ZERO_HORSE_REQUIRED_CONTEXT, classify_payout_layout, decode_hr_payouts,
    parse_payout_layout,
)
from jra_racing_ai.refunds import RefundValidationError, determine_refund_targets
from jra_racing_ai.settlement import (
    JST, JPY_PER_100, DataOrigin, NormalizedPayout, OutcomeState, PayoutTarget,
    SettlementStatus, SettlementValidationError, VirtualBet, settle_virtual_bet,
)


def raw(a="01", b="000000350", c="02"):
    return RawPayoutSlot(a, b, c)


def blank():
    return raw("  ", " " * 9, "  ")


def classify(*slots, wager="tansho"):
    return classify_payout_layout(parse_payout_layout(wager_type=wager, slots=slots))


class BlankAndCandidateFormatTests(unittest.TestCase):
    def test_win_three_slots_keep_one_candidate_and_two_unused_slots(self):
        value = classify(raw(), blank(), blank())
        self.assertEqual([slot.kind for slot in value.slots], [
            SlotKind.ORDINARY_CANDIDATE, SlotKind.UNUSED_BLANK, SlotKind.UNUSED_BLANK,
        ])
        self.assertEqual([slot.slot_number for slot in value.slots], [1, 2, 3])
        self.assertTrue(value.has_all_slots)
        self.assertEqual(value.format_status, "valid")

    def test_place_five_slots_preserve_additional_target_and_original_order(self):
        values = (raw("08"), raw("02"), raw("11"), raw("04"), blank())
        value = classify(*values, wager="fukusho")
        self.assertEqual([slot.raw.a for slot in value.ordinary_candidates], ["08", "02", "11", "04"])
        self.assertEqual(tuple(slot.raw for slot in value.slots), values)
        self.assertEqual(value.slots[4].kind, SlotKind.UNUSED_BLANK)
        self.assertTrue(value.has_all_slots)

    def test_all_blank_layout_is_valid_format_without_zero_money_or_winners(self):
        for wager, count in (("tansho", 3), ("fukusho", 5)):
            value = classify(*(blank() for _ in range(count)), wager=wager)
            self.assertEqual(value.format_status, "valid")
            self.assertEqual(value.ordinary_candidates, ())
            self.assertFalse(value.ready_for_settlement)
            self.assertEqual(value.status, "needs_confirmation")
            self.assertTrue(all(slot.raw.b == " " * 9 for slot in value.slots))

    def test_exact_width_ascii_spaces_are_classified_per_field(self):
        slot = classify(blank()).slots[0]
        self.assertEqual(slot.field_formats, (FieldFormat.BLANK,) * 3)
        self.assertEqual(slot.required_context, ())

    def test_leading_zeros_are_retained_before_any_monetary_interpretation(self):
        slot = classify(raw(b="000000007", c="01")).ordinary_candidates[0]
        self.assertIs(type(slot.raw.b), str)
        self.assertEqual((slot.raw.a, slot.raw.b, slot.raw.c), ("01", "000000007", "01"))
        self.assertEqual(slot.field_formats, (FieldFormat.DIGITS,) * 3)
        self.assertFalse(hasattr(slot, "payout_yen_per_100"))

    def test_multiple_win_targets_remain_distinct_without_inferred_dead_heat(self):
        value = classify(raw("05"), raw("03", "000000670"), blank())
        self.assertEqual([slot.raw.a for slot in value.ordinary_candidates], ["05", "03"])
        self.assertTrue(all(not hasattr(slot, "is_dead_heat") for slot in value.slots))
        self.assertFalse(value.ready_for_settlement)

    def test_multiple_place_targets_do_not_imply_dead_heat(self):
        value = classify(raw("01"), raw("03"), raw("05"), blank(), blank(), wager="fukusho")
        self.assertEqual(len(value.ordinary_candidates), 3)
        self.assertTrue(all(not hasattr(slot, "is_dead_heat") for slot in value.slots))

    def test_candidate_horse_range_accepts_01_and_28(self):
        value = classify(raw("01"), raw("28"))
        self.assertEqual([slot.raw.a for slot in value.ordinary_candidates], ["01", "28"])

    def test_digit_rank_is_only_a_format_not_a_confirmed_rank_range(self):
        value = classify(raw(c="99"))
        self.assertEqual(value.format_status, "valid")
        self.assertEqual(value.status, "needs_confirmation")
        self.assertIn("popularity_special_values", value.required_confirmations)

    def test_partial_layout_is_marked_without_inventing_missing_slots(self):
        for wager in ("tansho", "fukusho"):
            value = classify(raw(), wager=wager)
            self.assertFalse(value.has_all_slots)
            self.assertEqual(len(value.slots), 1)
            self.assertFalse(value.ready_for_settlement)


class ZeroCodeFormatTests(unittest.TestCase):
    def test_zero_horse_with_blank_fields_is_not_an_unused_or_winning_slot(self):
        for wager in ("tansho", "fukusho"):
            value = classify(raw("00", " " * 9, "  "), wager=wager)
            slot = value.slots[0]
            self.assertEqual(slot.kind, SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION)
            self.assertEqual(slot.format_status, "needs_confirmation")
            self.assertEqual(value.format_status, "needs_confirmation")
            self.assertEqual(value.ordinary_candidates, ())

    def test_zero_horse_requires_wager_flags_race_and_revision_context(self):
        slot = classify(raw("00", " " * 9, "  ")).slots[0]
        self.assertEqual(slot.required_context, ZERO_HORSE_REQUIRED_CONTEXT)
        self.assertIn("wager_specific_refund_flags", slot.required_context)
        self.assertIn("wager_specific_special_payment_flags", slot.required_context)
        self.assertIn("wager_specific_establishment_flags", slot.required_context)
        self.assertIn("wager_specific_sale_status", slot.required_context)
        self.assertIn("race_status", slot.required_context)
        self.assertIn("correction_and_deletion_selection", slot.required_context)

    def test_zero_horse_with_digits_still_has_no_special_payment_interpretation(self):
        value = classify(raw("00"))
        self.assertEqual(value.slots[0].kind, SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION)
        self.assertEqual(value.ordinary_candidates, ())
        self.assertFalse(value.ready_for_settlement)

    def test_all_zero_digits_are_retained_as_unresolved_not_blank(self):
        slot = classify(raw("00", "000000000", "00")).slots[0]
        self.assertEqual(slot.field_formats, (FieldFormat.DIGITS,) * 3)
        self.assertEqual(slot.kind, SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION)
        self.assertEqual(slot.raw.b, "000000000")

    def test_repeated_zero_placeholders_do_not_become_duplicate_winners(self):
        value = classify(raw("00", " " * 9, "  "), raw("00", " " * 9, "  "), wager="fukusho")
        self.assertEqual(len(value.slots), 2)
        self.assertEqual(value.ordinary_candidates, ())
        self.assertEqual(value.format_status, "needs_confirmation")

    def test_zero_horse_kept_alongside_multiple_candidates_does_not_complete_targets(self):
        value = classify(raw("01"), raw("04"), raw("00", " " * 9, "  "), blank(), blank(), wager="fukusho")
        self.assertEqual(len(value.ordinary_candidates), 2)
        self.assertTrue(value.has_all_slots)
        self.assertEqual(value.format_status, "needs_confirmation")
        self.assertFalse(value.ready_for_settlement)

    def test_zero_amount_is_not_zero_yen_or_an_ordinary_candidate(self):
        value = classify(raw(b="000000000"))
        self.assertEqual(value.slots[0].kind, SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION)
        self.assertEqual(value.ordinary_candidates, ())
        self.assertEqual(value.slots[0].raw.b, "000000000")

    def test_zero_popularity_is_retained_for_confirmation(self):
        slot = classify(raw(c="00")).slots[0]
        self.assertEqual(slot.kind, SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION)
        self.assertIn("zero_amount_or_popularity_meaning", slot.required_context)

    def test_zero_horse_with_only_one_blank_field_is_inconsistent(self):
        for slot in (raw("00", " " * 9, "01"), raw("00", "000000350", "  ")):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(slot)
            self.assertEqual(caught.exception.reason, "inconsistent_slot_fields")

    def test_refund_targets_alone_cannot_resolve_zero_horse(self):
        refunds = determine_refund_targets(
            henkan_umaban_joho="0" * 28, henkan_flag_tansho="0", henkan_flag_fukusho="0",
        )
        self.assertFalse(refunds.is_refunded(wager_type="tansho", horse_number="01"))
        parsed = parse_payout_layout(wager_type="tansho", slots=(raw("00", " " * 9, "  "),))
        with self.assertRaises(TypeError):
            classify_payout_layout(parsed, refund_targets=refunds)
        self.assertEqual(classify_payout_layout(parsed).format_status, "needs_confirmation")


class InvalidPayoutFormatTests(unittest.TestCase):
    def test_null_in_each_field_stops_without_replacing_it(self):
        for index in range(3):
            values = ["01", "000000350", "02"]
            values[index] = None
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(RawPayoutSlot(*values))
            self.assertEqual(caught.exception.reason, "missing_field")

    def test_empty_string_is_not_a_blank_slot(self):
        for index in range(3):
            values = ["  ", " " * 9, "  "]
            values[index] = ""
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(RawPayoutSlot(*values))
            self.assertEqual(caught.exception.reason, "invalid_field_width")

    def test_wrong_width_digits_or_blanks_are_not_padded_or_trimmed(self):
        for index, width in enumerate((2, 9, 2)):
            for size in (1, width - 1, width + 1):
                for char in ("0", " "):
                    values = ["01", "000000350", "02"]
                    values[index] = char * size
                    with self.subTest(index=index, size=size, char=char), self.assertRaises(PayoutDecodingError):
                        classify(RawPayoutSlot(*values))

    def test_numbers_bytes_and_bool_are_not_coerced_to_strings(self):
        for index in range(3):
            for invalid in (0, 350, True, 3.5, b"01"):
                values = ["01", "000000350", "02"]
                values[index] = invalid
                with self.assertRaises(PayoutDecodingError) as caught:
                    classify(RawPayoutSlot(*values))
                self.assertEqual(caught.exception.reason, "invalid_field_type")

    def test_non_ascii_digits_in_any_field_are_rejected(self):
        for index, invalid in enumerate(("０１", "００００００３５０", "０２")):
            values = ["01", "000000350", "02"]
            values[index] = invalid
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(RawPayoutSlot(*values))
            self.assertEqual(caught.exception.reason, "unsupported_field_characters")

    def test_tabs_newlines_unicode_spaces_and_nuls_are_not_blank(self):
        for index, width in enumerate((2, 9, 2)):
            for char in ("\t", "\n", "\r", "\x00", "\u00a0", "\u3000", "\u200b"):
                values = ["  ", " " * 9, "  "]
                values[index] = char * width
                with self.assertRaises(PayoutDecodingError) as caught:
                    classify(RawPayoutSlot(*values))
                self.assertEqual(caught.exception.reason, "unsupported_field_characters")

    def test_mixed_spaces_digits_signs_and_decimal_points_are_not_repaired(self):
        for index, invalid_values in enumerate((
            (" 1", "1 ", "+1", "-1", "1."),
            (" 00000350", "00000350 ", "+00000350", "0000003.5", "PRIVATE_X"),
            (" 2", "2 ", "+2", "-2", "2."),
        )):
            for invalid in invalid_values:
                values = ["01", "000000350", "02"]
                values[index] = invalid
                with self.assertRaises(PayoutDecodingError):
                    classify(RawPayoutSlot(*values))

    def test_every_partial_blank_combination_with_regular_horse_is_unresolved(self):
        digits = ("01", "000000350", "02")
        blanks = ("  ", " " * 9, "  ")
        for mask in range(1, 7):
            values = tuple(blanks[i] if mask & (1 << i) else digits[i] for i in range(3))
            with self.subTest(mask=mask), self.assertRaises(PayoutDecodingError) as caught:
                classify(RawPayoutSlot(*values))
            self.assertEqual(caught.exception.reason, "inconsistent_slot_fields")

    def test_horse_outside_01_to_28_is_not_a_target(self):
        for horse in ("29", "30", "99"):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(raw(horse))
            self.assertEqual(caught.exception.reason, "candidate_horse_out_of_range")

    def test_duplicate_populated_slot_is_rejected_while_repeated_blank_is_allowed(self):
        self.assertEqual(classify(raw(), blank(), blank()).format_status, "valid")
        with self.assertRaises(PayoutDecodingError) as caught:
            classify(raw(), raw(), blank())
        self.assertEqual(caught.exception.reason, "duplicate_raw_slot")

    def test_same_candidate_horse_with_different_amount_or_rank_is_rejected(self):
        for other in (raw(b="000000360"), raw(c="03"), raw(b="000000000")):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(raw(), other, wager="fukusho")
            self.assertEqual(caught.exception.reason, "duplicate_a_requires_confirmation")

    def test_extra_slot_is_rejected_even_when_it_is_blank(self):
        for wager, count in (("tansho", 4), ("fukusho", 6)):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(*(blank() for _ in range(count)), wager=wager)
            self.assertEqual(caught.exception.reason, "too_many_slots")

    def test_other_wager_families_are_not_given_the_single_horse_schema(self):
        for wager in ("umaren", "wide", "wakuren", "umatan", "sanrenpuku", "sanrentan"):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify(raw(), wager=wager)
            self.assertEqual(caught.exception.reason, "unsupported_wager_type")

    def test_classifier_rejects_arbitrary_objects_and_revalidates_mutated_layout(self):
        for value in (None, raw(), {"PRIVATE_ROW": "PRIVATE_VALUE"}):
            with self.assertRaises(PayoutDecodingError) as caught:
                classify_payout_layout(value)
            self.assertEqual(caught.exception.reason, "invalid_parsed_layout")
        parsed = parse_payout_layout(wager_type="tansho", slots=(raw(),))
        object.__setattr__(parsed, "slots", (raw(b=None),))
        with self.assertRaises(PayoutDecodingError) as caught:
            classify_payout_layout(parsed)
        self.assertEqual(caught.exception.reason, "missing_field")


class FormatGateAndCompatibilityTests(unittest.TestCase):
    def test_valid_complete_layout_does_not_open_the_monetary_gate(self):
        for wager, values in (("tansho", (raw(), blank(), blank())),
                              ("fukusho", (raw("01"), raw("02"), blank(), blank(), blank()))):
            parsed = parse_payout_layout(wager_type=wager, slots=values)
            classified = classify_payout_layout(parsed)
            self.assertTrue(classified.has_all_slots)
            self.assertEqual(classified.format_status, "valid")
            self.assertFalse(classified.ready_for_settlement)
            with self.assertRaises(PayoutDecodingError) as caught:
                decode_hr_payouts(parsed)
            self.assertEqual(caught.exception.reason, "payout_semantics_unconfirmed")
            self.assertIn("currency_and_base_stake", classified.required_confirmations)
            self.assertIn("correction_and_deletion_selection", classified.required_confirmations)

    def test_all_blank_layout_does_not_mean_zero_return_or_not_offered(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(blank(), blank(), blank()))
        with self.assertRaises(PayoutDecodingError) as caught:
            decode_hr_payouts(parsed)
        self.assertEqual(caught.exception.reason, "payout_semantics_unconfirmed")

    def test_zero_horse_and_zero_values_use_the_closed_special_value_gate(self):
        for value in (raw("00", " " * 9, "  "), raw("00"), raw(b="000000000"), raw(c="00")):
            parsed = parse_payout_layout(wager_type="tansho", slots=(value, blank(), blank()))
            with self.assertRaises(PayoutDecodingError) as caught:
                decode_hr_payouts(parsed)
            self.assertEqual(caught.exception.reason, "payout_special_values_unconfirmed")

    def test_no_confirmation_or_unit_override_can_bypass_classification(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(raw(),))
        for override in ({"assume_confirmed": True}, {"amount_basis": JPY_PER_100}, {"flags": {}}):
            with self.assertRaises(TypeError):
                classify_payout_layout(parsed, **override)
            with self.assertRaises(TypeError):
                decode_hr_payouts(parsed, **override)

    def test_layout_and_classified_results_are_not_settlement_input(self):
        start = datetime(2024, 2, 3, 12, tzinfo=JST)
        bet = VirtualBet("fake-format-ticket", "fake-format-race", "tansho", "01", 100,
                         start - timedelta(minutes=4))
        parsed = parse_payout_layout(wager_type="tansho", slots=(raw(), blank(), blank()))
        classified = classify_payout_layout(parsed)
        for candidate in (parsed, classified, classified.slots[0], classified.ordinary_candidates):
            with self.assertRaises(SettlementValidationError):
                settle_virtual_bet(bet, candidate)

    def test_synthetic_settlement_and_existing_refund_bitmap_still_work(self):
        start = datetime(2024, 2, 3, 12, tzinfo=JST)
        refunds = determine_refund_targets(
            henkan_umaban_joho="010" + "0" * 25,
            henkan_flag_tansho="1", henkan_flag_fukusho="1",
        )
        outcome = NormalizedPayout(
            race_ref="fake-format-race", wager_type="tansho", race_date=date(2024, 2, 3),
            origin=DataOrigin.SYNTHETIC, state=OutcomeState.NORMAL, amount_basis=JPY_PER_100,
            race_start_at=start, finalized_at=start + timedelta(minutes=8), revision_state="final",
            targets=(PayoutTarget("01", 450),), targets_complete=True, refund_targets=refunds,
        )
        for horse, status, returned, net in (("01", SettlementStatus.NORMAL_HIT, 900, 700),
                                              ("02", SettlementStatus.REFUND, 200, 0)):
            bet = VirtualBet("fake-format-" + horse, "fake-format-race", "tansho", horse, 200,
                             start - timedelta(minutes=4))
            result = settle_virtual_bet(bet, outcome)
            self.assertEqual((result.status, result.returned_yen, result.net_profit_yen), (status, returned, net))

    def test_mixed_refund_flags_still_require_confirmation(self):
        for win, place in (("0", "1"), ("1", "0")):
            with self.assertRaises(RefundValidationError):
                determine_refund_targets(
                    henkan_umaban_joho="1" + "0" * 27, henkan_flag_tansho=win, henkan_flag_fukusho=place,
                )

    def test_redacted_repr_and_fixed_diagnostics_hide_slot_values(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(raw(b="987654321"), blank(), blank()))
        value = classify_payout_layout(parsed)
        for item in (parsed, value, *value.slots, *[slot.raw for slot in value.slots]):
            self.assertNotIn("987654321", repr(item))
        with self.assertRaises(PayoutDecodingError) as caught:
            classify(raw(b="PRIVATE_X"))
        error = caught.exception
        self.assertNotIn("PRIVATE_X", str(error) + repr(error) + json.dumps(vars(error)) + json.dumps(error.diagnostic()))
        self.assertEqual(error.diagnostic()["status"], "needs_confirmation")

    def test_classified_results_and_raw_strings_are_immutable(self):
        value = classify(raw(), blank(), blank())
        for item, attribute, replacement in ((value, "slots", ()), (value.slots[0], "kind", SlotKind.UNUSED_BLANK),
                                               (value.slots[0].raw, "b", "000000000")):
            with self.assertRaises(FrozenInstanceError):
                setattr(item, attribute, replacement)

    def test_format_classification_has_no_io_or_logging_side_effects(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("jra_racing_ai.db_connection.read_only_session") as db, \
                patch.object(Path, "mkdir") as mkdir, patch("builtins.open") as opened, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            parsed = parse_payout_layout(wager_type="tansho", slots=(raw(), blank(), blank()))
            classify_payout_layout(parsed)
            with self.assertRaises(PayoutDecodingError):
                decode_hr_payouts(parsed)
            db.assert_not_called()
            mkdir.assert_not_called()
            opened.assert_not_called()
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
