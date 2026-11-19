"""Artificial field strings: no official records, monetary mapping, or SQL."""
from dataclasses import FrozenInstanceError
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.payout_decoder import (
    UNCONFIRMED_HR_RULES, PayoutDecodingError, RawPayoutSlot,
    decode_hr_payouts, parse_payout_layout,
)


def slot(a="01", b="000000420", c="02"):
    return RawPayoutSlot(a, b, c)


class PayoutLayoutTests(unittest.TestCase):
    def test_confirmed_widths_preserve_opaque_fields_as_strings(self):
        value = slot()
        parsed = parse_payout_layout(wager_type="tansho", slots=(value,))
        self.assertEqual(parsed.slots, (value,))
        self.assertIs(type(parsed.slots[0].b), str)
        self.assertEqual(parsed.slots[0].b, "000000420")
        self.assertFalse(parsed.ready_for_settlement)
        self.assertEqual(parsed.status, "needs_confirmation")

    def test_win_three_and_place_five_slots_are_retained(self):
        for wager, count in (("tansho", 3), ("fukusho", 5)):
            values = tuple(slot(f"{n:02d}") for n in range(1, count+1))
            parsed = parse_payout_layout(wager_type=wager, slots=values)
            self.assertEqual(len(parsed.slots), count)
            with self.assertRaises(PayoutDecodingError) as caught:
                decode_hr_payouts(parsed)
            self.assertEqual(caught.exception.reason, "payout_semantics_unconfirmed")

    def test_slot_bounds_and_missing_slots_stop(self):
        for wager, count in (("tansho", 4), ("fukusho", 6)):
            with self.assertRaises(PayoutDecodingError) as caught:
                parse_payout_layout(wager_type=wager, slots=tuple(slot() for _ in range(count)))
            self.assertEqual(caught.exception.reason, "too_many_slots")
        with self.assertRaises(PayoutDecodingError) as caught:
            parse_payout_layout(wager_type="tansho", slots=())
        self.assertEqual(caught.exception.reason, "missing_slots")

    def test_containers_and_slot_types_are_strict(self):
        for values in (None, [slot()], "PRIVATE_RAW_ROW", ({"a": "01", "b": "000000420", "c": "02"},)):
            with self.assertRaises(PayoutDecodingError):
                parse_payout_layout(wager_type="tansho", slots=values)

    def test_invalid_wager_type_stops_without_echo(self):
        for wager in ("win", "単勝", "PRIVATE_WAGER_VALUE", None, True):
            with self.assertRaises(PayoutDecodingError) as caught:
                parse_payout_layout(wager_type=wager, slots=(slot(),))
            self.assertNotIn("PRIVATE_WAGER_VALUE", str(caught.exception))

    def test_every_field_rejects_null_empty_wrong_width_and_types(self):
        defaults = {"a": "01", "b": "000000420", "c": "02"}
        for field, width in (("a", 2), ("b", 9), ("c", 2)):
            for bad in (None, "", "0"*(width-1), "0"*(width+1), 0, 1.0, True, b"01"):
                values = {**defaults, field: bad}
                with self.subTest(field=field), self.assertRaises(PayoutDecodingError):
                    parse_payout_layout(wager_type="tansho", slots=(RawPayoutSlot(**values),))

    def test_non_ascii_control_sign_and_space_are_not_corrected(self):
        for value in (" 1", "1 ", "１1", "0\n", "0\x00", "+1", "-1", "1."):
            with self.assertRaises(PayoutDecodingError):
                parse_payout_layout(wager_type="tansho", slots=(slot(a=value),))
        for value in ("0000004.2", "+00000420", " 00000420", "００００００４２０"):
            with self.assertRaises(PayoutDecodingError):
                parse_payout_layout(wager_type="tansho", slots=(slot(b=value),))

    def test_all_zero_slot_is_never_dropped_as_unused_or_zero_yen(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(slot("00", "000000000", "00"),))
        self.assertEqual(len(parsed.slots), 1)
        with self.assertRaises(PayoutDecodingError) as caught:
            decode_hr_payouts(parsed)
        self.assertEqual(caught.exception.reason, "payout_special_values_unconfirmed")

    def test_zero_a_amount_or_popularity_are_confirmation_gates(self):
        for value in (slot(a="00"), slot(b="000000000"), slot(c="00")):
            parsed = parse_payout_layout(wager_type="fukusho", slots=(value,))
            with self.assertRaises(PayoutDecodingError) as caught:
                decode_hr_payouts(parsed)
            self.assertEqual(caught.exception.reason, "payout_special_values_unconfirmed")

    def test_duplicate_raw_group_is_detected(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(slot(), slot()))
        with self.assertRaises(PayoutDecodingError) as caught:
            decode_hr_payouts(parsed)
        self.assertEqual(caught.exception.reason, "duplicate_raw_slot")

    def test_repeated_candidate_target_with_different_payload_stops(self):
        parsed = parse_payout_layout(wager_type="fukusho", slots=(slot(), slot(b="000000430")))
        with self.assertRaises(PayoutDecodingError) as caught:
            decode_hr_payouts(parsed)
        self.assertEqual(caught.exception.reason, "duplicate_a_requires_confirmation")

    def test_a_29_or_99_is_not_silently_promoted_to_a_winning_horse(self):
        for candidate in ("29", "99"):
            parsed = parse_payout_layout(wager_type="tansho", slots=(slot(a=candidate),))
            with self.assertRaises(PayoutDecodingError):
                decode_hr_payouts(parsed)

    def test_gate_cannot_be_bypassed_with_declared_unit_or_confirmation_bool(self):
        parsed = parse_payout_layout(wager_type="tansho", slots=(slot(),))
        with self.assertRaises(TypeError):
            decode_hr_payouts(parsed, assume_confirmed=True)
        with self.assertRaises(PayoutDecodingError):
            decode_hr_payouts(parsed)
        self.assertIn("currency_and_base_stake", UNCONFIRMED_HR_RULES)
        self.assertIn("correction_and_deletion_selection", UNCONFIRMED_HR_RULES)

    def test_decoder_does_not_accept_a_record_or_arbitrary_object(self):
        for value in (None, {"PRIVATE_RECORD": "PRIVATE_VALUE"}, slot(), (slot(),)):
            with self.assertRaises(PayoutDecodingError) as caught:
                decode_hr_payouts(value)
            self.assertEqual(caught.exception.reason, "invalid_parsed_layout")
            self.assertNotIn("PRIVATE_VALUE", str(caught.exception))

    def test_repr_and_errors_do_not_retain_source_values(self):
        raw = slot(b="SECRETXYZ")
        self.assertNotIn("SECRETXYZ", repr(raw))
        with self.assertRaises(PayoutDecodingError) as caught:
            parse_payout_layout(wager_type="tansho", slots=(raw,))
        error = caught.exception
        self.assertIsInstance(error, SafeAuditError)
        self.assertNotIn("SECRETXYZ", str(error)+repr(error)+json.dumps(error.diagnostic())+json.dumps(vars(error)))
        parsed = parse_payout_layout(wager_type="tansho", slots=(slot(b="987654321"),))
        self.assertNotIn("987654321", repr(parsed))

    def test_layout_is_immutable_and_has_no_io_side_effects(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("jra_racing_ai.db_connection.read_only_session") as db, \
                patch.object(Path, "mkdir") as mkdir, patch("builtins.open") as opened, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            parsed = parse_payout_layout(wager_type="tansho", slots=(slot(),))
            with self.assertRaises(PayoutDecodingError):
                decode_hr_payouts(parsed)
            with self.assertRaises(FrozenInstanceError):
                parsed.wager_type = "fukusho"
            db.assert_not_called()
            mkdir.assert_not_called()
            opened.assert_not_called()
        self.assertEqual(stdout.getvalue()+stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
