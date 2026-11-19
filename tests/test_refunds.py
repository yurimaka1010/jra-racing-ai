"""Fully fictitious refund bits/flags; no database driver or real records."""
from dataclasses import FrozenInstanceError, asdict
import importlib
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.payout_policy import describe_case
from jra_racing_ai.refunds import RefundTargets, RefundValidationError, determine_refund_targets


def bitmap(*numbers):
    # Fixture builder only. The production parser never builds/repairs input.
    return "".join("1" if number in numbers else "0" for number in range(1, 29))


def determine(bits=None, win="1", place="1"):
    return determine_refund_targets(henkan_umaban_joho=bitmap(1, 12, 28) if bits is None else bits,
                                    henkan_flag_tansho=win, henkan_flag_fukusho=place)


class RefundBitmapTests(unittest.TestCase):
    def test_no_refund_requires_valid_zero_bitmap_and_zero_flags(self):
        result = determine(bitmap(), "0", "0")
        self.assertEqual(result.tansho_umaban, ())
        self.assertEqual(result.fukusho_umaban, ())
        self.assertFalse(result.is_refunded(wager_type="tansho", horse_number="01"))
        self.assertFalse(result.is_refunded(wager_type="fukusho", horse_number="28"))

    def test_every_single_position_maps_to_exact_one_based_horse_number(self):
        for number in range(1, 29):
            with self.subTest(position=number):
                result = determine(bitmap(number))
                expected = (f"{number:02d}",)
                self.assertEqual(result.tansho_umaban, expected)
                self.assertEqual(result.fukusho_umaban, expected)
                for horse in range(1, 29):
                    for wager in ("tansho", "fukusho"):
                        self.assertEqual(result.is_refunded(wager_type=wager, horse_number=f"{horse:02d}"), horse == number)

    def test_multiple_targets_include_first_middle_and_last_positions(self):
        result = determine(bitmap(1, 12, 28))
        self.assertEqual(result.tansho_umaban, ("01", "12", "28"))
        self.assertEqual(result.fukusho_umaban, ("01", "12", "28"))
        self.assertFalse(result.is_refunded(wager_type="tansho", horse_number="02"))

    def test_all_28_targets_are_retained(self):
        result = determine("1" * 28)
        self.assertEqual(result.tansho_umaban, tuple(f"{n:02d}" for n in range(1, 29)))
        self.assertEqual(len(result.fukusho_umaban), 28)

    def test_target_order_is_ascending_unique(self):
        result = determine(bitmap(28, 7, 1, 7))
        self.assertEqual(result.tansho_umaban, ("01", "07", "28"))

    def test_null_is_not_interpreted_as_no_refund(self):
        with self.assertRaises(RefundValidationError) as caught:
            determine_refund_targets(henkan_umaban_joho=None, henkan_flag_tansho="0", henkan_flag_fukusho="0")
        self.assertEqual(caught.exception.reason, "missing_bitmap")

    def test_empty_and_blank_values_are_never_padded_or_trimmed(self):
        for value in ("", " " * 28, "\t" * 28, "0" * 27 + " ", " " + "0" * 27):
            with self.subTest(case_length=len(value)), self.assertRaises(RefundValidationError):
                determine(value, "0", "0")

    def test_every_length_except_28_is_rejected(self):
        for length in (0, 1, 2, 27, 29, 30, 56, 10000):
            with self.subTest(length=length), self.assertRaises(RefundValidationError):
                determine("0" * length, "0", "0")

    def test_invalid_ascii_unicode_and_control_characters_are_rejected(self):
        for character in ("2", "9", "x", "０", "１", "\n", "\r", "\x00", "\u200b", "\ufeff"):
            for index in (0, 13, 27):
                with self.subTest(index=index), self.assertRaises(RefundValidationError):
                    determine("0" * index + character + "0" * (27-index), "0", "0")

    def test_non_string_bitmap_is_not_coerced(self):
        for value in (0, 1, True, False, b"0" * 28, list("0" * 28), {"value": "0" * 28}):
            with self.assertRaises(RefundValidationError) as caught:
                determine(value, "0", "0")
            self.assertEqual(caught.exception.reason, "bitmap_type")

    def test_string_subclasses_are_not_implicitly_accepted(self):
        class UntrustedString(str):
            def __str__(self):
                raise AssertionError("must_not_coerce_input")
        with self.assertRaises(RefundValidationError):
            determine(UntrustedString("0" * 28), "0", "0")


class RefundFlagTests(unittest.TestCase):
    def test_exact_integer_and_character_flag_encodings_are_supported(self):
        for win in (1, "1"):
            for place in (1, "1"):
                self.assertEqual(determine(bitmap(3), win, place).tansho_umaban, ("03",))
        for win in (0, "0"):
            for place in (0, "0"):
                self.assertEqual(determine(bitmap(), win, place).tansho_umaban, ())

    def test_null_empty_or_missing_flags_stop(self):
        for bad in (None, ""):
            for win, place in ((bad, "1"), ("1", bad)):
                with self.assertRaises(RefundValidationError):
                    determine(bitmap(1), win, place)
        with self.assertRaises(RefundValidationError):
            determine_refund_targets(henkan_umaban_joho=bitmap(1))

    def test_flags_are_not_trimmed_padded_or_coerced_by_truthiness(self):
        from decimal import Decimal
        for value in (True, False, 1.0, 0.0, Decimal("1"), Decimal("0"), " 1", "1 ", "01", "００", "１", b"1", -1, 2, [], {}):
            for win, place in ((value, "1"), ("1", value)):
                with self.assertRaises(RefundValidationError):
                    determine(bitmap(1), win, place)

    def test_flag_one_without_any_target_is_inconsistent(self):
        for win, place in (("1", "1"), ("1", "0"), ("0", "1")):
            with self.assertRaises(RefundValidationError) as caught:
                determine(bitmap(), win, place)
            self.assertEqual(caught.exception.reason, "flag_without_targets")
            self.assertEqual(caught.exception.status, "needs_confirmation")

    def test_targets_with_both_flags_zero_are_inconsistent(self):
        with self.assertRaises(RefundValidationError) as caught:
            determine(bitmap(1, 7, 28), "0", "0")
        self.assertEqual(caught.exception.reason, "targets_without_flag")

    def test_mixed_wager_flags_stop_until_common_bitmap_applicability_is_confirmed(self):
        for win, place in (("0", "1"), ("1", "0")):
            with self.assertRaises(RefundValidationError) as caught:
                determine(bitmap(3, 9), win, place)
            self.assertEqual(caught.exception.reason, "mixed_wager_flags_unconfirmed")
            self.assertEqual(caught.exception.status, "needs_confirmation")


class RefundQueryAndSafetyTests(unittest.TestCase):
    def test_horse_number_requires_exact_ascii_01_through_28(self):
        result = determine(bitmap(1, 28))
        for value in (None, "", "00", "29", "99", "1", "001", " 01", "01 ", "０１", "01\n", 1, 28, True, b"01"):
            with self.assertRaises(RefundValidationError) as caught:
                result.is_refunded(wager_type="tansho", horse_number=value)
            self.assertEqual(caught.exception.reason, "invalid_horse_number")

    def test_supported_wager_type_is_explicit(self):
        result = determine()
        for wager in (None, "", "win", "place", "単勝", "複勝", "tansho ", "TANSHO", 1, True):
            with self.assertRaises(RefundValidationError) as caught:
                result.is_refunded(wager_type=wager, horse_number="01")
            self.assertEqual(caught.exception.reason, "unsupported_wager_type")

    def test_manually_created_target_sets_cannot_bypass_range_or_consistency(self):
        invalid = (("00",), ("29",), ("1",), (1,), ("01", "01"), ("28", "01"), ["01"])
        for collection in invalid:
            with self.assertRaises(RefundValidationError):
                RefundTargets(collection, collection)
        with self.assertRaises(RefundValidationError):
            RefundTargets(("01",), ())

    def test_result_is_immutable_and_repr_hides_targets(self):
        result = determine(bitmap(1, 12, 28))
        with self.assertRaises(FrozenInstanceError):
            result.tansho_umaban = ()
        self.assertNotIn("01", repr(result))
        self.assertNotIn("12", repr(result))
        self.assertNotIn("28", repr(result))

    def test_exception_diagnostics_do_not_include_source_values(self):
        inputs = {
            "henkan_umaban_joho": "SYNTHETIC_PRIVATE_BITMAP_SECRET",
            "henkan_flag_tansho": "SYNTHETIC_PRIVATE_FLAG_SECRET",
            "henkan_flag_fukusho": "1",
        }
        with self.assertRaises(RefundValidationError) as caught:
            determine_refund_targets(**inputs)
        exc = caught.exception
        self.assertIsInstance(exc, SafeAuditError)
        self.assertEqual(exc.code, "refund_needs_confirmation")
        text = str(exc) + repr(exc) + json.dumps(exc.diagnostic()) + json.dumps(vars(exc))
        for value in inputs.values():
            if value.startswith("SYNTHETIC_PRIVATE"):
                self.assertNotIn(value, text)
        self.assertEqual(set(exc.diagnostic()), {"status", "code", "reason", "field"})

    def test_invalid_horse_value_is_never_echoed(self):
        with self.assertRaises(RefundValidationError) as caught:
            determine().is_refunded(wager_type="fukusho", horse_number="SYNTHETIC_PRIVATE_HORSE")
        self.assertNotIn("SYNTHETIC_PRIVATE_HORSE", str(caught.exception) + repr(caught.exception.diagnostic()))

    def test_invalid_flag_values_are_not_retained_in_diagnostics(self):
        for win, place in (("SYNTHETIC_PRIVATE_FLAG", "1"), ("1", "SYNTHETIC_PRIVATE_FLAG")):
            with self.assertRaises(RefundValidationError) as caught:
                determine(bitmap(1, 12, 28), win, place)
            error = caught.exception
            self.assertEqual(error.reason, "invalid_flag")
            self.assertNotIn("SYNTHETIC_PRIVATE_FLAG", str(error) + json.dumps(error.diagnostic()) + json.dumps(vars(error)))

    def test_no_io_logging_or_database_side_effects_on_import_and_execution(self):
        output, errors = io.StringIO(), io.StringIO()
        name = "jra_racing_ai._refund_import_safety_test"
        path = Path(__file__).resolve().parents[1] / "src" / "jra_racing_ai" / "refunds.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with patch("jra_racing_ai.db_connection.read_only_session") as session, \
                patch.object(Path, "mkdir") as mkdir, \
                patch("builtins.open") as file_open, \
                patch.dict(sys.modules, {name: module}), \
                redirect_stdout(output), redirect_stderr(errors):
            spec.loader.exec_module(module)
            result = module.determine_refund_targets(henkan_umaban_joho=bitmap(1),
                                                     henkan_flag_tansho="1", henkan_flag_fukusho="1")
            self.assertTrue(result.is_refunded(wager_type="tansho", horse_number="01"))
            session.assert_not_called()
            mkdir.assert_not_called()
            file_open.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(errors.getvalue(), "")

    def test_unknown_error_metadata_is_replaced_with_fixed_diagnostics(self):
        for reason, field in (("SYNTHETIC_PRIVATE_REASON", "SYNTHETIC_PRIVATE_FIELD"), ({}, [])):
            error = RefundValidationError(reason, field)
            self.assertEqual(error.reason, "unconfirmed_refund_information")
            self.assertEqual(error.field, "targets")
            self.assertNotIn("SYNTHETIC_PRIVATE", str(error) + repr(error.diagnostic()))

    def test_refund_target_decision_does_not_classify_other_settlement_cases(self):
        result = determine(bitmap(2, 17))
        self.assertEqual(set(asdict(result)), {"tansho_umaban", "fukusho_umaban"})
        for case in ("normal_win_hit", "normal_place_hit", "special_payment", "not_established", "not_offered"):
            self.assertEqual(describe_case(case)["profit_calculation"], "not_implemented")
            self.assertEqual(describe_case(case)["source_mapping_status"], "needs_confirmation")

    def test_existing_report_generator_does_not_automatically_export_target_object(self):
        from jra_racing_ai.generate_report import PROJECT_ROOT, build_report, write_report
        report = build_report({"tables": []}, {"refunds": determine(bitmap(1, 12, 28))}, source="synthetic")
        with patch.object(Path, "mkdir") as mkdir, self.assertRaises(SafeAuditError) as caught:
            write_report(report, PROJECT_ROOT / "reports")
        self.assertEqual(caught.exception.code, "internal_error")
        mkdir.assert_not_called()

    def test_existing_cli_can_redact_refund_validation_errors(self):
        from jra_racing_ai import cli
        error = RefundValidationError("mixed_wager_flags_unconfirmed", "wager_flags")
        with patch("jra_racing_ai.cli.run", side_effect=error), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["demo"]), 2)
        self.assertIn("refund_needs_confirmation", output.getvalue())
        self.assertIn("要確認", output.getvalue())
        self.assertNotIn("henkan_umaban_joho", output.getvalue())


if __name__ == "__main__":
    unittest.main()
