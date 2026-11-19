"""Fictitious normalized payouts only; all prices and identities are invented."""
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.payout_decoder import RawPayoutSlot, parse_payout_layout
from jra_racing_ai.refunds import RefundValidationError, determine_refund_targets
from jra_racing_ai.settlement import (
    DataOrigin, JPY_PER_100, JST, NormalizedPayout, OutcomeState, PayoutTarget,
    SettlementResult, SettlementStatus, SettlementValidationError, VirtualBet,
    settle_virtual_bet, summarize_settlements,
)


START = datetime(2025, 1, 5, 12, 0, tzinfo=JST)


def refund_targets(*numbers):
    bits = "".join("1" if n in numbers else "0" for n in range(1, 29))
    flag = "1" if numbers else "0"
    return determine_refund_targets(henkan_umaban_joho=bits, henkan_flag_tansho=flag, henkan_flag_fukusho=flag)


def bet(**changes):
    values = dict(ticket_ref="fictional-ticket-1", race_ref="fictional-race-1", wager_type="tansho",
                  horse_number="01", stake_yen=100, decided_at=START-timedelta(minutes=5))
    return VirtualBet(**(values | changes))


def payout(**changes):
    values = dict(race_ref="fictional-race-1", wager_type="tansho", race_date=date(2025, 1, 5),
                  state=OutcomeState.NORMAL, origin=DataOrigin.SYNTHETIC, amount_basis=JPY_PER_100,
                  race_start_at=START, finalized_at=START+timedelta(minutes=10), revision_state="final",
                  targets=(PayoutTarget("01", 420),), targets_complete=True, refund_targets=refund_targets())
    return NormalizedPayout(**(values | changes))


class VirtualBetContractTests(unittest.TestCase):
    def test_stakes_are_positive_exact_integer_100_yen_units(self):
        for stake in (100, 200, 10000):
            self.assertEqual(bet(stake_yen=stake).stake_yen, stake)
        for invalid in (0, -100, 1, 99, 101, 150, 100.0, True, "100", Decimal("100"), None):
            with self.assertRaises(SettlementValidationError) as caught:
                bet(stake_yen=invalid)
            self.assertEqual(caught.exception.reason, "invalid_stake")

    def test_wager_and_horse_identifiers_are_strict(self):
        for wager in ("win", "place", "単勝", None, True):
            with self.assertRaises(SettlementValidationError):
                bet(wager_type=wager)
        for number in ("00", "29", "1", "０１", "01 ", 1, True, None):
            with self.assertRaises(SettlementValidationError):
                bet(horse_number=number)

    def test_ticket_and_race_identity_are_required_without_echo(self):
        for name in ("ticket_ref", "race_ref"):
            for value in ("", None, "PRIVATE ID WITH SPACE", "x"*129, "PRIVATE\nVALUE"):
                with self.assertRaises(SettlementValidationError) as caught:
                    bet(**{name: value})
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_decision_time_must_be_timezone_aware(self):
        for value in (None, "2025-01-05", START.replace(tzinfo=None), date(2025, 1, 5)):
            with self.assertRaises(SettlementValidationError):
                bet(decided_at=value)

    def test_target_amounts_are_positive_integers_without_coercion(self):
        for value in (0, -1, 420.0, "420", True, Decimal("420"), None):
            with self.assertRaises(SettlementValidationError):
                PayoutTarget("01", value)
        with self.assertRaises(SettlementValidationError):
            PayoutTarget("01", 420, 1)

    def test_normalized_origin_and_state_are_explicit_enums(self):
        for changes in ({"origin": "synthetic"}, {"state": "normal"}, {"origin": "hr_confirmed"}):
            with self.assertRaises(SettlementValidationError):
                payout(**changes)

    def test_normalized_containers_and_required_date_are_strict(self):
        for changes in ({"targets": []}, {"targets": ({"horse": "01"},)}, {"targets_complete": 1},
                        {"race_date": "2025-01-05"}, {"race_date": START}, {"refund_targets": ()}):
            with self.assertRaises(SettlementValidationError):
                payout(**changes)


class NormalSettlementTests(unittest.TestCase):
    def test_normal_win_uses_exact_supplied_official_amount(self):
        result = settle_virtual_bet(bet(), payout())
        self.assertEqual(result.status, SettlementStatus.NORMAL_HIT)
        self.assertEqual((result.invested_yen, result.returned_yen, result.net_profit_yen), (100, 420, 320))
        self.assertIsNone(result.reason_code)

    def test_multi_unit_purchase_scales_by_integer_units(self):
        result = settle_virtual_bet(bet(stake_yen=300), payout())
        self.assertEqual((result.returned_yen, result.net_profit_yen), (1260, 960))

    def test_normal_miss_is_the_only_zero_payout_loss_case(self):
        result = settle_virtual_bet(bet(horse_number="02", stake_yen=200), payout())
        self.assertEqual(result.status, SettlementStatus.NORMAL_MISS)
        self.assertEqual((result.returned_yen, result.net_profit_yen), (0, -200))

    def test_place_can_have_multiple_ordinary_targets_without_dead_heat(self):
        targets = (PayoutTarget("01", 180), PayoutTarget("02", 220), PayoutTarget("03", 310))
        for number, price in (("01", 180), ("02", 220), ("03", 310)):
            result = settle_virtual_bet(bet(wager_type="fukusho", horse_number=number), payout(wager_type="fukusho", targets=targets))
            self.assertEqual(result.status, SettlementStatus.NORMAL_HIT)
            self.assertEqual(result.returned_yen, price)

    def test_two_place_targets_are_not_forced_into_top_three(self):
        outcome = payout(wager_type="fukusho", targets=(PayoutTarget("01", 180), PayoutTarget("02", 220)))
        result = settle_virtual_bet(bet(wager_type="fukusho", horse_number="03"), outcome)
        self.assertEqual(result.status, SettlementStatus.NORMAL_MISS)

    def test_win_dead_heat_retains_three_distinct_targets_and_amounts(self):
        targets = (PayoutTarget("01", 220, True), PayoutTarget("04", 310, True), PayoutTarget("08", 540, True))
        for target in targets:
            result = settle_virtual_bet(bet(horse_number=target.horse_number), payout(targets=targets))
            self.assertEqual(result.status, SettlementStatus.DEAD_HEAT_HIT)
            self.assertEqual(result.returned_yen, target.payout_yen_per_100)

    def test_place_dead_heat_is_marked_per_target_not_all_horses(self):
        targets = (PayoutTarget("01", 150), PayoutTarget("02", 170), PayoutTarget("03", 180),
                   PayoutTarget("04", 210, True), PayoutTarget("05", 240, True))
        for number, expected in (("01", SettlementStatus.NORMAL_HIT), ("04", SettlementStatus.DEAD_HEAT_HIT),
                                 ("05", SettlementStatus.DEAD_HEAT_HIT)):
            result = settle_virtual_bet(bet(wager_type="fukusho", horse_number=number), payout(wager_type="fukusho", targets=targets))
            self.assertEqual(result.status, expected)

    def test_exact_integer_arithmetic_handles_large_values_without_float(self):
        result = settle_virtual_bet(bet(stake_yen=123456700), payout(targets=(PayoutTarget("01", 999999999),)))
        self.assertEqual(result.returned_yen, 1234567*999999999)
        self.assertIs(type(result.returned_yen), int)

    def test_no_odds_or_prediction_input_is_accepted(self):
        with self.assertRaises(TypeError):
            settle_virtual_bet(bet(), payout(), final_odds=4.2)
        with self.assertRaises(TypeError):
            VirtualBet(**{**vars_dict(bet()), "predicted_probability": 0.4})


def vars_dict(ticket):
    return {name: getattr(ticket, name) for name in ("ticket_ref", "race_ref", "wager_type", "horse_number", "stake_yen", "decided_at")}


class RefundAndBlockedStateTests(unittest.TestCase):
    def test_refund_uses_existing_target_validation_and_returns_stake(self):
        for wager in ("tansho", "fukusho"):
            result = settle_virtual_bet(bet(wager_type=wager, horse_number="02", stake_yen=500),
                                        payout(wager_type=wager, refund_targets=refund_targets(2, 7)))
            self.assertEqual(result.status, SettlementStatus.REFUND)
            self.assertEqual((result.returned_yen, result.net_profit_yen), (500, 0))

    def test_non_refunded_horse_uses_payout_target_membership(self):
        result = settle_virtual_bet(bet(), payout(refund_targets=refund_targets(2)))
        self.assertEqual(result.status, SettlementStatus.NORMAL_HIT)

    def test_conflicting_refund_and_winning_target_is_invalid(self):
        result = settle_virtual_bet(bet(), payout(refund_targets=refund_targets(1)))
        self.assertEqual(result.status, SettlementStatus.INVALID)
        self.assertEqual(result.reason_code, "refund_payout_overlap")
        self.assertIsNone(result.returned_yen)

    def test_missing_refund_validation_does_not_assume_no_refund(self):
        result = settle_virtual_bet(bet(), payout(refund_targets=None))
        self.assertEqual(result.reason_code, "refund_targets_unconfirmed")
        self.assertIsNone(result.net_profit_yen)

    def test_mixed_refund_flags_still_stop_in_phase12_validator(self):
        with self.assertRaises(RefundValidationError):
            determine_refund_targets(henkan_umaban_joho="1"+"0"*27, henkan_flag_tansho="1", henkan_flag_fukusho="0")

    def test_special_not_established_not_offered_and_cancelled_are_distinct_unresolved_states(self):
        cases = ((OutcomeState.SPECIAL_PAYMENT, SettlementStatus.SPECIAL_PAYMENT),
                 (OutcomeState.NOT_ESTABLISHED, SettlementStatus.NOT_ESTABLISHED),
                 (OutcomeState.NOT_OFFERED, SettlementStatus.NOT_OFFERED),
                 (OutcomeState.CANCELLED, SettlementStatus.CANCELLED),
                 (OutcomeState.UNCONFIRMED, SettlementStatus.UNCONFIRMED),
                 (OutcomeState.INVALID, SettlementStatus.INVALID))
        reasons = set()
        for state, expected in cases:
            result = settle_virtual_bet(bet(), payout(state=state))
            self.assertEqual(result.status, expected)
            self.assertIsNone(result.returned_yen)
            self.assertIsNone(result.net_profit_yen)
            self.assertFalse(result.is_settled)
            reasons.add(result.reason_code)
        self.assertEqual(len(reasons), len(cases))

    def test_special_states_do_not_implicitly_refund_even_with_refund_bitmap(self):
        for state in (OutcomeState.SPECIAL_PAYMENT, OutcomeState.NOT_ESTABLISHED, OutcomeState.NOT_OFFERED, OutcomeState.CANCELLED):
            result = settle_virtual_bet(bet(horse_number="02"), payout(state=state, refund_targets=refund_targets(2)))
            self.assertNotEqual(result.status, SettlementStatus.REFUND)
            self.assertIsNone(result.returned_yen)

    def test_real_hr_is_blocked_even_with_normal_state_and_declared_100_yen_unit(self):
        result = settle_virtual_bet(bet(), payout(origin=DataOrigin.HR_UNCONFIRMED))
        self.assertEqual(result.reason_code, "hr_mapping_unconfirmed")
        self.assertIsNone(result.returned_yen)
        self.assertIsNone(result.net_profit_yen)

    def test_default_normalized_origin_is_unconfirmed_not_synthetic(self):
        value = NormalizedPayout(race_ref="fictional-race-1", wager_type="tansho", race_date=date(2025, 1, 5))
        self.assertEqual(value.origin, DataOrigin.HR_UNCONFIRMED)
        result = settle_virtual_bet(bet(), value)
        self.assertEqual(result.reason_code, "hr_mapping_unconfirmed")
        self.assertIsNone(result.returned_yen)

    def test_real_hr_refund_is_also_blocked(self):
        result = settle_virtual_bet(bet(horse_number="02"), payout(origin=DataOrigin.HR_UNCONFIRMED, refund_targets=refund_targets(2)))
        self.assertEqual(result.status, SettlementStatus.UNCONFIRMED)
        self.assertIsNone(result.returned_yen)

    def test_unknown_unit_stops_hit_miss_and_refund(self):
        for amount_basis in (None, "JPY_PER_1", "PRIVATE_UNKNOWN_UNIT"):
            for horse, refunds in (("01", refund_targets()), ("03", refund_targets()), ("02", refund_targets(2))):
                result = settle_virtual_bet(bet(horse_number=horse), payout(amount_basis=amount_basis, refund_targets=refunds))
                self.assertEqual(result.reason_code, "unit_unconfirmed")
                self.assertIsNone(result.returned_yen)

    def test_incomplete_target_list_never_creates_a_miss(self):
        for number in ("01", "02"):
            result = settle_virtual_bet(bet(horse_number=number), payout(targets_complete=False))
            self.assertEqual(result.reason_code, "incomplete_winning_targets")
            self.assertIsNone(result.net_profit_yen)

    def test_empty_target_list_does_not_turn_all_horses_into_losses(self):
        result = settle_virtual_bet(bet(), payout(targets=()))
        self.assertEqual(result.reason_code, "missing_winning_targets")
        self.assertIsNone(result.returned_yen)

    def test_deleted_conflicting_preliminary_or_corrected_unconfirmed_versions_are_blocked(self):
        for revision in ("unconfirmed", "deleted", "conflicting_versions", "correction_unconfirmed"):
            result = settle_virtual_bet(bet(), payout(revision_state=revision))
            self.assertEqual(result.reason_code, "revision_unconfirmed")
            self.assertIsNone(result.net_profit_yen)


class SettlementConsistencyTests(unittest.TestCase):
    def test_race_or_wager_mismatch_does_not_settle(self):
        for changes, reason in (({"race_ref": "different-fictional-race"}, "race_mismatch"),
                                ({"wager_type": "fukusho"}, "wager_mismatch")):
            result = settle_virtual_bet(bet(), payout(**changes))
            self.assertEqual(result.reason_code, reason)
            self.assertIsNone(result.returned_yen)
            if reason == "race_mismatch":
                self.assertIsNone(result.race_month_jst)

    def test_duplicate_and_too_many_targets_are_invalid(self):
        result = settle_virtual_bet(bet(), payout(targets=(PayoutTarget("01", 420), PayoutTarget("01", 430))))
        self.assertEqual(result.reason_code, "duplicate_target")
        for wager, count in (("tansho", 4), ("fukusho", 6)):
            targets = tuple(PayoutTarget(f"{n:02d}", 420, True) for n in range(1, count+1))
            result = settle_virtual_bet(bet(wager_type=wager), payout(wager_type=wager, targets=targets))
            self.assertEqual(result.reason_code, "too_many_targets")

    def test_win_multiple_targets_require_explicit_consistent_dead_heat(self):
        for targets in ((PayoutTarget("01", 420), PayoutTarget("02", 430)),
                        (PayoutTarget("01", 420, True), PayoutTarget("02", 430)),
                        (PayoutTarget("01", 420, True),)):
            result = settle_virtual_bet(bet(), payout(targets=targets))
            self.assertEqual(result.reason_code, "invalid_dead_heat_targets")

    def test_missing_times_and_bad_timing_stop(self):
        for changes in ({"race_start_at": None}, {"finalized_at": None}):
            self.assertEqual(settle_virtual_bet(bet(), payout(**changes)).reason_code, "timing_unconfirmed")
        self.assertEqual(settle_virtual_bet(bet(decided_at=START), payout()).reason_code, "purchase_not_before_race")
        self.assertEqual(settle_virtual_bet(bet(decided_at=START+timedelta(minutes=1)), payout()).reason_code, "purchase_not_before_race")
        self.assertEqual(settle_virtual_bet(bet(), payout(finalized_at=START-timedelta(seconds=1))).reason_code, "invalid_result_timing")

    def test_race_date_must_match_jst_and_timezone_equivalence_is_preserved(self):
        equivalent_utc = START.astimezone(timezone.utc)
        result = settle_virtual_bet(bet(), payout(race_start_at=equivalent_utc))
        self.assertEqual(result.status, SettlementStatus.NORMAL_HIT)
        invalid = settle_virtual_bet(bet(), payout(race_date=date(2025, 1, 6)))
        self.assertEqual(invalid.reason_code, "race_date_mismatch")
        self.assertIsNone(invalid.race_month_jst)

    def test_out_of_range_timezone_conversion_is_safely_blocked(self):
        extreme = datetime(9999, 12, 31, 23, tzinfo=timezone.utc)
        result = settle_virtual_bet(bet(), payout(race_start_at=extreme, finalized_at=extreme, race_date=date(9999, 12, 31)))
        self.assertEqual(result.reason_code, "invalid_timestamp")
        self.assertIsNone(result.returned_yen)
        self.assertIsNone(result.race_month_jst)

    def test_raw_or_layout_input_cannot_be_sent_directly_to_settlement(self):
        layout = parse_payout_layout(wager_type="tansho", slots=(RawPayoutSlot("01", "000000420", "02"),))
        for value in (layout, {"PRIVATE_RAW_HR": "PRIVATE_VALUE"}, None):
            with self.assertRaises(SettlementValidationError) as caught:
                settle_virtual_bet(bet(), value)
            self.assertNotIn("PRIVATE_VALUE", str(caught.exception))


class SettlementAggregationAndSafetyTests(unittest.TestCase):
    def test_complete_batch_totals_use_all_tickets(self):
        hit = settle_virtual_bet(bet(), payout())
        miss = settle_virtual_bet(bet(ticket_ref="fictional-2", horse_number="03", stake_yen=200), payout())
        refund = settle_virtual_bet(bet(ticket_ref="fictional-3", horse_number="02"), payout(refund_targets=refund_targets(2)))
        totals = summarize_settlements((hit, miss, refund))
        self.assertEqual((totals.ticket_count, totals.settled_count, totals.unresolved_count), (3, 3, 0))
        self.assertEqual((totals.invested_yen, totals.returned_yen, totals.net_profit_yen), (400, 520, 120))

    def test_unresolved_ticket_prevents_a_false_complete_profit_total(self):
        hit = settle_virtual_bet(bet(), payout())
        blocked = settle_virtual_bet(bet(ticket_ref="fictional-2", stake_yen=300), payout(state=OutcomeState.SPECIAL_PAYMENT))
        totals = summarize_settlements((hit, blocked))
        self.assertEqual((totals.settled_count, totals.unresolved_count, totals.unresolved_invested_yen), (1, 1, 300))
        self.assertIsNone(totals.returned_yen)
        self.assertIsNone(totals.net_profit_yen)

    def test_duplicate_ticket_ids_and_invalid_aggregation_types_stop(self):
        result = settle_virtual_bet(bet(), payout())
        with self.assertRaises(SettlementValidationError) as caught:
            summarize_settlements((result, result))
        self.assertEqual(caught.exception.reason, "duplicate_ticket")
        for value in ([result], (None,), None):
            with self.assertRaises(SettlementValidationError):
                summarize_settlements(value)

    def test_empty_batch_has_no_invented_return_or_profit(self):
        totals = summarize_settlements(())
        self.assertEqual(totals.invested_yen, 0)
        self.assertIsNone(totals.returned_yen)

    def test_month_contract_uses_race_date_not_purchase_month(self):
        first = datetime(2025, 2, 1, 0, 2, tzinfo=JST)
        result = settle_virtual_bet(bet(decided_at=first-timedelta(minutes=5)),
                                   payout(race_start_at=first, finalized_at=first+timedelta(minutes=10), race_date=date(2025, 2, 1)))
        self.assertEqual(result.race_month_jst, "2025-02")

    def test_result_contract_rejects_forged_money_or_unknown_reason(self):
        result = settle_virtual_bet(bet(), payout())
        with self.assertRaises(SettlementValidationError):
            replace(result, net_profit_yen=0)
        with self.assertRaises(SettlementValidationError):
            replace(result, race_month_jst=None)
        with self.assertRaises(SettlementValidationError):
            replace(result, status=SettlementStatus.UNCONFIRMED, returned_yen=None, net_profit_yen=None, reason_code="PRIVATE_RAW_REASON")

    def test_immutability_and_redacted_reprs(self):
        ticket = bet(ticket_ref="SYNTHETIC_PRIVATE_TICKET")
        outcome = payout(targets=(PayoutTarget("01", 987654321),))
        result = settle_virtual_bet(ticket, outcome)
        for value in (ticket, outcome, outcome.targets[0], result, summarize_settlements((result,))):
            self.assertNotIn("SYNTHETIC_PRIVATE_TICKET", repr(value))
            self.assertNotIn("987654321", repr(value))
        with self.assertRaises(FrozenInstanceError):
            result.returned_yen = 0

    def test_errors_do_not_echo_raw_input_or_arbitrary_reason(self):
        with self.assertRaises(SettlementValidationError) as caught:
            bet(stake_yen="PRIVATE_STAKE_VALUE")
        self.assertIsInstance(caught.exception, SafeAuditError)
        self.assertNotIn("PRIVATE_STAKE_VALUE", str(caught.exception)+json.dumps(caught.exception.diagnostic()))
        error = SettlementValidationError("PRIVATE_RAW_REASON")
        self.assertNotIn("PRIVATE_RAW_REASON", str(error)+repr(error.diagnostic()))

    def test_engine_has_no_database_file_logging_or_network_path(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        ticket, outcome = bet(), payout()
        with patch("jra_racing_ai.db_connection.read_only_session") as db, patch("builtins.open") as opened, \
                patch.object(Path, "mkdir") as mkdir, redirect_stdout(stdout), redirect_stderr(stderr):
            result = settle_virtual_bet(ticket, outcome)
            summarize_settlements((result,))
            db.assert_not_called()
            opened.assert_not_called()
            mkdir.assert_not_called()
        self.assertEqual(stdout.getvalue()+stderr.getvalue(), "")

    def test_existing_report_generator_does_not_auto_export_financial_objects(self):
        from jra_racing_ai.generate_report import PROJECT_ROOT, build_report, write_report
        report = build_report({"tables": []}, {"settlement": settle_virtual_bet(bet(), payout())}, source="synthetic")
        with patch.object(Path, "mkdir") as mkdir, self.assertRaises(SafeAuditError):
            write_report(report, PROJECT_ROOT/"reports")
        mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
