"""Pure, in-memory settlement of explicitly normalized fictitious tickets.

Real HR input is blocked until its semantic/monetary/version mapping is verified.
No odds, predictions, database driver, files, logger, or network is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum

from .core import SafeAuditError
from .refunds import RefundTargets

JPY_PER_100 = "JPY_PER_100"
JST = timezone(timedelta(hours=9))
_WAGERS = ("tansho", "fukusho")
_HORSES = tuple(f"{number:02d}" for number in range(1, 29))
_MAX_TARGETS = {"tansho": 3, "fukusho": 5}


class DataOrigin(StrEnum):
    SYNTHETIC = "synthetic"
    HR_UNCONFIRMED = "hr_unconfirmed"


class OutcomeState(StrEnum):
    NORMAL = "normal"
    SPECIAL_PAYMENT = "special_payment"
    NOT_ESTABLISHED = "not_established"
    NOT_OFFERED = "not_offered"
    CANCELLED = "cancelled"
    UNCONFIRMED = "unconfirmed"
    INVALID = "invalid"


class SettlementStatus(StrEnum):
    NORMAL_HIT = "normal_hit"
    NORMAL_MISS = "normal_miss"
    DEAD_HEAT_HIT = "dead_heat_hit"
    REFUND = "refund"
    SPECIAL_PAYMENT = "special_payment"
    NOT_ESTABLISHED = "not_established"
    NOT_OFFERED = "not_offered"
    CANCELLED = "cancelled"
    UNCONFIRMED = "data_unconfirmed"
    INVALID = "data_invalid"


_SETTLED = frozenset({SettlementStatus.NORMAL_HIT, SettlementStatus.NORMAL_MISS,
                       SettlementStatus.DEAD_HEAT_HIT, SettlementStatus.REFUND})
_REASONS = frozenset({
    "invalid_ticket", "invalid_identity", "unsupported_wager_type", "invalid_horse_number",
    "invalid_stake", "invalid_timestamp", "invalid_race_date", "invalid_origin", "invalid_state",
    "invalid_normalized_data", "invalid_payout_target", "invalid_amount", "duplicate_target",
    "too_many_targets", "invalid_dead_heat_targets", "refund_payout_overlap", "race_mismatch",
    "wager_mismatch", "hr_mapping_unconfirmed", "unit_unconfirmed", "revision_unconfirmed",
    "timing_unconfirmed", "purchase_not_before_race", "invalid_result_timing", "race_date_mismatch",
    "special_payment_rule_unconfirmed", "not_established_rule_unconfirmed", "not_offered_rule_unconfirmed",
    "cancelled_rule_unconfirmed", "outcome_unconfirmed", "outcome_invalid", "refund_targets_unconfirmed",
    "incomplete_winning_targets", "missing_winning_targets", "invalid_settlement_result",
    "duplicate_ticket", "empty_aggregation", "invalid_aggregation",
})


class SettlementValidationError(SafeAuditError):
    status = "needs_confirmation"

    def __init__(self, reason: str):
        self.reason = reason if type(reason) is str and reason in _REASONS else "invalid_normalized_data"
        super().__init__("settlement_invalid_input")

    def diagnostic(self) -> dict[str, str]:
        return {"status": self.status, "code": self.code, "reason": self.reason}


def _identity(value):
    if type(value) is not str or not 1 <= len(value) <= 128 or any(not 33 <= ord(c) <= 126 for c in value):
        raise SettlementValidationError("invalid_identity")


def _wager(value):
    if type(value) is not str or value not in _WAGERS:
        raise SettlementValidationError("unsupported_wager_type")


def _horse(value):
    if type(value) is not str or value not in _HORSES:
        raise SettlementValidationError("invalid_horse_number")


def _timestamp(value):
    if type(value) is not datetime or value.tzinfo is None:
        raise SettlementValidationError("invalid_timestamp")
    try:
        if value.utcoffset() is None:
            raise SettlementValidationError("invalid_timestamp")
    except Exception:
        raise SettlementValidationError("invalid_timestamp") from None


@dataclass(frozen=True, slots=True, repr=False)
class VirtualBet:
    ticket_ref: str
    race_ref: str
    wager_type: str
    horse_number: str
    stake_yen: int
    decided_at: datetime

    def __post_init__(self):
        _identity(self.ticket_ref)
        _identity(self.race_ref)
        _wager(self.wager_type)
        _horse(self.horse_number)
        if type(self.stake_yen) is not int or self.stake_yen <= 0 or self.stake_yen % 100:
            raise SettlementValidationError("invalid_stake")
        _timestamp(self.decided_at)

    def __repr__(self):
        return "VirtualBet(details=local_only)"


@dataclass(frozen=True, slots=True, repr=False)
class PayoutTarget:
    horse_number: str
    payout_yen_per_100: int
    is_dead_heat: bool = False

    def __post_init__(self):
        _horse(self.horse_number)
        if type(self.payout_yen_per_100) is not int or self.payout_yen_per_100 <= 0:
            raise SettlementValidationError("invalid_amount")
        if type(self.is_dead_heat) is not bool:
            raise SettlementValidationError("invalid_payout_target")

    def __repr__(self):
        return "PayoutTarget(details=local_only)"


@dataclass(frozen=True, slots=True, repr=False)
class NormalizedPayout:
    race_ref: str
    wager_type: str
    race_date: date
    state: OutcomeState = OutcomeState.UNCONFIRMED
    origin: DataOrigin = DataOrigin.HR_UNCONFIRMED
    amount_basis: str | None = None
    race_start_at: datetime | None = None
    finalized_at: datetime | None = None
    revision_state: str = "unconfirmed"
    targets: tuple[PayoutTarget, ...] = ()
    targets_complete: bool = False
    refund_targets: RefundTargets | None = None

    def __post_init__(self):
        _identity(self.race_ref)
        _wager(self.wager_type)
        if type(self.race_date) is not date:
            raise SettlementValidationError("invalid_race_date")
        if type(self.origin) is not DataOrigin:
            raise SettlementValidationError("invalid_origin")
        if type(self.state) is not OutcomeState:
            raise SettlementValidationError("invalid_state")
        if self.amount_basis is not None and type(self.amount_basis) is not str:
            raise SettlementValidationError("invalid_normalized_data")
        for timestamp in (self.race_start_at, self.finalized_at):
            if timestamp is not None:
                _timestamp(timestamp)
        if type(self.revision_state) is not str or self.revision_state not in (
            "unconfirmed", "final", "deleted", "conflicting_versions", "correction_unconfirmed"):
            raise SettlementValidationError("invalid_normalized_data")
        if type(self.targets) is not tuple or type(self.targets_complete) is not bool:
            raise SettlementValidationError("invalid_normalized_data")
        if any(type(target) is not PayoutTarget for target in self.targets):
            raise SettlementValidationError("invalid_payout_target")
        for target in self.targets:
            target.__post_init__()
        if self.refund_targets is not None and type(self.refund_targets) is not RefundTargets:
            raise SettlementValidationError("invalid_normalized_data")
        if self.refund_targets is not None:
            self.refund_targets.__post_init__()

    def __repr__(self):
        return "NormalizedPayout(details=local_only)"


@dataclass(frozen=True, slots=True, repr=False)
class SettlementResult:
    ticket_ref: str
    race_month_jst: str | None
    status: SettlementStatus
    invested_yen: int
    returned_yen: int | None
    net_profit_yen: int | None
    reason_code: str | None

    def __post_init__(self):
        _identity(self.ticket_ref)
        if (type(self.status) is not SettlementStatus or type(self.invested_yen) is not int
                or self.invested_yen <= 0 or self.invested_yen % 100):
            raise SettlementValidationError("invalid_settlement_result")
        if self.race_month_jst is None:
            if self.status in _SETTLED:
                raise SettlementValidationError("invalid_settlement_result")
        elif (type(self.race_month_jst) is not str or len(self.race_month_jst) != 7
                or self.race_month_jst[4] != "-"
                or any(c not in "0123456789" for c in self.race_month_jst[:4]+self.race_month_jst[5:])
                or not 1 <= int(self.race_month_jst[:4]) <= 9999
                or not 1 <= int(self.race_month_jst[5:]) <= 12):
            raise SettlementValidationError("invalid_settlement_result")
        if self.status in _SETTLED:
            if (type(self.returned_yen) is not int or self.returned_yen < 0
                    or type(self.net_profit_yen) is not int
                    or self.net_profit_yen != self.returned_yen-self.invested_yen
                    or self.reason_code is not None):
                raise SettlementValidationError("invalid_settlement_result")
            if self.status == SettlementStatus.REFUND and self.returned_yen != self.invested_yen:
                raise SettlementValidationError("invalid_settlement_result")
            if self.status == SettlementStatus.NORMAL_MISS and self.returned_yen != 0:
                raise SettlementValidationError("invalid_settlement_result")
            if self.status in (SettlementStatus.NORMAL_HIT, SettlementStatus.DEAD_HEAT_HIT) and not self.returned_yen:
                raise SettlementValidationError("invalid_settlement_result")
        elif (self.returned_yen is not None or self.net_profit_yen is not None
              or type(self.reason_code) is not str or self.reason_code not in _REASONS):
            raise SettlementValidationError("invalid_settlement_result")

    @property
    def is_settled(self):
        return self.status in _SETTLED

    def __repr__(self):
        return "SettlementResult(details=local_only)"


def settle_virtual_bet(bet: VirtualBet, payout: NormalizedPayout) -> SettlementResult:
    """Settle normalized fictional data; all unconfirmed HR sources stay blocked."""
    if type(bet) is not VirtualBet:
        raise SettlementValidationError("invalid_ticket")
    if type(payout) is not NormalizedPayout:
        raise SettlementValidationError("invalid_normalized_data")
    bet.__post_init__()
    payout.__post_init__()
    month = f"{payout.race_date.year:04d}-{payout.race_date.month:02d}"

    def blocked(status, reason):
        month_key = None if reason in ("race_mismatch", "race_date_mismatch", "invalid_timestamp") else month
        return SettlementResult(bet.ticket_ref, month_key, status, bet.stake_yen, None, None, reason)

    def paid(status, amount):
        return SettlementResult(bet.ticket_ref, month, status, bet.stake_yen,
                                amount, amount-bet.stake_yen, None)

    if bet.race_ref != payout.race_ref:
        return blocked(SettlementStatus.INVALID, "race_mismatch")
    if bet.wager_type != payout.wager_type:
        return blocked(SettlementStatus.INVALID, "wager_mismatch")
    # No real-source bypass exists, even if someone supplies a unit string.
    if payout.origin != DataOrigin.SYNTHETIC:
        return blocked(SettlementStatus.UNCONFIRMED, "hr_mapping_unconfirmed")
    if payout.state != OutcomeState.NORMAL:
        state_map = {
            OutcomeState.SPECIAL_PAYMENT: (SettlementStatus.SPECIAL_PAYMENT, "special_payment_rule_unconfirmed"),
            OutcomeState.NOT_ESTABLISHED: (SettlementStatus.NOT_ESTABLISHED, "not_established_rule_unconfirmed"),
            OutcomeState.NOT_OFFERED: (SettlementStatus.NOT_OFFERED, "not_offered_rule_unconfirmed"),
            OutcomeState.CANCELLED: (SettlementStatus.CANCELLED, "cancelled_rule_unconfirmed"),
            OutcomeState.UNCONFIRMED: (SettlementStatus.UNCONFIRMED, "outcome_unconfirmed"),
            OutcomeState.INVALID: (SettlementStatus.INVALID, "outcome_invalid"),
        }
        status, reason = state_map[payout.state]
        return blocked(status, reason)
    if payout.amount_basis != JPY_PER_100:
        return blocked(SettlementStatus.UNCONFIRMED, "unit_unconfirmed")
    if payout.revision_state != "final":
        return blocked(SettlementStatus.UNCONFIRMED, "revision_unconfirmed")
    if payout.race_start_at is None or payout.finalized_at is None:
        return blocked(SettlementStatus.UNCONFIRMED, "timing_unconfirmed")
    try:
        if bet.decided_at >= payout.race_start_at:
            return blocked(SettlementStatus.INVALID, "purchase_not_before_race")
        if payout.finalized_at < payout.race_start_at:
            return blocked(SettlementStatus.INVALID, "invalid_result_timing")
        if payout.race_start_at.astimezone(JST).date() != payout.race_date:
            return blocked(SettlementStatus.INVALID, "race_date_mismatch")
    except (ValueError, OverflowError, TypeError):
        return blocked(SettlementStatus.INVALID, "invalid_timestamp")
    if payout.refund_targets is None:
        return blocked(SettlementStatus.UNCONFIRMED, "refund_targets_unconfirmed")
    if len(payout.targets) > _MAX_TARGETS[payout.wager_type]:
        return blocked(SettlementStatus.INVALID, "too_many_targets")
    numbers = [target.horse_number for target in payout.targets]
    if len(set(numbers)) != len(numbers):
        return blocked(SettlementStatus.INVALID, "duplicate_target")
    if payout.wager_type == "tansho" and payout.targets:
        if ((len(payout.targets) > 1 and not all(t.is_dead_heat for t in payout.targets))
                or (len(payout.targets) == 1 and payout.targets[0].is_dead_heat)):
            return blocked(SettlementStatus.INVALID, "invalid_dead_heat_targets")
    for target in payout.targets:
        target.__post_init__()
        if payout.refund_targets.is_refunded(wager_type=payout.wager_type, horse_number=target.horse_number):
            return blocked(SettlementStatus.INVALID, "refund_payout_overlap")
    if payout.refund_targets.is_refunded(wager_type=payout.wager_type, horse_number=bet.horse_number):
        # Explicit fictional policy from the task. No corresponding real-HR rule
        # has been enabled, and exceptional wager states were already blocked.
        return paid(SettlementStatus.REFUND, bet.stake_yen)
    if not payout.targets_complete:
        return blocked(SettlementStatus.UNCONFIRMED, "incomplete_winning_targets")
    if not payout.targets:
        return blocked(SettlementStatus.UNCONFIRMED, "missing_winning_targets")
    for target in payout.targets:
        if target.horse_number == bet.horse_number:
            status = SettlementStatus.DEAD_HEAT_HIT if target.is_dead_heat else SettlementStatus.NORMAL_HIT
            return paid(status, (bet.stake_yen // 100)*target.payout_yen_per_100)
    return paid(SettlementStatus.NORMAL_MISS, 0)


@dataclass(frozen=True, slots=True, repr=False)
class SettlementTotals:
    ticket_count: int
    settled_count: int
    unresolved_count: int
    invested_yen: int
    unresolved_invested_yen: int
    returned_yen: int | None
    net_profit_yen: int | None

    def __repr__(self):
        return "SettlementTotals(details=local_only)"


def summarize_settlements(results: tuple[SettlementResult, ...]) -> SettlementTotals:
    """Sum local results without silently dropping unresolved or duplicate tickets.

    Monthly grouping can use race_month_jst before calling this function.
    No monthly database report or ROI is computed by this module.
    """
    if type(results) is not tuple or any(type(r) is not SettlementResult for r in results):
        raise SettlementValidationError("invalid_aggregation")
    for result in results:
        result.__post_init__()
    refs = [result.ticket_ref for result in results]
    if len(set(refs)) != len(refs):
        raise SettlementValidationError("duplicate_ticket")
    if not results:
        return SettlementTotals(0, 0, 0, 0, 0, None, None)
    unresolved = [r for r in results if not r.is_settled]
    invested = sum(r.invested_yen for r in results)
    returned = None if unresolved else sum(r.returned_yen for r in results)
    return SettlementTotals(len(results), len(results)-len(unresolved), len(unresolved), invested,
                            sum(r.invested_yen for r in unresolved), returned,
                            None if returned is None else returned-invested)
