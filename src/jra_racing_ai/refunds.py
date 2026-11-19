"""Strict, local-only refund-target validation for the confirmed HR bitmap.

This module has no database, file, logging, network, or settlement side effects.
It interprets only the 28-position bitmap and the two win/place refund flags.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import SafeAuditError

_BITMAP_LENGTH = 28
_HORSE_NUMBERS = tuple(f"{number:02d}" for number in range(1, _BITMAP_LENGTH + 1))
_WAGER_TYPES = ("tansho", "fukusho")
_REASONS = frozenset({
    "missing_bitmap", "bitmap_type", "empty_bitmap", "bitmap_length", "bitmap_characters",
    "missing_flag", "invalid_flag", "flag_without_targets", "targets_without_flag",
    "mixed_wager_flags_unconfirmed", "invalid_horse_number", "unsupported_wager_type",
    "invalid_target_collection", "unconfirmed_refund_information",
})
_FIELDS = frozenset({"henkan_umaban_joho", "henkan_flag_tansho", "henkan_flag_fukusho",
                     "wager_flags", "horse_number", "wager_type", "targets"})


class RefundValidationError(SafeAuditError):
    """Stop processing with fixed diagnostics and no copy of source values."""
    status = "needs_confirmation"

    def __init__(self, reason: str, field: str):
        self.reason = reason if type(reason) is str and reason in _REASONS else "unconfirmed_refund_information"
        self.field = field if type(field) is str and field in _FIELDS else "targets"
        super().__init__("refund_needs_confirmation")

    def diagnostic(self) -> dict[str, str]:
        # No bitmap, horse number, flag value, race key, or source record.
        return {"status": self.status, "code": self.code,
                "reason": self.reason, "field": self.field}


def _parse_bitmap(value: object) -> tuple[str, ...]:
    if value is None:
        raise RefundValidationError("missing_bitmap", "henkan_umaban_joho")
    if type(value) is not str:
        raise RefundValidationError("bitmap_type", "henkan_umaban_joho")
    if value == "":
        raise RefundValidationError("empty_bitmap", "henkan_umaban_joho")
    if len(value) != _BITMAP_LENGTH:
        raise RefundValidationError("bitmap_length", "henkan_umaban_joho")
    if any(character not in ("0", "1") for character in value):
        raise RefundValidationError("bitmap_characters", "henkan_umaban_joho")
    return tuple(number for number, character in zip(_HORSE_NUMBERS, value) if character == "1")


def _flag(value: object, field: str) -> bool:
    if value is None:
        raise RefundValidationError("missing_flag", field)
    # Exact encodings only. bool, float, Decimal, bytes, spaces, and zero padding
    # are rejected; no str(), int(), strip(), padding, or truthiness conversion.
    if type(value) is str and value in ("0", "1"):
        return value == "1"
    if type(value) is int and value in (0, 1):
        return value == 1
    raise RefundValidationError("invalid_flag", field)


def _horse_number(value: object) -> str:
    if type(value) is not str or value not in _HORSE_NUMBERS:
        raise RefundValidationError("invalid_horse_number", "horse_number")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class RefundTargets:
    """Validated local target sets; repr never reveals horse numbers."""
    tansho_umaban: tuple[str, ...]
    fukusho_umaban: tuple[str, ...]

    def __post_init__(self):
        for collection in (self.tansho_umaban, self.fukusho_umaban):
            if type(collection) is not tuple:
                raise RefundValidationError("invalid_target_collection", "targets")
            for number in collection:
                _horse_number(number)
            if collection != tuple(sorted(set(collection))):
                raise RefundValidationError("invalid_target_collection", "targets")
        # Mixed win/place applicability of the common bitmap is unconfirmed.
        # Even manually constructed instances may not bypass this constraint.
        if self.tansho_umaban != self.fukusho_umaban:
            raise RefundValidationError("mixed_wager_flags_unconfirmed", "wager_flags")

    def __repr__(self) -> str:
        return "RefundTargets(validated=True, details=local_only)"

    def is_refunded(self, *, wager_type: str, horse_number: object) -> bool:
        if type(wager_type) is not str or wager_type not in _WAGER_TYPES:
            raise RefundValidationError("unsupported_wager_type", "wager_type")
        number = _horse_number(horse_number)
        targets = self.tansho_umaban if wager_type == "tansho" else self.fukusho_umaban
        return number in targets


def determine_refund_targets(*, henkan_umaban_joho: object = None,
                             henkan_flag_tansho: object = None,
                             henkan_flag_fukusho: object = None) -> RefundTargets:
    """Validate one local input and stop on invalid/unconfirmed information.

    A position i (1-based) corresponds to horse number i, from 01 through 28.
    No source values are stored in errors or printed. Returned numbers are for
    local in-memory use only, not audit logs/reports. Amounts are not interpreted.
    """
    targets = _parse_bitmap(henkan_umaban_joho)
    win = _flag(henkan_flag_tansho, "henkan_flag_tansho")
    place = _flag(henkan_flag_fukusho, "henkan_flag_fukusho")
    if not targets and (win or place):
        raise RefundValidationError("flag_without_targets", "wager_flags")
    if targets and not (win or place):
        raise RefundValidationError("targets_without_flag", "wager_flags")
    if win != place:
        raise RefundValidationError("mixed_wager_flags_unconfirmed", "wager_flags")
    return RefundTargets(tansho_umaban=targets, fukusho_umaban=targets)
