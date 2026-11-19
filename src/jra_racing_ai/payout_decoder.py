"""Lossless HR field classification with a closed monetary confirmation gate.

Widths, slot limits and blank patterns are based on user-confirmed local format
observations. They do not establish official meanings, currency, or settlement
rules. No database, logging, file, or network operations are performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import NoReturn

from .core import SafeAuditError


@dataclass(frozen=True, slots=True)
class _WagerFormat:
    max_slots: int
    fields: tuple[tuple[str, int], ...]


# Other wager families need their own verified field layout and validator.
# They must not inherit the single-horse layout merely by changing a slot limit.
_SINGLE_HORSE_FIELDS = (("a", 2), ("b", 9), ("c", 2))
_WAGER_FORMATS = MappingProxyType({
    "tansho": _WagerFormat(3, _SINGLE_HORSE_FIELDS),
    "fukusho": _WagerFormat(5, _SINGLE_HORSE_FIELDS),
})
UNCONFIRMED_HR_RULES = (
    "a_b_c_meanings", "currency_and_base_stake", "zero_padding_rules",
    "unused_slot_initial_values", "special_horse_codes", "popularity_special_values",
    "settlement_flag_interactions", "correction_and_deletion_selection",
)
_REASONS = frozenset({
    "unsupported_wager_type", "missing_slots", "invalid_slot_container", "too_many_slots",
    "invalid_slot_type", "missing_field", "invalid_field_type", "invalid_field_width",
    "unsupported_field_characters", "duplicate_raw_slot", "duplicate_a_requires_confirmation",
    "payout_special_values_unconfirmed", "payout_semantics_unconfirmed", "invalid_parsed_layout",
    "inconsistent_slot_fields", "candidate_horse_out_of_range",
})

ZERO_HORSE_REQUIRED_CONTEXT = (
    "wager_specific_refund_flags", "wager_specific_special_payment_flags",
    "wager_specific_establishment_flags", "wager_specific_sale_status",
    "race_status", "correction_and_deletion_selection",
)


class FieldFormat(str, Enum):
    DIGITS = "digits"
    BLANK = "blank"


class SlotKind(str, Enum):
    UNUSED_BLANK = "unused_blank"
    ORDINARY_CANDIDATE = "ordinary_candidate"
    ZERO_HORSE_REQUIRES_CONFIRMATION = "zero_horse_requires_confirmation"
    ZERO_VALUE_REQUIRES_CONFIRMATION = "zero_value_requires_confirmation"


class PayoutDecodingError(SafeAuditError):
    status = "needs_confirmation"

    def __init__(self, reason: str):
        self.reason = reason if type(reason) is str and reason in _REASONS else "payout_semantics_unconfirmed"
        super().__init__("payout_needs_confirmation")

    def diagnostic(self) -> dict[str, str]:
        return {"status": self.status, "code": self.code, "reason": self.reason}


@dataclass(frozen=True, slots=True, repr=False)
class RawPayoutSlot:
    """Original a/b/c strings; b is never a converted amount in this module."""
    a: object
    b: object
    c: object

    def __repr__(self):
        return "RawPayoutSlot(details=local_only)"


def _field_format(value, width: int) -> FieldFormat:
    if value is None:
        raise PayoutDecodingError("missing_field")
    if type(value) is not str:
        raise PayoutDecodingError("invalid_field_type")
    if len(value) != width:
        raise PayoutDecodingError("invalid_field_width")
    if value == " " * width:
        return FieldFormat.BLANK
    if all(char in "0123456789" for char in value):
        return FieldFormat.DIGITS
    raise PayoutDecodingError("unsupported_field_characters")


def _validate_slots(wager_type, slots):
    if type(wager_type) is not str or wager_type not in _WAGER_FORMATS:
        raise PayoutDecodingError("unsupported_wager_type")
    if type(slots) is not tuple:
        raise PayoutDecodingError("invalid_slot_container")
    if not slots:
        raise PayoutDecodingError("missing_slots")
    spec = _WAGER_FORMATS[wager_type]
    if len(slots) > spec.max_slots:
        raise PayoutDecodingError("too_many_slots")
    for slot in slots:
        if type(slot) is not RawPayoutSlot:
            raise PayoutDecodingError("invalid_slot_type")
        for name, width in spec.fields:
            _field_format(getattr(slot, name), width)


@dataclass(frozen=True, slots=True, repr=False)
class ParsedPayoutLayout:
    wager_type: str
    slots: tuple[RawPayoutSlot, ...]

    def __post_init__(self):
        _validate_slots(self.wager_type, self.slots)

    @property
    def ready_for_settlement(self) -> bool:
        return False

    @property
    def status(self) -> str:
        return "needs_confirmation"

    def __repr__(self):
        return "ParsedPayoutLayout(ready_for_settlement=False, details=local_only)"


def parse_payout_layout(*, wager_type: str, slots: tuple[RawPayoutSlot, ...]) -> ParsedPayoutLayout:
    """Validate widths/type/accepted characters while retaining all opaque slots.

    Exact-width ASCII blanks and digits are retained without stripping or
    padding. Cross-field combinations are checked by classify_payout_layout().
    All-zero digit slots are never silently removed or converted to zero yen.
    """
    return ParsedPayoutLayout(wager_type, slots)


@dataclass(frozen=True, slots=True, repr=False)
class ClassifiedPayoutSlot:
    """Local-memory format finding, never a normalized payable target."""
    slot_number: int
    raw: RawPayoutSlot
    field_formats: tuple[FieldFormat, ...]
    kind: SlotKind

    @property
    def format_status(self) -> str:
        if self.kind in (SlotKind.UNUSED_BLANK, SlotKind.ORDINARY_CANDIDATE):
            return "valid"
        return "needs_confirmation"

    @property
    def required_context(self) -> tuple[str, ...]:
        if self.kind is SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION:
            return ZERO_HORSE_REQUIRED_CONTEXT
        if self.kind is SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION:
            return ("zero_amount_or_popularity_meaning", "settlement_flag_interactions")
        return ()

    @property
    def ready_for_settlement(self) -> bool:
        return False

    def __repr__(self):
        return "ClassifiedPayoutSlot(ready_for_settlement=False, details=local_only)"


def _classify_slots(layout: ParsedPayoutLayout) -> tuple[ClassifiedPayoutSlot, ...]:
    if type(layout) is not ParsedPayoutLayout:
        raise PayoutDecodingError("invalid_parsed_layout")
    _validate_slots(layout.wager_type, layout.slots)
    spec = _WAGER_FORMATS[layout.wager_type]
    result = []
    seen, a_values = set(), set()
    for index, raw in enumerate(layout.slots, 1):
        formats = tuple(_field_format(getattr(raw, name), width) for name, width in spec.fields)
        if formats == (FieldFormat.BLANK,) * 3:
            kind = SlotKind.UNUSED_BLANK
        elif formats[0] is FieldFormat.DIGITS and raw.a == "00":
            # Both observed 00/blank/blank and all-digit special candidates
            # remain unresolved. A single missing b or c is an inconsistency.
            if formats[1] is not formats[2]:
                raise PayoutDecodingError("inconsistent_slot_fields")
            kind = SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION
        elif formats == (FieldFormat.DIGITS,) * 3:
            # Lexical range checking after exact ASCII-width validation keeps
            # all original leading zeros, including the entire b field.
            if not "01" <= raw.a <= "28":
                raise PayoutDecodingError("candidate_horse_out_of_range")
            kind = (SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION
                    if raw.b == "000000000" or raw.c == "00"
                    else SlotKind.ORDINARY_CANDIDATE)
            triple = (raw.a, raw.b, raw.c)
            if triple in seen:
                raise PayoutDecodingError("duplicate_raw_slot")
            if raw.a in a_values:
                raise PayoutDecodingError("duplicate_a_requires_confirmation")
            seen.add(triple)
            a_values.add(raw.a)
        else:
            raise PayoutDecodingError("inconsistent_slot_fields")
        result.append(ClassifiedPayoutSlot(index, raw, formats, kind))
    return tuple(result)


@dataclass(frozen=True, slots=True, repr=False)
class ClassifiedPayoutLayout:
    layout: ParsedPayoutLayout
    slots: tuple[ClassifiedPayoutSlot, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "slots", _classify_slots(self.layout))

    @property
    def wager_type(self) -> str:
        return self.layout.wager_type

    @property
    def ordinary_candidates(self) -> tuple[ClassifiedPayoutSlot, ...]:
        """Keep every distinct candidate; this is not a complete winner set."""
        return tuple(slot for slot in self.slots if slot.kind is SlotKind.ORDINARY_CANDIDATE)

    @property
    def has_all_slots(self) -> bool:
        return len(self.slots) == _WAGER_FORMATS[self.wager_type].max_slots

    @property
    def format_status(self) -> str:
        return ("valid" if all(slot.format_status == "valid" for slot in self.slots)
                else "needs_confirmation")

    @property
    def status(self) -> str:
        return "needs_confirmation"

    @property
    def ready_for_settlement(self) -> bool:
        return False

    @property
    def required_confirmations(self) -> tuple[str, ...]:
        return UNCONFIRMED_HR_RULES

    def __repr__(self):
        return "ClassifiedPayoutLayout(ready_for_settlement=False, details=local_only)"


def classify_payout_layout(layout: ParsedPayoutLayout) -> ClassifiedPayoutLayout:
    """Classify exact blanks, ordinary formats and unresolved zero codes.

    No flag meanings, dead heats, monetary units, or revisions are inferred.
    Even a full layout of ordinary-format candidates cannot be settled.
    """
    return ClassifiedPayoutLayout(layout)


def decode_hr_payouts(layout: ParsedPayoutLayout) -> NoReturn:
    """Stop: an independently confirmed HR monetary mapping is not installed.

    This deliberately has no assume_confirmed/amount_unit override. Later
    implementation must first verify the specification and PC-KEIBA mappings.
    """
    classified = classify_payout_layout(layout)
    for slot in classified.slots:
        if slot.kind in (SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION,
                         SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION):
            raise PayoutDecodingError("payout_special_values_unconfirmed")
    raise PayoutDecodingError("payout_semantics_unconfirmed")
