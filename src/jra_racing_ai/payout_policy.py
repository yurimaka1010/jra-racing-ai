"""Proposed case policies, never a decoder of JV-Data/PC-KEIBA records.

Inputs are invented, normalized case names. No horse numbers, flags, odds,
amounts, or source records are accepted or interpreted here.
"""
from .core import SafeAuditError


# No amount formula is implemented until unit/encoding/exception mappings exist.
_CASES = {
    "normal_win_hit": ("ordinary_win", "official_win_target_membership", "include_if_mapping_confirmed"),
    "normal_place_hit": ("ordinary_place", "official_place_target_membership", "include_if_mapping_confirmed"),
    "dead_heat_win_hit": ("ordinary_win_dead_heat", "official_win_target_membership", "include_if_mapping_confirmed"),
    "dead_heat_place_hit": ("ordinary_place_dead_heat", "official_place_target_membership", "include_if_mapping_confirmed"),
    "refund": ("refund", "no_standard_hit_label", "exclude_standard_hit_metrics"),
    "race_cancelled": ("cancelled", "no_standard_hit_label", "exclude_standard_hit_metrics"),
    "not_established": ("not_established", "no_standard_hit_label", "exclude_standard_hit_metrics"),
    "special_payment": ("special_payment", "no_standard_hit_label", "exclude_standard_hit_metrics"),
    "not_offered": ("not_offered", "no_standard_hit_label", "no_purchase_allowed"),
    "missing": ("unresolved_missing", "unknown_never_zero", "exclude_pending_review"),
    "initial_value": ("unresolved_initial", "unknown_never_zero", "exclude_pending_review"),
    "invalid": ("unresolved_invalid", "unknown_never_zero", "exclude_pending_review"),
    "unmapped": ("unresolved_mapping", "unknown_never_zero", "exclude_pending_review"),
}
PAYOUT_CASES = tuple(_CASES)


def describe_case(case: str) -> dict:
    """Describe a future policy without classifying an actual payout."""
    if not isinstance(case, str) or case not in _CASES:
        raise SafeAuditError("invalid_configuration")
    ledger_kind, label_rule, score_rule = _CASES[case]
    return {"case": case, "status": "proposed", "ledger_kind": ledger_kind,
            "standard_hit_label_rule": label_rule, "score_rule": score_rule,
            "source_mapping_status": "needs_confirmation",
            "amount_rule": "needs_confirmation", "profit_calculation": "not_implemented"}


def policy_summary() -> dict:
    return {"status": "proposed", "record_classification": "not_implemented",
            "cases": [describe_case(case) for case in PAYOUT_CASES],
            "note": "特殊ケースを別状態として設計。実フラグ・配列・金額の解釈や自動分類は行わない。"}
