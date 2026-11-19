"""Pure, bounded hypothesis comparison. Never authorizes HR settlement.

All record/identity containers redact repr. Only the aggregate summary is an
output contract. No database, file, network, logging, or settlement imports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
import re

from .core import DOMESTIC_VENUES, SafeAuditError
from .payout_decoder import (
    PayoutDecodingError, RawPayoutSlot, SlotKind, classify_payout_layout, parse_payout_layout,
)
from .refunds import RefundValidationError, determine_refund_targets

WAGERS = ("tansho", "fukusho")
SLOT_COUNTS = {"tansho": 3, "fukusho": 5}
FLAG_KINDS = ("not_established", "special_payment", "not_offered")
_DECODER_REASONS = (
    "unsupported_wager_type", "missing_slots", "invalid_slot_container", "too_many_slots",
    "invalid_slot_type", "missing_field", "invalid_field_type", "invalid_field_width",
    "unsupported_field_characters", "duplicate_raw_slot", "duplicate_a_requires_confirmation",
    "payout_special_values_unconfirmed", "payout_semantics_unconfirmed", "invalid_parsed_layout",
    "inconsistent_slot_fields", "candidate_horse_out_of_range",
)
REASONS = frozenset(_DECODER_REASONS + (
    "invalid_reference", "duplicate_reference_key", "duplicate_reference_case", "duplicate_reference_target",
    "conflicting_reference", "invalid_mapping", "invalid_limits", "invalid_hypothesis",
    "invalid_record", "unexpected_hr_race", "row_limit", "race_limit", "input_size_limit",
    "unsafe_input", "invalid_pdf", "invalid_json", "time_limit", "synthetic_reference_forbidden",
    "local_execution_required", "unsynced_confirmation_required", "endpoint_mismatch",
    "hr_table_unavailable", "hr_column_mapping_unconfirmed", "reference_source_unreviewed",
    "abc_mapping_unreviewed", "missing_hr", "multiple_hr_versions", "deleted_hr", "preliminary_hr",
    "cancelled_hr", "unknown_hr_status", "revision_unreviewed", "correction_unreviewed",
    "reference_refund", "reference_not_offered", "reference_not_established", "reference_special_payment",
    "reference_cancelled", "reference_unconfirmed", "reference_invalid", "incomplete_layout",
    "zero_horse_unconfirmed", "zero_value_unconfirmed", "place_two_paid_unconfirmed", "place_terms_unreviewed",
    "refund_context_unconfirmed", "refund_payout_overlap", "flag_mapping_unreviewed",
    "flag_value_unconfirmed", "hr_not_established", "hr_special_payment", "hr_not_offered",
    "target_set_mismatch", "fractional_yen_hypothesis", "oversized_hr_field",
    "invalid_h1_config", "invalid_h1_record", "missing_h1", "multiple_h1_records", "unexpected_h1_race",
    "h1_mapping_unreviewed", "h1_revision_unreviewed", "h1_correction_unreviewed", "h1_deletion_unreviewed",
    "h1_preliminary_unreviewed", "h1_category_mismatch", "hr_category_mismatch_for_h1",
    "h1_sale_flag_unconfirmed", "h1_place_key_unconfirmed", "h1_place_terms_mismatch",
    "hr_normal_flags_unconfirmed", "hr_refund_flags_not_zero", "oversized_h1_field",
    "h1_table_unavailable", "h1_column_mapping_unconfirmed", "h1_time_budget",
))


class ReconciliationError(SafeAuditError):
    status = "needs_confirmation"

    def __init__(self, reason: str):
        self.reason = reason if type(reason) is str and reason in REASONS else "invalid_reference"
        super().__init__("reconciliation_needs_confirmation")

    def diagnostic(self):
        return {"status": self.status, "code": self.code, "reason": self.reason}


class _LocalOnly:
    def __repr__(self):
        return type(self).__name__ + "(details=local_only)"


def _require(condition, reason="invalid_reference"):
    if not condition:
        raise ReconciliationError(reason)


def _digits(value, width):
    return type(value) is str and len(value) == width and all(c in "0123456789" for c in value)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _identifier(value):
    return type(value) is str and re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value) is not None


class CaseState(str, Enum):
    NORMAL = "normal"
    REFUND = "refund"
    NOT_OFFERED = "not_offered"
    NOT_ESTABLISHED = "not_established"
    SPECIAL_PAYMENT = "special_payment"
    CANCELLED = "cancelled"
    UNCONFIRMED = "unconfirmed"
    INVALID = "invalid"


class RevisionReview(str, Enum):
    UNCONFIRMED = "unconfirmed"
    SNAPSHOT_REVIEWED = "snapshot_reviewed_locally"
    PRELIMINARY = "preliminary"
    CORRECTION = "correction"
    DELETED = "deleted"


class AmountHypothesis(str, Enum):
    RAW_YEN = "raw_integer_as_yen_per_100"
    DIVIDE_10 = "raw_divided_by_10_as_yen_per_100"
    MULTIPLY_10 = "raw_times_10_as_yen_per_100"

    @property
    def ratio(self):
        return {self.RAW_YEN: (1, 1), self.DIVIDE_10: (1, 10), self.MULTIPLY_10: (10, 1)}[self]


ALL_HYPOTHESES = tuple(AmountHypothesis)


@dataclass(frozen=True, slots=True, repr=False)
class RaceKey(_LocalOnly):
    parts: tuple[str, ...]

    def __post_init__(self):
        _require(type(self.parts) is tuple and len(self.parts) == 6)
        _require(all(_digits(v, w) for v, w in zip(self.parts, (4, 4, 2, 2, 2, 2))))
        _require(self.parts[2] in DOMESTIC_VENUES and all(v != "00" for v in self.parts[3:]))
        try:
            date(int(self.parts[0]), int(self.parts[1][:2]), int(self.parts[1][2:]))
        except ValueError:
            raise ReconciliationError("invalid_reference") from None
        _require(1800 <= self.year <= 2200)

    @property
    def year(self):
        return int(self.parts[0])


@dataclass(frozen=True, slots=True, repr=False)
class ReferenceTarget(_LocalOnly):
    horse_number: str
    payout_yen_per_100: int
    plus10: bool | None = None

    def __post_init__(self):
        _require(_digits(self.horse_number, 2) and "01" <= self.horse_number <= "28")
        _require(_integer(self.payout_yen_per_100, 1, 10**12))
        _require(self.plus10 is None or type(self.plus10) is bool)


@dataclass(frozen=True, slots=True, repr=False)
class ReferenceCase(_LocalOnly):
    race: RaceKey
    wager_type: str
    state: CaseState
    revision_review: RevisionReview
    dead_heat: bool
    place_paid_positions: int | None
    targets: tuple[ReferenceTarget, ...]

    def __post_init__(self):
        _require(type(self.race) is RaceKey and type(self.wager_type) is str and self.wager_type in WAGERS)
        _require(type(self.state) is CaseState and type(self.revision_review) is RevisionReview)
        _require(type(self.dead_heat) is bool and type(self.targets) is tuple)
        _require(self.place_paid_positions is None or _integer(self.place_paid_positions, 2, 3))
        _require(self.wager_type == "fukusho" or self.place_paid_positions is None)
        _require(all(type(t) is ReferenceTarget for t in self.targets))
        _require(len(self.targets) <= SLOT_COUNTS[self.wager_type])
        _require(len({t.horse_number for t in self.targets}) == len(self.targets), "duplicate_reference_target")
        if self.state is CaseState.NORMAL:
            _require(bool(self.targets))
            if self.wager_type == "tansho":
                _require(self.dead_heat == (len(self.targets) > 1), "conflicting_reference")
        else:
            # Special payments/refunds are separate cases, never normal winners.
            _require(not self.targets and not self.dead_heat, "conflicting_reference")


@dataclass(frozen=True, slots=True, repr=False)
class FlagRule(_LocalOnly):
    wager_type: str
    kind: str
    column: str
    normal_value: str
    active_value: str
    reviewed_locally: bool

    def __post_init__(self):
        _require(type(self.wager_type) is str and self.wager_type in WAGERS, "invalid_mapping")
        _require(type(self.kind) is str and self.kind in FLAG_KINDS, "invalid_mapping")
        _require(_identifier(self.column), "invalid_mapping")
        for value in (self.normal_value, self.active_value):
            _require(type(value) is str and 1 <= len(value) <= 4
                     and all(c in "0123456789 " for c in value), "invalid_mapping")
        _require(self.normal_value != self.active_value and type(self.reviewed_locally) is bool, "invalid_mapping")


@dataclass(frozen=True, slots=True, repr=False)
class ReferenceDocument(_LocalOnly):
    source: str
    source_reviewed_locally: bool
    abc_mapping_reviewed_locally: bool
    status_column: str
    flag_rules: tuple[FlagRule, ...]
    cases: tuple[ReferenceCase, ...]

    def __post_init__(self):
        _require(type(self.source) is str and self.source in ("synthetic", "jra_official_manual"))
        _require(type(self.source_reviewed_locally) is bool and type(self.abc_mapping_reviewed_locally) is bool)
        _require(_identifier(self.status_column), "invalid_mapping")
        _require(type(self.flag_rules) is tuple and len(self.flag_rules) <= 6, "invalid_mapping")
        _require(all(type(rule) is FlagRule for rule in self.flag_rules), "invalid_mapping")
        _require(len({(r.wager_type, r.kind) for r in self.flag_rules}) == len(self.flag_rules), "invalid_mapping")
        names = [r.column for r in self.flag_rules]
        for name in set(names):
            shared = [r for r in self.flag_rules if r.column == name]
            # One physical code field may distinguish two non-normal states.
            # Such sharing is explicit, with one consistent normal code.
            _require(len({r.wager_type for r in shared}) == 1 and len({r.normal_value for r in shared}) == 1
                     and len({r.active_value for r in shared}) == len(shared), "invalid_mapping")
        reserved = {"kaisai_nen", "kaisai_tsukihi", "keibajo_code", "kaisai_kai", "kaisai_nichime", "race_bango",
                    "henkan_umaban_joho", "henkan_flag_tansho", "henkan_flag_fukusho"}
        reserved.update(f"haraimodoshi_{w}_{n}{f}" for w in WAGERS for n in range(1, SLOT_COUNTS[w]+1) for f in "abc")
        _require(self.status_column not in reserved and not (set(names) & (reserved | {self.status_column})), "invalid_mapping")
        _require(type(self.cases) is tuple and 1 <= len(self.cases) <= 40)
        _require(all(type(case) is ReferenceCase for case in self.cases))
        _require(len({(c.race, c.wager_type) for c in self.cases}) == len(self.cases), "duplicate_reference_case")
        revisions = {}
        for case in self.cases:
            if case.race in revisions:
                _require(revisions[case.race] is case.revision_review, "conflicting_reference")
            revisions[case.race] = case.revision_review

    @property
    def races(self):
        return tuple(dict.fromkeys(case.race for case in self.cases))


@dataclass(frozen=True, slots=True)
class ReconciliationLimits:
    year: int = 2025
    max_races: int = 5
    max_hr_rows: int = 40
    statement_timeout_ms: int = 5000
    max_run_seconds: int = 30

    def __post_init__(self):
        _require(_integer(self.year, 1800, 2200), "invalid_limits")
        _require(_integer(self.max_races, 1, 20) and _integer(self.max_hr_rows, 1, 200), "invalid_limits")
        _require(_integer(self.statement_timeout_ms, 1000, 10000), "invalid_limits")
        _require(_integer(self.max_run_seconds, 10, 120), "invalid_limits")
        # One connect (5s), role check + two catalog queries + one HR query.
        _require(5 + 4*self.statement_timeout_ms/1000 <= self.max_run_seconds - 1, "invalid_limits")

    def check_document(self, document):
        _require(type(document) is ReferenceDocument)
        _require(len(document.races) <= self.max_races, "race_limit")
        _require(all(r.year == self.year for r in document.races))


def _object(value, keys):
    _require(type(value) is dict and set(value) == set(keys))
    return value


def parse_reference_document(value) -> ReferenceDocument:
    """Strict decoded JSON contract. Unknown keys and coercion are rejected."""
    from .core import RACE_KEY
    _object(value, ("format_version", "source", "official_base_yen", "source_reviewed_locally",
                    "abc_mapping_reviewed_locally", "status_column", "flag_rules", "cases"))
    _require(type(value["format_version"]) is int and value["format_version"] == 1)
    _require(type(value["official_base_yen"]) is int and value["official_base_yen"] == 100)
    _require(type(value["flag_rules"]) is list and len(value["flag_rules"]) <= 6)
    rules = []
    for rule in value["flag_rules"]:
        _object(rule, ("wager_type", "kind", "column", "normal_value", "active_value", "reviewed_locally"))
        rules.append(FlagRule(**rule))
    _require(type(value["cases"]) is list and 1 <= len(value["cases"]) <= 40)
    cases = []
    for case in value["cases"]:
        _object(case, ("race_key", "wager_type", "state", "revision_review", "dead_heat", "place_paid_positions", "targets"))
        _object(case["race_key"], RACE_KEY)
        _require(type(case["targets"]) is list and len(case["targets"]) <= 5)
        targets = []
        for target in case["targets"]:
            _object(target, ("horse_number", "payout_yen_per_100", "plus10"))
            targets.append(ReferenceTarget(**target))
        try:
            state, revision = CaseState(case["state"]), RevisionReview(case["revision_review"])
        except (ValueError, TypeError):
            raise ReconciliationError("invalid_reference") from None
        cases.append(ReferenceCase(RaceKey(tuple(case["race_key"][k] for k in RACE_KEY)),
                                   case["wager_type"], state, revision, case["dead_heat"],
                                   case["place_paid_positions"], tuple(targets)))
    return ReferenceDocument(value["source"], value["source_reviewed_locally"],
                             value["abc_mapping_reviewed_locally"], value["status_column"], tuple(rules), tuple(cases))


@dataclass(frozen=True, slots=True, repr=False)
class LocalHRRecord(_LocalOnly):
    race: RaceKey
    category: object
    tansho: tuple[RawPayoutSlot, ...]
    fukusho: tuple[RawPayoutSlot, ...]
    refund_bitmap: object
    refund_tansho: object
    refund_fukusho: object
    flags: tuple[object, ...]
    oversized: bool = False

    def __post_init__(self):
        _require(type(self.race) is RaceKey and type(self.flags) is tuple and type(self.oversized) is bool, "invalid_record")


@dataclass(frozen=True, slots=True)
class HypothesisObservation:
    hypothesis: AmountHypothesis
    matched: int
    mismatched: int
    uncomparable: int

    def __post_init__(self):
        _require(type(self.hypothesis) is AmountHypothesis, "invalid_hypothesis")
        _require(all(_integer(v, 0, 10000) for v in (self.matched, self.mismatched, self.uncomparable)))

    def to_dict(self):
        total = self.matched + self.mismatched
        status = ("mismatch_observed" if self.mismatched else "incomplete_sample" if self.uncomparable
                  else "sample_matches_only" if self.matched else "no_comparable_targets")
        numerator, denominator = self.hypothesis.ratio
        return {"hypothesis": self.hypothesis.value, "numerator": numerator, "denominator": denominator,
                "matched": self.matched, "mismatched": self.mismatched, "uncomparable": self.uncomparable,
                "match_rate_among_comparable": round(self.matched/total, 6) if total else None,
                "observation_status": status, "unit_confirmed": False}


@dataclass(frozen=True, slots=True)
class WagerEvidence:
    wager_type: str
    cases: int
    eligible_cases: int
    blocked_cases: int
    reference_targets: int
    unexpected_hr_targets: int
    refund_present_cases: int
    dead_heat_cases: int
    place_two_paid_cases: int
    plus10_targets: int
    plus10_unknown_targets: int
    formats: tuple[tuple[str, int], ...]
    reasons: tuple[tuple[str, int], ...]
    observations: tuple[HypothesisObservation, ...]

    def __post_init__(self):
        _require(self.wager_type in WAGERS)
        for name in ("cases", "eligible_cases", "blocked_cases", "reference_targets", "unexpected_hr_targets",
                     "refund_present_cases", "dead_heat_cases", "place_two_paid_cases", "plus10_targets", "plus10_unknown_targets"):
            _require(_integer(getattr(self, name), 0, 10000))
        _require(self.cases == self.eligible_cases + self.blocked_cases)
        for values, allowed in ((self.formats, {s.value for s in SlotKind} | {"invalid_layout"}), (self.reasons, REASONS)):
            _require(type(values) is tuple)
            _require(all(type(t) is tuple and len(t) == 2 and t[0] in allowed and _integer(t[1], 0, 10000) for t in values))
            _require(len(dict(values)) == len(values))
        _require(type(self.observations) is tuple and all(type(o) is HypothesisObservation for o in self.observations))
        _require(all(o.matched + o.mismatched + o.uncomparable == self.reference_targets for o in self.observations))

    def to_dict(self):
        return {name: getattr(self, name) for name in (
            "wager_type", "cases", "eligible_cases", "blocked_cases", "reference_targets", "unexpected_hr_targets",
            "refund_present_cases", "dead_heat_cases", "place_two_paid_cases", "plus10_targets", "plus10_unknown_targets"
        )} | {"format_counts": dict(self.formats), "reason_counts": dict(self.reasons),
              "hypotheses": [o.to_dict() for o in self.observations]}


@dataclass(frozen=True, slots=True)
class H1Evidence:
    requested_races: int
    records: int
    compatible_races: int
    blocked_races: int
    categories: tuple[tuple[str, int], ...]
    reasons: tuple[tuple[str, int], ...]

    def __post_init__(self):
        _require(all(_integer(v, 0, 200) for v in (self.requested_races, self.records, self.compatible_races, self.blocked_races)))
        _require(self.requested_races == self.compatible_races + self.blocked_races and self.requested_races <= 20)
        for values, allowed in ((self.categories, set("0123456789") | {"null", "other"}), (self.reasons, REASONS)):
            _require(type(values) is tuple and all(type(t) is tuple and len(t) == 2 and t[0] in allowed
                     and _integer(t[1], 0, 200) for t in values))
            _require(len(dict(values)) == len(values))
        _require(sum(n for _, n in self.categories) == self.records)
        _require(sum(n for _, n in self.reasons) == self.blocked_races)

    def to_dict(self):
        return {"mode": "h1_required", "status": "completed_needs_confirmation",
                "requested_races": self.requested_races, "records": self.records,
                "compatible_races": self.compatible_races, "blocked_races": self.blocked_races,
                "category_counts": dict(self.categories), "reason_counts": dict(self.reasons),
                "rule_basis": "user_provided_hr2_h1_5_sale7_hr_flags0_place2_or3",
                "production_settlement_enabled": False}


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    year: int
    source: str
    requested_races: int
    hr_records: int
    categories: tuple[tuple[str, int], ...]
    wagers: tuple[WagerEvidence, ...]
    h1_evidence: H1Evidence | None = None

    def __post_init__(self):
        _require(_integer(self.year, 1800, 2200) and self.source in ("synthetic", "jra_official_manual"))
        _require(_integer(self.requested_races, 1, 20) and _integer(self.hr_records, 0, 200))
        _require(type(self.categories) is tuple and all(type(t) is tuple and len(t) == 2
                 and t[0] in ("0", "1", "2", "9", "other", "null") and _integer(t[1], 0, 200) for t in self.categories))
        _require(len(dict(self.categories)) == len(self.categories) and sum(n for _, n in self.categories) == self.hr_records)
        _require(type(self.wagers) is tuple and len(self.wagers) == 2 and all(type(w) is WagerEvidence for w in self.wagers))
        _require(tuple(w.wager_type for w in self.wagers) == WAGERS)
        _require(self.h1_evidence is None or type(self.h1_evidence) is H1Evidence)
        if self.h1_evidence is not None:
            self.h1_evidence.__post_init__()
            _require(self.h1_evidence.requested_races == self.requested_races)

    def to_safe_dict(self):
        payload = {"format_version": 1, "status": "comparison_completed_needs_confirmation", "year": self.year,
                "source": self.source, "requested_races": self.requested_races, "hr_records": self.hr_records,
                "hr_category_counts": dict(self.categories), "official_base_yen": 100,
                "official_basis_evidence": "user_provided_jra_display_basis",
                "jv_data_reference": "4.9.0.1_user_provided_not_independently_verified",
                "conditions": "selected_domestic_races_normal_cases_with_local_reviews_only",
                "amount_interpretation_status": "needs_confirmation", "production_settlement_enabled": False,
                "wagers": [w.to_dict() for w in self.wagers]}
        if self.h1_evidence is not None:
            payload["h1_validation"] = self.h1_evidence.to_dict()
        return payload


def _case_block_reason(document, case, record, classified, refunds, normal_context=None):
    if not document.source_reviewed_locally:
        return "reference_source_unreviewed"
    if case.state is not CaseState.NORMAL:
        return "reference_" + case.state.value
    category = record.category
    if type(category) is not str or category not in ("0", "1", "2", "9"):
        return "unknown_hr_status"
    if category != "2":
        return {"0": "deleted_hr", "1": "preliminary_hr", "9": "cancelled_hr"}[category]
    if case.revision_review is RevisionReview.CORRECTION:
        return "correction_unreviewed"
    if case.revision_review is not RevisionReview.SNAPSHOT_REVIEWED:
        return "revision_unreviewed"
    if record.oversized:
        return "oversized_hr_field"
    if normal_context is None and case.wager_type == "fukusho" and case.place_paid_positions == 2:
        return "place_two_paid_unconfirmed"
    if case.wager_type == "fukusho" and case.place_paid_positions is None:
        return "place_terms_unreviewed"
    if any(s.kind is SlotKind.ZERO_HORSE_REQUIRES_CONFIRMATION for s in classified.slots):
        return "zero_horse_unconfirmed"
    if any(s.kind is SlotKind.ZERO_VALUE_REQUIRES_CONFIRMATION for s in classified.slots):
        return "zero_value_unconfirmed"
    if not document.abc_mapping_reviewed_locally:
        return "abc_mapping_unreviewed"
    if normal_context is not None:
        reason = normal_context(case, record)
        if reason is not None:
            return reason
    else:
        reason = _legacy_flag_reason(document, case, record)
        if reason is not None:
            return reason
    if refunds is None:
        return "refund_context_unconfirmed"
    refunded = set(refunds.tansho_umaban if case.wager_type == "tansho" else refunds.fukusho_umaban)
    if any(s.raw.a in refunded for s in classified.ordinary_candidates):
        return "refund_payout_overlap"
    if {s.raw.a for s in classified.ordinary_candidates} != {t.horse_number for t in case.targets}:
        return "target_set_mismatch"
    return None


def _legacy_flag_reason(document, case, record):
    rules = [(i, r) for i, r in enumerate(document.flag_rules) if r.wager_type == case.wager_type]
    if {r.kind for _, r in rules} != set(FLAG_KINDS) or not all(r.reviewed_locally for _, r in rules):
        return "flag_mapping_unreviewed"
    if len(record.flags) != len(document.flag_rules):
        return "flag_value_unconfirmed"
    for column in dict.fromkeys(r.column for _, r in rules):
        shared = [(i, r) for i, r in rules if r.column == column]
        value = record.flags[shared[0][0]]
        if type(value) is not str or any(record.flags[i] != value for i, _ in shared):
            return "flag_value_unconfirmed"
        if value == shared[0][1].normal_value:
            continue
        for _, rule in shared:
            if value == rule.active_value:
                return "hr_" + rule.kind
        return "flag_value_unconfirmed"
    return None


def _reconcile_payouts(document: ReferenceDocument, records: tuple[LocalHRRecord, ...], *,
                      limits: ReconciliationLimits = ReconciliationLimits(),
                      hypotheses: tuple[AmountHypothesis, ...] = ALL_HYPOTHESES,
                      normal_context=None) -> ReconciliationSummary:
    """Compare reviewed normal cases under explicit, unconfirmed amount ratios.

    Per-target counts are separate from blocked case counts (special cases have
    no ordinary targets). Never infer misses, select latest HR, or emit details.
    """
    _require(type(limits) is ReconciliationLimits, "invalid_limits")
    limits.check_document(document)
    _require(type(records) is tuple and len(records) <= limits.max_hr_rows, "row_limit")
    _require(all(type(record) is LocalHRRecord for record in records), "invalid_record")
    _require(type(hypotheses) is tuple and bool(hypotheses) and len(hypotheses) <= 3
             and all(type(h) is AmountHypothesis for h in hypotheses)
             and len(set(hypotheses)) == len(hypotheses), "invalid_hypothesis")
    by_race = defaultdict(list)
    categories = Counter()
    for record in records:
        _require(record.race in document.races, "unexpected_hr_race")
        by_race[record.race].append(record)
        value = record.category
        categories[value if type(value) is str and value in ("0", "1", "2", "9") else "null" if value is None else "other"] += 1
    evidence = []
    for wager in WAGERS:
        cases = [c for c in document.cases if c.wager_type == wager]
        counts, formats, reasons = Counter(), Counter(), Counter()
        observations = {h: Counter() for h in hypotheses}
        for case in cases:
            counts["reference_targets"] += len(case.targets)
            counts["dead_heat_cases"] += case.dead_heat
            counts["place_two_paid_cases"] += case.place_paid_positions == 2
            counts["plus10_targets"] += sum(t.plus10 is True for t in case.targets)
            counts["plus10_unknown_targets"] += sum(t.plus10 is None for t in case.targets)
            records_for_race = by_race[case.race]
            classified, refunds = None, None
            reason = None
            if not records_for_race:
                reason = "missing_hr"
            elif len(records_for_race) != 1:
                reason = "multiple_hr_versions"
            else:
                record = records_for_race[0]
                slots = record.tansho if wager == "tansho" else record.fukusho
                try:
                    classified = classify_payout_layout(parse_payout_layout(wager_type=wager, slots=slots))
                    formats.update(s.kind.value for s in classified.slots)
                    if not classified.has_all_slots:
                        reason = "incomplete_layout"
                except PayoutDecodingError as error:
                    formats["invalid_layout"] += 1
                    reason = error.reason if error.reason in REASONS else "invalid_record"
                if record.oversized:
                    reason = "oversized_hr_field"
                try:
                    refunds = determine_refund_targets(henkan_umaban_joho=record.refund_bitmap,
                        henkan_flag_tansho=record.refund_tansho, henkan_flag_fukusho=record.refund_fukusho)
                    counts["refund_present_cases"] += bool(refunds.tansho_umaban if wager == "tansho" else refunds.fukusho_umaban)
                except RefundValidationError:
                    refunds = None
                if reason is None:
                    reason = _case_block_reason(document, case, record, classified, refunds, normal_context)
                if classified is not None:
                    counts["unexpected_hr_targets"] += len({s.raw.a for s in classified.ordinary_candidates}
                                                            - {t.horse_number for t in case.targets})
            if reason is not None:
                reasons[reason] += 1
                counts["blocked_cases"] += 1
                for hypothesis in hypotheses:
                    observations[hypothesis]["uncomparable"] += len(case.targets)
                continue
            counts["eligible_cases"] += 1
            candidates = {s.raw.a: s.raw.b for s in classified.ordinary_candidates}
            for target in case.targets:
                original_nine_characters = candidates[target.horse_number]
                # Classification validated all 9 ASCII digits first. This int is
                # used only for a named hypothesis comparison, never settlement.
                raw_integer = int(original_nine_characters)
                for hypothesis in hypotheses:
                    numerator, denominator = hypothesis.ratio
                    product = raw_integer * numerator
                    if product % denominator:
                        observations[hypothesis]["uncomparable"] += 1
                        reasons["fractional_yen_hypothesis"] += 1
                    elif product == target.payout_yen_per_100 * denominator:
                        observations[hypothesis]["matched"] += 1
                    else:
                        observations[hypothesis]["mismatched"] += 1
        evidence.append(WagerEvidence(wager, len(cases), counts["eligible_cases"], counts["blocked_cases"],
            counts["reference_targets"], counts["unexpected_hr_targets"], counts["refund_present_cases"],
            counts["dead_heat_cases"], counts["place_two_paid_cases"], counts["plus10_targets"], counts["plus10_unknown_targets"],
            tuple(sorted(formats.items())), tuple(sorted(reasons.items())),
            tuple(HypothesisObservation(h, observations[h]["matched"], observations[h]["mismatched"],
                                        observations[h]["uncomparable"]) for h in hypotheses)))
    return ReconciliationSummary(limits.year, document.source, len(document.races), len(records),
                                 tuple(sorted(categories.items())), tuple(evidence))


def reconcile_payouts(document: ReferenceDocument, records: tuple[LocalHRRecord, ...], *,
                      limits: ReconciliationLimits = ReconciliationLimits(),
                      hypotheses: tuple[AmountHypothesis, ...] = ALL_HYPOTHESES) -> ReconciliationSummary:
    """Phase 1.5 compatibility API; does not claim H1 validation."""
    return _reconcile_payouts(document, records, limits=limits, hypotheses=hypotheses)
