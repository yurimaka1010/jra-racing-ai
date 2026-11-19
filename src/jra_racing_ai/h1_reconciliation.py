"""Pure H1/HR normal-context checks. Record types have distinct status rules."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from .core import RACE_KEY
from .payout_reconciliation import (
    ALL_HYPOTHESES, H1Evidence, LocalHRRecord, RaceKey, ReconciliationError, ReconciliationLimits,
    ReferenceDocument, RevisionReview, _LocalOnly, _identifier, _integer, _object, _require, _reconcile_payouts,
)

H1_ROLES = ("status", "tansho_sale", "fukusho_sale", "place_paid_positions")
HR_ROLES = ("tansho_not_established", "tansho_special_payment", "fukusho_not_established", "fukusho_special_payment")


@dataclass(frozen=True, slots=True, repr=False)
class H1Review(_LocalOnly):
    race: RaceKey
    revision: RevisionReview

    def __post_init__(self):
        _require(type(self.race) is RaceKey and type(self.revision) is RevisionReview, "invalid_h1_config")


@dataclass(frozen=True, slots=True, repr=False)
class H1Config(_LocalOnly):
    h1_columns: tuple[str, ...]
    hr_columns: tuple[str, ...]
    column_meanings_reviewed_locally: bool
    reviews: tuple[H1Review, ...]

    def __post_init__(self):
        _require(type(self.column_meanings_reviewed_locally) is bool, "invalid_h1_config")
        for columns in (self.h1_columns, self.hr_columns):
            _require(type(columns) is tuple and len(columns) == 4 and all(_identifier(c) for c in columns), "invalid_h1_config")
            _require(len(set(columns)) == 4 and not set(columns) & set(RACE_KEY), "invalid_h1_config")
        _require(not any(c.startswith("haraimodoshi_") or c.startswith("henkan_") for c in self.hr_columns), "invalid_h1_config")
        _require(type(self.reviews) is tuple and len(self.reviews) <= 20
                 and all(type(r) is H1Review for r in self.reviews), "invalid_h1_config")
        _require(len({r.race for r in self.reviews}) == len(self.reviews), "invalid_h1_config")

    def check_document(self, document):
        _require(type(document) is ReferenceDocument, "invalid_h1_config")
        _require(all(r.race in document.races for r in self.reviews), "invalid_h1_config")
        _require(document.status_column not in self.hr_columns, "invalid_h1_config")


def parse_h1_config(value) -> H1Config:
    _object(value, ("format_version", "h1_columns", "hr_columns", "column_meanings_reviewed_locally", "h1_revision_reviews"))
    _require(type(value["format_version"]) is int and value["format_version"] == 1, "invalid_h1_config")
    _object(value["h1_columns"], H1_ROLES)
    _object(value["hr_columns"], HR_ROLES)
    _require(type(value["h1_revision_reviews"]) is list and len(value["h1_revision_reviews"]) <= 20, "invalid_h1_config")
    reviews = []
    for item in value["h1_revision_reviews"]:
        _object(item, ("race_key", "revision_review"))
        _object(item["race_key"], RACE_KEY)
        try:
            revision = RevisionReview(item["revision_review"])
        except (ValueError, TypeError):
            raise ReconciliationError("invalid_h1_config") from None
        reviews.append(H1Review(RaceKey(tuple(item["race_key"][k] for k in RACE_KEY)), revision))
    return H1Config(tuple(value["h1_columns"][k] for k in H1_ROLES),
                    tuple(value["hr_columns"][k] for k in HR_ROLES),
                    value["column_meanings_reviewed_locally"], tuple(reviews))


def check_h1_budget(limits: ReconciliationLimits, max_h1_rows: int):
    _require(type(limits) is ReconciliationLimits and _integer(max_h1_rows, 1, 200), "invalid_limits")
    # Role check + two catalog queries and one data query for each table.
    _require(5 + 7*limits.statement_timeout_ms/1000 <= limits.max_run_seconds - 1, "h1_time_budget")


@dataclass(frozen=True, slots=True, repr=False)
class LocalH1Record(_LocalOnly):
    race: RaceKey
    category: object
    tansho_sale: object
    fukusho_sale: object
    place_paid_positions: object
    oversized: bool = False

    def __post_init__(self):
        _require(type(self.race) is RaceKey and type(self.oversized) is bool, "invalid_h1_record")


def _normal_reason(document, config, race, hr, h1):
    if not hr:
        return "missing_hr"
    if len(hr) != 1:
        return "multiple_hr_versions"
    if not h1:
        return "missing_h1"
    if len(h1) != 1:
        return "multiple_h1_records"
    if not config.column_meanings_reviewed_locally:
        return "h1_mapping_unreviewed"
    review = next((r.revision for r in config.reviews if r.race == race), RevisionReview.UNCONFIRMED)
    if review is not RevisionReview.SNAPSHOT_REVIEWED:
        return {RevisionReview.CORRECTION: "h1_correction_unreviewed", RevisionReview.DELETED: "h1_deletion_unreviewed",
                RevisionReview.PRELIMINARY: "h1_preliminary_unreviewed"}.get(review, "h1_revision_unreviewed")
    hr, h1 = hr[0], h1[0]
    if type(hr.category) is not str or hr.category != "2":
        return "hr_category_mismatch_for_h1"
    if type(h1.category) is not str or h1.category != "5":
        return "h1_category_mismatch"
    if h1.oversized:
        return "oversized_h1_field"
    if hr.oversized:
        return "oversized_hr_field"
    if any(type(v) is not str or v != "7" for v in (h1.tansho_sale, h1.fukusho_sale)):
        return "h1_sale_flag_unconfirmed"
    if type(h1.place_paid_positions) is not str or h1.place_paid_positions not in ("2", "3"):
        return "h1_place_key_unconfirmed"
    if len(hr.flags) != 4 or any(type(v) is not str or v != "0" for v in hr.flags):
        return "hr_normal_flags_unconfirmed"
    if any(type(v) is not str or v != "0" for v in (hr.refund_tansho, hr.refund_fukusho)):
        return "hr_refund_flags_not_zero"
    for case in document.cases:
        if case.race == race and case.wager_type == "fukusho":
            if case.place_paid_positions is None or str(case.place_paid_positions) != h1.place_paid_positions:
                return "h1_place_terms_mismatch"
    return None


def reconcile_h1_payouts(document, hr_records, h1_records, config, *,
                         limits=ReconciliationLimits(max_run_seconds=45), max_h1_rows=40, hypotheses=ALL_HYPOTHESES):
    """H1/HR must each be unique; no inner-join loss or automatic latest choice."""
    _require(type(config) is H1Config, "invalid_h1_config")
    config.check_document(document)
    limits.check_document(document)
    check_h1_budget(limits, max_h1_rows)
    _require(type(hr_records) is tuple and len(hr_records) <= limits.max_hr_rows
             and all(type(r) is LocalHRRecord for r in hr_records), "invalid_record")
    _require(type(h1_records) is tuple and len(h1_records) <= max_h1_rows, "row_limit")
    _require(all(type(r) is LocalH1Record for r in h1_records), "invalid_h1_record")
    hrs, h1s, categories = defaultdict(list), defaultdict(list), Counter()
    for record in hr_records:
        _require(record.race in document.races, "unexpected_hr_race")
        hrs[record.race].append(record)
    for record in h1_records:
        _require(record.race in document.races, "unexpected_h1_race")
        h1s[record.race].append(record)
        v = record.category
        categories[v if type(v) is str and len(v) == 1 and v in "0123456789" else "null" if v is None else "other"] += 1
    decisions = {r: _normal_reason(document, config, r, hrs[r], h1s[r]) for r in document.races}
    reasons = Counter(v for v in decisions.values() if v is not None)
    result = _reconcile_payouts(document, hr_records, limits=limits, hypotheses=hypotheses,
                                normal_context=lambda case, _: decisions[case.race])
    evidence = H1Evidence(len(document.races), len(h1_records), sum(v is None for v in decisions.values()),
                          sum(reasons.values()), tuple(sorted(categories.items())), tuple(sorted(reasons.items())))
    return replace(result, h1_evidence=evidence)
