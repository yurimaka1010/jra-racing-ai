"""Pure win-only as-of selection, decisions, official-payment settlement and reports."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
import re
import statistics

from ..core import RACE_KEY, SafeAuditError
from ..ml.features import race_identity

JST = timezone(timedelta(hours=9))
REASONS = frozenset({
    "invalid_input", "invalid_policy", "invalid_probability", "probability_sum_invalid",
    "duplicate_prediction", "schedule_unavailable", "schedule_unverified", "schedule_conflict",
    "schedule_after_decision", "roster_mismatch", "odds_unavailable", "odds_unverified",
    "odds_time_precision",
    "odds_after_decision", "odds_conflict", "odds_incomplete", "odds_invalid", "odds_capped", "unsupported_status",
    "ev_below_threshold", "ev_selected", "favorite_selected", "result_unavailable",
    "result_unconfirmed", "result_invalid", "result_revision_unconfirmed", "result_duplicate",
    "special_pending", "refund_pending", "outside_scope", "settled_win", "settled_loss",
    "invalid_mapping", "mapping_unconfirmed", "unsafe_path", "input_limit", "io_failed",
    "needs_authentication", "explicit_execution_required", "query_timeout", "query_limit",
    "invalid_schema", "database_error", "artifact_invalid", "invalid_period", "internal_error",
})


class SimulationError(SafeAuditError):
    def __init__(self, code="invalid_input"):
        self.code = code if code in REASONS else "internal_error"
        Exception.__init__(self, self.code)


def decimal(value):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
            raise ValueError
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError
        return result
    except (ValueError, InvalidOperation):
        raise SimulationError("invalid_input") from None


def instant(value):
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(JST)
    except (ValueError, TypeError, OverflowError):
        raise SimulationError("invalid_input") from None


def key(value):
    try:
        if not isinstance(value, (tuple, list)) or len(value) != 6:
            raise ValueError
        return race_identity(dict(zip(RACE_KEY, value)))[0]
    except Exception:
        raise SimulationError("invalid_input") from None


def horse(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}", value) or not 1 <= int(value) <= 28:
        raise SimulationError("invalid_input")
    return value


@dataclass(frozen=True, repr=False)
class PredictionRace:
    race: tuple[str, ...]
    probabilities: dict[str, object]

    def __repr__(self):
        return "PredictionRace(local_only)"


@dataclass(frozen=True, repr=False)
class Schedule:
    race: tuple[str, ...]
    scheduled_start: datetime
    observed_at: datetime | None
    horses: tuple[str, ...]
    verified: bool = False
    availability_basis: str = "observed"


@dataclass(frozen=True, repr=False)
class Snapshot:
    race: tuple[str, ...]
    published_at: datetime
    observed_at: datetime | None
    odds: dict[str, object]
    source: str = "local_snapshot"
    verified: bool = False
    reason: str | None = None
    publication_precision: str = "second"


@dataclass(frozen=True, repr=False)
class Outcome:
    race: tuple[str, ...]
    status: str
    payouts_yen_per_100: dict[str, int] = field(default_factory=dict)
    verified: bool = False
    complete_winner_set: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class WinPolicy:
    threshold: str = "0.05"
    minutes_before_start: int = 10
    probability_method: str = "race_normalize"
    temperature: float = 1.0
    evaluation_start: str = "1950-01-01"
    evaluation_end: str = "2200-12-31"
    model_fit_end: str = "1949-12-31"
    calibration_end: str | None = None
    threshold_selection_end: str | None = None
    calibration_source: str = "fixed_not_fitted"
    timing_basis: str = "observed"

    def validate(self):
        try:
            if (not 0 <= decimal(self.threshold) <= 10
                    or type(self.minutes_before_start) is not int or not 1 <= self.minutes_before_start <= 1440
                    or self.probability_method not in ("race_normalize", "identity")
                    or isinstance(self.temperature, bool) or not math.isfinite(self.temperature)
                    or not .25 <= self.temperature <= 4
                    or self.calibration_source not in ("fixed_not_fitted", "independent_validation")
                    or self.timing_basis not in ("observed", "jra_publication_time")
                    or (self.probability_method == "identity" and self.temperature != 1)):
                raise ValueError
            start, end = date.fromisoformat(self.evaluation_start), date.fromisoformat(self.evaluation_end)
            fit_end = date.fromisoformat(self.model_fit_end)
            if fit_end >= start or start > end:
                raise ValueError
            for limit in (self.calibration_end, self.threshold_selection_end):
                if limit is not None and not fit_end < date.fromisoformat(limit) < start:
                    raise ValueError
            if self.calibration_source == "independent_validation" and self.calibration_end is None:
                raise ValueError
        except (ValueError, TypeError, SimulationError):
            raise SimulationError("invalid_policy") from None
        return self


def probabilities(values, policy):
    if not isinstance(values, dict) or not 2 <= len(values) <= 28:
        raise SimulationError("invalid_probability")
    try:
        original = {horse(h): decimal(p) for h, p in values.items()}
        if any(not 0 <= p <= 1 for p in original.values()):
            raise SimulationError("invalid_probability")
        total = sum(original.values())
        if total <= 0:
            raise SimulationError("probability_sum_invalid")
        if policy.probability_method == "identity":
            if abs(total - 1) > Decimal("0.000001"):
                raise SimulationError("probability_sum_invalid")
            return original, total
        # The mechanical normalization is predeclared, not fitted to test results.
        weighted = (original if policy.temperature == 1 else
                    {h: decimal(float(p) ** (1 / policy.temperature)) for h, p in original.items()})
        denominator = sum(weighted.values())
        return {h: p / denominator for h, p in weighted.items()}, total
    except SimulationError as exc:
        raise SimulationError(exc.code if exc.code == "probability_sum_invalid" else "invalid_probability") from None


def choose_schedule(race, versions, policy):
    if not versions:
        return None, None, "schedule_unavailable"
    try:
        if any(item.observed_at is None for item in versions):
            if (policy.timing_basis != "jra_publication_time" or len(versions) != 1
                    or versions[0].availability_basis not in ("ra_advance_card", "reviewed_historical_schedule")):
                return None, None, "schedule_unverified"
            item = versions[0]
            start = instant(item.scheduled_start)
            roster = tuple(sorted(horse(h) for h in item.horses))
            if (not item.verified or key(item.race) != race
                    or start.date().strftime("%Y%m%d") != race[0] + race[1]):
                return None, None, "schedule_unverified"
            if len(roster) != len(set(roster)) or not 2 <= len(roster) <= 28:
                return None, None, "roster_mismatch"
            return item, start - timedelta(minutes=policy.minutes_before_start), None
        versions = sorted(versions, key=lambda item: instant(item.observed_at))
        chosen, seen = None, {}
        for item in versions:
            observed = instant(item.observed_at)
            if chosen is not None and observed > chosen[1]:
                break
            if key(item.race) != race or not item.verified:
                return None, None, "schedule_unverified"
            observed, start = instant(item.observed_at), instant(item.scheduled_start)
            if start.date().strftime("%Y%m%d") != race[0] + race[1]:
                return None, None, "schedule_unverified"
            roster = tuple(sorted(horse(h) for h in item.horses))
            if len(roster) != len(set(roster)) or not 2 <= len(roster) <= 28:
                return None, None, "roster_mismatch"
            cutoff = start - timedelta(minutes=policy.minutes_before_start)
            signature = (start, roster)
            if observed in seen and seen[observed] != signature:
                return None, None, "schedule_conflict"
            seen[observed] = signature
            if observed > cutoff:
                return None, None, "schedule_after_decision"
            chosen = (Schedule(race, start, observed, roster, True), cutoff)
        return (*chosen, None) if chosen else (None, None, "schedule_unavailable")
    except (SimulationError, TypeError, AttributeError):
        return None, None, "schedule_unverified"


def choose_snapshot(race, snapshots, decision_at, roster, *, timing_basis="observed"):
    if timing_basis not in ("observed", "jra_publication_time"):
        return None, "odds_unverified"
    if not snapshots:
        return None, "odds_unavailable"
    eligible, imprecise_boundary = [], False
    for item in snapshots:
        try:
            if key(item.race) != race:
                continue
            published = instant(item.published_at)
            if timing_basis == "jra_publication_time":
                if item.publication_precision not in ("second", "minute"):
                    return None, "odds_unverified"
                upper = published + timedelta(minutes=1) if item.publication_precision == "minute" else published
                imprecise_boundary |= published <= decision_at < upper
                if upper <= decision_at:
                    eligible.append((published, published, item))
            else:
                observed = instant(item.observed_at)
                if published <= decision_at and observed <= decision_at:
                    eligible.append((observed, published, item))
        except (SimulationError, AttributeError):
            return None, "odds_unverified"
    if not eligible:
        return None, "odds_time_precision" if imprecise_boundary else "odds_after_decision"
    if timing_basis == "jra_publication_time" and len({item[2].publication_precision for item in eligible}) > 1:
        return None, "odds_conflict"
    latest = max((item[0], item[1]) for item in eligible)
    selected = [item[2] for item in eligible if item[:2] == latest]
    # Even a latest incomplete/unconfirmed snapshot prevents stale fallback.
    if len(selected) != 1:
        return None, "odds_conflict"
    item = selected[0]
    if timing_basis == "observed" and instant(item.observed_at) < instant(item.published_at):
        return None, "odds_invalid"
    if not item.verified:
        return None, item.reason if item.reason in REASONS else "odds_unverified"
    if item.reason:
        return None, item.reason if item.reason in REASONS else "odds_invalid"
    try:
        odds = {horse(h): decimal(price) for h, price in item.odds.items()}
        if set(odds) != set(roster):
            return None, "odds_incomplete"
        if any(price < 1 for price in odds.values()):
            return None, "odds_invalid"
        if any(price >= Decimal("999.9") for price in odds.values()):
            return None, "odds_capped"
        observed = instant(item.observed_at) if item.observed_at is not None else None
        return Snapshot(race, instant(item.published_at), observed, odds,
                        item.source, True, publication_precision=item.publication_precision), None
    except (SimulationError, AttributeError):
        return None, "odds_invalid"


def outcome_reason(outcome, roster):
    if outcome is None:
        return "result_unavailable"
    if outcome.status != "normal_final":
        return outcome.reason if outcome.reason in REASONS else "special_pending"
    if not outcome.verified or not outcome.complete_winner_set:
        return "result_unconfirmed"
    try:
        values = outcome.payouts_yen_per_100
        if not isinstance(values, dict) or not 1 <= len(values) <= 3:
            return "result_invalid"
        if any(horse(h) not in roster or type(amount) is not int or not 1 <= amount <= 999999999
               for h, amount in values.items()):
            return "result_invalid"
    except SimulationError:
        return "result_invalid"
    return None


def settle(selection, outcome, roster):
    if selection is None:
        return {"status": "no_bet", "reason": None, "stake_yen": 0, "payout_yen": 0,
                "net_profit_yen": 0, "hit": False}
    if horse(selection) not in roster:
        return {"status": "pending", "reason": "roster_mismatch", "stake_yen": 100,
                "payout_yen": None, "net_profit_yen": None, "hit": None}
    reason = outcome_reason(outcome, roster)
    if reason:
        return {"status": "pending", "reason": reason, "stake_yen": 100,
                "payout_yen": None, "net_profit_yen": None, "hit": None}
    payout = outcome.payouts_yen_per_100.get(selection, 0)
    return {"status": "settled", "reason": "settled_win" if payout else "settled_loss",
            "stake_yen": 100, "payout_yen": payout, "net_profit_yen": payout - 100, "hit": bool(payout)}


def run_simulation(predictions, schedules=(), snapshots=(), outcomes=(), *, policy=None, odds_failures=None):
    policy = (WinPolicy() if policy is None else policy).validate()
    if not isinstance(predictions, (list, tuple)) or len(predictions) > 20000:
        raise SimulationError("input_limit")
    by_schedule, by_snapshot, by_outcome = defaultdict(list), defaultdict(list), defaultdict(list)
    for items, index in ((schedules, by_schedule), (snapshots, by_snapshot), (outcomes, by_outcome)):
        if len(items) > 500000:
            raise SimulationError("input_limit")
        for item in items:
            index[key(item.race)].append(item)
    rows, seen = [], set()
    for prediction in predictions:
        race = key(prediction.race)
        if race in seen:
            raise SimulationError("duplicate_prediction")
        seen.add(race)
        day = date(int(race[0]), int(race[1][:2]), int(race[1][2:]))
        if not date.fromisoformat(policy.evaluation_start) <= day <= date.fromisoformat(policy.evaluation_end):
            raise SimulationError("invalid_period")
        row = {"race_key": list(race), "month": day.strftime("%Y-%m"), "decision_at": None,
               "scheduled_start": None, "schedule_observed_at": None, "odds_observed_at": None,
               "odds_published_at": None, "odds_acquired": False, "simulation_eligible": False,
               "common_settlement_eligible": False, "reason": None, "raw_probability_sum": None,
               "probability_sum": None, "probability_status": "valid", "runners": [], "ai_horse": None, "favorite_horse": None}
        row.update(timing_basis=policy.timing_basis, actual_observation_time_verified=False,
                   purchase_availability_verified=False, schedule_availability_basis=None,
                   odds_publication_precision=None)
        probs = None
        try:
            probs, raw_sum = probabilities(prediction.probabilities, policy)
            row.update(raw_probability_sum=float(raw_sum), probability_sum=float(sum(probs.values())))
        except SimulationError as exc:
            row["probability_status"] = exc.code
        schedule, decision, reason = choose_schedule(race, by_schedule[race], policy)
        if schedule is None:
            row["reason"] = reason
        else:
            row.update(decision_at=decision.isoformat(), scheduled_start=schedule.scheduled_start.isoformat(),
                       schedule_observed_at=schedule.observed_at.isoformat() if schedule.observed_at is not None else None,
                       schedule_availability_basis=schedule.availability_basis)
            if odds_failures and race in odds_failures:
                snapshot, reason = None, "odds_unverified"
            else:
                snapshot, reason = choose_snapshot(race, by_snapshot[race], decision, schedule.horses,
                                                   timing_basis=policy.timing_basis)
            if snapshot is None:
                row["reason"] = reason
            else:
                row.update(odds_acquired=True, odds_observed_at=snapshot.observed_at.isoformat() if snapshot.observed_at is not None else None,
                           odds_published_at=snapshot.published_at.isoformat(),
                           odds_publication_precision=snapshot.publication_precision,
                           actual_observation_time_verified=policy.timing_basis == "observed")
                try:
                    if probs is None:
                        raise SimulationError(row["probability_status"])
                    if set(probs) != set(schedule.horses):
                        raise SimulationError("roster_mismatch")
                    row.update(simulation_eligible=True, raw_probability_sum=float(raw_sum),
                               probability_sum=float(sum(probs.values())))
                    ev = {h: probs[h] * snapshot.odds[h] - 1 for h in probs}
                    candidates = sorted((h for h in probs if ev[h] >= decimal(policy.threshold)),
                                        key=lambda h: (-ev[h], h))
                    row["ai_horse"] = candidates[0] if candidates else None
                    row["favorite_horse"] = min(snapshot.odds, key=lambda h: (snapshot.odds[h], h))
                    row["reason"] = "ev_selected" if candidates else "ev_below_threshold"
                    row["runners"] = [{"horse_number": h, "raw_probability": float(decimal(prediction.probabilities[h])),
                                       "probability": float(probs[h]), "decision_odds": float(snapshot.odds[h]),
                                       "expected_return": float(ev[h])} for h in sorted(probs)]
                except SimulationError as exc:
                    row["reason"] = exc.code
        matches = by_outcome[race]
        outcome = matches[0] if len(matches) == 1 else None
        if len(matches) > 1:
            outcome = Outcome(race, "pending", reason="result_duplicate")
        roster = schedule.horses if schedule else ()
        row["common_settlement_eligible"] = row["simulation_eligible"] and outcome_reason(outcome, roster) is None
        row["settlement_reason"] = outcome_reason(outcome, roster)
        row["ai"] = settle(row["ai_horse"], outcome, roster)
        row["favorite"] = settle(row["favorite_horse"], outcome, roster)
        rows.append(row)
    notice = "判断時点の表示オッズは払戻額を保証しない。実購入・約定の実績ではない。"
    if policy.timing_basis == "jra_publication_time":
        notice += " JRA発表時刻基準の暫定的な事後シミュレーション。実取得時刻・購入可能時刻・通信遅延は未検証。前向き運用の実績ではない。"
    summary = summarize(rows)
    summary["timing_basis"] = policy.timing_basis
    summary["odds_coverage_basis"] = ("published_snapshots_not_actual_observations" if policy.timing_basis == "jra_publication_time"
                                      else "verified_observed_snapshots")
    return {"format_version": 2, "wager": "win", "mode": "hypothetical_backtest",
            "timing_basis": policy.timing_basis, "rows": rows, "summary": summary, "notice": notice}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p90": values[min(len(values)-1, int(len(values)*.9))], "max": values[-1],
            "mean": statistics.fmean(values)}


def financial(rows, strategy):
    bets = [row[strategy] for row in rows if row[strategy]["status"] != "no_bet"]
    settled = [bet for bet in bets if bet["status"] == "settled"]
    pending = len(bets) - len(settled)
    investment = len(bets) * 100
    settled_investment = len(settled) * 100
    paid = sum(bet["payout_yen"] for bet in settled)
    hits = sum(bet["hit"] for bet in settled)
    return {"bet_count": len(bets), "skip_count": len(rows)-len(bets), "investment_yen": investment,
            "hit_count": hits if not pending else None,
            "hit_rate_percent": 100*hits/len(bets) if bets and not pending else None,
            "payout_yen": paid if not pending else None,
            "net_profit_yen": paid-investment if not pending else None,
            "return_rate_percent": 100*paid/investment if investment and not pending else None,
            "unsettled_count": pending, "settled_bet_count": len(settled),
            "settled_subset": {"investment_yen": settled_investment, "payout_yen": paid,
                               "net_profit_yen": paid-settled_investment, "hit_count": hits,
                               "return_rate_percent": 100*paid/settled_investment if settled_investment else None},
            "pending_reasons": dict(Counter(bet["reason"] for bet in bets if bet["status"] == "pending"))}


def _summary(rows):
    acquired = sum(row["odds_acquired"] for row in rows)
    common = [row for row in rows if row["common_settlement_eligible"]]
    ai_bought = [row for row in common if row["ai_horse"] is not None]
    return {"target_race_count": len(rows), "odds_acquired_race_count": acquired,
            "odds_acquisition_rate_percent": 100*acquired/len(rows) if rows else None,
            "simulation_race_count": sum(row["simulation_eligible"] for row in rows),
            "race_reasons": dict(Counter(row["reason"] for row in rows)),
            "ai": financial(rows, "ai"), "favorite": financial(rows, "favorite"),
            "expected_return_distribution": distribution([runner["expected_return"] for row in rows for runner in row["runners"]]),
            "selected_expected_return_distribution": distribution([runner["expected_return"] for row in rows for runner in row["runners"]
                                                                   if runner["horse_number"] == row["ai_horse"]]),
            "raw_probability_sum_distribution": distribution([row["raw_probability_sum"] for row in rows
                                                               if row["raw_probability_sum"] is not None]),
            "probability_status_counts": dict(Counter(row["probability_status"] for row in rows)),
            "comparisons": {"all_common_eligible_races": {"race_count": len(common), "ai": financial(common, "ai"),
                                                          "favorite": financial(common, "favorite")},
                            "ai_purchased_common_races": {"race_count": len(ai_bought), "ai": financial(ai_bought, "ai"),
                                                           "favorite": financial(ai_bought, "favorite")}}}


def summarize(rows):
    result = _summary(rows)
    monthly = {month: _summary([row for row in rows if row["month"] == month])
               for month in sorted({row["month"] for row in rows})}
    result["by_month"] = monthly
    complete = [value["ai"]["net_profit_yen"] for value in monthly.values()
                if value["ai"]["bet_count"] and value["ai"]["unsettled_count"] == 0]
    result["monthly_stability"] = {"complete_active_month_count": len(complete),
                                    "profitable_month_count": sum(p > 0 for p in complete),
                                    "profit_mean_yen": statistics.fmean(complete) if complete else None,
                                    "profit_population_std_yen": statistics.pstdev(complete) if complete else None,
                                    "worst_month_yen": min(complete) if complete else None}
    return result


def fit_temperature(validation_predictions, winners, *, model_fit_end, test_start):
    """Small probability-only calibration on a separate, earlier period."""
    if not validation_predictions or date.fromisoformat(model_fit_end) >= date.fromisoformat(test_start):
        raise SimulationError("invalid_period")
    examples, days = [], []
    for item in validation_predictions:
        race = key(item.race)
        day = date(int(race[0]), int(race[1][:2]), int(race[1][2:]))
        if not date.fromisoformat(model_fit_end) < day < date.fromisoformat(test_start):
            raise SimulationError("invalid_period")
        winner = winners.get(race)
        if winner is not None:
            if winner not in item.probabilities:
                raise SimulationError("invalid_input")
            examples.append((item, winner))
            days.append(day)
    if not examples:
        raise SimulationError("invalid_input")
    scores = []
    for temperature in (.75, 1., 1.25, 2.):
        policy = WinPolicy(temperature=temperature)
        loss = 0.
        for item, winner in examples:
            probs, _ = probabilities(item.probabilities, policy)
            loss -= math.log(max(1e-15, float(probs[winner])))
        scores.append({"temperature": temperature, "race_log_loss": loss/len(examples)})
    best = min(scores, key=lambda item: (item["race_log_loss"], abs(item["temperature"]-1), item["temperature"]))
    return {"temperature": best["temperature"], "calibration_start": min(days).isoformat(),
            "calibration_end": max(days).isoformat(), "races": len(examples), "scores": scores}
