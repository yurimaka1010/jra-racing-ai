"""Fixed-cohort market benchmark. No training, purchase decisions or external I/O."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from ..win_sim import core, formats, publication
from . import v03

BIN_NAMES = ("0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-100%")
FINAL_CATEGORIES = ("5", "4", "3")


@dataclass(frozen=True, repr=False)
class MarketRace:
    race: tuple
    odds: dict
    source: str
    category: str
    basis: str
    published_at: object = None
    minutes_before_start: float | None = None
    cancelled_outside_roster: int = 0


def prices(row, roster):
    """Reuse the O1 decoder, masking only explicit non-roster cancellation slots."""
    payload = row.get("payload")
    if (row.get("oversized") or row.get("record_type") != "O1" or row.get("sale") != "7"
            or not isinstance(payload, str) or len(payload) != 224 or not payload.isascii()):
        raise core.SimulationError("odds_invalid")
    slots, cancelled = [], 0
    for i in range(28):
        slot = payload[i * 8:(i + 1) * 8]
        if slot != " " * 8 and slot[:2] != f"{i + 1:02d}":
            raise core.SimulationError("odds_invalid")
        if slot[:2] not in roster and slot[2:6] in ("----", "****"):
            slot = " " * 8
            cancelled += 1
        slots.append(slot)
    decoded = publication._raw_prices("".join(slots))
    if set(decoded) != set(roster):
        raise core.SimulationError("roster_mismatch")
    return decoded, cancelled


def _signature(row):
    return tuple(row.get(field) for field in ("record_type", "sale", "payload", "oversized"))


def select_final(race, rows, roster):
    if not rows:
        return None, "final_odds_unavailable"
    if any(row.get("status") not in tuple("12345") for row in rows):
        return None, "final_status_unconfirmed"
    for category in FINAL_CATEGORIES:
        selected = [row for row in rows if row.get("status") == category]
        if not selected:
            continue
        # Identical copies across two tables may be deduplicated. Multiple rows
        # within one table, or conflicting copies, never select an arbitrary row.
        tables = [row["source_table"] for row in selected]
        if len(set(tables)) != len(tables) or len({_signature(row) for row in selected}) != 1:
            return None, "final_odds_conflict"
        row = min(selected, key=lambda item: item["source_table"] != "jvd_o1")
        try:
            odds, cancelled = prices(row, roster)
        except core.SimulationError as exc:
            return None, exc.code
        return MarketRace(race, odds, row["source_table"], category, "final_category_" + category,
                          cancelled_outside_roster=cancelled), None
    return None, "final_odds_unavailable"


def select_pre_race(race, rows, roster, schedules, policy):
    candidates = [row for row in rows if row.get("status") == "1"]
    if not candidates:
        return None, "odds_history_unavailable"
    if any(row.get("status") not in tuple("12345") for row in rows):
        return None, "odds_unverified"
    schedule, cutoff, reason = core.choose_schedule(race, schedules, policy)
    if reason:
        return None, reason
    grouped = defaultdict(list)
    for row in candidates:
        try:
            stamp = formats.parse_time(row.get("published"), "mmddhhmm", race)
        except core.SimulationError:
            return None, "odds_unverified"
        grouped[stamp].append(row)
    snapshots, cancellation_counts = [], {}
    for stamp, versions in grouped.items():
        reason, odds, cancelled = None, {}, 0
        tables = [row["source_table"] for row in versions]
        if len(set(tables)) != len(tables) or len({_signature(row) for row in versions}) != 1:
            reason = "odds_conflict"
        row = min(versions, key=lambda item: item["source_table"] != "apd_sokuho_o1")
        if reason is None:
            try:
                odds, cancelled = prices(row, roster)
            except core.SimulationError as exc:
                reason = exc.code
        cancellation_counts[stamp] = cancelled
        snapshots.append(core.Snapshot(race, stamp, None, odds, row["source_table"], True, reason, "minute"))
    chosen, reason = core.choose_snapshot(race, snapshots, cutoff, roster, timing_basis="jra_publication_time")
    if reason:
        return None, reason
    return MarketRace(race, chosen.odds, chosen.source, "1", "pre_race_publication_time", chosen.published_at,
                      (schedule.scheduled_start - chosen.published_at).total_seconds() / 60,
                      cancellation_counts[chosen.published_at]), None


def select_cohorts(predictions, rows, ra_rows, policy):
    groups, schedule_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[core.key(row["race_key"])].append(row)
    schedules, _ = publication.schedules_from_ra(ra_rows, predictions)
    for schedule in schedules:
        schedule_groups[schedule.race].append(schedule)
    selected = {name: [] for name in ("pre_race_publication_time", *("final_category_" + c for c in FINAL_CATEGORIES))}
    reasons = {"pre_race": Counter(), "final": Counter()}
    excluded_runners = {"pre_race": 0, "final": 0}
    for prediction in predictions:
        race, roster = core.key(prediction.race), tuple(prediction.probabilities)
        for name, chooser in (("pre_race", lambda: select_pre_race(race, groups[race], roster, schedule_groups[race], policy)),
                              ("final", lambda: select_final(race, groups[race], roster))):
            quote, reason = chooser()
            if reason:
                reasons[name][reason] += 1
                excluded_runners[name] += len(roster)
            else:
                selected[quote.basis].append(quote)
    return selected, {name: {"excluded_race_count": sum(counts.values()),
                            "excluded_runner_count": excluded_runners[name], "reasons": dict(counts)}
                      for name, counts in reasons.items()}


def implied_probabilities(odds):
    values = {core.horse(h): core.decimal(o) for h, o in odds.items()}
    if not 2 <= len(values) <= 28 or any(o < 1 or o >= core.decimal("999.9") for o in values.values()):
        raise core.SimulationError("odds_invalid")
    inverse = {h: 1 / o for h, o in values.items()}
    total = sum(inverse.values())
    return {h: value / total for h, value in inverse.items()}, total


def _average_ranks(values):
    ordered = sorted(values, key=lambda h: (-values[h], h))
    ranks, start = {}, 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for horse in ordered[start:stop]:
            ranks[horse] = (start + 1 + stop) / 2
        start = stop
    return ranks


def spearman(first, second):
    if set(first) != set(second):
        raise core.SimulationError("roster_mismatch")
    a, b = _average_ranks(first), _average_ranks(second)
    n = len(a)
    mean = (n + 1) / 2
    va, vb = sum((x - mean) ** 2 for x in a.values()), sum((x - mean) ** 2 for x in b.values())
    if not va or not vb:
        return None
    return max(-1., min(1., sum((a[h] - mean) * (b[h] - mean) for h in a) / math.sqrt(va * vb)))


def calibration_table(value):
    return [dict(row, probability_band=BIN_NAMES[row["bin"]],
                 predicted_to_observed_ratio=row["mean_probability"] / row["observed_rate"] if row["observed_rate"] else None,
                 ratio_status="ok" if row["observed_rate"] else "no_observed_wins" if row["count"] else "empty_bin")
            for row in value["low_probability_calibration"]]


def measure(predictions, winners, quotes, policy):
    if not quotes:
        return {"status": "unavailable", "race_count": 0, "runner_count": 0}
    if len({q.basis for q in quotes}) != 1:
        raise core.SimulationError("invalid_input")
    by_race = {item.race: item for item in predictions}
    if len({item.race for item in quotes}) != len(quotes):
        raise core.SimulationError("odds_conflict")
    cohort, markets, totals, correlations, agreement = [], [], [], [], 0
    for quote in quotes:
        item = by_race[quote.race]
        if set(quote.odds) != set(item.probabilities) or quote.race not in winners:
            raise core.SimulationError("roster_mismatch")
        market, total = implied_probabilities(quote.odds)
        ai, _ = core.probabilities(item.probabilities, policy)
        cohort.append(item)
        markets.append(core.PredictionRace(quote.race, market))
        totals.append(float(total))
        correlation = spearman(ai, market)
        if correlation is not None:
            correlations.append(correlation)
        agreement += min(ai, key=lambda h: (-ai[h], h)) == min(market, key=lambda h: (-market[h], h))
    ai_metrics = v03.probability_evaluation(cohort, winners, policy)
    market_metrics = v03.probability_evaluation(markets, winners, core.WinPolicy(probability_method="identity"))
    ll = market_metrics["model"]["log_loss"]
    return {"status": "ok", "race_count": len(cohort), "runner_count": ai_metrics["model"]["count"],
            "metrics": {"v03_calibrated": ai_metrics["model"], "uniform": ai_metrics["uniform"], "market": market_metrics["model"]},
            "calibration": {"v03_calibrated": calibration_table(ai_metrics), "market": calibration_table(market_metrics)},
            "rank_relationship": {"spearman_per_race": core.distribution(correlations),
                "undefined_spearman_races": len(cohort) - len(correlations), "undefined_reason": "constant_rank_vector",
                "top1_agreement_count": agreement, "top1_agreement_percent": 100 * agreement / len(cohort),
                "top1_agreement_tie_rule": "lowest_horse_number", "spearman_tie_rule": "average_ranks"},
            "timing": {"basis": quotes[0].basis, "published_clock_used_for_final": False,
                "minutes_before_scheduled_start": core.distribution([q.minutes_before_start for q in quotes if q.minutes_before_start is not None]),
                "source_table_counts": dict(Counter(q.source for q in quotes)),
                "category_counts": dict(Counter(q.category for q in quotes)),
                "actual_observation_time_verified": False},
            "sum_inverse_odds": core.distribution(totals), "market_probability_sum_checked": True,
            "cancelled_slots_outside_evaluation_roster": sum(q.cancelled_outside_roster for q in quotes),
            "comparison_with_unmeasured_estimate": {"estimate": .21, "measured_log_loss": ll, "difference": ll - .21,
                "estimate_is_not_a_measured_baseline": True},
            "market_minus_model_log_loss": ll - ai_metrics["model"]["log_loss"],
            "market_minus_model_brier": market_metrics["model"]["brier_score"] - ai_metrics["model"]["brier_score"]}
