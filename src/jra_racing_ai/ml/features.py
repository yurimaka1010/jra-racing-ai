"""Causal features and explicit rank-proxy labels; no I/O or model imports."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import math
import re

from ..core import RACE_KEY
from .safety import MAX_ROWS, MLFailure

CATEGORY_FEATURES = ("venue", "surface", "going", "sex")
NUMERIC_FEATURES = (
    "distance", "declared_runners", "frame", "horse_number", "age", "weight_kg",
    "past_starts", "days_since_previous", "previous_finish", "previous_time_seconds",
    "past_mean_finish", "same_distance_mean_finish", "same_course_mean_finish",
)
FEATURES = CATEGORY_FEATURES + NUMERIC_FEATURES
OPTIONAL_FIELDS = ("horse_id", "surface", "going", "distance", "declared_runners",
                   "frame", "age", "sex", "weight_kg", "finish_time_seconds", "start_time")
DATA_FIELDS = RACE_KEY + ("horse_number", "finish_position") + OPTIONAL_FIELDS


def number(value, *, low=None, high=None):
    if value in (None, ""):
        return None
    try:
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(value)):
            return None
        result = float(value)
        if not math.isfinite(result) or (low is not None and result < low) or (high is not None and result > high):
            return None
        return result
    except (ValueError, TypeError, OverflowError):
        return None


def race_identity(row):
    values = tuple(row.get(key) for key in RACE_KEY)
    widths = (4, 4, 2, 2, 2, 2)
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9]{%d}" % w, v)
           for v, w in zip(values, widths)):
        raise MLFailure("invalid_identity")
    year, month_day, venue, meeting, day, race = values
    try:
        race_date = date(int(year), int(month_day[:2]), int(month_day[2:]))
    except ValueError:
        raise MLFailure("invalid_date") from None
    if (not 1950 <= int(year) <= 2200 or not 1 <= int(venue) <= 10
            or not 1 <= int(meeting) <= 99 or not 1 <= int(day) <= 99 or not 1 <= int(race) <= 12):
        raise MLFailure("invalid_identity")
    return values, race_date


def entry_identity(row):
    key, race_date = race_identity(row)
    horse = row.get("horse_number")
    if not isinstance(horse, str) or not re.fullmatch(r"[0-9]{2}", horse) or not 1 <= int(horse) <= 28:
        raise MLFailure("invalid_identity")
    return (*key, horse), race_date


def race_cutoff(row):
    _, day = race_identity(row)
    clock = row.get("start_time", "")
    # Without an independently reviewed scheduled time, midnight is conservative.
    if clock in (None, ""):
        return datetime.combine(day, datetime.min.time(), tzinfo=timezone(timedelta(hours=9)))
    if not isinstance(clock, str) or not re.fullmatch(r"[0-9]{4}", clock):
        raise MLFailure("invalid_date")
    try:
        return datetime(day.year, day.month, day.day, int(clock[:2]), int(clock[2:]),
                        tzinfo=timezone(timedelta(hours=9)))
    except ValueError:
        raise MLFailure("invalid_date") from None


def validate_rows(rows):
    if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
        raise MLFailure("input_limit")
    seen, race_values = set(), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) - set(DATA_FIELDS):
            raise MLFailure("invalid_input")
        key, _ = entry_identity(row)
        if key in seen:
            raise MLFailure("duplicate_entry")
        seen.add(key)
        # Race-level features must agree across runners; no silent duplicate joins.
        common = tuple(row.get(k, "") for k in ("surface", "going", "distance", "declared_runners", "start_time"))
        if key[:-1] in race_values and race_values[key[:-1]] != common:
            raise MLFailure("invalid_input")
        race_values[key[:-1]] = common
        race_cutoff(row)
    return rows


def _mean(items):
    return sum(items) / len(items) if items else None


def build_features(rows, *, history=(), history_lag_days=7, feature_profile=None):
    if feature_profile not in (None,'conditioned_previous_time_v1'):
        raise MLFailure('invalid_model')
    """History is available only on date+lag and strictly before target date.

    A date-batched sweep prevents same-day and current-result contamination.
    Race results are kept out of the returned feature dictionaries.
    """
    validate_rows(rows)
    if type(history_lag_days) is not int or not 1 <= history_lag_days <= 365:
        raise MLFailure("invalid_input")
    combined = {}
    for row in [*history, *rows]:
        key, day = entry_identity(row)
        if key in combined and combined[key] != row:
            raise MLFailure("duplicate_entry")
        combined[key] = row
    if len(combined) > MAX_ROWS:
        raise MLFailure("input_limit")
    candidates = []
    for row in combined.values():
        _, day = entry_identity(row)
        rank = number(row.get("finish_position"), low=1, high=28)
        if row.get("horse_id") and rank is not None and rank.is_integer():
            candidates.append((day + timedelta(days=history_lag_days), day, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    targets = sorted(enumerate(rows), key=lambda item: entry_identity(item[1])[1])
    available, cursor, output = defaultdict(list), 0, [None] * len(rows)
    for index, row in targets:
        key, day = entry_identity(row)
        while cursor < len(candidates) and candidates[cursor][0] <= day:
            _, old_day, old = candidates[cursor]
            available[old["horse_id"]].append((old_day, old))
            cursor += 1
        past = available.get(row.get("horse_id"), [])
        previous = past[-1] if past else None
        distance = number(row.get("distance"), low=100, high=10000)
        ranks = [number(old.get("finish_position"), low=1, high=28) for _, old in past]
        matching_distance = [number(old.get("finish_position"), low=1, high=28) for _, old in past
                             if distance is not None and number(old.get("distance")) == distance]
        matching_course = [number(old.get("finish_position"), low=1, high=28) for _, old in past
                           if distance is not None and number(old.get("distance")) == distance
                           and old.get("keibajo_code") == row["keibajo_code"]
                           and row.get("surface") and old.get("surface") == row.get("surface")]
        previous_time=number(previous[1].get('finish_time_seconds'),low=1,high=1000) if previous else None
        if feature_profile=='conditioned_previous_time_v1':
            previous_time=next((number(old.get('finish_time_seconds'),low=1,high=1000)
                for _,old in reversed(past) if distance is not None
                and number(old.get('distance'))==distance and row.get('surface')
                and old.get('surface')==row['surface']
                and number(old.get('finish_time_seconds'),low=1,high=1000) is not None),None)
        output[index] = {
            "venue": key[2], "surface": row.get("surface") or None,
            "going": row.get("going") or None, "sex": row.get("sex") or None,
            "distance": distance,
            "declared_runners": number(row.get("declared_runners"), low=1, high=28),
            "frame": number(row.get("frame"), low=1, high=8),
            "horse_number": int(key[-1]), "age": number(row.get("age"), low=2, high=30),
            "weight_kg": number(row.get("weight_kg"), low=20, high=100),
            "past_starts": len(past),
            "days_since_previous": (day - previous[0]).days if previous else None,
            "previous_finish": number(previous[1].get("finish_position"), low=1, high=28) if previous else None,
            "previous_time_seconds": previous_time,
            "past_mean_finish": _mean(ranks), "same_distance_mean_finish": _mean(matching_distance),
            "same_course_mean_finish": _mean(matching_course),
        }
    return output


def rank_labels(rows):
    """Rank proxies, NOT verified HR ticket labels (dead heats/specials unresolved)."""
    validate_rows(rows)
    counts = Counter(entry_identity(row)[0][:-1] for row in rows)
    labels, reasons = [], Counter()
    for row in rows:
        key, _ = entry_identity(row)
        rank = number(row.get("finish_position"), low=1, high=28)
        if rank is None or not rank.is_integer():
            labels.append({"win": None, "place": None})
            reasons["rank_unavailable"] += 1
            continue
        # Observed SE runner count is label-only; never a pre-race feature.
        count = counts[key[:-1]]
        paid = 3 if count >= 8 else 2 if count >= 5 else None
        labels.append({"win": int(rank == 1), "place": int(rank <= paid) if paid else None})
        if paid is None:
            reasons["place_proxy_field_too_small"] += 1
    return labels, dict(reasons)


def chronological_split(rows, *, train_end=None, valid_end=None, test_end=None):
    dates = sorted({entry_identity(row)[1] for row in rows})
    if len(dates) < 3:
        raise MLFailure("invalid_split")
    if (train_end is None) != (valid_end is None):
        raise MLFailure("invalid_split")
    if train_end is None:
        first = max(1, min(len(dates) - 2, int(len(dates) * .70)))
        second = max(first + 1, min(len(dates) - 1, int(len(dates) * .85)))
        train_end, valid_end = dates[first - 1], dates[second - 1]
    if not isinstance(train_end, date) or not isinstance(valid_end, date) or train_end >= valid_end:
        raise MLFailure("invalid_split")
    if test_end is not None and (not isinstance(test_end, date) or test_end <= valid_end):
        raise MLFailure("invalid_split")
    result = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        _, day = entry_identity(row)
        if test_end is not None and day > test_end:
            continue
        name = "train" if day <= train_end else "validation" if day <= valid_end else "test"
        result[name].append(index)
    if not all(result.values()):
        raise MLFailure("invalid_split")
    return result
