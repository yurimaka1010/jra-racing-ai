"""Small pure helpers for fixed-cohort nested windows and race-paired bootstrap."""
from collections import Counter, defaultdict
from datetime import date, timedelta
import math

from . import features, safety, v03
from ..win_sim import core


def validate_plan(plan):
    names = ("history_start", "train_start", "train_end", "validation_start", "validation_end",
             "calibration_start", "calibration_end", "test_start", "test_end")
    try:
        days = {name: date.fromisoformat(plan[name]) for name in names}
        if (not days["history_start"] + timedelta(days=365) <= days["train_start"] <= days["train_end"]
                or not days["train_end"] + timedelta(days=7) < days["validation_start"] <= days["validation_end"]
                or not days["validation_end"] + timedelta(days=7) < days["calibration_start"] <= days["calibration_end"]
                or not days["calibration_end"] + timedelta(days=7) < days["test_start"] <= days["test_end"]
                or plan["history_lag_days"] != 7 or plan["max_rounds"] != 100
                or plan["temperature_series"]["recalibrated_candidates"] != [.75, 1., 1.25, 2.]):
            raise ValueError
        return days
    except (KeyError, TypeError, ValueError):
        raise safety.MLFailure("invalid_split") from None


def prepare_cohorts(rows, matrix, plan):
    days = validate_plan(plan)
    winners = v03.single_winners(rows)
    grouped, excluded = defaultdict(list), Counter()
    for i, row in enumerate(rows):
        identity, _ = features.entry_identity(row)
        grouped[identity[:-1]].append(i)
    eligible = {}
    for race, indices in grouped.items():
        if race not in winners:
            excluded["missing_rank_or_multiple_winners"] += 1
        elif any(matrix[i][name] is None for i in indices for name in v03.PRIORITY_FEATURES):
            excluded["missing_priority_feature"] += 1
        else:
            eligible[race] = indices
    parts = {}
    bounds = {"train": (days["train_start"], days["train_end"]),
              "validation": (days["validation_start"], days["validation_end"]),
              "calibration": (days["calibration_start"], days["calibration_end"]),
              "test": (days["test_start"], days["test_end"]),
              "reference": (date.fromisoformat(plan["reference_start"]), date.fromisoformat(plan["reference_end"]))}
    for name, (start, end) in bounds.items():
        parts[name] = [race for race in sorted(eligible) if start <= features.race_identity(dict(zip(core.RACE_KEY, race)))[1] <= end]
    if not all(parts.values()):
        raise safety.MLFailure("insufficient_history")
    if len(parts["validation"]) < 150 or len(parts["calibration"]) < 150 or len(parts["test"]) < 200:
        raise safety.MLFailure("insufficient_history")
    return parts, eligible, winners, dict(excluded)


def nested_windows(races, requested=(144, 500, 1000, 2000, 5000)):
    ordered = sorted(races)
    sizes = sorted({size for size in requested if 0 < size <= len(ordered)} | {len(ordered)})
    if not ordered:
        raise safety.MLFailure("insufficient_history")
    return [(size, tuple(ordered[-size:])) for size in sizes]


def race_scores(predictions, winners, policy):
    result = {}
    for item in predictions:
        if item.race not in winners:
            continue
        p, _ = core.probabilities(item.probabilities, policy)
        winner = winners[item.race]
        if winner not in p:
            raise safety.MLFailure("invalid_identity")
        loss, brier = 0., 0.
        for horse, value in p.items():
            probability = float(value)
            clipped = max(1e-15, min(1 - 1e-15, probability))
            y = int(horse == winner)
            loss -= y * math.log(clipped) + (1 - y) * math.log(1 - clipped)
            brier += (probability - y) ** 2
        best = max(p.values())
        top = 1 / sum(value == best for value in p.values()) if p[winner] == best else 0.
        result[item.race] = (loss, brier, len(p), top)
    return result


def paired_bootstrap(new, baseline, *, repetitions=1000, seed=20261018):
    import numpy as np
    keys = sorted(new)
    if (not keys or set(new) != set(baseline) or repetitions < 100
            or any(new[key][2] != baseline[key][2] for key in keys)):
        raise safety.MLFailure("invalid_input")
    a = np.asarray([new[k] for k in keys], dtype=float)
    b = np.asarray([baseline[k] for k in keys], dtype=float)
    sample = np.random.default_rng(seed).integers(0, len(keys), (repetitions, len(keys)))
    # Weight races by their horse counts AFTER resampling, preserving the
    # established horse-binary loss rather than averaging race mean losses.
    n = a[sample, 2].sum(axis=1)
    values = {
        "log_loss": ((a[sample, 0] - b[sample, 0]).sum(axis=1) / n,
                     (a[:, 0].sum() - b[:, 0].sum()) / a[:, 2].sum()),
        "brier_score": ((a[sample, 1] - b[sample, 1]).sum(axis=1) / n,
                        (a[:, 1].sum() - b[:, 1].sum()) / a[:, 2].sum()),
        "top1_accuracy_percentage_points": ((a[sample, 3] - b[sample, 3]).mean(axis=1) * 100,
                                           (a[:, 3] - b[:, 3]).mean() * 100),
    }
    result = {}
    for name, (distribution, point) in values.items():
        low, high = np.quantile(distribution, [.025, .975], method="linear")
        result[name] = {"difference_new_minus_baseline": float(point), "ci95_low": float(low), "ci95_high": float(high)}
    ll = result["log_loss"]
    result.update(repetitions=repetitions, seed=seed, resampling_unit="race_cluster_paired",
        interpretation="improvement_supported" if ll["ci95_high"] < 0 else
                       "worsening_supported" if ll["ci95_low"] > 0 else "uncertain")
    return result


def probability_distribution(predictions, policy):
    return core.distribution([float(value) for item in predictions
        for value in core.probabilities(item.probabilities, policy)[0].values()])
