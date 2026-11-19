"""Version 0.3 helpers: reviewed features, causal history and existing win pipeline."""
from collections import defaultdict
import math
import sys

from . import data, features, model, safety
from ..win_sim import core, formats, local, settle_saved

PRIORITY_FEATURES = ("venue", "surface", "distance", "age", "weight_kg")
BASIC_FIELDS = ("horse_id", "surface", "distance", "declared_runners", "frame", "age", "sex", "weight_kg")


def feature_mapping(catalog):
    columns = {name: None for name in data.SOURCES}
    for field in (*data.REQUIRED, *BASIC_FIELDS):
        candidates = (("toroku_tosu",) if field == "declared_runners" else data.CANDIDATES[field])
        available = {item["name"] for item in catalog["tables"][data.SOURCES[field]]
                     if item.get("type_is_builtin") is True and item.get("type_name") in data.BUILTINS}
        matches = [column for column in candidates if column in available]
        if len(matches) != 1:
            raise safety.MLFailure("invalid_mapping")
        columns[field] = matches[0]
    return {"format_version": 1, "reviewed": False, "columns": columns,
            "pre_race_fields": sorted(set(BASIC_FIELDS) & data.PRE_RACE),
            "scales": {"weight_kg": .1}, "history_lag_days": 7}


def confirm_mapping(candidate):
    print("Version 0.3: 列名と単位だけを確認します。実レコードは表示しません。", file=sys.stderr)
    for field, column in candidate["columns"].items():
        if column:
            print(field + " -> " + column, file=sys.stderr)
    print("負担重量はJV原形式0.1kgをPC列が保持する場合の×0.1です。登録頭数toroku_tosuを使用します。"
          "レース前の宣言情報として使える列対応・単位を本人PCで確認できた場合だけ "
          "CONFIRM_V03_FEATURES と入力してください。", file=sys.stderr)
    try:
        if input() != "CONFIRM_V03_FEATURES":
            raise safety.MLFailure("mapping_not_reviewed")
    except EOFError:
        raise safety.MLFailure("mapping_not_reviewed") from None
    return data.validate_mapping(dict(candidate, reviewed=True))


def coverage(matrix):
    count = len(matrix)
    missing = {name: sum(row[name] is None for row in matrix) for name in features.FEATURES}
    return {"rows": count, "missing_count": missing,
            "missing_rate": {name: value / count if count else None for name, value in missing.items()},
            "without_past_run": sum(row["past_starts"] == 0 for row in matrix)}


def check_training_coverage(value):
    if (not value["rows"] or any(value["missing_count"][name] == value["rows"] for name in PRIORITY_FEATURES)
            or value["missing_count"]["previous_finish"] == value["rows"]):
        raise safety.MLFailure("insufficient_history")


def prediction_races(rows, probabilities):
    if len(rows) != len(probabilities):
        raise safety.MLFailure("invalid_input")
    grouped = defaultdict(dict)
    for row, probability in zip(rows, probabilities):
        identity, _ = features.entry_identity(row)
        if identity[-1] in grouped[identity[:-1]]:
            raise safety.MLFailure("duplicate_entry")
        grouped[identity[:-1]][identity[-1]] = float(probability)
    return [core.PredictionRace(race, values) for race, values in sorted(grouped.items())]


def single_winners(rows):
    labels, _ = features.rank_labels(rows)
    grouped = defaultdict(list)
    for row, label in zip(rows, labels):
        identity, _ = features.entry_identity(row)
        grouped[identity[:-1]].append((identity[-1], label["win"]))
    return {race: next(horse for horse, label in items if label == 1) for race, items in grouped.items()
            if all(label is not None for _, label in items) and sum(label == 1 for _, label in items) == 1}


def outcome_winners(outcomes, predictions):
    grouped = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.race].append(outcome)
    return {item.race: next(iter(grouped[item.race][0].payouts_yen_per_100)) for item in predictions
            if len(grouped[item.race]) == 1
            and core.outcome_reason(grouped[item.race][0], tuple(item.probabilities)) is None
            and len(grouped[item.race][0].payouts_yen_per_100) == 1}


def probability_evaluation(predictions, winners, policy):
    labels, probabilities, uniform = [], [], []
    losses, uniform_losses, spreads = [], [], []
    top1, winner_ranks, field_sizes = [], [], []
    bins = [{"count": 0, "probability_sum": 0., "wins": 0} for _ in range(6)]
    for item in predictions:
        winner = winners.get(item.race)
        if winner is None:
            continue
        normalized, _ = core.probabilities(item.probabilities, policy)
        if winner not in normalized:
            raise safety.MLFailure("invalid_identity")
        losses.append(-math.log(max(1e-15, float(normalized[winner]))))
        uniform_losses.append(math.log(len(normalized)))
        spreads.append(float(max(normalized.values()) - min(normalized.values())))
        winner_p = normalized[winner]
        tied_top = sum(value == max(normalized.values()) for value in normalized.values())
        top1.append(1 / tied_top if winner_p == max(normalized.values()) else 0.)
        winner_ranks.append(1 + sum(value > winner_p for value in normalized.values())
                            + (sum(value == winner_p for value in normalized.values()) - 1) / 2)
        field_sizes.append(len(normalized))
        for horse, value in normalized.items():
            p = float(value)
            label = int(horse == winner)
            labels.append(label)
            probabilities.append(p)
            uniform.append(1 / len(normalized))
            index = next((i for i, upper in enumerate((.01, .02, .05, .1, .2, 1.01)) if p < upper), 5)
            bins[index]["count"] += 1
            bins[index]["probability_sum"] += p
            bins[index]["wins"] += label
    if not labels:
        return {"status": "unavailable", "races": 0, "excluded_races": len(predictions)}
    def brief(values):
        return {k: v for k, v in model.probability_metrics(labels, values).items()
                if k in ("count", "log_loss", "brier_score")}
    return {"status": "ok", "races": len(losses), "excluded_races": len(predictions) - len(losses),
            "model": dict(brief(probabilities), race_log_loss=sum(losses) / len(losses),
                          top1_accuracy_percent=100 * sum(top1) / len(top1), winner_mean_rank=sum(winner_ranks) / len(winner_ranks)),
            "uniform": dict(brief(uniform), race_log_loss=sum(uniform_losses) / len(uniform_losses),
                            top1_accuracy_percent=100 * sum(1 / n for n in field_sizes) / len(field_sizes),
                            winner_mean_rank=sum((n + 1) / 2 for n in field_sizes) / len(field_sizes)),
            "race_probability_spread": core.distribution(spreads),
            "low_probability_calibration": [dict(bin=i, count=value["count"], wins=value["wins"],
                mean_probability=value["probability_sum"] / value["count"] if value["count"] else None,
                observed_rate=value["wins"] / value["count"] if value["count"] else None) for i, value in enumerate(bins)]}


def ev_check(ledger):
    ps, odds, products, returns, top_odds, failures = [], [], [], [], 0, 0
    for row in ledger["rows"]:
        if row["ai_horse"] is None:
            continue
        runner = next(item for item in row["runners"] if item["horse_number"] == row["ai_horse"])
        p, price = core.decimal(runner["probability"]), core.decimal(runner["decision_odds"])
        calculated = p * price - 1
        failures += abs(calculated - core.decimal(runner["expected_return"])) > core.decimal("0.00000001")
        ps.append(float(p))
        odds.append(float(price))
        products.append(float(p * price))
        returns.append(float(calculated))
        top_odds += price == max(core.decimal(item["decision_odds"]) for item in row["runners"])
    if failures:
        raise safety.MLFailure("invalid_odds")
    return {"checked_bets": len(ps), "formula_mismatch_count": 0, "formula": "probability * decision_odds - 1",
            "selected_probability": core.distribution(ps), "decision_odds": core.distribution(odds),
            "probability_times_odds": core.distribution(products), "expected_return": core.distribution(returns),
            "highest_odds_selected_count": top_odds}


def predict_win(bundle, rows, history):
    # Remove the target identities before blanking labels, avoiding a conflicting
    # duplicate with the final-result history. The generator still enforces date+lag.
    identities = {features.entry_identity(row)[0] for row in rows}
    safe_history = [row for row in history if features.entry_identity(row)[0] not in identities]
    entries = [dict(row, finish_position="", finish_time_seconds="") for row in rows]
    matrix = features.build_features(entries, history=safe_history, history_lag_days=bundle["metadata"]["history_lag_days"],
                                     feature_profile=bundle['metadata'].get('feature_profile'))
    values = bundle["boosters"]["win"].predict(model.frame(matrix, bundle["metadata"]["categories"]), num_threads=4)
    return prediction_races(entries, values), coverage(matrix)


def reference_inputs(simulation_id):
    directory, ledger, policy_document, _, manifest = settle_saved.load_saved(simulation_id)
    provenance = manifest["provenance"]
    for name, field in (("hr_settlement_review.json", "hr_review_sha256"), ("hr_payout_source.json", "hr_source_sha256")):
        if provenance.get(field) != local.sha(directory / name):
            raise safety.MLFailure("artifact_invalid")
    source = local.read(directory / "hr_payout_source.json")
    formats.validate_payout_source(source)
    outcomes = settle_saved.decode_existing_hr(local.read(directory / "hr_settlement_review.json")["hr_rows"], source)
    parent, _, _, _, parent_manifest = settle_saved.load_saved(provenance["parent_simulation_id"])
    for name in ("input-schedules.json", "input-odds.json"):
        if parent_manifest["provenance"]["replay_input_sha256"].get(name) != local.sha(parent / name):
            raise safety.MLFailure("artifact_invalid")
    return {"ledger": ledger, "policy_document": policy_document, "outcomes": outcomes,
            "schedules": formats.parse_schedules(local.read(parent / "input-schedules.json")),
            "snapshots": formats.parse_snapshots(local.read(parent / "input-odds.json")),
            "ledger_sha256": local.sha(directory / "ledger.json"), "manifest_sha256": local.sha(directory / "manifest.json")}


def comparison(old, new):
    old_rows = {core.key(row["race_key"]): row for row in old["rows"] if row["common_settlement_eligible"]}
    new_rows = {core.key(row["race_key"]): row for row in new["rows"] if row["common_settlement_eligible"]}
    shared = sorted(set(old_rows) & set(new_rows))
    return {"common_race_count": len(shared),
            "v02_ai": core.financial([old_rows[key] for key in shared], "ai"),
            "v03_ai": core.financial([new_rows[key] for key in shared], "ai"),
            "favorite": core.financial([new_rows[key] for key in shared], "favorite")}
