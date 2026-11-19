"""Frozen paper-bet rules, trusted as-of prices, and separate outcome evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math

from .features import entry_identity, race_cutoff, rank_labels
from .safety import MAX_ROWS, MLFailure, digest, local_path, read_json, write_json

DECISION_REASONS = frozenset({"odds_unavailable", "odds_after_cutoff", "place_range_unpriced",
                              "below_threshold", "ev_threshold_met", "race_budget_limit"})


@dataclass(frozen=True)
class PurchasePolicy:
    min_expected_return: float = .10
    stake_yen: int = 100
    max_bets_per_race: int = 2

    def validate(self):
        if (isinstance(self.min_expected_return, bool) or not math.isfinite(self.min_expected_return)
                or not 0 <= self.min_expected_return <= 10
                or type(self.stake_yen) is not int or not 100 <= self.stake_yen <= 10000
                or self.stake_yen % 100 or type(self.max_bets_per_race) is not int
                or not 1 <= self.max_bets_per_race <= 56):
            raise MLFailure("invalid_input")

    def as_dict(self):
        self.validate()
        return dict(asdict(self), selection="expected_return_desc_then_wager_then_horse",
                    threshold_source="fixed_before_final_test", production_settlement_enabled=False)

    @classmethod
    def from_dict(cls, value):
        try:
            if (value["threshold_source"] != "fixed_before_final_test"
                    or value["production_settlement_enabled"] is not False):
                raise MLFailure("invalid_input")
            result = cls(value["min_expected_return"], value["stake_yen"], value["max_bets_per_race"])
            result.validate()
            return result
        except (TypeError, KeyError):
            raise MLFailure("invalid_input") from None


def timestamp(value):
    try:
        if not isinstance(value, str):
            raise ValueError
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result
    except (ValueError, TypeError):
        raise MLFailure("invalid_input") from None


def parse_odds(value):
    """Explicit scalar purchase snapshots only. Ranges are preserved as a hold."""
    if (not isinstance(value, dict) or set(value) != {"format_version", "source", "records"}
            or value.get("format_version") != 1 or value.get("source") != "pre_race_snapshot"
            or not isinstance(value.get("records"), list) or len(value["records"]) > MAX_ROWS * 2):
        raise MLFailure("invalid_odds")
    output = {}
    try:
        for row in value["records"]:
            from ..core import RACE_KEY
            fields = set(RACE_KEY) | {"horse_number", "wager", "captured_at", "decision_at", "price", "price_kind"}
            if not isinstance(row, dict) or set(row) != fields or row["wager"] not in ("win", "place"):
                raise MLFailure("invalid_odds")
            key, _ = entry_identity(row)
            key = (*key, row["wager"])
            if key in output:
                raise MLFailure("duplicate_entry")
            captured, decision = timestamp(row["captured_at"]), timestamp(row["decision_at"])
            if captured > decision:
                raise MLFailure("invalid_odds")
            if row["price_kind"] not in ("scalar", "range"):
                raise MLFailure("invalid_odds")
            price = row["price"]
            values = [price] if row["price_kind"] == "scalar" else price
            if (not isinstance(values, list) or len(values) != (1 if row["price_kind"] == "scalar" else 2)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                           or not 1 <= x <= 100000 for x in values)
                    or (len(values) == 2 and values[0] > values[1])):
                raise MLFailure("invalid_odds")
            output[key] = dict(row)
    except MLFailure:
        raise
    except Exception:
        raise MLFailure("invalid_odds") from None
    return output


def paper_decisions(rows, predictions, *, policy=None, odds=None, mode="backtest", now=None):
    policy = PurchasePolicy() if policy is None else policy
    policy.validate()
    odds = {} if odds is None else odds
    now = datetime.now(timezone.utc) if now is None else now
    if mode not in ("backtest", "forward") or now.tzinfo is None:
        raise MLFailure("invalid_input")
    if (set(predictions) != {"win", "place"}
            or any(len(probabilities) != len(rows) for probabilities in predictions.values())):
        raise MLFailure("invalid_input")
    output = []
    for index, row in enumerate(rows):
        key, _ = entry_identity(row)
        cutoff = race_cutoff(row)
        if mode == "forward" and (now >= cutoff or any(row.get(field) not in (None, "")
                                                       for field in ("finish_position", "finish_time_seconds"))):
            raise MLFailure("forward_cutoff")
        for wager in ("win", "place"):
            probability = predictions[wager][index]
            if not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise MLFailure("invalid_input")
            price = odds.get((*key, wager))
            decision_at = now if mode == "forward" else cutoff - timedelta(minutes=5)
            reason, expected, scalar = "odds_unavailable", None, None
            if price is not None:
                captured, specified = timestamp(price["captured_at"]), timestamp(price["decision_at"])
                if mode == "backtest":
                    decision_at = specified
                if captured > decision_at or decision_at >= cutoff or captured >= cutoff:
                    reason = "odds_after_cutoff"
                elif price["price_kind"] != "scalar":
                    reason = "place_range_unpriced"
                else:
                    scalar = float(price["price"])
                    expected = float(probability) * scalar - 1
                    reason = "candidate" if expected >= policy.min_expected_return else "below_threshold"
            output.append({"entry_key": list(key), "wager": wager, "probability": float(probability),
                           "decision_at": decision_at.isoformat(),
                           "race_cutoff_at": cutoff.isoformat(),
                           "odds_captured_at": price["captured_at"] if price else None,
                           "purchase_odds": scalar, "expected_return": expected,
                           "decision": "skip", "reason": reason, "stake_yen": 0})
    grouped = defaultdict(list)
    for item in output:
        if item["reason"] == "candidate":
            grouped[tuple(item["entry_key"][:-1])].append(item)
    for group in grouped.values():
        group.sort(key=lambda item: (-item["expected_return"], item["wager"], item["entry_key"][-1]))
        for index, item in enumerate(group):
            if index < policy.max_bets_per_race:
                item.update(decision="paper_buy", reason="ev_threshold_met", stake_yen=policy.stake_yen)
            else:
                item["reason"] = "race_budget_limit"
    return output


def _distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {"count": len(values), "min": ordered[0], "median": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(len(ordered) * .9))], "max": ordered[-1],
            "mean": sum(values) / len(values)}


def summarize(decisions, *, labels=None):
    labels = {} if labels is None else labels
    if len(decisions) > MAX_ROWS * 2:
        raise MLFailure("input_limit")
    from ..core import RACE_KEY
    seen = set()
    for item in decisions:
        if (not isinstance(item, dict) or item.get("reason") not in DECISION_REASONS
                or item.get("wager") not in ("win", "place")
                or item.get("decision") not in ("skip", "paper_buy")
                or not isinstance(item.get("entry_key"), list) or len(item["entry_key"]) != 7
                or type(item.get("stake_yen")) is not int):
            raise MLFailure("invalid_input")
        entry, _ = entry_identity(dict(zip((*RACE_KEY, "horse_number"), item["entry_key"])))
        key = (*entry, item["wager"])
        if key in seen:
            raise MLFailure("duplicate_entry")
        seen.add(key)
        if ((item["decision"] == "skip" and item["stake_yen"] != 0)
                or (item["decision"] == "paper_buy" and (not 100 <= item["stake_yen"] <= 10000 or item["stake_yen"] % 100))):
            raise MLFailure("invalid_input")
    # None is intentional: no final odds or proxy label becomes an official payment.
    def one(items):
        purchases = [item for item in items if item["decision"] == "paper_buy"]
        known = [labels[(*item["entry_key"], item["wager"])] for item in purchases
                 if labels.get((*item["entry_key"], item["wager"])) is not None]
        return {"prediction_count": len(items),
                "race_count": len({tuple(item["entry_key"][:-1]) for item in items}),
                "paper_bet_count": len(purchases), "investment_yen": sum(item["stake_yen"] for item in purchases),
                "rank_proxy_hit_count": sum(known) if known else None,
                "rank_proxy_known_bets": len(known), "payout_yen": None, "return_rate": None,
                "roi": None, "net_profit_yen": None,
                "financial_status": "not_computed_official_payout_contract_unconfirmed",
                "expected_return_distribution": _distribution([item["expected_return"] for item in items
                                                                 if item["expected_return"] is not None]),
                "decision_reasons": dict(Counter(item["reason"] for item in items))}
    result = one(decisions)
    result["by_wager"] = {wager: one([item for item in decisions if item["wager"] == wager])
                          for wager in ("win", "place")}
    result["by_month"] = {month: one([item for item in decisions if
                                     item["entry_key"][0] + "-" + item["entry_key"][1][:2] == month])
                          for month in sorted({item["entry_key"][0] + "-" + item["entry_key"][1][:2]
                                               for item in decisions})}
    return result


def save_ledger(directory, *, decisions, mode, model_hash, policy, root=None):
    if mode not in ("backtest", "forward"):
        raise MLFailure("invalid_input")
    records = directory / "predictions.json"
    if mode == "forward":
        now = datetime.now(timezone.utc)
        if any(now >= timestamp(item["race_cutoff_at"]) or timestamp(item["decision_at"]) > now
               for item in decisions):
            raise MLFailure("forward_cutoff")
    write_json(records, {"format_version": 1, "mode": mode, "records": decisions}, root=root)
    if mode == "forward" and any(datetime.now(timezone.utc) >= timestamp(item["race_cutoff_at"]) for item in decisions):
        # No completion manifest exists, so this file is never a valid forecast.
        raise MLFailure("forward_cutoff")
    manifest = {"format_version": 1, "mode": mode,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_manifest_sha256": model_hash, "purchase_policy": policy.as_dict(),
                "predictions_sha256": digest(records, root=root),
                "integrity": "exclusive_write_and_sha256_not_trusted_timestamp",
                "production_hr_settlement_enabled": False}
    write_json(directory / "manifest.json", manifest, root=root)
    # Application never updates prediction files. Windows read-only attribute is an
    # additional accident guard, not a claim of tamper-proof storage or a WORM device.
    for path in (records, directory / "manifest.json"):
        try:
            path.chmod(0o444)
        except OSError:
            pass
    return manifest


def evaluate_ledger(directory, outcomes, *, root=None):
    manifest = read_json(directory / "manifest.json", root=root)
    if (manifest.get("format_version") != 1 or manifest.get("production_hr_settlement_enabled") is not False
            or digest(directory / "predictions.json", root=root) != manifest.get("predictions_sha256")):
        raise MLFailure("ledger_tampered")
    ledger = read_json(directory / "predictions.json", root=root)
    if ledger.get("mode") != manifest.get("mode") or ledger.get("format_version") != 1:
        raise MLFailure("ledger_tampered")
    decisions = ledger["records"]
    labels, _ = rank_labels(outcomes)
    lookup = {(*entry_identity(row)[0], wager): label[wager]
              for row, label in zip(outcomes, labels) for wager in ("win", "place")}
    result = summarize(decisions, labels=lookup)
    result["mode"] = manifest["mode"]
    from .model import probability_metrics
    result["probability_metrics"] = {}
    for wager in ("win", "place"):
        scored = [item for item in decisions if item["wager"] == wager
                  and lookup.get((*item["entry_key"], wager)) is not None]
        result["probability_metrics"][wager] = probability_metrics(
            [lookup[(*item["entry_key"], wager)] for item in scored],
            [item["probability"] for item in scored]) if scored else None
    result["missing_outcome_predictions"] = sum(lookup.get((*item["entry_key"], item["wager"])) is None
                                               for item in decisions)
    return result
