"""Explicit local contracts. Unknown DB field meanings are never auto-confirmed."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
import re

from ..payout_decoder import (RawPayoutSlot, SlotKind, classify_payout_layout, parse_payout_layout)
from .core import (JST, Outcome, PredictionRace, Schedule, SimulationError, Snapshot,
                   decimal, horse, instant, key)

ODDS_TABLES = ("jvd_o1", "apd_sokuho_o1")
TIME_FORMATS = ("iso", "timestamp_jst", "timestamptz", "yyyymmddhhmmss", "mmddhhmm", "hhmm_on_race_date")
JV_O1_WIN = {"version": "4.9.0.1", "pdf_page": 15, "printed_page": 12, "record_bytes": 962,
             "published_format": "mmddhhmm_intermediate_only", "win_position_1based": 44,
             "slots": 28, "stride": 8, "horse_width": 2, "price_width": 4, "popularity_width": 2,
             "raw_price_numerator": 1, "raw_price_denominator": 10,
             "capped_token": "9999", "capped_lower_bound": "999.9",
             "non_price_tokens": ["0000", "----", "****", "    "],
             "pre_race_candidate_categories": ["1", "2"]}


def document(value):
    if (not isinstance(value, dict) or value.get("format_version") != 2
            or value.get("source") not in ("synthetic", "local_reviewed")
            or not isinstance(value.get("records"), list) or len(value["records"]) > 500000):
        raise SimulationError("invalid_input")
    return value["records"]


def parse_schedules(value):
    result = []
    for row in document(value):
        try:
            race = key(row["race_key"])
        except (KeyError, TypeError):
            raise SimulationError("invalid_input") from None
        try:
            observed = instant(row["observed_at"]) if row.get("observed_at") is not None else None
            result.append(Schedule(race, instant(row["scheduled_start"]), observed,
                                   tuple(horse(h) for h in row["horses"]), row.get("availability_verified") is True,
                                   row.get("availability_basis", "observed")))
        except (KeyError, TypeError, SimulationError):
            result.append(Schedule(race, None, None, (), False))
    return result


def parse_snapshots(value):
    result = []
    for row in document(value):
        try:
            race = key(row["race_key"])
        except (KeyError, TypeError):
            raise SimulationError("invalid_input") from None
        published, observed = None, None
        try:
            published = instant(row["published_at"])
            observed = instant(row["observed_at"]) if row.get("observed_at") is not None else None
            prices = {}
            for item in row["odds"]:
                h = horse(item["horse_number"])
                if h in prices:
                    raise SimulationError("odds_conflict")
                prices[h] = decimal(item["value"])
            time_reviewed = (row.get("publication_time_reviewed") is True
                             if row.get("timing_basis") == "jra_publication_time"
                             else row.get("availability_verified") is True and observed is not None)
            reason = row.get("reason")
            result.append(Snapshot(race, published, observed, prices,
                                   verified=time_reviewed and row.get("unit_verified") is True,
                                   reason=reason if reason in ("odds_invalid", "odds_conflict", "odds_capped", "unsupported_status", "odds_unverified") else None,
                                   publication_precision=row.get("publication_precision", "second")))
        except (KeyError, TypeError, SimulationError):
            result.append(Snapshot(race, published, observed, {}, verified=False, reason="odds_invalid"))
    return result


def parse_outcomes(value):
    result = []
    for row in document(value):
        try:
            race = key(row["race_key"])
        except (KeyError, TypeError):
            raise SimulationError("invalid_input") from None
        try:
            payouts = {}
            for item in row.get("payouts", []):
                h = horse(item["horse_number"])
                if h in payouts or type(item["yen_per_100"]) is not int:
                    raise SimulationError("result_invalid")
                payouts[h] = item["yen_per_100"]
            status = row["status"]
            if status not in ("normal_final", "refund", "special", "canceled", "no_sale", "unestablished", "pending"):
                raise SimulationError("result_invalid")
            verified = (row.get("reviewed") is True and row.get("currency") == "JPY"
                        and row.get("base_stake_yen") == 100
                        and row.get("evidence_basis") == "official_published_payout")
            result.append(Outcome(race, status, payouts, verified,
                                  row.get("complete_winner_set") is True,
                                  "refund_pending" if status == "refund" else None))
        except (KeyError, TypeError, SimulationError):
            result.append(Outcome(race, "pending", reason="result_invalid"))
    return result


def parse_time(value, encoding, race):
    if encoding == "iso" or encoding == "timestamptz":
        return instant(value)
    if encoding == "timestamp_jst":
        if not isinstance(value, datetime) or value.tzinfo is not None:
            raise SimulationError("invalid_mapping")
        return value.replace(tzinfo=JST)
    widths = {"yyyymmddhhmmss": 14, "mmddhhmm": 8, "hhmm_on_race_date": 4}
    if encoding not in widths or not isinstance(value, str) or not re.fullmatch(r"[0-9]{%d}" % widths[encoding], value):
        raise SimulationError("invalid_mapping")
    text = value if encoding == "yyyymmddhhmmss" else (race[0] + value + "00" if encoding == "mmddhhmm"
                                                       else race[0] + race[1] + value + "00")
    try:
        return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=JST)
    except ValueError:
        raise SimulationError("invalid_mapping") from None


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value):
        raise SimulationError("invalid_mapping")
    return '"' + value + '"'


def validate_source(source):
    try:
        if (not isinstance(source, dict) or source["table"] not in ODDS_TABLES
                or source.get("layout") not in ("per_horse", "packed")):
            raise SimulationError("invalid_mapping")
        columns = source["columns"]
        if set(columns) != {"status", "published", "observed", "horse", "price", "payload"}:
            raise SimulationError("invalid_mapping")
        for name in columns.values():
            if name is not None:
                _identifier(name)
        required = ("status", "observed", "horse", "price") if source["layout"] == "per_horse" else ("status", "observed", "payload")
        if any(not columns[field] for field in required):
            raise SimulationError("invalid_mapping")
        if any(source["time_formats"].get(field) not in TIME_FORMATS for field in ("observed", "published")
               if columns[field]):
            raise SimulationError("invalid_mapping")
        statuses = source["allowed_statuses"]
        if (not isinstance(statuses, list) or not statuses or any(s not in ("1", "2") for s in statuses)):
            raise SimulationError("invalid_mapping")
        encoding = source["price_encoding"]
        if encoding["format"] not in ("digits", "decimal"):
            raise SimulationError("invalid_mapping")
        if encoding["format"] == "digits" and (type(encoding["width"]) is not int or not 1 <= encoding["width"] <= 12):
            raise SimulationError("invalid_mapping")
        if (any(type(encoding[field]) is not int or not 1 <= encoding[field] <= 10000
                for field in ("numerator", "denominator")) or not isinstance(encoding.get("unavailable_tokens"), list)
                or any(not isinstance(s, str) or len(s) > 12 for s in encoding["unavailable_tokens"])):
            raise SimulationError("invalid_mapping")
        if source["layout"] == "packed":
            layout = source["packed"]
            if (any(type(layout[field]) is not int for field in ("slots", "stride", "horse_offset", "horse_width", "price_offset", "price_width"))
                    or not 1 <= layout["slots"] <= 28 or not 1 <= layout["stride"] <= 64
                    or layout["horse_width"] != 2
                    or not isinstance(layout.get("empty_horse_codes"), list)):
                raise SimulationError("invalid_mapping")
            for offset, width in ((layout["horse_offset"], 2), (layout["price_offset"], layout["price_width"])):
                if offset < 0 or width < 1 or offset+width > layout["stride"]:
                    raise SimulationError("invalid_mapping")
            horse_positions = set(range(layout["horse_offset"], layout["horse_offset"]+2))
            price_positions = set(range(layout["price_offset"], layout["price_offset"]+layout["price_width"]))
            if (horse_positions & price_positions or any(not isinstance(code, str) or not re.fullmatch(r"[0-9 ]{2}", code)
                                                         for code in layout["empty_horse_codes"])):
                raise SimulationError("invalid_mapping")
            if encoding["format"] == "digits" and layout["price_width"] != encoding["width"]:
                raise SimulationError("invalid_mapping")
        confirmations = source["confirmed"]
        if any(confirmations.get(field) is not True for field in
               ("official_format", "pc_keiba_mapping", "observed_time_semantics", "odds_unit", "revision_policy")):
            raise SimulationError("mapping_unconfirmed")
        if not all(isinstance(source.get(field), str) and source[field].strip()
                   for field in ("official_reference", "local_mapping_reference")):
            raise SimulationError("mapping_unconfirmed")
    except (KeyError, TypeError, ValueError):
        raise SimulationError("invalid_mapping") from None
    return source


def price(value, encoding):
    if not isinstance(value, str) or value in encoding["unavailable_tokens"]:
        raise SimulationError("odds_invalid")
    regex = r"[0-9]{%d}" % encoding["width"] if encoding["format"] == "digits" else r"[0-9]+(?:\.[0-9]+)?"
    if len(value) > 16 or not re.fullmatch(regex, value):
        raise SimulationError("odds_invalid")
    result = decimal(value) * encoding["numerator"] / encoding["denominator"]
    if result < 1:
        raise SimulationError("odds_invalid")
    if result >= Decimal("999.9"):
        raise SimulationError("odds_capped")
    return result


def decode_odds_rows(rows, source):
    """Rows use fixed aliases from the SQL adapter; no raw row is logged."""
    validate_source(source)
    groups, failures = {}, {}
    for row in rows:
        race = key(row["race_key"])
        stamp, observed = None, None
        try:
            observed = parse_time(row.get("observed"), source["time_formats"]["observed"], race)
            if row.get("status") not in source["allowed_statuses"]:
                # Final O1 records have no intermediate publication clock.
                # Anchor a non-price quarantine record at its known observation
                # time so a later final record cannot poison an earlier quote.
                stamp = (race, observed, observed)
                groups.setdefault(stamp, {"odds": {}, "reason": "unsupported_status", "packed_rows": 0})
                groups[stamp]["reason"] = "unsupported_status"
                continue
            published = (parse_time(row.get("published"), source["time_formats"]["published"], race)
                         if source["columns"]["published"] else observed)
            stamp = (race, published, observed)
            group = groups.setdefault(stamp, {"odds": {}, "reason": None, "packed_rows": 0})
            if row.get("oversized") or observed < published:
                raise SimulationError("odds_invalid")
            if source["layout"] == "per_horse":
                entries = [(row.get("horse"), row.get("price"))]
            else:
                layout, payload = source["packed"], row.get("payload")
                group["packed_rows"] += 1
                if group["packed_rows"] != 1:
                    raise SimulationError("odds_conflict")
                if not isinstance(payload, str) or len(payload) != layout["slots"] * layout["stride"]:
                    raise SimulationError("odds_invalid")
                entries = []
                for index in range(layout["slots"]):
                    slot = payload[index*layout["stride"]:(index+1)*layout["stride"]]
                    h = slot[layout["horse_offset"]:layout["horse_offset"]+2]
                    amount = slot[layout["price_offset"]:layout["price_offset"]+layout["price_width"]]
                    if slot == " " * layout["stride"] or (h in layout["empty_horse_codes"] and amount == " " * layout["price_width"]):
                        continue
                    entries.append((h, amount))
            for h, amount in entries:
                h = horse(h)
                if h in group["odds"]:
                    raise SimulationError("odds_conflict")
                group["odds"][h] = price(amount, source["price_encoding"])
        except SimulationError as exc:
            reason = exc.code if exc.code in ("odds_conflict", "unsupported_status", "odds_capped") else "odds_invalid"
            if stamp is not None and stamp in groups:
                groups[stamp]["reason"] = reason
            elif observed is not None:
                stamp = (race, observed, observed)
                groups.setdefault(stamp, {"odds": {}, "reason": reason, "packed_rows": 0})
                groups[stamp]["reason"] = reason
            else:
                failures[race] = "odds_unverified"
    snapshots = [Snapshot(race, published, observed, group["odds"], source["table"], True, group["reason"])
                 for (race, published, observed), group in groups.items()]
    return snapshots, failures


def validate_payout_source(source):
    try:
        if (source.get("table") != "jvd_hr" or source.get("reviewed_normal_conditions") is not True
                or source.get("currency") != "JPY" or source.get("base_stake_yen") != 100
                or source.get("final_category") not in ("1", "2") or source.get("revision_policy") != "single_reviewed_final"
                or source.get("semantics_reviewed") is not True or not source.get("official_reference")
                or not source.get("local_reconciliation_reference")):
            raise SimulationError("mapping_unconfirmed")
        if (not isinstance(source["years"], list) or not source["years"]
                or any(type(y) is not int or not 1950 <= y <= 2200 for y in source["years"])
                or not isinstance(source["venues"], list) or any(not re.fullmatch(r"0[1-9]|10", v) for v in source["venues"])):
            raise SimulationError("invalid_mapping")
        if set(source["columns"]) != {"status", "unestablished", "special", "refund"}:
            raise SimulationError("invalid_mapping")
        for column in source["columns"].values():
            _identifier(column)
        for field in ("numerator", "denominator"):
            if type(source[field]) is not int or not 1 <= source[field] <= 10000:
                raise SimulationError("invalid_mapping")
    except (KeyError, TypeError, ValueError):
        raise SimulationError("invalid_mapping") from None
    return source


def decode_hr_rows(rows, source):
    """An explicit scoped, locally reviewed normal-win adapter; Phase 1 stays closed."""
    validate_payout_source(source)
    grouped, result = defaultdict(list), []
    for row in rows:
        grouped[key(row["race_key"])].append(row)
    for race, versions in grouped.items():
        reason, payouts = None, {}
        if int(race[0]) not in source["years"] or race[2] not in source["venues"]:
            reason = "outside_scope"
        elif len(versions) != 1:
            reason = "result_duplicate"
        else:
            row = versions[0]
            if row.get("oversized"):
                reason = "result_invalid"
            elif row.get("status") != source["final_category"]:
                reason = "result_revision_unconfirmed"
            elif row.get("refund") == "1":
                reason = "refund_pending"
            elif row.get("unestablished") != "0" or row.get("special") != "0" or row.get("refund") != "0":
                reason = "special_pending"
            else:
                try:
                    layout = classify_payout_layout(parse_payout_layout(wager_type="tansho", slots=tuple(
                        RawPayoutSlot(*(row.get(f"slot_{i}{part}") for part in "abc")) for i in range(1, 4))))
                    if not layout.has_all_slots or layout.format_status != "valid" or not layout.ordinary_candidates:
                        raise SimulationError("result_unconfirmed")
                    for item in layout.slots:
                        if item.kind is SlotKind.ORDINARY_CANDIDATE:
                            # Width/ASCII/leading zero validation above precedes conversion.
                            amount = Decimal(item.raw.b) * source["numerator"] / source["denominator"]
                            if amount != amount.to_integral_value() or not 1 <= amount <= 999999999:
                                raise SimulationError("result_invalid")
                            payouts[item.raw.a] = int(amount)
                except Exception:
                    reason = "result_unconfirmed"
        result.append(Outcome(race, "pending" if reason else "normal_final", payouts if not reason else {},
                              not reason, not reason, reason))
    return result
