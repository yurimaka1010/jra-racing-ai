"""Bounded local O1 publication-time replay. No observation times are invented."""
from collections import Counter, defaultdict
from datetime import date, datetime
import re

from ..core import RACE_KEY
from ..ml import data
from . import core, formats, local

TIMING_BASIS = "jra_publication_time"
PRICE_ENCODING = {"format": "digits", "width": 4, "numerator": 1, "denominator": 10,
                  "unavailable_tokens": ["0000", "----", "****", "    "]}
RA_FIELDS = {"record_type": ("record_id", 2), "status": ("data_kubun", 1),
             "created": ("data_sakusei_nengappi", 8), "start": ("hasso_jikoku", 4),
             "previous_start": ("hasso_jikoku_henkomae", 4)}
O1_FIELDS = {"record_type": ("record_id", 2), "status": ("data_kubun", 1),
             "published": ("happyo_tsukihi_jifun", 8), "sale": ("hatsubai_flag_tansho", 1),
             "payload": ("odds_tansho", 224)}


def validate_entries(path, predictions, prediction_manifest):
    rows, metadata = data.read_dataset(path, allow_prediction_snapshot=True)
    expected = {tuple(prediction.race): set(prediction.probabilities) for prediction in predictions}
    actual = defaultdict(set)
    for row in rows:
        identity, _ = data.entry_identity(row)
        actual[identity[:-1]].add(identity[-1])
    if (metadata.get("source") != data.ENTRY_SOURCE or metadata.get("purpose") != "prediction_only"
            or metadata.get("current_results_excluded") is not True or dict(actual) != expected
            or metadata.get("source_status_counts") != {"ra_races": {"2": len(expected)}, "se_runners": {"2": len(rows)}}
            or prediction_manifest.get("entries_source") != data.ENTRY_SOURCE
            or prediction_manifest.get("entries_sha256") != local.sha(path, root=local.old_io.ML_ROOT)):
        raise core.SimulationError("artifact_invalid")
    return metadata


def _columns(db, table, fields, budget):
    columns = local.table_info(db, table, budget)
    if columns is None:
        return None
    if any(columns.get(physical, {}).get("type_is_builtin") is not True
           or columns[physical].get("type_name") not in local.TEXT_TYPES for physical, _ in fields.values()):
        raise core.SimulationError("invalid_schema")
    return columns


def fetch(db, races, *, table="apd_sokuho_o1", max_rows=20000, max_seconds=60, include_ra=True):
    if table not in formats.ODDS_TABLES or not races or len(races) > 50:
        raise core.SimulationError("input_limit")
    budget = local.Budget(max_seconds)
    ra_rows, odds_rows, status = [], [], []
    if include_ra:
        columns = _columns(db, "jvd_ra", RA_FIELDS, budget)
        if columns is not None:
            ra_rows = local._rows(db, budget, "jvd_ra", RA_FIELDS, columns, races, max_rows)
        status.append({"table": "jvd_ra", "status": "bounded_read_complete" if columns is not None else "not_found",
                       "rows": len(ra_rows)})
    remaining = max_rows - len(ra_rows)
    if remaining < 1:
        raise core.SimulationError("input_limit")
    columns = _columns(db, table, O1_FIELDS, budget)
    if columns is not None:
        odds_rows = local._rows(db, budget, table, O1_FIELDS, columns, races, remaining)
    status.append({"table": table, "status": "bounded_read_complete" if columns is not None else "not_found",
                   "rows": len(odds_rows)})
    return ra_rows, odds_rows, status


def schedules_from_ra(rows, predictions):
    grouped = defaultdict(list)
    for row in rows:
        grouped[core.key(row["race_key"])].append(row)
    schedules, counts = [], Counter()
    for prediction in predictions:
        race = core.key(prediction.race)
        versions = grouped[race]
        if not versions:
            counts["schedule_unavailable"] += 1
            continue
        try:
            if len(versions) != 1:
                raise core.SimulationError("schedule_conflict")
            row = versions[0]
            day = date(int(race[0]), int(race[1][:2]), int(race[1][2:]))
            created = row.get("created")
            # Date alone is not availability proof: also require the official
            # category-2 advance card, unchanged scheduled clock, and entries-2.
            if (row.get("oversized") or row.get("record_type") != "RA" or row.get("status") != "2"
                    or not isinstance(created, str) or not re.fullmatch(r"[0-9]{8}", created)
                    or datetime.strptime(created, "%Y%m%d").date() >= day
                    or row.get("previous_start") not in (None, "", "0000", "    ")
                    or row.get("start") == "0000"):
                raise core.SimulationError("schedule_unverified")
            start = formats.parse_time(row.get("start"), "hhmm_on_race_date", race)
            schedules.append(core.Schedule(race, start, None, tuple(prediction.probabilities), True, "ra_advance_card"))
            counts["advance_card_schedule"] += 1
        except (core.SimulationError, ValueError, TypeError):
            counts["schedule_unverified"] += 1
            schedules.append(core.Schedule(race, None, None, (), False, "ra_advance_card"))
    return schedules, dict(counts)


def _raw_prices(payload):
    if not isinstance(payload, str) or len(payload) != 224 or not payload.isascii():
        raise core.SimulationError("odds_invalid")
    prices = {}
    for index in range(28):
        slot = payload[index * 8:(index + 1) * 8]
        if slot == " " * 8:
            continue
        number, amount, popularity = slot[:2], slot[2:6], slot[6:]
        if number != f"{index + 1:02d}":
            raise core.SimulationError("odds_invalid")
        # Official blank odds/popularity mean unregistered. If that horse is in
        # the reviewed roster the later exact-set check still excludes the race.
        if amount == " " * 4 and popularity == " " * 2:
            continue
        value = formats.price(amount, PRICE_ENCODING)
        if not re.fullmatch(r"[0-9]{2}", popularity) or not 1 <= int(popularity) <= 28:
            raise core.SimulationError("odds_invalid")
        prices[core.horse(number)] = value
    return prices


def decode_odds(rows, *, source="apd_sokuho_o1", format_reviewed=False):
    grouped, failures, statistics, examples = {}, {}, Counter(), []
    for row in rows:
        race = core.key(row["race_key"])
        status = row.get("status")
        statistics["category_" + status if status in ("0", "1", "2", "3", "4", "5", "9") else "category_unknown"] += 1
        # O1's publication clock is defined for intermediate records only.
        # Never timestamp a final record using its creation date or race time.
        if status != "1":
            if status not in ("2", "3", "4", "5"):
                failures[race] = "odds_unverified"
            continue
        try:
            published = formats.parse_time(row.get("published"), "mmddhhmm", race)
        except core.SimulationError:
            failures[race] = "odds_unverified"
            statistics["invalid_publication_clock"] += 1
            continue
        stamp, prices, reason = (race, published), {}, None
        try:
            if row.get("oversized") or row.get("record_type") != "O1":
                raise core.SimulationError("odds_invalid")
            if row.get("sale") != "7":
                raise core.SimulationError("unsupported_status")
            prices = _raw_prices(row.get("payload"))
        except core.SimulationError as exc:
            reason = exc.code if exc.code in ("odds_capped", "unsupported_status") else "odds_invalid"
            statistics[reason] += 1
        if stamp in grouped:
            reason = "odds_conflict"
            statistics[reason] += 1
        if isinstance(row.get("payload"), str) and len(row["payload"]) <= 224:
            priced_candidate = reason is None and bool(prices)
            if priced_candidate and examples and not examples[0]["normal_priced_candidate"]:
                examples.clear()
            if len(examples) < 3 and (priced_candidate or not examples or not examples[0]["normal_priced_candidate"]):
                examples.append({"race_key": list(race), "published_at": published.isoformat(),
                    "raw_win_block": row["payload"], "normal_priced_candidate": priced_candidate,
                    "interpretation": "raw_fields_pending_local_format_review"})
        grouped[stamp] = core.Snapshot(race, published, None, prices, source, format_reviewed, reason, "minute")
    statistics["structurally_valid_intermediate_snapshots"] = sum(item.reason is None for item in grouped.values())
    return list(grouped.values()), failures, dict(statistics), examples


def confirm_raw_format(review_path):
    # No rows or prices reach stdout/stderr. The user inspects the local file.
    import sys
    print("Local-only format review: " + str(review_path), file=sys.stderr)
    print("確認箇所: examples 内の raw_win_block がDB列 odds_tansho の元文字列です。", file=sys.stderr)
    print("PC-KEIBAのodds_tanshoがJV O1の単勝領域28組x8文字を無変換で保存し、"
          "各組2/4/2・4桁オッズの倍率1/10と確認できた場合のみ CONFIRM_RAW_O1 と入力してください。"
          "確認用ファイルの実値をチャットへ貼る必要はありません。", file=sys.stderr)
    try:
        if input() != "CONFIRM_RAW_O1":
            raise core.SimulationError("mapping_unconfirmed")
    except EOFError:
        raise core.SimulationError("mapping_unconfirmed") from None


def save_replay_inputs(directory, schedules, snapshots, failures):
    schedule_records = [{"race_key": list(item.race),
                         "scheduled_start": item.scheduled_start.isoformat() if item.scheduled_start is not None else None,
                         "observed_at": None if item.observed_at is None else item.observed_at.isoformat(),
                         "horses": list(item.horses), "availability_verified": item.verified,
                         "availability_basis": item.availability_basis} for item in schedules]
    odds_records = [{"race_key": list(item.race), "published_at": item.published_at.isoformat(),
                     "observed_at": None, "timing_basis": TIMING_BASIS, "publication_time_reviewed": True,
                     "publication_precision": item.publication_precision,
                     "unit_verified": item.verified, "reason": item.reason,
                     "odds": [{"horse_number": h, "value": str(value)} for h, value in item.odds.items()]} for item in snapshots]
    odds_records.extend({"race_key": list(race), "published_at": None, "observed_at": None, "odds": [],
                         "reason": "odds_unverified"} for race in failures)
    for name, records in (("input-schedules.json", schedule_records), ("input-odds.json", odds_records)):
        local.write(directory / name, {"format_version": 2, "source": "local_reviewed", "records": records})
        (directory / name).chmod(0o444)
