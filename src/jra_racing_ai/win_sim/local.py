"""Local-only artifacts and bounded SELECT adapters. No connections on import."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
import re
import time
from uuid import uuid4

from ..core import RACE_KEY, SafeAuditError
from ..inspect_schema import COLUMNS_SQL
from ..ml import safety as old_io
from ..ml.data import TABLE_SQL
from .core import (Outcome, PredictionRace, SimulationError, Snapshot, WinPolicy, instant, key, horse)
from .formats import (ODDS_TABLES, _identifier, decode_hr_rows, decode_odds_rows,
                      validate_payout_source, validate_source, JV_O1_WIN)

ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v02")
SPEC_PDF = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\references\JRA-VAN\JV-Data4901.pdf")
TEXT_TYPES = {"bpchar", "varchar", "text"}
BUILTINS = TEXT_TYPES | {"int2", "int4", "int8", "numeric", "timestamp", "timestamptz"}


def artifact(kind, *, root=None):
    root = ROOT if root is None else root
    if kind not in ("inspection", "policies", "predictions", "simulations", "demo"):
        raise SimulationError("invalid_input")
    try:
        path = old_io.local_path(root / kind / uuid4().hex, root=root)
        path.mkdir(parents=True, exist_ok=False)
        return path
    except Exception:
        raise SimulationError("unsafe_path") from None


def read(path, *, root=None):
    try:
        return old_io.read_json(path, root=ROOT if root is None else root)
    except Exception:
        raise SimulationError("unsafe_path") from None


def write(path, value, *, root=None):
    try:
        old_io.write_json(path, value, root=ROOT if root is None else root)
    except Exception:
        raise SimulationError("io_failed") from None


def sha(path, *, root=None):
    try:
        return old_io.digest(path, root=ROOT if root is None else root)
    except Exception:
        raise SimulationError("artifact_invalid") from None


def model_info(directory, *, v01_root=None):
    root = old_io.ML_ROOT if v01_root is None else v01_root
    info = read(directory / "model.json", root=root)
    if (info.get("format_version") != 1 or info.get("model_kind") != "lightgbm_rank_proxy_v01"
            or info.get("production_hr_settlement_enabled") is not False):
        raise SimulationError("artifact_invalid")
    try:
        date.fromisoformat(info["split"]["validation"]["end"])
    except (KeyError, TypeError, ValueError):
        raise SimulationError("artifact_invalid") from None
    return info, sha(directory / "model.json", root=root)


def load_predictions(directory, *, expected_model_hash=None, v01_root=None, root=None):
    root = ROOT if root is None else root
    v01_root = old_io.ML_ROOT if v01_root is None else v01_root
    # Legacy prediction ledgers remain untouched. Both roots are fixed local paths.
    base = v01_root if directory == v01_root or v01_root in directory.parents else root
    manifest = read(directory / "manifest.json", root=base)
    if (manifest.get("format_version") not in (1, 2)
            or manifest.get("mode") not in ("backtest", "hypothetical_backtest")
            or manifest.get("predictions_sha256") != sha(directory / "predictions.json", root=base)
            or (expected_model_hash and manifest.get("model_manifest_sha256") != expected_model_hash)):
        raise SimulationError("artifact_invalid")
    payload = read(directory / "predictions.json", root=base)
    if payload.get("format_version") != manifest["format_version"] or payload.get("mode") != manifest["mode"]:
        raise SimulationError("artifact_invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) > 1000000:
        raise SimulationError("input_limit")
    grouped = {}
    for row in records:
        if row.get("wager") != "win":
            continue
        try:
            parts = row["entry_key"]
            if not isinstance(parts, list) or len(parts) != 7:
                raise SimulationError("invalid_input")
            race, h = key(parts[:6]), horse(parts[6])
            group = grouped.setdefault(race, {})
            if h in group:
                raise SimulationError("duplicate_prediction")
            group[h] = row["probability"]
        except (KeyError, TypeError):
            raise SimulationError("invalid_input") from None
    if not grouped:
        raise SimulationError("invalid_input")
    return [PredictionRace(race, group) for race, group in sorted(grouped.items())], manifest


def policy_document(policy, *, model_hash, calibration=None):
    policy.validate()
    return {"format_version": 2, "wager": "win", "policy": asdict(policy),
            "model_manifest_sha256": model_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "threshold_origin": "fixed_before_simulation_not_test_optimized",
            "stake_yen": 100, "max_bets_per_race": 1,
            "favorite_tie_break": "lowest_horse_number_at_same_decision_odds",
            "calibration": calibration, "live_purchase_enabled": False}


def load_policy(path, *, root=None):
    document = read(path, root=root)
    try:
        if (document.get("format_version") != 2 or document.get("wager") != "win"
                or document.get("stake_yen") != 100 or document.get("max_bets_per_race") != 1
                or document.get("live_purchase_enabled") is not False
                or document.get("threshold_origin") != "fixed_before_simulation_not_test_optimized"):
            raise SimulationError("invalid_policy")
        policy = WinPolicy(**document["policy"]).validate()
        return policy, document
    except (KeyError, TypeError):
        raise SimulationError("invalid_policy") from None


def draft_config():
    return {"format_version": 2, "odds_sources": [
        {"table": table, "enabled": False, "layout": None,
         "columns": {field: None for field in ("status", "published", "observed", "horse", "price", "payload")},
         "time_formats": {"published": None, "observed": None}, "allowed_statuses": [],
         "price_encoding": {"format": None, "width": None, "numerator": None, "denominator": None,
                            "unavailable_tokens": []}, "packed": None,
         "confirmed": {name: False for name in ("official_format", "pc_keiba_mapping", "observed_time_semantics",
                                                "odds_unit", "revision_policy")},
         "official_reference": str(SPEC_PDF), "official_definition": dict(JV_O1_WIN),
         "local_mapping_reference": None} for table in ODDS_TABLES],
            "payout_source": None}


class Budget:
    def __init__(self, seconds=120, clock=None):
        if type(seconds) is not int or not 1 <= seconds <= 600:
            raise SimulationError("invalid_input")
        self.clock = time.monotonic if clock is None else clock
        self.deadline = self.clock()+seconds

    def check(self):
        if self.clock() >= self.deadline:
            raise SimulationError("query_timeout")

    def aggregate(self, db, query_id, sql, params=(), max_rows=1000):
        self.check()
        result = db.aggregate(query_id, sql, params, max_rows=max_rows)
        self.check()
        return result


def table_info(db, table, budget):
    if table not in (*ODDS_TABLES, "jvd_hr", "jvd_ra", "jvd_se"):
        raise SimulationError("invalid_mapping")
    rows = budget.aggregate(db, "v02.table."+table, TABLE_SQL, ("public", table), max_rows=1)
    if not rows:
        return None
    row = rows[0]
    if (row.get("kind") != "r" or row.get("access_method") != "heap"
            or any(row.get(field, True) for field in ("is_partition", "descendants", "rls", "forced_rls"))):
        raise SimulationError("invalid_schema")
    columns = budget.aggregate(db, "v02.columns."+table, COLUMNS_SQL, ("public", table), max_rows=1600)
    return {column["name"]: column for column in columns}


def inspect_availability(db, *, sample_limit=100, max_seconds=30):
    if type(sample_limit) is not int or not 1 <= sample_limit <= 1000:
        raise SimulationError("input_limit")
    budget, output = Budget(max_seconds), []
    for table in ODDS_TABLES:
        try:
            columns = table_info(db, table, budget)
        except SimulationError as exc:
            if exc.code != "invalid_schema":
                raise
            output.append({"table": table, "status": "unsupported_table", "columns": []})
            continue
        if columns is None:
            output.append({"table": table, "status": "not_found", "columns": []})
            continue
        item = {"table": table, "status": "structure_observed_format_unconfirmed", "columns": list(columns.values()),
                "sample_scope": "bounded_recent_rows_not_full_history", "sample_limit": sample_limit}
        if all(columns.get(k, {}).get("type_is_builtin") is True and columns[k].get("type_name") in TEXT_TYPES | {"int2", "int4", "int8"}
               for k in RACE_KEY):
            sql = ('SELECT count(*) AS sampled_rows, min(s.y || s.d) AS first_date, max(s.y || s.d) AS last_date FROM ('
                   'SELECT t.kaisai_nen::text AS y, t.kaisai_tsukihi::text AS d FROM ONLY public.'+_identifier(table)
                   +' t ORDER BY '+', '.join('t.'+_identifier(k)+' DESC' for k in RACE_KEY)+' LIMIT %s) s')
            sample = budget.aggregate(db, "v02.recent."+table, sql, (sample_limit,), max_rows=1)
            if len(sample) != 1 or type(sample[0].get("sampled_rows")) is not int:
                raise SimulationError("invalid_schema")
            # Aggregate dates only; unexpected strings never reach console/report.
            item["recent_sample"] = {"sampled_rows": sample[0]["sampled_rows"]}
            for field in ("first_date", "last_date"):
                value = sample[0].get(field)
                item["recent_sample"][field] = value if isinstance(value, str) and re.fullmatch(r"[0-9]{8}", value) else None
        else:
            item["status"] = "race_key_mapping_unconfirmed"
        output.append(item)
    return {"format_version": 2, "tables": output, "official_spec_reviewed_by_agent": True,
            "official_reference": str(SPEC_PDF), "official_o1_win_definition": dict(JV_O1_WIN),
            "pc_keiba_odds_unit_confirmed": False, "observation_time_confirmed": False}


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != {"format_version", "odds_sources", "payout_source"}
            or config["format_version"] != 2 or not isinstance(config["odds_sources"], list)
            or len(config["odds_sources"]) > 2):
        raise SimulationError("invalid_mapping")
    seen = set()
    for source in config["odds_sources"]:
        if not isinstance(source, dict) or source.get("table") not in ODDS_TABLES or source["table"] in seen:
            raise SimulationError("invalid_mapping")
        seen.add(source["table"])
        if source.get("enabled") is True:
            validate_source(source)
    if config["payout_source"] is not None:
        validate_payout_source(config["payout_source"])
    return config


def build_rows_query(table, fields, columns, races, max_rows, *, join_targets=False):
    if (table not in (*ODDS_TABLES, "jvd_hr", "jvd_ra", "jvd_se") or not races or len(races) > 5000
            or type(max_rows) is not int or not 1 <= max_rows <= 200000):
        raise SimulationError("input_limit")
    for name in (*RACE_KEY, *(physical for physical, _ in fields.values())):
        col = columns.get(name, {})
        if col.get("type_is_builtin") is not True or col.get("type_name") not in BUILTINS:
            raise SimulationError("invalid_schema")
    selections = ['t.'+_identifier(k)+'::text AS '+_identifier(k) for k in RACE_KEY]
    too_long = []
    for alias, (physical, limit) in fields.items():
        name = 't.'+_identifier(physical)
        check = (f'pg_catalog.octet_length({name}) > {limit}' if columns[physical]["type_name"] in TEXT_TYPES else 'false')
        too_long.append(check)
        selections.append(f'CASE WHEN {check} THEN NULL ELSE {name} END AS '+_identifier(alias))
    selections.append('('+' OR '.join(too_long)+') AS oversized')
    if join_targets:
        # Frozen local ticket keys are parameters, not uploaded tables or SQL text.
        # The ordinary adapter's WHERE path remains unchanged for existing callers.
        values = ', '.join('('+', '.join('%s::text' for _ in RACE_KEY)+')' for _ in races)
        names = ', '.join(_identifier(k) for k in RACE_KEY)
        condition = ' AND '.join('t.'+_identifier(k)+'::text = requested.'+_identifier(k) for k in RACE_KEY)
        suffix = ' JOIN (VALUES '+values+') AS requested('+names+') ON '+condition
    else:
        where = ' OR '.join('('+' AND '.join('t.'+_identifier(k)+' = %s' for k in RACE_KEY)+')' for _ in races)
        suffix = ' WHERE ('+where+')'
    sql = 'SELECT '+', '.join(selections)+' FROM ONLY public.'+_identifier(table)+' t'+suffix+' LIMIT %s'
    params = tuple(value for race in races for value in key(race))+(max_rows+1,)
    from ..db_connection import validate_audit_sql
    validate_audit_sql(sql)
    return sql, params


def _rows(db, budget, table, fields, columns, races, max_rows, *, join_targets=False):
    sql, params = build_rows_query(table, fields, columns, races, max_rows, join_targets=join_targets)
    rows = []
    for batch in db.iter_batches("v02.rows."+table, sql, params, max_rows=max_rows, batch_size=1000,
                                 max_seconds=max(1, min(600, int(budget.deadline-budget.clock())))):
        budget.check()
        for row in batch:
            row["race_key"] = key([row.pop(k, None) for k in RACE_KEY])
            if row["race_key"] not in races:
                raise SimulationError("invalid_schema")
            for alias in ("status", "unestablished", "special", "refund"):
                if type(row.get(alias)) is int and 0 <= row[alias] <= 9:
                    row[alias] = str(row[alias])
            rows.append(row)
    return rows


def fetch_inputs(db, config, races, *, max_rows=50000, max_seconds=120):
    validate_config(config)
    budget, snapshots, failures, status = Budget(max_seconds), [], {}, []
    races = tuple(key(race) for race in races)
    if len(set(races)) != len(races):
        raise SimulationError("invalid_input")
    count, outcomes = 0, []
    for source in config["odds_sources"]:
        if source.get("enabled") is not True:
            status.append({"table": source["table"], "status": "mapping_unconfirmed", "rows": 0})
            continue
        try:
            columns = table_info(db, source["table"], budget)
        except SimulationError as exc:
            if exc.code != "invalid_schema":
                raise
            status.append({"table": source["table"], "status": "unsupported_table", "rows": 0})
            continue
        if columns is None:
            status.append({"table": source["table"], "status": "not_found", "rows": 0})
            continue
        sizes = {"status": 4, "published": 32, "observed": 32, "horse": 2, "price": 16, "payload": 1792}
        fields = {alias: (name, sizes[alias]) for alias, name in source["columns"].items() if name}
        rows = _rows(db, budget, source["table"], fields, columns, races, max_rows-count)
        count += len(rows)
        # Native numeric columns have no fixed-width leading zeros to preserve.
        if source["price_encoding"]["format"] == "decimal":
            for row in rows:
                if row.get("price") is not None and not isinstance(row["price"], str):
                    row["price"] = str(row["price"])
        parsed, failed = decode_odds_rows(rows, source)
        snapshots.extend(parsed)
        failures.update(failed)
        status.append({"table": source["table"], "status": "bounded_read_complete", "rows": len(rows),
                       "snapshots": len(parsed), "timestamp_failure_races": len(failed)})
    source = config["payout_source"]
    if source:
        columns = table_info(db, "jvd_hr", budget)
        if columns is not None:
            fields = {alias: (name, 4) for alias, name in source["columns"].items()}
            fields.update({f"slot_{i}{part}": (f"haraimodoshi_tansho_{i}{part}", width)
                           for i in range(1, 4) for part, width in zip("abc", (2, 9, 2))})
            rows = _rows(db, budget, "jvd_hr", fields, columns, races, max_rows-count)
            outcomes = decode_hr_rows(rows, source)
            status.append({"table": "jvd_hr", "status": "bounded_read_complete", "rows": len(rows)})
    return snapshots, outcomes, failures, status


def write_report(directory, simulation, *, policy_payload, provenance, root=None):
    data_source = "synthetic" if provenance.get("source") == "synthetic" else "local_inputs"
    ledger = {k: v for k, v in simulation.items() if k != "summary"}
    ledger["data_source"] = data_source
    write(directory / "ledger.json", ledger, root=root)
    write(directory / "summary.json", simulation["summary"], root=root)
    write(directory / "policy.json", policy_payload, root=root)
    write(directory / "manifest.json", {"format_version": 2, "mode": "hypothetical_backtest", "wager": "win",
          "created_at_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
          "hashes": {name: sha(directory / name, root=root) for name in ("ledger.json", "summary.json", "policy.json")},
          "live_purchase_enabled": False}, root=root)
    label = ("**架空データのみの動作確認です。実データの精度・収益性を示しません。**" if data_source == "synthetic"
             else "ローカル入力による仮想評価。入力時点・公式額の根拠はローカル確認記録を参照してください。")
    version = "0.3" if provenance.get("pipeline_version") == "0.3" else "0.2"
    lines = ["# Version " + version + " 単勝仮想バックテスト", "", label, "", simulation["notice"], "",
             "判断用オッズと公式確定払戻は別情報。未精算を0円や不的中に置換しません。", "",
             "| 月 | 全対象 | オッズ取得 | 取得率% | AI購入 | 1番人気購入 | AI未精算 | AI純利益 | AI回収率% | 比較共通レース | 共通AI純利益 | 共通1番人気純利益 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if simulation.get("timing_basis") == "jra_publication_time":
        lines[8:8] = ["発表履歴の対応率を表示します。PCでの実取得率・購入可能性を検証した値ではありません。", ""]
    for month, result in simulation["summary"]["by_month"].items():
        common = result["comparisons"]["all_common_eligible_races"]
        values = (month, result["target_race_count"], result["odds_acquired_race_count"],
                  result["odds_acquisition_rate_percent"], result["ai"]["bet_count"], result["favorite"]["bet_count"],
                  result["ai"]["unsettled_count"],
                  result["ai"]["net_profit_yen"], result["ai"]["return_rate_percent"], common["race_count"],
                  common["ai"]["net_profit_yen"], common["favorite"]["net_profit_yen"])
        lines.append("| "+" | ".join("未算出" if value is None else str(value) for value in values)+" |")
    lines += ["", "詳細な件数・的中率・投資・払戻・期待利益率分布・共通集合比較・月別安定性は summary.json を参照。",
              "個別レース明細はローカル ledger.json のみ。実購入・約定の実績ではありません。", ""]
    try:
        path = old_io.local_path(directory / "report.md", root=ROOT if root is None else root)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        for name in ("ledger.json", "summary.json", "policy.json", "manifest.json", "report.md"):
            (directory / name).chmod(0o444)
    except Exception:
        raise SimulationError("io_failed") from None
