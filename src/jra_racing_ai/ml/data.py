"""Explicit RA/SE mappings and bounded read-only export to local CSV."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import date, datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
import time

from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..inspect_schema import COLUMNS_SQL
from .features import DATA_FIELDS, entry_identity, number, validate_rows
from .safety import MAX_ROWS, MLFailure, digest, identifier, local_path, write_json
from .encodings import jv_time_seconds

REQUIRED = ("race_status", "horse_status", "horse_number", "finish_position")
SOURCES = {
    "race_status": "ra", "horse_status": "se", "horse_number": "se", "finish_position": "se",
    "horse_id": "se", "surface": "ra", "going": "ra", "distance": "ra", "declared_runners": "ra",
    "frame": "se", "age": "se", "sex": "se", "weight_kg": "se", "finish_time_seconds": "se", "start_time": "ra",
}
PRE_RACE = frozenset({"surface", "going", "distance", "declared_runners", "frame", "age", "sex", "weight_kg", "start_time"})
NUMERIC = frozenset({"distance", "declared_runners", "frame", "age", "weight_kg", "finish_time_seconds"})
CANDIDATES = {
    "race_status": ("data_kubun",), "horse_status": ("data_kubun",),
    "horse_number": ("umaban", "baban"), "finish_position": ("kakutei_chakujun",),
    "horse_id": ("ketto_toroku_bango",), "distance": ("kyori",), "surface": ("track_code",),
    "declared_runners": ("shusso_tosu",), "frame": ("wakuban",), "age": ("barei",),
    "sex": ("seibetsu_code",), "weight_kg": ("futan_juryo",),
    "finish_time_seconds": ("soha_time",), "start_time": ("hasso_jikoku",),
}
TABLE_SQL = """
SELECT c.relkind AS kind, am.amname AS access_method,
       c.relispartition AS is_partition, c.relhassubclass AS descendants,
       c.relrowsecurity AS rls, c.relforcerowsecurity AS forced_rls
FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_am am ON am.oid = c.relam
WHERE n.nspname = %s AND c.relname = %s
"""
BUILTINS = {"bpchar", "varchar", "text", "int2", "int4", "int8", "numeric", "float4", "float8"}
ENTRY_CATEGORIES = tuple("1234567")
ENTRY_SOURCE = "local_ra_se_prediction_snapshot"


def validate_mapping(value):
    base_keys={"format_version", "reviewed", "columns", "pre_race_fields", "scales", "history_lag_days"}
    if (not isinstance(value, dict) or value.get('format_version') not in (1,2)
            or set(value) != (base_keys if value.get('format_version')==1 else base_keys|{'formats'})):
        raise MLFailure("invalid_mapping")
    formats=value.get('formats',{})
    if formats not in ({},{'finish_time_seconds':'jv_mss_tenths_4'}):
        raise MLFailure('invalid_mapping')
    if value.get("reviewed") is not True:
        raise MLFailure("mapping_not_reviewed")
    columns = value.get("columns")
    if (not isinstance(columns, dict) or set(columns) != set(SOURCES)
            or any(not columns.get(key) for key in REQUIRED)
            or not isinstance(value["pre_race_fields"], list)
            or len(set(value["pre_race_fields"])) != len(value["pre_race_fields"])
            or not set(value["pre_race_fields"]) <= PRE_RACE
            or not isinstance(value["scales"], dict) or not set(value["scales"]) <= NUMERIC
            or type(value["history_lag_days"]) is not int or not 1 <= value["history_lag_days"] <= 365):
        raise MLFailure("invalid_mapping")
    for name, column in columns.items():
        if column is not None:
            identifier(column)
    for field, scale in value["scales"].items():
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 10000:
            raise MLFailure("invalid_mapping")
    # Weight/time units must be explicit; raw time encodings are not guessed.
    for field in ("weight_kg", "finish_time_seconds"):
        if columns[field] and field not in value["scales"] and field not in formats:
            raise MLFailure("invalid_mapping")
        if field in formats and (field in value['scales'] or not columns[field]):
            raise MLFailure('invalid_mapping')
    return value


def inspect_sources(runner, *, include_years=True):
    schema = {}
    for alias in ("ra", "se"):
        table = "jvd_" + alias
        rows = runner.aggregate("ml.table." + alias, TABLE_SQL, ("public", table), max_rows=1)
        if (len(rows) != 1 or rows[0].get("kind") != "r" or rows[0].get("access_method") != "heap"
                or any(rows[0].get(field, True) for field in ("is_partition", "descendants", "rls", "forced_rls"))):
            raise MLFailure("invalid_schema")
        schema[alias] = runner.aggregate("ml.columns." + alias, COLUMNS_SQL, ("public", table), max_rows=1000)
        by_name = {row["name"]: row for row in schema[alias]}
        for key in RACE_KEY:
            if (key not in by_name or not by_name[key].get("type_is_builtin")
                    or by_name[key].get("type_name") not in BUILTINS):
                raise MLFailure("invalid_schema")
    draft = {"format_version": 1, "reviewed": False,
             "columns": {field: None for field in SOURCES}, "pre_race_fields": [],
             "scales": {}, "history_lag_days": 7}
    for field, names in CANDIDATES.items():
        present = {row["name"] for row in schema[SOURCES[field]]}
        matches = [name for name in names if name in present]
        # An exact name is only a candidate, never a semantic confirmation.
        draft["columns"][field] = matches[0] if len(matches) == 1 else None
    # Optional uncertain units are disabled to let the minimal model start now.
    draft["columns"]["weight_kg"] = None
    draft["columns"]["finish_time_seconds"] = None
    years = runner.aggregate("ml.years", """
SELECT ra.kaisai_nen::text AS year, count(*) AS races
FROM ONLY public.jvd_ra ra GROUP BY ra.kaisai_nen ORDER BY ra.kaisai_nen
""", max_rows=300) if include_years else []
    return {"format_version": 1, "tables": schema, "race_year_counts": years,
            "candidate_status": "name_candidates_need_local_review"}, draft


def check_mapping_schema(mapping, schema):
    validate_mapping(mapping)
    for field, column in mapping["columns"].items():
        if column is None:
            continue
        candidates = {row["name"]: row for row in schema[SOURCES[field]]}
        item = candidates.get(column, {})
        if not item.get("type_is_builtin") or item.get("type_name") not in BUILTINS:
            raise MLFailure("invalid_schema")


def export_query(mapping, *, start: date, end: date, max_rows=100000, prediction_only=False, horse_ids=None):
    validate_mapping(mapping)
    if (not isinstance(start, date) or not isinstance(end, date) or start > end
            or not 1 <= max_rows <= MAX_ROWS or type(prediction_only) is not bool):
        raise MLFailure("invalid_input")
    columns = mapping["columns"]
    selected = [f"ra.{identifier(key)}::text AS {identifier(key)}" for key in RACE_KEY]
    for field in DATA_FIELDS[len(RACE_KEY):]:
        column = columns.get(field)
        if field in PRE_RACE and field not in mapping["pre_race_fields"]:
            column = None
        if prediction_only and field in ("finish_position", "finish_time_seconds"):
            column = None
        selected.append((f"{SOURCES[field]}.{identifier(column)}::text" if column else "NULL::text")
                        + f" AS {identifier(field)}")
    join = " AND ".join(f"ra.{identifier(key)} = se.{identifier(key)}" for key in RACE_KEY)
    categories = ENTRY_CATEGORIES if prediction_only else ("7",)
    status_test = "IN (" + ", ".join(["%s"] * len(categories)) + ")" if prediction_only else "= %s"
    if prediction_only:
        selected += [f"ra.{identifier(columns['race_status'])}::text AS _ra_status",
                     f"se.{identifier(columns['horse_status'])}::text AS _se_status"]
    sql = ("SELECT " + ", ".join(selected)
           + " FROM ONLY public.jvd_ra ra JOIN ONLY public.jvd_se se ON " + join
           + f" WHERE ra.{identifier(columns['race_status'])}::text " + status_test
           + f" AND se.{identifier(columns['horse_status'])}::text " + status_test
           + " AND (ra.kaisai_nen::text || ra.kaisai_tsukihi::text) BETWEEN %s AND %s"
           + " AND ra.keibajo_code::text IN (" + ",".join(["%s"] * 10) + ")"
           + " ORDER BY " + ", ".join("ra." + identifier(key) for key in RACE_KEY)
           + ", se." + identifier(columns["horse_number"]) + " LIMIT %s")
    params = (*categories, *categories, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
              *(f"{i:02d}" for i in range(1, 11)), max_rows + 1)
    if horse_ids is not None:
        if (not isinstance(horse_ids,(list,tuple,set)) or not 1<=len(horse_ids)<=20000
                or any(not isinstance(h,str) or not h or len(h)>40 for h in horse_ids)
                or not columns.get('horse_id')):
            raise MLFailure('invalid_input')
        sql=sql.replace(' ORDER BY ',' AND se.'+identifier(columns['horse_id'])+'::text = ANY(%s) ORDER BY ',1)
        params=(*params[:-1],sorted(set(horse_ids)),params[-1])
    validate_audit_sql(sql)
    return sql, params


def export_dataset(runner, mapping, directory, *, start, end, max_rows=100000, max_seconds=180, root=None,
                   prediction_only=False, catalog=None, horse_ids=None):
    validate_mapping(mapping)
    if not 1 <= max_seconds <= 600:
        raise MLFailure("invalid_input")
    deadline = time.monotonic() + max_seconds
    # Only RA and SE are inspected; no expansion of the Phase 1 audit.
    if catalog is None:
        catalog, _ = inspect_sources(runner)
    check_mapping_schema(mapping, catalog["tables"])
    arguments=dict(start=start,end=end,max_rows=max_rows,prediction_only=prediction_only)
    if horse_ids is not None:
        arguments['horse_ids']=horse_ids
    sql, params = export_query(mapping, **arguments)
    path = local_path(directory / "runners.csv", root=root)
    count, seen, reasons, created = 0, set(), Counter(), False
    race_statuses, horse_statuses = {}, Counter()
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            created = True
            writer = csv.DictWriter(stream, fieldnames=DATA_FIELDS)
            writer.writeheader()
            for batch in runner.iter_batches("ml.export", sql, params, max_rows=max_rows,
                                              batch_size=1000, max_seconds=max_seconds):
                if time.monotonic() > deadline:
                    raise MLFailure("query_timeout")
                for row in batch:
                    normalized = {key: "" if row.get(key) is None else str(row[key]) for key in DATA_FIELDS}
                    if prediction_only:
                        # Defense in depth: even a broken adapter or mock must
                        # not carry current outcomes into a prediction artifact.
                        normalized["finish_position"] = normalized["finish_time_seconds"] = ""
                        if row.get("_ra_status") not in ENTRY_CATEGORIES or row.get("_se_status") not in ENTRY_CATEGORIES:
                            raise MLFailure("invalid_schema")
                    key, _ = entry_identity(normalized)
                    if key in seen:
                        raise MLFailure("duplicate_entry")
                    seen.add(key)
                    if prediction_only:
                        if key[:-1] in race_statuses and race_statuses[key[:-1]] != row["_ra_status"]:
                            raise MLFailure("duplicate_entry")
                        race_statuses[key[:-1]] = row["_ra_status"]
                        horse_statuses[row["_se_status"]] += 1
                    for field in NUMERIC:
                        if normalized[field]:
                            if field=='finish_time_seconds' and mapping.get('formats',{}).get(field)=='jv_mss_tenths_4':
                                seconds=jv_time_seconds(normalized[field])
                                normalized[field]='' if seconds is None else str(seconds)
                                if seconds is None:
                                    reasons['invalid_or_initial_jv_time']+=1
                                continue
                            raw = number(normalized[field])
                            if raw is None:
                                normalized[field] = ""
                                reasons["invalid_optional_numeric_" + field] += 1
                            else:
                                normalized[field] = format(Decimal(normalized[field]) * Decimal(str(mapping["scales"].get(field, 1))), "f")
                    writer.writerow(normalized)
                    count += 1
                    if count > max_rows:
                        raise MLFailure("input_limit")
        if not count:
            raise MLFailure("insufficient_history")
        metadata = {"format_version": 1, "source": ENTRY_SOURCE if prediction_only else "local_ra_se_final_snapshot",
                    "rows": count, "races": len({key[:-1] for key in seen}),
                    "start": start.isoformat(), "end": end.isoformat(),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mapping": mapping, "missing_reasons": dict(reasons),
                    "csv_sha256": digest(path, root=root), "complete": True,
                    "production_hr_settlement_enabled": False}
        if prediction_only:
            metadata.update(purpose="prediction_only", current_results_excluded=True,
                            source_status_counts={"ra_races": dict(Counter(race_statuses.values())),
                                                  "se_runners": dict(horse_statuses)},
                            availability="retrospective_entry_snapshot_not_verified_point_in_time")
        write_json(directory / "dataset.json", metadata, root=root)
        return metadata
    except Exception:
        # Failed exports have no completion manifest and cannot be trained on.
        if created and path.exists():
            path.unlink()
        raise


def read_dataset(path: Path, *, root=None, require_manifest=True, allow_prediction_snapshot=False):
    path = local_path(path, exists=True, root=root)
    if path.suffix.lower() != ".csv":
        raise MLFailure("invalid_input")
    from .safety import read_json
    metadata = None
    if require_manifest:
        metadata = read_json(path.parent / "dataset.json", root=root)
        sources = ("synthetic", "local_ra_se_final_snapshot") + ((ENTRY_SOURCE,) if allow_prediction_snapshot is True else ())
        if (metadata.get("format_version") != 1 or metadata.get("complete") is not True
                or metadata.get("source") not in sources
                or metadata.get("csv_sha256") != digest(path, root=root)):
            raise MLFailure("artifact_invalid")
        if metadata["source"] == ENTRY_SOURCE and (metadata.get("purpose") != "prediction_only"
                                                   or metadata.get("current_results_excluded") is not True):
            raise MLFailure("artifact_invalid")
    rows = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(DATA_FIELDS):
                raise MLFailure("invalid_input")
            for row in reader:
                if None in row or any(v is None or len(v) > 128 for v in row.values()):
                    raise MLFailure("invalid_input")
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise MLFailure("input_limit")
        validate_rows(rows)
        if metadata is not None and metadata.get("rows") != len(rows):
            raise MLFailure("artifact_invalid")
        if metadata is not None and metadata["source"] == ENTRY_SOURCE and any(
                row.get(field) for row in rows for field in ("finish_position", "finish_time_seconds")):
            raise MLFailure("artifact_invalid")
        return rows, metadata
    except MLFailure:
        raise
    except Exception:
        raise MLFailure("invalid_input") from None


def write_synthetic_dataset(rows, directory, *, root=None):
    validate_rows(rows)
    path = local_path(directory / "runners.csv", root=root)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATA_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in DATA_FIELDS} for row in rows)
    write_json(directory / "dataset.json", {"format_version": 1, "source": "synthetic",
               "rows": len(rows), "complete": True, "history_lag_days": 7,
               "csv_sha256": digest(path, root=root)}, root=root)
    return path
