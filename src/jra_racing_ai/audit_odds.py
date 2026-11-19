"""Bounded aggregate investigation of odds storage, without odds values.

Column names only provide investigation candidates. No candidate timestamp or
odds payload is fetched, and no published JRA-VAN field meaning is inferred.
"""

from __future__ import annotations

from typing import Any

from .core import SafeAuditError, column_names, qualified, quote_ident, scope_where, status_column


RACE_KEY = (
    "kaisai_nen", "kaisai_tsukihi", "keibajo_code", "kaisai_kai", "kaisai_nichime", "race_bango",
)
SAFE_TYPE_NAMES = {"text", "varchar", "bpchar", "int2", "int4", "int8", "numeric"}


def _numeric_summary(rows: list[dict[str, Any]], fields: tuple[str, ...], nullable: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Allowlist aggregate fields so future query edits cannot expose a row."""
    if len(rows) != 1:
        raise SafeAuditError("internal_error")
    output = {}
    for field in fields:
        value = rows[0].get(field)
        if value is None and field in nullable:
            output[field] = None
        elif type(value) is int and value >= 0:
            output[field] = value
        else:
            raise SafeAuditError("internal_error")
    return [output]


def _buckets(rows: list[dict[str, Any]], label: str, count: str, allowed: set[str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get(label) not in allowed or type(row.get(count)) is not int or row[count] < 0:
            raise SafeAuditError("internal_error")
        output.append({label: row[label], count: row[count]})
    return output


def _candidates(columns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    times: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []
    for column in columns:
        name = column["name"]
        lower = name.lower()
        data_type = column.get("data_type", "")
        if name not in RACE_KEY and (
            any(term in lower for term in ("happyo", "sakusei", "nichiji", "jifun", "jikan", "timestamp", "time", "date"))
            or any(term in data_type.lower() for term in ("timestamp", "date", "time"))
        ):
            times.append({"name": name, "data_type": data_type, "evidence": "column_name_or_type_only", "meaning": "要確認"})
        if any(term in lower for term in ("odds", "tansho", "tansyo", "fukusho", "fukusyo")):
            odds.append({"name": name, "data_type": data_type, "evidence": "column_name_only", "meaning": "要確認"})
    return times, odds


def audit_odds(ctx: Any) -> dict[str, Any]:
    """Audit only verified race-key columns and explicitly mapped status codes.

    Repeated race keys cannot establish whether a record is a snapshot, packed
    payload, revision, or horse row. The report therefore keeps grain unconfirmed.
    """
    table_name = "jvd_o1"
    table = ctx.schema.get(table_name)
    result: dict[str, Any] = {
        "status": "needs_confirmation",
        "table": table_name,
        "scope": {"year": ctx.scope.year, "venues": list(ctx.scope.venues)},
        "record_grain": "要確認",
        "pre_race_vs_final_odds": "要確認",
        "availability_time": "要確認",
        "leakage_risk": "更新時点・上書き履歴・最終オッズの定義が未確認。予測時点の特徴量としての利用は未判定。",
        "warnings": [],
        "specification_questions": [
            "1行の単位（レース、馬、時点、版、固定長・配列格納など）を仕様と照合する。",
            "単勝・複勝の格納列、桁・倍率、欠測・発売なし等のコードを仕様と照合する。",
            "更新・発表時点候補の意味、タイムゾーン、開催日との対応、履歴保存・上書き動作を確認する。",
            "発走前に取得できるオッズと最終オッズを区別できるか確認する。",
            "作成日時だけで利用可能時点を確定しない。公開時点・訂正・遡及更新の扱いを確認する。",
        ],
    }
    if table is None:
        result["warnings"].append("public.jvd_o1 が構造一覧にありません。テーブル名を確認してください。")
        return result
    if (
        table.get("auditable") is not True or table.get("kind") != "table"
        or table.get("table_access_method") != "heap"
        or table.get("row_security") is not False
        or table.get("force_row_security") is not False
    ):
        result["warnings"].append("heap通常テーブル以外またはRLS有効・状態未確認のため、実レコード監査対象外。")
        return result
    times, odds = _candidates(table.get("columns", []))
    result["time_column_candidates"] = times
    result["odds_column_candidates"] = odds
    result["declared_primary_key_columns"] = list(table.get("primary_key") or [])
    names = column_names(ctx, table_name)
    result["status_column_candidates"] = [name for name in sorted(names) if "kubun" in name.lower() or "status" in name.lower()]
    missing = [name for name in RACE_KEY if name not in names]
    if missing:
        result["missing_race_key_columns"] = missing
        result["warnings"].append("指定されたレース識別列が不足するため、値の集計は実行していません。")
        return result

    status_name = status_column(ctx, table_name)
    metadata = {column["name"]: column for column in table.get("columns", [])}
    used_columns = list(RACE_KEY) + ([status_name] if status_name is not None else [])
    unconfirmed = [
        name for name in used_columns
        if metadata[name].get("type_is_builtin") is not True
        or metadata[name].get("type_name") not in SAFE_TYPE_NAMES
    ]
    if unconfirmed:
        result["unconfirmed_column_types"] = unconfirmed
        result["warnings"].append("集計対象列の型が確認済みのPostgreSQL組み込み型ではないため、実集計は実行していません。")
        return result

    where, params = scope_where(ctx.scope, "t")
    relation = "ONLY " + qualified(table_name)
    key_sql = ", ".join(f"btrim(t.{quote_ident(name)}::text)" for name in RACE_KEY)
    complete = " AND ".join(f"t.{quote_ident(name)} IS NOT NULL AND btrim(t.{quote_ident(name)}::text) <> ''" for name in RACE_KEY)
    nulls = ",\n       ".join(
        f"count(*) FILTER (WHERE t.{quote_ident(name)} IS NULL) AS {quote_ident(name + '_null_records')}"
        for name in RACE_KEY
    )
    blank = " OR ".join(f"t.{quote_ident(name)} IS NOT NULL AND btrim(t.{quote_ident(name)}::text) = ''" for name in RACE_KEY)
    summary_sql = f"""
SELECT count(*) AS records,
       count(*) FILTER (WHERE NOT ({complete})) AS incomplete_key_records,
       count(*) FILTER (WHERE {blank}) AS blank_key_records,
       {nulls}
FROM {relation} t
WHERE {where}
"""
    result["record_summary"] = _numeric_summary(
        ctx.db.aggregate("odds.summary", summary_sql, params, max_rows=1),
        ("records", "incomplete_key_records", "blank_key_records") + tuple(name + "_null_records" for name in RACE_KEY),
    )
    grain_sql = f"""
WITH grouped AS (
    SELECT count(*) AS records
    FROM {relation} t
    WHERE {where} AND ({complete})
    GROUP BY {key_sql}
)
SELECT count(*) AS complete_race_key_groups,
       COALESCE(sum(records), 0)::bigint AS records_with_complete_keys,
       count(*) FILTER (WHERE records = 1) AS single_record_groups,
       count(*) FILTER (WHERE records > 1) AS multiple_record_groups,
       min(records) AS minimum_records_per_race,
       max(records) AS maximum_records_per_race
FROM grouped
"""
    result["race_key_grain_summary"] = _numeric_summary(
        ctx.db.aggregate("odds.grain", grain_sql, params, max_rows=1),
        ("complete_race_key_groups", "records_with_complete_keys", "single_record_groups", "multiple_record_groups",
         "minimum_records_per_race", "maximum_records_per_race"),
        ("minimum_records_per_race", "maximum_records_per_race"),
    )
    distribution_sql = f"""
WITH grouped AS (
    SELECT count(*) AS records
    FROM {relation} t
    WHERE {where} AND ({complete})
    GROUP BY {key_sql}
), bands AS (
    SELECT CASE WHEN records = 1 THEN '1'
                WHEN records <= 5 THEN '2-5'
                WHEN records <= 10 THEN '6-10'
                WHEN records <= 20 THEN '11-20'
                WHEN records <= 50 THEN '21-50'
                WHEN records <= 100 THEN '51-100'
                ELSE '101+' END AS records_per_race_band
    FROM grouped
)
SELECT records_per_race_band, count(*) AS race_key_groups
FROM bands
GROUP BY records_per_race_band
ORDER BY records_per_race_band
"""
    result["records_per_race_distribution"] = _buckets(
        ctx.db.aggregate("odds.grain_distribution", distribution_sql, params, max_rows=7),
        "records_per_race_band", "race_key_groups", {"1", "2-5", "6-10", "11-20", "21-50", "51-100", "101+"},
    )

    if status_name is None:
        result["status_summary"] = {"status": "not_run", "reason": "区分列の意味が未確認。--odds-status-column による明示指定が必要。"}
    else:
        # A mapped column can still contain arbitrary content. Return validated
        # one-digit codes only; all other values are reduced to a count.
        field = f"btrim(t.{quote_ident(status_name)}::text)"
        status_sql = f"""
SELECT CASE WHEN {field} ~ '^[0-9]$' THEN {field}
            WHEN t.{quote_ident(status_name)} IS NULL THEN 'NULL'
            ELSE 'INVALID' END AS status_code,
       count(*) AS records
FROM {relation} t
WHERE {where}
GROUP BY CASE WHEN {field} ~ '^[0-9]$' THEN {field}
              WHEN t.{quote_ident(status_name)} IS NULL THEN 'NULL'
              ELSE 'INVALID' END
ORDER BY status_code
"""
        result["status_summary"] = {
            "column": status_name, "meaning": "要確認",
            "counts": _buckets(ctx.db.aggregate("odds.status", status_sql, params, max_rows=12),
                               "status_code", "records", set("0123456789") | {"NULL", "INVALID"}),
        }
    if getattr(ctx.scope, "global_summary", False):
        year = f"btrim(t.{quote_ident('kaisai_nen')}::text)"
        coverage_sql = f"""
WITH years AS (
    SELECT CASE WHEN {year} ~ '^[0-9]{{4}}$' THEN {year}::integer
                ELSE NULL END AS year_number,
           (t.{quote_ident('kaisai_nen')} IS NULL) AS is_null
    FROM {relation} t
)
SELECT min(year_number) FILTER (WHERE year_number BETWEEN 1 AND 9999) AS minimum_year,
       max(year_number) FILTER (WHERE year_number BETWEEN 1 AND 9999) AS maximum_year,
       count(*) FILTER (WHERE is_null) AS null_year_records,
       count(*) FILTER (WHERE NOT is_null AND (year_number IS NULL OR year_number = 0)) AS invalid_year_records
FROM years
"""
        coverage = _numeric_summary(
            ctx.db.aggregate("odds.global_year_coverage", coverage_sql, (), max_rows=1),
            ("minimum_year", "maximum_year", "null_year_records", "invalid_year_records"),
            ("minimum_year", "maximum_year"),
        )
        if any(coverage[0][field] is not None and not 1 <= coverage[0][field] <= 9999
               for field in ("minimum_year", "maximum_year")):
            raise SafeAuditError("internal_error")
        result["global_year_coverage"] = {
            "status": "ok", "scope": "all_years_all_venues",
            "summary": coverage,
        }
    else:
        result["global_year_coverage"] = {
            "status": "not_run",
            "reason": "全期間の走査は既定で省略。--include-global-summary の明示指定で年範囲を集計します。",
        }
    result["warnings"].extend([
        "race keyは前後の空白だけを除去します。桁補完や別表現の同一視は行いません。",
        "複数recordのrace keyがあっても更新時点別・馬別・重複のいずれかは判定できません。",
        "時点・odds候補は名前と型だけの候補です。列の値、オッズ、生キー、個別時刻を取得しません。",
        "指定年・指定競馬場の抽出条件外のNULL・不正値は、この範囲の異常集計に含まれません。",
    ])
    return result
