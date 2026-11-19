"""Year-scoped race aggregates. No source race identifiers leave PostgreSQL."""

from __future__ import annotations

from datetime import date as calendar_date

from .core import RACE_KEY, column_names, qualified, quote_ident, scope_where, status_column, usable_columns


def key_expressions(alias: str = "t") -> list[str]:
    """Preserve zero padding; trim storage padding without guessing key formats."""
    return [f"btrim({alias}.{quote_ident(name)}::text)" for name in RACE_KEY]


def key_projection(alias: str = "t") -> str:
    return ", ".join(
        f"{expression} AS {quote_ident(name)}"
        for expression, name in zip(key_expressions(alias), RACE_KEY)
    )


def complete_key(alias: str = "t") -> str:
    return " AND ".join(f"{alias}.{quote_ident(name)} IS NOT NULL" for name in RACE_KEY)


def nonempty_key(alias: str = "t") -> str:
    return " AND ".join(f"{value} <> ''" for value in key_expressions(alias))


def valid_key_format(alias: str = "t") -> str:
    """Conservative syntax checks, not a declaration of JRA-VAN semantics."""
    year, date, venue, kai, day, race = key_expressions(alias)
    guarded = " AND ".join(
        [f"{year} ~ '^[0-9]{{4}}$'", f"{date} ~ '^[0-9]{{4}}$'"]
        + [f"{value} ~ '^[0-9]{{1,2}}$'" for value in (venue, kai, day, race)]
    )
    # CASE keeps all integer casts behind bounded ASCII digit checks.
    month = f"substring({date}, 1, 2)::integer"
    dom = f"substring({date}, 3, 2)::integer"
    days = (
        f"CASE WHEN {month} = 2 THEN CASE WHEN (mod({year}::integer, 400) = 0 "
        f"OR (mod({year}::integer, 4) = 0 AND mod({year}::integer, 100) <> 0)) "
        f"THEN 29 ELSE 28 END WHEN {month} IN (4,6,9,11) THEN 30 ELSE 31 END"
    )
    checks = [f"{year}::integer BETWEEN 1 AND 9999", f"{month} BETWEEN 1 AND 12", f"{dom} BETWEEN 1 AND ({days})"]
    checks.extend(f"{value}::integer BETWEEN 1 AND 99" for value in (venue, kai, day, race))
    return f"CASE WHEN {guarded} THEN ({' AND '.join(checks)}) ELSE FALSE END"


def valid_calendar_date(alias: str = "t") -> str:
    year, date = key_expressions(alias)[:2]
    month = f"substring({date}, 1, 2)::integer"
    dom = f"substring({date}, 3, 2)::integer"
    days = (
        f"CASE WHEN {month} = 2 THEN CASE WHEN (mod({year}::integer, 400) = 0 "
        f"OR (mod({year}::integer, 4) = 0 AND mod({year}::integer, 100) <> 0)) "
        f"THEN 29 ELSE 28 END WHEN {month} IN (4,6,9,11) THEN 30 ELSE 31 END"
    )
    return (
        f"CASE WHEN {year} ~ '^[0-9]{{4}}$' AND {date} ~ '^[0-9]{{4}}$' THEN "
        f"({year}::integer BETWEEN 1 AND 9999 AND {month} BETWEEN 1 AND 12 "
        f"AND {dom} BETWEEN 1 AND ({days})) ELSE FALSE END"
    )


def category_expression(column: str, alias: str = "t") -> str:
    value = f"btrim({alias}.{quote_ident(column)}::text)"
    return (
        f"CASE WHEN {alias}.{quote_ident(column)} IS NULL THEN 'null' "
        f"WHEN {value} ~ '^[0-9]$' THEN {value} ELSE 'other' END"
    )


def category_column(ctx, table: str) -> str | None:
    """Only an explicit mapping may give a column semantic meaning."""
    mapped = status_column(ctx, table)
    return mapped if mapped is not None and usable_columns(ctx, table, (mapped,)) else None


def category_mapping_note(ctx, table: str) -> dict:
    mapped = category_column(ctx, table)
    candidates = [name for name in ("data_kubun", "data_kbn") if name in column_names(ctx, table)]
    return {
        "status": "user_mapped_unverified_specification" if mapped else "needs_confirmation",
        "mapped_column": mapped,
        "name_candidates_only": candidates,
        "note": "区分7=確定成績・9=中止はユーザー提示の解釈。列の意味は公開仕様で要確認。候補名だけで集計しない。",
    }


def required_key_status(ctx, table: str) -> dict | None:
    if table not in ctx.schema:
        return {"status": "needs_confirmation", "reason": "table_not_found", "table": table}
    metadata = ctx.schema[table]
    if (metadata.get("auditable", False) is not True
            or metadata.get("kind") != "table"
            or metadata.get("table_access_method") != "heap"
            or metadata.get("row_security") is not False
            or metadata.get("force_row_security") is not False):
        return {"status": "needs_confirmation", "reason": "table_not_locally_auditable", "table": table}
    missing = [name for name in RACE_KEY if name not in column_names(ctx, table)]
    if missing:
        return {"status": "needs_confirmation", "reason": "race_key_columns_missing", "table": table, "missing_columns": missing}
    if not usable_columns(ctx, table, RACE_KEY):
        return {"status": "needs_confirmation", "reason": "race_key_column_type_requires_review", "table": table}
    return None


def count_value(value) -> int:
    """Aggregate outputs must be counts, never an accidental source value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid_aggregate_count")
    return value


def take_count_rows(rows: list[dict], label: str, allowed: set[str]) -> list[dict]:
    output = []
    for row in rows:
        bucket = row.get(label)
        if bucket not in allowed:
            raise ValueError("unexpected_aggregate_bucket")
        item = {label: bucket, "records": count_value(row.get("records"))}
        if "distinct_races" in row:
            item["distinct_races"] = count_value(row["distinct_races"])
        output.append(item)
    return output


def audit_period(ctx, table: str, where: str, params: tuple, prefix: str) -> dict:
    year, monthday = key_expressions()[:2]
    valid = valid_calendar_date()
    iso = f"{year} || '-' || substring({monthday}, 1, 2) || '-' || substring({monthday}, 3, 2)"
    sql = f"""
WITH dates AS (
 SELECT CASE WHEN {valid} THEN ({iso}) ELSE NULL END AS calendar_date
 FROM ONLY {qualified(table)} t WHERE {where}
)
SELECT min(calendar_date) AS earliest_date, max(calendar_date) AS latest_date,
       count(*) FILTER (WHERE calendar_date IS NOT NULL) AS valid_date_records,
       count(*) FILTER (WHERE calendar_date IS NULL) AS invalid_or_null_date_records
FROM dates
"""
    rows = ctx.db.aggregate(prefix + ".period", sql, params, max_rows=1)
    if len(rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    result = {}
    for field in ("earliest_date", "latest_date"):
        value = rows[0].get(field)
        if value is not None:
            if not isinstance(value, str) or len(value) != 10:
                raise ValueError("invalid_period_aggregate")
            calendar_date.fromisoformat(value)
        result[field] = value
    result.update({field: count_value(rows[0][field]) for field in ("valid_date_records", "invalid_or_null_date_records")})
    return result


def audit_global_years(ctx, table: str, prefix: str) -> dict:
    year = key_expressions()[0]
    # Only bounded year labels are emitted; malformed year strings never leave DB.
    sql = f"""
SELECT CASE WHEN {year} ~ '^[0-9]{{4}}$' AND {year} <> '0000' THEN {year}
            WHEN t.{quote_ident('kaisai_nen')} IS NULL THEN 'null' ELSE 'other' END AS year,
       count(*) AS records
FROM ONLY {qualified(table)} t GROUP BY 1 ORDER BY 1
"""
    rows = ctx.db.aggregate(prefix + ".global_years", sql, (), max_rows=10001)
    counts = take_count_rows(rows, "year", {f"{n:04}" for n in range(1, 10000)} | {"null", "other"})
    quality_selects = [
        "count(*) AS records",
        f"count(*) FILTER (WHERE NOT ({complete_key()})) AS null_key_records",
        f"count(*) FILTER (WHERE {complete_key()} AND NOT ({nonempty_key()})) AS empty_key_records",
        f"count(*) FILTER (WHERE {complete_key()} AND NOT ({valid_key_format()})) AS invalid_format_records",
    ]
    for name in RACE_KEY:
        quality_selects.extend([
            f"count(*) FILTER (WHERE t.{quote_ident(name)} IS NULL) AS {quote_ident(name + '_null_records')}",
            f"count(*) FILTER (WHERE btrim(t.{quote_ident(name)}::text) = '') AS {quote_ident(name + '_empty_records')}",
        ])
    quality_rows = ctx.db.aggregate(prefix + ".global_key_quality", f"SELECT {', '.join(quality_selects)} FROM ONLY {qualified(table)} t", (), max_rows=1)
    if len(quality_rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    quality = {name: count_value(quality_rows[0][name]) for name in ("records", "null_key_records", "empty_key_records", "invalid_format_records")}
    quality["by_field"] = [{"field": name, "null_records": count_value(quality_rows[0][name + "_null_records"]), "empty_records": count_value(quality_rows[0][name + "_empty_records"])} for name in RACE_KEY]
    return {
        "status": "completed_explicit_full_table_scan",
        "year_counts": counts,
        "period": audit_period(ctx, table, "TRUE", (), prefix + ".global"),
        "key_quality": quality,
        "scope": "all years and all venue codes",
    }


def reference_comparison(ctx, table: str, categories) -> dict:
    expected = {
        "jvd_ra": {"7": {"records": 3455, "distinct_races": 3455}, "9": {"records": 13, "distinct_races": 13}},
        "jvd_se": {"7": {"records": 47884, "distinct_races": 3455}, "9": {"records": 174, "distinct_races": 13}},
    }[table]
    applies = ctx.scope.year == 2025 and set(ctx.scope.venues) == {f"{n:02}" for n in range(1, 11)}
    result = {
        "status": "applicable" if applies else "outside_reference_scope",
        "reference_year": 2025, "reference_venues": [f"{n:02}" for n in range(1, 11)],
        "expected": expected,
        "interpretation": "参考値。更新・条件差による変化があり得るため、差異だけで欠損と断定しない。",
    }
    if applies and isinstance(categories, list):
        observed = {row["category"]: row for row in categories}
        result["difference"] = {
            category: {field: observed.get(category, {}).get(field, 0) - value for field, value in fields.items()}
            for category, fields in expected.items()
        }
    elif applies:
        result["comparison_status"] = "needs_explicit_status_column_mapping"
    return result


def audit_races(ctx) -> dict:
    table = "jvd_ra"
    unavailable = required_key_status(ctx, table)
    if unavailable:
        return unavailable
    where, params = scope_where(ctx.scope)
    key_fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    valid = valid_key_format()
    projection = key_projection()
    complete = complete_key()
    nonempty = nonempty_key()
    summary_sql = f"""
WITH scoped AS (
    SELECT {projection} FROM ONLY {qualified(table)} t WHERE {where}
), groups AS (
    SELECT {key_fields}, count(*) AS n FROM scoped t
    WHERE {complete} GROUP BY {key_fields}
)
SELECT (SELECT count(*) FROM scoped) AS records,
       (SELECT count(*) FROM scoped t WHERE NOT ({complete})) AS null_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete} AND NOT ({nonempty})) AS empty_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete} AND NOT ({valid})) AS invalid_format_records,
       (SELECT count(*) FROM groups) AS distinct_complete_races,
       (SELECT count(*) FROM groups WHERE n > 1) AS duplicate_key_groups,
       (SELECT coalesce(sum(n - 1), 0)::bigint FROM groups WHERE n > 1) AS duplicate_excess_records
"""
    rows = ctx.db.aggregate("ra.summary", summary_sql, params, max_rows=1)
    if len(rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    summary = {field: count_value(rows[0][field]) for field in (
        "records", "null_key_records", "empty_key_records", "invalid_format_records",
        "distinct_complete_races", "duplicate_key_groups", "duplicate_excess_records",
    )}
    fields_sql = " UNION ALL ".join(
        f"SELECT '{name}' AS field, count(*) FILTER (WHERE {quote_ident(name)} IS NULL) AS null_records, "
        f"count(*) FILTER (WHERE btrim({quote_ident(name)}::text) = '') AS empty_records FROM scoped"
        for name in RACE_KEY
    )
    null_rows = ctx.db.aggregate(
        "ra.key_quality", f"WITH scoped AS (SELECT {projection} FROM ONLY {qualified(table)} t WHERE {where}) {fields_sql}",
        params, max_rows=len(RACE_KEY),
    )
    field_quality = []
    for row in null_rows:
        if row.get("field") not in RACE_KEY:
            raise ValueError("unexpected_aggregate_field")
        field_quality.append({"field": row["field"], "null_records": count_value(row["null_records"]), "empty_records": count_value(row["empty_records"])})
    venue_value = f"btrim(t.{quote_ident('keibajo_code')}::text)"
    venue_sql = f"""
SELECT CASE WHEN {venue_value} ~ '^[0-9]{{2}}$' THEN {venue_value} ELSE 'other' END AS venue,
       count(*) AS records
FROM ONLY {qualified(table)} t WHERE {where} GROUP BY 1 ORDER BY 1
"""
    venue_counts = take_count_rows(ctx.db.aggregate("ra.venues", venue_sql, params, max_rows=101), "venue", {f"{n:02}" for n in range(100)} | {"other"})
    result = {
        "status": "completed", "table": table,
        "scope": {"year": ctx.scope.year, "venues": list(ctx.scope.venues)},
        "year_counts": [{"year": ctx.scope.year, "records": summary["records"]}],
        "summary": summary, "key_field_quality": field_quality, "venue_counts": venue_counts,
        "period": audit_period(ctx, table, where, params, "ra"),
        "key_comparison_policy": "btrim(column::text), zero padding preserved, non-NULL six-column keys",
        "format_rules_status": "provisional_syntax_checks_require_specification_review",
        "coverage": {"status": "scoped_year_only", "year": ctx.scope.year, "note": "年を限定した監査。全DBの最古・最新年は未走査。選択条件に合わない年・競馬場の不正値は検出範囲外。"},
        "data_category_mapping": category_mapping_note(ctx, table),
        "notes": ["欠損・異常は件数のみ出力。個別レースキーは出力しない。", "有効形式は暫定の数字・日付形式検査。仕様上の妥当性を保証しない。"],
    }
    mapped = category_column(ctx, table)
    if mapped:
        cat = category_expression(mapped)
        category_sql = f"""
WITH scoped AS (
 SELECT {projection}, {cat} AS category FROM ONLY {qualified(table)} t WHERE {where}
)
SELECT category, count(*) AS records,
       count(DISTINCT ({key_fields})) FILTER (WHERE {complete_key('scoped')}) AS distinct_races
FROM scoped GROUP BY category ORDER BY category
"""
        result["category_counts"] = take_count_rows(ctx.db.aggregate("ra.categories", category_sql, params, max_rows=12), "category", set("0123456789") | {"null", "other"})
        result["execution_categories"] = {"7": "user_reported_final_results", "9": "user_reported_cancelled", "other_categories": "needs_confirmation"}
    else:
        result["category_counts"] = {"status": "needs_confirmation", "reason": "explicit_data_category_mapping_required"}
        result["execution_categories"] = {"status": "needs_confirmation"}
    if getattr(ctx.scope, "global_summary", False):
        result["global_summary"] = audit_global_years(ctx, table, "ra")
    else:
        result["global_summary"] = {"status": "not_requested", "reason": "full_table_scan_requires_explicit_flag"}
    result["reference_2025"] = reference_comparison(ctx, table, result["category_counts"])
    return result
