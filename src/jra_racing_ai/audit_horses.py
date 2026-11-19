"""Horse-record and race-set audits using only bounded, anonymous aggregates."""

from __future__ import annotations

from .core import RACE_KEY, column_names, qualified, quote_ident, scope_where, usable_columns
from .audit_races import (
    audit_global_years, audit_period, category_column, category_expression,
    category_mapping_note, complete_key, count_value, key_projection, nonempty_key,
    reference_comparison, required_key_status, take_count_rows, valid_key_format,
)


def race_set_audit(ctx) -> dict:
    unavailable = required_key_status(ctx, "jvd_ra")
    if unavailable:
        return unavailable
    where, params = scope_where(ctx.scope)
    fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    sql = f"""
WITH ra AS (
 SELECT DISTINCT {key_projection()} FROM ONLY {qualified('jvd_ra')} t WHERE {where} AND {complete_key()}
), se AS (
 SELECT DISTINCT {key_projection()} FROM ONLY {qualified('jvd_se')} t WHERE {where} AND {complete_key()}
), ra_valid AS (SELECT {fields} FROM ra t WHERE {valid_key_format()}),
   se_valid AS (SELECT {fields} FROM se t WHERE {valid_key_format()}),
   only_ra AS (SELECT {fields} FROM ra EXCEPT SELECT {fields} FROM se),
   only_se AS (SELECT {fields} FROM se EXCEPT SELECT {fields} FROM ra),
   shared AS (SELECT {fields} FROM ra INTERSECT SELECT {fields} FROM se),
   only_ra_valid AS (SELECT {fields} FROM ra_valid EXCEPT SELECT {fields} FROM se_valid),
   only_se_valid AS (SELECT {fields} FROM se_valid EXCEPT SELECT {fields} FROM ra_valid)
SELECT (SELECT count(*) FROM ra) AS ra_complete_distinct_races,
       (SELECT count(*) FROM se) AS se_complete_distinct_races,
       (SELECT count(*) FROM shared) AS shared_complete_races,
       (SELECT count(*) FROM only_ra) AS races_only_in_ra,
       (SELECT count(*) FROM only_se) AS races_only_in_se,
       (SELECT count(*) FROM ra_valid) AS ra_valid_format_distinct_races,
       (SELECT count(*) FROM se_valid) AS se_valid_format_distinct_races,
       (SELECT count(*) FROM only_ra_valid) AS valid_format_races_only_in_ra,
       (SELECT count(*) FROM only_se_valid) AS valid_format_races_only_in_se
"""
    rows = ctx.db.aggregate("se.race_sets", sql, params + params, max_rows=1)
    if len(rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    metrics = (
        "ra_complete_distinct_races", "se_complete_distinct_races", "shared_complete_races",
        "races_only_in_ra", "races_only_in_se", "ra_valid_format_distinct_races", "se_valid_format_distinct_races",
        "valid_format_races_only_in_ra", "valid_format_races_only_in_se",
    )
    result = {
        "status": "completed", "counts": {field: count_value(rows[0][field]) for field in metrics},
        "comparison_policy": "exact six-column btrim(::text) sets, zero padding preserved",
        "null_policy": "rows with any NULL key excluded and reported in each table's key quality",
        "invalid_policy": "non-NULL malformed and empty keys included in complete sets; provisional valid-format sets reported separately",
        "note": "件数が一致しても集合が異なる場合を検出。差分のレースキーは出力しない。表現差・区分・更新状態を確認して解釈する。",
    }
    ra_status = category_column(ctx, "jvd_ra")
    se_status = category_column(ctx, "jvd_se")
    if ra_status and se_status:
        cat_sql = f"""
WITH ra AS (
 SELECT DISTINCT {key_projection()}, {category_expression(ra_status)} AS category
 FROM ONLY {qualified('jvd_ra')} t WHERE {where} AND {complete_key()}
), se AS (
 SELECT DISTINCT {key_projection()}, {category_expression(se_status)} AS category
 FROM ONLY {qualified('jvd_se')} t WHERE {where} AND {complete_key()}
), only_ra AS (SELECT * FROM ra EXCEPT SELECT * FROM se),
   only_se AS (SELECT * FROM se EXCEPT SELECT * FROM ra),
   buckets AS (SELECT category FROM ra UNION SELECT category FROM se)
SELECT category,
       (SELECT count(*) FROM ra r WHERE r.category = buckets.category) AS ra_distinct_races,
       (SELECT count(*) FROM se s WHERE s.category = buckets.category) AS se_distinct_races,
       (SELECT count(*) FROM only_ra r WHERE r.category = buckets.category) AS races_only_in_ra,
       (SELECT count(*) FROM only_se s WHERE s.category = buckets.category) AS races_only_in_se
FROM buckets ORDER BY category
"""
        category_rows = ctx.db.aggregate("se.category_race_sets", cat_sql, params + params, max_rows=12)
        result["by_category"] = []
        for row in category_rows:
            if row.get("category") not in set("0123456789") | {"null", "other"}:
                raise ValueError("unexpected_aggregate_bucket")
            result["by_category"].append({"category": row["category"], **{
                field: count_value(row[field]) for field in ("ra_distinct_races", "se_distinct_races", "races_only_in_ra", "races_only_in_se")
            }})
    else:
        result["by_category"] = {"status": "needs_confirmation", "reason": "both_tables_need_explicit_status_column_mapping"}
    return result


def horse_duplicate_audit(ctx) -> dict:
    table = "jvd_se"
    identities = tuple(getattr(ctx, "horse_identity_columns", ()))
    primary_key = ctx.schema[table].get("primary_key", [])
    # PK metadata describes database grain, not necessarily horse-record identity.
    metadata = {"database_primary_key_columns": list(primary_key), "primary_key_semantics": "needs_confirmation"}
    if not identities:
        return {
            "status": "needs_confirmation", "reason": "explicit_horse_identity_columns_required",
            **metadata,
            "note": "レースキーの繰り返しは馬単位レコードで想定される。馬番・馬ID・DB主キーの意味を推測して重複判定しない。",
        }
    if len(set(identities)) != len(identities) or any(name not in column_names(ctx, table) for name in identities):
        return {"status": "needs_confirmation", "reason": "horse_identity_columns_invalid_or_missing", **metadata}
    if not set(identities) - set(RACE_KEY):
        return {"status": "needs_confirmation", "reason": "horse_identity_needs_at_least_one_nonrace_column", **metadata}
    if not usable_columns(ctx, table, identities):
        return {"status": "needs_confirmation", "reason": "horse_identity_column_type_requires_review", **metadata}
    combined = tuple(dict.fromkeys(RACE_KEY + identities))
    fields = ", ".join(quote_ident(name) for name in combined)
    projection = ", ".join(f"t.{quote_ident(name)} AS {quote_ident(name)}" for name in combined)
    notnull = " AND ".join(f"{quote_ident(name)} IS NOT NULL" for name in combined)
    where, params = scope_where(ctx.scope)
    sql = f"""
WITH scoped AS (
 SELECT {projection} FROM ONLY {qualified(table)} t WHERE {where}
), groups AS (
 SELECT {fields}, count(*) AS n FROM scoped WHERE {notnull} GROUP BY {fields}
)
SELECT (SELECT count(*) FROM groups WHERE n > 1) AS duplicate_identity_groups,
       (SELECT coalesce(sum(n - 1), 0)::bigint FROM groups WHERE n > 1) AS duplicate_excess_records,
       (SELECT count(*) FROM scoped WHERE NOT ({notnull})) AS null_identity_records
"""
    rows = ctx.db.aggregate("se.horse_duplicates", sql, params, max_rows=1)
    if len(rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    return {
        "status": "completed_user_mapped_identity", "identity_columns": list(combined), **metadata,
        "counts": {field: count_value(rows[0][field]) for field in ("duplicate_identity_groups", "duplicate_excess_records", "null_identity_records")},
        "note": "ユーザー指定の識別列によるDB値の重複。時点別・区分別レコードの正当な繰り返しとの区別は仕様で要確認。",
    }


def audit_horses(ctx) -> dict:
    table = "jvd_se"
    unavailable = required_key_status(ctx, table)
    if unavailable:
        return unavailable
    where, params = scope_where(ctx.scope)
    fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    sql = f"""
WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified(table)} t WHERE {where}),
     groups AS (SELECT {fields}, count(*) AS n FROM scoped t WHERE {complete_key()} GROUP BY {fields})
SELECT (SELECT count(*) FROM scoped) AS records,
       (SELECT count(*) FROM scoped t WHERE NOT ({complete_key()})) AS null_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete_key()} AND NOT ({nonempty_key()})) AS empty_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete_key()} AND NOT ({valid_key_format()})) AS invalid_format_records,
       (SELECT count(*) FROM groups) AS distinct_complete_races,
       (SELECT count(*) FROM groups WHERE n > 1) AS races_with_multiple_records,
       (SELECT coalesce(sum(n - 1), 0)::bigint FROM groups WHERE n > 1) AS records_beyond_one_per_race
"""
    rows = ctx.db.aggregate("se.summary", sql, params, max_rows=1)
    if len(rows) != 1:
        raise ValueError("unexpected_aggregate_shape")
    summary = {field: count_value(rows[0][field]) for field in (
        "records", "null_key_records", "empty_key_records", "invalid_format_records",
        "distinct_complete_races", "races_with_multiple_records", "records_beyond_one_per_race",
    )}
    quality_sql = " UNION ALL ".join(
        f"SELECT '{name}' AS field, count(*) FILTER (WHERE {quote_ident(name)} IS NULL) AS null_records, "
        f"count(*) FILTER (WHERE btrim({quote_ident(name)}::text) = '') AS empty_records FROM scoped"
        for name in RACE_KEY
    )
    quality_rows = ctx.db.aggregate("se.key_quality", f"WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified(table)} t WHERE {where}) {quality_sql}", params, max_rows=len(RACE_KEY))
    quality = []
    for row in quality_rows:
        if row.get("field") not in RACE_KEY:
            raise ValueError("unexpected_aggregate_field")
        quality.append({"field": row["field"], "null_records": count_value(row["null_records"]), "empty_records": count_value(row["empty_records"])})
    histogram_sql = f"""
WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified(table)} t WHERE {where}),
     counts AS (SELECT count(*) AS n FROM scoped t WHERE {complete_key()} GROUP BY {fields})
SELECT CASE WHEN n <= 40 THEN n::text WHEN n <= 100 THEN '41-100'
            WHEN n <= 1000 THEN '101-1000' ELSE '1001+' END AS records_per_race,
       count(*) AS races
FROM counts GROUP BY 1 ORDER BY 1
"""
    histogram = []
    for row in ctx.db.aggregate("se.horse_count_distribution", histogram_sql, params, max_rows=43):
        if row.get("records_per_race") not in {str(n) for n in range(1, 41)} | {"41-100", "101-1000", "1001+"}:
            raise ValueError("unexpected_aggregate_bucket")
        histogram.append({"records_per_race": row["records_per_race"], "races": count_value(row["races"])})
    result = {
        "status": "completed", "table": table,
        "scope": {"year": ctx.scope.year, "venues": list(ctx.scope.venues)},
        "year_counts": [{"year": ctx.scope.year, "records": summary["records"]}],
        "summary": summary, "key_field_quality": quality,
        "horse_records_per_race_distribution": histogram,
        "distribution_scope": "non-NULL race keys; counts are records, not proven unique horses",
        "period": audit_period(ctx, table, where, params, "se"),
        "race_set_comparison": race_set_audit(ctx),
        "horse_identity_duplicates": horse_duplicate_audit(ctx),
        "data_category_mapping": category_mapping_note(ctx, table),
        "notes": ["1レースに複数の馬単位レコードがあることは重複異常とはしない。", "集計対象年・競馬場条件に合わない不正な年・競馬場値は検出範囲外。"],
    }
    mapped = category_column(ctx, table)
    if mapped:
        category_sql = f"""
WITH scoped AS (
 SELECT {key_projection()}, {category_expression(mapped)} AS category FROM ONLY {qualified(table)} t WHERE {where}
)
SELECT category, count(*) AS records,
       count(DISTINCT ({fields})) FILTER (WHERE {complete_key('scoped')}) AS distinct_races
FROM scoped GROUP BY category ORDER BY category
"""
        result["category_counts"] = take_count_rows(ctx.db.aggregate("se.categories", category_sql, params, max_rows=12), "category", set("0123456789") | {"null", "other"})
    else:
        result["category_counts"] = {"status": "needs_confirmation", "reason": "explicit_data_category_mapping_required"}
    if getattr(ctx.scope, "global_summary", False):
        result["global_summary"] = audit_global_years(ctx, table, "se")
    else:
        result["global_summary"] = {"status": "not_requested", "reason": "full_table_scan_requires_explicit_flag"}
    result["reference_2025"] = reference_comparison(ctx, table, result["category_counts"])
    return result
