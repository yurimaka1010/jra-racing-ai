"""HR record-presence audit. No payout payloads or amounts leave PostgreSQL."""
from __future__ import annotations

from .core import RACE_KEY, SafeAuditError, column_names, qualified, quote_ident, scope_where, status_column, usable_columns
from .audit_races import (
    audit_global_years, category_expression, complete_key, count_value,
    key_projection, nonempty_key, required_key_status, take_count_rows,
    valid_key_format,
)
from .payout_policy import policy_summary


_HR_STATUS_REFERENCE = {
    "0": "削除", "1": "速報成績（払戻金確定）", "2": "成績（月曜）", "9": "レース中止",
}


def _one_counts(ctx, query_id: str, sql: str, params: tuple, fields: tuple[str, ...]) -> dict:
    rows = ctx.db.aggregate(query_id, sql, params, max_rows=1)
    if len(rows) != 1:
        raise SafeAuditError("internal_error")
    # Extra fields are discarded; arbitrary data cannot pass as a count.
    try:
        return {field: count_value(rows[0].get(field)) for field in fields}
    except (ValueError, TypeError):
        raise SafeAuditError("internal_error") from None


def _structure(ctx) -> dict:
    table = ctx.schema.get("jvd_hr")
    if table is None:
        return {"status": "needs_confirmation", "exists": False}
    columns = [{k: c.get(k) for k in ("name", "data_type", "type_is_builtin", "nullable")}
               for c in table.get("columns", [])]
    names = column_names(ctx, "jvd_hr")
    # These are lexical candidates only; their values are never selected.
    groups = {
        "status": ("kubun", "kbn", "status"),
        "win_payout": ("tansho", "tansyo", "win_pay"),
        "place_payout": ("fukusho", "fukusyo", "place_pay"),
        "not_established": ("fuseiritsu", "fuseiritu", "not_established"),
        "special_payment": ("tokubarai", "tokuharai", "special_pay"),
        "refund": ("henkan", "refund"),
        "registered_or_starter_count": ("toroku", "syusso", "shusso", "starter", "entry_count"),
    }
    candidates = {group: [{"name": col["name"], "data_type": col["data_type"],
                            "status": "needs_confirmation", "evidence": "name_only"}
                           for col in columns if any(token in col["name"].lower() for token in tokens)]
                  for group, tokens in groups.items()}
    return {"status": "needs_confirmation", "catalog_structure_observed": True,
            "evidence": "catalog_structure_only_not_field_semantics",
            "exists": True, "auditable": table.get("auditable") is True,
            "kind": table.get("kind"), "columns": columns,
            "primary_key": list(table.get("primary_key") or []),
            "indexes": [{k: idx.get(k) for k in (
                "name", "access_method", "unique", "primary", "valid", "ready",
                "columns", "included_columns", "has_expressions", "has_predicate")}
                for idx in table.get("indexes", [])],
            "race_key_columns_present": [name for name in RACE_KEY if name in names],
            "missing_race_key_columns": [name for name in RACE_KEY if name not in names],
            "column_candidates": candidates, "record_grain_status": "needs_confirmation"}


def _race_sets(ctx, where: str, params: tuple) -> dict:
    unavailable = required_key_status(ctx, "jvd_ra")
    if unavailable:
        return unavailable
    fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    sql = f"""
WITH ra AS (
 SELECT DISTINCT {key_projection()} FROM ONLY {qualified('jvd_ra')} t
 WHERE {where} AND {complete_key()}
), hr AS (
 SELECT DISTINCT {key_projection()} FROM ONLY {qualified('jvd_hr')} t
 WHERE {where} AND {complete_key()}
), ra_valid AS (SELECT {fields} FROM ra t WHERE {valid_key_format()}),
   hr_valid AS (SELECT {fields} FROM hr t WHERE {valid_key_format()}),
   only_ra AS (SELECT {fields} FROM ra EXCEPT SELECT {fields} FROM hr),
   only_hr AS (SELECT {fields} FROM hr EXCEPT SELECT {fields} FROM ra),
   shared AS (SELECT {fields} FROM ra INTERSECT SELECT {fields} FROM hr),
   only_ra_valid AS (SELECT {fields} FROM ra_valid EXCEPT SELECT {fields} FROM hr_valid),
   only_hr_valid AS (SELECT {fields} FROM hr_valid EXCEPT SELECT {fields} FROM ra_valid)
SELECT (SELECT count(*) FROM ra) AS ra_complete_distinct_races,
       (SELECT count(*) FROM hr) AS hr_complete_distinct_races,
       (SELECT count(*) FROM shared) AS shared_complete_races,
       (SELECT count(*) FROM only_ra) AS races_only_in_ra,
       (SELECT count(*) FROM only_hr) AS races_only_in_hr,
       (SELECT count(*) FROM only_ra_valid) AS valid_format_races_only_in_ra,
       (SELECT count(*) FROM only_hr_valid) AS valid_format_races_only_in_hr
"""
    counts = _one_counts(ctx, "hr.race_sets", sql, params + params, (
        "ra_complete_distinct_races", "hr_complete_distinct_races", "shared_complete_races",
        "races_only_in_ra", "races_only_in_hr", "valid_format_races_only_in_ra", "valid_format_races_only_in_hr"))
    return {"status": "completed", "counts": counts,
            "scope": "all_hr_categories_including_deletion_and_unmapped_categories",
            "note": "非NULLキー全体の構造比較。区分0も含むため払戻利用可能性の証明ではない。NULLは分離、形式正常の差分も併記。"}


def _coverage(ctx, where: str, params: tuple, hr_status: str | None) -> dict:
    unavailable = required_key_status(ctx, "jvd_ra")
    if unavailable:
        return unavailable
    ra_status = status_column(ctx, "jvd_ra")
    if hr_status is None or ra_status is None:
        return {"status": "needs_confirmation", "reason": "both_ra_and_hr_status_mappings_required"}
    if not usable_columns(ctx, "jvd_ra", (ra_status,)):
        return {"status": "needs_confirmation", "reason": "ra_status_column_type_requires_review"}
    fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    match = " AND ".join(f"h.{quote_ident(name)} = r.{quote_ident(name)}" for name in RACE_KEY)
    # Categories are normalized to 0..9/null/other inside PostgreSQL. No raw codes.
    # A repeated HR 1/2 pair might be versions; count existence, never select a winner.
    sql = f"""
WITH ra AS (
 SELECT DISTINCT {key_projection()}, {category_expression(ra_status)} AS category
 FROM ONLY {qualified('jvd_ra')} t WHERE {where} AND {valid_key_format()}
), hr AS (
 SELECT {key_projection()}, {category_expression(hr_status)} AS category
 FROM ONLY {qualified('jvd_hr')} t WHERE {where} AND {valid_key_format()}
), ra_state AS (
 SELECT {fields}, max(category) AS category FROM ra GROUP BY {fields}
 HAVING count(DISTINCT category) = 1
), ra_conflicts AS (
 SELECT {fields} FROM ra GROUP BY {fields} HAVING count(DISTINCT category) > 1
), hr_final AS (
 SELECT {fields} FROM hr GROUP BY {fields} HAVING bool_and(category IN ('1','2'))
), hr_review AS (
 SELECT {fields} FROM hr GROUP BY {fields}
 HAVING bool_or(category NOT IN ('0','1','2','9'))
     OR (bool_or(category = '0') AND bool_or(category <> '0'))
     OR (bool_or(category = '9') AND bool_or(category IN ('1','2')))
), hr_deleted_only AS (
 SELECT {fields} FROM hr GROUP BY {fields} HAVING bool_and(category = '0')
)
SELECT (SELECT count(*) FROM ra_state WHERE category = '7') AS executed_races,
       (SELECT count(*) FROM ra_state WHERE category = '9') AS cancelled_races,
       (SELECT count(*) FROM ra_conflicts) AS ra_conflicting_status_races,
       (SELECT count(*) FROM hr_review) AS hr_review_required_races,
       (SELECT count(*) FROM ra_state r WHERE category = '7' AND EXISTS
           (SELECT 1 FROM hr_final h WHERE {match})) AS executed_races_with_payout_record,
       (SELECT count(*) FROM ra_state r WHERE category = '7' AND NOT EXISTS
           (SELECT 1 FROM hr_final h WHERE {match})) AS executed_races_without_payout_record,
       (SELECT count(*) FROM ra_state r WHERE category = '7' AND NOT EXISTS
           (SELECT 1 FROM hr h WHERE {match})) AS executed_races_without_any_hr_record,
       (SELECT count(*) FROM ra_state r WHERE category = '7' AND EXISTS
           (SELECT 1 FROM hr_review h WHERE {match})) AS executed_races_with_review_required_hr,
       (SELECT count(*) FROM ra_state r WHERE category = '7' AND EXISTS
           (SELECT 1 FROM hr_deleted_only h WHERE {match})) AS executed_races_with_only_deleted_hr,
       (SELECT count(*) FROM ra_state r WHERE category = '9' AND EXISTS
           (SELECT 1 FROM hr h WHERE {match})) AS cancelled_races_with_any_hr_record,
       (SELECT count(*) FROM ra_state r WHERE category = '9' AND EXISTS
           (SELECT 1 FROM hr h WHERE {match} AND h.category = '9')) AS cancelled_races_with_hr_cancelled_status,
       (SELECT count(*) FROM ra_state r WHERE category = '9' AND EXISTS
           (SELECT 1 FROM hr_final h WHERE {match})) AS cancelled_races_with_hr_final_status,
       (SELECT count(*) FROM hr h WHERE h.category IN ('1','2','9') AND EXISTS
           (SELECT 1 FROM ra_state r WHERE r.category = '9' AND {match})) AS cancelled_hr_nondeleted_known_status_records,
       (SELECT count(*) FROM hr h WHERE h.category = '9' AND EXISTS
           (SELECT 1 FROM ra_state r WHERE r.category = '9' AND {match})) AS cancelled_hr_cancelled_status_records
"""
    fields_out = (
        "executed_races", "cancelled_races", "ra_conflicting_status_races", "hr_review_required_races",
        "executed_races_with_payout_record", "executed_races_without_payout_record",
        "executed_races_without_any_hr_record", "executed_races_with_review_required_hr",
        "executed_races_with_only_deleted_hr", "cancelled_races_with_any_hr_record",
        "cancelled_races_with_hr_cancelled_status", "cancelled_races_with_hr_final_status",
        "cancelled_hr_nondeleted_known_status_records", "cancelled_hr_cancelled_status_records",
    )
    return {"status": "completed_user_mapped_status", "counts": _one_counts(
                ctx, "hr.coverage", sql, params + params, fields_out),
            "scope": "provisional_valid_keys_only_ra_single_status",
            "rule": "RA7実施/RA9中止。HRの全レコードが1/2のキーだけを払戻レコード存在と数える。0・9・未知との混在は採用しない。",
            "note": "区分解釈は依頼文の仕様情報が根拠。金額・発売有無・返還・的中対象の可読性は未検証。欠損と即断しない。"}


def audit_payouts(ctx) -> dict:
    """Return counts, structural candidates and explicit unresolved mappings."""
    result = {"status": "needs_confirmation", "table": "jvd_hr",
              "scope": {"year": ctx.scope.year, "venues": list(ctx.scope.venues)},
              "structure": _structure(ctx),
              "jv_data_reference": {"record_type": "HR", "source": "https://jra-van.jp/dlb/sdv/sdk.html",
                  "verification": "user_provided_not_independently_verified", "status": "needs_confirmation",
                  "status_codes": dict(_HR_STATUS_REFERENCE),
                  "win_repetitions": 3, "place_repetitions": 5,
                  "pc_keiba_mapping": "needs_confirmation"},
              "special_case_policy": policy_summary(),
              "payload_decoding": "not_implemented", "profit_calculation": "not_implemented",
              "unresolved": ["JV-Data仕様版とPC-KEIBA列の対応・変換・配列展開。",
                             "金額単位、馬番の桁数、的中組の繰返し・パディング・特殊値。",
                             "不成立、特払、返還、発売なし、欠損・初期値の区別。",
                             "複数レコードの版・更新・削除区分の保持方式。"],
              "notes": ["個別レースキー・馬ID・払戻金額・payload・フラグ値は取得しない。",
                        "年・競馬場がNULL/不正で抽出条件外の行は年限定では検出範囲外。",
                        "集合差や区分差・重複は、更新方式・収録条件を確認して解釈する。"]}
    unavailable = required_key_status(ctx, "jvd_hr")
    if unavailable:
        result["availability"] = unavailable
        return result
    hr_status = status_column(ctx, "jvd_hr")
    if hr_status is not None and not usable_columns(ctx, "jvd_hr", (hr_status,)):
        result["availability"] = {"status": "needs_confirmation", "reason": "payout_status_column_type_requires_review"}
        return result
    # Validate an explicitly named RA status before running heavy queries.
    if "jvd_ra" in ctx.schema:
        status_column(ctx, "jvd_ra")
    where, params = scope_where(ctx.scope)
    fields = ", ".join(quote_ident(name) for name in RACE_KEY)
    sql = f"""
WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified('jvd_hr')} t WHERE {where}),
 groups AS (SELECT {fields}, count(*) AS n FROM scoped t WHERE {complete_key()} GROUP BY {fields})
SELECT (SELECT count(*) FROM scoped) AS records,
       (SELECT count(*) FROM scoped t WHERE NOT ({complete_key()})) AS null_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete_key()} AND NOT ({nonempty_key()})) AS empty_key_records,
       (SELECT count(*) FROM scoped t WHERE {complete_key()} AND NOT ({valid_key_format()})) AS invalid_format_records,
       (SELECT count(*) FROM groups) AS distinct_complete_races,
       (SELECT count(*) FROM groups WHERE n > 1) AS duplicate_key_groups,
       (SELECT coalesce(sum(n - 1), 0)::bigint FROM groups WHERE n > 1) AS duplicate_excess_records
"""
    summary = _one_counts(ctx, "hr.summary", sql, params, (
        "records", "null_key_records", "empty_key_records", "invalid_format_records",
        "distinct_complete_races", "duplicate_key_groups", "duplicate_excess_records"))
    quality_sql = " UNION ALL ".join(
        f"SELECT '{name}' AS field, count(*) FILTER (WHERE {quote_ident(name)} IS NULL) AS null_records, "
        f"count(*) FILTER (WHERE btrim({quote_ident(name)}::text) = '') AS empty_records FROM scoped" for name in RACE_KEY)
    quality = []
    for row in ctx.db.aggregate("hr.key_quality", f"WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified('jvd_hr')} t WHERE {where}) {quality_sql}", params, max_rows=6):
        if row.get("field") not in RACE_KEY:
            raise SafeAuditError("internal_error")
        try:
            quality.append({"field": row["field"], "null_records": count_value(row.get("null_records")),
                            "empty_records": count_value(row.get("empty_records"))})
        except ValueError:
            raise SafeAuditError("internal_error") from None
    venue_sql = f"""
SELECT btrim(t.{quote_ident('keibajo_code')}::text) AS venue, count(*) AS records
FROM ONLY {qualified('jvd_hr')} t WHERE {where} GROUP BY 1 ORDER BY 1
"""
    # Scope predicate makes only the approved domestic codes reachable.
    venues = take_count_rows(ctx.db.aggregate("hr.venues", venue_sql, params, max_rows=10), "venue", set(ctx.scope.venues))
    histogram_sql = f"""
WITH scoped AS (SELECT {key_projection()} FROM ONLY {qualified('jvd_hr')} t WHERE {where}),
 groups AS (SELECT count(*) AS n FROM scoped t WHERE {complete_key()} GROUP BY {fields})
SELECT CASE WHEN n <= 10 THEN n::text WHEN n <= 100 THEN '11-100' ELSE '101+' END AS records_per_race,
       count(*) AS races FROM groups GROUP BY 1 ORDER BY 1
"""
    histogram = []
    for row in ctx.db.aggregate("hr.records_per_race", histogram_sql, params, max_rows=12):
        if row.get("records_per_race") not in {str(n) for n in range(1, 11)} | {"11-100", "101+"}:
            raise SafeAuditError("internal_error")
        try:
            histogram.append({"records_per_race": row["records_per_race"], "races": count_value(row.get("races"))})
        except ValueError:
            raise SafeAuditError("internal_error") from None
    result.update(status="completed_aggregates_semantics_need_confirmation", summary=summary,
                  year_counts=[{"year": ctx.scope.year, "records": summary["records"]}],
                  venue_counts=venues, key_field_quality=quality,
                  records_per_race_distribution=histogram,
                  key_comparison_policy="btrim(::text), no zero padding, six columns, NULL keys isolated",
                  record_grain_status="needs_confirmation_repetition_does_not_prove_corruption")
    result["status_mapping"] = {"status": "needs_confirmation" if hr_status is None else "user_mapped_unverified_specification",
                                "mapped_column": hr_status, "interpretation_basis": "user_provided_jv_data_hr_reference"}
    if hr_status is None:
        result["category_counts"] = {"status": "needs_confirmation", "reason": "explicit_payout_status_column_required"}
    else:
        category_sql = f"""
WITH scoped AS (SELECT {key_projection()}, {category_expression(hr_status)} AS category
 FROM ONLY {qualified('jvd_hr')} t WHERE {where})
SELECT category, count(*) AS records,
       count(DISTINCT ({fields})) FILTER (WHERE {complete_key('scoped')}) AS distinct_races
FROM scoped GROUP BY category ORDER BY category
"""
        result["category_counts"] = take_count_rows(ctx.db.aggregate("hr.categories", category_sql, params, max_rows=12), "category", set("0123456789") | {"null", "other"})
    result["race_set_comparison"] = _race_sets(ctx, where, params)
    result["settlement_record_coverage"] = _coverage(ctx, where, params, hr_status)
    result["global_summary"] = (audit_global_years(ctx, "jvd_hr", "hr") if ctx.scope.global_summary else
                                {"status": "not_requested", "reason": "full_table_scan_requires_explicit_flag"})
    return result
