"""Inspect PostgreSQL catalog structure without selecting application records.

Definitions, comments, defaults, statistics sample values, and index expressions
are deliberately excluded: those fields can contain data or secrets.
"""

from __future__ import annotations

from typing import Any


TABLES_SQL = """
SELECT c.relname AS name,
       CASE c.relkind
         WHEN 'r' THEN 'table'
         WHEN 'p' THEN 'partitioned_table'
         WHEN 'f' THEN 'foreign_table'
         WHEN 'v' THEN 'view'
         WHEN 'm' THEN 'materialized_view'
       END AS kind,
       CASE WHEN c.reltuples < 0 THEN NULL
            ELSE round(c.reltuples)::bigint END AS estimated_rows,
       am.amname AS table_access_method,
       CASE WHEN c.relkind IN ('r', 'm') AND am.amname = 'heap'
            THEN pg_catalog.pg_table_size(c.oid) ELSE NULL END AS size_bytes,
       CASE WHEN c.relkind IN ('r', 'm') AND am.amname = 'heap'
            THEN pg_catalog.pg_indexes_size(c.oid) ELSE NULL END AS index_size_bytes,
       CASE WHEN c.relkind IN ('r', 'm') AND am.amname = 'heap'
            THEN pg_catalog.pg_total_relation_size(c.oid) ELSE NULL END AS total_size_bytes,
       c.relispartition AS is_partition,
       c.relhassubclass AS has_descendants,
       c.relrowsecurity AS row_security,
       c.relforcerowsecurity AS force_row_security,
       stats.last_analyze,
       stats.last_autoanalyze,
       stats.n_live_tup AS statistics_live_rows,
       stats.n_dead_tup AS statistics_dead_rows,
       stats.n_mod_since_analyze AS modifications_since_analyze,
       stats.analyze_count,
       stats.autoanalyze_count
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_am am ON am.oid = c.relam
LEFT JOIN pg_catalog.pg_stat_all_tables stats ON stats.relid = c.oid
WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'f', 'v', 'm')
ORDER BY c.relname
"""

COLUMNS_SQL = """
SELECT a.attname AS name,
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
       typ.typname AS type_name,
       (typ_n.nspname = 'pg_catalog') AS type_is_builtin,
       (NOT a.attnotnull) AS nullable,
       a.attnum AS ordinal_position
FROM pg_catalog.pg_attribute a
JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_type typ ON typ.oid = a.atttypid
JOIN pg_catalog.pg_namespace typ_n ON typ_n.oid = typ.typnamespace
WHERE n.nspname = %s AND c.relname = %s
  AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""

PRIMARY_KEY_SQL = """
SELECT con.conname AS name,
       array_agg(a.attname ORDER BY keys.position) AS columns
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN LATERAL unnest(con.conkey)
    WITH ORDINALITY AS keys(attnum, position)
JOIN pg_catalog.pg_attribute a
  ON a.attrelid = con.conrelid AND a.attnum = keys.attnum
WHERE n.nspname = %s AND c.relname = %s AND con.contype = 'p'
GROUP BY con.conname
ORDER BY con.conname
"""

FOREIGN_KEYS_SQL = """
SELECT con.conname AS name,
       array_agg(local_a.attname ORDER BY keys.position) AS columns,
       referenced_n.nspname AS referenced_schema,
       referenced_c.relname AS referenced_table,
       array_agg(referenced_a.attname ORDER BY keys.position) AS referenced_columns,
       con.convalidated AS validated
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_class referenced_c ON referenced_c.oid = con.confrelid
JOIN pg_catalog.pg_namespace referenced_n ON referenced_n.oid = referenced_c.relnamespace
CROSS JOIN LATERAL unnest(con.conkey, con.confkey)
    WITH ORDINALITY AS keys(local_attnum, referenced_attnum, position)
JOIN pg_catalog.pg_attribute local_a
  ON local_a.attrelid = con.conrelid AND local_a.attnum = keys.local_attnum
JOIN pg_catalog.pg_attribute referenced_a
  ON referenced_a.attrelid = con.confrelid AND referenced_a.attnum = keys.referenced_attnum
WHERE n.nspname = %s AND c.relname = %s AND con.contype = 'f'
GROUP BY con.conname, referenced_n.nspname, referenced_c.relname, con.convalidated
ORDER BY con.conname
"""

INDEXES_SQL = """
SELECT index_c.relname AS name,
       am.amname AS access_method,
       i.indisunique AS unique,
       i.indisprimary AS primary,
       i.indisvalid AS valid,
       i.indisready AS ready,
       (i.indexprs IS NOT NULL) AS has_expressions,
       (i.indpred IS NOT NULL) AS has_predicate,
       COALESCE(array_agg(a.attname ORDER BY keys.position)
           FILTER (WHERE keys.position <= i.indnkeyatts AND keys.attnum > 0),
           ARRAY[]::name[]) AS columns,
       COALESCE(array_agg(a.attname ORDER BY keys.position)
           FILTER (WHERE keys.position > i.indnkeyatts AND keys.attnum > 0),
           ARRAY[]::name[]) AS included_columns
FROM pg_catalog.pg_index i
JOIN pg_catalog.pg_class c ON c.oid = i.indrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_class index_c ON index_c.oid = i.indexrelid
JOIN pg_catalog.pg_am am ON am.oid = index_c.relam
CROSS JOIN LATERAL unnest(i.indkey)
    WITH ORDINALITY AS keys(attnum, position)
LEFT JOIN pg_catalog.pg_attribute a
  ON a.attrelid = i.indrelid AND a.attnum = keys.attnum
WHERE n.nspname = %s AND c.relname = %s
GROUP BY index_c.relname, am.amname, i.indisunique, i.indisprimary,
         i.indisvalid, i.indisready, i.indexprs IS NOT NULL, i.indpred IS NOT NULL
ORDER BY index_c.relname
"""


def inspect_schema(db: Any) -> dict[str, Any]:
    """Return bounded public-schema metadata from catalogs only.

    ``db.aggregate`` must enforce its row limit and redact database errors.
    This function does not discover foreign servers or inspect their options.
    Estimates come from ``pg_class.reltuples``; no full table count is used.
    Sizes are for each relation itself, excluding partition descendants.
    """
    inventory = db.aggregate("schema.tables", TABLES_SQL, ("public",), max_rows=1000)
    tables: list[dict[str, Any]] = []
    warnings = [
        "概算件数は pg_class.reltuples に基づき、統計未取得・更新遅延で不明または不正確になり得ます。",
        "サイズは各relation自体の値です。親テーブルに子・partitionのサイズを合算していません。",
        "heapの通常テーブル以外とRLS有効テーブルは構造一覧だけに含め、実レコード監査から除外します。",
        "式、条件、デフォルト値、コメント、統計のサンプル値は出力しません。",
    ]
    for row in inventory:
        # Explicit allowlists also prevent accidental leakage after query edits.
        table = {
            key: row.get(key)
            for key in (
                "name", "kind", "estimated_rows", "size_bytes", "index_size_bytes",
                "total_size_bytes", "is_partition", "has_descendants",
                "row_security", "force_row_security",
                "table_access_method",
                "last_analyze", "last_autoanalyze", "statistics_live_rows", "statistics_dead_rows",
                "modifications_since_analyze", "analyze_count", "autoanalyze_count",
            )
        }
        table["estimated_rows_source"] = "pg_class.reltuples"
        table["statistics_update_status"] = (
            "analyze_timestamp_available"
            if table["last_analyze"] is not None or table["last_autoanalyze"] is not None
            else "analyze_timestamp_unavailable_or_counters_reset"
        )
        table["auditable"] = (
            table["kind"] == "table"
            and table["table_access_method"] == "heap"
            and table["row_security"] is False
            and table["force_row_security"] is False
        )
        table["warnings"] = []
        if not table["auditable"]:
            table["warnings"].append("heap通常テーブル以外またはRLS有効・状態未確認のため、実レコード監査対象外。")
        if table["estimated_rows"] is None:
            table["warnings"].append("統計未取得などにより概算件数は不明。")
        if table["has_descendants"]:
            table["warnings"].append("実監査はONLYにより子を除外。子の収録範囲は要確認。")
        params = ("public", table["name"])
        table["columns"] = [
            {key: column.get(key) for key in ("name", "data_type", "type_name", "type_is_builtin", "nullable", "ordinal_position")}
            for column in db.aggregate("schema.columns", COLUMNS_SQL, params, max_rows=1600)
        ]
        primary = db.aggregate("schema.primary_key", PRIMARY_KEY_SQL, params, max_rows=1)
        table["primary_key"] = list(primary[0].get("columns") or []) if primary else []
        table["primary_key_name"] = primary[0].get("name") if primary else None
        table["foreign_keys"] = [
            {key: fk.get(key) for key in (
                "name", "columns", "referenced_schema", "referenced_table", "referenced_columns", "validated",
            )}
            for fk in db.aggregate("schema.foreign_keys", FOREIGN_KEYS_SQL, params, max_rows=1000)
        ]
        table["indexes"] = [
            {key: index.get(key) for key in (
                "name", "access_method", "unique", "primary", "valid", "ready",
                "has_expressions", "has_predicate", "columns", "included_columns",
            )}
            for index in db.aggregate("schema.indexes", INDEXES_SQL, params, max_rows=1000)
        ]
        tables.append(table)
    by_kind: dict[str, int] = {}
    for table in tables:
        by_kind[table["kind"]] = by_kind.get(table["kind"], 0) + 1
    return {
        "status": "ok", "tables": tables, "warnings": warnings,
        "catalog_counts": {
            "listed_relations": len(tables), "by_kind": by_kind,
            "table_relations": sum(by_kind.get(kind, 0) for kind in ("table", "partitioned_table", "foreign_table")),
            "auditable_heap_tables": sum(table["auditable"] for table in tables),
        },
    }
