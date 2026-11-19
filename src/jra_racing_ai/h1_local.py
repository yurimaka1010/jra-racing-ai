"""Bounded H1 reads using the same local-only, read-only session as HR."""
from __future__ import annotations

from .core import RACE_KEY, quote_ident
from .h1_reconciliation import H1Config, LocalH1Record, check_h1_budget, parse_h1_config
from .inspect_schema import COLUMNS_SQL
from .payout_reconciliation import RaceKey, ReconciliationError, _require
from .reconciliation_local import _HR_TABLE_SQL, _query, _read_json_file


def read_h1_config_file(path):
    return _read_json_file(path, parse_h1_config)


def fetch_local_h1(db, document, config, limits, budget, *, max_rows=40):
    _require(type(config) is H1Config, "invalid_h1_config")
    config.check_document(document)
    limits.check_document(document)
    check_h1_budget(limits, max_rows)
    table = _query(db, budget, "reconciliation.h1_table", _HR_TABLE_SQL, ("public", "jvd_h1"), 1)
    if (len(table) != 1 or table[0].get("kind") != "r" or table[0].get("method") != "heap"
            or table[0].get("row_security") is not False or table[0].get("force_row_security") is not False):
        raise ReconciliationError("h1_table_unavailable")
    columns = _query(db, budget, "reconciliation.h1_columns", COLUMNS_SQL, ("public", "jvd_h1"), 1600)
    metadata = {c["name"]: c for c in columns}
    fields = list(zip(RACE_KEY, (4, 4, 2, 2, 2, 2))) + [(c, 1) for c in config.h1_columns]
    if any(metadata.get(c, {}).get("type_is_builtin") is not True
           or metadata.get(c, {}).get("type_name") not in ("text", "varchar", "bpchar") for c, _ in fields):
        raise ReconciliationError("h1_column_mapping_unconfirmed")
    aliases = dict(zip(config.h1_columns, ("h1_category", "h1_win_sale", "h1_place_sale", "h1_paid_positions")))
    selects, oversized = [], []
    for name, width in fields:
        col = "t." + quote_ident(name)
        selects.append(f"CASE WHEN pg_catalog.octet_length({col}) <= {width} THEN {col} ELSE NULL END AS "
                       + quote_ident(aliases.get(name, name)))
        oversized.append(f"pg_catalog.octet_length({col}) > {width}")
    selects.append("(" + " OR ".join(oversized) + ") IS TRUE AS oversized_fields")
    where = " OR ".join("(" + " AND ".join(f"t.{quote_ident(k)} = %s" for k in RACE_KEY) + ")" for _ in document.races)
    sql = 'SELECT ' + ", ".join(selects) + ' FROM ONLY "public"."jvd_h1" t WHERE (' + where + ') LIMIT %s'
    params = tuple(v for r in document.races for v in r.parts) + (max_rows + 1,)
    rows = _query(db, budget, "reconciliation.h1_rows", sql, params, max_rows)
    result = []
    for row in rows:
        race = RaceKey(tuple(row.get(k) for k in RACE_KEY))
        _require(race in document.races, "unexpected_h1_race")
        result.append(LocalH1Record(race, row.get("h1_category"), row.get("h1_win_sale"), row.get("h1_place_sale"),
                                     row.get("h1_paid_positions"), row.get("oversized_fields") is True))
    return tuple(result)
