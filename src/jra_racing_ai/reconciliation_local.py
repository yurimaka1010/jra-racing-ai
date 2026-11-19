"""Explicit local I/O for reconciliation; no I/O or connection at import time."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .core import RACE_KEY, quote_ident
from .db_connection import validate_audit_sql
from .generate_report import (
    PROJECT_ROOT, REAL_REPORT_DIR, _has_link, _local_disk, validate_output_dir,
)
from .inspect_schema import COLUMNS_SQL
from .payout_decoder import RawPayoutSlot
from .payout_reconciliation import (
    LocalHRRecord, RaceKey, ReconciliationError, ReconciliationLimits, ReconciliationSummary,
    SLOT_COUNTS, WAGERS, parse_reference_document,
    _identifier,
)

MAX_REFERENCE_BYTES = 262144
INPUT_SUBDIR = "reconciliation-input"
_UNSAFE_ATTRIBUTES = 0x400 | 0x1000 | 0x40000 | 0x400000  # reparse / offline / recall


def _cloud_or_reparse(path):
    for part in reversed((path, *path.parents)):
        try:
            if getattr(part.lstat(), "st_file_attributes", 0) & _UNSAFE_ATTRIBUTES:
                return True
        except FileNotFoundError:
            continue
    return False


def validate_local_input(path: Path, *, suffix: str, approved_root: Path = REAL_REPORT_DIR,
                         project_root: Path = PROJECT_ROOT, env=None) -> Path:
    """Only regular direct children of the approved local input directory."""
    root = validate_output_dir(approved_root, real=True, approved_root=approved_root,
                               project_root=project_root, env=env)
    try:
        if (not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
                or not _local_disk(path) or path.suffix.lower() != suffix):
            raise ReconciliationError("unsafe_input")
        # Lexical containment is checked before touching an untrusted leaf.
        if path.parent != root / INPUT_SUBDIR or _has_link(path) or _cloud_or_reparse(path):
            raise ReconciliationError("unsafe_input")
        resolved = path.resolve()
        if resolved.parent != root / INPUT_SUBDIR or not resolved.is_file():
            raise ReconciliationError("unsafe_input")
        # Hard links can point at a synchronized file while looking local.
        if resolved.stat().st_nlink != 1:
            raise ReconciliationError("unsafe_input")
        return resolved
    except ReconciliationError:
        raise
    except Exception:
        raise ReconciliationError("unsafe_input") from None


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ReconciliationError("duplicate_reference_key")
        value[key] = item
    return value


def _reject_constant(_):
    raise ReconciliationError("invalid_json")


def _read_json_file(path: Path, decoder, *, approved_root=REAL_REPORT_DIR, project_root=PROJECT_ROOT, env=None):
    path = validate_local_input(path, suffix=".json", approved_root=approved_root, project_root=project_root, env=env)
    try:
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ReconciliationError("input_size_limit")
        with path.open("rb") as stream:
            data = stream.read(MAX_REFERENCE_BYTES + 1)
        if len(data) > MAX_REFERENCE_BYTES:
            raise ReconciliationError("input_size_limit")
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return decoder(value)
    except ReconciliationError:
        raise
    except Exception:
        # JSON/Unicode/file exceptions may embed input, paths or entire lines.
        raise ReconciliationError("invalid_json") from None


def read_reference_file(path: Path, *, approved_root=REAL_REPORT_DIR, project_root=PROJECT_ROOT, env=None):
    return _read_json_file(path, parse_reference_document, approved_root=approved_root, project_root=project_root, env=env)


def check_local_spec_pdf(path: Path, *, approved_root=REAL_REPORT_DIR, project_root=PROJECT_ROOT, env=None):
    """Presence/header check only. The user reads the PDF manually, offline."""
    path = validate_local_input(path, suffix=".pdf", approved_root=approved_root, project_root=project_root, env=env)
    try:
        if not 5 <= path.stat().st_size <= 64*1024*1024:
            raise ReconciliationError("invalid_pdf")
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ReconciliationError("invalid_pdf")
    except ReconciliationError:
        raise
    except Exception:
        raise ReconciliationError("invalid_pdf") from None
    return True  # Never means the contents/edition/semantics were verified.


class ExecutionBudget:
    def __init__(self, limits: ReconciliationLimits, *, clock=time.monotonic):
        self._clock = clock
        self._deadline = clock() + limits.max_run_seconds
        self._query_seconds = limits.statement_timeout_ms / 1000

    def check(self):
        if self._clock() >= self._deadline:
            raise ReconciliationError("time_limit")

    def before_query(self):
        if self._deadline - self._clock() <= self._query_seconds:
            raise ReconciliationError("time_limit")


_HR_TABLE_SQL = """
SELECT c.relkind AS kind, am.amname AS method,
       c.relrowsecurity AS row_security, c.relforcerowsecurity AS force_row_security
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_am am ON am.oid = c.relam
WHERE n.nspname = %s AND c.relname = %s
"""


def _query(db, budget, query_id, sql, params, max_rows):
    budget.before_query()
    validate_audit_sql(sql)
    rows = db.aggregate(query_id, sql, params, max_rows=max_rows)
    budget.check()
    if len(rows) > max_rows:
        raise ReconciliationError("row_limit")
    return rows


def fetch_local_hr(db, document, limits: ReconciliationLimits, budget: ExecutionBudget, *, flag_columns=None) -> tuple[LocalHRRecord, ...]:
    """Only user invocation reaches here. Four SQL calls including role check.

    Data fields are not cast to text or trimmed (bpchar casts can lose spaces).
    Oversized values are rejected, not fetched/truncated into plausible values.
    """
    limits.check_document(document)
    if flag_columns is None:
        flag_columns = tuple(r.column for r in document.flag_rules)
    else:
        if (type(flag_columns) is not tuple or len(flag_columns) != 4 or not all(_identifier(c) for c in flag_columns)
                or len(set(flag_columns)) != 4
                or document.status_column in flag_columns
                or any(c in RACE_KEY or c.startswith(("haraimodoshi_", "henkan_")) for c in flag_columns)):
            raise ReconciliationError("invalid_h1_config")
    table = _query(db, budget, "reconciliation.hr_table", _HR_TABLE_SQL, ("public", "jvd_hr"), 1)
    if (len(table) != 1 or table[0].get("kind") != "r" or table[0].get("method") != "heap"
            or table[0].get("row_security") is not False or table[0].get("force_row_security") is not False):
        raise ReconciliationError("hr_table_unavailable")
    columns = _query(db, budget, "reconciliation.hr_columns", COLUMNS_SQL, ("public", "jvd_hr"), 1600)
    metadata = {c["name"]: c for c in columns}
    fields = list(zip(RACE_KEY, (4, 4, 2, 2, 2, 2)))
    fields += [(document.status_column, 1), ("henkan_umaban_joho", 28),
               ("henkan_flag_tansho", 1), ("henkan_flag_fukusho", 1)]
    for wager in WAGERS:
        fields.extend((f"haraimodoshi_{wager}_{n}{part}", width)
                      for n in range(1, SLOT_COUNTS[wager] + 1) for part, width in zip("abc", (2, 9, 2)))
    fields.extend((name, 4) for name in dict.fromkeys(flag_columns))
    for name, _ in fields:
        col = metadata.get(name, {})
        if col.get("type_is_builtin") is not True or col.get("type_name") not in ("text", "varchar", "bpchar"):
            raise ReconciliationError("hr_column_mapping_unconfirmed")
    # Alias physical columns into a small, fixed internal row contract.
    aliases = {name: name for name in RACE_KEY}
    aliases.update({document.status_column: "hr_category", "henkan_umaban_joho": "refund_bitmap",
                    "henkan_flag_tansho": "refund_tansho", "henkan_flag_fukusho": "refund_fukusho"})
    for i, column in enumerate(flag_columns):
        aliases.setdefault(column, f"flag_{i}")
    selects, too_long = [], []
    for name, width in fields:
        col = "t." + quote_ident(name)
        too_long.append(f"pg_catalog.octet_length({col}) > {width}")
        selects.append(f"CASE WHEN pg_catalog.octet_length({col}) <= {width} THEN {col} ELSE NULL END AS "
                       + quote_ident(aliases.get(name, name)))
    selects.append("(" + " OR ".join(too_long) + ") IS TRUE AS oversized_fields")
    where = " OR ".join("(" + " AND ".join(f"t.{quote_ident(k)} = %s" for k in RACE_KEY) + ")"
                        for _ in document.races)
    sql = 'SELECT ' + ", ".join(selects) + ' FROM ONLY "public"."jvd_hr" t WHERE (' + where + ') LIMIT %s'
    params = tuple(p for race in document.races for p in race.parts) + (limits.max_hr_rows + 1,)
    rows = _query(db, budget, "reconciliation.hr_rows", sql, params, limits.max_hr_rows)
    records = []
    for row in rows:
        try:
            race = RaceKey(tuple(row.get(k) for k in RACE_KEY))
            slots = {w: tuple(RawPayoutSlot(*(row.get(f"haraimodoshi_{w}_{n}{f}") for f in "abc"))
                              for n in range(1, SLOT_COUNTS[w]+1)) for w in WAGERS}
            record = LocalHRRecord(race, row.get("hr_category"), slots["tansho"], slots["fukusho"],
                row.get("refund_bitmap"), row.get("refund_tansho"), row.get("refund_fukusho"),
                tuple(row.get(aliases[c]) for c in flag_columns), row.get("oversized_fields") is True)
            if record.race not in document.races:
                raise ReconciliationError("unexpected_hr_race")
            records.append(record)
        except ReconciliationError:
            raise
        except Exception:
            raise ReconciliationError("invalid_record") from None
    return tuple(records)


def write_reconciliation_summary(summary, output: Path, *, local_spec_pdf_supplied=False):
    """Only aggregates reach disk; filenames and output location are fixed."""
    if type(summary) is not ReconciliationSummary or type(local_spec_pdf_supplied) is not bool:
        raise ReconciliationError("invalid_reference")
    summary.__post_init__()
    for evidence in summary.wagers:
        evidence.__post_init__()
        for observation in evidence.observations:
            observation.__post_init__()
    root = validate_output_dir(output, real=True)
    payload = summary.to_safe_dict() | {"local_spec_pdf_supplied": local_spec_pdf_supplied,
                                       "pdf_content_verified_by_program": False}
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    lines = ["# ローカルHR払戻照合（集計のみ）", "", "金額単位は未確定。本番決済は無効です。", "",
             "一致率の分母は一致＋不一致。照合不能と保留ケースは除外せず別掲します。", "",
             "| 券種 | 金額仮説 | 一致 | 不一致 | 照合不能 |", "|---|---|---:|---:|---:|"]
    for wager in payload["wagers"]:
        for observation in wager["hypotheses"]:
            lines.append("| " + " | ".join(str(v) for v in (
                wager["wager_type"], observation["hypothesis"], observation["matched"],
                observation["mismatched"], observation["uncomparable"])) + " |")
    lines.extend(["", "## 形式・理由・条件の集計", "", "```json", json_text.rstrip(), "```", ""])
    from .core import SafeAuditError
    try:
        if _cloud_or_reparse(root):
            raise ReconciliationError("unsafe_input")
        root.mkdir(parents=True, exist_ok=True)
        validate_output_dir(root, real=True)
        if _cloud_or_reparse(root):
            raise ReconciliationError("unsafe_input")
        run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_reconciliation")
        run.mkdir(exist_ok=False)
        if _has_link(run) or _cloud_or_reparse(run):
            raise ReconciliationError("unsafe_input")
        for name, text in (("reconciliation.json", json_text), ("reconciliation.md", "\n".join(lines))):
            validate_output_dir(root, real=True)
            with (run / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
    except SafeAuditError:
        raise
    except Exception:
        raise SafeAuditError("output_failed") from None
    return payload
