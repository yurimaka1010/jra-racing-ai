"""Shared contracts; no database driver is imported here."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

RACE_KEY = (
    "kaisai_nen", "kaisai_tsukihi", "keibajo_code",
    "kaisai_kai", "kaisai_nichime", "race_bango",
)
DOMESTIC_VENUES = tuple(f"{number:02d}" for number in range(1, 11))

ERROR_MESSAGES = {
    "invalid_configuration": "設定値が不正です。READMEの設定条件を確認してください。",
    "unsafe_connection": "接続先または接続環境設定を安全に確認できません。",
    "missing_driver": "Psycopg 3が未導入です。導入の承認と準備が必要です。",
    "connection_failed": "ローカル接続に失敗しました。起動状態・設定・認証をPC上で確認してください。",
    "unsafe_role": "管理者権限・所有権・書き込み権限のあるロールでは監査できません。専用読取ロールを使用してください。",
    "not_read_only": "読み取り専用・一貫したスナップショットの接続を確認できませんでした。",
    "unsafe_sql": "許可された監査SQLとして検証できませんでした。",
    "query_timeout": "監査がタイムアウトしました。対象範囲や既存インデックスをPC上で確認してください。",
    "permission_denied": "参照権限が不足しています。読取ロールの権限をPC上で確認してください。",
    "database_error": "監査クエリを完了できませんでした。詳細エラーは出力しません。",
    "result_limit": "集計結果が上限を超えました。範囲を限定してください。",
    "refund_needs_confirmation": "返還情報が不正または未確認です（要確認）。処理を停止しました。",
    "payout_needs_confirmation": "払戻フィールドの仕様または値が未確認です（要確認）。変換を停止しました。",
    "settlement_invalid_input": "仮想決済の入力を安全に検証できません。処理を停止しました。",
    "reconciliation_needs_confirmation": "ローカル払戻照合の入力・条件を確認できません（要確認）。処理を停止しました。",
    "unsafe_output": "出力先がプロジェクト・同期領域内、リンク経由、または許可外です。",
    "output_failed": "レポート保存に失敗しました。PC上で保存先の権限を確認してください。",
    "internal_error": "処理を完了できませんでした。秘密を保護するため例外本文は表示しません。",
}


class SafeAuditError(Exception):
    def __init__(self, code: str):
        self.code = code if code in ERROR_MESSAGES else "internal_error"
        super().__init__(ERROR_MESSAGES[self.code])


class AggregateDB(Protocol):
    def aggregate(self, query_id: str, sql: str, params: tuple = (),
                  max_rows: int = 1000) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class Scope:
    year: int
    venues: tuple[str, ...] = DOMESTIC_VENUES
    global_summary: bool = False

    def __post_init__(self):
        if (type(self.year) is not int or not 1800 <= self.year <= 2200
                or not self.venues or len(set(self.venues)) != len(self.venues)
                or any(v not in DOMESTIC_VENUES for v in self.venues)):
            raise SafeAuditError("invalid_configuration")


@dataclass
class AuditContext:
    db: AggregateDB
    scope: Scope
    schema: dict[str, dict[str, Any]]
    status_columns: dict[str, str] = field(default_factory=dict)
    horse_identity_columns: tuple[str, ...] = ()


def quote_ident(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise SafeAuditError("invalid_configuration")
    return '"' + name.replace('"', '""') + '"'


def qualified(name: str) -> str:
    return '"public".' + quote_ident(name)


def column_names(ctx: AuditContext, name: str) -> set[str]:
    return {column["name"] for column in ctx.schema.get(name, {}).get("columns", [])}


def status_column(ctx: AuditContext, table: str) -> str | None:
    column = ctx.status_columns.get(table)
    if column is not None and column not in column_names(ctx, table):
        raise SafeAuditError("invalid_configuration")
    return column


def usable_columns(ctx: AuditContext, table: str, names) -> bool:
    """Avoid invoking custom type output functions in data aggregation."""
    metadata = {c["name"]: c for c in ctx.schema.get(table, {}).get("columns", [])}
    permitted = re.compile(r"(?:text|character(?: varying)?(?:\(\d+\))?|varchar(?:\(\d+\))?|bpchar|"
                           r"smallint|integer|bigint|int[248]|numeric(?:\(\d+(?:,\s*\d+)?\))?)")
    return all(name in metadata and metadata[name].get("type_is_builtin", False)
               and bool(permitted.fullmatch(metadata[name].get("data_type", ""))) for name in names)


def scope_where(scope: Scope, alias: str = "t") -> tuple[str, tuple]:
    prefix = quote_ident(alias)
    # No silent zero padding: text/integer representation differences need review.
    sql = (f"btrim({prefix}.{quote_ident('kaisai_nen')}::text) = %s AND "
           f"btrim({prefix}.{quote_ident('keibajo_code')}::text) = ANY(%s)")
    return sql, (str(scope.year), list(scope.venues))
