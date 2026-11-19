"""Explicit local read-only connections. No connect happens at import time."""
from __future__ import annotations

import importlib
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

from .core import SafeAuditError


@dataclass(frozen=True)
class ConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 5432
    database: str = "pckeiba"
    user: str = "jra_audit_reader"
    connect_timeout: int = 5
    statement_timeout_ms: int = 60000
    lock_timeout_ms: int = 2000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None,
                 statement_timeout_ms: int = 60000) -> "ConnectionConfig":
        env = os.environ if env is None else env
        # Prevent inherited service / option configuration from changing endpoints.
        if any(env.get(key) for key in ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR", "PGOPTIONS")):
            raise SafeAuditError("unsafe_connection")
        try:
            config = cls(host=env.get("PGHOST", "127.0.0.1"),
                         port=int(env.get("PGPORT", "5432")),
                         database=env.get("PGDATABASE", "pckeiba"),
                         user=env.get("PGUSER", "jra_audit_reader"),
                         statement_timeout_ms=statement_timeout_ms)
        except (ValueError, TypeError):
            raise SafeAuditError("invalid_configuration") from None
        config.validate()
        return config

    def validate(self):
        identifier = r"[A-Za-z_][A-Za-z0-9_-]{0,62}"
        if self.host not in ("127.0.0.1", "::1"):
            raise SafeAuditError("unsafe_connection")
        if (not 1 <= self.port <= 65535 or not re.fullmatch(identifier, self.database)
                or not re.fullmatch(identifier, self.user)
                or not 1000 <= self.statement_timeout_ms <= 600000):
            raise SafeAuditError("invalid_configuration")
        if self.user.lower() == "postgres":
            raise SafeAuditError("unsafe_role")


_FORBIDDEN = re.compile(
    r"\b(?:insert|update|delete|drop|truncate|alter|create|grant|revoke|copy|call|do|"
    r"vacuum|analyze|refresh|merge|execute|prepare|set|reset|commit|rollback|"
    r"pg_sleep|dblink|lo_export|lo_import|pg_read_file|pg_read_binary_file|"
    r"pg_write_file|pg_terminate_backend|pg_cancel_backend|pg_advisory_lock|"
    r"set_config|nextval|setval)\b", re.IGNORECASE,
)


def validate_audit_sql(sql: str):
    """Defense in depth for fixed project SQL; not an arbitrary-SQL sandbox."""
    # Ignore identifiers and literal contents, keeping SQL tokens outside them.
    tokens = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", sql)
    if (";" in tokens or "--" in tokens or "/*" in tokens or "$" in tokens
            or not re.match(r"^\s*(SELECT|WITH)\b", tokens, re.IGNORECASE)
            or _FORBIDDEN.search(tokens)):
        raise SafeAuditError("unsafe_sql")


def _safe_query_error(exc, query_id):
    """Only standard SQLSTATE and fixed ML operation IDs may leave this layer."""
    state = getattr(exc, "sqlstate", None)
    result = SafeAuditError({"57014": "query_timeout", "42501": "permission_denied"}.get(
        state, "database_error"))
    fixed_ids = {"connection.safety", "ml.table.ra", "ml.columns.ra", "ml.table.se",
                 "ml.columns.se", "ml.years", "ml.export"}
    result.diagnostics = {}
    if query_id in fixed_ids:
        result.diagnostics["query_id"] = query_id
        if isinstance(state, str) and re.fullmatch(r"[0-9A-Z]{5}", state):
            result.diagnostics["sqlstate"] = state
    return result


class QueryRunner:
    def __init__(self, connection):
        self._connection = connection

    def aggregate(self, query_id: str, sql: str, params: tuple = (),
                  max_rows: int = 1000) -> list[dict]:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", query_id) or not 1 <= max_rows <= 12000:
            raise SafeAuditError("invalid_configuration")
        validate_audit_sql(sql)
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                raise SafeAuditError("result_limit")
            return [dict(row) for row in rows]
        except SafeAuditError:
            raise
        except Exception as exc:
            raise _safe_query_error(exc, query_id) from None

    def iter_batches(self, query_id: str, sql: str, params: tuple = (), *,
                     max_rows: int = 100000, batch_size: int = 1000,
                     max_seconds: int = 180):
        """Bounded server cursor for explicitly requested local ML exports."""
        if (not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", query_id)
                or not 1 <= max_rows <= 500000 or not 1 <= batch_size <= 2000
                or not 1 <= max_seconds <= 600):
            raise SafeAuditError("invalid_configuration")
        validate_audit_sql(sql)
        deadline, count = time.monotonic() + max_seconds, 0
        try:
            with self._connection.cursor(name="jra_ml_export") as cursor:
                cursor.execute(sql, params)
                while True:
                    if time.monotonic() >= deadline:
                        raise SafeAuditError("query_timeout")
                    rows = cursor.fetchmany(min(batch_size, max_rows + 1 - count))
                    if time.monotonic() >= deadline:
                        raise SafeAuditError("query_timeout")
                    if not rows:
                        break
                    count += len(rows)
                    if count > max_rows:
                        raise SafeAuditError("result_limit")
                    yield [dict(row) for row in rows]
        except SafeAuditError:
            raise
        except Exception as exc:
            raise _safe_query_error(exc, query_id) from None


_ROLE_CHECK = """
SELECT current_setting('transaction_read_only') AS read_only,
       current_setting('transaction_isolation') AS isolation,
       EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = current_user
               AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls)) AS privileged,
       EXISTS (SELECT 1 FROM pg_catalog.pg_class c
               JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'f', 'v', 'm')
                 AND (pg_catalog.pg_has_role(c.relowner, 'USAGE')
                      OR pg_catalog.has_table_privilege(c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'))) AS writable,
       EXISTS (SELECT 1 FROM pg_catalog.pg_namespace n WHERE n.nspname = 'public'
               AND (pg_catalog.pg_has_role(n.nspowner, 'USAGE')
                    OR pg_catalog.has_schema_privilege(n.oid, 'CREATE'))) AS schema_writable
"""


@contextmanager
def read_only_session(config: ConnectionConfig, password: str, *, driver=None, gate_timeout_ms=20000):
    """Serialize cooperating PC readers; retain the existing RO transaction contract."""
    from .reader_gate import reader_gate
    config.validate()
    with reader_gate(config, gate_timeout_ms):
        with _read_only_session(config, password, driver=driver) as runner:
            yield runner


@contextmanager
def _read_only_session(config: ConnectionConfig, password: str, *, driver=None):
    """Only the CLI's explicit local command invokes this function."""
    config.validate()
    if any(os.environ.get(key) for key in ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR", "PGOPTIONS")):
        raise SafeAuditError("unsafe_connection")
    if not password:
        raise SafeAuditError("invalid_configuration")
    if driver is None:
        try:
            driver = importlib.import_module("psycopg")
        except ImportError:
            raise SafeAuditError("missing_driver") from None
    connection = None
    try:
        # Explicit hostaddr/dbname/password/passfile suppress libpq fallback routes.
        connection = driver.connect(
            host=config.host, hostaddr=config.host, port=config.port,
            dbname=config.database, user=config.user, password=password,
            passfile=os.devnull, sslmode="disable", gssencmode="disable",
            require_auth="scram-sha-256,md5,password",
            target_session_attrs="any", connect_timeout=config.connect_timeout,
            application_name="jra_phase1_audit", autocommit=False,
            row_factory=driver.rows.dict_row,
            options=("-c default_transaction_read_only=on -c search_path=pg_catalog "
                     f"-c statement_timeout={config.statement_timeout_ms} "
                     f"-c lock_timeout={config.lock_timeout_ms} "
                     "-c idle_in_transaction_session_timeout=60000"),
        )
        connection.isolation_level = driver.IsolationLevel.REPEATABLE_READ
        connection.read_only = True
        runner = QueryRunner(connection)
        checks = runner.aggregate("connection.safety", _ROLE_CHECK, max_rows=1)
        if not checks or checks[0].get("read_only") != "on" or checks[0].get("isolation") != "repeatable read":
            raise SafeAuditError("not_read_only")
        if any(checks[0].get(key, True) for key in ("privileged", "writable", "schema_writable")):
            raise SafeAuditError("unsafe_role")
        yield runner
    except SafeAuditError:
        raise
    except Exception:
        raise SafeAuditError("connection_failed" if connection is None else "database_error") from None
    finally:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
