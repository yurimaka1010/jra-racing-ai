"""User-run, one-day check for the empty Version 0.2 export. Counts only."""
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.db_connection import ConnectionConfig, read_only_session
from jra_racing_ai.ml import data, safety
from jra_racing_ai.ml_cli import Parser, _password
from jra_racing_ai.ml.safety import MLFailure


def check(race_date):
    day = date.fromisoformat(race_date)
    mapping_path = safety.ML_ROOT / "schema" / "258d31477be9478d83e6dc9a624bc00b" / "mapping.reviewed.json"
    mapping = data.validate_mapping(safety.read_json(mapping_path))
    keys = ", ".join(data.identifier(name) for name in RACE_KEY)
    scopes, params = [], []
    for alias, table, field in (("ra", "jvd_ra", "race_status"), ("se", "jvd_se", "horse_status")):
        scopes.append(alias + " AS (SELECT " + keys + ", " + data.identifier(mapping["columns"][field])
                      + "::text AS record_status FROM ONLY public." + data.identifier(table)
                      + " WHERE kaisai_nen::text = %s AND kaisai_tsukihi::text = %s"
                      + " AND keibajo_code::text IN (" + ", ".join(["%s"] * 10) + "))")
        params.extend((str(day.year), day.strftime("%m%d"), *(f"{n:02d}" for n in range(1, 11))))
    sql = ("WITH " + ", ".join(scopes) + " SELECT "
           "(SELECT count(*) FROM ra) AS ra_rows, "
           "(SELECT count(*) FROM ra WHERE record_status = '7') AS ra_final_rows, "
           "(SELECT count(*) FROM se) AS se_rows, "
           "(SELECT count(*) FROM se WHERE record_status = '7') AS se_final_rows, "
           "(SELECT count(*) FROM ra JOIN se USING (" + keys + ") "
           " WHERE ra.record_status = '7' AND se.record_status = '7') AS matched_final_runner_rows")
    secret = _password()
    try:
        with read_only_session(ConnectionConfig(statement_timeout_ms=10000), secret) as db:
            rows = db.aggregate("ml.export", sql, tuple(params), max_rows=1)
    finally:
        secret = None
    fields = ("ra_rows", "ra_final_rows", "se_rows", "se_final_rows", "matched_final_runner_rows")
    if (len(rows) != 1 or any(type(rows[0].get(name)) is not int or rows[0][name] < 0 for name in fields)):
        raise MLFailure("invalid_schema")
    counts = {name: rows[0][name] for name in fields}
    if not counts["ra_rows"] and not counts["se_rows"]:
        reason = "ra_se_missing"
    elif not counts["ra_rows"]:
        reason = "ra_missing"
    elif not counts["se_rows"]:
        reason = "se_missing"
    elif not counts["ra_final_rows"] or not counts["se_final_rows"]:
        reason = "final_category_missing"
    elif not counts["matched_final_runner_rows"]:
        reason = "race_keys_unmatched"
    else:
        reason = "source_ready"
    return {"status": "ok", "date": day.isoformat(), "reason": reason, "counts": counts}


def main(argv=None):
    try:
        parser = Parser(description="Check one export date using counts only")
        parser.add_argument("--race-date", default="2026-10-04")
        result = check(parser.parse_args(argv).race_date)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (SafeAuditError, MLFailure) as exc:
        result = {"status": "stopped", "reason": exc.code}
        if getattr(exc, "diagnostics", None):
            result["diagnostics"] = exc.diagnostics
        print(json.dumps(result, ensure_ascii=True))
        return 2
    except KeyboardInterrupt:
        print('{"status":"stopped","reason":"interrupted"}')
        return 130
    except Exception:
        print('{"status":"stopped","reason":"source_check_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
