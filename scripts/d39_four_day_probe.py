"""One aggregate for today's declared races; no labels, inference, or saves."""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.db_connection import ConnectionConfig, read_only_session, validate_audit_sql


SQL = ("SELECT count(*) AS entry_races FROM ONLY public.jvd_ra "
       "WHERE kaisai_nen::text=%s AND kaisai_tsukihi::text=%s "
       "AND data_kubun::text IN ('1','2','3','4','5','6') "
       "AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10')")


def probe(db, day, *, today):
    if day != today:
        raise ValueError('current_day_only')
    validate_audit_sql(SQL)
    rows = db.aggregate('four.day_probe', SQL, (str(day.year), day.strftime('%m%d')), max_rows=1)
    if len(rows) != 1 or type(rows[0].get('entry_races')) is not int or rows[0]['entry_races'] < 0:
        raise ValueError('invalid_count')
    count = rows[0]['entry_races']
    return {'status': 'ready' if count else 'no_entries', 'date': str(day),
            'entry_races': count, 'results_read': False, 'new_fits': 0}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    args = parser.parse_args(argv)
    try:
        today = datetime.now(ZoneInfo('Asia/Tokyo')).date()
        day = date.fromisoformat(args.date)
        if day != today:
            raise ValueError('current_day_only')
        config = ConnectionConfig.from_env(statement_timeout_ms=10000)
        with read_only_session(config, os.environ.get('PGPASSWORD', '')) as db:
            result = probe(db, day, today=today)
        print(json.dumps(result, separators=(',', ':')))
        return 0
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'day_probe_failed', 'results_read': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
