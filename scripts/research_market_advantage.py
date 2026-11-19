"""Run only the predeclared, local-file-only, zero-fit market exploration."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai import market_advantage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    try:
        value = (market_advantage.publish if args.publish else market_advantage.run)(args.run_id)
    except Exception as exc:
        # Do not put paths, record data or exception detail into the console.
        reason = str(exc) if isinstance(exc, ValueError) and str(exc).replace('_', '').isalnum() else 'local_analysis_error'
        print(json.dumps(dict(status='stopped', reason=reason), ensure_ascii=False))
        return 1
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
