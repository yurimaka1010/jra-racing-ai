"""Explicit local zero-fit preparation. Only fixed aggregate diagnostics on stdout."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.ml import safety
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.pedigree_research import contracts as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('inventory', 'prepare', 'verify', 'report'))
    parser.add_argument('--run-id', default='')
    args = parser.parse_args()
    stage = args.command
    try:
        from jra_racing_ai.pedigree_research import preparation
        if args.run_id:
            import re
            if not re.fullmatch('[a-f0-9]{32}', args.run_id):
                raise safety.MLFailure('invalid_identity')
            directory = safety.local_path(c.ROOT / 'evaluations' / args.run_id, root=c.ROOT)
        else:
            directory = safety.new_artifact('evaluations', root=c.ROOT)
        # Process-local guard; loading and predicting saved trees stays available.
        import lightgbm as lgb
        from unittest.mock import patch
        with patch.object(lgb, 'train', side_effect=safety.MLFailure('explicit_execution_required')), \
             patch.object(lgb.LGBMModel, 'fit', side_effect=safety.MLFailure('explicit_execution_required')):
            if args.command == 'report':
                from jra_racing_ai.pedigree_research.report import finalize
                value = finalize(directory)
            else:
                value = getattr(preparation, args.command)(directory)
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, SafeAuditError) else 'internal_error'
        # No exception text: a pandas/DB error can contain an actual key or value.
        if 'directory' in locals() and directory.exists():
            import traceback
            from uuid import uuid4
            try:
                c.write(directory / ('error_' + uuid4().hex + '.json'),
                        {'error_type': type(exc).__name__, 'reason': code,
                         'frames': [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                                    for f in traceback.extract_tb(exc.__traceback__)]})
            except Exception:
                pass  # Reporting an I/O failure must not expose an original exception.
        print(json.dumps({'status': 'stopped', 'reason': code, 'stage': stage}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
