import argparse
import json
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c


def main(argv=None):
    parser = argparse.ArgumentParser(description='Clean-race D39: independent preparation and approved finite learning')
    parser.add_argument('mode', choices=('prepare', 'train', '_worker', 'record', 'evaluate', 'preview'))
    parser.add_argument('--run-id')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'prepare':
            from .data import prepare
            value = prepare()
        elif args.mode in ('train', '_worker'):
            from . import learning
            value = (learning.execute if args.mode == 'train' else learning.worker)(c.folder(args.run_id))
        else:
            from . import forward
            if args.mode == 'preview':
                value = forward.preview(c.folder(args.run_id))
            else:
                value = (forward.run if args.mode == 'record' else forward.evaluate)(c.folder(args.run_id), once=args.once)
        print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)
        return 0 if value.get('status') not in ('stopped',) else 1
    except Exception as exc:
        reason = exc.code if isinstance(exc, SafeAuditError) else 'internal_error'
        print(json.dumps({'status': 'stopped', 'reason': reason, 'stage': args.mode}), flush=True)
        return 1
