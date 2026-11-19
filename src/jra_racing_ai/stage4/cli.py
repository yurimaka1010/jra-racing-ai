"""Explicit finite development steps. No final-evaluation or production entry."""
import argparse,json,re,traceback
from pathlib import Path
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c,store


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage 4 bounded local development, through 2025-09-30 only.')
    parser.add_argument('--stage',required=True,choices=('init','reproduce','oof','direct','quotes60','h60','accuracy','finance','freeze'))
    parser.add_argument('--run-id',default='');args=parser.parse_args(argv)
    try:
        if args.stage=='init':directory=safety.new_artifact('evaluations',root=c.ROOT)
        else:
            if not re.fullmatch('[a-f0-9]{32}',args.run_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(c.ROOT/'evaluations'/args.run_id,root=c.ROOT)
            if not directory.is_dir():raise safety.MLFailure('artifact_invalid')
            if store.read(directory,'plan.json')['version']!=c.VERSION:raise safety.MLFailure('artifact_invalid')
        if args.stage=='init':value=store.init(directory)
        elif args.stage=='reproduce':
            from .data import reproduce
            value=reproduce(directory)
        else:
            module=__import__('jra_racing_ai.stage4.'+{'quotes60':'quotes','h60':'history','freeze':'report'}.get(args.stage,args.stage),fromlist=['run'])
            value=module.run(directory)
        payload=dict(value)
        if 'status' in payload:payload['unit_status']=payload.pop('status')
        payload.update(status='ok',stage=args.stage,run_id=directory.name)
        print(json.dumps(payload),flush=True);return 0
    except SafeAuditError as exc:
        print(json.dumps(dict(status='stopped',stage=args.stage,reason=exc.code)),flush=True);return 2
    except Exception as exc:
        location=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps(dict(status='stopped',stage=args.stage,reason='internal_error',exception_type=type(exc).__name__,
            location=dict(file=Path(location.filename).name,line=location.lineno))),flush=True);return 2
