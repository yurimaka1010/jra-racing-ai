import argparse,json,re,traceback
from pathlib import Path
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c,store as s


def main(argv=None):
    p=argparse.ArgumentParser(description='Stage 5 finite local development; no final evaluation or operation adoption.')
    p.add_argument('--stage',required=True,choices=('init','baseline','place','fit','report'))
    p.add_argument('--run-id',default='');args=p.parse_args(argv)
    try:
        if args.stage=='init':directory=safety.new_artifact('evaluations',root=c.ROOT);value=s.snapshot(directory)
        else:
            if not re.fullmatch('[a-f0-9]{32}',args.run_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(c.ROOT/'evaluations'/args.run_id,root=c.ROOT)
            if s.read(directory,'plan.json')['version']!=c.VERSION:raise safety.MLFailure('artifact_invalid')
            from . import experiment
            value=getattr(experiment,args.stage)(directory)
        print(json.dumps(dict(value,status='ok',stage=args.stage,run_id=directory.name)),flush=True);return 0
    except SafeAuditError as exc:
        print(json.dumps(dict(status='stopped',stage=args.stage,reason=exc.code)),flush=True);return 2
    except Exception as exc:
        line=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps(dict(status='stopped',stage=args.stage,reason='internal_error',exception_type=type(exc).__name__,
            location={'file':Path(line.filename).name,'line':line.lineno})),flush=True);return 2
