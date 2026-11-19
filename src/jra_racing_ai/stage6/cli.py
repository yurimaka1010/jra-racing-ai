import argparse,json,re,traceback
from pathlib import Path
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c,store as s,experiment


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage 6: fixed six-fit local development. No database or 2026 entry.')
    parser.add_argument('--stage',required=True,choices=('init','run'));parser.add_argument('--run-id',default='');args=parser.parse_args(argv)
    try:
        if args.stage=='init':directory=safety.new_artifact('evaluations',root=c.ROOT);value=s.init(directory)
        else:
            if not re.fullmatch('[a-f0-9]{32}',args.run_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(c.ROOT/'evaluations'/args.run_id,root=c.ROOT);value=experiment.run(directory)
        print(json.dumps(dict(value,status='ok',stage=args.stage,run_id=directory.name),ensure_ascii=True,allow_nan=False),flush=True);return 0
    except SafeAuditError as exc:
        print(json.dumps({'status':'stopped','stage':args.stage,'reason':exc.code}),flush=True);return 2
    except Exception as exc:
        location=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps({'status':'stopped','stage':args.stage,'reason':'internal_error','exception_type':type(exc).__name__,
            'location':{'file':Path(location.filename).name,'line':location.lineno}}),flush=True);return 2
