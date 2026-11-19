"""Development-only commands; no path here accepts a 2026 evaluation period."""
import argparse
import json
from pathlib import Path
import re
import traceback
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c,io


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('init','collect','build','market','ability-ab','anchor-ab','ability-cd','anchor-full','annual','freeze'),required=True)
    p.add_argument('--work-id',default='')
    args=p.parse_args(argv)
    try:
        if args.stage=='init':
            directory=safety.new_artifact('evaluations',root=c.ROOT)
        else:
            if not re.fullmatch('[a-f0-9]{32}',args.work_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(c.ROOT/'evaluations'/args.work_id,root=c.ROOT)
        if args.stage in ('init','collect'):
            result=getattr(io,args.stage)(directory)
        else:
            from . import experiment
            result=getattr(experiment,args.stage.replace('-','_'))(directory)
        print(json.dumps(dict(status='ok',stage=args.stage,work_id=directory.name,**result),ensure_ascii=True),flush=True)
        return 0
    except SafeAuditError as exc:
        print(json.dumps(dict(status='stopped',stage=args.stage,reason=exc.code)),flush=True)
        return 2
    except Exception as exc:
        f=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps(dict(status='stopped',stage=args.stage,reason='internal_error',
            exception_type=type(exc).__name__,location={'file':Path(f.filename).name,'line':f.lineno})),flush=True)
        return 2
