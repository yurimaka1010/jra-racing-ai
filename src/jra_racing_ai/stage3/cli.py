import argparse,json,re,traceback
from pathlib import Path
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c,store


def main(argv=None):
    p=argparse.ArgumentParser(description='Bounded Stage 3 development; sealed 2026 is unavailable.')
    p.add_argument('--stage',choices=('init','reuse','oof','integrate','e','freeze'),required=True)
    p.add_argument('--run-id',default='');args=p.parse_args(argv)
    try:
        if args.stage=='init':directory=safety.new_artifact('evaluations',root=c.ROOT)
        else:
            if not re.fullmatch('[a-f0-9]{32}',args.run_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(c.ROOT/'evaluations'/args.run_id,root=c.ROOT)
            if not directory.is_dir():raise safety.MLFailure('artifact_invalid')
        if args.stage=='init':value=store.init(directory)
        else:
            module=__import__('jra_racing_ai.stage3.'+{'e':'supplement','freeze':'report'}.get(args.stage,args.stage),fromlist=['run'])
            value=module.run(directory)
        payload=dict(value)
        if 'status' in payload:payload['unit_status']=payload.pop('status')
        payload.update(status='ok',stage=args.stage,run_id=directory.name)
        print(json.dumps(payload),flush=True)
        return 0
    except SafeAuditError as exc:
        print(json.dumps({'status':'stopped','stage':args.stage,'reason':exc.code}),flush=True);return 2
    except Exception as exc:
        f=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps({'status':'stopped','stage':args.stage,'reason':'internal_error','exception_type':type(exc).__name__,
            'location':{'file':Path(f.filename).name,'line':f.lineno}}),flush=True);return 2
