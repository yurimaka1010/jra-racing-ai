import argparse
import json
from ..core import SafeAuditError
from ..ml import safety
from . import contracts as c


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('inventory','prepare','train','_worker','preview','record','observe'))
    parser.add_argument('--run-id');parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.mode=='inventory':
            from .sources import inventory
            directory=safety.new_artifact('evaluations',root=c.ROOT)
            c.write(directory/'plan.json',c.plan())
            c.write(directory/'preservation_before.json',c.prior.preservation())
            value=inventory(directory)
        elif args.mode=='prepare':
            from .data import prepare
            value=prepare(c.folder(args.run_id))
        elif args.mode in ('train','_worker'):
            from . import learning
            value=(learning.execute if args.mode=='train' else learning.worker)(c.folder(args.run_id))
        elif args.mode=='observe':
            from .sources import observe
            value=observe(c.folder(args.run_id),once=args.once)
        else:
            from . import forward
            value=(forward.preview if args.mode=='preview' else forward.run)(c.folder(args.run_id),once=args.once)
        print(json.dumps(value,ensure_ascii=True,allow_nan=False),flush=True)
        return 0 if value.get('status')!='stopped' else 1
    except Exception as exc:
        print(json.dumps({'status':'stopped','reason':exc.code if isinstance(exc,SafeAuditError) else 'internal_error',
                          'stage':args.mode,'error_type':type(exc).__name__}),flush=True)
        return 1
