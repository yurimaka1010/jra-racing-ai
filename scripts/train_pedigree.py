"""Approved two-fit study only; no DB credentials, production registration or retries."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai.ml import safety
from jra_racing_ai.core import SafeAuditError
from jra_racing_ai.pedigree_research import contracts as c, learning


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','train','_worker','report'))
    parser.add_argument('--run-id')
    args=parser.parse_args()
    try:
        if args.command=='prepare':value=learning.prepare()
        else:
            directory=learning.folder(args.run_id)
            value=(learning.worker(directory) if args.command=='_worker' else learning.report(directory)
                   if args.command=='report' else learning.execute(directory))
        print(json.dumps(value,ensure_ascii=False,allow_nan=False))
        return 0 if value.get('status')!='stopped' else 1
    except Exception as exc:
        code=exc.code if isinstance(exc,SafeAuditError) else 'internal_error'
        if 'directory' in locals():
            import traceback
            from uuid import uuid4
            try:
                c.write(directory/('failure_'+uuid4().hex+'.json'),{'reason':code,'error_type':type(exc).__name__,
                    'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
                              for f in traceback.extract_tb(exc.__traceback__)]})
            except Exception:pass
        print(json.dumps({'status':'stopped','reason':code,'stage':args.command}))
        return 1


if __name__=='__main__':raise SystemExit(main())
