"""Zero-fit research commands. Only fixed aggregate diagnostics reach stdout."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai.pedigree_forward import contracts as c
from jra_racing_ai.core import SafeAuditError


def execute(a):
    if a.command=='prepare':return c.prepare(a.supersedes_run_id)
    if a.command=='measure':
        from jra_racing_ai.pedigree_forward.measurement import measure
        return measure(c.folder(a.run_id))
    if a.command=='freeze':
        from jra_racing_ai.pedigree_forward.protocol import freeze
        return freeze(c.folder(a.run_id))
    if a.command=='replay':
        from jra_racing_ai.pedigree_forward.replay import run
        return run(c.folder(a.run_id))
    if a.command in ('preflight','record'):
        from jra_racing_ai.pedigree_forward import recorder
        return recorder.preflight(c.folder(a.run_id)) if a.command=='preflight' else recorder.run(c.folder(a.run_id),once=a.once)
    from collections import Counter
    count=Counter();directory=c.folder(a.run_id)
    for path in (directory/'forward').glob('*/races/*/receipt.json'):
        value=c.read(path);count[value['status']]+=1
        if value.get('status')=='saved' and not value.get('odds_ready'):count['odds_unavailable']+=1
        if value.get('status')=='saved' and not value.get('all_models_complete'):count['models_incomplete']+=1
    return {'status':'ok','operational_counts':dict(count),'scores_computed':False,'results_read':False,'new_fits':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','measure','freeze','replay','preflight','record','ops'))
    p.add_argument('--run-id');p.add_argument('--supersedes-run-id');p.add_argument('--once',action='store_true');a=p.parse_args()
    try:
        # Guard this process only. Prediction from saved trees remains available.
        import lightgbm as lgb
        from unittest.mock import patch
        with patch.object(lgb,'train',side_effect=RuntimeError('zero_fit_scope')), \
             patch.object(lgb.LGBMModel,'fit',side_effect=RuntimeError('zero_fit_scope')):
            v=execute(a)
        print(json.dumps(v,ensure_ascii=False,allow_nan=False));return 0
    except Exception as e:
        code=e.code if isinstance(e,SafeAuditError) else 'internal_error'
        if a.run_id:
            try:
                from uuid import uuid4
                import traceback
                directory=c.folder(a.run_id);log=c.mkdir(directory/'operational-logs')
                c.write(log/('failure_'+uuid4().hex+'.json'),{'reason':code,'stage':a.command,'error_type':type(e).__name__,
                    'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
            except Exception:pass
        print(json.dumps({'status':'stopped','reason':code,'stage':a.command}));return 1


if __name__=='__main__':raise SystemExit(main())
