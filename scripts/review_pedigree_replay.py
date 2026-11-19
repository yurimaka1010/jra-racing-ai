"""Review only committed Oct24/25 replay forecasts; never fit or open formal results."""
from pathlib import Path
import argparse,json,os,sys,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai import pedigree_replay_review as review


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('collect','render','popularity','market-favorite'));p.add_argument('--artifact-id');a=p.parse_args()
    try:
        import lightgbm as lgb
        from unittest.mock import patch
        with patch.object(lgb,'train',side_effect=RuntimeError('zero_fit_scope')),patch.object(lgb.LGBMModel,'fit',side_effect=RuntimeError('zero_fit_scope')):
            if a.command in ('render','popularity','market-favorite'):
                import re
                if not re.fullmatch('[a-f0-9]{32}',a.artifact_id or ''):raise ValueError('invalid_identity')
                if a.command=='render':value=review.render(review.ROOT/'evaluations'/a.artifact_id)
                elif a.command=='popularity':
                    from jra_racing_ai.pedigree_popularity import stage
                    value=stage(review.ROOT/'evaluations'/a.artifact_id)
                else:
                    from jra_racing_ai.pedigree_market_favorite import stage
                    value=stage(review.ROOT/'evaluations'/a.artifact_id)
            else:
                from jra_racing_ai.db_connection import read_only_session
                from jra_racing_ai.win_sim_cli import connection_config
                def session():return read_only_session(connection_config(10000),os.environ.get('PGPASSWORD'),gate_timeout_ms=3000)
                directory,report=review.collect(session);value=review.aggregate_output(directory,report)
        print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0
    except Exception as e:
        code=getattr(e,'code','internal_error')
        try:
            from uuid import uuid4
            root=review.safety.local_path(review.ROOT/'failures',root=review.ROOT);root.mkdir(parents=True,exist_ok=True)
            review.write(root/(uuid4().hex+'.json'),{'reason':code,'stage':a.command,'error_type':type(e).__name__,
                'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
        except Exception:pass
        print(json.dumps({'status':'stopped','reason':code,'stage':a.command}));return 1


if __name__=='__main__':raise SystemExit(main())
