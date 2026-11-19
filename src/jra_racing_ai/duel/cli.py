import argparse,json,os,re,traceback
from pathlib import Path
from ..core import SafeAuditError
from ..ml import safety
from ..stage4 import store as old_store
from . import replay as r,scoring,render


def build(directory,parent=None):
    output=directory/'予想対決.html'
    if output.exists():
        r.verify_commit(directory);old=r.read(directory/'dashboard_manifest.json')
        if r.sha(output)!=old['HTML_sha256'] or r.sha(directory/'summary.json')!=old['summary_sha256']:raise safety.MLFailure('artifact_invalid')
        return r.read(directory/'summary.json')
    commit=r.freeze(directory,parent)
    forecast=r.read(directory/'forecasts.local.json')
    print(json.dumps({'status':'running','stage':'forecasts_frozen','target_races':len(forecast['races']),
        'forecast_sha256':commit['forecast_sha256'],'saved_forecast_races':sum(x['saved']['status']=='ready' for x in forecast['races']),
        'H_replayed_races':sum(x['H']['status']=='ready' for x in forecast['races'])}),flush=True)
    results=r.results(directory)
    if results['forecast_sha256']!=commit['forecast_sha256']:raise safety.MLFailure('artifact_invalid')
    forecast['committed_sha256']=commit['forecast_sha256']
    payload=scoring.build(forecast,results);payload.update(forecast_sha256=commit['forecast_sha256'],venues=r.VENUES)
    r.verify_commit(directory)
    tasks=old_store.task_configuration();before=forecast['tasks_before'];match=tasks==before if tasks['status']==before['status']=='observed' else None
    if match is False:raise safety.MLFailure('artifact_invalid')
    summary={'kind':'Canopus_best242_and_separate_legacy_V03_vs_H_dashboard','schema_version':2,
        'run_id':directory.name,'status':'ok',**payload['summary'],'saved_models':r.SAVED_MODELS,
        'forecast_sha256':commit['forecast_sha256'],'HTML':str(output),'2026_Jan_Sep_read':False,'models_retrained':False,
        'frozen_references_unchanged':True,'task_configuration_match':match,
        'saved_probabilities_unchanged':True,'legacy_V03_separate':True,'reused_parent':forecast.get('reuse_parent_id'),
        'evaluation_mode':'result_blinded_posthoc_replay_not_strict_forward_validation','real_bets_placed':False}
    r.write(directory,'dashboard.local.json',payload);r.write(directory,'summary.json',summary)
    output=safety.local_path(output,root=r.ROOT)
    with output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(render.html(payload))
    r.write(directory,'dashboard_manifest.json',{'forecast_sha256':commit['forecast_sha256'],'HTML_sha256':r.sha(output),
        'summary_sha256':r.sha(directory/'summary.json'),'result_sha256':r.sha(directory/'results.local.json'),
        'generated_at':r.now(),'result_access_started_after_forecast_commit':True,'task_configuration_match':match,
        'no_training_no_sealed_data_no_deployment':True})
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline HTML duel for exactly 2026-10-17/18; no training.')
    parser.add_argument('--run-id',default='');parser.add_argument('--reuse-run',default='')
    parser.add_argument('--open',action='store_true');args=parser.parse_args(argv)
    try:
        if args.run_id:
            if not re.fullmatch('[a-f0-9]{32}',args.run_id):raise safety.MLFailure('invalid_input')
            directory=safety.local_path(r.ROOT/'evaluations'/args.run_id,root=r.ROOT)
        else:directory=safety.new_artifact('evaluations',root=r.ROOT)
        summary=build(directory,r.parent_directory(args.reuse_run) if args.reuse_run else None)
        print(json.dumps(summary,ensure_ascii=True,allow_nan=False),flush=True)
        if args.open:os.startfile(summary['HTML'])
        return 0
    except SafeAuditError as exc:
        print(json.dumps({'status':'stopped','reason':exc.code,'kind':'prediction_duel'}),flush=True);return 2
    except Exception as exc:
        line=traceback.extract_tb(exc.__traceback__)[-1]
        print(json.dumps({'status':'stopped','reason':'internal_error','exception_type':type(exc).__name__,
            'location':{'file':Path(line.filename).name,'line':line.lineno}}),flush=True);return 2
