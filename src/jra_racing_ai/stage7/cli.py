import argparse,json,os,traceback
from pathlib import Path
from datetime import date
from ..core import SafeAuditError
from ..db_connection import read_only_session
from ..ml import safety
from ..win_sim_cli import connection_config
from ..win_sim import publication,local
from . import contracts as c,store as s,planning,forward,survey

def session():
    if not os.environ.get('PGPASSWORD'):raise safety.MLFailure('needs_authentication')
    return read_only_session(connection_config(60000),os.environ['PGPASSWORD'])

def preflight(folder):
    plan=planning.verify(folder,'forward');bundle=s.fixed_model(plan)
    # Synthetic quotes only, read/reload H twice; no fit and no real predictions.
    from .inputs import predict
    race=('2026','1024','05','04','06','01')
    q={'race_key':list(race),'odds':{'01':'2.0','02':'4.0'},'source':'synthetic','category':'1',
        'basis':'pre_race_publication_time','published_at':'2026-10-24T09:19:00+09:00',
        'minutes_before_start':11.,'cancelled_outside_roster':0}
    first,_=predict(bundle,race,q,None);second,_=predict(s.fixed_model(plan),race,q,None)
    if first!=second:raise safety.MLFailure('artifact_invalid')
    value={'status':'ready','H_model_id':c.H_ID,'temperature':1.,'reload_max_probability_difference':0.,
        'synthetic_H_probability_sum':sum(first['H'].values()),'synthetic_market_probability_sum':sum(first['raw_market_T10'].values()),
        'new_fits':0,'database_connected':False}
    if not (folder/'preflight.json').exists():s.write(folder/'preflight.json',value)
    return value

def probe(folder):
    plan=planning.verify(folder,'forward');day=date.fromisoformat(plan['start_date'])
    with session() as db:
        cols=publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,local.Budget(60))
        if cols is None:raise safety.MLFailure('invalid_schema')
        cards=forward.live_cards(db,day)
    value={'status':'ok','scope_date':str(day),'RA_race_keys_observed':len(cards),
        'pre_race_complete_rosters':sum(v['reason'] is None for v in cards.values()),
        'pre_race_runners':sum(len(v['roster']) for v in cards.values() if v['reason'] is None),
        'schema_confirmed':True,'results_columns_selected':False,'fits':0}
    s.event(folder,'forward_input_probe',**value);return value

def report(folder):
    plan=planning.verify(folder,'forward');summary=s.read(folder/'T3_summary.json') if (folder/'T3_summary.json').exists() else None
    value={'status':'ok','kind':'stage7a_completion','run_id':folder.name,'H_model_id':c.H_ID,'temperature':1.,
        'forward_plan':{k:plan[k] for k in ('start_date','end_date','planned_weeks')},
        'forward_preflight':s.read(folder/'preflight.json') if (folder/'preflight.json').exists() else None,
        'T3':{k:summary[k] for k in ('status','complete','B','C','years','roles','bounded_queries','elapsed_seconds','fits','stop_reason')} if summary else None,
        'preservation':s.verify_old(folder),'new_real_data_fits':0,'stage7b_approved':False,'2026_Jan_Sep_opened':False}
    p=folder/'completion.json'
    if not p.exists():s.write(p,value)
    return value

def main(argv=None):
    parser=argparse.ArgumentParser(description='Approved Stage 7a only: frozen H record and label-free T3 survey. No fit command.')
    parser.add_argument('mode',choices=('init','preflight','prepare-survey','probe','survey','record','report'))
    parser.add_argument('--run-id',default='');parser.add_argument('--start-date',default='2026-10-24');parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);folder=None
    try:
        if args.mode=='init':
            folder=safety.new_artifact('evaluations',root=c.ROOT);value=planning.initialize(folder,args.start_date)
        else:
            folder=s.directory(args.run_id)
            if args.mode=='preflight':value=preflight(folder)
            elif args.mode=='prepare-survey':value=planning.prepare_survey(folder)
            elif args.mode=='probe':value=probe(folder)
            elif args.mode=='survey':
                full=survey.run(folder,session)
                value={k:full[k] for k in ('status','complete','stop_reason','B','C','years','roles','target_races','completed_races','bounded_queries','elapsed_seconds','fits')}
            elif args.mode=='record':value=forward.record(folder,session,once=args.once)
            else:value=report(folder)
        print(json.dumps({'status':'ok','run_id':folder.name,'mode':args.mode,'summary':value},ensure_ascii=True,allow_nan=False),flush=True);return 0
    except SafeAuditError as exc:
        if folder is not None and folder.exists():s.event(folder,'stage7a_stopped',exc.code)
        print(json.dumps({'status':'stopped','reason':exc.code,'stage':args.mode,'run_id':folder.name if folder else None}),flush=True);return 2
    except Exception as exc:
        last=traceback.extract_tb(exc.__traceback__)[-1]
        if folder is not None and folder.exists():s.event(folder,'stage7a_stopped','internal_error',exception_type=type(exc).__name__,file=Path(last.filename).name,line=last.lineno)
        print(json.dumps({'status':'stopped','reason':'internal_error','stage':args.mode,'exception_type':type(exc).__name__,
            'file':Path(last.filename).name,'line':last.lineno,'run_id':folder.name if folder else None}),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
