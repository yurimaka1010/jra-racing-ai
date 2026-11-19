"""Local-only new D39 generation. Existing sealed development entry points stay closed."""
import argparse
import json
from pathlib import Path

from ..ml import safety
from ..research.io import session
from ..tuner_pilot import experiment as tuner
from . import contracts as c,data,learning


def initialize():
    newest,days=data.latest(session)
    original=tuner.read(c.old.OLD_RUN,'plan.json')
    directory=safety.new_artifact('evaluations',root=c.ROOT)
    refs={str(path):c.sha(path) for path in (c.OLD_MODEL/'model.json',c.OLD_MODEL/'booster.lgb.txt',
        c.h.MODEL/'model.json',c.h.MODEL/'win.lgb.txt',c.S2/'source_manifest.json',c.S2/'feature_manifest.json')}
    plan={'version':c.VERSION,'created_at_jst':c.clock().isoformat(),'latest_final_date':str(newest),
        '2026_training_authorized':True,'authorization_scope':'new_general_D39_training_only;old_seal_records_unchanged',
        'feature_spec':c.FEATURE_SPEC,'spans':c.split(newest),'parameters':original['parameters'],
        'steps':original['steps'],'calibration':original['calibration'],'versions':tuner.versions(),
        'parameter_initialization':'original_common_initial_parameters_not_old_tuned_parameters',
        'source_references':refs,'budget':{'tuner_actual_fits_max':8,'final_refits_max':1,'total_fits_max':9,
        'rounds_max':300,'early_stopping_rounds':20,'threads':4,'private_bytes_max':12*1024**3,
        'fit_worker_seconds_max':1200,'tuner_seconds_max':600,'collect_seconds_max':900,'build_seconds_max':900,
        'monthly_rows_max':20000,'statement_seconds_max':60,'automatic_retry':False},
        'comparison':{'models':['old_D39','new_D39','H'],'primary':'future_same_roster_same_T10_cutoff',
        'past_fitted_dates_are_not_unused_tests':True,'calibration_diagnostic_is_not_test':True,
        'Stage7_interim_results_used':False,'automatic_adoption':False},
        'latest_days_aggregate':days,'H_and_tasks_must_be_unchanged':True}
    c.write(directory/'plan.json',plan)
    c.write(directory/'authorization.json',{'source':'human_user','max_fits':9,'plan_sha256':c.sha(directory/'plan.json'),
        'basis':'explicit_2026_training_authorization;8_Tuner_plus_at_most_1_announced_final_refit'})
    c.preserve(directory)
    return {'status':'prepared','run_id':directory.name,'latest_final_date':str(newest),'spans':plan['spans'],'max_fits':9}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Independent current-data D39; original H and tasks unchanged')
    parser.add_argument('command',choices=('init','collect','build','train','_worker','report','preflight','record','evaluate',
        'prepare-control','train-control','_control-worker','verify-four','preflight-four','record-four','evaluate-four',
        'prepare-addendum','diagnose-control','dry-run-four','verify-addendum'))
    parser.add_argument('--run-id');parser.add_argument('--control-id');parser.add_argument('--once',action='store_true');parser.add_argument('--date')
    args=parser.parse_args(argv)
    try:
        if args.command=='init':result=initialize()
        else:
            directory=c.folder(args.run_id);plan=c.approved(directory)
            for path,value in plan['source_references'].items():
                if c.sha(Path(path))!=value:raise safety.MLFailure('artifact_invalid')
            if args.command=='collect':result=dict(status='ok',**data.collect(directory,session))
            elif args.command=='build':result=dict(status='ok',**data.build(directory))
            elif args.command=='train':result=learning.execute(directory)
            elif args.command=='_worker':result=learning.worker(directory)
            elif args.command in ('prepare-control','train-control','_control-worker','verify-four'):
                from . import control
                if args.command=='prepare-control':result=control.prepare(directory)
                elif args.command=='verify-four':result=control.verify_four(directory)
                elif args.command=='train-control':result=control.execute(directory,control.folder(directory,args.control_id))
                else:result=control.worker(directory,control.folder(directory,args.control_id))
            elif args.command=='report':
                from .report import report
                result=report(directory)
            elif args.command=='prepare-addendum':
                from .addendum import prepare
                result=prepare(directory)
            elif args.command=='diagnose-control':
                from .addendum import diagnostics
                result=diagnostics(directory)
            elif args.command=='verify-addendum':
                from .addendum import verify_raw
                result=verify_raw(directory)
            elif args.command=='preflight-four':
                from .four_forward import preflight
                result=preflight(directory)
            elif args.command=='record-four':
                from .four_forward import run
                result=run(directory,session,once=args.once)
            elif args.command=='dry-run-four':
                from .four_dry import run
                result=run(directory,session,args.date)
            elif args.command=='evaluate-four':
                from .four_final import run
                result=run(directory,session)
            else:
                from . import forward
                if args.command in ('preflight','preflight-four'):result=forward.preflight(directory,four=args.command.endswith('-four'))
                elif args.command in ('record','record-four'):result=forward.run(directory,session,once=args.once,four=args.command.endswith('-four'))
                else:
                    from .outcomes import evaluate
                    result=evaluate(directory,session,args.date,four=args.command.endswith('-four'))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if result.get('status') in ('ok','prepared','complete','ready') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status':'stopped','reason':error.code,'stage':args.command}),flush=True);return 1
    except Exception:
        print(json.dumps({'status':'stopped','reason':'internal_error','stage':args.command}),flush=True);return 1
