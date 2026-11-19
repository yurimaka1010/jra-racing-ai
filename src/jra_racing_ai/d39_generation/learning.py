"""Eight official Tuner fits and one fixed-round refit; no historical test claims."""
from pathlib import Path
from uuid import uuid4
import json
import os
import subprocess
import sys
import time

from ..ml import safety
from ..research import model as matrices
from ..tuner_pilot import experiment as tuner
from ..objective_pilot import model as adapters
from ..objective_pilot.runner import calibration_table,tree_memory,memory_bytes,terminate_worker
from .. import tuner_reduction as prior
from ..stage5 import metrics
from . import contracts as c,data


def point(frame,probabilities):
    import numpy as np
    data.validate_frame(frame)
    y=frame.winner_label.to_numpy(int);p=np.asarray(probabilities,float)
    _,codes=np.unique(frame.entry_key.str[:16].to_numpy(str),return_inverse=True)
    if p.shape!=y.shape or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not np.allclose(np.bincount(codes,weights=p),1.,atol=1e-12,rtol=0):
        raise safety.MLFailure('invalid_input')
    values=metrics.statistics(y,p,codes,np.zeros(len(frame),int),1)[0]
    return {'races':int(values[0]),'runners':int(values[1]),'race_log_loss':float(values[2]/values[0]),
            'horse_log_loss':float(values[3]/values[1]),'brier_score':float(values[4]/values[1]),
            'roc_auc':metrics.auc(y,p),'within_race_auc':float(values[6]/values[0]),
            'top1_accuracy_percent':float(100*values[5]/values[0]),'calibration_bands':calibration_table(y,p)}


class Budget:
    def __init__(self,directory,deadline):
        self.directory=directory;self.deadline=deadline;self.calls=0;self.cpu=prior.CPU();self.seconds=0.

    def reserve(self,phase):
        if phase not in ('tuner','final_refit') or self.calls>=9 or (phase=='tuner' and self.calls>=8):
            raise safety.MLFailure('input_limit')
        began=time.monotonic()
        while self.cpu.percent()>85:
            if time.monotonic()-began>90 or time.monotonic()>=self.deadline:raise safety.MLFailure('input_limit')
            time.sleep(2)
        self.calls+=1
        c.write(self.directory/f'actual_fit_{self.calls:02d}.started.json',{'phase':phase,'failed_attempt_consumes_budget':True})
        return self.calls


def load(path):
    import lightgbm as lgb
    meta=c.read(path/'model.json')
    if (meta.get('version')!=c.VERSION or meta.get('feature_spec')!=c.FEATURE_SPEC
            or meta.get('features')!=c.FEATURE_SPEC['features'] or meta.get('temperature') not in (.75,1.,1.25,2.)
            or c.sha(path/'booster.lgb.txt')!=meta['tree_sha256']):raise safety.MLFailure('invalid_model')
    booster=lgb.Booster(model_str=(path/'booster.lgb.txt').read_text(encoding='utf-8'))
    if booster.feature_name()!=meta['features'] or booster.current_iteration()!=meta['best_iteration']:
        raise safety.MLFailure('invalid_model')
    return booster,meta


def predict(bundle,frame):
    booster,meta=bundle
    raw=booster.predict(matrices.matrix(frame,meta['feature_spec'],meta['categories']),num_threads=4)
    return matrices.normalize(frame,raw,meta['temperature'])


def worker(directory):
    import lightgbm as lgb
    import numpy as np
    plan=c.approved(directory)
    if c.read(c.BASE/'research-training.lock').get('run_id')!=directory.name or not (directory/'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    if tuner.versions()!=plan['versions']:raise safety.MLFailure('missing_library')
    c.write(directory/'worker_entered.json',{'started_at_jst':c.clock().isoformat(),'retry':False})
    frames={name:data.load_part(directory,name) for name in ('train','early_stopping','final_train','calibration')}
    if not max(frames['train'].date)<min(frames['early_stopping'].date)<=max(frames['final_train'].date)<min(frames['calibration'].date):
        raise safety.MLFailure('invalid_split')
    spec=plan['feature_spec'];categories=matrices.schema(frames['train'],spec)
    sets=[]
    for name in ('train','early_stopping'):
        sets.append(lgb.Dataset(matrices.matrix(frames[name],spec,categories),label=frames[name].winner_label.to_numpy(int),
                    categorical_feature=spec['categorical_features'],reference=sets[0] if sets else None))
    budget=Budget(directory,time.monotonic()+1170);reservations={}
    def ledger(number,state,details):
        if state=='started':reservations[number]=budget.reserve('tuner')
        else:
            budget.seconds+=details['actual_train_call_seconds']
            c.write(directory/f'actual_fit_{reservations[number]:02d}.{state}.json',dict(phase='tuner',**details))
    counter=tuner.FitCounter(8,min(budget.deadline,time.monotonic()+600),ledger)
    began=time.monotonic()
    best,params,trials=tuner.tune(sets[0],sets[1],plan['parameters'],counter=counter)
    tuning_wall=time.monotonic()-began
    selected_rounds=int(best.best_iteration)
    if not 1<=selected_rounds<=300 or not 1<=counter.calls<=8:raise safety.MLFailure('training_failed')
    c.write(directory/'tuning_result.json',{'parameters':params,'selected_rounds':selected_rounds,
        'actual_tuner_fits':counter.calls,'trials':trials,'tuning_wall_seconds':tuning_wall,'selection':'early_stopping_binary_logloss_only'})
    # Refit includes the early-stop/gap dates, but never the independent calibration window.
    categories=matrices.schema(frames['final_train'],spec)
    final_set=lgb.Dataset(matrices.matrix(frames['final_train'],spec,categories),
        label=frames['final_train'].winner_label.to_numpy(int),categorical_feature=spec['categorical_features'])
    def final_ledger(number,state,details):
        if state=='started':reservations['final']=budget.reserve('final_refit')
        else:
            budget.seconds+=details['actual_train_call_seconds']
            c.write(directory/f'actual_fit_{reservations["final"]:02d}.{state}.json',dict(phase='final_refit',**details))
    final_counter=tuner.FitCounter(1,budget.deadline,final_ledger)
    final=final_counter.wrap(lgb.train)(dict(params),final_set,num_boost_round=selected_rounds)
    raw=final.predict(matrices.matrix(frames['calibration'],spec,categories),num_threads=4)
    temperature,grid=adapters.calibrate(frames['calibration'],raw,{'objective':'binary','temperature_candidates':plan['calibration']['grid']})
    path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT);path.mkdir(parents=True)
    tree=path/'booster.lgb.txt'
    with tree.open('x',encoding='utf-8',newline='\n') as stream:stream.write(final.model_to_string())
    manifest=c.read(directory/'data_manifest.json')
    meta={'version':c.VERSION,'model_id':path.name,'dataset_id':manifest['dataset_id'],'created_at_jst':c.clock().isoformat(),
        'features':spec['features'],'feature_spec':spec,'categories':categories,'formula_version':c.features.FORMULA_VERSION,
        'input_format':c.features.INPUT_VERSION,'parameters':params,'best_iteration':int(final.current_iteration()),
        'early_stopping_selected_rounds':selected_rounds,'temperature':temperature,'calibration':grid,'spans':plan['spans'],
        'calibration_role':'temperature_selection_only_not_independent_test','latest_training_or_calibration_date':plan['latest_final_date'],
        'tree_sha256':c.sha(tree),'plan_sha256':c.sha(directory/'plan.json'),'data_manifest_sha256':c.sha(directory/'data_manifest.json'),
        'versions':tuner.versions(),'actual_train_calls':budget.calls,'final_refit':1,'production_adoption':False,
        '2026_training_authorized_for_new_D39_only':True,'old_sealed_evaluation_records_unchanged':True}
    c.write(path/'model.json',meta)
    loaded=load(path);probabilities=predict(loaded,frames['calibration'])
    delta=float(np.max(np.abs(probabilities-matrices.normalize(frames['calibration'],raw,temperature))))
    if delta>1e-12:raise safety.MLFailure('invalid_model')
    old_bundle=tuner.load_model(c.OLD_MODEL);old_p=tuner.predict(*old_bundle,frames['calibration'])
    with (directory/'calibration_diagnostic.predictions.local.csv').open('x',encoding='utf-8',newline='') as stream:
        frames['calibration'][['entry_key','date','winner_label']].assign(old_D39=old_p,new_D39=probabilities).to_csv(stream,index=False)
    report={'status':'ok','kind':'new_D39_training_complete','model_id':path.name,'model_path':str(path),
        'dataset_id':manifest['dataset_id'],'actual_train_calls':budget.calls,'tuner_fits':counter.calls,'final_refits':1,
        'selected_rounds':selected_rounds,'temperature':temperature,'temperature_grid':grid,'parameters':params,
        'training_seconds':budget.seconds,'tuning_wall_seconds':tuning_wall,'CPU_observed_percent':budget.cpu.observed,
        'reload_max_probability_difference':delta,'calibration_diagnostic_only':{'old_D39':point(frames['calibration'],old_p),
        'new_D39':point(frames['calibration'],probabilities)},'future_common_three_model_evaluation':{'status':'pending_future_pre_result_records','races':0},
        'no_learning_date_claimed_as_unused_test':True,'automatic_promotion':False,'Stage7_interim_results_used':False}
    c.write(directory/'training_report.json',report)
    c.write(directory/'candidate_freeze.json',{'model_id':path.name,'model_path':str(path),'model_sha256':c.sha(path/'model.json'),
        'tree_sha256':c.sha(tree),'dataset_id':manifest['dataset_id'],'report_sha256':c.sha(directory/'training_report.json'),
        'future_comparators':{'old_D39':old_bundle[1]['model_id'],'H':c.h.H_ID},'automatic_promotion':False})
    return {key:report[key] for key in ('status','kind','model_id','dataset_id','actual_train_calls','temperature','selected_rounds')}


def execute(directory):
    c.approved(directory)
    with tuner.training_lock(directory):
        c.write(directory/'execution_started.json',{'started_at_jst':c.clock().isoformat(),'max_fits':9,'automatic_retry':False})
        began=time.monotonic();peak=0;reason=None
        args=[sys.executable,'-B',str(c.PROJECT/'scripts/d39_current.py'),'_worker','--run-id',directory.name]
        # Fitting cannot access DB credentials and does not need network access.
        env={k:v for k,v in os.environ.items() if k not in ('PGPASSWORD','PGPASSFILE')}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with subprocess.Popen(args,cwd=c.PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8') as child:
            while child.poll() is None:
                used=tree_memory(child.pid);own=memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:break
                    reason='input_limit'
                else:
                    peak=max(peak,used+own)
                    if used+own>12*1024**3:reason='input_limit'
                if time.monotonic()-began>1200:reason='query_timeout'
                if reason:terminate_worker(child);break
                time.sleep(.25)
            output,_=child.communicate(timeout=10)
        try:response=json.loads(output.strip())
        except ValueError:response={'status':'stopped','reason':'training_failed'}
        result={'status':'complete' if not reason and child.returncode==0 and response.get('status')=='ok' else 'stopped',
            'reason':reason or response.get('reason'),'actual_fits':len(list(directory.glob('actual_fit_*.started.json'))),
            'wall_seconds':time.monotonic()-began,'peak_parent_worker_private_bytes':peak,'worker':response,'automatic_retry':False}
        c.write(directory/'completion.json',result)
        return result
