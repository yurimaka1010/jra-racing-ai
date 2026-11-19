"""One fixed-parameter fit. No tuner, retry, or feedback from forward outcomes."""
from pathlib import Path
from uuid import uuid4
import json
import os
import subprocess
import sys
import time
from ..ml import safety
from ..research import model as matrices
from ..objective_pilot import model as adapters
from ..tuner_pilot import experiment as budget
from ..d39_generation.learning import point
from . import contracts as c, data


def load(path):
    import lightgbm as lgb
    m=c.read(path/'model.json')
    if (m.get('version')!=c.VERSION or m.get('feature_spec')!=c.spec()
            or m.get('features')!=c.spec()['features'] or m.get('actual_train_calls')!=1
            or m.get('temperature') not in (.75,1.,1.25,2.)
            or c.sha(path/'booster.lgb.txt')!=m['tree_sha256']):
        raise safety.MLFailure('invalid_model')
    booster=lgb.Booster(model_str=(path/'booster.lgb.txt').read_text(encoding='utf-8'))
    if booster.feature_name()!=m['features'] or booster.current_iteration()!=m['best_iteration']:
        raise safety.MLFailure('invalid_model')
    return booster,m


def predict(bundle,frame,*,raw=False):
    booster,meta=bundle
    values=booster.predict(data.matrix(frame,meta['feature_spec'],meta['categories']),num_threads=4)
    return values if raw else matrices.normalize(frame,values,meta['temperature'])


def worker(directory):
    import lightgbm as lgb
    import numpy as np
    from ..d39_next import learning as previous
    p=c.approved(directory)
    if (not (directory/'execution_started.json').exists()
            or c.read(c.BASE/'research-training.lock').get('run_id')!=directory.name):
        raise safety.MLFailure('explicit_execution_required')
    c.write(directory/'worker_entered.json',{'at':c.clock().isoformat(),'automatic_retry':False})
    began=time.monotonic();deadline=began+880
    frames={name:data.load_part(directory,name) for name in ('final_train','calibration')}
    if max(frames['final_train'].date)>=min(frames['calibration'].date):raise safety.MLFailure('invalid_split')
    if p['rounds']!=219 or p['parameters']['num_threads']!=4:raise safety.MLFailure('invalid_input')
    parent=c.read(c.PARENT/'candidate_freeze.json');old=previous.load(Path(parent['model_path']))
    if p['parameters']!=old[1]['parameters'] or budget.versions()!=old[1]['versions']:
        raise safety.MLFailure('invalid_model')
    spec=p['feature_spec'];cats=matrices.schema(frames['final_train'],spec)
    dataset=lgb.Dataset(data.matrix(frames['final_train'],spec,cats),
        label=frames['final_train'].winner_label.to_numpy(int),categorical_feature=spec['categorical_features'])
    def ledger(number,state,details):c.write(directory/f'fit_{number:02d}.{state}.json',details)
    counter=budget.FitCounter(1,deadline,ledger)
    booster=counter.wrap(lgb.train)(dict(p['parameters']),dataset,num_boost_round=p['rounds'])
    raw=booster.predict(data.matrix(frames['calibration'],spec,cats),num_threads=4)
    temperature,grid=adapters.calibrate(frames['calibration'],raw,
        {'objective':'binary','temperature_candidates':p['calibration']['grid']})
    manifest=c.read(directory/'data_manifest.json')
    path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT);path.mkdir(parents=True)
    tree=path/'booster.lgb.txt'
    with tree.open('x',encoding='utf-8',newline='\n') as stream:stream.write(booster.model_to_string())
    meta={'version':c.VERSION,'model_id':path.name,'features':spec['features'],'feature_spec':spec,
        'categories':cats,'parameters':p['parameters'],'best_iteration':int(booster.current_iteration()),
        'temperature':temperature,'calibration':grid,'dataset_id':manifest['dataset_id'],
        'data_manifest_sha256':c.sha(directory/'data_manifest.json'),'plan_sha256':c.sha(directory/'plan.json'),
        'spans':p['spans'],'created_at':c.clock().isoformat(),'latest_training_or_calibration_date':p['source_end'],
        'tree_sha256':c.sha(tree),'versions':budget.versions(),'actual_train_calls':1,
        'production_replacement':False,'Stage7_results_used':False,
        'going_training_available':manifest['going_training_available']}
    c.write(path/'model.json',meta);loaded=load(path)
    predicted=predict(loaded,frames['calibration'])
    delta=float(np.max(np.abs(predicted-matrices.normalize(frames['calibration'],raw,temperature))))
    if delta>1e-12:raise safety.MLFailure('invalid_model')
    old_frame=__import__('jra_racing_ai.d39_next.data',fromlist=['load_part']).load_part(c.PARENT,'calibration')
    if list(old_frame.entry_key)!=list(frames['calibration'].entry_key):raise safety.MLFailure('artifact_invalid')
    report={'status':'ok','model_id':path.name,'model_path':str(path),'actual_train_calls':counter.calls,
        'selected_rounds':p['rounds'],'temperature':temperature,'calibration_grid':grid,
        'parameters':p['parameters'],'reload_max_difference':delta,
        'calibration_diagnostic_only':point(frames['calibration'],predicted),
        'previous47_calibration_diagnostic':point(old_frame,previous.predict(old,old_frame)),
        'not_an_unused_test':True,'prospective_evaluation_races':0,
        'training_seconds':counter.training_seconds,'worker_seconds':time.monotonic()-began,
        'going_training_available':manifest['going_training_available'],
        'feature_gain':dict(zip(meta['features'],map(float,booster.feature_importance('gain'))))}
    c.write(directory/'training_report.json',report)
    c.write(directory/'candidate_freeze.json',{'model_id':path.name,'model_path':str(path),
        'model_sha256':c.sha(path/'model.json'),'tree_sha256':c.sha(tree),'plan_sha256':c.sha(directory/'plan.json'),
        'created_at':c.clock().isoformat(),'live_participation_authorized':True,'replace_existing_models':False})
    return {k:report[k] for k in ('status','model_id','actual_train_calls','selected_rounds','temperature','training_seconds')}


def execute(directory):
    from ..objective_pilot.runner import tree_memory,memory_bytes,terminate_worker
    p=c.approved(directory)
    with budget.training_lock(directory):
        # Exclusive marker is deliberately never removed, even after a failed attempt.
        c.write(directory/'execution_started.json',{'at':c.clock().isoformat(),'automatic_retry':False})
        env={k:v for k,v in os.environ.items() if k not in ('PGPASSWORD','PGPASSFILE')}
        env.update(PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        command=[sys.executable,'-B',str(c.prior.old.PROJECT/'scripts/d39_enrichment.py'),'_worker','--run-id',directory.name]
        began=time.monotonic();peak=0;reason=None
        options={'creationflags':subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name=='nt' else {}
        with subprocess.Popen(command,cwd=c.prior.old.PROJECT,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,text=True,encoding='utf-8',**options) as child:
            while child.poll() is None:
                used,own=tree_memory(child.pid),memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:break
                    reason='input_limit'
                else:
                    peak=max(peak,used+own)
                    if used+own>p['budget']['private_bytes_max']:reason='input_limit'
                if time.monotonic()-began>p['budget']['whole_seconds']:reason='query_timeout'
                if reason:terminate_worker(child);break
                time.sleep(.25)
            output,_=child.communicate(timeout=10)
        try:response=json.loads(output.strip())
        except ValueError:response={'status':'stopped','reason':'training_failed'}
        result={'status':'complete' if not reason and child.returncode==0 and response.get('status')=='ok' else 'stopped',
            'reason':reason or response.get('reason'),'actual_train_calls':len(list(directory.glob('fit_*.started.json'))),
            'wall_seconds':time.monotonic()-began,'peak_private_bytes':peak,'worker':response,
            'preservation':c.prior.verify_preservation(c.read(directory/'preservation_before.json'))}
        c.write(directory/'completion.json',result)
        return result
