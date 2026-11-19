"""One fit only: old D39 parameters/rounds/T on the new D39 final-train bytes."""
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from ..ml import safety
from ..research import model as matrices
from ..tuner_pilot import experiment as tuner
from ..objective_pilot.runner import memory_bytes,tree_memory,terminate_worker
from .. import tuner_reduction as prior
from . import contracts as c,data,learning

VERSION='fixed_old_D39_on_same_current_final_train_20261023_v1'


def folder(parent,identifier):
    if not re.fullmatch('[a-f0-9]{32}',str(identifier)):raise safety.MLFailure('invalid_identity')
    return safety.local_path(parent/'controls'/identifier,root=c.ROOT)


def preserve(parent,directory):
    from ..stage4.store import task_configuration
    # Static artifacts of the completed three-model study are append-only.
    # Its live folders can legitimately grow while this independent fit runs.
    refs={str(p):c.sha(p) for p in parent.rglob('*') if p.is_file()
          and not {'controls','forward','readiness','future-evaluations','forward-four','readiness-four','future-evaluations-four'}
                  .intersection(p.relative_to(parent).parts)}
    for path in (c.OLD_MODEL/'model.json',c.OLD_MODEL/'booster.lgb.txt',c.h.MODEL/'model.json',c.h.MODEL/'win.lgb.txt'):
        refs[str(path)]=c.sha(path)
    c.write(directory/'preservation_before.json',{'references':refs,'tasks':task_configuration()})


def verify(parent,directory):
    from ..stage4.store import task_configuration
    before=c.read(directory/'preservation_before.json')
    for path,sha in before['references'].items():
        if c.sha(Path(path))!=sha:raise safety.MLFailure('artifact_invalid')
    if task_configuration()!=before['tasks']:raise safety.MLFailure('artifact_invalid')
    return dict(c.verify(parent),three_model_static_artifacts_unchanged=len(before['references']))


def prepare(parent):
    c.approved(parent)
    if (parent/'control_candidate_freeze.json').exists() or (parent/'control_plan_commit.json').exists():
        raise safety.MLFailure('artifact_exists')
    old,old_meta=tuner.load_model(c.OLD_MODEL)
    source=c.read(parent/'candidate_freeze.json');new_path=Path(source['model_path'])
    _,new_meta=learning.load(new_path)
    if c.sha(new_path/'model.json')!=source['model_sha256']:raise safety.MLFailure('artifact_invalid')
    frame=data.load_part(parent,'final_train');manifest=c.read(parent/'data_manifest.json')
    categories=matrices.schema(frame,c.FEATURE_SPEC)
    if (old_meta['feature_spec']!=c.FEATURE_SPEC or new_meta['feature_spec']!=c.FEATURE_SPEC
            or new_meta['categories']!=categories or new_meta['dataset_id']!=manifest['dataset_id']
            or old.current_iteration()!=old_meta['best_iteration'] or old_meta['temperature']!=.75):
        raise safety.MLFailure('invalid_model')
    directory=folder(parent,uuid4().hex);directory.mkdir(parents=True,exist_ok=False)
    plan={'version':VERSION,'parent_run_id':parent.name,'created_at_jst':c.clock().isoformat(),
        'source_old_model_id':old_meta['model_id'],'source_new_model_id':new_meta['model_id'],
        'source_old_model_path':str(c.OLD_MODEL),'source_new_model_path':str(new_path),
        'feature_spec':c.FEATURE_SPEC,'features':new_meta['features'],'categories':categories,
        'parameters':old_meta['parameters'],'rounds':old_meta['best_iteration'],'temperature':old_meta['temperature'],
        'calibration_policy':'frozen_old_D39_temperature;no_new_calibration_or_temperature_selection',
        'dataset_id':manifest['dataset_id'],'final_train':manifest['parts']['final_train'],
        'spans':c.read(parent/'plan.json')['spans'],'versions':tuner.versions(),
        'references':{str(path):c.sha(path) for path in (c.OLD_MODEL/'model.json',c.OLD_MODEL/'booster.lgb.txt',
            new_path/'model.json',new_path/'booster.lgb.txt',parent/'data_manifest.json',parent/manifest['parts']['final_train']['file'])},
        'budget':{'actual_train_calls_max':1,'Tuner_fits':0,'calibration_fits':0,'rounds':old_meta['best_iteration'],
            'CPU_threads':4,'private_bytes_max':12*1024**3,'wall_seconds_max':300,'automatic_retry':False},
        'automatic_promotion':False,'Stage7_interim_results_used':False}
    c.write(directory/'plan.json',plan)
    c.write(directory/'authorization.json',{'source':'human_user','max_fits':1,'plan_sha256':c.sha(directory/'plan.json'),
        'basis':'explicit_fourth_model_fixed_old_parameters_onefit_request'})
    c.write(parent/'control_plan_commit.json',{'control_id':directory.name,'plan_sha256':c.sha(directory/'plan.json'),
        'max_total_control_fits_for_parent':1,'automatic_reprepare_or_retry':False})
    preserve(parent,directory)
    return {'status':'prepared','control_id':directory.name,'parent_run_id':parent.name,'max_fits':1,
        'rounds':plan['rounds'],'temperature':plan['temperature'],'races':plan['final_train']['races'],'runners':len(frame)}


def approved(parent,directory):
    c.approved(parent);plan=c.read(directory/'plan.json');authorization=c.read(directory/'authorization.json')
    commit=c.read(parent/'control_plan_commit.json')
    if (plan.get('version')!=VERSION or plan.get('parent_run_id')!=parent.name or authorization.get('source')!='human_user'
            or authorization.get('max_fits')!=1 or authorization.get('plan_sha256')!=c.sha(directory/'plan.json')
            or plan['budget']['actual_train_calls_max']!=1 or plan['budget']['Tuner_fits']!=0
            or commit.get('control_id')!=directory.name or commit.get('plan_sha256')!=c.sha(directory/'plan.json')
            or commit.get('max_total_control_fits_for_parent')!=1):
        raise safety.MLFailure('explicit_execution_required')
    for path,sha in plan['references'].items():
        if c.sha(Path(path))!=sha:raise safety.MLFailure('artifact_invalid')
    old_meta=c.read(c.OLD_MODEL/'model.json')
    if (plan['parameters']!=old_meta['parameters'] or plan['rounds']!=old_meta['best_iteration']
            or plan['temperature']!=old_meta['temperature'] or plan['feature_spec']!=c.FEATURE_SPEC):
        raise safety.MLFailure('invalid_mapping')
    return plan


def worker(parent,directory):
    import lightgbm as lgb
    import numpy as np
    plan=approved(parent,directory)
    if c.read(c.BASE/'research-training.lock').get('run_id')!=directory.name or not (directory/'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    if tuner.versions()!=plan['versions']:raise safety.MLFailure('missing_library')
    c.write(directory/'worker_entered.json',{'started_at':c.clock().isoformat(),'fit_max':1})
    frame=data.load_part(parent,'final_train')
    if matrices.schema(frame,plan['feature_spec'])!=plan['categories']:raise safety.MLFailure('artifact_invalid')
    training=lgb.Dataset(matrices.matrix(frame,plan['feature_spec'],plan['categories']),
        label=frame.winner_label.to_numpy(int),categorical_feature=plan['feature_spec']['categorical_features'])
    cpu=prior.CPU();began=time.monotonic();deadline=began+270
    while cpu.percent()>85:
        if time.monotonic()-began>90:raise safety.MLFailure('input_limit')
        time.sleep(2)
    def ledger(number,state,details):
        c.write(directory/f'actual_fit_{number:02d}.{state}.json',dict(number=number,**details))
    counter=tuner.FitCounter(1,deadline,ledger)
    booster=counter.wrap(lgb.train)(dict(plan['parameters']),training,num_boost_round=plan['rounds'])
    if booster.current_iteration()!=plan['rounds']:raise safety.MLFailure('training_failed')
    model_path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT);model_path.mkdir(parents=True)
    tree=model_path/'booster.lgb.txt'
    with tree.open('x',encoding='utf-8',newline='\n') as stream:stream.write(booster.model_to_string())
    metadata={'version':c.VERSION,'research_version':VERSION,'model_role':'fixed_old_parameters_current_data_control',
        'model_id':model_path.name,'dataset_id':plan['dataset_id'],'created_at_jst':c.clock().isoformat(),
        'feature_spec':plan['feature_spec'],'features':plan['features'],'categories':plan['categories'],
        'formula_version':c.features.FORMULA_VERSION,'input_format':c.features.INPUT_VERSION,
        'parameters':plan['parameters'],'best_iteration':plan['rounds'],'temperature':plan['temperature'],
        'calibration':{'method':'frozen_old_D39_temperature','source_model_id':plan['source_old_model_id'],'refitted':False},
        'spans':plan['spans'],'training_part':'final_train','latest_training_date':plan['final_train']['end'],
        'latest_training_or_calibration_date':plan['final_train']['end'],'tree_sha256':c.sha(tree),
        'plan_sha256':c.sha(directory/'plan.json'),'data_manifest_sha256':c.sha(parent/'data_manifest.json'),
        'same_final_train_sha256':plan['final_train']['sha256'],'versions':plan['versions'],
        'actual_train_calls':counter.calls,'Tuner_fits':0,'new_calibration_fits':0,'production_adoption':False}
    c.write(model_path/'model.json',metadata)
    loaded=learning.load(model_path)
    # A small real-data reload check is a numerical check, never scored or used
    # for tuning. All samples are from this model's declared training bytes.
    sample=frame.iloc[:min(1000,len(frame))]
    a=booster.predict(matrices.matrix(sample,plan['feature_spec'],plan['categories']),num_threads=4)
    b=loaded[0].predict(matrices.matrix(sample,plan['feature_spec'],plan['categories']),num_threads=4)
    delta=float(np.max(np.abs(a-b)))
    if delta>1e-12:raise safety.MLFailure('invalid_model')
    preserved=verify(parent,directory)
    result={'status':'ok','control_id':directory.name,'model_id':model_path.name,'model_path':str(model_path),
        'source_old_model_id':plan['source_old_model_id'],'source_new_model_id':plan['source_new_model_id'],
        'dataset_id':plan['dataset_id'],'final_train':plan['final_train'],'parameters':plan['parameters'],
        'rounds':plan['rounds'],'temperature':plan['temperature'],'actual_fits':counter.calls,'Tuner_fits':0,
        'temperature_refitted':False,'training_seconds':counter.training_seconds,'reload_probability_max_difference':delta,
        'CPU_observed_percent':cpu.observed,'preservation':preserved,'future_performance_status':'not_yet_measured'}
    c.write(directory/'training_report.json',result)
    c.write(parent/'control_candidate_freeze.json',{'control_id':directory.name,'model_id':model_path.name,'model_path':str(model_path),
        'model_sha256':c.sha(model_path/'model.json'),'tree_sha256':c.sha(tree),'dataset_id':plan['dataset_id'],
        'source_old_model_id':plan['source_old_model_id'],'same_final_train_sha256':plan['final_train']['sha256'],
        'temperature':plan['temperature'],'rounds':plan['rounds'],'report_sha256':c.sha(directory/'training_report.json')})
    return {key:result[key] for key in ('status','control_id','model_id','dataset_id','rounds','temperature','actual_fits')}


def execute(parent,directory):
    approved(parent,directory)
    with tuner.training_lock(directory):
        c.write(directory/'execution_started.json',{'started_at':c.clock().isoformat(),'actual_fit_max':1,'retry':False})
        args=[sys.executable,'-B',str(c.PROJECT/'scripts/d39_current.py'),'_control-worker',
              '--run-id',parent.name,'--control-id',directory.name]
        env={k:v for k,v in os.environ.items() if k not in ('PGPASSWORD','PGPASSFILE')}
        env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        began=time.monotonic();peak=0;reason=None
        with subprocess.Popen(args,cwd=c.PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8') as child:
            while child.poll() is None:
                used=tree_memory(child.pid);own=memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:break
                    reason='input_limit'
                else:
                    peak=max(peak,used+own)
                    if used+own>12*1024**3:reason='input_limit'
                if time.monotonic()-began>300:reason='query_timeout'
                if reason:terminate_worker(child);break
                time.sleep(.25)
            output,_=child.communicate(timeout=10)
        try:response=json.loads(output.strip())
        except ValueError:response={'status':'stopped','reason':'training_failed'}
        result={'status':'complete' if child.returncode==0 and response.get('status')=='ok' and not reason else 'stopped',
            'reason':reason or response.get('reason'),'actual_fits':len(list(directory.glob('actual_fit_*.started.json'))),
            'wall_seconds':time.monotonic()-began,'peak_private_bytes':peak,'worker':response,'automatic_retry':False}
        c.write(directory/'completion.json',result)
        return result


def verify_four(parent):
    """Saved real models, fictional inputs, real serialization; no fit or labels."""
    from copy import deepcopy
    from datetime import datetime
    import pandas as pd
    import numpy as np
    from . import forward
    from ..win_sim import core
    from ..research import features as rf
    ready=forward.preflight(parent,four=True);plan=c.read(parent/'forward_four_plan.json')
    frozen=c.read(parent/'control_candidate_freeze.json');directory=folder(parent,frozen['control_id'])
    plan=deepcopy(plan);plan['synthetic_verification']=True
    day=plan['start_date'];race=(day[:4],day[5:7]+day[8:10],'05','01','01','01')
    now=datetime.fromisoformat(day+'T09:50:05+09:00')
    registered={'race_key':list(race),'roster':['01','02','03'],'scheduled_start':day+'T10:00:00+09:00',
        'registered_at':day+'T08:00:00+09:00','reason':None}
    rows=[]
    for i in range(1,4):
        rows.append(dict(zip(('kaisai_nen','kaisai_tsukihi','keibajo_code','kaisai_kai','kaisai_nichime','race_bango'),race),
            horse_number=f'{i:02d}',horse_id=f'FAKE{i:06d}',surface='11',distance='1600',declared_runners='03',
            frame=str(i),age='04',sex='1',weight_raw='550',grade=' ',previous_grade_revision=' ',
            condition_2='005',condition_3='005',condition_4='005',condition_5='005',race_type='13',weight_type='4',
            jockey=f'{i:05d}',previous_jockey_revision='00000',trainer=f'{i:05d}'))
    bundles={'old_D39':tuner.load_model(c.OLD_MODEL),'new_D39':learning.load(Path(plan['new_model_path'])),
        'control_D39':learning.load(Path(plan['control_model_path'])),'H':matrices.load(c.h.MODEL,root=c.S2)}
    # Keep hash-named input files below legacy Windows MAX_PATH without
    # changing the normal live recorder's file layout or registry settings.
    out=forward.mkdir(parent/'verification-four'/uuid4().hex);records={};max_sum_error=0.
    for four in (False,True):
        selected=deepcopy(plan) if four else c.read(parent/'forward_plan.json')
        selected['synthetic_verification']=True
        target=forward.mkdir(out/('four' if four else 'three'))
        forward.capture(target,'entries',rows,datetime.fromisoformat(day+'T09:49:40+09:00'))
        for hourminute,prices,seen in (('0929',(50,30,18),'09:29:30'),('0949',(40,30,20),'09:49:30')):
            payload=''.join(f'{i:02d}{price:04d}{i:02d}' for i,price in enumerate(prices,1))+' '*200
            raw={'race_key':list(race),'record_type':'O1','status':'1','published':race[1]+hourminute,'sale':'7',
                'payload':payload,'source_table':'apd_sokuho_o1','oversized':False,'unsafe_version':False}
            forward.capture(target,'odds',raw,datetime.fromisoformat(day+'T'+seen+'+09:00'))
        receipt=forward.decide(target,registered,registered,selected,bundles,rf.Builder(),clock=lambda:now)
        if receipt['status']!='saved':raise safety.MLFailure('invalid_model')
        saved=pd.read_csv(target/'predictions.csv',dtype={'entry_key':str},float_precision='round_trip')
        record=c.read(target/'prediction.json');records[four]=record
        names=forward.FOUR_NAMES if four else forward.NAMES
        if (set(saved.model)!=set(names) or len(saved)!=3*len(names) or saved.generated_at.nunique()!=1
                or saved.input_cutoff.nunique()!=1 or record.get('synthetic_verification') is not True):
            raise safety.MLFailure('artifact_invalid')
        for name,part in saved.groupby('model'):
            if set(part.entry_key)!= {''.join(race)+h for h in registered['roster']} or set(part.model_id)!={selected['model_ids'][name]}:
                raise safety.MLFailure('invalid_identity')
            if not np.isfinite(part.probability).all():raise safety.MLFailure('invalid_model')
            max_sum_error=max(max_sum_error,abs(float(part.probability.sum())-1.))
    for name in forward.NAMES:
        if (records[False]['probabilities'][name]!=records[True]['probabilities'][name]
                or records[False]['ledgers'][name]!=records[True]['ledgers'][name]):raise safety.MLFailure('invalid_model')
    if max_sum_error>1e-12:raise safety.MLFailure('invalid_model')
    result={'status':'ok','verification_id':out.name,'models_loaded':4,'same_fictional_runner_count':3,'saved_CSV_rows':12,
        'model_ids':plan['model_ids'],'all_models_same_roster':True,'all_models_same_T10_cutoff':True,
        'all_models_same_generation_timestamp':True,'maximum_race_probability_sum_error':max_sum_error,
        'old_three_models_probabilities_and_decisions_unchanged':True,'synthetic_inputs_only':True,'results_read':False,
        'new_fits':0,'not_a_real_pre_race_record_or_performance_evaluation':True,'saved_folder':str(out),
        'preservation':verify(parent,directory),'ready':ready}
    c.write(out/'verification.json',result)
    return result
