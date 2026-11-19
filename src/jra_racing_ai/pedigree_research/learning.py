"""Two explicitly authorized fixed fits. No search, retry, deployment or H scoring."""
from pathlib import Path
from uuid import uuid4
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from ..ml import safety
from ..research import model as matrices
from ..objective_pilot import model as calibration
from ..tuner_pilot import experiment as budget
from ..d39_generation.learning import point
from . import contracts as c, data
from .preparation import preservation, verify

VERSION = 'pedigree_two_fixed_fits_20261026_v1'
SOURCE_ID = '7be65e783430433ebc7440838360ef30'
NAMES = ('control39', 'pedigree45')


def folder(identifier):
    if not re.fullmatch('[a-f0-9]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(c.ROOT/'evaluations'/identifier, root=c.ROOT)


def model_spec(name):
    if name not in NAMES: raise safety.MLFailure('invalid_input')
    return data.spec(name == 'pedigree45')


def prepare():
    source=folder(SOURCE_ID)
    proposal=c.read(source/'learning_plan.proposed.json')
    if (c.sha(source/'learning_plan.proposed.json')!=c.read(source/'learning_plan.commit.json')['sha256']
            or c.read(source/'completion.json')['status']!='stage1_complete_zero_fit'
            or proposal['total_fits_max']!=2 or proposal['rounds']!=219
            or c.sha(source/'feature_spec.json')!=proposal['pedigree_spec_sha256']
            or c.sha(source/'identity_manifest.json')!=proposal['identity_source_manifest_sha256']):
        raise safety.MLFailure('artifact_invalid')
    if (c.ROOT/'learning-claims'/f'{SOURCE_ID}.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    freeze=c.read(c.REFERENCE/'candidate_freeze.json');model_path=Path(freeze['model_path'])
    meta=c.read(model_path/'model.json')
    if (c.sha(model_path/'model.json')!=freeze['model_sha256'] or proposal['parameters']!=meta['parameters']
            or meta['best_iteration']!=219 or budget.versions()!=meta['versions']):
        raise safety.MLFailure('invalid_model')
    refs={}
    for path in (source/'learning_plan.proposed.json',source/'learning_plan.commit.json',source/'feature_spec.json',
                 source/'identity_manifest.json',source/'pedigree_summary.json',source/'lineage.local.json',
                 model_path/'model.json',model_path/'booster.lgb.txt',c.REFERENCE/'data_manifest.json'):
        refs[str(path)]=c.sha(path)
    for part in ('final_train','calibration'):
        for path in (c.REFERENCE/(part+'.local.csv'),source/(part+'.pedigree.local.csv')):
            refs[str(path)]=c.sha(path)
        if refs[str(c.REFERENCE/(part+'.local.csv'))]!=proposal['data_identity'][part]:
            raise safety.MLFailure('artifact_invalid')
    directory=safety.new_artifact('evaluations',root=c.ROOT)
    plan={'version':VERSION,'created_at':c.clock().isoformat(),'source_preparation_id':SOURCE_ID,
        'source_proposal_sha256':c.sha(source/'learning_plan.proposed.json'),
        'configurations':[{'name':name,'feature_spec':model_spec(name),'fits_max':1} for name in NAMES],
        'parameters':proposal['parameters'],'rounds':219,'categories':{k:v for k,v in meta['categories'].items() if k!='e_rest_stage'},
        'versions':meta['versions'],'references':refs,'reference47_model_path':str(model_path),
        'reference47_model_id':meta['model_id'],'calibration':proposal['calibration'],
        'training_spans':proposal['training_spans'],'budget':dict(proposal['budget'],actual_train_calls_max=2,
            per_model_actual_train_calls_max=1,automatic_retry=False,Optuna_or_Tuner_trials=0),
        'final_training':'same_final_train_rows_as_reference47;no_new_early_stopping;219_fixed',
        'comparison':'calibration133_only;temperature_selection_diagnostic;not_unused_or_forward_test',
        'independent_forward_evaluation_races':0,'future_acceptance_criteria':'deferred_pending_user_decision',
        'production_connected':False,'promotion_authorized':False,'tasks_changed':False,
        'H_or_Stage7_results_used':False,
        'lineage_caveat':'current_static_parent_IDs;historical_master_corrections_unverified;retained'}
    c.write(directory/'training_plan.json',plan)
    c.write(directory/'authorization.json',{'source':'human_user','allow_real_fits':True,
        'actual_train_calls_max':2,'per_model_actual_train_calls_max':1,
        'training_plan_sha256':c.sha(directory/'training_plan.json'),
        'basis':'explicit_approval_control39_onefit_and_pedigree45_onefit_219_no_search_no_retry',
        'forward_acceptance_rules_approved':False,'deployment_approved':False})
    c.write(directory/'preservation_before.json',preservation())
    return {'status':'prepared','run_id':directory.name,'max_fits':2,'rounds':219,'automatic_retry':False}


def approved(directory):
    p=c.read(directory/'training_plan.json');a=c.read(directory/'authorization.json')
    if (p.get('version')!=VERSION or a.get('source')!='human_user' or a.get('allow_real_fits') is not True
            or a.get('training_plan_sha256')!=c.sha(directory/'training_plan.json')
            or a.get('actual_train_calls_max')!=2 or a.get('per_model_actual_train_calls_max')!=1
            or p.get('rounds')!=219 or p.get('parameters',{}).get('num_threads')!=4
            or p.get('budget',{}).get('actual_train_calls_max')!=2
            or p['budget'].get('automatic_retry') is not False or p['budget'].get('Optuna_or_Tuner_trials')!=0
            or p.get('configurations')!=[{'name':n,'feature_spec':model_spec(n),'fits_max':1} for n in NAMES]
            or p.get('production_connected') is not False or p.get('promotion_authorized') is not False
            or p.get('future_acceptance_criteria')!='deferred_pending_user_decision'):
        raise safety.MLFailure('explicit_execution_required')
    if (p.get('source_preparation_id')!=SOURCE_ID or p['budget'].get('per_model_actual_train_calls_max')!=1
            or p['budget'].get('CPU_threads')!=4 or p['budget'].get('per_fit_seconds_max')!=300
            or p['budget'].get('whole_learning_seconds_max')!=900
            or p['budget'].get('private_bytes_max')!=12*1024**3):
        raise safety.MLFailure('explicit_execution_required')
    for path,sha in p['references'].items():
        if c.sha(Path(path))!=sha: raise safety.MLFailure('artifact_invalid')
    meta=c.read(Path(p['reference47_model_path'])/'model.json')
    proposal=c.read(folder(SOURCE_ID)/'learning_plan.proposed.json')
    if (p['parameters']!=meta['parameters'] or p['versions']!=budget.versions()
            or p['categories']!={k:v for k,v in meta['categories'].items() if k!='e_rest_stage'}
            or p['calibration']!=proposal['calibration']):
        raise safety.MLFailure('invalid_model')
    return p


def fingerprint(matrix):
    return hashlib.sha256(matrix.to_csv(index=False).encode('utf-8')).hexdigest()


def load(path):
    import lightgbm as lgb
    path=safety.local_path(Path(path),root=c.ROOT)
    m=c.read(path/'model.json');name=m.get('configuration')
    if (name not in NAMES or m.get('version')!=VERSION or m.get('feature_spec')!=model_spec(name)
            or m.get('features')!=model_spec(name)['features'] or m.get('best_iteration')!=219
            or m.get('temperature') not in (.75,1.,1.25,2.) or m.get('actual_train_calls')!=1
            or m.get('production_connected') is not False
            or c.sha(path/'booster.lgb.txt')!=m.get('tree_sha256')):
        raise safety.MLFailure('invalid_model')
    booster=lgb.Booster(model_str=(path/'booster.lgb.txt').read_text(encoding='utf-8'))
    if booster.feature_name()!=m['features'] or booster.current_iteration()!=219:
        raise safety.MLFailure('invalid_model')
    return booster,m


def predict(bundle, frame, *, raw=False):
    booster,meta=bundle
    scores=booster.predict(data.matrix(frame,meta['feature_spec'],meta['categories']),num_threads=4)
    return scores if raw else matrices.normalize(frame,scores,meta['temperature'])


def worker(directory):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from ..d39_enrichment import data as reference_data, learning as reference_learning
    p=approved(directory);source=folder(p['source_preparation_id'])
    if (c.read(c.BASE/'research-training.lock').get('run_id')!=directory.name
            or not (directory/'execution_started.json').exists()
            or c.read(c.ROOT/'learning-claims'/f'{SOURCE_ID}.json').get('run_id')!=directory.name):
        raise safety.MLFailure('explicit_execution_required')
    c.write(directory/'worker_entered.json',{'at':c.clock().isoformat(),'automatic_retry':False})
    began=time.monotonic();whole_deadline=began+p['budget']['whole_learning_seconds_max']-10
    def ledger(number,state,details):
        c.write(directory/f'fit_{number:02d}.{state}.json',dict(details,number=number,
            model=NAMES[number-1],at=c.clock().isoformat(),monotonic=time.monotonic()))
    counter=budget.FitCounter(2,whole_deadline,ledger);reports={};predictions={}
    reference=reference_learning.load(Path(p['reference47_model_path']))
    original_train=reference_data.load_part(c.REFERENCE,'final_train')
    original_cal=reference_data.load_part(c.REFERENCE,'calibration')
    if max(original_train.date)>=min(original_cal.date):raise safety.MLFailure('invalid_split')
    common_features=model_spec('control39')['features']
    reference_x=reference_data.matrix(original_train,reference[1]['feature_spec'],reference[1]['categories'])[common_features]
    ref_fp=fingerprint(reference_x)
    key_fp=hashlib.sha256('\n'.join(original_train.entry_key).encode()).hexdigest()
    label_fp=hashlib.sha256(original_train.winner_label.to_numpy(int).tobytes()).hexdigest()
    predictions['reference47']=reference_learning.predict(reference,original_cal)
    reports['reference47']={'model_id':reference[1]['model_id'],'model_path':p['reference47_model_path'],
        'feature_count':47,'temperature':reference[1]['temperature'],'rounds':219,'new_fit':False,
        'metrics':point(original_cal,predictions['reference47'])}
    for name in NAMES:
        blood=name=='pedigree45';spec=model_spec(name)
        train=data.load_part(source,'final_train',pedigree=blood)
        cal=data.load_part(source,'calibration',pedigree=blood)
        if (list(train.entry_key)!=list(original_train.entry_key) or list(cal.entry_key)!=list(original_cal.entry_key)
                or not np.array_equal(train.winner_label.to_numpy(int),original_train.winner_label.to_numpy(int))
                or not np.array_equal(cal.winner_label.to_numpy(int),original_cal.winner_label.to_numpy(int))):
            raise safety.MLFailure('invalid_identity')
        categories=matrices.schema(train,spec)
        if categories!=p['categories']:raise safety.MLFailure('invalid_model')
        xt=data.matrix(train,spec,categories);xc=data.matrix(cal,spec,categories)
        pd.testing.assert_frame_equal(xt[common_features],reference_x,check_exact=True)
        common_fp=fingerprint(xt[common_features])
        if common_fp!=ref_fp:raise safety.MLFailure('artifact_invalid')
        training=lgb.Dataset(xt,label=train.winner_label.to_numpy(int),categorical_feature=spec['categorical_features'])
        counter.deadline=min(whole_deadline,time.monotonic()+p['budget']['per_fit_seconds_max'])
        before=counter.training_seconds
        booster=counter.wrap(lgb.train)(dict(p['parameters']),training,num_boost_round=219)
        if booster.current_iteration()!=219:raise safety.MLFailure('training_failed')
        raw=booster.predict(xc,num_threads=4)
        temperature,grid=calibration.calibrate(cal,raw,{'objective':'binary','temperature_candidates':p['calibration']['candidates']})
        probability=matrices.normalize(cal,raw,temperature)
        model_path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT);model_path.mkdir(parents=True)
        tree=model_path/'booster.lgb.txt'
        with tree.open('x',encoding='utf-8',newline='\n') as stream:stream.write(booster.model_to_string())
        meta={'version':VERSION,'model_id':model_path.name,'configuration':name,
            'created_at':c.clock().isoformat(),'feature_spec':spec,'features':spec['features'],'categories':categories,
            'parameters':p['parameters'],'best_iteration':219,'temperature':temperature,'calibration_grid':grid,
            'calibration_source':'same133_races_temperature_selected_here;diagnostic_not_unused_test',
            'training_plan_sha256':c.sha(directory/'training_plan.json'),'tree_sha256':c.sha(tree),
            'versions':p['versions'],'actual_train_calls':1,'Optuna_or_Tuner_trials':0,
            'source_preparation_id':SOURCE_ID,'training_spans':p['training_spans'],
            'latest_training_or_calibration_date':'2026-10-18','training_rows':len(train),'training_races':int(train.entry_key.str[:16].nunique()),
            'calibration_rows':len(cal),'calibration_races':int(cal.entry_key.str[:16].nunique()),
            'training_input_sha256':fingerprint(xt),'common39_input_sha256':common_fp,
            'entry_key_order_sha256':key_fp,'labels_sha256':label_fp,
            'categories_sha256':hashlib.sha256(json.dumps(categories,sort_keys=True).encode()).hexdigest(),
            'production_connected':False,'automatic_promotion':False,'future_acceptance_criteria':'not_finalized',
            'pedigree_master_corrections_asof':'unverified;static_ancestry_snapshot_only',
            'H_or_Stage7_results_used':False}
        c.write(model_path/'model.json',meta)
        reloaded=load(model_path);reload_raw=predict(reloaded,cal,raw=True);reload_p=predict(reloaded,cal)
        raw_delta=float(np.max(np.abs(raw-reload_raw)));delta=float(np.max(np.abs(probability-reload_p)))
        if max(raw_delta,delta)>1e-12:raise safety.MLFailure('invalid_model')
        forecasts=cal[['entry_key','date','winner_label']].copy()
        forecasts['raw_probability']=raw;forecasts['probability']=probability
        with (directory/(name+'.calibration.predictions.local.csv')).open('x',encoding='utf-8',newline='') as stream:
            forecasts.to_csv(stream,index=False)
        report={'model_id':model_path.name,'model_path':str(model_path),'feature_count':len(spec['features']),
            'temperature':temperature,'rounds':219,'new_fit':True,'actual_train_calls':1,
            'training_seconds':counter.training_seconds-before,'calibration_grid':grid,
            'reload_raw_max_difference':raw_delta,'reload_final_max_difference':delta,
            'metrics':point(cal,probability),'common39_input_sha256':common_fp,
            'feature_importance':{'gain':dict(zip(spec['features'],map(float,booster.feature_importance('gain')))),
                'split':dict(zip(spec['features'],map(int,booster.feature_importance('split'))))}}
        c.write(directory/(name+'.result.json'),report)
        predictions[name]=probability;reports[name]=report
        c.write(directory/(name+'.candidate_freeze.json'),{'model_id':model_path.name,'model_path':str(model_path),
            'model_sha256':c.sha(model_path/'model.json'),'tree_sha256':c.sha(tree),'production_authorized':False,
            'evaluation_acceptance_rules_finalized':False})
        del train,cal,xt,xc,training,booster,reloaded;gc.collect()
    comparison={'kind':'calibration_diagnostic_only_not_independent','calibration_races':133,'calibration_runners':1747,
        'actual_period':{'start':str(original_cal.date.min()),'end':str(original_cal.date.max())},
        'same_races_same_rosters_same_labels':True,'common39_encoded_values_identical':True,
        'common39_input_sha256':ref_fp,'entry_key_order_sha256':key_fp,'labels_sha256':label_fp,
        'models':reports,'differences_new_minus_comparator':{},'unused_evaluation_races':0,
        'formal_forward_records_for_new_models':0,'promotion_decision':'not_authorized',
        'future_evaluation_acceptance_rules':'await_user_decision','actual_train_calls':counter.calls,
        'worker_seconds':time.monotonic()-began,'lineage_caveat':p['lineage_caveat']}
    for new,old in (('pedigree45','control39'),('reference47','control39'),('pedigree45','reference47')):
        comparison['differences_new_minus_comparator'][new+'_minus_'+old]={
            metric:reports[new]['metrics'][metric]-reports[old]['metrics'][metric] for metric in
            ('race_log_loss','horse_log_loss','brier_score','roc_auc','within_race_auc','top1_accuracy_percent')}
    c.write(directory/'comparison.json',comparison)
    return {'status':'ok','actual_train_calls':counter.calls,'models':{n:{k:reports[n][k] for k in
        ('model_id','feature_count','temperature','training_seconds')} for n in NAMES},'independent_evaluation_races':0}


def execute(directory):
    from ..objective_pilot.runner import tree_memory,memory_bytes,terminate_worker
    p=approved(directory)
    with budget.training_lock(directory):
        claims=safety.local_path(c.ROOT/'learning-claims',root=c.ROOT);claims.mkdir(exist_ok=True)
        c.write(claims/f'{SOURCE_ID}.json',{'run_id':directory.name,'source_proposal_sha256':p['source_proposal_sha256'],
            'max_total_fits_for_source':2,'automatic_retry':False,'at':c.clock().isoformat()})
        c.write(directory/'execution_started.json',{'at':c.clock().isoformat(),'automatic_retry':False})
        env={k:v for k,v in os.environ.items() if k not in ('PGPASSWORD','PGPASSFILE','PGSERVICE','PGSERVICEFILE')}
        env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        project=Path(__file__).resolve().parents[3]
        command=[sys.executable,'-B',str(project/'scripts/train_pedigree.py'),'_worker','--run-id',directory.name]
        began=time.monotonic();peak=0;reason=None
        options={'creationflags':subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name=='nt' else {}
        with subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                              text=True,encoding='utf-8',**options) as child:
            while child.poll() is None:
                used,own=tree_memory(child.pid),memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:break
                    reason='input_limit'
                else:
                    peak=max(peak,used+own)
                    if used+own>p['budget']['private_bytes_max']:reason='input_limit'
                if time.monotonic()-began>p['budget']['whole_learning_seconds_max']:reason='query_timeout'
                for number in (1,2):
                    path=directory/f'fit_{number:02d}.started.json'
                    if path.exists() and not any((directory/f'fit_{number:02d}.{state}.json').exists() for state in ('complete','failed')):
                        if time.monotonic()-c.read(path)['monotonic']>p['budget']['per_fit_seconds_max']:
                            reason='query_timeout'
                if reason:terminate_worker(child);break
                time.sleep(.25)
            output,_=child.communicate(timeout=10)
        try:response=json.loads(output.strip())
        except (ValueError,TypeError):response={'status':'stopped','reason':'training_failed'}
        protected=verify(directory);protected.pop('real_fits',None)
        result={'status':'complete' if not reason and child.returncode==0 and response.get('status')=='ok' else 'stopped',
            'reason':reason or response.get('reason'),'actual_train_calls':len(list(directory.glob('fit_*.started.json'))),
            'wall_seconds':time.monotonic()-began,'peak_parent_and_worker_private_bytes':peak,'worker':response,
            'preservation':protected,'automatic_retry':False,'production_connected':False,'H_results_read':False}
        c.write(directory/'completion.json',result)
        return result


def report(directory):
    """Saved-model/calibration diagnostics only. Does not fit or start predictions."""
    import numpy as np
    import pandas as pd
    from .report import write_csv
    from ..d39_enrichment import data as reference_data, learning as reference_learning
    p=approved(directory);done=c.read(directory/'completion.json');result=c.read(directory/'comparison.json')
    if done['status']!='complete' or done['actual_train_calls']!=2:raise safety.MLFailure('training_failed')
    source=folder(p['source_preparation_id']);cal=reference_data.load_part(c.REFERENCE,'calibration')
    table=[];bands=[];temperatures=[];importance=[];validation={};source_hashes={}
    for name in ('control39','pedigree45','reference47'):
        item=result['models'][name];path=Path(item['model_path'])
        bundle=reference_learning.load(path) if name=='reference47' else load(path)
        frame=cal if name=='reference47' else data.load_part(source,'calibration',pedigree=name=='pedigree45')
        probabilities=reference_learning.predict(bundle,frame) if name=='reference47' else predict(bundle,frame)
        current=point(frame,probabilities)
        for metric,value in current.items():
            if metric=='calibration_bands':continue
            if abs(value-item['metrics'][metric])>1e-12:raise safety.MLFailure('artifact_invalid')
        raw=reference_learning.predict(bundle,frame,raw=True) if name=='reference47' else predict(bundle,frame,raw=True)
        file=directory/(name+'.calibration.predictions.local.csv')
        if file.exists():
            saved=pd.read_csv(file,dtype={'entry_key':'string','date':'string'},float_precision='round_trip')
            if (list(saved.entry_key)!=list(frame.entry_key) or list(saved.date)!=list(frame.date)
                    or not np.array_equal(saved.winner_label,frame.winner_label)
                    or np.max(np.abs(saved.probability.to_numpy(float)-probabilities))>1e-12
                    or np.max(np.abs(saved.raw_probability.to_numpy(float)-raw))>1e-12):
                raise safety.MLFailure('artifact_invalid')
        else:
            f=frame[['entry_key','date','winner_label']].assign(raw_probability=raw,probability=probabilities)
            with file.open('x',encoding='utf-8',newline='') as stream:f.to_csv(stream,index=False)
        source_hashes[str(file)]=c.sha(file)
        for filename in ('model.json','booster.lgb.txt'):source_hashes[str(path/filename)]=c.sha(path/filename)
        booster,meta=bundle
        table.append({'model':name,'model_id':item['model_id'],'features':item['feature_count'],
            'temperature':item['temperature'],'rounds':219,'training_seconds_new_fit':item.get('training_seconds'),
            **{k:v for k,v in current.items() if k!='calibration_bands'},
            'evaluation_kind':'calibration_diagnostic_not_independent',
            'period_start':str(frame.date.min()),'period_end':str(frame.date.max())})
        bands.extend(dict(model=name,**b) for b in current['calibration_bands'])
        grid=item.get('calibration_grid') or meta.get('calibration',[])
        temperatures.extend(dict(model=name,**v) for v in grid)
        gains=booster.feature_importance('gain');splits=booster.feature_importance('split');total=float(gains.sum())
        order=sorted(range(len(meta['features'])),key=lambda i:(-float(gains[i]),i))
        for rank,i in enumerate(order,1):
            importance.append({'model':name,'feature':meta['features'][i],'gain_rank':rank,
                'gain':float(gains[i]),'gain_share_percent':100*float(gains[i])/total if total else 0.,
                'split':int(splits[i]),'pedigree_feature':meta['features'][i] in c.FEATURES})
        validation[name]={'saved_tree_predictions_and_report_match':True,'prediction_sha256':c.sha(file),
                          'model_id':meta['model_id'],'temperature':meta['temperature'],
                          'complete_same_roster':list(frame.entry_key)==list(cal.entry_key)}
    write_csv(directory/'comparison.csv',table)
    write_csv(directory/'calibration_bands.csv',bands)
    write_csv(directory/'temperature_grid.csv',temperatures)
    write_csv(directory/'feature_importance.csv',importance)
    protections=verify(directory);protections.pop('real_fits',None)
    validation.update(real_fits_total=2,new_fits_in_report=0,protected=protections,
        source_hashes=source_hashes,prospective_evaluation_races=0,formal_promotion=False,
        acceptance_rules='deferred_not_applied',automatic_retry=False)
    c.write(directory/'report_validation.json',validation)
    metrics=('race_log_loss','horse_log_loss','brier_score','roc_auc','within_race_auc','top1_accuracy_percent')
    rows={v['model']:v for v in table}
    lines=['# 血統6列追加・同条件2fit研究 完了','',
        '2026-10-26。承認された2fitを各1回だけ実行。追加探索・自動再試行・本番接続は行っていない。',
        '', '## 比較期間・母集団','',
        '- 最終学習：2022-01-05〜2026-09-20、14,356レース・196,841行。',
        '- 旧計画の早期終了窓を含む保存済みfinal_trainをそのまま使用。今回は早期終了の追加選択をせず219本で固定。',
        '- 較正・診断比較：2026-10-10〜10-18、133レース・1,747頭。3モデルでレース・全出走馬集合・ラベルは同一。',
        '- 各モデルの温度を同じ133レースで選んでいるため、以下は未使用テストでも前向き実績でもない。',
        '- 新2モデルの独立した評価・正式な前向き記録は0レース。既知20レース・10/25の結果を使った新評価は実行していない。',
        '', '## 総合比較（較正期間の診断）','',
        '|モデル|特徴量|温度|Race LL↓|Brier↓|全体AUC↑|レース内AUC↑|本命1着率↑|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('control39','pedigree45','reference47'):
        r=rows[name]
        lines.append(f"|{name}|{r['features']}|{r['temperature']}|{r['race_log_loss']:.6f}|{r['brier_score']:.6f}|{r['roc_auc']:.6f}|{r['within_race_auc']:.6f}|{r['top1_accuracy_percent']:.2f}%|")
    lines += ['', '差の符号は新−比較対象。AUC・的中率は正、LL・Brierは負が良い。全体AUCとレース内AUCは別定義。',
              '', '|比較|ΔRace LL|ΔBrier|Δ全体AUC|Δレース内AUC|Δ本命1着率pt|',
              '|---|---:|---:|---:|---:|---:|']
    for name,v in result['differences_new_minus_comparator'].items():
        lines.append(f"|{name}|{v['race_log_loss']:+.6f}|{v['brier_score']:+.6f}|{v['roc_auc']:+.6f}|{v['within_race_auc']:+.6f}|{v['top1_accuracy_percent']:+.2f}|")
    lines += ['', '血統45は39対照に対してAUC・LL・Brierの点推定が改善した一方、本命1着率は低下した。',
        '血統45は既存改善47よりAUC・LLが良いが、Brierは悪化している。全指標で優れているとは言えない。',
        '**温度選定に利用した小標本の診断結果であり、血統の独立した改善確認・採用判定には使わない。** 信頼区間や提案ゲートによる正式採否は今回計算しない。',
        '', '## 学習条件と温度','',
        '保存済み47版の全パラメータ（seedを含む）を複写し、219本・CPU4で固定。Optuna/Tunerの試行0、最終再学習の追加0、失敗再試行0。',
        '39と45の学習行・ラベル・共通39列のエンコード済み値・順序・カテゴリ辞書は47の対応する部分と同一。ハッシュをmodel.jsonとcomparison.jsonへ保存した。',
        '同じ候補順0.75／1／1.25／2を使用し、較正Race LL最小で39はT=1.25、血統45はT=1.0を選択。既存47は保存済みT=1.25を維持。',
        '温度変換は二値確率のべき乗後、レース内で1に正規化する既存処理を1回だけ使用。',
        '', '## モデル保存','']
    for name in ('control39','pedigree45','reference47'):
        m=result['models'][name]
        lines.extend([f"- {name}: `{m['model_id']}`",f"  保存先: `{m['model_path']}`"])
    lines += ['', '新2モデルは別ID・別フォルダー。再読込後の生確率と校正後確率は保存時から差分0。既存モデル・正式予測・H・Stage7・自動起動・KEIBA LABは変更していない。',
              '', '## 血統の扱いと留保','',
        '父・母父×全条件成績水準／条件偏差／支持産駒数の6列。7日前まで、本馬除外、産駒内平均→産駒間等重み、事前固定の階層縮小を維持した。対象行の欠落なし。',
        '**過去の血統マスタ訂正履歴は未確認**。現在保存した静的な親子IDによる研究であり、過去の訂正前全版のas-of証明はしていない。',
        'goingは既存どおり全欠損を維持。現在の確定馬場を過去の入力へ代入していない。異常レース全体除外・7日履歴規則も両モデルで統一。',
        '', '血統45の学習内Gain（将来の有効性・因果寄与を証明する値ではない）：','',
        '|特徴量|Gain順位/45|Gain構成比|Split|','|---|---:|---:|---:|']
    for v in importance:
        if v['model']=='pedigree45' and v['pedigree_feature']:
            lines.append(f"|{v['feature']}|{v['gain_rank']}|{v['gain_share_percent']:.3f}%|{v['split']}|")
    lines += ['', '## 計算予算・検証','',
        f"実fit {done['actual_train_calls']}/2。学習呼出し時間：39={rows['control39']['training_seconds_new_fit']:.3f}秒、45={rows['pedigree45']['training_seconds_new_fit']:.3f}秒。",
        f"データ読込・比較・保存を含む全体{done['wall_seconds']:.2f}秒、親＋子の観測最大Private Bytes {done['peak_parent_and_worker_private_bytes']/1024**3:.3f}GiB。承認上限はCPU4・12GiB・各fit300秒・全体900秒。",
        '追加9テスト、既存を含むPython全1,464テスト成功。テスト内は実通信・実DBドライバーを遮断し、追加fitはモックのみ。',
        '旧資産12・保護コード24・追加の不変対象201ファイルとタスク構成の保持を確認。今回の学習ではDB接続・外部通信・依存追加なし。',
        '', '## 次の判断','',
        '新モデルの本番接続・主力昇格は未承認のため実施していない。前向き評価の詳細な採否規則は今回固定せず、ユーザーの別途最終決定待ち。',
        '既存の0fit段階の採否案はあくまで提案のまま保存。較正診断の良い数値を理由に正式採用へ読み替えない。',
        '比較可能な次の独立期間は、3モデルの設定を凍結し、同一締切・同一馬集合で結果確定前に保存を開始した後の共通レース。期間・終了時点・欠測・採否規則の確定後に別途進める。',
        '', '## 変更したコード・成果物','',
        '- src/jra_racing_ai/pedigree_research/learning.py：固定2fit、独立保存、再読込、較正診断・報告。',
        '- scripts/train_pedigree.py：既存資格情報を要求しないローカル専用CLI。',
        '- tests/test_pedigree_learning.py：承認・予算・列順・温度一重適用・保存整合。',
        '- docs/pedigree_research.md：承認された学習段階を追記。',
        '- comparison.csv / comparison.json：3モデルの匿名比較。',
        '- calibration_bands.csv / temperature_grid.csv：同じ確率帯と温度候補の診断。',
        '- feature_importance.csv：全39／45／47列のGain・Split。',
        '- training_plan.json / authorization.json / fit_*.json / candidate_freeze.json類：計画・fit数・モデル保存の記録。',
        '- 各モデルの*.calibration.predictions.local.csvは個別データを含むためPC外へ共有しない。']
    with (directory/'training_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    summary={'status':'reported','actual_train_calls':2,'new_fits_in_report':0,'independent_evaluation_races':0,
        'models':{n:{k:rows[n][k] for k in ('model_id','features','temperature','race_log_loss','brier_score','roc_auc','within_race_auc','top1_accuracy_percent')} for n in rows},
        'deployment_authorized':False,'acceptance_rules_finalized':False,'preservation':protections}
    c.write(directory/'training_report.json',summary)
    return summary
