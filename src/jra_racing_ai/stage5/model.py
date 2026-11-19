"""Thin numeric H extension; shared offset/sigmoid/normalizer and native saver."""
from pathlib import Path
from datetime import date
from uuid import uuid4
import numpy as np
from ..ml import safety,model as legacy
from ..research import model as old
from ..stage4 import data as source
from . import contracts as c,store as s


def matrix(frame):
    source.dates(frame);out=frame.loc[:,list(c.FEATURES)].astype(float)
    if np.isinf(out.to_numpy()).any():raise safety.MLFailure('invalid_input')
    return out


def predict(bundle,frame):
    residual=bundle['boosters']['win'].predict(matrix(frame),raw_score=True,num_threads=4)
    return old.normalize(frame,old.sigmoid(residual+old.offset(frame.market_probability.to_numpy(float))),1.)


def load(directory):
    import lightgbm as lgb
    meta=safety.read_json(directory/'model.json',root=c.ROOT)
    if (meta.get('format_version')!=5 or meta.get('model_kind')!='H_place_binary_market_offset_v1'
            or meta.get('features')!=list(c.FEATURES) or meta.get('categories')!={} or meta.get('formula_version')!=c.VERSION
            or meta.get('input_format')!='same_T10_O1_numeric_H8_plus_place3_v1' or meta.get('production_hr_settlement_enabled') is not False
            or meta.get('calibration')!={'temperature':1.,'method':'identity_after_external_offset'}
            or s.previous.comparable(meta.get('split'))!=s.previous.comparable(c.split(meta.get('evaluation_year')))):
        raise safety.MLFailure('invalid_model')
    tree=safety.local_path(directory/'win.lgb.txt',root=c.ROOT,exists=True)
    if s.sha(tree)!=meta['artifacts']['win.lgb.txt']:raise safety.MLFailure('artifact_invalid')
    return {'metadata':meta,'boosters':{'win':lgb.Booster(model_str=tree.read_text(encoding='utf-8'))}}


def fit(directory,year,parts,parameters):
    if not s.read(directory,'place_gate.json')['fit_allowed'] or not (directory/'training_plan.json').exists():raise safety.MLFailure('invalid_input')
    split=c.split(year);done=s.completed(directory,f'fit_{year}')
    if done:return load(directory/'models'/done['model_id']),done
    name=f'fit_{year}.started.json';started=directory/name;recovery=False
    if started.exists():
        init=s.read(directory,name);path=directory/'models'/init['model_id']
        if not (path/'model.json').exists():raise safety.MLFailure('training_failed')
        bundle=load(path);recovery=True
        if bundle['metadata']['evaluation_year']!=year:raise safety.MLFailure('invalid_model')
    else:
        if len(list(directory.glob('fit_*.started.json')))>=3:raise safety.MLFailure('input_limit')
        for label,f in zip(('train','validation','calibration'),parts):
            source.dates(f)
            if str(f.date.min())<split[label+'_start'] or str(f.date.max())>split[label+'_end']:raise safety.MLFailure('invalid_split')
        path=directory/'models'/uuid4().hex;path.mkdir(parents=True,exist_ok=False)
        s.write(directory,name,{'year':year,'model_id':path.name,'split':split,'status':'started','no_additional_fit_on_resume':True})
        import lightgbm as lgb
        train,validation,calibration=parts
        t=lgb.Dataset(matrix(train),label=train.winner_label.to_numpy(),init_score=old.offset(train.market_probability))
        v=lgb.Dataset(matrix(validation),label=validation.winner_label.to_numpy(),reference=t,init_score=old.offset(validation.market_probability))
        booster=lgb.train(parameters,t,num_boost_round=300,valid_sets=[v],callbacks=[lgb.early_stopping(20,verbose=False)])
        meta={'format_version':5,'model_kind':'H_place_binary_market_offset_v1','features':list(c.FEATURES),'categories':{},
            'formula_version':c.VERSION,'evaluation_year':year,'split':split,'input_format':'same_T10_O1_numeric_H8_plus_place3_v1',
            'calibration':{'temperature':1.,'method':'identity_after_external_offset'},'history_lag_days':7,
            'training_parameters':parameters,'best_iteration':booster.best_iteration,'max_rounds':300,'early_stopping_rounds':20,
            'library_versions':legacy.versions(),'production_hr_settlement_enabled':False,
            'training_plan_sha256':s.sha(directory/'training_plan.json'),'place_gate_sha256':s.sha(directory/'place_gate.json'),
            'place_mapping_sha256':s.sha(directory/'place_mapping.json'),
            'counts':{n:{'races':int(f.entry_key.str[:16].nunique()),'runners':len(f)} for n,f in zip(('train','validation','calibration'),parts)},
            'feature_missing':{n:{field:int(f[field].isna().sum()) for field in c.FEATURES} for n,f in zip(('train','validation','calibration'),parts)}}
        bundle={'metadata':meta,'boosters':{'win':booster}};legacy.save_models(bundle,path,root=c.ROOT)
    before=predict(bundle,parts[1]);loaded=load(path);after=predict(loaded,parts[1]);difference=float(np.max(np.abs(before-after)))
    if difference>1e-12:raise safety.MLFailure('artifact_invalid')
    summary={'year':year,'model_id':path.name,'best_iteration':loaded['metadata']['best_iteration'],'split':split,
        'roundtrip_max_probability_difference':difference,'resumed_without_refitting':recovery,'new_fit_count':1,
        'model_sha256':s.sha(path/'model.json'),'tree_sha256':s.sha(path/'win.lgb.txt')}
    s.complete(directory,f'fit_{year}',[path/'model.json',path/'win.lgb.txt'],summary)
    return loaded,summary
