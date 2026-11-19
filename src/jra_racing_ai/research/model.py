"""Versioned binary LightGBM, with an explicitly external logit-market offset."""
from datetime import date,datetime
import math
from pathlib import Path
from uuid import uuid4

from ..ml import model as legacy,safety
from ..win_sim import core
from . import contracts as c,io


def schema(frame,spec):
    import pandas as pd
    return {f:['__MISSING__','__UNKNOWN__']+sorted({'v:'+str(v) for v in frame[f] if not pd.isna(v)})
            for f in spec['categorical_features']}


def matrix(frame,spec,categories):
    import pandas as pd
    c.validate_spec(spec)
    if set(categories)!=set(spec['categorical_features']):raise safety.MLFailure('invalid_model')
    result=frame.loc[:,spec['features']].copy()
    for name in spec['features']:
        if name in categories:
            cats=categories[name]
            if cats[:2]!=['__MISSING__','__UNKNOWN__'] or len(set(cats))!=len(cats):
                raise safety.MLFailure('invalid_model')
            allowed=set(cats)
            values=['__MISSING__' if pd.isna(v) else 'v:'+str(v) if 'v:'+str(v) in allowed else '__UNKNOWN__' for v in frame[name]]
            result[name]=pd.Categorical(values,categories=cats)
        else:
            result[name]=pd.to_numeric(result[name],errors='raise').astype('float64')
    return result


def offset(probabilities):
    import numpy as np
    q=np.asarray(probabilities,dtype=float)
    if not np.isfinite(q).all() or not ((q>0)&(q<1)).all():raise safety.MLFailure('invalid_input')
    return np.log(q)-np.log1p(-q)


def sigmoid(raw):
    import numpy as np
    r=np.asarray(raw,dtype=float)
    if not np.isfinite(r).all():raise safety.MLFailure('invalid_input')
    # Stable sigmoid without overflow or a new library dependency.
    out=np.empty_like(r);mask=r>=0
    out[mask]=1/(1+np.exp(-r[mask]));ex=np.exp(r[~mask]);out[~mask]=ex/(1+ex)
    return out


def normalize(frame,probabilities,temperature=1.):
    import numpy as np
    p=np.asarray(probabilities,dtype=float)
    if len(p)!=len(frame) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise safety.MLFailure('invalid_input')
    if temperature!=1.:p=p**(1/temperature)
    _,codes=np.unique(frame['entry_key'].str[:16].to_numpy(dtype=str),return_inverse=True)
    sums=np.bincount(codes,weights=p)
    if not (sums>0).all():raise safety.MLFailure('invalid_input')
    return p/sums[codes]


def uncalibrated(bundle,frame):
    meta=bundle['metadata'];spec=meta['feature_spec']
    x=matrix(frame,spec,meta['categories'])
    if spec['mode']=='market_offset':
        residual=bundle['boosters']['win'].predict(x,raw_score=True,num_threads=4)
        return sigmoid(residual+offset(frame['market_probability'].to_numpy(dtype=float)))
    return bundle['boosters']['win'].predict(x,num_threads=4)


def predict(bundle,frame):
    return normalize(frame,uncalibrated(bundle,frame),bundle['metadata']['calibration']['temperature'])


def predict_inputs(bundle,rows,*,history=()):
    """Ability inference shares the causal generator used for the training cache."""
    import pandas as pd
    from .features import build
    from ..ml.features import entry_identity
    spec=bundle['metadata']['feature_spec']
    if spec['mode']!='ability':raise safety.MLFailure('invalid_input')
    x=pd.DataFrame(build(rows,history=history),columns=c.ALL_FEATURES)
    x['entry_key']=[''.join(entry_identity(r)[0]) for r in rows]
    return predict(bundle,x)


def fit(train,validation,calibration,spec,*,split,parameters):
    import numpy as np
    import lightgbm as lgb
    from .experiment import predictions,winners
    for part in (train,validation,calibration):
        if not len(part) or any(date.fromisoformat(str(v))>=c.SEALED_START for v in part['date']):
            raise safety.MLFailure('invalid_split')
    if not max(train['date'])<min(validation['date']) or not max(validation['date'])<min(calibration['date']):
        raise safety.MLFailure('invalid_split')
    categories=schema(train,spec)
    xt=matrix(train,spec,categories);xv=matrix(validation,spec,categories)
    anchor=spec['mode']=='market_offset'
    if anchor and min(train['date'])<'2020-01-01':raise safety.MLFailure('invalid_split')
    t=lgb.Dataset(xt,label=train['winner_label'].to_numpy(),categorical_feature=spec['categorical_features'],
        init_score=offset(train['market_probability']) if anchor else None,free_raw_data=True)
    v=lgb.Dataset(xv,label=validation['winner_label'].to_numpy(),reference=t,
        init_score=offset(validation['market_probability']) if anchor else None,free_raw_data=True)
    evaluation_history={}
    booster=lgb.train(parameters,t,num_boost_round=300,valid_sets=[v],
        callbacks=[lgb.early_stopping(20,verbose=False),lgb.record_evaluation(evaluation_history)])
    metadata={'format_version':2,'model_kind':'versioned_binary_lgbm_v2','targets':['win'],
        'features':spec['features'],'categories':categories,'feature_spec':spec,
        'input_format':c.INPUT_VERSION,'formula_version':c.FORMULA_VERSION,'split':split,
        'calibration':{'method':'identity_after_external_offset' if anchor else 'independent_power_temperature',
                       'temperature':1.,'source':'calibration_period_only'},
        'training_parameters':parameters,'max_rounds':300,'early_stopping_rounds':20,
        'best_iteration':booster.best_iteration,'library_versions':legacy.versions(),
        'production_hr_settlement_enabled':False,'created_at_jst':datetime.now(core.JST).isoformat(),
        'missing_features':{name:{f:int(part[f].isna().sum()) for f in spec['features']}
                            for name,part in (('train',train),('validation',validation),('calibration',calibration))},
        'counts':{name:{'rows':len(part),'races':part['entry_key'].str[:16].nunique()}
                  for name,part in (('train',train),('validation',validation),('calibration',calibration))}}
    metadata['counts']={n:{k:int(v) for k,v in fields.items()} for n,fields in metadata['counts'].items()}
    bundle={'boosters':{'win':booster},'metadata':metadata}
    raw=uncalibrated(bundle,calibration)
    if not anchor:
        selected=core.fit_temperature(predictions(calibration,raw),winners(calibration),
            model_fit_end=split['validation_end'],test_start=split['evaluation'][0][0])
        metadata['calibration'].update(selected)
    else:
        metadata['calibration'].update(races=int(calibration['entry_key'].str[:16].nunique()),
                                      start=min(calibration['date']),end=max(calibration['date']))
    return bundle


def save(bundle,directory):
    path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT)
    path.mkdir(parents=True,exist_ok=False)
    metadata=legacy.save_models(bundle,path,root=c.ROOT)
    return path,metadata


def load(directory,*,root=None):
    root=c.ROOT if root is None else root
    metadata=safety.read_json(directory/'model.json',root=root)
    if metadata.get('format_version')==1:
        return legacy.load_models(directory,root=root,required_targets=('win',))
    import lightgbm as lgb
    if (metadata.get('format_version')!=2 or metadata.get('model_kind')!='versioned_binary_lgbm_v2'
            or metadata.get('features')!=metadata.get('feature_spec',{}).get('features')
            or metadata.get('formula_version')!=c.FORMULA_VERSION):
        raise safety.MLFailure('invalid_model')
    c.validate_spec(metadata['feature_spec'])
    if metadata['feature_spec']['mode']=='market_offset' and metadata['calibration']['temperature']!=1.:
        raise safety.MLFailure('invalid_model')
    path=safety.local_path(directory/'win.lgb.txt',root=root,exists=True)
    if safety.digest(path,root=root)!=metadata['artifacts']['win.lgb.txt']:
        raise safety.MLFailure('artifact_invalid')
    return {'metadata':metadata,'boosters':{'win':lgb.Booster(model_str=path.read_text(encoding='utf-8'))}}
