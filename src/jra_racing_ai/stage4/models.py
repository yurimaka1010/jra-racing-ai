"""New H fits reuse the old contract; H60 has a separate, strict numeric format."""
from datetime import date,datetime
from uuid import uuid4
import numpy as np
from ..ml import safety,model as legacy
from ..research import model as old,contracts as oc
from ..win_sim import core
from . import contracts as c,store as s,data


def parts(directory,split,augment=None):
    values=[]
    for name in ('train','validation','calibration'):
        start,end=(date.fromisoformat(split[name+'_start']),date.fromisoformat(split[name+'_end']))
        c.guard(start,end);f=data.market_frame(directory,start,end)
        if augment is not None:f=augment(f)
        if not len(f):raise safety.MLFailure('insufficient_history')
        values.append(f)
    if not max(values[0].date)<min(values[1].date) or not max(values[1].date)<min(values[2].date):
        raise safety.MLFailure('invalid_split')
    return values


def matrix(frame):
    result=frame.loc[:,list(c.H_FEATURES+c.EXTRA)].astype(float)
    if np.isinf(result.to_numpy()).any():raise safety.MLFailure('invalid_input')
    return result


def predict(bundle,frame):
    data.dates(frame)
    if bundle['metadata'].get('model_kind')!='stage4_H60_v1':return old.predict(bundle,frame)
    raw=bundle['boosters']['win'].predict(matrix(frame),raw_score=True,num_threads=4)
    return old.normalize(frame,old.sigmoid(raw+old.offset(frame.market_probability.to_numpy())),1.)


def load(path):
    meta=safety.read_json(path/'model.json',root=c.ROOT)
    if meta.get('model_kind')!='stage4_H60_v1':return old.load(path,root=c.ROOT)
    if (meta.get('format_version')!=4 or meta.get('features')!=list(c.H_FEATURES+c.EXTRA)
            or meta.get('formula_version')!=c.H60_VERSION or meta.get('calibration')!={'temperature':1.,'method':'identity_after_offset'}
            or meta.get('input_format')!='stage4_numeric_market_11_v1' or meta.get('categories')!={}):
        raise safety.MLFailure('invalid_model')
    if s.comparable(meta['split'])!=s.comparable(c.h_split(int(meta['fold_id']))):raise safety.MLFailure('invalid_model')
    tree=safety.local_path(path/'win.lgb.txt',root=c.ROOT,exists=True)
    if safety.digest(tree,root=c.ROOT)!=meta['artifacts']['win.lgb.txt']:raise safety.MLFailure('artifact_invalid')
    import lightgbm as lgb
    return {'metadata':meta,'boosters':{'win':lgb.Booster(model_str=tree.read_text(encoding='utf-8'))}}


def fit(directory,kind,year,split,frames):
    if kind not in ('H','H60') or (kind=='H' and year not in (2021,2022)) or (kind=='H60' and year not in c.YEARS):
        raise safety.MLFailure('invalid_input')
    if split!=c.h_split(year):raise safety.MLFailure('invalid_split')
    folder=directory/'fits';folder.mkdir(exist_ok=True);unit=f'{year}_{kind}'
    done=s.completed(folder,unit)
    if done:return load(directory/'models'/done['summary']['model_id']),done['summary']
    train,validation,calibration=frames
    for name,f in zip(('train','validation','calibration'),frames):
        c.guard(date.fromisoformat(str(f.date.min())),date.fromisoformat(str(f.date.max())))
        if str(f.date.min())<split[name+'_start'] or str(f.date.max())>split[name+'_end']:raise safety.MLFailure('invalid_split')
    origin=s.provenance(directory,kind,year,split=split)
    if (folder/(unit+'.started.json')).exists():
        # A fully serialized fit may be recovered after a loader/summary interruption,
        # but a missing/incomplete model must never trigger an extra training run.
        candidates=[]
        for manifest in (directory/'models').glob('*/model.json'):
            meta=safety.read_json(manifest,root=c.ROOT);source=meta.get('stage4_origin',{})
            if source.get('config_id')==kind and source.get('fold_id')==str(year):candidates.append(manifest.parent)
        if len(candidates)!=1:raise safety.MLFailure('training_failed')
        path=candidates[0];reloaded=load(path);metadata=reloaded['metadata']
        summary=dict(year=year,kind=kind,model_id=path.name,best_iteration=metadata['best_iteration'],
            reload_max_difference=None,recovered_serialized_fit_without_refitting=True,
            counts={n:{'rows':len(p),'races':int(p.entry_key.str[:16].nunique())} for n,p in zip(('train','validation','calibration'),frames)},
            model_manifest_sha256=s.digest(path/'model.json'),tree_sha256=s.digest(path/'win.lgb.txt'))
        s.complete(folder,unit,[path/'model.json',path/'win.lgb.txt'],summary);return reloaded,summary
    if len(list(folder.glob('*.started.json')))>=5:raise safety.MLFailure('training_failed')
    s.write(folder,unit+'.started.json',dict(origin,status='started'))
    params=s.ref_json(directory,c.S2/'plan.json')['anchor_parameters']
    if kind=='H':
        bundle=old.fit(train,validation,calibration,oc.spec(anchor=True,ability=False,history=True),split=split,parameters=params)
    else:
        import lightgbm as lgb
        t=lgb.Dataset(matrix(train),label=train.winner_label.to_numpy(),init_score=old.offset(train.market_probability))
        v=lgb.Dataset(matrix(validation),label=validation.winner_label.to_numpy(),reference=t,init_score=old.offset(validation.market_probability))
        booster=lgb.train(params,t,num_boost_round=300,valid_sets=[v],callbacks=[lgb.early_stopping(20,verbose=False)])
        bundle={'boosters':{'win':booster},'metadata':dict(format_version=4,model_kind='stage4_H60_v1',
            features=list(c.H_FEATURES+c.EXTRA),categories={},formula_version=c.H60_VERSION,input_format='stage4_numeric_market_11_v1',
            calibration={'temperature':1.,'method':'identity_after_offset'},split=split,fold_id=year,
            training_parameters=params,max_rounds=300,early_stopping_rounds=20,best_iteration=booster.best_iteration,
            production_hr_settlement_enabled=False,library_versions=legacy.versions(),created_at_jst=datetime.now(core.JST).isoformat(),
            missing_features={n:{f:int(p[f].isna().sum()) for f in c.H_FEATURES+c.EXTRA} for n,p in zip(('train','validation','calibration'),frames)})}
    path=directory/'models'/uuid4().hex;path.mkdir(parents=True,exist_ok=False)
    bundle['metadata']['stage4_origin']=origin
    metadata=legacy.save_models(bundle,path,root=c.ROOT);reloaded=load(path)
    before=predict(bundle,validation);after=predict(reloaded,validation)
    delta=float(np.max(np.abs(before-after)))
    if delta>1e-12:raise safety.MLFailure('artifact_invalid')
    summary=dict(year=year,kind=kind,model_id=path.name,best_iteration=metadata['best_iteration'],
        reload_max_difference=delta,counts={n:{'rows':len(p),'races':int(p.entry_key.str[:16].nunique())} for n,p in zip(('train','validation','calibration'),frames)},
        model_manifest_sha256=s.digest(path/'model.json'),tree_sha256=s.digest(path/'win.lgb.txt'))
    s.complete(folder,unit,[path/'model.json',path/'win.lgb.txt'],summary)
    return reloaded,summary
