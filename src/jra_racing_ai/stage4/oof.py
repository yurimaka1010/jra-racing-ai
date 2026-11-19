from datetime import date,timedelta
import json
import numpy as np
import pandas as pd
from ..ml import safety
from ..research import model as old,experiment as exp
from ..stage3 import scoring
from . import contracts as c,store as s,data,models


def validate_rows(f,summary,year):
    if year not in range(2021,2026) or s.comparable(summary['split'])!=s.comparable(c.h_split(year)):
        raise safety.MLFailure('invalid_split')
    data.dates(f)
    if f.entry_key.duplicated().any() or not (f.date.str[:4]==str(year)).all():raise safety.MLFailure('invalid_split')
    if not (f.source_H_model_id==summary['model_id']).all():raise safety.MLFailure('invalid_model')
    for name in ('train_end','validation_end','calibration_end'):
        if not (f[name]==c.h_split(year)[name]).all() or f[name].max()>=f'{year}-01-01':raise safety.MLFailure('invalid_split')
    expected=[(date.fromisoformat(str(d))-timedelta(days=7)).isoformat() for d in f.date]
    if not (f.feature_history_cutoff==expected).all():raise safety.MLFailure('invalid_split')


def csv_write(path,frame):
    safety.local_path(path,root=c.ROOT)
    if path.exists():raise safety.MLFailure('artifact_exists')
    frame.to_csv(path,index=False,mode='x',lineterminator='\n')


def read(directory,start,end):
    c.guard(start,end);result=[]
    for year in range(start.year,end.year+1):
        if year not in range(2021,2026):raise safety.MLFailure('invalid_split')
        folder=directory/'oof';meta=s.completed(folder,str(year))
        if not meta:raise safety.MLFailure('artifact_invalid')
        f=pd.read_csv(folder/f'{year}.csv',dtype={'entry_key':'string','date':'string'},float_precision='round_trip')
        validate_rows(f,meta['summary'],year)
        result.append(f.loc[(f.date>=start.isoformat())&(f.date<=end.isoformat())])
    return pd.concat(result,ignore_index=True)


def lineage(directory):
    """Explicit per-runner provenance for BOTH OOF sides; no labels are exported."""
    folder=directory/'lineage';folder.mkdir(exist_ok=True);done=s.completed(directory,'lineage')
    if done:return done['summary']
    catalog=[];files=[]
    for year in range(2021,2026):
        unit=s.completed(folder,str(year))
        if unit:catalog.append(unit['summary']);files.append(folder/f'{year}.complete.json');continue
        start,end=date(year,1,1),min(date(year,12,31),c.END)
        h=read(directory,start,end);a=data.ability(directory,start,end)
        a_meta=s.ref_json(directory,c.S3/'oof'/f'{year}.complete.json')['summary']
        for name in ('train_end','validation_end','calibration_end'):
            if a_meta['split'][name]>=start.isoformat():raise safety.MLFailure('invalid_split')
        result=data.merge_probability(h,a[['entry_key','ability_probability','ability_model_id']], 'ability_probability')
        for name in ('train_end','validation_end','calibration_end'):result['ABCD_'+name]=a_meta['split'][name]
        result['H_OOF_manifest_sha256']=s.digest(directory/'oof'/f'{year}.complete.json')
        result['ABCD_OOF_manifest_sha256']=safety.digest(c.S3/'oof'/f'{year}.complete.json',root=c.S3)
        csv_write(folder/f'{year}.csv',result)
        summary={'year':year,'rows':len(result),'both_model_ids_and_all_deadlines_and_hashes':True,'labels_exported':False}
        s.complete(folder,str(year),[folder/f'{year}.csv'],summary);catalog.append(summary);files.append(folder/f'{year}.complete.json')
    summary={'years':catalog,'source_predictions_in_sample':False,'no_future_year_model':True}
    s.complete(directory,'lineage',files,summary);return summary


def combined(directory,start,end):
    f=data.market_frame(directory,start,end)
    h=read(directory,start,end);a=data.ability(directory,start,end)
    f=data.merge_probability(f,h[['entry_key','history_probability','source_H_model_id']], 'history_probability')
    return data.merge_probability(f,a[['entry_key','ability_probability','ability_model_id']], 'ability_probability')


def run(directory):
    if not s.completed(directory,'reproduce'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'oof')
    if done:return done['summary']
    folder=directory/'oof';folder.mkdir(exist_ok=True);origins=[];files=[]
    for year in range(2021,2026):
        completed=s.completed(folder,str(year))
        if completed:
            origins.append(completed['summary']);files.append(folder/f'{year}.complete.json');continue
        split=c.h_split(year);start,end=map(date.fromisoformat,split['evaluation'][0])
        if year in (2021,2022):
            bundle,source=models.fit(directory,'H',year,split,models.parts(directory,split))
            model_id=source['model_id'];model_path=directory/'models'/model_id;root=c.ROOT
        else:
            previous=data.trial(directory,year,'H');model_id=previous['model_id'];model_path=c.S2/'models'/model_id;root=c.S2
            bundle=old.load(model_path,root=root)
        if bundle['metadata']['split']['calibration_end']>=start.isoformat():raise safety.MLFailure('invalid_split')
        frame=data.market_frame(directory,start,end);values=models.predict(bundle,frame)
        # Every primary OOF row must have an original ABCD OOF counterpart and identical rosters.
        a=data.ability(directory,start,end);joined=data.merge_probability(frame,a[['entry_key','ability_probability']], 'ability_probability')
        if year in c.YEARS:
            subset=frame.loc[frame.date<=f'{year}-09-30'].copy()
            mask=frame.date<=f'{year}-09-30'
            x,y=scoring.align(exp.predictions(subset,values[mask]),data.source_predictions(directory,year,'H'))
            if max(abs(x[r].probabilities[h]-y[r].probabilities[h]) for r in x for h in x[r].probabilities)>1e-12:
                raise safety.MLFailure('artifact_invalid')
            current=exp.predictions(subset,joined.loc[mask.to_numpy(),'ability_probability'].to_numpy());race_set={p.race for p in current}
            x,y=scoring.align(current,[p for p in data.source_predictions(directory,year,'A') if p.race in race_set])
            if max(abs(x[r].probabilities[h]-y[r].probabilities[h]) for r in x for h in x[r].probabilities)>1e-12:
                raise safety.MLFailure('artifact_invalid')
        meta_sha=safety.digest(model_path/'model.json',root=root);tree_sha=safety.digest(model_path/'win.lgb.txt',root=root)
        rows=frame[['entry_key','date']].copy();rows['history_probability']=values;rows['source_H_model_id']=model_id
        for name in ('train_end','validation_end','calibration_end'):rows[name]=split[name]
        rows['source_model_manifest_sha256']=meta_sha;rows['source_tree_sha256']=tree_sha
        rows['reference_manifest_sha256']=s.digest(directory/'references.json')
        rows['feature_history_cutoff']=[(date.fromisoformat(str(d))-timedelta(days=7)).isoformat() for d in rows.date]
        csv_write(folder/f'{year}.csv',rows)
        summary=dict(year=year,model_id=model_id,new_fit=year<2023,rows=len(rows),races=int(rows.entry_key.str[:16].nunique()),
            split=split,model_manifest_sha256=meta_sha,tree_sha256=tree_sha,temperature=1.,source_features=list(c.H_FEATURES),
            ABCD_OOF_manifest_sha256=safety.digest(c.S3/'oof'/f'{year}.complete.json',root=c.S3),
            first_date=str(rows.date.min()),last_date=str(rows.date.max()),latest_label_available=split['calibration_end'])
        s.complete(folder,str(year),[folder/f'{year}.csv'],summary);origins.append(summary);files.append(folder/f'{year}.complete.json')
        print(json.dumps(dict(status='running',stage='oof',year=year,rows=len(rows),races=summary['races'],new_fit=year<2023)),flush=True)
    s.write(directory,'oof_origins.json',origins);files.append(directory/'oof_origins.json')
    summary={'years':origins,'new_LightGBM_fits':2,'reused_annual_H_models':3,'ABCD_source':'unchanged_Stage3_fixed_ABCD_OOF'}
    s.complete(directory,'oof',files,summary);return summary
