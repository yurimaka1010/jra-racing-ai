"""Preserve the clean 47-column cohort; only the two reviewed connections change."""
from datetime import date
import time
from uuid import uuid4
from ..ml import safety
from ..d39_generation import data as history_data
from ..d39_next import data as previous
from ..objective_pilot.data import read_frame
from . import contracts as c, sources
from .features import Builder


def matrix(frame,spec,categories):
    if spec!=c.spec():raise safety.MLFailure('invalid_model')
    return previous.matrix(frame,c.prior.spec(),categories)


def prepare(directory):
    import pandas as pd
    import numpy as np
    began=time.monotonic();sources.collect(directory)
    index,resolver,proof=sources.load(directory)
    builder=Builder(index,resolver);frames=[]
    p=c.read(directory/'plan.json')
    for rows in history_data.iter_days(c.RAW_PARENT):
        if time.monotonic()-began>600:raise safety.MLFailure('query_timeout')
        x,meta=builder.transform_day(rows)
        if meta[0]['date']<'2022-01-01':continue
        if meta[0]['date']>p['source_end']:raise safety.MLFailure('invalid_split')
        frames.extend(dict(entry_key=''.join(m['key']),date=m['date'],winner_label=m['winner_label'],**v)
            for v,m in zip(x,meta) if m['eligible'])
    fresh=pd.DataFrame(frames).sort_values('entry_key').reset_index(drop=True)
    previous.validate_frame(fresh)
    manifest=c.read(c.PARENT/'data_manifest.json');parts={};oldmeta=c.read(__import__('pathlib').Path(c.read(c.PARENT/'candidate_freeze.json')['model_path'])/'model.json')
    for name,item in manifest['parts'].items():
        old=previous.load_part(c.PARENT,name).sort_values('entry_key').reset_index(drop=True)
        span=p['spans'][name]
        frame=fresh.loc[fresh.date.between(span['start'],span['end'])].reset_index(drop=True)
        if list(frame.entry_key)!=list(old.entry_key) or list(frame.winner_label)!=list(old.winner_label):
            raise safety.MLFailure('artifact_invalid')
        # Existing 45 features must retain exactly the same encoded values.
        other=[v for v in c.spec()['features'] if v not in ('going','previous_time_seconds')]
        a=previous.matrix(frame,c.prior.spec(),oldmeta['categories'])
        b=previous.matrix(old,c.prior.spec(),oldmeta['categories'])
        pd.testing.assert_frame_equal(a[other],b[other],check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
        path=directory/(name+'.local.csv')
        with path.open('x',encoding='utf-8',newline='') as stream:frame.to_csv(stream,index=False)
        missing={f:int(frame[f].isna().sum()) for f in c.spec()['features']}
        parts[name]={'file':path.name,'sha256':c.sha(path),'rows':len(frame),
            'races':int(frame.entry_key.str[:16].nunique()),'missing':missing,
            'original_entry_keys_and_labels_identical':True,'other_45_features_unchanged':True,
            'previous_missing':item['missing']}
    value={'version':c.VERSION,'feature_spec':c.spec(),'dataset_id':uuid4().hex,'parts':parts,
        'parent_manifest_sha256':c.sha(c.PARENT/'data_manifest.json'),
        'source_manifest_sha256':c.sha(directory/'source_manifest.json'),
        'archived_time_sources':proof,'history_lag_days':7,
        'historical_final_RA_going_used':False,'going_training_available':parts['final_train']['missing']['going']<parts['final_train']['rows'],
        'whole_race_clean_mask_preserved':True,'stage7_outcomes_read':False}
    c.write(directory/'data_manifest.json',value)
    return {'status':'prepared','run_id':directory.name,'real_fits':0,
        'elapsed_seconds':time.monotonic()-began,'parts':{n:{k:v[k] for k in ('rows','races')} for n,v in parts.items()},
        'feature_missing':{n:{f:parts[n]['missing'][f] for f in ('previous_time_seconds','going')} for n in parts},
        'going_training_available':value['going_training_available']}


def load_part(directory,name):
    manifest=c.read(directory/'data_manifest.json')
    if manifest.get('feature_spec')!=c.spec():raise safety.MLFailure('artifact_invalid')
    item=manifest['parts'][name];path=directory/item['file']
    if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
    frame=read_frame(path);previous.validate_frame(frame)
    if len(frame)!=item['rows'] or not item['original_entry_keys_and_labels_identical']:
        raise safety.MLFailure('artifact_invalid')
    return frame
