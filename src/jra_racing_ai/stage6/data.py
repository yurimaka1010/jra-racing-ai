"""Read only three C-cache fields and join the existing T10 horse roster exactly."""
import csv
from datetime import date
import numpy as np
import pandas as pd
from ..ml import safety
from ..research.experiment import key_parts
from ..stage4 import data as old,store as old_store
from . import contracts as c


def identities(frame):
    if not len(frame) or not {'entry_key','date'}<=set(frame):raise safety.MLFailure('invalid_input')
    if frame.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    for key,day in zip(frame.entry_key,frame.date):
        parts=key_parts(str(key));d=date.fromisoformat(str(day));c.guard(d,d)
        if str(key)[:8]!=d.strftime('%Y%m%d') or not 1<=int(parts[-1])<=28:raise safety.MLFailure('invalid_identity')


def read_style(start,end):
    c.guard(start,end);rows=[];fields=('entry_key','date',c.SOURCE_COLUMN)
    for year in range(start.year,end.year+1):
        path=c.S2/'features'/f'{year}.csv';old_store.check_reference(c.S4,path)
        with path.open(encoding='utf-8',newline='') as stream:
            header=next(csv.reader([stream.readline()]))
            if header[:2]!=['entry_key','date'] or any(header.count(k)!=1 for k in fields):raise safety.MLFailure('invalid_input')
            indices=[header.index(k) for k in fields];last=None
            for line in stream:
                prefix=line.split(',',2)
                if len(prefix)<3:raise safety.MLFailure('invalid_input')
                day=date.fromisoformat(prefix[1])
                if last is not None and day<last:raise safety.MLFailure('invalid_split')
                last=day
                if day>min(end,c.END):break  # Stop before interpreting any post-boundary fields.
                if day<start:continue
                values=next(csv.reader([line]))
                if len(values)!=len(header):raise safety.MLFailure('invalid_input')
                rows.append(tuple(values[i] for i in indices))
    frame=pd.DataFrame(rows,columns=fields);identities(frame)
    try:frame[c.SOURCE_COLUMN]=pd.to_numeric(frame[c.SOURCE_COLUMN].replace('',np.nan),errors='raise').astype(float)
    except (TypeError,ValueError):raise safety.MLFailure('invalid_input') from None
    return frame


def augment(frame,style):
    identities(frame);identities(style)
    if c.SOURCE_COLUMN not in style:raise safety.MLFailure('invalid_input')
    if any(k in frame for k in c.STYLE+c.PACE):raise safety.MLFailure('invalid_input')
    races=set(frame.entry_key.str[:16]);selected=style.loc[style.entry_key.str[:16].isin(races),['entry_key','date',c.SOURCE_COLUMN]]
    if set(selected.entry_key)!=set(frame.entry_key):raise safety.MLFailure('invalid_identity')
    values=frame[['entry_key','date']].merge(selected,on='entry_key',how='left',sort=False,validate='one_to_one',suffixes=('_base','_style'))
    if not np.array_equal(values.date_base.to_numpy(str),values.date_style.to_numpy(str)):raise safety.MLFailure('invalid_identity')
    try:s=values[c.SOURCE_COLUMN].to_numpy(dtype=float)
    except (TypeError,ValueError):raise safety.MLFailure('invalid_input') from None
    missing=np.isnan(s);known=~missing
    if (not np.isfinite(s[known]).all()) or ((s[known]<0)|(s[known]>1)).any():raise safety.MLFailure('invalid_input')
    _,codes=np.unique(frame.entry_key.str[:16].to_numpy(str),return_inverse=True)
    sizes=np.bincount(codes)
    if (sizes<2).any():raise safety.MLFailure('invalid_identity')
    front=known&(s>=2/3);counts=np.bincount(codes,weights=known)[codes]-known
    fronts=np.bincount(codes,weights=front)[codes]-front
    sums=np.bincount(codes,weights=np.where(known,s,0.))[codes]-np.where(known,s,0.)
    denominator=sizes[codes]-1
    fraction=np.divide(fronts,denominator,out=np.full(len(s),np.nan),where=counts>0)
    mean=np.divide(sums,counts,out=np.full(len(s),np.nan),where=counts>0)
    out=frame.copy()
    out['style_proxy']=s;out['style_missing']=missing.astype(float)
    out['others_front_fraction']=fraction;out['others_style_mean']=mean
    out['others_style_known_fraction']=counts/denominator
    out['style_pressure_interaction']=(s-.5)*fraction
    return out


def load(start,end):
    c.guard(start,end)
    return augment(old.market_frame(c.S4,start,end),read_style(start,end))


def coverage(frame):
    result={'races':int(frame.entry_key.str[:16].nunique()),'runners':len(frame),'features':{}}
    for name in c.CONFIGS['H_pace']:
        values=frame[name];result['features'][name]={'missing_count':int(values.isna().sum()),'missing_rate':float(values.isna().mean()),
            'finite_distinct_values':int(values.dropna().nunique())}
    return result


def training_gate(frame):
    return all(frame[n].dropna().nunique()>1 for n in ('style_proxy','style_pressure_interaction'))
