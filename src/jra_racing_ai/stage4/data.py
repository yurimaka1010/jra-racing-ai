from datetime import date,timedelta
from io import StringIO
from pathlib import Path
import json,re
import pandas as pd
import numpy as np
from ..ml import safety
from ..research import market_data,experiment as exp
from ..stage3 import scoring
from ..win_sim.core import PredictionRace
from . import contracts as c,store as s

META=('entry_key','date','eligible','winner_label','start_time','previous_start')
_frames={};_markets={}


def dates(frame):
    if not len(frame) or 'date' not in frame:raise safety.MLFailure('invalid_split')
    c.guard(date.fromisoformat(str(frame.date.min())),date.fromisoformat(str(frame.date.max())))


def frame(directory,start,end):
    c.guard(start,end);parts=[]
    for year in range(start.year,end.year+1):
        key=(str(directory),year)
        if key not in _frames:
            path=c.S2/'features'/f'{year}.csv';s.check_reference(directory,path)
            with path.open(encoding='utf-8',newline='') as stream:
                header=stream.readline()
                if not header.startswith('entry_key,date,eligible,winner_label,'):raise safety.MLFailure('invalid_input')
                lines=[header]
                for line in stream:
                    prefix=line.split(',',2)
                    day=date.fromisoformat(prefix[1])
                    if day>c.END:break  # No labels/results after the Stage 4 boundary are parsed.
                    if day<c.START:continue
                    lines.append(line)
            f=pd.read_csv(StringIO(''.join(lines)),usecols=list(META),dtype={n:'string' for n in ('entry_key','date','start_time','previous_start')},
                keep_default_na=False,na_values=[''],float_precision='round_trip')
            _frames[key]=f.loc[f.eligible==1].copy()
        f=_frames[key];parts.append(f.loc[(f.date>=start.isoformat())&(f.date<=end.isoformat())].copy())
    return pd.concat(parts,ignore_index=True)


def packet_date(text):
    match=re.match(r'\s*\{\s*"race_key":\s*\[\s*"([0-9]{4})",\s*"([0-9]{4})"',text)
    return date(int(match[1]),int(match[2][:2]),int(match[2][2:])) if match else None


def packet_rows(path):
    """Stream the trusted Stage 2 pretty-JSON container, stop before post-boundary records."""
    current=[];day=None;records=False
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not records:
                if line.strip()=='"records": [':records=True
                continue
            if not current:
                if line=='    {\n' or line=='    {\r\n':current=[line];day=None
                continue
            current.append(line)
            if day is None:
                day=packet_date(''.join(current))
                if day is not None and day>c.END:return
            if line.startswith('    }'):
                if day is None:raise safety.MLFailure('artifact_invalid')
                row=json.loads(''.join(current).rstrip().removesuffix(','))
                if c.START<=day<=c.END:yield row
                current=[];day=None
    if current:raise safety.MLFailure('artifact_invalid')


def market(directory,start,end):
    c.guard(start,end);result={}
    for year in range(start.year,end.year+1):
        key=(str(directory),year)
        if key not in _markets:
            values={}
            for path in sorted((c.S2/'market').glob(f'{year}-*.local.json')):
                s.check_reference(directory,path)
                for row in packet_rows(path):
                    race=tuple(row['race_key'])
                    if race in values:raise safety.MLFailure('duplicate_entry')
                    # Reuse provenance and verified HR; do not duplicate raw O1 payload caches.
                    values[race]={k:row[k] for k in ('race_key','scheduled_start','schedule_asof_verified','quotes','failures','outcome')}
            _markets[key]=values
        for race,value in _markets[key].items():
            day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
            if start<=day<=end:result[race]=value
    return result


def market_frame(directory,start,end):
    return market_data.augment(frame(directory,start,end),market(directory,start,end))


def trial(directory,year,kind):
    if year not in c.YEARS:raise safety.MLFailure('invalid_split')
    name='annual_market_control' if kind=='H' else 'annual_ability'
    return s.ref_json(directory,c.S2/'trials'/f'{year}_{name}.json')


def source_predictions(directory,year,kind):
    name='annual_market_control' if kind=='H' else 'annual_ability'
    raw=s.ref_json(directory,c.S2/'trials'/f'{year}_{name}.predictions.local.json')
    return [PredictionRace(tuple(p['race_key']),p['probabilities']) for part in raw for p in part]


def ability(directory,start,end):
    c.guard(start,end);parts=[]
    for year in range(start.year,end.year+1):
        if year<2021:raise safety.MLFailure('invalid_split')
        path=c.S3/'oof'/f'{year}.csv';s.check_reference(directory,path)
        meta=s.ref_json(directory,c.S3/'oof'/f'{year}.complete.json')['summary']
        if not meta['split']['calibration_end']<f'{year}-01-01':raise safety.MLFailure('invalid_split')
        f=pd.read_csv(path,dtype={'entry_key':'string','date':'string','source_model_id':'string'},float_precision='round_trip')
        if not (f.date.str[:4]==str(year)).all() or set(f.source_model_id)!={meta['model_id']} or f.entry_key.duplicated().any():
            raise safety.MLFailure('artifact_invalid')
        if f.date.max()>c.END.isoformat():raise safety.MLFailure('invalid_split')
        for d,cutoff in zip(f.date,f.feature_history_cutoff):
            if (date.fromisoformat(str(d))-timedelta(days=7)).isoformat()!=str(cutoff):raise safety.MLFailure('invalid_split')
        parts.append(f.loc[(f.date>=start.isoformat())&(f.date<=end.isoformat())].rename(columns={'source_model_id':'ability_model_id'}))
    return pd.concat(parts,ignore_index=True)


def merge_probability(frame,probabilities,column):
    before=frame.entry_key.tolist()
    if frame.entry_key.duplicated().any() or probabilities.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    races=set(frame.entry_key.str[:16]);same_races=probabilities.loc[probabilities.entry_key.str[:16].isin(races),'entry_key']
    if set(same_races)!=set(before):raise safety.MLFailure('invalid_identity')
    result=frame.merge(probabilities,on='entry_key',how='left',sort=False,validate='one_to_one')
    if before!=result.entry_key.tolist() or result[column].isna().any():raise safety.MLFailure('invalid_identity')
    return result


def reproduce(directory):
    done=s.completed(directory,'reproduce')
    if done:return done['summary']
    from ..research import model as vm
    summaries=[];maximum=0.;files=[]
    for year in c.YEARS:
        start,end=date(year,1,1),date(year,9,30);f=market_frame(directory,start,end)
        expected=s.ref_json(directory,c.S3/'integration'/f'{year}_predictions.local.json')
        ph=source_predictions(directory,year,'H');raw=[PredictionRace(tuple(p['race_key']),p['probabilities']) for p in expected['raw_market']]
        scoring.align(ph,raw)
        actual=exp.predictions(f,f.market_probability.to_numpy());a,b=scoring.align(actual,raw)
        maximum=max(maximum,*[abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities])
        meta=trial(directory,year,'H');folder=c.S2/'models'/meta['model_id']
        for name in ('model.json','win.lgb.txt'):s.check_reference(directory,folder/name)
        bundle=vm.load(folder,root=c.S2);actual=exp.predictions(f,vm.predict(bundle,f));a,b=scoring.align(actual,ph)
        maximum=max(maximum,*[abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities])
        if len(ph)!=c.plan()['expected_year_races'][str(year)]:raise safety.MLFailure('invalid_identity')
        value={'year':year,'races':len(ph),'runners':len(f),'max_probability_difference':maximum,
            'cohort_sha256':__import__('hashlib').sha256(json.dumps([(p.race,sorted(p.probabilities)) for p in ph]).encode()).hexdigest()}
        s.write(directory,f'cohort_{year}.json',value);files.append(directory/f'cohort_{year}.json');summaries.append(value)
    if maximum>1e-12 or sum(x['races'] for x in summaries)!=6373:raise safety.MLFailure('artifact_invalid')
    summary={'years':summaries,'common_races':6373,'max_probability_difference':maximum,'no_main_2025_reexecution':True}
    s.complete(directory,'reproduce',files,summary);return summary
