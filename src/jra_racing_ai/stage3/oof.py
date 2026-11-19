from datetime import date,timedelta
from pathlib import Path
from uuid import uuid4
import json
import numpy as np
from ..ml import safety,model as legacy
from ..research import contracts as oldc,model as vm,experiment as exp
from . import contracts as c,store as s,scoring


def verify_timing(split,first_prediction):
    day=date.fromisoformat(str(first_prediction))
    for key in ('train_end','validation_end','calibration_end'):
        if date.fromisoformat(split[key])>=day:raise safety.MLFailure('invalid_split')
    if not split['train_end']<split['validation_start']<=split['validation_end']<split['calibration_start']<=split['calibration_end']:
        raise safety.MLFailure('invalid_split')
    c.guard(date.fromisoformat(split['train_start']),day)


def fit_once(directory,config,fold,split,parts,*,extra=False):
    folder=directory/'fit_units';folder.mkdir(exist_ok=True)
    unit=config+'_'+str(fold)
    done=s.completed(folder,unit)
    if done:
        model_path=directory/'models'/done['summary']['model_id']
        if extra:
            from .supplement import load_model
            return load_model(model_path),done['summary']
        return vm.load(model_path,root=c.ROOT),done['summary']
    if (folder/(unit+'.started.json')).exists():raise safety.MLFailure('training_failed')
    started=list(folder.glob('*.started.json'))
    if len(started)>=7 or sum(p.name.startswith(config+'_') for p in started)>=(4 if extra else 3):
        raise safety.MLFailure('input_limit')
    s.write(folder,unit+'.started.json',dict(s.provenance(directory,config,fold,split=split),status='started',new_LightGBM_fit=True))
    parameters=s.source('plan.json')['ability_parameters']
    if extra:
        from .supplement import fit_model
        bundle=fit_model(parts,split,parameters)
    else:
        bundle=vm.fit(parts['train'],parts['validation'],parts['calibration'],oldc.spec(tuple('ABCD')),split=split,parameters=parameters)
    bundle['metadata']['stage3_provenance']=s.provenance(directory,config,fold,split=split)
    path=safety.local_path(directory/'models'/uuid4().hex,root=c.ROOT);path.mkdir(parents=True)
    metadata=legacy.save_models(bundle,path,root=c.ROOT)
    if extra:
        from .supplement import load_model,predict
        loaded=load_model(path);before=predict(bundle,parts['calibration']);after=predict(loaded,parts['calibration'])
    else:
        loaded=vm.load(path,root=c.ROOT);before=vm.predict(bundle,parts['calibration']);after=vm.predict(loaded,parts['calibration'])
    difference=float(np.max(np.abs(before-after)))
    if difference>1e-12:raise safety.MLFailure('artifact_invalid')
    summary={'model_id':path.name,'model_manifest_sha256':s.digest(path/'model.json'),
        'best_iteration':metadata['best_iteration'],'temperature':metadata['calibration']['temperature'],
        'reload_max_difference':difference,'split':split,'new_fit':True}
    s.complete(folder,unit,[path/'model.json',path/'win.lgb.txt'],summary)
    return loaded,summary


def run(directory):
    if not s.completed(directory,'reuse'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'oof')
    if done:return done['summary']
    folder=directory/'oof';folder.mkdir(exist_ok=True);files=[];origins=[]
    for year in range(2020,2026):
        unit=str(year);done=s.completed(folder,unit)
        if done:
            origins.append(done['summary']);files.extend([folder/(unit+'.complete.json'),folder/(unit+'.csv')]);continue
        sp=c.split(year);verify_timing(sp,f'{year}-01-01')
        if year<=2022:
            parts,_=exp.partition(c.SOURCE,sp)
            bundle,source=fit_once(directory,'OOF_ABCD',year,sp,parts)
            del parts
            path=directory/'models'/source['model_id']
        else:
            source=s.origin(str(year),'ABCD');path=Path(source['path']);bundle=vm.load(path)
            source=dict(source,new_fit=False)
            if bundle['metadata']['split']['calibration_end']!=sp['calibration_end']:raise safety.MLFailure('artifact_invalid')
        verify_timing(bundle['metadata']['split'],f'{year}-01-01')
        end=date(year,12,31) if year<2025 else date(year,9,30)
        frame=s.frame(date(year,1,1),end);p=vm.predict(bundle,frame)
        check=0.
        if year>=2023:
            saved=scoring.flatten(exp.saved_predictions(c.SOURCE,s.trial(str(year),'ABCD')))
            earlier=frame.loc[frame['date']<=f'{year}-09-30']
            actual=exp.predictions(earlier,p[(frame['date']<=f'{year}-09-30').to_numpy()]);a,b=scoring.align(actual,saved)
            check=max(abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities)
            if check>1e-12:raise safety.MLFailure('artifact_invalid')
        out=frame.loc[:,['entry_key','date']].copy();out['ability_probability']=p;out['source_model_id']=source['model_id']
        out['feature_history_cutoff']=[(date.fromisoformat(str(day))-timedelta(days=7)).isoformat() for day in frame['date']]
        target=safety.local_path(folder/(unit+'.csv'),root=c.ROOT)
        with target.open('x',encoding='utf-8',newline='') as stream:out.to_csv(stream,index=False,lineterminator='\n')
        summary=dict(s.provenance(directory,'OOF_ABCD',year,parents=[source],split=sp),
            model_id=source['model_id'],new_fit=year<=2022,rows=len(frame),races=int(frame.entry_key.str[:16].nunique()),
            prediction_start=f'{year}-01-01',prediction_end=end.isoformat(),timing_valid=True,
            categories_available_through=sp['train_end'],temperature=bundle['metadata']['calibration']['temperature'],
            saved_annual_max_difference=check,extra_months_prediction_only=year in (2023,2024))
        s.complete(folder,unit,[target],summary);origins.append(summary);files.extend([target,folder/(unit+'.complete.json')])
        print(json.dumps({'status':'running','stage':'oof','year':year,'rows':len(frame),'new_fit':year<=2022}),flush=True)
    s.write(directory,'oof_origins.json',{'origins':origins,'new_fits':3,'in_sample_predictions_used':False})
    files.append(directory/'oof_origins.json')
    summary={'new_LightGBM_fits':3,'years':[x['fold_id'] for x in origins],'total_rows':sum(x['rows'] for x in origins),
        'all_timing_valid':True,'Stage2_temperature_procedure_unchanged':True}
    s.complete(directory,'oof',files,summary);return summary


def load(directory,start,end):
    import pandas as pd
    c.guard(start,end)
    if start<date(2020,1,1):raise safety.MLFailure('invalid_split')
    parts=[]
    for year in range(start.year,end.year+1):
        done=s.completed(directory/'oof',str(year))
        if not done:raise safety.MLFailure('artifact_invalid')
        verify_timing(done['summary']['split'],f'{year}-01-01')
        frame=pd.read_csv(directory/'oof'/f'{year}.csv',dtype={'entry_key':'string','date':'string','source_model_id':'string'},float_precision='round_trip')
        validate_rows(frame,done['summary'],year)
        parts.append(frame.loc[(frame.date>=start.isoformat())&(frame.date<=end.isoformat())])
    return pd.concat(parts,ignore_index=True)


def validate_rows(frame,summary,year):
    verify_timing(summary['split'],f'{year}-01-01')
    if frame.entry_key.duplicated().any() or set(frame.source_model_id)!={summary['model_id']}:
        raise safety.MLFailure('artifact_invalid')
    if not (frame.date.str[:4]==str(year)).all() or not (frame.entry_key.str[:4]==str(year)).all():
        raise safety.MLFailure('invalid_split')
    if not np.isfinite(frame.ability_probability).all() or not ((frame.ability_probability>0)&(frame.ability_probability<1)).all():
        raise safety.MLFailure('invalid_input')
    for day,cutoff in zip(frame.date,frame.feature_history_cutoff):
        if (date.fromisoformat(str(day))-timedelta(days=7)).isoformat()!=str(cutoff):raise safety.MLFailure('invalid_split')
