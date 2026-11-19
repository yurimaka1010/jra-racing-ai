"""Optional signed time difference, strictly joined to the frozen Stage 2 population."""
from collections import Counter,defaultdict,deque
from datetime import date,timedelta
from itertools import groupby
from pathlib import Path
import csv,json,re
import numpy as np
from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..inspect_schema import COLUMNS_SQL
from ..ml import safety,model as legacy
from ..research import io as oldio,contracts as oldc,features as oldf,model as vm,experiment as exp
from ..win_sim import core,local
from . import contracts as c,store as s,scoring,oof


def parse(raw,rank,anomaly):
    if raw is None:return None,'null'
    if raw in ('','    '):return None,'blank'
    if raw=='0000':return None,'initial_unsigned_zero'
    if raw=='9999':return None,'invalid_sentinel'
    if not isinstance(raw,str) or not re.fullmatch('[+-][0-9]{3}',raw,flags=re.ASCII):return None,'invalid_format'
    if anomaly!='0':return None,'non_normal_anomaly'
    if not isinstance(rank,str) or not re.fullmatch('[0-9]{2}',rank) or not 1<=int(rank)<=28:return None,'rank_unavailable'
    if (rank=='01')!=(raw[0]=='-'):return None,'sign_rank_conflict'
    value=int(raw[1:])/10
    if raw[0]=='-':value=-value
    return value,'signed_zero_valid' if int(raw[1:])==0 else 'signed_value_valid'


class History:
    def __init__(self):self.pending=deque();self.horses={};self.last=None
    def day(self,rows):
        if not rows:return []
        day=date(int(rows[0]['kaisai_nen']),int(rows[0]['kaisai_tsukihi'][:2]),int(rows[0]['kaisai_tsukihi'][2:]))
        if self.last and day<=self.last:raise safety.MLFailure('invalid_split')
        while self.pending and self.pending[0][0]+timedelta(days=7)<=day:
            _,past=self.pending.popleft()
            for row in past:
                if row['anomaly'] in oldf.STARTERS:
                    value,_=parse(row['time_sa'],row['finish_position'],row['anomaly'])
                    self.horses.setdefault(row['horse_id'],deque(maxlen=3)).append(value)
        result=[]
        for row in rows:
            values=self.horses.get(row['horse_id'],[])
            weights=[(v,w) for v,w in zip(reversed(values),(1.,.5,.25)) if v is not None]
            result.append({'entry_key':''.join(row[k] for k in RACE_KEY)+row['horse_number'],
                c.E_FEATURES[0]:values[-1] if values else None,
                c.E_FEATURES[1]:sum(v*w for v,w in weights)/sum(w for _,w in weights) if weights else None})
        self.pending.append((day,rows));self.last=day;return result


def query(start,end):
    c.guard(start,end);ident=safety.identifier
    fields=['se.'+ident(k)+'::text AS '+ident(k) for k in RACE_KEY]
    fields+=['se.umaban::text AS horse_number','se.ketto_toroku_bango::text AS horse_id',
        'se.kakutei_chakujun::text AS finish_position','se.ijo_kubun_code::text AS anomaly',
        'CASE WHEN pg_catalog.octet_length(se.time_sa)<=4 THEN se.time_sa::text ELSE NULL::text END AS time_sa']
    join=' AND '.join('ra.'+ident(k)+'=se.'+ident(k) for k in RACE_KEY)
    sql='SELECT '+','.join(fields)+' FROM ONLY public.jvd_se se JOIN ONLY public.jvd_ra ra ON '+join+\
        " WHERE se.data_kubun::text='7' AND ra.data_kubun::text='7' AND se.kaisai_nen::text=%s AND se.kaisai_tsukihi::text BETWEEN %s AND %s AND se.keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') ORDER BY "+\
        ','.join('se.'+ident(k) for k in RACE_KEY)+',se.umaban LIMIT %s'
    validate_audit_sql(sql)
    return sql,(str(start.year),start.strftime('%m%d'),end.strftime('%m%d'),20001)


def collect(directory):
    done=s.completed(directory,'E_source')
    if done:return done['summary']
    folder=directory/'E_source';folder.mkdir(exist_ok=True);files=[];months=[];counts=Counter();join_errors=0
    with oldio.session() as db:
        cols=db.aggregate('s3.e_columns',COLUMNS_SQL,('public','jvd_se'),max_rows=1000)
    column=next((r for r in cols if r['name']=='time_sa'),None)
    if column is None or not column['type_is_builtin'] or column['type_name'] not in local.TEXT_TYPES:
        return {'status':'skipped','reason':'time_sa_column_or_type_unconfirmed'}
    fields=[*RACE_KEY,'horse_number','horse_id','finish_position','anomaly','time_sa']
    for item in s.source('source_manifest.json')['months']:
        unit=Path(item['file']).stem;done=s.completed(folder,unit)
        if done:
            months.append(done['summary']);counts.update(done['summary']['format_counts']);files.append(folder/(unit+'.complete.json'));continue
        start=date.fromisoformat(item['start']);end=date.fromisoformat(item['end']);sql,params=query(start,end)
        with (c.SOURCE/'source'/item['file']).open(encoding='utf-8',newline='') as stream:
            original={tuple(row[k] for k in RACE_KEY)+(row['horse_number'],):row for row in csv.DictReader(stream)}
        seen=set();local_counts=Counter();out=folder/(unit+'.csv')
        with oldio.session() as db,out.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for batch in db.iter_batches('s3.e_month',sql,params,max_rows=20000,batch_size=1000,max_seconds=180):
                for row in batch:
                    key=tuple(row[k] for k in RACE_KEY)+(row['horse_number'],)
                    old=original.get(key)
                    if key in seen or old is None or any(row[k]!=old[k] for k in ('horse_id','finish_position','anomaly')):
                        join_errors+=1;continue
                    seen.add(key);_,reason=parse(row['time_sa'],row['finish_position'],row['anomaly']);local_counts[reason]+=1
                    writer.writerow({k:'' if row[k] is None else row[k] for k in fields})
        if seen!=set(original) or join_errors:
            return {'status':'skipped','reason':'frozen_source_key_or_label_mismatch','matched_rows':len(seen),'expected_rows':len(original)}
        value={'month':unit,'rows':len(seen),'format_counts':dict(local_counts),'old_keys_and_labels_unchanged':True}
        s.complete(folder,unit,[out],value);months.append(value);counts.update(local_counts);files.append(folder/(unit+'.complete.json'))
        if start.month==1:print(json.dumps({'status':'running','stage':'E_source','through':unit}),flush=True)
    quality={'status':'confirmed','official_reference':str(local.SPEC_PDF),
        'official_pdf_sha256':safety.digest(local.SPEC_PDF,root=oldc.ROOT.parent),'official_SE_field':61,'official_PDF_page':12,
        'physical_column':'public.jvd_se.time_sa','unit':'0.1_seconds','width':4,
        'sign':'winner_minus_to_second;others_plus_to_first','signed_zero':'valid_preserve_sign;winner_not_forced_zero',
        'initial':'0000_unsigned_is_missing','sentinel':'9999_missing',
        'anomaly_policy':'only_normal_0_time_values_used;other_started_events_keep_missing_E',
        'format_counts':dict(counts),'months':months,'no_source_population_or_label_update':True,
        'going':{'classification':'intentionally_disabled_and_not_collected_in_Stage2','evidence':'research.features.Builder sets None;RAW_FIELDS has no going',
                 'upstream_missing_rate':'not_measured;not_claimed_source_missing'},
        'previous_time_seconds':{'classification':'intentionally_disabled_and_not_collected_in_Stage2',
            'evidence':'research.features.Builder sets None;RAW_FIELDS omits soha_time',
            'upstream_missing_rate':'not_measured;not_claimed_source_missing'}}
    s.write(directory,'E_data_quality.json',quality);files.append(directory/'E_data_quality.json')
    s.complete(directory,'E_source',files,quality);return quality


def build(directory):
    import pandas as pd
    done=s.completed(directory,'E_features')
    if done:return
    history=History();folder=directory/'E_features';folder.mkdir(exist_ok=True);years=defaultdict(list)
    for item in s.source('source_manifest.json')['months']:
        path=directory/'E_source'/item['file']
        unit=Path(item['file']).stem
        if not s.completed(directory/'E_source',unit):raise safety.MLFailure('artifact_invalid')
        with path.open(encoding='utf-8',newline='') as stream:
            for day,rows in groupby(csv.DictReader(stream),lambda r:r['kaisai_nen']+r['kaisai_tsukihi']):
                rows=list(rows);years[int(day[:4])].extend(history.day(rows))
    files=[];missing={}
    for year,rows in sorted(years.items()):
        frame=pd.DataFrame(rows);path=folder/f'{year}.csv'
        with path.open('x',encoding='utf-8',newline='') as stream:frame.to_csv(stream,index=False,lineterminator='\n')
        missing[str(year)]={'rows':len(frame),'missing':{k:int(frame[k].isna().sum()) for k in c.E_FEATURES}}
        files.append(path)
    summary={'version':c.E_VERSION,'features':list(c.E_FEATURES),'years':missing,'lag_days':7,'weights':[1.,.5,.25],'no_current_time_sa_used':True}
    s.complete(directory,'E_features',files,summary)


def augment(directory,frame):
    import pandas as pd
    if not s.completed(directory,'E_features'):raise safety.MLFailure('artifact_invalid')
    c.guard(date.fromisoformat(str(frame.date.min())),date.fromisoformat(str(frame.date.max())))
    frames=[pd.read_csv(directory/'E_features'/f'{year}.csv',dtype={'entry_key':'string'},float_precision='round_trip') for year in sorted(set(frame.date.str[:4]))]
    values=pd.concat(frames,ignore_index=True);keys=frame.entry_key.tolist()
    result=frame.merge(values,on='entry_key',how='left',validate='one_to_one',sort=False,indicator=True)
    if result.entry_key.tolist()!=keys or (result['_merge']!='both').any():raise safety.MLFailure('invalid_identity')
    return result.drop(columns=['_merge'])


def matrix(frame,metadata):
    base=vm.matrix(frame,oldc.spec(tuple('ABCD')),metadata['categories'])
    for name in c.E_FEATURES:base[name]=frame[name].astype('float64')
    if list(base)!=list(oldc.ALL_FEATURES)+list(c.E_FEATURES):raise safety.MLFailure('invalid_model')
    return base


def predict(bundle,frame):
    raw=bundle['boosters']['win'].predict(matrix(frame,bundle['metadata']),num_threads=4)
    return vm.normalize(frame,raw,bundle['metadata']['calibration']['temperature'])


def fit_model(parts,split,parameters):
    import lightgbm as lgb
    categories=vm.schema(parts['train'],oldc.spec(tuple('ABCD')))
    for part in parts.values():c.guard(date.fromisoformat(str(part.date.min())),date.fromisoformat(str(part.date.max())))
    if not parts['train'].date.max()<parts['validation'].date.min() or not parts['validation'].date.max()<parts['calibration'].date.min():
        raise safety.MLFailure('invalid_split')
    metadata={'format_version':3,'model_kind':'Stage3_ABCD_E_binary','formula_version':c.E_VERSION,
        'base_formula_version':oldc.FORMULA_VERSION,'features':list(oldc.ALL_FEATURES)+list(c.E_FEATURES),
        'categories':categories,'split':split,'library_versions':legacy.versions(),
        'calibration':{'temperature':1.,'method':'original_Stage2_four_temperature_candidates'},
        'training_parameters':parameters,'max_rounds':300,'early_stopping_rounds':20,
        'missing_features':{part:{k:int(f[k].isna().sum()) for k in list(oldc.ALL_FEATURES)+list(c.E_FEATURES)} for part,f in parts.items()},
        'counts':{part:{'rows':len(f),'races':int(f.entry_key.str[:16].nunique())} for part,f in parts.items()}}
    train=lgb.Dataset(matrix(parts['train'],metadata),label=parts['train'].winner_label.to_numpy(),categorical_feature=list(categories))
    valid=lgb.Dataset(matrix(parts['validation'],metadata),label=parts['validation'].winner_label.to_numpy(),reference=train)
    booster=lgb.train(parameters,train,num_boost_round=300,valid_sets=[valid],callbacks=[lgb.early_stopping(20,verbose=False)])
    metadata['best_iteration']=booster.best_iteration;bundle={'boosters':{'win':booster},'metadata':metadata}
    raw=booster.predict(matrix(parts['calibration'],metadata),num_threads=4)
    metadata['calibration'].update(core.fit_temperature(exp.predictions(parts['calibration'],raw),exp.winners(parts['calibration']),
        model_fit_end=split['validation_end'],test_start=split['evaluation'][0][0]))
    return bundle


def load_model(path):
    import lightgbm as lgb
    metadata=s.read(path,'model.json')
    if (metadata['model_kind']!='Stage3_ABCD_E_binary' or metadata['formula_version']!=c.E_VERSION or
        metadata['base_formula_version']!=oldc.FORMULA_VERSION or metadata['features']!=list(oldc.ALL_FEATURES)+list(c.E_FEATURES)):
        raise safety.MLFailure('invalid_model')
    if s.digest(path/'win.lgb.txt')!=metadata['artifacts']['win.lgb.txt']:raise safety.MLFailure('artifact_invalid')
    return {'metadata':metadata,'boosters':{'win':lgb.Booster(model_str=(path/'win.lgb.txt').read_text(encoding='utf-8'))}}


def run(directory):
    if not s.completed(directory,'integrate'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'E')
    if done:return done['summary']
    quality=collect(directory)
    if quality['status']!='confirmed':
        s.write(directory,'E_skipped.json',quality);s.complete(directory,'E',[directory/'E_skipped.json'],dict(quality,new_LightGBM_fits=0));return quality
    build(directory);folder=directory/'E_trials';folder.mkdir(exist_ok=True);files=[];results=[]
    for fold in ('main','2023','2024','2025'):
        done=s.completed(folder,fold)
        if done:results.append(done['summary']);files.append(folder/(fold+'.complete.json'));continue
        split=s.trial(fold,'ABCD')['split'];parts,evals=exp.partition(c.SOURCE,split)
        parts={part:augment(directory,f) for part,f in parts.items()}
        bundle,origin=oof.fit_once(directory,'ABCD_E',fold,split,parts,extra=True)
        del parts
        output=[];periods=[];base=exp.saved_predictions(c.SOURCE,s.trial(fold,'ABCD'))
        for i,frame in enumerate(evals):
            frame=augment(directory,frame);p=exp.predictions(frame,predict(bundle,frame));output.append(p)
            periods.append({'metrics':scoring.measure(p,exp.winners(frame)),'versus_Stage2_ABCD':scoring.compare(p,base[i],exp.winners(frame)),
                'E_missing':{k:{'count':int(frame[k].isna().sum()),'percent':100*int(frame[k].isna().sum())/len(frame)} for k in c.E_FEATURES}})
        record=dict(s.provenance(directory,'ABCD_E',fold,parents=[origin],split=split),E_version=c.E_VERSION,periods=periods,model=origin,
            pooled=scoring.measure(scoring.flatten(output),{k:v for frame in evals for k,v in exp.winners(frame).items()}))
        s.write(folder,fold+'.json',record);s.write_predictions(folder,fold+'.predictions.local.json',output)
        s.complete(folder,fold,[folder/(fold+'.json'),folder/(fold+'.predictions.local.json')],{'fold':fold,'model_id':origin['model_id'],'race_LL':record['pooled']['race_log_loss']})
        results.append({'fold':fold,'model_id':origin['model_id'],'race_LL':record['pooled']['race_log_loss']});files.append(folder/(fold+'.complete.json'))
        print(json.dumps({'status':'running','stage':'E','fold':fold,'race_LL':record['pooled']['race_log_loss']}),flush=True)
    summary={'status':'complete','new_LightGBM_fits':4,'results':results,'not_added_to_OOF_integration':True}
    s.complete(directory,'E',files,summary);return summary
