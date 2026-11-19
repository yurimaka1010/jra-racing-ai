"""One finite T3 input-only survey. No labels, result loaders or model inference."""
from collections import Counter,defaultdict
from datetime import timedelta
from pathlib import Path
import json,math,time
import numpy as np
from ..core import SafeAuditError
from ..ml import safety
from ..win_sim import core,publication,local
from . import contracts as c,store as s,inputs,planning

def saved_quote(document,race,roster,cutoff,max_age):
    if document is None:return None
    try:
        p=core.instant(document['published_at'])
        if (document['category']!='1' or document['basis']!='pre_race_publication_time'
                or set(document['odds'])!=set(roster) or p+timedelta(minutes=1)>cutoff
                or not 0<=(cutoff-p).total_seconds()<=max_age):return None
        return document
    except (KeyError,TypeError,SafeAuditError):return None

def assess(target,rows):
    race=tuple(target['race_key']);c.historical(race);start=core.instant(target['scheduled_start']);roster=target['roster']
    current,reason=inputs.select(race,rows,roster,start-timedelta(minutes=3),120)
    early=saved_quote(target['quotes']['30'],race,roster,start-timedelta(minutes=30),300)
    ten=saved_quote(target['quotes']['10'],race,roster,start-timedelta(minutes=10),300)
    frame=inputs.features(race,current,early) if current else None
    record={'race_key':list(race),'roles':target['roles'],'in_B':target['in_B'],'roster_size':len(roster),
        'quote_T3':current,'reason':reason,'in_C':bool(target['in_B'] and current and ten),
        'C_exclusion':reason if current is None else 'T10_saved_control_unavailable' if ten is None else None,
        'T30_missing':early is None,'T3_age_seconds':(start-timedelta(minutes=3)-core.instant(current['published_at'])).total_seconds() if current else None,
        'schedule_asof_verified':target['schedule_asof_verified'],'original_PC_receipt_verified':False}
    return record,frame

def run(folder,session):
    if (folder/'T3_summary.json').exists():return s.read(folder/'T3_summary.json')
    if (folder/'T3_started.json').exists():raise safety.MLFailure('artifact_exists')
    plan=planning.verify(folder,'T3')
    if s.sha(folder/plan['targets_file'])!=plan['targets_sha256']:raise safety.MLFailure('artifact_invalid')
    for p,h in plan['source_references'].items():
        if s.sha(Path(p))!=h:raise safety.MLFailure('artifact_invalid')
    targets=s.read(folder/plan['targets_file'])['records']
    for item in targets:c.historical(item['race_key'])
    s.write(folder/'T3_started.json',{'started_at':c.clock().isoformat(),'plan_sha256':s.sha(folder/'T3_plan.json'),
        'budget_seconds':3600,'fits':0,'retry_permitted':False})
    batches=s.mkdir(folder/'T3_batches');begin=time.monotonic();records=[];files=[];total_rows=0;queries=0;error=None
    feature_stats=defaultdict(lambda:{name:{'missing':0,'finite':0,'distinct':set()} for name in c.H_FIELDS})
    try:
        with session() as db:
            columns=publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,local.Budget(60))
            if columns is None:raise safety.MLFailure('invalid_schema')
            for offset in range(0,len(targets),32):
                if time.monotonic()-begin>=3600:error='survey_time_budget';break
                batch=targets[offset:offset+32];races=[tuple(t['race_key']) for t in batch]
                starts={tuple(t['race_key']):t['scheduled_start'] for t in batch}
                sql,args=inputs.query(races,starts)
                raw=db.aggregate('stage7.T3_input_only',sql,args,max_rows=64);queries+=1;total_rows+=len(raw)
                grouped=inputs.grouped(raw);out=[]
                for target in batch:
                    race=tuple(target['race_key']);record,frame=assess(target,grouped.get((race,3),[]));out.append(record)
                    if frame is not None:
                        for role in target['roles']:
                            for name in c.H_FIELDS:
                                values=frame[name].to_numpy(float);v=feature_stats[role][name]
                                v['missing']+=int(np.isnan(values).sum());finite=values[np.isfinite(values)];v['finite']+=len(finite)
                                if len(v['distinct'])<=1024:v['distinct'].update(map(float,finite))
                path=batches/f'{offset//32:04d}.local.json'
                s.write(path,{'records':out,'requested_races':len(batch),'returned_rows':len(raw),'completed_at':c.clock().isoformat()})
                files.append(path);records.extend(out)
                if queries%25==0:print(json.dumps({'status':'running','stage':'T3_input_survey','completed_races':len(records),
                    'total_target_races':len(targets),'bounded_queries':queries,'elapsed_seconds':round(time.monotonic()-begin,1),'fits':0}),flush=True)
    except SafeAuditError as exc:error=exc.code
    requested={tuple(x['race_key']):x for x in targets};done={tuple(x['race_key']):x for x in records}
    dimensions={key:defaultdict(Counter) for key in ('year','month','venue','race_number','roster_band')};role_counts=defaultdict(Counter)
    b_years=defaultdict(Counter);C=[];B=Counter();all_reasons=Counter();ages=[]
    for race,target in requested.items():
        item=done.get(race);ok=item is not None and item['quote_T3'] is not None;reason=item['reason'] if item else 'not_surveyed'
        for role in target['roles']:
            v=role_counts[role];v['original_T10_input_races']+=1;v['original_T10_input_runners']+=len(target['roster']);v['surveyed']+=item is not None
            v['T3_usable_races']+=ok;v['T3_usable_runners']+=len(target['roster']) if ok else 0
            v['T30_missing_among_T3_usable']+=bool(ok and item['T30_missing'])
        if not ok:all_reasons[reason]+=1
        if ok:ages.append(item['T3_age_seconds'])
        if not target['in_B']:continue
        B['races']+=1;B['runners']+=len(target['roster']);by=b_years[race[0]];by['B_races']+=1;by['B_runners']+=len(target['roster'])
        valid=bool(item and item['in_C']);by['C_races']+=valid;by['C_runners']+=len(target['roster']) if valid else 0
        if valid:C.append({'race_key':list(race),'roster':target['roster']})
        attrs={'year':race[0],'month':race[0]+'-'+race[1][:2],'venue':race[2],'race_number':race[-1],
            'roster_band':'2_8' if len(target['roster'])<=8 else '9_12' if len(target['roster'])<=12 else '13_plus'}
        for key,value in attrs.items():
            counts=dimensions[key][value];counts['B_races']+=1;counts['C_races']+=valid
            if not valid:counts['missing_'+str(item['C_exclusion'] if item else 'not_surveyed')]+=1
    stat={role:{name:{'missing_count':v['missing'],'finite_count':v['finite'],
        'distinct_values_lower_bound':len(v['distinct']),'distinct_count_capped':len(v['distinct'])>1024,
        'nonconstant':len(v['distinct'])>1} for name,v in values.items()} for role,values in feature_stats.items()}
    feasible=all(role_counts[f'{y}_{r}']['T3_usable_races']>=2 for y in c.YEARS for r in ('train','early','calibration','evaluation'))
    variable=all(all(stat.get(f'{y}_train',{}).get(f,{}).get('nonconstant',False) for f in c.MARKET_FEATURES) for y in c.YEARS)
    complete=len(records)==len(targets) and error is None
    status='publication_time_replay_possible' if complete and feasible and variable else 'replay_unavailable_or_incomplete'
    for counts in b_years.values():counts['coverage_percent']=100*counts['C_races']/counts['B_races']
    for counts in role_counts.values():counts['T3_coverage_percent']=100*counts['T3_usable_races']/counts['original_T10_input_races']
    for values in dimensions.values():
        for counts in values.values():counts['missing_rate']=1-counts['C_races']/counts['B_races']
    s.write(folder/'T3_C.local.json',{'records':C,'created_without_labels':True})
    result={'status':status,'complete':complete,'stop_reason':error,'B':dict(B),'C':{'races':len(C),'runners':sum(len(x['roster']) for x in C),
        'list_sha256':s.sha(folder/'T3_C.local.json'),'key_roster_sha256':s.digest(C)},'years':dict(b_years),'roles':dict(role_counts),
        'missingness_dimensions':{k:dict(v) for k,v in dimensions.items()},'all_target_missing_reasons':dict(all_reasons),
        'T3_age_seconds':core.distribution(ages),'H8_feature_stats':stat,'H8_formula_reused':'research.market_data.augment_current_T3_reference_T30',
        'all_three_origins_have_input':feasible,'all_three_train_current_fields_vary':variable,
        'target_races':len(targets),'completed_races':len(records),'not_surveyed_races':len(targets)-len(records),
        'bounded_queries':queries,'returned_rows':total_rows,'elapsed_seconds':time.monotonic()-begin,
        'publication_precision':'minute_end_before_cutoff','T3_max_age_seconds':120,'T10_and_T30_max_age_seconds':300,
        'original_PC_receipt_verified':False,'schedule_asof_verified_races':sum(t['schedule_asof_verified'] for t in targets),
        'fits':0,'labels_queried':0,'HR_rows_queried':0,'final_odds_queried':0,'2026_rows_queried':0,
        'coverage_threshold_used':False,'Stage7b_authorized':False,'proposed_next_budget_fits':3 if status=='publication_time_replay_possible' else 0,
        'batch_hashes':{p.name:s.sha(p) for p in files},'formal_survey_projection_only':True,
        'preparation_incident':plan['preparation_incident']}
    s.write(folder/'T3_summary.json',result);return result
