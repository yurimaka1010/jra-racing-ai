"""Finite development schedule and artifact assembly; final evaluation is excluded."""
from collections import Counter,defaultdict
from datetime import date,timedelta
import gc
import json
from pathlib import Path

from ..core import RACE_KEY
from ..ml import model as old_model,safety
from ..win_sim import core
from . import contracts as c,features,io


def key_parts(text):
    if not isinstance(text,str) or len(text)!=18 or not text.isascii() or not text.isdigit():
        raise safety.MLFailure('invalid_identity')
    return (text[:4],text[4:8],text[8:10],text[10:12],text[12:14],text[14:16],text[16:])


def build(directory):
    import pandas as pd
    builder=features.Builder();pending_x=[];pending_meta=[];current=None;entries=[]
    folder=safety.local_path(directory/'features',root=c.ROOT);folder.mkdir(exist_ok=True)
    def save(year):
        frame=pd.DataFrame(pending_x,columns=c.ALL_FEATURES)
        meta=pd.DataFrame({'entry_key':[''.join(x['key']) for x in pending_meta],
            'date':[x['date'] for x in pending_meta], 'eligible':[int(x['eligible']) for x in pending_meta],
            'winner_label':[x['winner_label'] for x in pending_meta],
            'start_time':[x['start_time'] for x in pending_meta], 'previous_start':[x['previous_start'] for x in pending_meta]})
        path=safety.local_path(folder/(str(year)+'.csv'),root=c.ROOT)
        with path.open('x',encoding='utf-8',newline='') as stream:
            pd.concat([meta,frame],axis=1).to_csv(stream,index=False,lineterminator='\n')
        good=meta['eligible']==1
        entry={'file':path.name,'year':year,'rows':len(frame),'eligible_rows':int(good.sum()),
            'eligible_races':len({x['entry_key'][:16] for x in meta.loc[good].to_dict('records')}),
            'missing_all':{k:int(frame[k].isna().sum()) for k in c.ALL_FEATURES},
            'missing_eligible':{k:int(frame.loc[good,k].isna().sum()) for k in c.ALL_FEATURES},
            'sha256':safety.digest(path,root=c.ROOT)}
        entries.append(entry)
        print(json.dumps(dict(status='running',stage='build',year=year,eligible_races=entry['eligible_races'])),flush=True)
    for rows in io.iter_days(directory):
        year=int(rows[0]['kaisai_nen'])
        if current is not None and year!=current:
            save(current);pending_x.clear();pending_meta.clear();gc.collect()
        current=year
        x,m=builder.transform_day(rows);pending_x.extend(x);pending_meta.extend(m)
    if current is not None:save(current)
    io.write(directory,'feature_manifest.json',{'input_format':c.INPUT_VERSION,'formula_version':c.FORMULA_VERSION,
        'features':c.ALL_FEATURES,'categories':c.CATEGORIES,'years':entries,
        'population_D':'all_JRA_01_10_starters_in_source_not_target_horse_filtered',
        'population_C':'all_actual_starters_per_source_race_relative_metrics_missing_if_incomplete',
        'sealed_2026_rows':0})
    return {'years':len(entries),'rows':sum(e['rows'] for e in entries),'sealed_2026_rows':0}


def load_cache(directory,start,end):
    import pandas as pd
    c.guard_development(start,end)
    manifest=io.read(directory,'feature_manifest.json')
    if manifest['formula_version']!=c.FORMULA_VERSION:raise safety.MLFailure('artifact_invalid')
    parts=[]
    dtype={name:'string' for name in c.CATEGORIES}
    dtype.update(entry_key='string',date='string',start_time='string',previous_start='string')
    for entry in manifest['years']:
        if start.year<=entry['year']<=end.year:
            path=safety.local_path(directory/'features'/entry['file'],root=c.ROOT,exists=True)
            if safety.digest(path,root=c.ROOT)!=entry['sha256']:raise safety.MLFailure('artifact_invalid')
            f=pd.read_csv(path,dtype=dtype,keep_default_na=False,na_values=[''],float_precision='round_trip')
            mask=(f['date']>=start.isoformat())&(f['date']<=end.isoformat())&(f['eligible']==1)
            parts.append(f.loc[mask].copy())
    if not parts:raise safety.MLFailure('insufficient_history')
    return pd.concat(parts,ignore_index=True)


def predictions(frame,probabilities):
    grouped=defaultdict(dict)
    for key,p in zip(frame['entry_key'],probabilities):
        parts=key_parts(str(key));h=parts[-1];race=parts[:-1]
        if h in grouped[race]:raise safety.MLFailure('duplicate_entry')
        grouped[race][h]=float(p)
    return [core.PredictionRace(k,v) for k,v in sorted(grouped.items())]


def winners(frame):
    return {key_parts(str(k))[:-1]:key_parts(str(k))[-1] for k,y in zip(frame['entry_key'],frame['winner_label']) if y==1}


def market(directory):
    from .market_data import collect
    return collect(directory)


def annual_split(year):
    if year not in (2023,2024,2025):raise safety.MLFailure('invalid_split')
    previous=year-1
    return {'train_start':'2016-07-02','train_end':f'{previous}-08-31',
        'validation_start':f'{previous}-09-08','validation_end':f'{previous}-10-31',
        'calibration_start':f'{previous}-11-08','calibration_end':f'{previous}-12-24',
        'evaluation':[(f'{year}-01-01',f'{year}-09-30')]}


def partition(directory,split,*,anchor=False):
    from . import market_data
    parts={}
    for name in ('train','validation','calibration'):
        start=date.fromisoformat(split[name+'_start']);end=date.fromisoformat(split[name+'_end'])
        if anchor:start=max(start,date(2020,1,1))
        f=load_cache(directory,start,end)
        if anchor:f=market_data.augment(f,market_data.load(directory,start,end))
        parts[name]=f
    evaluations=[]
    for start,end in split['evaluation']:
        f=load_cache(directory,date.fromisoformat(start),date.fromisoformat(end))
        if anchor:f=market_data.augment(f,market_data.load(directory,date.fromisoformat(start),date.fromisoformat(end)))
        evaluations.append(f)
    return parts,evaluations


def legacy_reproduction(directory,evaluations):
    import numpy as np
    from . import model as vm
    from .. import stage1
    root=safety.ML_ROOT/'models'/stage1.MODEL_ID
    old=vm.load(root,root=safety.ML_ROOT)
    records=stage1.read(stage1.ROOT/'evaluations/53fe500d9063471eb0cacd16b0bec85a','predictions.local.json')
    expected={''.join(r['race_key'])+h:float(p) for r in records if r['period']!='recent_coverage' for h,p in r['probabilities'].items()}
    maximum=0.;count=0
    for f in evaluations:
        x=vm.matrix(f,c.spec(),old['metadata']['categories'])
        raw=old['boosters']['win'].predict(x,num_threads=4)
        actual=vm.normalize(f,raw,.75)
        for key,value in zip(f['entry_key'],actual):
            if str(key) not in expected:raise safety.MLFailure('artifact_invalid')
            maximum=max(maximum,abs(float(value)-expected[str(key)]));count+=1
    if count!=len(expected) or maximum>1e-12:raise safety.MLFailure('artifact_invalid')
    io.write(directory,'cache_legacy_parity.json',{'rows':count,'maximum_probability_difference':maximum,
        'old_model_id':stage1.MODEL_ID,'temperature':.75})
    return maximum


def trial_path(directory,fold,name):
    return directory/'trials'/(fold+'_'+name+'.json')


def read_trial(directory,fold,name):
    return io.read(directory/'trials',fold+'_'+name+'.json')


def saved_predictions(directory,trial):
    document=io.read(directory/'trials',trial['prediction_file'])
    return [[core.PredictionRace(tuple(p['race_key']),p['probabilities']) for p in part] for part in document]


def run_trial(directory,fold,name,spec,split,parts,evaluations,comparator=None):
    from . import model as vm,metrics,market_data
    file=trial_path(directory,fold,name)
    if file.exists():
        record=read_trial(directory,fold,name)
        if record['feature_spec']!=spec:raise safety.MLFailure('artifact_invalid')
        return record
    plan=io.read(directory,'plan.json')
    parameters=plan['anchor_parameters' if spec['mode']=='market_offset' else 'ability_parameters']
    print(json.dumps(dict(status='running',stage='train',fold=fold,configuration=name,
        rows=len(parts['train']),features=len(spec['features']))),flush=True)
    bundle=vm.fit(parts['train'],parts['validation'],parts['calibration'],spec,split=split,parameters=parameters)
    model_dir,metadata=vm.save(bundle,directory)
    reloaded=vm.load(model_dir)
    before=vm.predict(bundle,parts['calibration']);after=vm.predict(reloaded,parts['calibration'])
    import numpy as np
    reload_diff=float(np.max(np.abs(before-after)))
    if reload_diff>1e-12:raise safety.MLFailure('artifact_invalid')
    all_predictions=[];summaries=[];paper_ledgers=[]
    references=saved_predictions(directory,comparator) if comparator else None
    for i,f in enumerate(evaluations):
        p=predictions(f,vm.predict(reloaded,f));w=winners(f);all_predictions.append(p)
        start,end=split['evaluation'][i]
        packets=market_data.load(directory,date.fromisoformat(start),date.fromisoformat(end))
        common=[x for x in p if packets.get(x.race,{}).get('quotes',{}).get('10') is not None]
        row={'start':start,'end':end,'ability_population':metrics.measure(p,w),
             'market_common_population':metrics.measure(common,w),
             'market_coverage':{'target_races':len(p),'usable_races':len(common),
                                'percent':100*len(common)/len(p) if p else None}}
        if references:
            row['versus_control']=metrics.compare(p,references[i],w)
        summary,ledger=metrics.paper(common,packets,w,start=start,end=end)
        row['paper']=summary;paper_ledgers.append(ledger)
        summaries.append(row)
    folder=safety.local_path(directory/'trials',root=c.ROOT);folder.mkdir(exist_ok=True)
    predfile=fold+'_'+name+'.predictions.local.json'
    io.write(folder,predfile,[[{'race_key':p.race,'probabilities':p.probabilities} for p in part] for part in all_predictions])
    ledgerfile=fold+'_'+name+'.ledgers.local.json';io.write(folder,ledgerfile,paper_ledgers)
    allw={k:v for f in evaluations for k,v in winners(f).items()}
    pooled=metrics.pooled(all_predictions,allw)
    record={'fold':fold,'configuration':name,'model_id':model_dir.name,'feature_spec':spec,
        'split':split,'model_manifest_sha256':safety.digest(model_dir/'model.json',root=c.ROOT),
        'feature_manifest_sha256':safety.digest(directory/'feature_manifest.json',root=c.ROOT),
        'plan_sha256':safety.digest(directory/'plan.json',root=c.ROOT),
        'best_iteration':metadata['best_iteration'],'temperature':metadata['calibration']['temperature'],
        'reload_max_difference':reload_diff,'periods':summaries,'pooled':pooled,
        'prediction_file':predfile,'ledger_file':ledgerfile,'exploration_only':True,
        'training_counts':metadata['counts'],'training_missing':metadata['missing_features']}
    if references:
        flat_ref=[p for part in references for p in part]
        record['pooled_versus_control']=metrics.compare([p for part in all_predictions for p in part],flat_ref,allw)
    io.write(folder,file.name,record)
    print(json.dumps(dict(status='running',stage='trained',fold=fold,configuration=name,model_id=model_dir.name,
        best_iteration=metadata['best_iteration'],temperature=metadata['calibration']['temperature'],
        pooled_race_LL=pooled['race_log_loss'])),flush=True)
    return record


def ability_ab(directory):
    split=io.read(directory,'plan.json')['main_split']
    parts,evals=partition(directory,split)
    if not (directory/'cache_legacy_parity.json').exists():legacy_reproduction(directory,evals)
    control=run_trial(directory,'main','control',c.spec(),split,parts,evals)
    for name in ('A','B','AB'):
        run_trial(directory,'main',name,c.spec(tuple(name)),split,parts,evals,control)
    io.write(directory,'ability_ab_done.json',{'configurations':['control','A','B','AB']})
    return {'completed':['control','A','B','AB']}


def ability_cd(directory):
    if not (directory/'anchor_ab_done.json').exists():raise safety.MLFailure('invalid_split')
    split=io.read(directory,'plan.json')['main_split'];parts,evals=partition(directory,split)
    control=read_trial(directory,'main','control')
    for name in ('C','D','ABCD'):
        record=run_trial(directory,'main',name,c.spec(tuple(name)),split,parts,evals,control)
    promising=(record['pooled']['race_log_loss']<control['pooled']['race_log_loss'] and
        sum(a['ability_population']['race_log_loss']<b['ability_population']['race_log_loss']
            for a,b in zip(record['periods'],control['periods']))>=2)
    ablations=[]
    if promising:
        for omitted in c.GROUPS:
            name='without_'+omitted
            groups=tuple(g for g in c.GROUPS if g!=omitted)
            run_trial(directory,'main',name,c.spec(groups),split,parts,evals,control);ablations.append(name)
    io.write(directory,'ability_full_done.json',{'all_group_promising':promising,'conditional_ablations':ablations})
    return {'completed':['C','D','ABCD',*ablations],'all_group_promising':promising}


def anchor_specs():
    return {'market_only':c.spec(anchor=True,ability=False),
        'market_history':c.spec(anchor=True,ability=False,history=True),
        'market_AB':c.spec(('A','B'),anchor=True),
        'market_AB_history':c.spec(('A','B'),anchor=True,history=True),
        'market_ABCD_history':c.spec(tuple('ABCD'),anchor=True,history=True)}


def anchor_baselines(directory,fold,split,parts,evaluations,control):
    from . import model as vm,metrics
    from .. import stage1_analysis as s1a
    file=directory/'trials'/(fold+'_market_baselines.json')
    if file.exists():return io.read(file.parent,file.name)
    cp=predictions(parts['calibration'],parts['calibration']['market_probability'].to_numpy())
    selected=core.fit_temperature(cp,winners(parts['calibration']),model_fit_end=split['validation_end'],
        test_start=split['evaluation'][0][0])
    references=saved_predictions(directory,control);periods=[]
    io.write(directory/'trials',fold+'_market_calibration.json',selected)
    for i,f in enumerate(evaluations):
        w=winners(f)
        raw=predictions(f,f['market_probability'].to_numpy())
        recal=predictions(f,vm.normalize(f,f['market_probability'].to_numpy(),selected['temperature']))
        supported=f.loc[f['market_history_missing']==0]
        subset={key_parts(str(k))[:-1] for k in supported['entry_key']}
        periods.append({'start':split['evaluation'][i][0], 'raw_market':metrics.measure(raw,w),
            'recalibrated_market':metrics.measure(recal,w),'market_information_control':metrics.measure(references[i],w),
            'history_complete_races':len(subset),'primary_races':len(raw),
            'raw_market_history_complete':metrics.measure([p for p in raw if p.race in subset],w),
            'control_history_complete':metrics.measure([p for p in references[i] if p.race in subset],w),
            'control_versus_raw':metrics.compare(references[i],raw,w),
            'control_versus_recalibrated':metrics.compare(references[i],recal,w)})
    value={'temperature_selected_on_calibration':selected,'periods':periods,'all_2025_periods_are_exploration':True}
    io.write(file.parent,file.name,value)
    return value


def history_subset_comparison(directory,fold,names,evaluations,control):
    from . import metrics
    baseline=saved_predictions(directory,control);result={}
    for name in names:
        rec=read_trial(directory,fold,name);values=saved_predictions(directory,rec);periods=[]
        for i,f in enumerate(evaluations):
            good={key_parts(str(k))[:-1] for k in f.loc[f['market_history_missing']==0,'entry_key']}
            a=[p for p in values[i] if p.race in good];b=[p for p in baseline[i] if p.race in good];w=winners(f)
            periods.append({'races':len(a),'candidate':metrics.measure(a,w),'control':metrics.measure(b,w),
                            'comparison':metrics.compare(a,b,w) if a else None})
        result[name]=periods
    io.write(directory/'trials',fold+'_history_complete_comparison.json',result)


def anchor_ab(directory):
    if not (directory/'ability_ab_done.json').exists():raise safety.MLFailure('invalid_split')
    split=dict(io.read(directory,'plan.json')['main_split'],train_start='2020-01-01')
    parts,evals=partition(directory,split,anchor=True)
    configs=anchor_specs()
    control=run_trial(directory,'main','market_only',configs['market_only'],split,parts,evals)
    for name in ('market_history','market_AB','market_AB_history'):
        run_trial(directory,'main',name,configs[name],split,parts,evals,control)
    anchor_baselines(directory,'main',split,parts,evals,control)
    io.write(directory,'anchor_ab_done.json',{'configurations':list(configs)[:4], 'model_temperature':1.})
    return {'completed':list(configs)[:4]}


def anchor_full(directory):
    if not (directory/'ability_full_done.json').exists():raise safety.MLFailure('invalid_split')
    split=dict(io.read(directory,'plan.json')['main_split'],train_start='2020-01-01')
    parts,evals=partition(directory,split,anchor=True)
    control=read_trial(directory,'main','market_only')
    run_trial(directory,'main','market_ABCD_history',anchor_specs()['market_ABCD_history'],split,parts,evals,control)
    history_subset_comparison(directory,'main',list(anchor_specs())[1:],evals,control)
    io.write(directory,'anchor_full_done.json',{'configuration':'market_ABCD_history'})
    return {'completed':['market_ABCD_history']}


def select_candidates(directory):
    path=directory/'candidates.development.json'
    if path.exists():return io.read(directory,path.name)
    if not (directory/'anchor_full_done.json').exists():raise safety.MLFailure('invalid_split')
    ablations=io.read(directory,'ability_full_done.json')['conditional_ablations']
    ability_names=['A','B','AB','C','D','ABCD',*ablations]
    ability=min((read_trial(directory,'main',n) for n in ability_names),key=lambda r:(r['pooled']['race_log_loss'],r['configuration']))
    market_control=min((read_trial(directory,'main',n) for n in ('market_only','market_history')),
                       key=lambda r:(r['pooled']['race_log_loss'],r['configuration']))
    anchor=min((read_trial(directory,'main',n) for n in ('market_AB','market_AB_history','market_ABCD_history')),
               key=lambda r:(r['pooled']['race_log_loss'],r['configuration']))
    result={'selection_metric':'pooled_race_winner_log_loss','selection_periods':'all_three_2025_exploration_periods',
        'ability_candidate':ability['configuration'],'ability_spec':ability['feature_spec'],
        'anchor_candidate':anchor['configuration'],'anchor_spec':anchor['feature_spec'],
        'market_control':market_control['configuration'],'market_control_spec':market_control['feature_spec'],
        'annual_candidates_per_family':1,'final_2026_not_used':True,
        'annual_may_retain_baseline_if_improvement_uncertain':True,
        'annual_retention_rule':{'pooled_race_LL_difference_must_be_negative':True,
            'minimum_improved_years':2,'years':3,'confidence_interval_crossing_zero':'report_uncertain_not_no_effect',
            'status':'development_candidate_only_no_formal_final_claim'}}
    io.write(directory,path.name,result)
    return result


def annual(directory):
    selected=select_candidates(directory)
    results=[]
    for year in (2023,2024,2025):
        fold=str(year);split=annual_split(year)
        parts,evals=partition(directory,split)
        control=run_trial(directory,fold,'annual_control',c.spec(),split,parts,evals)
        candidate=run_trial(directory,fold,'annual_ability',selected['ability_spec'],split,parts,evals,control)
        del parts,evals;gc.collect()
        anchor_split=dict(split,train_start='2020-01-01')
        parts,evals=partition(directory,anchor_split,anchor=True)
        market_control=run_trial(directory,fold,'annual_market_control',selected['market_control_spec'],anchor_split,parts,evals)
        anchor=run_trial(directory,fold,'annual_anchor',selected['anchor_spec'],anchor_split,parts,evals,market_control)
        baselines=anchor_baselines(directory,fold,anchor_split,parts,evals,market_control)
        results.append({'year':year,'ability_control_model_id':control['model_id'],'ability_candidate_model_id':candidate['model_id'],
            'ability_control':control['pooled'],'ability_candidate':candidate['pooled'],
            'ability_comparison':candidate['pooled_versus_control'],
            'market_control_model_id':market_control['model_id'],'anchor_candidate_model_id':anchor['model_id'],
            'market_control':market_control['pooled'],'anchor_candidate':anchor['pooled'],
            'anchor_comparison':anchor['pooled_versus_control'],'exploration_only':True})
        del parts,evals;gc.collect()
        print(json.dumps(dict(status='running',stage='annual',year=year,
            ability_LL_difference=results[-1]['ability_comparison']['race_week_cluster']['difference_new_minus_baseline'],
            anchor_LL_difference=results[-1]['anchor_comparison']['race_week_cluster']['difference_new_minus_baseline'])),flush=True)
    io.write(directory,'annual_results.json',{'years':results,'candidates':selected,'sealed_2026_evaluated':False})
    return {'years':results,'candidates':selected}


def freeze(directory):
    from .report import freeze as create_report
    return create_report(directory)
