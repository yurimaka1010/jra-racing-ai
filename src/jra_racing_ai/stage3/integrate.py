from datetime import date
import json
import numpy as np
from ..ml import safety
from ..research import experiment as exp,model as vm,market_data
from . import contracts as c,store as s,scoring,softmax,oof


def dataset(directory,start,end):
    c.guard(start,end)
    frame=s.frame(start,end)
    frame=market_data.augment(frame,market_data.load(c.SOURCE,start,end))
    ability=oof.load(directory,start,end)
    before=frame.entry_key.tolist()
    frame=frame.merge(ability[['entry_key','ability_probability','source_model_id','feature_history_cutoff']],
        on='entry_key',how='left',validate='one_to_one',sort=False)
    if frame.entry_key.tolist()!=before or frame.ability_probability.isna().any():raise safety.MLFailure('invalid_identity')
    if any(str(cutoff)>=str(day) for cutoff,day in zip(frame.feature_history_cutoff,frame.date)):
        raise safety.MLFailure('invalid_split')
    return frame


def period_summary(frame,variants):
    labels=exp.winners(frame);good={exp.key_parts(str(k))[:-1] for k in frame.loc[frame.market_history_missing==0,'entry_key']}
    values={name:scoring.measure(p,labels) for name,p in variants.items()}
    contrasts=[('market_history','raw_market'),('lambda_market_history','raw_market'),('lambda_market_ABCD_history','lambda_market_history'),
        ('MA','M'),('MHA','MH'),('MA','lambda_market_history'),('MHA','lambda_market_history'),('arithmetic','lambda_market_history')]
    comparisons={a+'_minus_'+b:scoring.compare(variants[a],variants[b],labels) for a,b in contrasts if a in variants and b in variants}
    subset={name:[p for p in series if p.race in good] for name,series in variants.items()}
    return {'metrics':values,'comparisons':comparisons,'primary_races':int(frame.entry_key.str[:16].nunique()),
        'primary_runners':len(frame),'complete_history_races':len(good),
        'early_missing_retained_races':int(frame.entry_key.str[:16].nunique())-len(good),
        'history_complete_metrics':{name:scoring.measure(p,labels) for name,p in subset.items()},
        'history_complete_comparisons':{a+'_minus_'+b:scoring.compare(subset[a],subset[b],labels) for a,b in [('MA','M'),('MHA','MH')] if a in subset and b in subset and good}}


def run(directory):
    if not s.completed(directory,'oof'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'integrate')
    if done:return done['summary']
    folder=directory/'integration';folder.mkdir(exist_ok=True);files=[];years=[]
    for year in (2023,2024,2025):
        unit=str(year);done=s.completed(folder,unit)
        if done:
            years.append(done['summary']);files.append(folder/(unit+'.complete.json'));continue
        sp=c.integration_split(year)
        train=dataset(directory,date.fromisoformat(sp['train_start']),date.fromisoformat(sp['train_end']))
        valid=dataset(directory,date.fromisoformat(sp['validation_start']),date.fromisoformat(sp['validation_end']))
        if max(train.date)>sp['train_end'] or max(valid.date)>sp['validation_end']:
            raise safety.MLFailure('invalid_split')
        evaluation=dataset(directory,date(year,1,1),date(year,9,30));labels=exp.winners(evaluation)
        variants={'raw_market':exp.predictions(evaluation,evaluation.market_probability.to_numpy()),
            'Stage2_ABCD':exp.predictions(evaluation,evaluation.ability_probability.to_numpy())}
        for name in ('market_history','market_ABCD_history'):
            p=scoring.flatten(exp.saved_predictions(c.SOURCE,s.trial(str(year),name)))
            scoring.align(p,variants['raw_market']);variants[name]=p
            p=scoring.flatten(s.predictions(directory/'reuse',f'{year}_lambda_{name}.predictions.local.json'))
            scoring.align(p,variants['raw_market']);variants['lambda_'+name]=p
        selections=[];current=[]
        for config in c.TERMS:
            candidates=[softmax.fit(directory,config,str(year),train,valid,strength,sp) for strength in c.STRENGTH]
            good=[r for r in candidates if r['status']=='complete']
            if not good:
                s.write(folder,f'{year}_{config}_skipped.json',{'status':'skipped','reason':'all_fixed_optimizations_failed'})
                continue
            scores=[dict(r['validation_metrics'],strength=r['strength']) for r in good]
            selected=scoring.choose(scores,lambda x:-x['strength'])
            record=next(r for r in good if r['strength']==selected['strength'])
            values=softmax.predict(evaluation,record)
            name=f'{year}_{config}.json';s.write(folder,name,dict(record,selected=True,
                validation_candidates=scores,evaluation_base_model_ids=sorted(set(evaluation.source_model_id))))
            loaded=s.read(folder,name);again=softmax.predict(evaluation,loaded)
            if np.max(np.abs(values-again))>1e-12:raise safety.MLFailure('artifact_invalid')
            variants[config]=exp.predictions(evaluation,values)
            current.append(folder/name);selections.append({'config':config,'strength':record['strength'],'coefficients':record['coefficients']})
        # Arithmetic blend: only six weights and only the previous Nov/Dec OOF labels.
        cal=dataset(directory,date.fromisoformat(sp['calibration_start']),date.fromisoformat(sp['calibration_end']))
        scores=[dict(weight=w,**scoring.evaluate(cal,(1-w)*cal.market_probability.to_numpy()+w*cal.ability_probability.to_numpy())) for w in c.WEIGHTS]
        chosen=scoring.choose(scores,lambda x:x['weight']);w=chosen['weight']
        s.write(folder,f'{year}_arithmetic.json',dict(s.provenance(directory,'arithmetic',year,split=sp),
            selected_weight=w,calibration_scores=scores,base_OOF_manifest_sha256=s.digest(directory/'oof_origins.json'),
            base_model_ids=sorted(set(cal.source_model_id)|set(evaluation.source_model_id))))
        variants['arithmetic']=exp.predictions(evaluation,(1-w)*evaluation.market_probability.to_numpy()+w*evaluation.ability_probability.to_numpy())
        summary=dict(s.provenance(directory,'integration_comparison',year,split=sp),**period_summary(evaluation,variants),
            selections=selections,arithmetic_weight=w,full_ability_races=s.trial(str(year),'ABCD')['pooled']['races'],
            coverage_percent=100*len(variants['raw_market'])/s.trial(str(year),'ABCD')['pooled']['races'],
            missing_h_policy='zero_for_model;market_history_missing_and_actual_publication_times_preserved_in_source_packets',
            no_future_OOF_in_fit=True)
        s.write(folder,f'{year}_results.json',summary)
        s.write(folder,f'{year}_predictions.local.json',{name:[{'race_key':p.race,'probabilities':p.probabilities} for p in values] for name,values in variants.items()})
        current.extend([folder/f'{year}_results.json',folder/f'{year}_predictions.local.json',folder/f'{year}_arithmetic.json'])
        s.complete(folder,unit,current,{'year':year,'races':len(variants['raw_market']),'configurations':list(variants)})
        files.append(folder/(unit+'.complete.json'));years.append({'year':year,'races':len(variants['raw_market']),'configurations':list(variants)})
        print(json.dumps({'status':'running','stage':'integrate','year':year,'races':len(variants['raw_market']),'arithmetic_weight':w}),flush=True)
    summary={'years':years,'optimizations':len(list((directory/'optimizations').glob('*.json')))//2,
        'new_LightGBM_fits':0,'base_ability_unchanged':True,'unseen_2026_evaluated':False}
    s.complete(directory,'integrate',files,summary);return summary


def saved(directory,year):
    raw=s.read(directory/'integration',f'{year}_predictions.local.json')
    from ..win_sim.core import PredictionRace
    return {name:[PredictionRace(tuple(x['race_key']),x['probabilities']) for x in values] for name,values in raw.items()}
