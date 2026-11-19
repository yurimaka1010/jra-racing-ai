"""Accuracy decisions are saved before any Stage 4 profit calculation."""
from collections import defaultdict
from datetime import date,timedelta
import hashlib,json
import numpy as np
from ..ml import safety
from ..research import experiment as exp
from ..stage3 import scoring
from ..win_sim.core import PredictionRace
from . import contracts as c,store as s,data,oof,direct


def parse(values):return [PredictionRace(tuple(p['race_key']),p['probabilities']) for p in values]


def validate_predictions(values):
    seen=set()
    for p in values:
        d=date(int(p.race[0]),int(p.race[1][:2]),int(p.race[1][2:]));c.guard(d,d)
        if d.year not in c.YEARS or d.month>9 or p.race in seen:raise safety.MLFailure('invalid_split')
        seen.add(p.race)
        v=np.asarray(list(p.probabilities.values()),dtype=float)
        if not np.isfinite(v).all() or not (v>0).all() or not np.isclose(v.sum(),1.,atol=1e-12,rtol=0.):raise safety.MLFailure('invalid_input')


def load(directory):
    results={};frames={};winners={}
    for year in c.YEARS:
        f=oof.combined(directory,date(year,1,1),date(year,9,30));frames[year]=f;winners.update(exp.winners(f))
        pred={'H':exp.predictions(f,f.history_probability.to_numpy()),'ABCD':exp.predictions(f,f.ability_probability.to_numpy()),
              'raw_market':exp.predictions(f,f.market_probability.to_numpy())}
        values=s.read(directory/'direct',f'{year}_predictions.local.json')
        for name,rows in values.items():pred[name]=parse(rows)
        h=s.completed(directory/'H60',str(year))
        if h and h['summary']['status']=='complete':pred['H60']=parse(s.read(directory/'H60',f'{year}_predictions.local.json'))
        target={p.race for p in pred['H']}
        eref=s.ref_json(directory,c.S3/'E_trials'/f'{year}.predictions.local.json')
        pred['ABCD_E']=[p for part in eref for p in parse(part) if p.race in target]
        for name,values in pred.items():
            validate_predictions(values);scoring.align(values,pred['H'])
        if len(target)!=c.plan()['expected_year_races'][str(year)]:raise safety.MLFailure('invalid_identity')
        results[year]=pred
    return frames,results,winners


def compare(new,baseline,winners):
    validate_predictions(new);validate_predictions(baseline)
    a=scoring.compare(new,baseline,winners,confidence=.95,repetitions=c.REPETITIONS)
    b=scoring.compare(new,baseline,winners,confidence=c.CORRECTED,repetitions=c.REPETITIONS)
    weeks=sorted({week(p.race) for p in new});indices=np.random.default_rng(c.SEED).integers(len(weeks),size=(c.REPETITIONS,len(weeks)))
    return {'difference_sign':'new_minus_baseline','95_percent':a,'98_333333_percent':b,
        'resample_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'week_order_sha256':hashlib.sha256(json.dumps(weeks).encode()).hexdigest()}


def week(race):
    d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));c.guard(d,d)
    return (d-timedelta(days=d.weekday())).isoformat()


def retain(comparisons,names):
    eligible=True;uncertain=False;detail={}
    for name in names:
        v=comparisons.get(name)
        if not v or 'pooled' not in v:
            eligible=False;detail[name]={'status':'unavailable'};continue
        point=v['pooled']['95_percent']['race_log_loss']['difference_new_minus_baseline']
        directions={year:stats['95_percent']['race_log_loss']['difference_new_minus_baseline'] for year,stats in v['years'].items()}
        enough=len(directions)==3 and sum(x<0 for x in directions.values())>=2
        high=v['pooled']['98_333333_percent']['race_log_loss']['ci_high'];low=v['pooled']['98_333333_percent']['race_log_loss']['ci_low']
        ok=point<0 and enough;eligible&=ok;uncertain|=low<=0<=high
        detail[name]={'pooled_difference':point,'annual_differences':directions,'improved_years':sum(x<0 for x in directions.values()),
            'mean_and_year_direction_pass':ok,'corrected_interval_crosses_zero':low<=0<=high}
    return {'retain_development_candidate':bool(eligible),'status':'uncertain_provisional' if eligible and uncertain else 'development_candidate' if eligible else 'no_candidate',
        'criteria':detail,'not_production_adoption':True,'not_independent_test_evidence':True}


def conditional(directory,frames,results,winners):
    labels=[];groups=defaultdict(lambda:defaultdict(list));yearly={}
    horse_bands={name:{'runners':0,'wins':0,'sum_pA':0.,'sum_pH':0.} for name in ('below_.8','.8_to_1.25','at_least_1.25')}
    for year in c.YEARS:
        threshold=s.read(directory/'direct',f'{year}_boundaries.json');keys,js=direct.divergence(frames[year]);dmap=dict(zip(keys,js))
        am,hm=scoring.align(results[year]['ABCD'],results[year]['H']);counts=defaultdict(int)
        for race in sorted(am):
            a,h=am[race],hm[race];pa,ph=a.probabilities,h.probabilities
            st=direct.stratum(dmap[''.join(race)],threshold)
            topa=min(pa,key=lambda x:(-pa[x],x));toph=min(ph,key=lambda x:(-ph[x],x));agreement='top1_agreement' if topa==toph else 'top1_disagreement'
            row={'race_key':race,'D':float(dmap[''.join(race)]),'stratum':st,'top1_relation':agreement,'threshold_year':year,
                'q50':threshold['q50'],'q90':threshold['q90']};labels.append(row)
            for layer in (st,agreement):
                groups[layer]['ABCD'].append(a);groups[layer]['H'].append(h);counts[layer]+=1
            for number in pa:
                ratio=pa[number]/ph[number];band=ratio_band(ratio);r=horse_bands[band]
                r['runners']+=1;r['wins']+=number==winners[race];r['sum_pA']+=pa[number];r['sum_pH']+=ph[number]
        yearly[str(year)]={'boundaries':threshold,'stratum_counts':dict(counts),'races':len(am)}
    strata={}
    for layer,values in groups.items():
        strata[layer]={'races':len(values['H']),'ABCD':scoring.measure(values['ABCD'],winners),'H':scoring.measure(values['H'],winners),
            'ABCD_minus_H':compare(values['ABCD'],values['H'],winners)}
    for band,r in horse_bands.items():
        n=r['runners'];r.update(mean_pA=r['sum_pA']/n if n else None,mean_pH=r['sum_pH']/n if n else None,actual_win_rate=r['wins']/n if n else None)
        r['mean_pA_over_actual']=r['sum_pA']/r['wins'] if r['wins'] else None;r['mean_pH_over_actual']=r['sum_pH']/r['wins'] if r['wins'] else None
        r['no_subset_renormalization']=True
    s.write(directory,'conditional_labels.local.json',labels)
    return {'annual_boundaries_and_counts':yearly,'pooled_strata':strata,'horse_ratio_calibration':horse_bands,
        'ratio_definition':'pA/pH_not_break_even_ratio','top1_tie_rule':'lowest_horse_number;fractional_ties_in_existing_accuracy_metrics',
        'no_other_subgroup_search':True}


def ratio_band(ratio):return 'below_.8' if ratio<.8 else '.8_to_1.25' if ratio<1.25 else 'at_least_1.25'


def run(directory):
    if not s.completed(directory,'direct') or not s.completed(directory,'h60'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'accuracy')
    if done:return done['summary']
    frames,results,winners=load(directory);names=set.intersection(*(set(r) for r in results.values()))
    pooled={name:[p for year in c.YEARS for p in results[year][name]] for name in sorted(names)}
    if len(pooled['H'])!=6373:raise safety.MLFailure('invalid_identity')
    metrics={'years':{str(y):{k:scoring.measure(v,winners) for k,v in result.items()} for y,result in results.items()},
        'pooled':{k:scoring.measure(v,winners) for k,v in pooled.items()}}
    comparisons={}
    for key,new,base in (('Hstack_minus_H','Hstack','H'),('Hstack_minus_Hcal','Hstack','Hcal'),('H60_minus_H','H60','H'),('Hcal_minus_H','Hcal','H')):
        years={str(y):compare(result[new],result[base],winners) for y,result in results.items() if new in result and base in result}
        comparisons[key]={'years':years,'status':'complete' if len(years)==3 else 'unavailable_full_three_year_comparison'}
        if new in pooled and base in pooled:comparisons[key]['pooled']=compare(pooled[new],pooled[base],winners)
    candidate={'Hstack':retain(comparisons,['Hstack_minus_H','Hstack_minus_Hcal']),'H60':retain(comparisons,['H60_minus_H'])}
    # This file must exist before finance.run is allowed to access HR settlement.
    s.write(directory,'accuracy_candidate_freeze.json',{'version':c.VERSION,'decision_basis':'accuracy_only_before_Stage4_finance',
        'candidates':candidate,'model_adoption':False,'no_2026_evaluation':True})
    s.write(directory,'accuracy.json',{'metrics':metrics,'comparisons':comparisons,'candidates':candidate,
        'coverage':{str(y):{'eligible_races':int(data.frame(directory,date(y,1,1),date(y,9,30)).entry_key.str[:16].nunique()),
            'market_common_races':len(result['H']),'runners':len(frames[y])} for y,result in results.items()},
        'development_reuse_not_independent_final_test':True})
    condition=conditional(directory,frames,results,winners);s.write(directory,'conditional_accuracy.json',condition)
    e={'pooled_ABCD_E':metrics['pooled']['ABCD_E'],'pooled_ABCD':metrics['pooled']['ABCD'],
       'ABCD_E_minus_ABCD':compare(pooled['ABCD_E'],pooled['ABCD'],winners),'new_E_fits':0,'E_candidate_status':'unchanged_no_promotion'}
    s.write(directory,'E_accuracy_supplement.json',e)
    value={'common_races':6373,'runners':sum(len(f) for f in frames.values()),'metrics':{n:{k:v for k,v in metrics['pooled'][n].items() if k!='calibration'} for n in ('H','Hcal','Hstack','H60') if n in metrics['pooled']},
        'accuracy_candidates':candidate,'accuracy_frozen_before_finance':True}
    files=[directory/n for n in ('accuracy.json','accuracy_candidate_freeze.json','conditional_accuracy.json','conditional_labels.local.json','E_accuracy_supplement.json')]
    files.extend(directory/n for n in ('direct.complete.json','h60.complete.json'))
    s.complete(directory,'accuracy',files,value);return value
