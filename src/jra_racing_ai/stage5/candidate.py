from datetime import date
import json
import numpy as np
import pandas as pd
from ..ml import safety
from ..research import experiment as exp
from ..stage4 import data as source,accuracy as old_accuracy
from . import contracts as c,store as s,metrics,place,model


def decide(new,baseline,comparisons,annual,*,consistency=True,highest_existing=None):
    delta=new['roc_auc']-baseline['roc_auc'];ci=comparisons['roc_auc']['95_percent_ci'];directions=[v['roc_auc']['difference_new_minus_H'] for v in annual.values()]
    tests={'pooled_delta_AUC_at_least_.002':delta>=.002,'AUC_delta_95_low_above_zero':ci is not None and ci[0]>0,
        'positive_AUC_years_at_least_2':sum(d>0 for d in directions)>=2,'no_year_delta_leq_minus_.002':all(d>-.002 for d in directions),
        'within_race_AUC_at_least_H':new['within_race_auc']>=baseline['within_race_auc']-1e-12,
        'top1_accuracy_at_least_H':new['top1_accuracy_percent']>=baseline['top1_accuracy_percent']-1e-12,
        'race_LL_at_most_H':new['race_log_loss']<=baseline['race_log_loss']+1e-12,
        'Brier_at_most_H':new['brier_score']<=baseline['brier_score']+1e-12,'consistency_verified':consistency}
    accepted=all(tests.values())
    reason='all_development_retention_conditions_met' if accepted else 'AUC_point_estimate_improved_but_overall_retention_conditions_not_met' if delta>0 else 'AUC_improvement_not_observed_and_retention_conditions_not_met'
    return {'retain_development_candidate':accepted,'status':'development_candidate' if accepted else 'not_retained','reason':reason,
        'AUC_difference_vs_fixed_H':delta,'AUC_95_percent_ci':ci,'criteria':tests,'failed_criteria':[k for k,v in tests.items() if not v],
        'existing_highest_AUC':highest_existing,'exceeds_existing_highest_AUC':new['roc_auc']>highest_existing['roc_auc'] if highest_existing else None,
        'highest_existing_comparison_is_descriptive_only':True,'H_is_fixed_formal_comparator':True,'production_adoption':False,
        'point_above_080':new['roc_auc']>.80,'not_independent_final_evaluation':True}


def measure(directory,frame,predictions,scope,base,replicas):
    y,p,races,weeks=metrics.assemble(frame,predictions);boot=metrics.WeekBootstrap(weeks);point,draw=boot.measure(y,p,races)
    if boot.fingerprint!=base['bootstrap']['sample_indices_sha256']:raise safety.MLFailure('invalid_identity')
    original={key:replicas[f'{scope}__H__{key}'] for key in metrics.NAMES}
    return {'metrics':dict(point,roc_auc_95_percent_ci=metrics.bounds(draw['roc_auc']),races=int(frame.entry_key.str[:16].nunique()),runners=len(frame)),
        'versus_H':metrics.compare(point,draw,base['metrics']['H'],original),'bootstrap':boot.metadata()}


def run(directory):
    complete=s.completed(directory,'candidate')
    if complete:return complete
    gate=s.completed(directory,'place')
    if gate is None:raise safety.MLFailure('invalid_input')
    if not gate['fit_allowed']:
        result={'status':'skipped','reason':gate['reason'],'new_LightGBM_fits':0,'candidate':None}
        s.write(directory,'candidate.json',result);s.complete(directory,'candidate',[directory/'candidate.json'],result);return result
    base=s.read(directory,'baseline.json');params=safety.read_json(c.S2/'plan.json',root=c.S2)['anchor_parameters']
    training_plan={'initial_plan_sha256':s.sha(directory/'plan.json'),'gate_sha256':s.sha(directory/'place_gate.json'),
        'mapping_sha256':s.sha(directory/'place_mapping.json'),'baseline_sha256':s.sha(directory/'baseline.json'),
        'config':'H_place_only','max_fits':3,'features':list(c.FEATURES),'parameters':params,'temperature':1.,
        'splits':{str(y):c.split(y) for y in c.YEARS},'main_population_races':6373,'main_population_runners':87601,
        'full_gate_coverage':gate['folds'],'selection_criteria':c.plan()['retain_all_conditions'],'ROI_not_used':True}
    s.write(directory,'training_plan.json',training_plan)
    # This read is side-effect-free and also provides the authoritative saved labels.
    reference_frames,reference_predictions,_=old_accuracy.load(c.S4)
    saved=safety.local_path(directory/'baseline_bootstrap.local.npz',root=c.ROOT,exists=True)
    files=[];trials=[];evaluations={};frames={};predictions={};diagnostics={}
    with np.load(saved,allow_pickle=False) as replicas:
        for year in c.YEARS:
            sp=c.split(year);parts=[]
            for part in ('train','validation','calibration'):
                f=source.market_frame(c.S4,date.fromisoformat(sp[part+'_start']),date.fromisoformat(sp[part+'_end']))
                augmented,_=place.augment(directory,f);parts.append(augmented)
            bundle,trial=model.fit(directory,year,parts,params);trials.append(trial)
            evaluation=source.market_frame(c.S4,date(year,1,1),date(year,9,30));f,_=place.augment(directory,evaluation)
            metrics.same_labels(reference_frames[year],f);frames[year]=f
            name=f'{year}_predictions.local.json'
            if (directory/name).exists():pred=old_accuracy.parse(s.read(directory,name))
            else:
                pred=exp.predictions(f,model.predict(bundle,f));s.write(directory,name,[{'race_key':p.race,'probabilities':p.probabilities} for p in pred])
            # Prediction replay after serialized model loading is exact; labels are not model input.
            actual=exp.predictions(f,model.predict(model.load(directory/'models'/trial['model_id']),f));a,b=old_accuracy.scoring.align(pred,actual)
            difference=max(abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities)
            if difference>1e-12:raise safety.MLFailure('artifact_invalid')
            predictions[year]=pred;evaluations[str(year)]=measure(directory,f,pred,str(year),base['years'][str(year)],replicas)
            parts_diag={}
            for flag,label in ((0,'complete_place'),(1,'missing_place')):
                subset=f.loc[f.place_missing==flag];races=set(subset.entry_key.str[:16])
                if not len(subset):parts_diag[label]={'status':'unavailable','reason':'empty_subset','races':0,'runners':0};continue
                p=[v for v in pred if ''.join(v.race) in races];h=[v for v in reference_predictions[year]['H'] if ''.join(v.race) in races]
                parts_diag[label]={'H_place':metrics.point_only(subset,p),'H':metrics.point_only(subset,h),'not_used_for_retention':True}
            diagnostics[str(year)]=parts_diag
            files.extend([directory/name,directory/f'fit_{year}.complete.json'])
            print(json.dumps(dict(status='running',stage='fit',year=year,best_iteration=trial['best_iteration'],
                ROC_AUC=evaluations[str(year)]['metrics']['roc_auc'],delta_AUC_vs_H=evaluations[str(year)]['versus_H']['roc_auc']['difference_new_minus_H'])),flush=True)
        allframes=pd.concat([frames[y] for y in c.YEARS],ignore_index=True);allpred=[p for y in c.YEARS for p in predictions[y]]
        pooled=measure(directory,allframes,allpred,'pooled',base['pooled'],replicas)
    if (pooled['metrics']['races'],pooled['metrics']['runners'])!=(6373,87601):raise safety.MLFailure('invalid_identity')
    decision=decide(pooled['metrics'],base['pooled']['metrics']['H'],pooled['versus_H'],{y:v['versus_H'] for y,v in evaluations.items()},
        consistency=True,highest_existing=base['pooled']['highest_existing_AUC'])
    result={'pooled':pooled,'years':evaluations,'diagnostic_subsets_not_used_for_retention':diagnostics,'decision':decision,
        'trials':trials,'new_LightGBM_fits':len(trials),'configurations':1,'no_ROI_or_purchase_rule_analysis':True,
        'same_saved_label_and_roster_cohort':True,'all_place_missing_races_retained':True}
    s.write(directory,'candidate.json',result);files.extend([directory/'candidate.json',directory/'training_plan.json'])
    s.complete(directory,'candidate',files,result)
    return {'decision':decision,'new_LightGBM_fits':len(trials),'ROC_AUC':pooled['metrics']['roc_auc']}
