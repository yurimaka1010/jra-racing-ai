from datetime import date
import numpy as np
from ..ml import safety
from ..research import experiment as exp,model as vm,market_data
from . import contracts as c,store as s,scoring


def load_origin(fold,parent):
    origin=s.origin(fold,parent)
    bundle=vm.load(c.SOURCE/'models'/origin['model_id'])
    return bundle,origin


def shrink(frame,bundle,coefficient):
    spec=bundle['metadata']['feature_spec']
    x=vm.matrix(frame,spec,bundle['metadata']['categories'])
    raw=bundle['boosters']['win'].predict(x,raw_score=True,num_threads=4)
    return vm.normalize(frame,vm.sigmoid(vm.offset(frame['market_probability'])+coefficient*raw))


def run(directory):
    done=s.completed(directory,'reuse')
    if done:return done['summary']
    folder=directory/'reuse';folder.mkdir(exist_ok=True)
    results=[];files=[];max_reload=0.;max_metric=0.;parity=0.
    for fold in ('main','2023','2024','2025'):
        sp=s.trial(fold,'ABCD')['split'];parts,evals=exp.partition(c.SOURCE,sp)
        ability,origin=load_origin(fold,'ABCD')
        raw_cal=vm.uncalibrated(ability,parts['calibration']);oldT=ability['metadata']['calibration']['temperature']
        scores=[dict(temperature=t,**scoring.evaluate(parts['calibration'],vm.normalize(parts['calibration'],raw_cal,t))) for t in c.TEMPERATURE]
        selected=scoring.choose(scores,lambda x:(abs(x['temperature']-oldT),x['temperature']))
        output=[];periods=[]
        saved=exp.saved_predictions(c.SOURCE,s.trial(fold,'ABCD'))
        for i,frame in enumerate(evals):
            raw=vm.uncalibrated(ability,frame)
            expected=exp.predictions(frame,vm.normalize(frame,raw,oldT));a,b=scoring.align(expected,saved[i])
            max_reload=max(max_reload,*[abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities])
            points=scoring.measure(expected,exp.winners(frame));old=s.trial(fold,'ABCD')['periods'][i]['ability_population']
            max_metric=max(max_metric,*[abs(points[k]-old[k]) for k in ('race_log_loss','log_loss','brier_score','top1_accuracy_percent')])
            p=exp.predictions(frame,vm.normalize(frame,raw,selected['temperature']));output.append(p)
            periods.append({'metrics':scoring.measure(p,exp.winners(frame)),'versus_Stage2_ABCD':scoring.compare(p,expected,exp.winners(frame))})
        record=dict(s.provenance(directory,'expanded_temperature',fold,parents=[origin],split=sp),
            old_temperature=oldT,selected_temperature=selected['temperature'],boundary=selected['temperature'] in (c.TEMPERATURE[0],c.TEMPERATURE[-1]),
            calibration_scores=scores,periods=periods,pooled=scoring.measure(scoring.flatten(output),{k:v for f in evals for k,v in exp.winners(f).items()}))
        name=f'{fold}_temperature.json';s.write(folder,name,record);s.write_predictions(folder,f'{fold}_temperature.predictions.local.json',output)
        files.extend([folder/name,folder/f'{fold}_temperature.predictions.local.json']);results.append({'fold':fold,'kind':'temperature','selected':selected['temperature']})
        del parts
        # Both parents share the same market cohort, independent of early-quote missingness.
        asp=dict(sp,train_start='2020-01-01');aparts,aevals=exp.partition(c.SOURCE,asp,anchor=True)
        for parent in ('market_history','market_ABCD_history'):
            bundle,origin=load_origin(fold,parent);cal=aparts['calibration']
            scores=[dict(coefficient=l,**scoring.evaluate(cal,shrink(cal,bundle,l))) for l in c.LAMBDA]
            selected=scoring.choose(scores,lambda x:x['coefficient']);output=[];periods=[]
            saved=exp.saved_predictions(c.SOURCE,s.trial(fold,parent))
            for i,frame in enumerate(aevals):
                p0=shrink(frame,bundle,0.);p1=shrink(frame,bundle,1.)
                parity=max(parity,float(np.max(np.abs(p0-frame['market_probability'].to_numpy()))))
                a,b=scoring.align(exp.predictions(frame,p1),saved[i])
                max_reload=max(max_reload,*[abs(a[r].probabilities[h]-b[r].probabilities[h]) for r in a for h in a[r].probabilities])
                points=scoring.evaluate(frame,p1);old=s.trial(fold,parent)['periods'][i]['ability_population']
                max_metric=max(max_metric,*[abs(points[k]-old[k]) for k in ('race_log_loss','log_loss','brier_score','top1_accuracy_percent')])
                values=shrink(frame,bundle,selected['coefficient']);p=exp.predictions(frame,values);output.append(p)
                periods.append({'metrics':scoring.measure(p,exp.winners(frame)),
                    'all_fixed_lambda_metrics':[dict(coefficient=l,**scoring.evaluate(frame,shrink(frame,bundle,l))) for l in c.LAMBDA],
                    'versus_raw_market':scoring.compare(p,exp.predictions(frame,p0),exp.winners(frame))})
            record=dict(s.provenance(directory,'lambda_'+parent,fold,parents=[origin],split=asp),
                selected_lambda=selected['coefficient'],calibration_scores=scores,periods=periods,
                pooled=scoring.measure(scoring.flatten(output),{k:v for f in aevals for k,v in exp.winners(f).items()}))
            name=f'{fold}_lambda_{parent}.json';s.write(folder,name,record)
            s.write_predictions(folder,f'{fold}_lambda_{parent}.predictions.local.json',output)
            files.extend([folder/name,folder/f'{fold}_lambda_{parent}.predictions.local.json']);results.append({'fold':fold,'kind':'lambda_'+parent,'selected':selected['coefficient']})
        print(__import__('json').dumps({'status':'running','stage':'reuse','fold':fold,'reproduction_max_difference':max_reload}),flush=True)
    if max_reload>1e-12 or max_metric>1e-12 or parity>1e-12:raise safety.MLFailure('artifact_invalid')
    summary={'new_LightGBM_fits':0,'lambda_evaluations':40,'fold_selections':results,
        'prediction_reproduction_max_difference':max_reload,'metric_reproduction_max_difference':max_metric,'lambda_zero_market_max_difference':parity}
    s.complete(directory,'reuse',files,summary)
    return summary
