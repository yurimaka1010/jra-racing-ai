"""Two centered-log-probability configurations; exactly twelve finite optimizations."""
from datetime import date
import json
import numpy as np
from scipy.optimize import minimize
from ..ml import safety
from ..stage3.softmax import normalized_scores
from ..stage3 import scoring
from ..research import experiment as exp
from . import contracts as c,store as s,oof,data


def inputs(frame):
    data.dates(frame)
    if frame.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    keys,codes=np.unique(frame.entry_key.str[:16].to_numpy(dtype=str),return_inverse=True)
    h=frame.history_probability.to_numpy(dtype=float);a=frame.ability_probability.to_numpy(dtype=float)
    for p in (h,a):
        if not np.isfinite(p).all() or not ((p>0)&(p<1)).all() or not np.allclose(np.bincount(codes,weights=p),1.,atol=1e-12,rtol=0.):
            raise safety.MLFailure('invalid_input')
    counts=np.bincount(codes);logs=np.column_stack((np.log(h),np.log(a)))
    centered=logs-np.column_stack([np.bincount(codes,weights=logs[:,i])/counts for i in range(2)])[codes]
    return logs[:,0],centered,codes,keys


def scales(frame):
    _,x,codes,_=inputs(frame);n=np.bincount(codes)
    values=np.sqrt([np.mean(np.bincount(codes,weights=x[:,i]**2)/n) for i in range(2)])
    if not np.isfinite(values).all() or not (values>1e-12).all():raise safety.MLFailure('training_failed')
    return values


def design(frame,config,scale):
    if config not in ('Hcal','Hstack'):raise safety.MLFailure('invalid_model')
    scale=np.asarray(scale,dtype=float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or not (scale>1e-12).all():raise safety.MLFailure('invalid_model')
    base,x,codes,keys=inputs(frame)
    return base,(x/scale)[:,:1 if config=='Hcal' else 2],codes,keys


def targets(frame,codes):
    y=frame.winner_label.to_numpy(dtype=float)
    if not np.isin(y,[0,1]).all() or not (np.bincount(codes,weights=y)==1).all():raise safety.MLFailure('invalid_input')
    return y


def loss_gradient(theta,base,x,codes,y,strength):
    p,logs=normalized_scores(base+x@theta,codes);n=codes.max()+1
    loss=-float(y@logs)/n+strength/2*float(theta@theta)
    gradient=x.T@(p-y)/n+strength*theta
    if not np.isfinite(loss) or not np.isfinite(gradient).all():raise safety.MLFailure('training_failed')
    return loss,gradient


def predict(frame,doc):
    if doc['version']!=c.VERSION:raise safety.MLFailure('invalid_model')
    base,x,codes,_=design(frame,doc['config_id'],doc['scales'])
    theta=np.asarray(doc['coefficients'],dtype=float)
    if theta.shape!=(x.shape[1],) or not np.isfinite(theta).all():raise safety.MLFailure('invalid_model')
    return normalized_scores(base+x@theta,codes)[0]


def divergence(frame):
    _,_,codes,keys=inputs(frame);a=frame.ability_probability.to_numpy(float);h=frame.history_probability.to_numpy(float);mid=(a+h)/2
    return keys,np.bincount(codes,weights=.5*(a*np.log(a/mid)+h*np.log(h/mid)))


def boundaries(frame):
    _,values=divergence(frame)
    q=np.quantile(values,[.5,.9],method='linear')
    return {'q50':float(q[0]),'q90':float(q[1]),'method':'linear','source':'direct_training_OOF_only',
        'races':len(values),'first_date':str(frame.date.min()),'last_date':str(frame.date.max()),
        'ties':'low<q50;middle>=q50_and<q90;high>=q90'}


def stratum(value,thresholds):
    return 'low' if value<thresholds['q50'] else 'middle' if value<thresholds['q90'] else 'high'


def fit(directory,config,year,train,validation,strength,split):
    if config not in ('Hcal','Hstack') or strength not in c.STRENGTHS or year not in c.YEARS or split!=c.direct_split(year):
        raise safety.MLFailure('invalid_input')
    folder=directory/'optimizations';folder.mkdir(exist_ok=True);unit=f'{year}_{config}_{strength}'
    path=folder/(unit+'.json')
    if path.exists():return s.read(folder,path.name)
    for part,name in ((train,'train'),(validation,'validation')):
        c.guard(date.fromisoformat(str(part.date.min())),date.fromisoformat(str(part.date.max())))
        if str(part.date.min())<split[name+'_start'] or str(part.date.max())>split[name+'_end']:raise safety.MLFailure('invalid_split')
    if not split['train_end']<split['validation_start']<=split['validation_end']<split['evaluation'][0][0]:raise safety.MLFailure('invalid_split')
    if (folder/(unit+'.started.json')).exists() or len(list(folder.glob('*.started.json')))>=12:raise safety.MLFailure('training_failed')
    origin=s.provenance(directory,config,year,split=split,parents=[s.digest(directory/'oof_origins.json')])
    s.write(folder,unit+'.started.json',dict(origin,status='started'))
    try:
        scale=scales(train);base,x,codes,_=design(train,config,scale);y=targets(train,codes)
        result=minimize(loss_gradient,np.zeros(x.shape[1]),args=(base,x,codes,y,strength),jac=True,method='L-BFGS-B',
            options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
        finite=bool(np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all())
        doc=dict(origin,config_id=config,strength=strength,scales=scale.tolist(),coefficients=result.x.tolist() if finite else None,
            status='complete' if finite and result.success else 'failed',solver_status=int(result.status),iterations=int(result.nit),
            objective=float(result.fun) if finite else None,gradient_norm=float(np.linalg.norm(result.jac)) if finite else None,
            train_races=int(codes.max()+1),train_rows=len(train),validation_rows=len(validation),
            base_H_ids=sorted(set(train.source_H_model_id)|set(validation.source_H_model_id)),
            base_ABCD_ids=sorted(set(train.ability_model_id)|set(validation.ability_model_id)),
            terms=['centered_log_pH/sH']+(['centered_log_pA/sA'] if config=='Hstack' else []),
            scales_source='training_OOF_only_equal_race_mean_square',no_refit=True)
        if doc['status']=='complete':doc['validation_metrics']=scoring.evaluate(validation,predict(validation,doc))
    except (ValueError,ArithmeticError,safety.MLFailure):
        doc=dict(origin,strength=strength,status='failed',reason='zero_scale_nonfinite_or_optimizer_failure')
    s.write(folder,path.name,doc);return doc


def document(predictions):return [{'race_key':p.race,'probabilities':p.probabilities} for p in predictions]


def run(directory):
    if not s.completed(directory,'oof'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'direct')
    if done:return done['summary']
    folder=directory/'direct';folder.mkdir(exist_ok=True);summary=[];files=[]
    for year in c.YEARS:
        cached=s.completed(folder,str(year))
        if cached:summary.append(cached['summary']);files.append(folder/f'{year}.complete.json');continue
        split=c.direct_split(year)
        train=oof.combined(directory,date.fromisoformat(split['train_start']),date.fromisoformat(split['train_end']))
        validation=oof.combined(directory,date.fromisoformat(split['validation_start']),date.fromisoformat(split['validation_end']))
        threshold=boundaries(train);s.write(folder,f'{year}_boundaries.json',threshold)
        f=oof.combined(directory,date(year,1,1),date(year,9,30));selected={};all_trials=[];outputs={}
        for config in ('Hcal','Hstack'):
            trials=[fit(directory,config,year,train,validation,st,split) for st in c.STRENGTHS];all_trials.extend(trials)
            good=[t for t in trials if t['status']=='complete']
            if not good:selected[config]={'status':'failed','reason':'no_finite_optimizer'};continue
            chosen=scoring.choose([dict(t,race_log_loss=t['validation_metrics']['race_log_loss']) for t in good],tie=lambda t:-t['strength'])
            selected[config]=chosen
            p=predict(f,chosen);outputs[config]=document(exp.predictions(f,p))
            # Saved coefficients/scales reproduce predictions; coefficients zero reproduce pH.
            zero=dict(chosen,coefficients=[0.]*len(chosen['coefficients']))
            zero_delta=float(np.max(np.abs(predict(f,zero)-f.history_probability.to_numpy(float))))
            if zero_delta>1e-12:raise safety.MLFailure('artifact_invalid')
            selected[config]['zero_coefficient_max_difference']=zero_delta
        s.write(folder,f'{year}_selection.json',{'split':split,'selected':selected,'boundaries':threshold,
            'selection_reason':'minimum_validation_race_LL;tie_strength_.1;no_refit',
            'origin_shift':'integration_OOF_2021_H_has_8_month_training;later_eval_H_has_longer_history;not_corrected_with_extra_fits'})
        s.write(folder,f'{year}_predictions.local.json',outputs)
        local_files=[folder/f'{year}_selection.json',folder/f'{year}_predictions.local.json',folder/f'{year}_boundaries.json']
        local_files.extend(directory/'optimizations'/f'{year}_{cfg}_{st}.json' for cfg in ('Hcal','Hstack') for st in c.STRENGTHS)
        value={'year':year,'races':int(f.entry_key.str[:16].nunique()),'rows':len(f),'optimizations':len(all_trials),
            'failures':sum(t['status']!='complete' for t in all_trials),'selected':{k:dict(strength=v.get('strength'),coefficients=v.get('coefficients')) for k,v in selected.items()},
            'boundary_q50':threshold['q50'],'boundary_q90':threshold['q90']}
        s.complete(folder,str(year),local_files,value);summary.append(value);files.append(folder/f'{year}.complete.json')
        print(json.dumps(dict(status='running',stage='direct',**value)),flush=True)
    result={'years':summary,'optimizations':sum(v['optimizations'] for v in summary),'failures':sum(v['failures'] for v in summary),
        'no_evaluation_refit':True,'ABCD_E_not_used':True}
    s.complete(directory,'direct',files,result);return result
