"""Race likelihood + fixed ridge towards the market, with analytic gradients."""
import numpy as np
from datetime import date
from ..ml import safety
from . import contracts as c,store as s


def normalized_scores(scores,codes):
    scores=np.asarray(scores,dtype=float);codes=np.asarray(codes,dtype=int)
    if not len(scores) or not np.isfinite(scores).all() or len(scores)!=len(codes):raise safety.MLFailure('invalid_input')
    maxima=np.full(codes.max()+1,-np.inf);np.maximum.at(maxima,codes,scores)
    shifted=scores-maxima[codes];e=np.exp(shifted);sums=np.bincount(codes,weights=e)
    logs=shifted-np.log(sums[codes])
    return np.exp(logs),logs


def design(frame,config):
    if config not in c.TERMS:raise safety.MLFailure('invalid_model')
    q=frame.market_probability.to_numpy(dtype=float);a=frame.ability_probability.to_numpy(dtype=float)
    if not ((q>0)&(q<1)&(a>0)&(a<1)).all():raise safety.MLFailure('invalid_input')
    values={'q':np.log(q),'a':np.log(a),'h':frame.market_log_probability_change.fillna(0.).to_numpy(dtype=float)}
    x=np.column_stack([values[name] for name in c.TERMS[config]])
    if not np.isfinite(x).all():raise safety.MLFailure('invalid_input')
    _,codes=np.unique(frame.entry_key.str[:16].to_numpy(dtype=str),return_inverse=True)
    y=frame.winner_label.to_numpy(dtype=float)
    if not np.isin(y,[0,1]).all() or not (np.bincount(codes,weights=y)==1).all():raise safety.MLFailure('invalid_input')
    return x,codes,y


def loss_gradient(theta,x,codes,y,strength):
    p,logs=normalized_scores(x@theta,codes);nr=codes.max()+1
    target=np.zeros(len(theta));target[0]=1.
    loss=-float(y@logs)/nr+strength/2*float((theta-target)@(theta-target))
    gradient=x.T@(p-y)/nr+strength*(theta-target)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():raise safety.MLFailure('training_failed')
    return loss,gradient


def predict(frame,document):
    if document['version']!=c.VERSION or document['config_id'] not in c.TERMS:raise safety.MLFailure('invalid_model')
    if tuple(document['terms'])!=c.TERMS[document['config_id']]:raise safety.MLFailure('invalid_model')
    x,codes,_=design(frame,document['config_id']);theta=np.asarray(document['coefficients'],dtype=float)
    if len(theta)!=x.shape[1]:raise safety.MLFailure('invalid_model')
    return normalized_scores(x@theta,codes)[0]


def fit(directory,config,fold,train,validation,strength,split):
    from scipy.optimize import minimize
    from .scoring import evaluate
    for part,start,end in ((train,split['train_start'],split['train_end']),
                           (validation,split['validation_start'],split['validation_end'])):
        c.guard(date.fromisoformat(start),date.fromisoformat(end))
        if not len(part) or str(part.date.min())<start or str(part.date.max())>end:
            raise safety.MLFailure('invalid_split')
    unit=f'{fold}_{config}_{strength}';folder=directory/'optimizations';folder.mkdir(exist_ok=True)
    if (folder/(unit+'.json')).exists():return s.read(folder,unit+'.json')
    if len(list(folder.glob('*.started.json')))>=24:raise safety.MLFailure('input_limit')
    if (folder/(unit+'.started.json')).exists():raise safety.MLFailure('training_failed')
    s.write(folder,unit+'.started.json',dict(s.provenance(directory,config,fold,split=split),status='started'))
    x,codes,y=design(train,config);theta=np.zeros(x.shape[1]);theta[0]=1.
    try:
        result=minimize(loss_gradient,theta,args=(x,codes,y,strength),jac=True,method='L-BFGS-B',
            options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
    except (ValueError,ArithmeticError,safety.MLFailure):
        value=dict(s.provenance(directory,config,fold,split=split),status='failed',reason='optimizer_nonfinite_or_failed')
        s.write(folder,unit+'.json',value);return value
    finite=bool(np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all())
    value=dict(s.provenance(directory,config,fold,split=split),
        fit_label_available_through=split['validation_end'],terms=list(c.TERMS[config]),strength=strength,
        objective='mean_race_winner_NLL+strength/2*sum((theta-market_reference)^2)',
        market_reference=[1.]+[0.]*(x.shape[1]-1),coefficients=[float(v) for v in result.x] if finite else None,
        status='complete' if result.success and finite else 'failed',
        solver_status=int(result.status),iterations=int(result.nit),gradient_norm=float(np.linalg.norm(result.jac)) if finite else None,
        penalized_training_loss=float(result.fun) if finite else None,training_rows=len(train),training_races=int(codes.max()+1),
        source_OOF_years=sorted(set(train.date.str[:4])),latest_training_date=str(train.date.max()),
        source_OOF_manifest_sha256=s.digest(directory/'oof_origins.json'),
        base_ability_model_ids=sorted(set(train.source_model_id)|set(validation.source_model_id)),
        no_intercept=True,no_auxiliary_terms=True)
    if value['status']=='complete':
        value['validation_metrics']=evaluate(validation,predict(validation,value))
    s.write(folder,unit+'.json',value)
    return value
