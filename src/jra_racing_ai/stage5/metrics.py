"""Exact global ROC-AUC under clustered resampling, not averaged per-week AUC."""
from datetime import date,timedelta
import hashlib
import numpy as np
from sklearn.metrics import roc_auc_score
from ..ml import safety
from ..stage3.scoring import align
from . import contracts as c

NAMES=('roc_auc','within_race_auc','race_log_loss','horse_log_loss','brier_score','top1_accuracy_percent')


def auc(y,p,weights=None):
    y=np.asarray(y);p=np.asarray(p,float)
    if y.ndim!=1 or p.shape!=y.shape or not np.isin(y,[0,1]).all() or set(y)!={0,1} or not np.isfinite(p).all():
        raise safety.MLFailure('invalid_input')
    if weights is not None:
        weights=np.asarray(weights,float)
        if weights.shape!=y.shape or not np.isfinite(weights).all() or (weights<0).any() or not all(weights[y==k].sum()>0 for k in (0,1)):
            raise safety.MLFailure('invalid_input')
    return float(roc_auc_score(y,p,sample_weight=weights))


def assemble(frame,predictions):
    if frame.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    y=frame.winner_label.to_numpy(float)
    if not np.isin(y,[0,1]).all():raise safety.MLFailure('invalid_input')
    races=frame.entry_key.str[:16].to_numpy(dtype=str);unique,codes=np.unique(races,return_inverse=True)
    if not (np.bincount(codes,weights=y)==1).all() or not (np.bincount(codes)>=2).all():raise safety.MLFailure('invalid_identity')
    mapped={};seen=set()
    for p in predictions:
        if p.race in seen:raise safety.MLFailure('duplicate_entry')
        seen.add(p.race);values=np.asarray(list(p.probabilities.values()),float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any() or not np.isclose(values.sum(),1.,atol=1e-12,rtol=0.):
            raise safety.MLFailure('invalid_input')
        for horse,value in p.probabilities.items():mapped[''.join(p.race)+horse]=value
    if set(mapped)!=set(frame.entry_key):raise safety.MLFailure('invalid_identity')
    p=np.asarray([mapped[k] for k in frame.entry_key],float)
    weeks=[]
    for key,day in zip(frame.entry_key,frame.date):
        d=date.fromisoformat(str(day));c.guard(d,d)
        if d.year not in c.YEARS or d.month>9 or str(key)[:8]!=d.strftime('%Y%m%d'):raise safety.MLFailure('invalid_split')
        weeks.append((d-timedelta(days=d.weekday())).isoformat())
    return y.astype(int),p,codes,np.asarray(weeks,dtype=str)


def same_labels(first,second):
    if first.entry_key.duplicated().any() or second.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    a=first.set_index('entry_key')[['date','winner_label']].sort_index();b=second.set_index('entry_key')[['date','winner_label']].sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.date.to_numpy(str),b.date.to_numpy(str)) or not np.array_equal(a.winner_label.to_numpy(float),b.winner_label.to_numpy(float)):
        raise safety.MLFailure('invalid_identity')


def point_only(frame,predictions):
    y,p,races,weeks=assemble(frame,predictions);ws,code=np.unique(weeks,return_inverse=True);v=statistics(y,p,races,code,len(ws)).sum(axis=0)
    return {'roc_auc':auc(y,p),'within_race_auc':float(v[6]/v[0]),'race_log_loss':float(v[2]/v[0]),
        'horse_log_loss':float(v[3]/v[1]),'brier_score':float(v[4]/v[1]),'top1_accuracy_percent':float(100*v[5]/v[0]),
        'races':int(v[0]),'runners':int(v[1])}


def pair_matrix(y,p,week_codes,k):
    """U[a,b]: positive in week a outranks negative in week b (ties half)."""
    auc(y,p);positive=p[y==1];posweek=week_codes[y==1];out=np.zeros((k,k))
    for b in range(k):
        neg=np.sort(p[(y==0)&(week_codes==b)])
        below=np.searchsorted(neg,positive,side='left');upper=np.searchsorted(neg,positive,side='right')
        out[:,b]=np.bincount(posweek,weights=(below+upper)/2,minlength=k)
    return out


def statistics(y,p,race_codes,week_codes,k):
    # [race count, runner count, race NLL, binary LL, Brier, top1 credit, within-race AUC]
    stats=np.zeros((k,7));clip=np.clip(p,1e-15,1-1e-15)
    stats[:,1]=np.bincount(week_codes,minlength=k)
    stats[:,3]=np.bincount(week_codes,weights=-y*np.log(clip)-(1-y)*np.log1p(-clip),minlength=k)
    stats[:,4]=np.bincount(week_codes,weights=(p-y)**2,minlength=k)
    order=np.argsort(race_codes,kind='stable');ordered=race_codes[order];borders=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(order)]
    for begin,end in zip(borders[:-1],borders[1:]):
        idx=order[begin:end];values=p[idx];labels=y[idx]
        winner=np.flatnonzero(labels==1)
        if len(winner)!=1 or len(set(week_codes[idx]))!=1:raise safety.MLFailure('invalid_identity')
        win=values[winner[0]];week=week_codes[idx[0]];neg=values[labels==0];tops=values==values.max()
        stats[week,0]+=1;stats[week,2]+=-np.log(max(win,1e-15))
        stats[week,5]+=float(tops[winner[0]])/int(tops.sum())
        stats[week,6]+=(np.count_nonzero(win>neg)+.5*np.count_nonzero(win==neg))/len(neg)
    return stats


def bounds(draw):
    a=np.asarray(draw,float);a=a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a,[.025,.975])] if len(a) else None


class WeekBootstrap:
    def __init__(self,weeks,*,repetitions=c.REPETITIONS,sample_indices=None):
        self.weeks,self.codes=np.unique(weeks,return_inverse=True);k=len(self.weeks)
        self.sample=np.random.default_rng(c.SEED).integers(k,size=(repetitions,k)) if sample_indices is None else np.asarray(sample_indices,dtype=int)
        if self.sample.ndim!=2 or self.sample.shape[1]!=k or ((self.sample<0)|(self.sample>=k)).any():raise safety.MLFailure('invalid_input')
        self.counts=np.asarray([np.bincount(row,minlength=k) for row in self.sample],float)
        self.fingerprint=hashlib.sha256(self.sample.tobytes()).hexdigest()

    def measure(self,y,p,race_codes):
        k=len(self.weeks);y=np.asarray(y);p=np.asarray(p,float)
        if len(y)!=len(self.codes) or len(p)!=len(y):raise safety.MLFailure('invalid_identity')
        u=pair_matrix(y,p,self.codes,k);pos=np.bincount(self.codes,weights=y,minlength=k);neg=np.bincount(self.codes,weights=1-y,minlength=k)
        base=auc(y,p);pair_point=float(u.sum()/pos.sum()/neg.sum())
        if abs(base-pair_point)>1e-12:raise safety.MLFailure('artifact_invalid')
        den=(self.counts@pos)*(self.counts@neg);num=np.sum((self.counts@u)*self.counts,axis=1)
        au=np.divide(num,den,out=np.full(len(den),np.nan),where=den>0)
        stats=statistics(y,p,race_codes,self.codes,k);total=stats.sum(axis=0);draw=self.counts@stats
        point={'roc_auc':base};rep={'roc_auc':au}
        for name,n,d,scale in (('within_race_auc',6,0,1),('race_log_loss',2,0,1),('horse_log_loss',3,1,1),('brier_score',4,1,1),('top1_accuracy_percent',5,0,100)):
            point[name]=float(scale*total[n]/total[d]);rep[name]=scale*draw[:,n]/draw[:,d]
        return point,rep

    def metadata(self):
        return {'weeks':len(self.weeks),'seed':c.SEED,'repetitions':len(self.sample),'confidence':.95,
            'resampling_unit':'Monday_starting_week_all_venues','sample_indices_sha256':self.fingerprint,
            'AUC_method':'exact_global_cross_week_U_with_bootstrap_week_multiplicities;ties_half;not_mean_of_week_AUC',
            'all_models_and_metrics_share_draws':True}


def compare(point,rep,base,basedraw):
    return {name:{'difference_new_minus_H':float(point[name]-base[name]),'95_percent_ci':bounds(rep[name]-basedraw[name]),
        'positive_is_improvement':name in ('roc_auc','within_race_auc','top1_accuracy_percent')} for name in NAMES}
