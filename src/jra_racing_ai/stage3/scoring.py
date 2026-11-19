from collections import defaultdict
from datetime import date,timedelta
import math
import numpy as np
from ..ml import safety
from ..research import experiment as exp,metrics as previous
from . import contracts as c

measure=previous.measure


def align(first,second):
    a={p.race:p for p in first};b={p.race:p for p in second}
    if len(a)!=len(first) or len(b)!=len(second) or set(a)!=set(b):raise safety.MLFailure('invalid_identity')
    for race in a:
        if set(a[race].probabilities)!=set(b[race].probabilities):raise safety.MLFailure('invalid_identity')
    return a,b


def compare(new,baseline,winners,*,confidence=.95,repetitions=c.REPETITIONS):
    a,b=align(new,baseline);weeks=defaultdict(lambda:np.zeros(7))
    for race in sorted(a):
        x=a[race].probabilities;y=b[race].probabilities;win=winners[race]
        if win not in x:raise safety.MLFailure('invalid_identity')
        d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));week=(d-timedelta(days=d.weekday())).isoformat()
        row=np.zeros(7);row[0]=-math.log(max(x[win],1e-15))+math.log(max(y[win],1e-15));row[1]=1
        for h in x:
            px=min(1-1e-15,max(1e-15,x[h]));py=min(1-1e-15,max(1e-15,y[h]));z=int(h==win)
            row[2]+=-(z*math.log(px)+(1-z)*math.log1p(-px))+(z*math.log(py)+(1-z)*math.log1p(-py))
            row[3]+=(px-z)**2-(py-z)**2;row[4]+=1
        tx=[h for h in x if x[h]==max(x.values())];ty=[h for h in y if y[h]==max(y.values())]
        row[5]=float(win in tx)/len(tx)-float(win in ty)/len(ty);row[6]=1
        weeks[week]+=row
    if not weeks:raise safety.MLFailure('insufficient_history')
    mat=np.asarray([v for _,v in sorted(weeks.items())]);rng=np.random.default_rng(c.SEED)
    sampled=rng.integers(len(mat),size=(repetitions,len(mat)));sums=mat[sampled].sum(axis=1);total=mat.sum(axis=0)
    output={'resampling_unit':'Monday_starting_race_week_all_venues','repetitions':repetitions,'seed':c.SEED,'confidence':confidence,'weeks':len(mat),'races':len(a)}
    alpha=(1-confidence)/2
    for name,n,d,scale in [('race_log_loss',0,1,1),('horse_log_loss',2,4,1),('brier_score',3,4,1),('top1_accuracy_percentage_points',5,6,100)]:
        lo,hi=np.quantile(scale*sums[:,n]/sums[:,d],[alpha,1-alpha]);delta=scale*total[n]/total[d]
        output[name]={'difference_new_minus_baseline':float(delta),'ci_low':float(lo),'ci_high':float(hi)}
    return output


def choose(scores,tie):
    best=min(row['race_log_loss'] for row in scores)
    return min((row for row in scores if abs(row['race_log_loss']-best)<=1e-12),key=tie)


def evaluate(frame,values):
    return measure(exp.predictions(frame,values),exp.winners(frame))


def flatten(parts):return [p for part in parts for p in part]


def annual_pool(parts_by_year):
    if set(parts_by_year)!={'2023','2024','2025'}:raise safety.MLFailure('invalid_split')
    values=[p for year in ('2023','2024','2025') for p in flatten(parts_by_year[year])]
    if len({p.race for p in values})!=len(values):raise safety.MLFailure('duplicate_entry')
    return values
