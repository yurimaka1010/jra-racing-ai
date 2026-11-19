"""Future-only pure adapter. No label loader/DB path or interim scoring command."""
from datetime import date,timedelta
import numpy as np
from ..ml import safety
from ..win_sim import core
from ..stage5 import metrics

def evaluate(records,results,plan,now,*,repetitions=5000):
    if now.date()<=date.fromisoformat(plan['end_date']):raise safety.MLFailure('forward_cutoff')
    ys=[];hs=[];qs=[];races=[];weeks=[];excluded={};race_index=0
    for record in records:
        race=core.key(record['race_key']);d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
        if not date.fromisoformat(plan['start_date'])<=d<=date.fromisoformat(plan['end_date']) or d<core.instant(plan['created_at']).date():
            raise safety.MLFailure('invalid_split')
        result=results.get(race)
        reason='result_unconfirmed'
        if result and result.get('status')=='normal_final' and len(result.get('winners',[]))==1:
            if set(record['roster'])!=set(result['horses']):reason='post_save_roster_change'
            elif result['winners'][0] not in record['roster']:reason='invalid_winner'
            elif result.get('actual_start') and core.instant(record['save_completed_at'])>=core.instant(result['actual_start']):reason='saved_after_actual_start'
            else:reason=None
        if reason:excluded[reason]=excluded.get(reason,0)+1;continue
        p=record['probabilities'];roster=record['roster']
        if any(set(p[n])!=set(roster) or abs(sum(p[n].values())-1)>1e-12 for n in ('H','raw_market_T10')):raise safety.MLFailure('invalid_identity')
        for horse in roster:
            ys.append(int(horse==result['winners'][0]));hs.append(p['H'][horse]);qs.append(p['raw_market_T10'][horse]);races.append(race_index)
            weeks.append((d-timedelta(days=d.weekday())).isoformat())
        race_index+=1
    if not ys:return {'status':'unavailable','excluded':excluded,'races':0}
    bootstrap=metrics.WeekBootstrap(np.asarray(weeks),repetitions=repetitions)
    y=np.asarray(ys);rs=np.asarray(races);hp,hd=bootstrap.measure(y,np.asarray(hs),rs);qp,qd=bootstrap.measure(y,np.asarray(qs),rs)
    delta=hp['within_race_auc']-qp['within_race_auc'];ci=metrics.bounds(hd['within_race_auc']-qd['within_race_auc'])
    return {'H':hp,'raw_market_T10':qp,'races':race_index,'runners':len(y),'excluded':excluded,
        'difference_H_minus_market':delta,'within_race_auc_95_week_CI':ci,'MPID':.002,
        'practical_discrimination_gain_confirmed':delta>=.002 and ci[0]>0,'uncertain_if_CI_crosses_zero':ci[0]<=0<=ci[1],
        'bootstrap':bootstrap.metadata(),'not_new_model_training':True}
