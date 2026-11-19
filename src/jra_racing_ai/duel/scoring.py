"""Pure comparisons of immutable forecasts; absent results never become losses."""
from collections import Counter
import numpy as np
from ..ml import safety
from ..stage5 import metrics
from ..win_sim import core
from .replay import DAYS,VENUES,SAVED_MODELS,race_day

NONSTARTERS={'1','2','3'}
NONFINISH={'4','5'}
REASONS={'1':'出走取消','2':'発走除外','3':'競走除外','4':'競走中止','5':'失格','6':'落馬再騎乗','7':'降着'}


def ranks(probabilities):
    if not probabilities:return {},None
    if not all(np.isfinite(p) and 0<=p<=1 for p in probabilities.values()) or abs(sum(probabilities.values())-1)>1e-12:
        raise safety.MLFailure('invalid_input')
    order=sorted(probabilities,key=lambda h:(-probabilities[h],int(h)))
    return {h:i+1 for i,h in enumerate(order)},order[0]


def race_view(item,result_rows,outcome):
    race=tuple(item['race_key']);race_day(race);roster=item['roster'];rows={};duplicates=False
    for row in result_rows:
        h=row['horse_number']
        if h in rows:duplicates=True
        rows[h]=row
    result_ok=not duplicates and set(rows)==set(roster) and all(r.get('status')=='7' and not r.get('oversized') and r.get('horse_id')==roster[h]['horse_id'] for h,r in rows.items())
    model_views={};horse_views=[];parsed={}
    for h in sorted(set(roster)|set(rows)):
        row=rows.get(h,{});value=row.get('rank');anomaly=row.get('anomaly')
        rank=int(value) if result_ok and isinstance(value,str) and len(value)==2 and value.isascii() and value.isdigit() and 1<=int(value)<=28 else None
        started=result_ok and anomaly in ('0','4','5','6','7')
        label=str(rank) if rank is not None else REASONS.get(anomaly,'未確定・欠損') if result_ok else '未確定・欠損'
        parsed[h]={'rank':rank,'anomaly':anomaly,'started':started,'label':label,
            'top_evaluable':bool(started and (rank is not None or anomaly in NONFINISH))}
        horse_views.append({'number':h,'name':row.get('name') or '',**parsed[h]})
    winners=[h for h,v in parsed.items() if v['rank']==1]
    auc_allowed=result_ok and len(winners)==1 and all(v['started'] and (v['rank'] is not None or v['anomaly'] in NONFINISH) for v in parsed.values())
    for name in ('saved','H'):
        forecast=item[name];p=forecast['probabilities'];order,top=ranks(p)
        value={'status':forecast['status'],'reason':forecast.get('reason'),'top':top,'probabilities':p,'ranks':order,
            'top_result':parsed.get(top,{}),'within_race_auc':None,'winner_predicted_rank':None,'paper':{'status':'pending','payout_yen':None}}
        if auc_allowed and set(p)==set(parsed):
            horses=sorted(p);labels=np.asarray([int(h==winners[0]) for h in horses]);prob=np.asarray([p[h] for h in horses])
            values=metrics.statistics(labels,prob,np.zeros(len(horses),int),np.zeros(len(horses),int),1)
            value['within_race_auc']=float(values[0,6])
            value['winner_predicted_rank']=order[winners[0]]
        if top and result_ok and outcome:
            o=core.Outcome(race,outcome['status'],outcome['payouts_yen_per_100'],outcome['verified'],outcome['complete_winner_set'],outcome.get('reason'))
            # HR and SE winner sets must agree before treating a normal loss as zero.
            if set(winners)==set(o.payouts_yen_per_100) and parsed[top]['started']:
                value['paper']=core.settle(top,o,tuple(p))
        model_views[name]=value
    common_reason=None
    for name in ('saved','H'):
        if model_views[name]['status']!='ready':common_reason='missing_'+name;break
    if common_reason is None and set(model_views['saved']['probabilities'])!=set(model_views['H']['probabilities']):common_reason='prediction_roster_mismatch'
    if common_reason is None and not result_ok:common_reason='result_or_identity_unconfirmed'
    if common_reason is None and any(not model_views[n]['top_result'].get('top_evaluable') for n in model_views):common_reason='top_nonstarter_or_unconfirmed'
    common=common_reason is None;a,b=model_views['saved']['top'],model_views['H']['top']
    relation='unavailable'
    if common:
        if a==b:relation='agree'
        elif parsed[a]['rank'] is None or parsed[b]['rank'] is None:relation='rank_unavailable'
        elif parsed[a]['rank']<parsed[b]['rank']:relation='saved_ahead'
        elif parsed[a]['rank']>parsed[b]['rank']:relation='H_ahead'
        else:relation='same_finish'
    for row in horse_views:
        for name,value in model_views.items():
            row[name+'_probability']=value['probabilities'].get(row['number']);row[name+'_rank']=value['ranks'].get(row['number']);row[name+'_top']=value['top']==row['number']
    return {'id':''.join(race),'date':item['date'],'venue':item['venue'],'venue_code':race[2],'race_number':item['race_number'],'start':item['start'],
        'comparison_id':item['comparison_id'],'models':model_views,'forecast_metadata':{n:{k:v for k,v in item[n].items() if k!='probabilities'} for n in ('saved','H')},
        'horses':horse_views,'result_ok':result_ok,'auc_eligible':auc_allowed,'common':common,'common_reason':common_reason,
        'top_agreement':bool(common and a==b),'relation':relation,'dead_heat':len(winners)>1,
        'nonstarter_count':sum(v['anomaly'] in NONSTARTERS for v in parsed.values()) if result_ok else None}


def model_summary(races,name):
    forecasts=[r for r in races if r['models'][name]['status']=='ready'];tops=[r['models'][name] for r in forecasts]
    evaluable=[v for v in tops if v['top_result'].get('top_evaluable')];distribution=Counter(v['top_result']['label'] for v in evaluable)
    excluded=Counter(v['top_result'].get('label','結果未取得') for v in tops if not v['top_result'].get('top_evaluable'))
    n=len(evaluable);rates={f'rank{k}_rate':100*sum(v['top_result']['rank']==k for v in evaluable)/n if n else None for k in (1,2,3)}
    auc=[v['within_race_auc'] for v in tops if v['within_race_auc'] is not None];winner_ranks=Counter(str(v['winner_predicted_rank']) for v in tops if v['winner_predicted_rank'] is not None)
    return {'forecast_races':len(forecasts),'top_evaluable_races':n,'top_excluded':dict(excluded),'distribution':dict(distribution),
        'rank1_count':distribution['1'],'rank2_count':distribution['2'],'rank3_count':distribution['3'],**rates,
        'top3_rate':100*sum(distribution[str(k)] for k in (1,2,3))/n if n else None,
        'within_race_auc':float(np.mean(auc)) if auc else None,'auc_races':len(auc),'winner_rank_distribution':dict(winner_ranks),
        'winner_mean_predicted_rank':sum(int(k)*v for k,v in winner_ranks.items())/sum(winner_ranks.values()) if winner_ranks else None}


def summarize(races):
    # A saved-model slot is generic; never pool different model identities.
    identities={(r['comparison_id'],r['forecast_metadata']['saved'].get('model_id')) for r in races}
    if len(identities)>1:raise safety.MLFailure('invalid_model')
    common=[r for r in races if r['common']];agree=[r for r in common if r['top_agreement']];different=[r for r in common if not r['top_agreement']]
    for race in agree:
        if race['models']['saved']['top_result']!=race['models']['H']['top_result']:raise safety.MLFailure('invalid_identity')
    financial=[r for r in common if all(r['models'][n]['paper']['status']=='settled' for n in ('saved','H'))]
    paper={}
    for name in ('saved','H'):
        paid=sum(r['models'][name]['paper']['payout_yen'] for r in financial);stake=100*len(financial)
        paper[name]={'settled_common_races':len(financial),'investment_yen':stake,'payout_yen':paid,'net_profit_yen':paid-stake,
            'return_rate_percent':100*paid/stake if stake else None,'unsettled_common_races':len(common)-len(financial),
            'rule':'flat_top_pick_100_yen_reference_not_EV_strategy'}
    relations=Counter(r['relation'] for r in different)
    return {'target_races':len(races),'common_races':len(common),'exclusions':dict(Counter(r['common_reason'] for r in races if not r['common'])),
        'available':{n:model_summary(races,n) for n in ('saved','H')},'common':{n:model_summary(common,n) for n in ('saved','H')},
        'agreement':{'races':len(agree),'rate':100*len(agree)/len(common) if common else None,
            'both_top1_hits':sum(r['models']['saved']['top_result']['rank']==1 for r in agree)},
        'disagreement':{'races':len(different),'saved_top1_hits':sum(r['models']['saved']['top_result']['rank']==1 for r in different),
            'H_top1_hits':sum(r['models']['H']['top_result']['rank']==1 for r in different),'finish_comparison':dict(relations)},
        'paper':paper,'dead_heat_races':sum(r['dead_heat'] for r in races),'nonstarter_races':sum(bool(r['nonstarter_count']) for r in races),
        'result_unconfirmed_races':sum(not r['result_ok'] for r in races)}


def build(forecast,results):
    if results['forecast_sha256']!=forecast['committed_sha256']:raise safety.MLFailure('artifact_invalid')
    races=[race_view(r,results['races'].get(''.join(r['race_key']),[]),results['outcomes'].get(''.join(r['race_key']))) for r in forecast['races']]
    days={day:summarize([r for r in races if r['date']==day]) for day in DAYS}
    coverage={'target_races':len(races),'common_races':sum(s['common_races'] for s in days.values()),
        'saved_forecast_races':sum(s['available']['saved']['forecast_races'] for s in days.values()),
        'H_replayed_races':sum(s['available']['H']['forecast_races'] for s in days.values()),
        'performance_pooled_across_saved_models':False}
    summary={'coverage':coverage,'days':days}
    return {'races':races,'summary':summary,'saved_sources':forecast['saved_sources'],
        'saved_models':forecast['saved_models'],'H_source':forecast['H_source'],
        'freeze_time':forecast['created_at'],'results_collected_at':results['collected_at'],
        'results_joined_at':results.get('joined_at',results['collected_at']),'dates':list(DAYS)}
