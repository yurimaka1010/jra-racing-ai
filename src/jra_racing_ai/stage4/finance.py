"""Fixed paper bets and shared-week uncertainty; never used to select predictions."""
from copy import deepcopy
from collections import defaultdict
from datetime import date
import hashlib,json
import numpy as np
from ..ml import safety
from ..research import metrics as oldmetrics
from ..win_sim import core,observed
from . import contracts as c,store as s,data,accuracy

MONTHS=tuple(f'{y}-{m:02d}' for y in c.YEARS for m in range(1,10))
MAJOR=('Hstack','H60','high_disagreement_ABCD')
CONTROLS=('ABCD','H','favorite','no_buy')


def interval(values,confidence):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v,[(1-confidence)/2,1-(1-confidence)/2])] if len(v) else None


class Weekly:
    def __init__(self,races,*,repetitions=c.REPETITIONS):
        self.weeks=sorted({accuracy.week(tuple(r)) for r in races});self.index={w:i for i,w in enumerate(self.weeks)}
        if not self.weeks:raise safety.MLFailure('insufficient_history')
        self.sample=np.random.default_rng(c.SEED).integers(len(self.weeks),size=(repetitions,len(self.weeks)))
        self.repetitions=repetitions;self.cache={}

    def draw(self,rows):
        matrix=np.zeros((len(self.weeks),5))
        for row in rows:
            day=accuracy.week(tuple(row['race_key']))
            if day not in self.index:raise safety.MLFailure('invalid_identity')
            bet=row['ai']
            if bet['status']=='no_bet':continue
            if bet['stake_yen']!=100:raise safety.MLFailure('invalid_input')
            idx=self.index[day];matrix[idx,0]+=100;matrix[idx,4]+=1
            if bet['status']=='settled':matrix[idx,1]+=bet['payout_yen'];matrix[idx,3]+=bool(bet['hit'])
            else:matrix[idx,2]+=1
        sampled=matrix[self.sample].sum(axis=1);total=matrix.sum(axis=0)
        profit=np.where(sampled[:,2]==0,sampled[:,1]-sampled[:,0],np.nan)
        roi=np.divide(100*sampled[:,1],sampled[:,0],out=np.full(self.repetitions,np.nan),where=(sampled[:,0]>0)&(sampled[:,2]==0))
        return dict(profit=profit,ROI=roi,zero=sampled[:,0]==0,pending=sampled[:,2]>0,total=total,matrix=matrix)

    def summarize(self,rows):
        values=self.draw(rows)
        return {'unit':'Monday_starting_week_all_venues','weeks':len(self.weeks),'repetitions':self.repetitions,'seed':c.SEED,
            'shared_indices_sha256':hashlib.sha256(self.sample.tobytes()).hexdigest(),
            '95_percent':{'profit_yen_ci':interval(values['profit'],.95),'ROI_percent_ci':interval(values['ROI'],.95)},
            '98_333333_percent':{'profit_yen_ci':interval(values['profit'],c.CORRECTED),'ROI_percent_ci':interval(values['ROI'],c.CORRECTED)},
            'zero_purchase_replica_fraction':float(values['zero'].mean()),'pending_replica_fraction':float(values['pending'].mean()),
            'ROI_defined_replicas':int(np.isfinite(values['ROI']).sum()),'hit_weeks':int((values['matrix'][:,3]>0).sum()),
            'active_weeks':int((values['matrix'][:,4]>0).sum()),'few_hits_caution':bool(values['total'][3]<30),
            'all_losses_sample':bool(values['total'][0]>0 and values['total'][1]==0 and values['total'][2]==0),
            'zero_denominator_ROI':None,'all_loss_degenerate_CI_does_not_prove_true_ROI_zero':True,
            'not_independent_or_familywide_error_control':True}

    def paired(self,new,baseline):
        a=self.draw(new);b=self.draw(baseline);delta=a['profit']-b['profit']
        point=None if a['total'][2] or b['total'][2] else float((a['total'][1]-a['total'][0])-(b['total'][1]-b['total'][0]))
        return {'difference_sign':'new_minus_control','profit_difference_yen':point,
            'monthly_mean_difference_yen':point/27 if point is not None else None,
            '95_percent_ci':interval(delta,.95),'98_333333_percent_ci':interval(delta,c.CORRECTED),
            'unit':'paired_Monday_starting_week','weeks':len(self.weeks),'repetitions':self.repetitions,'seed':c.SEED,
            'shared_indices_sha256':hashlib.sha256(self.sample.tobytes()).hexdigest(),
            'unsettled_replica_fraction':float((~np.isfinite(delta)).mean())}


def compact(rows,*,strategy='ai'):
    fields=('race_key','month','scheduled_start','decision_at','odds_published_at','ai_horse','favorite_horse','ai','favorite',
        'reason','common_settlement_eligible','settlement_reason')
    output=[]
    for row in rows:
        result={k:row[k] for k in fields};h=row['ai_horse'] if strategy=='ai' else row['favorite_horse']
        result['selected']=next((r for r in row['runners'] if r['horse_number']==h),None)
        if strategy=='favorite':result.update(ai_horse=h,ai=row['favorite'],reason='same_time_favorite')
        output.append(deepcopy(result))
    return output


def skip(row,reason):
    value=deepcopy(row);value.update(ai_horse=None,ai=core.settle(None,None,()),selected=None,reason=reason);return value


def restrict_high(rows,high):return [deepcopy(r) if tuple(r['race_key']) in high else skip(r,'outside_fixed_high_D') for r in rows]


def financial(rows,weekly):
    total=core.financial(rows,'ai');monthly={month:core.financial([r for r in rows if r['month']==month],'ai') for month in MONTHS}
    annual={str(y):core.financial([r for r in rows if r['month'].startswith(str(y))],'ai') for y in c.YEARS}
    paid=total['payout_yen'];net=total['net_profit_yen'];bets=total['bet_count']
    settled=[r for r in rows if r['ai']['status']=='settled'];largest=max((r['ai']['payout_yen'] for r in settled),default=0)
    risk=observed.risk_metrics(rows);risk['scope']='original_chronological_order_evaluated_Jan_Sep_months_only_not_full_3_calendar_years'
    risk['largest_payout_yen']=largest if not total['unsettled_count'] else None
    risk['profit_after_zeroing_largest_payout_keeping_its_stake_yen']=net-largest if net is not None else None
    risk['profit_after_removing_largest_payout_ticket_yen']=net-largest+100 if net is not None and bets else None
    selected=[r['selected'] for r in rows if r.get('selected') is not None]
    return {'financial':total,'monthly':monthly,'annual':annual,'monthly_mean_profit_yen_including_27_months':net/27 if net is not None else None,
        'evaluated_months':27,'no_purchase_months':sum(v['bet_count']==0 for v in monthly.values()),
        'no_purchase_years':sum(v['bet_count']==0 for v in annual.values()),
        'strictly_profitable_years':sum(v['net_profit_yen'] is not None and v['net_profit_yen']>0 and v['bet_count']>0 for v in annual.values()),
        'monthly_purchase_concentration_max_percent':100*max(v['bet_count'] for v in monthly.values())/bets if bets else None,
        'monthly_payout_concentration_max_percent':100*max(v['payout_yen'] or 0 for v in monthly.values())/paid if paid else None,
        'weekly_uncertainty':weekly.summarize(rows),'risk':risk,
        'selected_probability_distribution':core.distribution([r['probability'] for r in selected]),
        'selected_odds_distribution':core.distribution([r['decision_odds'] for r in selected]),
        'selected_EV_distribution':core.distribution([r['expected_return'] for r in selected]),
        'pending_not_zero_payout':True,'minimum_probability':None,'fixed_EV_threshold':.05,'stake_yen':100,'max_per_race':1,
        'timing':'T10_JRA_publication_posthoc_not_PC_receipt_or_actual_execution','settlement':'verified_official_HR_not_displayed_odds'}


def retain(result):
    f=result['financial'];profit=f['net_profit_yen'];roi=f['return_rate_percent']
    passes=profit is not None and profit>0 and roi is not None and roi>100 and result['strictly_profitable_years']>=2
    bounds=result['weekly_uncertainty']['98_333333_percent']['profit_yen_ci'];cross=bounds is None or bounds[0]<=0<=bounds[1]
    return {'retain_financial_research_candidate':bool(passes),'status':'unconfirmed_provisional' if passes and cross else 'development_candidate' if passes else 'no_candidate',
        'strictly_profitable_years':result['strictly_profitable_years'],'total_profit_yen':profit,'ROI_percent':roi,
        'mean_monthly_profit_yen':result['monthly_mean_profit_yen_including_27_months'],
        'corrected_profit_interval_crosses_zero':cross,'no_purchase_year_is_not_profitable':True,
        'not_profit_guarantee_or_operation_adoption':True}


def run(directory):
    if not s.completed(directory,'accuracy') or not (directory/'accuracy_candidate_freeze.json').exists():raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'finance')
    if done:return done['summary']
    frames,byyear,winners=accuracy.load(directory);packets=data.market(directory,date(2023,1,1),date(2025,9,30))
    names=set.intersection(*(set(v) for v in byyear.values()));pool={n:[p for y in c.YEARS for p in byyear[y][n]] for n in names}
    folder=directory/'paper';folder.mkdir(exist_ok=True);ledgers={};files=[]
    for name in ('ABCD','H','Hstack','H60','ABCD_E'):
        if name not in pool:continue
        complete=s.completed(folder,name)
        if complete:rows=s.read(folder,name+'.ledger.local.json')
        else:
            _,full=oldmetrics.paper(pool[name],packets,winners,start='2023-01-01',end='2025-09-30')
            rows=compact(full['rows']);s.write(folder,name+'.ledger.local.json',rows)
            if name=='H':
                favorite=compact(full['rows'],strategy='favorite');s.write(folder,'favorite.ledger.local.json',favorite)
            local_files=[folder/(name+'.ledger.local.json')]+([folder/'favorite.ledger.local.json'] if name=='H' else [])
            s.complete(folder,name,local_files,{'races':len(rows),'new_training':False,'minimum_probability':None})
        ledgers[name]=rows;files.append(folder/(name+'.complete.json'))
    ledgers['favorite']=s.read(folder,'favorite.ledger.local.json')
    labels=s.read(directory,'conditional_labels.local.json');high={tuple(r['race_key']) for r in labels if r['stratum']=='high'}
    ledgers['high_disagreement_ABCD']=restrict_high(ledgers['ABCD'],high)
    ledgers['no_buy']=[skip(r,'fixed_no_purchase_control') for r in ledgers['H']]
    for name in ('high_disagreement_ABCD','no_buy'):
        s.write(folder,name+'.ledger.local.json',ledgers[name]);files.append(folder/(name+'.ledger.local.json'))
    weekly=Weekly([r['race_key'] for r in ledgers['H']]);results={n:financial(rows,weekly) for n,rows in ledgers.items()}
    paired={name:{base:weekly.paired(ledgers[name],ledgers[base]) for base in CONTROLS} for name in MAJOR if name in ledgers}
    supplements={'E_minus_ABCD':weekly.paired(ledgers['ABCD_E'],ledgers['ABCD']),
        'E_financial':results['ABCD_E'],'E_training_runs':0,'E_candidate':'unchanged_no_promotion'}
    strata={};ratio_ROI={}
    for layer in ('low','middle','high','top1_agreement','top1_disagreement'):
        key='stratum' if layer in ('low','middle','high') else 'top1_relation'
        races={tuple(r['race_key']) for r in labels if r[key]==layer}
        strata[layer]={n:{'financial':core.financial([r for r in rows if tuple(r['race_key']) in races],'ai'),
            'weekly_uncertainty':weekly.summarize([r for r in rows if tuple(r['race_key']) in races])}
            for n,rows in ledgers.items() if n in (*CONTROLS,'high_disagreement_ABCD')}
    pa={p.race:p.probabilities for p in pool['ABCD']};ph={p.race:p.probabilities for p in pool['H']}
    for name,rows in ledgers.items():
        bands=defaultdict(list)
        for row in rows:
            horse=row['ai_horse']
            if horse is None:continue
            race=tuple(row['race_key']);bands[accuracy.ratio_band(pa[race][horse]/ph[race][horse])].append(row)
        ratio_ROI[name]={band:core.financial(bands[band],'ai') for band in ('below_.8','.8_to_1.25','at_least_1.25')}
    financial_candidates={n:retain(results[n]) if n in results else {'status':'skipped_model_unavailable','retain_financial_research_candidate':False} for n in MAJOR}
    s.write(directory,'financial_candidate_freeze.json',{'candidates':financial_candidates,'accuracy_decision_kept_separate':True,
        'accuracy_freeze_sha256':s.digest(directory/'accuracy_candidate_freeze.json'),'no_rule_retuning':True,'no_production_adoption':True})
    s.write(directory,'finance.json',{'strategies':results,'major_strategy_paired_profit_differences':paired,
        'fixed_conditional_strata':strata,'horse_ratio_band_selected_ticket_ROI':ratio_ROI,
        'financial_candidates':financial_candidates,'all_same_common_population_races':6373,
        'no_market_min_probability_filter':True,'T10_decision_and_HR_settlement_separate':True,
        'no_threshold_or_stake_optimization':True,'development_evidence_not_independent_test':True})
    s.write(directory,'E_financial_supplement.json',supplements)
    files.extend(directory/n for n in ('finance.json','financial_candidate_freeze.json','E_financial_supplement.json','accuracy.complete.json'))
    result={'major_strategies':{n:{**results[n]['financial'],'mean_monthly_profit_yen':results[n]['monthly_mean_profit_yen_including_27_months'],
            'profitable_years':results[n]['strictly_profitable_years']} for n in MAJOR if n in results},
        'financial_candidates':financial_candidates,'months':27,'no_new_purchase_rule_except_high_D_ABCD':True}
    s.complete(directory,'finance',files,result);return result
