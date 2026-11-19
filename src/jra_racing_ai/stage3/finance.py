"""Fixed 100-yen paper diagnostics; profit is never a model-selection input."""
from collections import defaultdict
from copy import deepcopy
from datetime import date,timedelta
import numpy as np
from ..ml import safety
from ..research import metrics as oldmetrics
from ..win_sim import core,observed
from . import contracts as c


def weekly_financial(rows,strategy='ai',*,confidence=.95):
    weeks=defaultdict(lambda:np.zeros(4))
    for row in rows:
        r=row['race_key'];day=date(int(r[0]),int(r[1][:2]),int(r[1][2:]));week=(day-timedelta(days=day.weekday())).isoformat()
        stat=weeks[week];bet=row[strategy]
        if bet['status']!='no_bet':
            stat[0]+=100;stat[1]+=bet['payout_yen'] if bet['status']=='settled' else 0
            stat[2]+=int(bet['status']!='settled');stat[3]+=1
    if not weeks:return {'status':'unavailable','reason':'empty_population','ROI_ci':None,'zero_purchase_fraction':None}
    mat=np.asarray([v for _,v in sorted(weeks.items())]);rng=np.random.default_rng(c.SEED)
    draw=mat[rng.integers(len(mat),size=(c.REPETITIONS,len(mat)))].sum(axis=1)
    zero=draw[:,0]==0;pending=draw[:,2]>0
    roi=np.divide(100*draw[:,1],draw[:,0],out=np.full(c.REPETITIONS,np.nan),where=~zero&~pending)
    profit=np.where(pending,np.nan,draw[:,1]-draw[:,0]);alpha=(1-confidence)/2
    def interval(v):
        v=v[np.isfinite(v)]
        return [float(x) for x in np.quantile(v,[alpha,1-alpha])] if len(v) else None
    return {'repetitions':c.REPETITIONS,'seed':c.SEED,'unit':'Monday_starting_race_week','confidence':confidence,
        'profit_yen_ci':interval(profit),'ROI_percent_ci':interval(roi),'zero_purchase_fraction':float(zero.mean()),
        'unsettled_replica_fraction':float(pending.mean()),'ROI_defined_replicas':int(np.isfinite(roi).sum()),
        'purchase_count':int(mat[:,3].sum()),'few_bets_warning':bool(mat[:,3].sum()<100),
        'few_bets_note':'under_100_is_descriptive_caution_not_a_power_test',
        'ROI_zero_denominator':'undefined_not_zero','DD_bootstrapped':False}


def apply_probability_filter(ledger,packets,winners,minimum=.05):
    result=deepcopy(ledger)
    for row in result['rows']:
        if not row['simulation_eligible']:continue
        candidates=[r for r in row['runners'] if core.decimal(r['probability'])>=core.decimal(minimum)
            and core.decimal(r['probability'])*core.decimal(r['decision_odds'])-1>=core.decimal('.05')]
        candidates.sort(key=lambda r:(-(core.decimal(r['probability'])*core.decimal(r['decision_odds'])-1),r['horse_number']))
        h=candidates[0]['horse_number'] if candidates else None
        row['ai_horse']=h;row['reason']='ev_selected' if h else 'probability_filter_no_candidate'
        packet=packets[tuple(row['race_key'])];value=packet.get('outcome')
        if value and row['common_settlement_eligible']:
            outcome=core.Outcome(tuple(row['race_key']),value['status'],value['payouts_yen_per_100'],value['verified'],value['complete_winner_set'],value['reason'])
        else:outcome=core.Outcome(tuple(row['race_key']),'pending',reason=row['settlement_reason'])
        row['ai']=core.settle(h,outcome,tuple(r['horse_number'] for r in row['runners']))
    result['summary']=core.summarize(result['rows']);result['additional_min_probability']=minimum
    return result


def diagnostic(predictions,packets,winners,*,minimum=None):
    _,ledger=oldmetrics.paper(predictions,packets,winners,start='2023-01-01',end='2025-09-30')
    if minimum is not None:ledger=apply_probability_filter(ledger,packets,winners,minimum)
    rows=ledger['rows'];summary=core.summarize(rows)
    summary.update(risk=observed.risk_metrics(rows),favorite_risk=observed.risk_metrics(rows,'favorite'),
        weekly_uncertainty=weekly_financial(rows),favorite_weekly_uncertainty=weekly_financial(rows,'favorite'),
        retrospective_publication_time_not_execution=True,schedule_original_asof_unverified=True,
        minimum_probability=minimum,probability_filter_applied_before_EV_argmax=minimum is not None)
    chosen=[x for row in rows for x in row['runners'] if x['horse_number']==row['ai_horse']]
    summary['selected_probability_distribution']=core.distribution([x['probability'] for x in chosen])
    summary['selected_odds_distribution']=core.distribution([x['decision_odds'] for x in chosen])
    # Persist only the selected tickets and decision provenance, not repeated whole-race prices.
    keys=('race_key','month','decision_at','scheduled_start','odds_published_at','ai_horse','favorite_horse',
          'ai','favorite','reason','common_settlement_eligible','settlement_reason')
    details=[dict({k:row[k] for k in keys},selected=next((x for x in row['runners'] if x['horse_number']==row['ai_horse']),None)) for row in rows]
    return summary,details
