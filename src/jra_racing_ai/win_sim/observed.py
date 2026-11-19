"""Small local snapshot/decision helpers; existing EV and settlement remain authoritative."""
from collections import defaultdict
from datetime import timedelta
from . import core

MAX_AGE_SECONDS=120

def fresh_snapshot(race,snapshots,cutoff,roster,*,max_age_seconds=MAX_AGE_SECONDS):
    if type(max_age_seconds) is not int or not 1<=max_age_seconds<=600:
        raise core.SimulationError('invalid_input')
    selected,reason=core.choose_snapshot(race,snapshots,cutoff,roster,timing_basis='observed')
    if reason:
        return None,reason
    upper=selected.published_at+(timedelta(seconds=59,microseconds=999999) if selected.publication_precision=='minute' else timedelta())
    if upper>cutoff:
        return None,'odds_time_precision'
    if (selected.observed_at is None or (cutoff-selected.observed_at).total_seconds()>max_age_seconds
            or (cutoff-upper).total_seconds()>max_age_seconds):
        return None,'odds_stale'
    return selected,None

def decision(prediction,schedule,snapshots,policy,*,recorded_at,max_age_seconds=MAX_AGE_SECONDS):
    _,cutoff,reason=core.choose_schedule(prediction.race,[schedule],policy)
    if reason:
        return None,reason
    if recorded_at<cutoff:
        return None,'decision_not_due'
    snapshot,reason=fresh_snapshot(prediction.race,snapshots,cutoff,schedule.horses,max_age_seconds=max_age_seconds)
    ledger=core.run_simulation([prediction],[schedule],[snapshot] if snapshot else [],[],policy=policy)
    row=ledger['rows'][0]
    if reason:
        row['reason']=reason
    p,_=core.probabilities(prediction.probabilities,policy)
    row.update(prediction_top_horse=min(p,key=lambda h:(-p[h],h)),
        decision_recorded_at=recorded_at.isoformat(),information_cutoff=cutoff.isoformat(),
        max_odds_age_seconds=max_age_seconds,
        decision_class='data_unavailable' if not row['simulation_eligible'] else
            'virtual_purchase' if row['ai_horse'] else 'conditions_not_met',
        decision_record_type='calculation_from_frozen_predeadline_inputs',
        actual_bet_placed=False)
    return row,None

def risk_metrics(rows,strategy='ai'):
    bets=[r for r in rows if r[strategy]['status']!='no_bet']
    settled=[r for r in bets if r[strategy]['status']=='settled']
    pending=len(bets)-len(settled)
    grouped=defaultdict(int)
    for row in settled:
        grouped[row.get('scheduled_start') or row.get('decision_at') or row.get('month','')]+=row[strategy]['payout_yen']-100
    total=peak=drawdown=0
    for stamp in sorted(grouped):
        total+=grouped[stamp]; peak=max(peak,total); drawdown=max(drawdown,peak-total)
    payouts=[row[strategy]['payout_yen'] for row in settled]
    gains=[p-100 for p in payouts if p>100]
    largest=max(gains,default=0); paid=sum(payouts)
    value={'max_drawdown_yen':drawdown if settled else None,
        'largest_payout_share_percent':100*max(payouts)/paid if paid else None,
        'net_profit_without_largest_win_yen':total-largest if settled else None,
        'largest_win_net_to_total_profit_percent':100*largest/total if total>0 else None,
        'ordering':'scheduled_start_order_simultaneous_races_grouped_not_bankroll_execution',
        'settled_bets':len(settled)}
    if pending:
        return {'max_drawdown_yen':None,'largest_payout_share_percent':None,
                'net_profit_without_largest_win_yen':None,'largest_win_net_to_total_profit_percent':None,
                'unsettled_bets':pending,'settled_subset':value}
    return dict(value,unsettled_bets=0)
