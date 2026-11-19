"""Pure final-population rules and end-of-window guard; no current result I/O."""
from datetime import date
from ..d39_next.features import race_reasons
from ..ml import safety
from . import contracts as c,protocol
from .timing import quote_valid


def population(record,official_rows):
    """Apply whole-race exclusion first. Do not condition P3 on the outcome."""
    all3=all(record.get('models',{}).get(n,{}).get('status')=='saved' for n in c.NAMES)
    expected=set(record.get('roster',()))
    all3=all3 and all(set(record['models'][n].get('probability',{}))==expected for n in c.NAMES)
    quote=record.get('quote_T10');odds=quote_valid(quote,expected,record.get('input_cutoff'))
    reasons=race_reasons(official_rows) if official_rows else ['outcomes_missing']
    if {r.get('horse_number') for r in official_rows}!=expected:reasons.append('result_roster_mismatch')
    p1=all3 and not reasons
    remaining=[r['horse_number'] for r in official_rows if r.get('anomaly') not in ('1','2','3')]
    # Usually unreachable for cancellations because the existing race_reasons
    # already rejects their whole race, but the sequence remains explicit.
    normalized={}
    if p1:
        for name in c.NAMES:
            p=record['models'][name]['probability'];total=sum(p[h] for h in remaining)
            if total<=0:raise safety.MLFailure('invalid_input')
            normalized[name]={h:p[h]/total for h in remaining}
    return {'P1':p1,'P2':p1 and odds,'P3':all3 and odds,
        'exclusion_reasons_P1':reasons,'P1_P2_remaining_probabilities':normalized,
        'P3_is_not_a_P1_subset':True}


def r1_outcome(selected,official_rows,payouts,*,payout_state='needs_confirmation'):
    """Use supplied official decoded payouts, never decision odds. No DB access."""
    rows=[r for r in official_rows if r.get('horse_number')==selected]
    if len(rows)!=1:return {'status':'unsettled','reason':'official_result_missing'}
    row=rows[0];anomaly=row.get('anomaly')
    if anomaly in ('1','2','3'):return {'status':'refunded','stake_yen':100,'payout_yen':100,'profit_yen':0,'hit':False}
    if payout_state!='normal_confirmed':return {'status':'unsettled','reason':'special_or_payout_state_unconfirmed'}
    if anomaly in ('4','5'):return {'status':'settled','stake_yen':100,'payout_yen':0,'profit_yen':-100,'hit':False}
    try:rank=int(row['finish_position'])
    except (ValueError,TypeError,KeyError):return {'status':'unsettled','reason':'official_rank_missing'}
    if rank<1 or anomaly not in ('0','6','7'):return {'status':'unsettled','reason':'official_state_unconfirmed'}
    if rank!=1:return {'status':'settled','stake_yen':100,'payout_yen':0,'profit_yen':-100,'hit':False}
    amount=payouts.get(selected) if payouts else None
    if type(amount) is not int or amount<=0:return {'status':'unsettled','reason':'official_payout_unconfirmed'}
    return {'status':'settled','stake_yen':100,'payout_yen':amount,'profit_yen':amount-100,'hit':True}


def market_probability(record,remaining):
    prices=record['quote_T10']['odds']
    if not remaining or any(h not in prices or prices[h]<=0 for h in remaining):raise safety.MLFailure('invalid_odds')
    inverse={h:1/float(prices[h]) for h in remaining};total=sum(inverse.values())
    return {h:v/total for h,v in inverse.items()}


def acceptance(difference,interval,LL_difference,Brier_difference,rules):
    g=rules['thresholds']
    return (difference>=g['minimum_within_race_AUC_gain'] and interval[0]>0
        and LL_difference<=g['race_LL_increase_max'] and Brier_difference<=g['Brier_increase_max'])


def require_final_window(directory,now=None):
    return protocol.require_closed(directory,now)
