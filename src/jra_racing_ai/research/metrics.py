"""Race-primary, Monday-week cluster comparisons and fixed paper-money diagnostics."""
from collections import defaultdict
from datetime import date,timedelta
import math

from .. import stage1_analysis as s1a
from ..ml import learning_curve,market
from ..win_sim import core,observed
from . import contracts as c


def measure(predictions,winners):
    return s1a.measure(predictions,winners)


def compare(new,baseline,winners,*,confidence=.95,repetitions=1000):
    import numpy as np
    result=s1a.paired(new,baseline,winners)
    first={p.race:p for p in new};second={p.race:p for p in baseline}
    if set(first)!=set(second):raise core.SimulationError('roster_mismatch')
    weeks=defaultdict(list)
    for race in sorted(first):
        pa=first[race].probabilities;pb=second[race].probabilities
        if set(pa)!=set(pb):raise core.SimulationError('roster_mismatch')
        day=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
        week=(day-timedelta(days=day.weekday())).isoformat()
        diff=-math.log(max(1e-15,pa[winners[race]]))+math.log(max(1e-15,pb[winners[race]]))
        weeks[week].append(diff)
    sums=np.asarray([sum(v) for _,v in sorted(weeks.items())]);counts=np.asarray([len(v) for _,v in sorted(weeks.items())])
    rng=np.random.default_rng(20261019);sample=rng.integers(0,len(sums),(repetitions,len(sums)))
    distribution=sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
    alpha=(1-confidence)/2;low,high=np.quantile(distribution,[alpha,1-alpha])
    result['race_week_cluster']={'difference_new_minus_baseline':float(sums.sum()/counts.sum()),
        'ci_low':float(low),'ci_high':float(high),'confidence':confidence,'repetitions':repetitions,
        'weeks':len(weeks),'resampling_unit':'Monday_starting_race_week_all_venues',
        'interpretation':'mean_improved_uncertainty_remains' if sums.sum()<0 and high>=0 else
            'improvement_supported' if high<0 else 'mean_worse_uncertainty_remains' if low<=0 else 'worsening_supported'}
    return result


def paper(predictions,packets,winners,*,start,end):
    checked=[];schedules=[];snapshots=[];outcomes=[]
    for p in predictions:
        item=packets.get(p.race)
        if not item or not item['quotes']['10']:continue
        q=item['quotes']['10'];odds={h:core.decimal(v) for h,v in q['odds'].items()}
        if set(odds)!=set(p.probabilities):raise core.SimulationError('roster_mismatch')
        checked.append(p)
        schedules.append(core.Schedule(p.race,core.instant(item['scheduled_start']),None,tuple(p.probabilities),True,'reviewed_historical_schedule'))
        snapshots.append(core.Snapshot(p.race,core.instant(q['published_at']),None,odds,q['source'],True,publication_precision='minute'))
        o=item.get('outcome')
        if o:
            outcome=core.Outcome(p.race,o['status'],o['payouts_yen_per_100'],o['verified'],o['complete_winner_set'],o['reason'])
            if outcome.status=='normal_final' and set(outcome.payouts_yen_per_100)!={winners[p.race]}:
                outcome=core.Outcome(p.race,'pending',reason='result_unconfirmed')
            outcomes.append(outcome)
    policy=core.WinPolicy(probability_method='identity',timing_basis='jra_publication_time',
        evaluation_start=start,evaluation_end=end,model_fit_end='2014-12-31')
    ledger=core.run_simulation(checked,schedules,snapshots,outcomes,policy=policy)
    result=ledger['summary'];result['risk']=observed.risk_metrics(ledger['rows'])
    result['hypothetical_only']=True;result['schedule_asof_verified']=False
    return result,ledger


def pooled(predictions_by_period,winners):
    values=[p for group in predictions_by_period for p in group]
    return measure(values,winners)
