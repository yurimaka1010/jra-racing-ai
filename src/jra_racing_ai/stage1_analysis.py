"""Pure Stage 1 selection and probability diagnostics, reusing existing evaluators."""
from collections import Counter, defaultdict
from datetime import timedelta
import math

from .ml import market, v03
from .win_sim import core, formats

OFFSETS = (30, 15, 10, 5)
TEMPERATURES = (.75, 1., 1.25, 2.)
WEIGHTS = (0., .05, .1, .2, .3, .5, .75, 1.)
IDENTITY = core.WinPolicy(probability_method='identity')


def reference_schedule(race, versions, roster):
    """Unchanged saved RA clock is a historical reference, NOT verified as-of evidence."""
    if not versions:
        return None, 'schedule_unavailable'
    if len(versions) != 1:
        return None, 'schedule_conflict'
    row = versions[0]
    if (row.get('oversized') or row.get('record_type') != 'RA'
            or row.get('status') not in tuple('1234567')):
        return None, 'schedule_unverified'
    if row.get('previous_start') not in (None, '', '0000', '    '):
        return None, 'schedule_changed_asof_unavailable'
    try:
        start = formats.parse_time(row.get('start'), 'hhmm_on_race_date', race)
    except core.SimulationError:
        return None, 'schedule_unverified'
    if row.get('start') == '0000':
        return None, 'schedule_unverified'
    return core.Schedule(race, start, None, tuple(roster), True,
                         'reviewed_historical_schedule'), None


def select_quote(race, rows, roster, schedule, minutes=10, max_age_seconds=300):
    """Final rows never enter selection; minute clocks must end before the cutoff."""
    if schedule is None:
        return None, 'schedule_unavailable'
    quote, reason = market.select_pre_race(race, rows, roster, [schedule],
        core.WinPolicy(minutes_before_start=minutes, timing_basis='jra_publication_time'))
    if reason:
        return None, reason
    cutoff = schedule.scheduled_start - timedelta(minutes=minutes)
    age = (cutoff - quote.published_at).total_seconds()
    if age > max_age_seconds:
        return None, 'odds_stale'
    return quote, None


def normalized_ai(predictions):
    policy = core.WinPolicy(temperature=.75)
    return [core.PredictionRace(p.race, {h: float(v) for h, v in
        core.probabilities(p.probabilities, policy)[0].items()}) for p in predictions]


def market_predictions(quotes, temperature=1.):
    result = []
    for quote in quotes:
        if (quote.category != '1' or quote.basis != 'pre_race_publication_time'
                or quote.published_at is None):
            raise core.SimulationError('odds_unverified')
        raw, _ = market.implied_probabilities(quote.odds)
        values, _ = core.probabilities(raw, core.WinPolicy(temperature=temperature))
        result.append(core.PredictionRace(quote.race, {h: float(v) for h, v in values.items()}))
    return result


def blend(ai, markets, weight, *, geometric=False):
    if isinstance(weight, bool) or weight not in WEIGHTS:
        raise core.SimulationError('invalid_input')
    a = {p.race: p for p in ai}
    b = {p.race: p for p in markets}
    if len(a) != len(ai) or len(b) != len(markets) or set(a) != set(b):
        raise core.SimulationError('roster_mismatch')
    result = []
    for race in sorted(a):
        pa, _ = core.probabilities(a[race].probabilities, IDENTITY)
        pb, _ = core.probabilities(b[race].probabilities, IDENTITY)
        if set(pa) != set(pb):
            raise core.SimulationError('roster_mismatch')
        if weight == 0:
            values = {h: float(v) for h, v in pb.items()}
        elif weight == 1:
            values = {h: float(v) for h, v in pa.items()}
        elif geometric:
            scores = {h: weight*math.log(max(1e-15, float(pa[h]))) +
                (1-weight)*math.log(max(1e-15, float(pb[h]))) for h in pa}
            peak = max(scores.values())
            values = {h: math.exp(v-peak) for h, v in scores.items()}
        else:
            values = {h: weight*float(pa[h])+(1-weight)*float(pb[h]) for h in pa}
        total = sum(values.values())
        result.append(core.PredictionRace(race, {h: v/total for h, v in values.items()}))
    return result


def measure(predictions, winners):
    if not predictions:
        return {'status': 'unavailable', 'races': 0}
    if any(p.race not in winners or winners[p.race] not in p.probabilities for p in predictions):
        raise core.SimulationError('roster_mismatch')
    value = v03.probability_evaluation(predictions, winners, IDENTITY)
    return dict(status='ok', races=value['races'], **value['model'],
                calibration=market.calibration_table(value))


def fit_development(ai, quotes, winners, *, development_end, evaluation_start, development_start='2025-03-01'):
    """Only development labels may select market temperature or blend weight."""
    if development_end >= evaluation_start or not ai:
        raise core.SimulationError('invalid_period')
    for p in ai:
        day = p.race[0]+'-'+p.race[1][:2]+'-'+p.race[1][2:]
        if not development_start <= day <= development_end or day >= evaluation_start:
            raise core.SimulationError('invalid_period')
    if {p.race for p in ai} != {q.race for q in quotes}:
        raise core.SimulationError('roster_mismatch')
    candidates = [{'temperature': t, 'log_loss': measure(market_predictions(quotes, t), winners)['log_loss']}
                  for t in TEMPERATURES]
    temperature = min(candidates, key=lambda c: (c['log_loss'], abs(c['temperature']-1)))['temperature']
    raw, recal = market_predictions(quotes), market_predictions(quotes, temperature)
    choices, scores = {}, {}
    for name, base, geometric in (('linear_raw', raw, False), ('linear_recalibrated', recal, False),
                                 ('geometric_recalibrated', recal, True)):
        values = [{'ai_weight': w, 'log_loss': measure(blend(ai, base, w, geometric=geometric), winners)['log_loss']}
                  for w in WEIGHTS]
        # Exact ties keep the market-only endpoint, limiting gratuitous AI weight.
        choices[name] = min(values, key=lambda c: (c['log_loss'], c['ai_weight']))['ai_weight']
        scores[name] = values
    return {'market_temperature': temperature, 'market_temperature_candidates': candidates,
            'ai_weights': choices, 'weight_candidates': scores, 'primary_metric': 'horse_binary_log_loss',
            'selected_on': 'development_only', 'development_end': development_end,
            'first_evaluation_start': evaluation_start}


def variants(ai, quotes, selection):
    raw = market_predictions(quotes)
    recal = market_predictions(quotes, selection['market_temperature'])
    weights = selection['ai_weights']
    return {'best242': ai, 'market_raw': raw, 'market_recalibrated': recal,
        'linear_raw': blend(ai, raw, weights['linear_raw']),
        'linear_recalibrated': blend(ai, recal, weights['linear_recalibrated']),
        'geometric_recalibrated': blend(ai, recal, weights['geometric_recalibrated'], geometric=True)}


def disagreement(ai, markets, winners):
    b = {p.race: p for p in markets}
    counts = {name: Counter() for name in ('agreement', 'disagreement')}
    for p in ai:
        other = b[p.race]
        if set(p.probabilities) != set(other.probabilities):
            raise core.SimulationError('roster_mismatch')
        first = min(p.probabilities, key=lambda h: (-p.probabilities[h], h))
        second = min(other.probabilities, key=lambda h: (-other.probabilities[h], h))
        row = counts['agreement' if first == second else 'disagreement']
        row['races'] += 1
        row['ai_hits'] += first == winners[p.race]
        row['market_hits'] += second == winners[p.race]
    result = {name: dict(races=c['races'], ai_hits=c['ai_hits'], market_hits=c['market_hits'],
                        ai_misses=c['races']-c['ai_hits']) for name, c in counts.items()}
    if result['agreement']['ai_hits'] != result['agreement']['market_hits']:
        raise core.SimulationError('invalid_input')
    return dict(result, tie_rule='lowest_horse_number')


def paired(new, baseline, winners):
    if not new:
        return {'status': 'unavailable'}
    from .post_step2 import paired as existing_paired
    return existing_paired(new, baseline, winners, IDENTITY, IDENTITY)


def price_drift(quotes, finals):
    """Postrace prices are diagnostics only and never inputs to selection."""
    changes = []
    for q in quotes:
        final = finals.get(q.race)
        if final is not None and set(final.odds) == set(q.odds):
            changes.extend(float(final.odds[h]/q.odds[h]-1) for h in q.odds)
    return {'horse_quote_count': len(changes), 'final_over_pre_minus_one': core.distribution(changes)}


def match_outcomes(outcomes, winners):
    """Keep existing labels fixed and hold only contradictory HR settlements."""
    checked, mismatches = {}, 0
    for race, outcome in outcomes.items():
        if (race in winners and outcome.status == 'normal_final'
                and set(outcome.payouts_yen_per_100) != {winners[race]}):
            checked[race] = core.Outcome(race, 'pending', reason='result_unconfirmed')
            mismatches += 1
        else:
            checked[race] = outcome
    return checked, mismatches
