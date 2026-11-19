"""Same causal builder for training and inference; abnormal ranks never enter history."""
from collections import Counter, defaultdict, deque
from datetime import timedelta
import math

from ..ml import safety
from ..research import features as legacy
from . import contracts as c

ANOMALIES = {'1': 'scratched', '2': 'excluded_by_starter', '3': 'excluded_by_stewards',
             '4': 'did_not_finish', '5': 'disqualified', '6': 'remounted', '7': 'demoted'}
REST_DAYS = 84


def race_reasons(rows):
    reasons = set()
    identities = [legacy.legacy.entry_identity(r)[0] for r in rows]
    if len(identities) != len(set(identities)):
        reasons.add('duplicate_entry')
    for row in rows:
        code = row.get('anomaly')
        if code in ANOMALIES:
            reasons.add(ANOMALIES[code])
        elif code != '0':
            reasons.add('anomaly_unknown')
    n = len(rows)
    if n < 2 or n > 28 or any(legacy.num(r.get('actual_runners')) != n for r in rows):
        reasons.add('incomplete_starter_set')
    ranks = [legacy.num(r.get('finish_position'), 1, 28) for r in rows]
    if any(v is None or not v.is_integer() for v in ranks):
        reasons.add('invalid_finish')
    elif sorted(ranks) != list(range(1, n + 1)):
        reasons.add('tied_or_inconsistent_finish')
    return sorted(reasons)


def stage(gap, previous):
    if gap is None:
        return None
    if gap >= REST_DAYS:
        return 'first'
    return {'first': 'second', 'second': 'third', 'third': 'later', 'later': 'later'}.get(previous, 'none')


class Builder(legacy.Builder):
    def __init__(self):
        super().__init__()
        self.start_events = deque()
        self.starts = {}
        self.exclusions = Counter()
        self.excluded_races = 0
        self.source_races = 0

    def _available(self, day):
        super()._available(day)
        while self.start_events and self.start_events[0][0] + timedelta(days=7) <= day:
            old_day, rows = self.start_events.popleft()
            for row, clean in rows:
                if row.get('anomaly') not in legacy.STARTERS or not row.get('horse_id'):
                    continue
                h = self.starts.setdefault(row['horse_id'], {'last_day': None, 'stage': None,
                    'gaps': deque(maxlen=3), 'quality': defaultdict(lambda: [0, 0.])})
                gap = (old_day - h['last_day']).days if h['last_day'] else None
                current = stage(gap, h['stage'])
                if gap is not None:
                    h['gaps'].append(gap)
                # Only dates/known irregular flags are retained for abnormal starts.
                # Results from the entire affected race never enter a performance mean.
                if clean and current:
                    q = legacy.percentile(legacy.num(row.get('finish_position')), legacy.num(row.get('actual_runners')))
                    if q is not None:
                        h['quality'][current][0] += 1
                        h['quality'][current][1] += q
                h.update(last_day=old_day, stage=current, irregular=float(row.get('anomaly') != '0'))

    def transform_day(self, rows, *, prediction=False):
        if not rows:
            return [], []
        day = legacy.legacy.entry_identity(rows[0])[1]
        races = defaultdict(list)
        for row in rows:
            races[legacy.legacy.entry_identity(row)[0][:-1]].append(row)
        reasons = {key: race_reasons(group) for key, group in races.items()} if not prediction else {}
        # Current labels only affect the training mask and future, delayed events.
        values, meta = super().transform_day(rows)
        old_day, events = self.pending.pop()
        clean_events = [] if prediction else [e for e in events if not reasons[e['identity'][:-1]]]
        self.pending.append((old_day, clean_events))
        if not prediction:
            self.start_events.append((day, [(row, not reasons[key]) for key, group in races.items() for row in group]))
            self.source_races += len(races)
            self.excluded_races += sum(bool(v) for v in reasons.values())
            self.exclusions.update(reason for part in reasons.values() for reason in part)
        # Legacy transform sorts races. Join by identity, not input row order.
        by_key = {legacy.legacy.entry_identity(r)[0]: r for r in rows}
        for x, m in zip(values, meta):
            key = tuple(m['key']); row = by_key[key]; h = self.starts.get(row.get('horse_id'), {})
            gap = (day - h['last_day']).days if h.get('last_day') else None
            current = stage(gap, h.get('stage')); gaps = h.get('gaps', ())
            first = h.get('quality', {}).get('first', (0, 0.))
            conditional = h.get('quality', {}).get(current, (0, 0.)) if current else None
            x.update(e_days_since_actual_start=gap,
                e_previous_start_gap_days=gaps[-1] if gaps else None,
                e_recent3_start_gap_mean_days=sum(gaps) / len(gaps) if gaps else None,
                e_rest_stage=current,
                e_rest_first_past_starts=first[0],
                e_rest_first_smoothed_finish_percentile=(first[1] + 1.5) / (first[0] + 3),
                e_rest_stage_smoothed_finish_percentile=(conditional[1] + 1.5) / (conditional[0] + 3) if conditional else None,
                e_last_start_irregular=h.get('irregular'))
            if set(x) != set(c.spec()['features']):
                raise safety.MLFailure('invalid_input')
            if not prediction:
                m['exclusion_reasons'] = reasons[key[:-1]]
                m['eligible'] = bool(m['eligible'] and not reasons[key[:-1]])
        return values, meta
