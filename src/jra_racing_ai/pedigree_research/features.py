"""Six causal, offspring-balanced features, independent of every live builder."""
from collections import Counter, defaultdict, deque
from datetime import timedelta
import math
import re

from ..d39_next.features import race_reasons
from ..research import features as existing
from ..ml.safety import MLFailure
from . import contracts as c


def stable_id(value):
    return value if isinstance(value, str) and re.fullmatch('[0-9]{10}', value) and value != '0000000000' else None


def cell(surface, distance):
    # Flat turf/dirt only. Sand, jumps, mixed/unknown tracks are not reassigned.
    kind = 'turf' if surface in tuple(map(str, range(10, 23))) else 'dirt' if surface in ('23','24','25','26','29') else None
    d = existing.num(distance, 100, 10000)
    if kind is None or d is None:
        return None
    return kind + ':' + ('le1400' if d <= 1400 else 'gt1400_le1800' if d <= 1800 else 'gt1800')


class Pool:
    """O(1) leave-one-horse-out mean of per-horse means, not start-weighted."""
    def __init__(self):
        self.horses = {}
        self.total = 0.

    def add(self, horse, score):
        count, summed = self.horses.get(horse, (0, 0.))
        if count:
            self.total -= summed / count
        self.horses[horse] = (count + 1, summed + score)
        self.total += (summed + score) / (count + 1)

    def without(self, horse):
        item = self.horses.get(horse)
        return (len(self.horses) - int(item is not None),
                self.total - item[1] / item[0] if item else self.total)


class Builder:
    def __init__(self, lineage):
        self.lineage = {stable_id(h): {r: stable_id(v.get(r)) for r in ('sire', 'damsire')}
                        for h, v in lineage.items() if stable_id(h)}
        self.pending = deque()
        self.global_pool = Pool()
        self.overall = defaultdict(Pool)
        self.cells = defaultdict(Pool)
        self.last_history_day = None
        self.asof_day = None
        self.counts = Counter()

    def _available(self, day):
        if self.asof_day is not None and day < self.asof_day:
            raise MLFailure('invalid_split')
        self.asof_day = day
        while self.pending and self.pending[0][0] + timedelta(days=7) <= day:
            _, events = self.pending.popleft()
            for horse, key, score in events:
                self.global_pool.add(horse, score)
                for role, parent in self.lineage.get(horse, {}).items():
                    if parent:
                        self.overall[(role, parent)].add(horse, score)
                        self.cells[(role, parent, key)].add(horse, score)
                self.counts['available_clean_starts'] += 1

    def values(self, horse, surface, distance):
        horse = stable_id(horse)
        n, summed = self.global_pool.without(horse)
        prior = summed / n if n else .5
        key = cell(surface, distance)
        out = {}
        for role in ('sire', 'damsire'):
            parent = self.lineage.get(horse, {}).get(role)
            whole = self.overall.get((role, parent)) if parent else None
            count, summed = whole.without(horse) if whole else (0, 0.)
            level = (summed + 10. * prior) / (count + 10.)
            local = self.cells.get((role, parent, key)) if parent and key else None
            support, local_sum = local.without(horse) if local else (0, 0.)
            conditioned = (local_sum + 5. * level) / (support + 5.)
            out.update({f'p_{role}_level': level,
                        f'p_{role}_condition_deviation': conditioned - level,
                        f'p_{role}_support': support})
        return out

    def transform_day(self, rows, *, prediction=False):
        if not rows:
            return []
        identities = [existing.legacy.entry_identity(r) for r in rows]
        day = identities[0][1]
        if any(d != day for _, d in identities) or len({k for k, _ in identities}) != len(rows):
            raise MLFailure('invalid_identity')
        if not prediction and self.last_history_day is not None and day <= self.last_history_day:
            raise MLFailure('invalid_split')
        self._available(day)
        values = [self.values(r.get('horse_id'), r.get('surface'), r.get('distance')) for r in rows]
        # The prediction path never inspects the current finish, margins, time or anomaly.
        if not prediction:
            groups = defaultdict(list)
            for row, (key, _) in zip(rows, identities):
                groups[key[:-1]].append(row)
            events = []
            for group in groups.values():
                reasons = race_reasons(group)
                ids = [stable_id(r.get('horse_id')) for r in group]
                if None in ids or len(set(ids)) != len(ids):
                    reasons.append('invalid_or_duplicate_horse_identity')
                if reasons:
                    self.counts['excluded_history_races'] += 1
                    continue
                for r, horse in zip(group, ids):
                    key = cell(r.get('surface'), r.get('distance'))
                    score = existing.percentile(existing.num(r.get('finish_position')), existing.num(r.get('actual_runners')))
                    if key and score is not None and math.isfinite(score):
                        events.append((horse, key, score))
                    else:
                        self.counts['unsupported_condition_starts'] += 1
            self.pending.append((day, events))
            self.last_history_day = day
        return values
