"""Date-batched causal features; full-race C and full-JRA-population D have separate paths."""
from collections import defaultdict, deque
from datetime import date, timedelta
import math
import re

from ..ml import features as legacy
from ..ml.safety import MLFailure
from .contracts import ALL_FEATURES, GROUPS

STARTERS = frozenset('04567')  # Official 2101; 1/2/3 never started.
GRADES = frozenset('ABCDEFGHL')
MARGINS = {' 12':.5,' 34':.75,'1  ':1.,'112':1.5,'114':1.25,'134':1.75,
    '2  ':2.,'212':2.5,'3  ':3.,'312':3.5, **{f'{n}  ':float(n) for n in range(4,10)},
    'A  ':10.,'D  ':0.,'H  ':None,'K  ':None,'T  ':None,'Z  ':None,
    ' 14':.25,'214':2.25,'734':7.75}


def num(value, low=None, high=None):
    return legacy.number(value, low=low, high=high)


def person(value):
    return value if isinstance(value,str) and re.fullmatch('[0-9]{5}',value) and value!='00000' else None


def condition(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9]{3}',value):
        return None
    n=int(value)
    return value if 1<=n<=100 or value in ('701','702','703','999') else None


def grade(value):
    # A blank is the published general-race/initial code, kept as a named category.
    return 'GENERAL_OR_INITIAL' if value==' ' else value if value in GRADES else None


def late3f(value, race_type):
    # JV SE item59: 999/000 unavailable. Obstacle records contain a 1F average, not 3F.
    if (race_type not in ('11','12','13','14','21','22','23','24') or not isinstance(value,str)
            or not re.fullmatch('[0-9]{3}',value) or value in ('000','999')):
        return None
    return int(value)/10.


def percentile(rank, count, *, ties=False):
    if rank is None or count is None or (not ties and not float(rank).is_integer()) or not float(count).is_integer() or not 2<=count<=28 or not 1<=rank<=count:
        return None
    return (count-rank)/(count-1)


def weighted(recent, field):
    pairs=[(e.get(field), w) for e,w in zip(reversed(recent), (1.,.5,.25)) if e.get(field) is not None]
    return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None


def source_info(row):
    identity, day=legacy.entry_identity(row)
    age=num(row.get('age'),2,30)
    group=min(int(age),5) if age is not None and age.is_integer() else None
    c=condition(row.get('condition_'+str(group))) if group is not None else None
    g=grade(row.get('grade'))
    if row.get('previous_grade_revision') not in (None,'',' '):
        g=None  # Revision time is not available; do not backdate the changed grade.
    jockey=person(row.get('jockey'))
    if row.get('previous_jockey_revision') not in (None,'','00000','     '):
        jockey=None
    return dict(identity=identity, day=day, grade=g, historical_grade=grade(row.get('grade')),
                condition=c, jockey=jockey, actual_jockey=person(row.get('jockey')),
                trainer=person(row.get('trainer')), rank=num(row.get('finish_position'),1,28),
                actual=num(row.get('actual_runners'),2,28), started=row.get('anomaly') in STARTERS,
                late=late3f(row.get('late3f_raw'),row.get('race_type')))


def race_summaries(rows):
    """Uses every actual starter before any modeling-cohort filtering."""
    infos=[source_info(r) for r in rows]
    starters=[i for i in infos if i['started']]
    totals={i['actual'] for i in infos}
    complete=(len(totals)==1 and None not in totals and len(starters)==next(iter(totals))
              and all(i['late'] is not None for i in starters))
    times=[i['late'] for i in starters] if complete else []
    mean=sum(times)/len(times) if times else None
    out=[]
    for row,info in zip(rows,infos):
        rank=info['rank']
        value=info['late']
        time_rank=(1+sum(t<value for t in times)+(sum(t==value for t in times)-1)/2) if times and value is not None else None
        margin=row.get('margin1')
        out.append(dict(info, row=row, a_finish=percentile(rank,info['actual']),
            c_time=value, c_rank=percentile(time_rank, float(len(times)), ties=True) if time_rank is not None else None,
            c_delta=value-mean if value is not None and mean is not None else None,
            c_corner=percentile(num(row.get('corner4'),1,28),info['actual']),
            c_margin=MARGINS.get(margin), c_code=margin.replace(' ','_') if margin in MARGINS else None,
            c_extra=float(any(row.get(k) not in (None,'','   ') for k in ('margin2','margin3')))))
    return out


class Builder:
    """Streaming source dates; only seven-day-old events reach either history path."""
    def __init__(self):
        self.last_day=None
        self.pending=deque()
        self.horses={}
        self.people=defaultdict(lambda:[0,0,0])
        self.population=deque()
        self.global_counts=[0,0,0]

    def _available(self, day):
        while self.pending and self.pending[0][0]+timedelta(days=7)<=day:
            old_day, events=self.pending.popleft()
            for e in events:
                e=dict(e,grade=e['historical_grade'])
                raw=e['row']; horse=raw.get('horse_id')
                if horse:
                    h=self.horses.setdefault(horse,{'recent':deque(maxlen=3),'count':0,'sum':0.,'distance':{},'course':{},'last':None})
                    r=e['rank']
                    if r is not None and r.is_integer():
                        h['count']+=1;h['sum']+=r;h['last']=e
                        distance=num(raw.get('distance'),100,10000)
                        if distance is not None:
                            for field,key in (('distance',distance),('course',(distance,raw.get('keibajo_code'),raw.get('surface')))):
                                pair=h[field].setdefault(key,[0.,0]);pair[0]+=r;pair[1]+=1
                    if e['started']:
                        h['recent'].append(e)
                if e['started']:
                    if e['rank'] is not None and e['rank'].is_integer():
                        y=int(e['rank']==1)
                    elif raw.get('anomaly') in ('4','5'):
                        y=0
                    else:
                        y=None
                    entry=(old_day,e['actual_jockey'],e['trainer'],y)
                    self.population.append(entry)
                    self._person_update(entry,1)
        cutoff=day-timedelta(days=365)
        while self.population and self.population[0][0]<cutoff:
            self._person_update(self.population.popleft(),-1)

    def _person_update(self, entry, sign):
        _,jockey,trainer,y=entry
        for counts in [self.global_counts]+[self.people[(kind,p)] for kind,p in (('jockey',jockey),('trainer',trainer)) if p]:
            counts[0]+=sign;counts[1]+=sign*(y or 0);counts[2]+=sign*(y is None)

    def transform_day(self, rows):
        if not rows:
            return [], []
        day=legacy.entry_identity(rows[0])[1]
        if self.last_day is not None and day<=self.last_day:
            raise MLFailure('invalid_split')
        if any(legacy.entry_identity(r)[1]!=day for r in rows):
            raise MLFailure('invalid_split')
        self._available(day)
        races=defaultdict(list)
        for row in rows:
            races[legacy.entry_identity(row)[0][:-1]].append(row)
        all_features,metadata,events=[],[],[]
        for race, group in sorted(races.items()):
            summaries=race_summaries(group)
            valid_labels=all(e['rank'] is not None and e['rank'].is_integer() for e in summaries)
            single=valid_labels and sum(e['rank']==1 for e in summaries)==1
            pairs=[]
            for e in summaries:
                raw=e['row']; identity=e['identity']
                h=self.horses.get(raw.get('horse_id'),{})
                last=h.get('last');recent=h.get('recent',())
                prev=recent[-1] if recent else None
                distance=num(raw.get('distance'),100,10000)
                dist=h.get('distance',{}).get(distance)
                course=h.get('course',{}).get((distance,race[2],raw.get('surface'))) if raw.get('surface') else None
                x={'venue':race[2],'surface':raw.get('surface') or None,'going':None,'sex':raw.get('sex') or None,
                    'distance':distance,'declared_runners':num(raw.get('declared_runners'),1,28),
                    'frame':num(raw.get('frame'),1,8),'horse_number':int(identity[-1]),'age':num(raw.get('age'),2,30),
                    'weight_kg':num(raw.get('weight_kg'),20,100),'past_starts':h.get('count',0),
                    'days_since_previous':(day-last['day']).days if last else None,
                    'previous_finish':last['rank'] if last else None,'previous_time_seconds':None,
                    'past_mean_finish':h['sum']/h['count'] if h.get('count') else None,
                    'same_distance_mean_finish':dist[0]/dist[1] if dist else None,
                    'same_course_mean_finish':course[0]/course[1] if course else None,
                    'a_previous_finish_percentile':prev['a_finish'] if prev else None,
                    'a_recent3_finish_percentile':weighted(recent,'a_finish'),
                    'b_grade':e['grade'],'b_condition':e['condition'],
                    'b_race_type':raw.get('race_type') if raw.get('race_type') in ('11','12','13','14','18','19','21','22','23','24') else None,
                    'b_weight_type':raw.get('weight_type') if raw.get('weight_type') in tuple('1234') else None,
                    'b_previous_grade':prev['grade'] if prev else None,'b_previous_condition':prev['condition'] if prev else None,
                    'b_condition_transition':prev['condition']+'>'+e['condition'] if prev and prev['condition'] and e['condition'] else None,
                    'b_grade_transition':prev['grade']+'>'+e['grade'] if prev and prev['grade'] and e['grade'] else None,
                    'c_previous_late3f_seconds':prev['c_time'] if prev else None,
                    'c_previous_late3f_rank_percentile':prev['c_rank'] if prev else None,
                    'c_previous_late3f_minus_race_mean':prev['c_delta'] if prev else None,
                    'c_recent3_late3f_rank_percentile':weighted(recent,'c_rank'),
                    'c_previous_corner4_percentile':prev['c_corner'] if prev else None,
                    'c_previous_adjacent_margin_lengths':prev['c_margin'] if prev else None,
                    'c_previous_margin_code':prev['c_code'] if prev else None,
                    'c_previous_additional_margin_present':prev['c_extra'] if prev else None}
                gn,gw,gu=self.global_counts
                prior=gw/gn if gn and not gu else None
                for role,p in (('jockey',e['jockey']),('trainer',e['trainer'])):
                    counts=self.people.get((role,p),[0,0,0]) if p else None
                    x[f'd_{role}_starts_365d']=counts[0] if counts is not None else None
                    x[f'd_{role}_smoothed_win_365d']=(counts[1]+20*prior)/(counts[0]+20) if counts is not None and not counts[2] and prior is not None else None
                if set(x)!=set(ALL_FEATURES):
                    raise MLFailure('invalid_input')
                pairs.append((x,{'key':identity,'date':day.isoformat(),'winner_label':int(e['rank']==1) if single else None,
                    'eligible':single,'start_time':raw.get('start_time'),'previous_start':raw.get('previous_start'),
                    'record_status':'7','jockey_population_scope':'all_JRA_domestic_starters'}))
            # Optional A/B/C/D values never affect this common legacy-cohort mask.
            priority=('venue','surface','distance','age','weight_kg')
            eligible=single and all(x[k] is not None for x,_ in pairs for k in priority)
            for x,m in pairs:
                m['eligible']=eligible
                all_features.append(x);metadata.append(m)
            events.extend(summaries)
        self.pending.append((day,events));self.last_day=day
        return all_features,metadata


def build(rows, *, history=()):
    """The same generator is used by training-cache preparation and inference."""
    wanted={legacy.entry_identity(r)[0] for r in rows}
    combined={}
    for row in [*history,*rows]:
        key,day=legacy.entry_identity(row)
        if key in combined and combined[key]!=row:
            raise MLFailure('duplicate_entry')
        combined[key]=row
    days=defaultdict(list)
    for row in combined.values():
        days[legacy.entry_identity(row)[1]].append(row)
    b=Builder();out={}
    for day,group in sorted(days.items()):
        x,m=b.transform_day(group)
        out.update({tuple(v['key']):f for f,v in zip(x,m) if tuple(v['key']) in wanted})
    return [out[legacy.entry_identity(r)[0]] for r in rows]
