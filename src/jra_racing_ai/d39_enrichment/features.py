from collections import deque
from datetime import timedelta
from ..d39_next import features as original


class Builder(original.Builder):
    def __init__(self,time_index,going_resolver):
        super().__init__()
        self.time_index=time_index;self.going_resolver=going_resolver
        self.time_pending=deque();self.past_times={}

    def _available(self,day):
        super()._available(day)
        while self.time_pending and self.time_pending[0][0]+timedelta(days=7)<=day:
            _,events=self.time_pending.popleft()
            for e in events:
                row=e['row'];key=''.join(e['identity']);item=self.time_index.get(key)
                if (not item or item['horse_id']!=row.get('horse_id')
                        or original.legacy.num(item['finish_position'])!=original.legacy.num(row.get('finish_position'))):
                    continue
                seconds=item.get('time_seconds')
                if seconds is not None:
                    self.past_times[(row['horse_id'],original.legacy.num(row.get('distance')),row.get('surface'))]=seconds

    def transform_day(self,rows,*,prediction=False):
        x,meta=super().transform_day(rows,prediction=prediction)
        if not rows:return x,meta
        if not prediction:
            self.time_pending.append((self.pending[-1][0],list(self.pending[-1][1])))
        by_key={original.legacy.legacy.entry_identity(r)[0]:r for r in rows}
        for values,m in zip(x,meta):
            row=by_key[tuple(m['key'])]
            values['previous_time_seconds']=self.past_times.get((row.get('horse_id'),original.legacy.num(row.get('distance')),row.get('surface')))
            values['going']=self.going_resolver.resolve(row) if self.going_resolver else None
        return x,meta
