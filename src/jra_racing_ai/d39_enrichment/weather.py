"""Official WE initial state is required; future/ambiguous changes are never backdated."""
from datetime import date, datetime, timedelta
from ..win_sim import core

SCOPE = ('kaisai_nen','kaisai_tsukihi','keibajo_code','kaisai_kai','kaisai_nichime')
VALID = ('1','2','3','4')


def surface_field(track):
    if track in tuple(str(i) for i in range(10,23)) or track in ('51','53','54','55','56','57','58','59'):
        return 'babajotai_code_shiba'
    if track in ('23','24','25','26','29'):
        return 'babajotai_code_dirt'
    return None  # Sand and mixed turf/dirt jumps need a separate definition.


def published(row):
    raw = row.get('happyo_tsukihi_jifun')
    if not isinstance(raw,str) or len(raw)!=8 or not raw.isascii() or not raw.isdigit():
        return None
    try:
        value=datetime(int(row['kaisai_nen']),int(raw[:2]),int(raw[2:4]),int(raw[4:6]),int(raw[6:]),tzinfo=core.JST)
        day=date(int(row['kaisai_nen']),int(row['kaisai_tsukihi'][:2]),int(row['kaisai_tsukihi'][2:]))
        if value.date()!=day:
            return None
        return value+timedelta(minutes=1)
    except (ValueError,TypeError,KeyError):
        return None


def cutoff_for(row):
    raw=row.get('start_time')
    if row.get('previous_start') not in (None,'','0000','    '):
        return None
    if not isinstance(raw,str) or len(raw)!=4 or not raw.isascii() or not raw.isdigit():
        return None
    try:
        return datetime(int(row['kaisai_nen']),int(row['kaisai_tsukihi'][:2]),int(row['kaisai_tsukihi'][2:]),
            int(raw[:2]),int(raw[2:]),tzinfo=core.JST)-timedelta(minutes=10)
    except (ValueError,TypeError):
        return None


class Resolver:
    def __init__(self,rows):
        self.scopes={}
        for row in rows:
            self.scopes.setdefault(tuple(row.get(k) for k in SCOPE),[]).append(row)

    def resolve(self,row,cutoff=None):
        cutoff=cutoff or cutoff_for(row);field=surface_field(row.get('surface'))
        if cutoff is None or field is None:
            return None
        events=[]
        for event in self.scopes.get(tuple(row.get(k) for k in SCOPE),[]):
            stamp=published(event)
            if stamp is None:
                return None  # Cannot prove an unparseable change was after the cutoff.
            if stamp<=cutoff:
                events.append((stamp,event))
        events.sort(key=lambda v:v[0])
        state=None; seen={}
        for stamp,event in events:
            kind=event.get('henko_shikibetsu')
            signature=(kind,event.get(field),event.get(field+'_henkomae'))
            if stamp in seen:
                if seen[stamp]!=signature:
                    return None
                continue
            seen[stamp]=signature
            if event.get('data_kubun')!='1' or kind not in ('1','2','3'):
                return None
            if kind=='2':
                continue
            value=event.get(field)
            if value not in VALID:
                return None
            if kind=='1':
                if state is not None:
                    return None
                state=value
            elif state is None or event.get(field+'_henkomae')!=state:
                return None
            else:
                state=value
        return state
