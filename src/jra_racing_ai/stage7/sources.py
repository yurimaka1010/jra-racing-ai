"""Projection of trusted pretty JSON: outcome/raw payload fields are never parsed."""
import json,re
from datetime import date
from ..ml import safety
from . import contracts as c,store as s

ALLOWED=frozenset({'race_key','scheduled_start','schedule_asof_verified','quotes','failures'})

def packets(path):
    row=None;field=None;buf=[];records=False
    def value():
        if not buf:return {}
        return json.loads('{'+''.join(buf).strip().rstrip(',')+'}')
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not records:
                if line.strip()=='"records": [':records=True
                continue
            if row is None:
                if line.rstrip()=='    {':row={};field=None;buf=[]
                continue
            match=re.match(r'^      "([a-z_]+)":',line)
            if match or line.startswith('    }'):
                if field in ALLOWED:
                    row.update(value())
                    if field=='race_key':
                        key=row['race_key'];d=date(int(key[0]),int(key[1][:2]),int(key[1][2:]))
                        if d>date(2025,9,30):return
                        c.historical(key)
                field=match[1] if match else None;buf=[]
            if line.startswith('    }'):
                if not ALLOWED<=set(row):raise safety.MLFailure('artifact_invalid')
                yield row;row=None;continue
            # Denied fields (including malformed outcome text) are not accumulated
            # or JSON-decoded. Only structurally scoped allowlisted values exist.
            if field in ALLOWED:buf.append(line)
    if row is not None:raise safety.MLFailure('artifact_invalid')

def b_rosters():
    values={};references={}
    for year in c.YEARS:
        path=c.S2/'trials'/f'{year}_annual_market_control.predictions.local.json'
        references[str(path)]=s.sha(path);raw=s.read(path);n=0
        for part in raw:
            for item in part:
                race=tuple(item['race_key']);d=c.historical(race)
                if d.year!=year or d.month>9 or race in values:raise safety.MLFailure('invalid_identity')
                values[race]=tuple(sorted(item['probabilities']));n+=1
        if n!=c.EXPECTED[year]:raise safety.MLFailure('invalid_identity')
    if sum(map(len,values.values()))!=87601:raise safety.MLFailure('invalid_identity')
    return values,references

def roles(day):
    result=[]
    for year in c.YEARS:
        p=year-1
        if date(2020,1,1)<=day<=date(p,8,31):result.append(f'{year}_train')
        if date(p,9,8)<=day<=date(p,10,31):result.append(f'{year}_early')
        if date(p,11,8)<=day<=date(p,12,24):result.append(f'{year}_calibration')
    return result

def prepare():
    baseline,refs=b_rosters();targets={};seen=set()
    known=s.read(c.S4/'references.json')
    for year in range(2020,2026):
        for path in sorted((c.S2/'market').glob(f'{year}-*.local.json')):
            digest=s.sha(path)
            if known.get(str(path))!=digest:raise safety.MLFailure('artifact_invalid')
            refs[str(path)]=digest
            for row in packets(path):
                race=tuple(row['race_key']);day=c.historical(race)
                if race in seen:raise safety.MLFailure('duplicate_entry')
                seen.add(race);groups=roles(day)
                if race in baseline:groups.append(f'{day.year}_evaluation')
                if not groups:continue
                q=row['quotes']['10']
                # Original H input universe: complete saved T10 market rosters.
                if q is None:
                    if race in baseline:raise safety.MLFailure('invalid_identity')
                    continue
                roster=tuple(sorted(q['odds']))
                if race in baseline and roster!=baseline[race]:raise safety.MLFailure('invalid_identity')
                targets[race]={'race_key':list(race),'roster':list(roster),'scheduled_start':row['scheduled_start'],
                    'schedule_asof_verified':bool(row['schedule_asof_verified']),'quotes':row['quotes'],'roles':groups,
                    'source_packet_file':str(path),'in_B':race in baseline}
    if {r for r,v in targets.items() if v['in_B']}!=set(baseline):raise safety.MLFailure('invalid_identity')
    return [targets[r] for r in sorted(targets)],refs
