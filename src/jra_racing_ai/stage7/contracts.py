from datetime import date,datetime,timedelta
from pathlib import Path
from ..ml.safety import MLFailure
from ..research.contracts import MARKET_FEATURES,HISTORY_FEATURES
from ..win_sim import core

BASE=Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
ROOT=BASE/'stage7-development'
PROJECT=Path(__file__).resolve().parents[3]
S2=BASE/'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S4=BASE/'stage4-development/evaluations/2f26b6573d2249d38ad96b4456f89c0f'
S5=BASE/'stage5-development/evaluations/598c62b4dbe54142a2abe4a98b435b12'
S6=BASE/'stage6-development/evaluations/719a0d3915194bc59b6e55eba2659576'
H_ID='4efe34cdef5b4da9a33c254c4586a2bf'
MODEL=S2/'models'/H_ID
H_FIELDS=MARKET_FEATURES+HISTORY_FEATURES
VERSION='stage7a_frozen_H_observed_T10_and_label_free_T3_v1'
SEED=20261019
YEARS=(2023,2024,2025)
EXPECTED={2023:2103,2024:2101,2025:2169}
TASK='JraRacingAI_Stage7_H_Research'

def historical(race):
    r=core.key(race);d=date(int(r[0]),int(r[1][:2]),int(r[1][2:]))
    if not date(2020,1,1)<=d<=date(2025,9,30) or r[2] not in tuple(f'{i:02d}' for i in range(1,11)):
        raise MLFailure('invalid_split')
    return d

def future(race,plan,now):
    r=core.key(race);d=date(int(r[0]),int(r[1][:2]),int(r[1][2:]))
    created=core.instant(plan['created_at'])
    if (not date.fromisoformat(plan['start_date'])<=d<=date.fromisoformat(plan['end_date'])
            or d<created.date() or now<created or d!=now.date() or r[2] not in tuple(f'{i:02d}' for i in range(1,11))):
        raise MLFailure('invalid_split')
    return d

def week_end(start,weeks):
    d=date.fromisoformat(start);return (d-timedelta(days=d.weekday())+timedelta(weeks=weeks-1,days=6)).isoformat()

def clock():return datetime.now(core.JST)
