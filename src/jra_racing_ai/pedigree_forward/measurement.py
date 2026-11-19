"""SE design from old ABCD/E predictions only; no H, 2026 labels or new calibration."""
from collections import Counter
from datetime import date,timedelta
from itertools import groupby
import csv
import math
import time

from ..core import RACE_KEY
from ..ml import safety
from ..stage4 import contracts as sources
from ..stage5.metrics import WeekBootstrap
from ..d39_next.features import race_reasons
from . import contracts as c

METRICS=('within_race_auc','roc_auc','race_log_loss','brier_score')


def monday(day):return day-timedelta(days=day.weekday())


def old_predictions(path,year):
    out={}
    for part in c.read(path):
        for row in part:
            race=tuple(row['race_key']);d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
            if d.year!=year or d.month>9 or year not in (2023,2024,2025) or race in out:
                raise safety.MLFailure('invalid_split')
            values=row['probabilities']
            if (not 2<=len(values)<=28 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values.values())
                    or abs(sum(values.values())-1)>1e-12):raise safety.MLFailure('invalid_input')
            out[race]=values
    return out


def calendar_bootstrap(weeks,calendar,*,repetitions=5000,seed=20261026):
    """Reuse unchanged WeekBootstrap methods, retaining empty calendar weeks too."""
    import numpy as np
    calendar=np.asarray(calendar,dtype=str);mapped={w:i for i,w in enumerate(calendar)}
    if len(calendar)!=21 or len(set(calendar))!=21 or any(w not in mapped for w in weeks):
        raise safety.MLFailure('invalid_split')
    instance=WeekBootstrap.__new__(WeekBootstrap)
    instance.weeks=calendar;instance.codes=np.asarray([mapped[w] for w in weeks],int)
    instance.sample=np.random.default_rng(seed).integers(21,size=(repetitions,21))
    instance.counts=np.asarray([np.bincount(row,minlength=21) for row in instance.sample],float)
    import hashlib
    instance.fingerprint=hashlib.sha256(instance.sample.tobytes()).hexdigest()
    return instance


def measure(directory):
    import numpy as np
    if (directory/'measurement.json').exists():raise safety.MLFailure('artifact_exists')
    began=time.monotonic();pairs={};refs={};all_races=set()
    for year in (2023,2024,2025):
        paths=(sources.S2/'trials'/f'{year}_annual_ability.predictions.local.json',
               sources.S3/'E_trials'/f'{year}.predictions.local.json')
        a,b=(old_predictions(p,year) for p in paths)
        if set(a)!=set(b) or any(set(a[r])!=set(b[r]) for r in a):raise safety.MLFailure('invalid_identity')
        pairs[year]=(a,b);all_races.update(a)
        refs.update({str(p):c.sha(p) for p in paths})
    # The source manifest contains only <=2025; filter months before opening.
    manifest=c.read(sources.S2/'source_manifest.json');truth={};excluded=Counter();seen=set()
    for item in manifest['months']:
        start=date.fromisoformat(item['start']);end=date.fromisoformat(item['end'])
        if start.year not in pairs or start.month>9:continue
        if end.year!=start.year or end>date(2025,9,30):raise safety.MLFailure('invalid_split')
        path=sources.S2/'source'/item['file']
        if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        refs[str(path)]=item['sha256']
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for key,group in groupby(csv.DictReader(stream),key=lambda r:tuple(r[k] for k in RACE_KEY)):
                if key not in all_races:continue
                if key in seen:raise safety.MLFailure('duplicate_entry')
                seen.add(key);rows=list(group);reasons=race_reasons(rows)
                expected=pairs[int(key[0])][0][key]
                if set(r['horse_number'] for r in rows)!=set(expected):reasons.append('roster_mismatch')
                if reasons:excluded.update(reasons);continue
                truth[key]={r['horse_number']:int(r['finish_position']=='01' or r['finish_position']=='1') for r in rows}
    if seen!=all_races:raise safety.MLFailure('artifact_invalid')
    windows=[]
    for year,(a,b) in pairs.items():
        ds=sorted(date(int(r[0]),int(r[1][:2]),int(r[1][2:])) for r in a)
        first=monday(ds[0]);last=monday(ds[-1])-timedelta(weeks=20)
        for tag,start in (('first21',first),('last21',last)):
            windows.append({'name':f'{year}_{tag}','start':str(start),'end':str(start+timedelta(weeks=20,days=6))})
    plan={'new_fits':0,'source_pair':'Stage2_annual_ABCD_vs_Stage3_annual_ABCD_plus_E',
        'sources':refs,'periods':[2023,2024,2025],'P1_filter':'existing_d39_next.race_reasons_full_raw_race',
        'windows':windows,'window_rule_fixed_before_bootstrap':'first_and_last21_calendar_weeks_of_each_saved_JanSep_cohort;may_overlap_not_independent',
        'bootstrap':{'repetitions':5000,'seed':20261026,'unit':'Monday_week_21_calendar_weeks_including_empty','paired':True},
        'H_Stage7_or_2026_results_used':False,'new3model_calibration_predictions_used':False}
    c.write(directory/'measurement_plan.json',plan)
    c.write(directory/'measurement_plan.commit.json',{'sha256':c.sha(directory/'measurement_plan.json'),'recorded_at':c.clock().isoformat()})
    output=[]
    for window in windows:
        year=int(window['name'][:4]);a,b=pairs[year]
        chosen=sorted(r for r in truth if r in a and window['start']<=f'{r[0]}-{r[1][:2]}-{r[1][2:]}'<=window['end'])
        y=[];pa=[];pb=[];race_codes=[];weeks=[]
        for code,race in enumerate(chosen):
            d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]));week=str(monday(d))
            for h in sorted(a[race]):
                y.append(truth[race][h]);pa.append(a[race][h]);pb.append(b[race][h]);race_codes.append(code);weeks.append(week)
        if not y:raise safety.MLFailure('insufficient_history')
        calendar=[str(date.fromisoformat(window['start'])+timedelta(weeks=i)) for i in range(21)]
        bootstrap=calendar_bootstrap(weeks,calendar)
        _,base=bootstrap.measure(np.asarray(y,int),np.asarray(pa),np.asarray(race_codes,int))
        _,new=bootstrap.measure(np.asarray(y,int),np.asarray(pb),np.asarray(race_codes,int))
        se={name:float(np.std(new[name]-base[name],ddof=1)) for name in METRICS}
        row=dict(window,races=len(chosen),runners=len(y),calendar_weeks=21,nonempty_weeks=len(set(weeks)),standard_error=se,
                 shared_draw_sha256=bootstrap.fingerprint)
        output.append(row)
        print(__import__('json').dumps({'status':'running','stage':'measurement','completed_windows':len(output),'races':len(chosen)}),flush=True)
        if time.monotonic()-began>600:raise safety.MLFailure('query_timeout')
    summary={name:{'median':float(np.median([w['standard_error'][name] for w in output])),
        'minimum':min(w['standard_error'][name] for w in output),'maximum':max(w['standard_error'][name] for w in output)} for name in METRICS}
    up=lambda value,step:math.ceil(value/step)*step
    suggestions={'primary_minimum_gain':up(1.96*summary['within_race_auc']['maximum'],.001),
        'point_guardrails':{'race_LL_increase_max':up(summary['race_log_loss']['maximum'],.001),
                            'Brier_increase_max':up(summary['brier_score']['maximum'],.00001)},
        'point_rule':'primary_point>=minimum_and_primary_95CI_lower>0;secondary_point<=margin',
        'interval_alternative':'same_primary_rule;secondary_95CI_upper<=same_margin(noninferiority)',
        'not_user_approved':True,'basis':'primary_1.96_times_max_SE_rounded_up;guardrails_at_least_max_measured_SE',
        'caveat':'old_different_model_pair;overlapping_windows;precision_guide_not_new_model_effect_or_power_guarantee'}
    result={'status':'measured','real_fits':0,'windows':output,'standard_error_summary':summary,
        'races_before_P1':len(all_races),'races_after_P1':len(truth),'excluded_reason_counts_overlapping':dict(excluded),
        'threshold_suggestions':suggestions,'source_hashes_unchanged':all(c.sha(__import__('pathlib').Path(p))==h for p,h in refs.items()),
        'elapsed_seconds':time.monotonic()-began,'used_2026_or_H_or_new_calibration_labels':False}
    c.write(directory/'measurement.json',result)
    lines=['# 0fit測定・閾値提案（第1報）','',
        '対象はStage2の年次ABCDとStage3のABCD＋Eの保存済み対予測。2023〜2025年各1〜9月。H／Stage7／2026年／今回3モデルの較正予測は不使用。',
        '各年の先頭21暦週・末尾21暦週の計6窓を事前に固定。同年の2窓は重複し得るため独立な6実験ではない。既定race_reasonsを全出走馬へ適用後、同一レース・同一馬集合で計算。',
        '既存WeekBootstrapの測定法を再利用し、5000回、seed20261026、同一の週抽出を両モデルへ使用。空週も21暦週の母数へ含める。',
        '', '|窓|レース数|頭数|実データのある週|','|---|---:|---:|---:|']
    for w in output:lines.append(f"|{w['name']} {w['start']}〜{w['end']}|{w['races']}|{w['runners']}|{w['nonempty_weeks']}/21|")
    lines+=['','|差の指標|SE中央値|最小|最大|','|---|---:|---:|---:|']
    for name,v in summary.items():lines.append(f"|{name}|{v['median']:.8f}|{v['minimum']:.8f}|{v['maximum']:.8f}|")
    g=suggestions['point_guardrails']
    lines+=['','## 承認前の提案','',
        f"主指標の最小改善幅：レース内AUC +{suggestions['primary_minimum_gain']:.3f}。95%区間下限>0も要求する案。",
        f"点推定型：ΔRace LL ≤ +{g['race_LL_increase_max']:.3f}、ΔBrier ≤ +{g['Brier_increase_max']:.5f}。各幅は測定SEの最大値以上。",
        '区間型：上記と同じマージンに対し、各悪化差の95%区間上限がマージン以内である非劣性判定案。点推定型より保守的。',
        'これらは旧モデル対のばらつきによる設計上の目安。新しい血統3モデルの効果量・必要レース数・検出力の保証ではない。閾値はユーザーが確定するまで未承認。',
        f"P1適用前{len(all_races)}レース、適用後{len(truth)}レース。理由別重複件数はmeasurement.json。追加学習0fit。"]
    with (directory/'measurement_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    return {'status':'measured','run_id':directory.name,'real_fits':0,'windows':len(output),'SE':summary,'suggestions':suggestions}
