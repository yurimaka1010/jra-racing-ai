"""Local CSV snapshots of already joined D39 results; no fitting or DB access."""
from datetime import datetime
from io import StringIO
from pathlib import Path
from threading import RLock
from uuid import uuid4
import csv
import hashlib
import json
import math
import os

from ..ml import safety
from ..keiba_lab_export import COLUMNS
from ..win_sim import core
from . import contracts as c

ROOT=c.BASE/'keiba-lab-next-results'
CHOICES={'original':('old_D39','new_D39','control_D39','next_D39'),
         'improved':('old_D39','new_D39','control_D39','next_D39','enriched_D39')}
DETAIL_COLUMNS=(*COLUMNS,'input_cutoff','result_status','payout_status','result_reason',
    'virtual_selected','virtual_status','virtual_stake_yen','virtual_payout_yen','virtual_net_profit_yen','virtual_hit',
    'common_probability_cohort','common_financial_cohort','result_content_sha256')
SUMMARY_COLUMNS=('date','candidate','model_id','model_artifact_id','evaluation_races','evaluation_runners',
    'race_log_loss','horse_log_loss','brier_score','roc_auc','within_race_auc','top1_accuracy_percent',
    'top2_percent','top3_percent','winner_prediction_mean_rank','financial_common_forecast_races',
    'bet_count','skip_count','investment_yen','hit_count','hit_rate_percent','payout_yen','net_profit_yen',
    'return_rate_percent','unsettled_count','settled_bet_count','evaluation_kind','source_run_id','result_content_sha256')
_LOCK=RLock()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def stable_body(body,*,without_names=False):
    value={k:body.get(k) for k in ('date','races','metrics','financial','common_result_races','pending_result_races','kind')}
    if without_names:
        value=dict(value,races=[{k:v for k,v in r.items() if k!='horse_names'} for r in body.get('races',[])])
    return value


def safe_cell(value):
    if value is None:return ''
    if isinstance(value,float) and not math.isfinite(value):raise safety.MLFailure('invalid_input')
    # Preserve numeric negatives; protect text cells when opened in Excel.
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r','\n')):
        return "'"+value
    return value


def csv_text(rows,columns):
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\r\n')
    writer.writeheader();writer.writerows({k:safe_cell(r.get(k)) for k in columns} for r in rows)
    value='\ufeff'+stream.getvalue()
    if len(rows)>20000 or len(value.encode('utf-8'))>8*1024*1024:raise safety.MLFailure('input_limit')
    return value


def build(packet,choice):
    if choice not in CHOICES:raise safety.MLFailure('invalid_input')
    day=packet.get('date')
    if day not in ('2026-10-24','2026-10-25'):raise safety.MLFailure('invalid_date')
    body=packet.get('results')
    if not body or not body.get('races'):
        return csv_text([],DETAIL_COLUMNS),csv_text([],SUMMARY_COLUMNS),{'status':'pending','races':0,'rows':0,'date':day}
    names=CHOICES[choice]
    if (body.get('date')!=day or body.get('H_results_included') is not False
            or body.get('Stage7_results_read') is not False
            or any(n not in names for field in ('metrics','financial') for n in body.get(field,{}))
            or any(n not in names for r in body['races'] for n in r.get('models',{}))):
        raise safety.MLFailure('invalid_input')
    run=packet['activation']['run_id'];contexts={r['race_id']:r for r in packet.get('races',[])}
    stable=stable_body(body)
    fingerprint=digest(stable);detail=[];identities=set();model_ids={}
    financial_races=sum(all(r['models'].get(n,{}).get('status')=='saved' for n in names) for r in body['races'])
    for race in body['races']:
        context=contexts.get(race['race_id'])
        if not context or context.get('source_prediction_sha256')!=race.get('source_prediction_sha256'):
            raise safety.MLFailure('artifact_invalid')
        common_financial=all(race['models'].get(n,{}).get('status')=='saved' for n in names)
        common_probability=common_financial and race.get('result_status')=='confirmed'
        for name in names:
            model=race['models'].get(name,{})
            if model.get('status')!='saved':continue
            original=context['models'].get(name,{})
            if original.get('status')!='saved' or model['model_id']!=original['model_id']:
                raise safety.MLFailure('artifact_invalid')
            saved={r['horse_no']:r for r in original['rows']}
            if (len(saved)!=len(model['rows']) or set(saved)!={r['horse_no'] for r in model['rows']}
                    or any(r['rank']!=saved[r['horse_no']]['rank'] or r['probability']!=saved[r['horse_no']]['probability'] for r in model['rows'])):
                raise safety.MLFailure('artifact_invalid')
            model_ids.setdefault(name,set()).add(model['model_id'])
            selected=model.get('paper_horse');ticket=model.get('paper_ticket')
            for row in model['rows']:
                identity=(race['race_id'],name,row['horse_no'])
                if identity in identities:raise safety.MLFailure('duplicate_entry')
                identities.add(identity);is_selected=selected==row['horse_no'] and selected is not None
                money=ticket if is_selected and ticket else {}
                detail.append(dict(race_id=race['race_id'],date=day,venue=race.get('venue'),race_no=race.get('race_no'),
                    race_name='',surface=context.get('next_surface'),distance=context.get('next_distance'),going='',
                    **{'class':''},horse_no=int(row['horse_no']),horse_name=(race.get('horse_names') or {}).get(row['horse_no']),
                    model_id=name,model_artifact_id=model['model_id'],pred_rank=row['rank'],win_prob=row['probability'],
                    finish_position=row.get('finish_position'),win_payout_100=row.get('win_payout_100'),
                    predicted_at=model['predicted_at'],start_time=race.get('scheduled_start'),input_cutoff=race.get('input_cutoff'),
                    schema_version='keiba-lab.next-results.v1',dataset_id='local-next-'+day+'-'+choice,
                    evaluation_kind='formal_T10_D39_prospective_exploratory',source_stage='d39_enrichment' if choice=='improved' else 'd39_next',
                    source_run_id=run,source_sha256=race['source_prediction_sha256'],result_status=race.get('result_status'),
                    payout_status=race.get('payout_status'),result_reason=race.get('reason') or race.get('payout_reason'),
                    virtual_selected=is_selected,virtual_status=money.get('status','not_selected'),
                    virtual_stake_yen=money.get('stake_yen',0),virtual_payout_yen=money.get('payout_yen') if is_selected else None,
                    virtual_net_profit_yen=money.get('net_profit_yen') if is_selected else None,
                    virtual_hit=money.get('hit') if is_selected else None,common_probability_cohort=common_probability,
                    common_financial_cohort=common_financial,result_content_sha256=fingerprint))
    summary=[]
    for name in names:
        m=body.get('metrics',{}).get(name,{});f=body.get('financial',{}).get(name,{})
        if not m and not f:continue
        ids=model_ids.get(name,set())
        if len(ids)>1:raise safety.MLFailure('invalid_model')
        summary.append(dict(date=day,candidate=choice,model_id=name,model_artifact_id=next(iter(ids),None),
            evaluation_races=m.get('races'),evaluation_runners=m.get('runners'),financial_common_forecast_races=financial_races,
            **{k:m.get(k) for k in ('race_log_loss','horse_log_loss','brier_score','roc_auc','within_race_auc',
                'top1_accuracy_percent','top2_percent','top3_percent','winner_prediction_mean_rank')},
            **{k:f.get(k) for k in ('bet_count','skip_count','investment_yen','hit_count','hit_rate_percent','payout_yen',
                'net_profit_yen','return_rate_percent','unsettled_count','settled_bet_count')},
            evaluation_kind='formal_T10_D39_prospective_exploratory',source_run_id=run,result_content_sha256=fingerprint))
    info={'status':'saved','date':day,'races':len(body['races']),'rows':len(detail),'summary_rows':len(summary),
        'candidate':choice,'source_run_id':run,'result_content_sha256':fingerprint,
        'raw_result_content_sha256':digest(stable_body(body,without_names=True)),
        'source_checked_at':body.get('checked_at'),'H_results_included':False,'Stage7_results_read':False}
    return csv_text(detail,DETAIL_COLUMNS),csv_text(summary,SUMMARY_COLUMNS),info


def atomic_text(path,value):
    temp=safety.local_path(path.with_name(uuid4().hex+'.tmp'),root=ROOT)
    with temp.open('x',encoding='utf-8',newline='') as stream:stream.write(value)
    os.replace(temp,path)


def persist(packet,choice):
    detail,summary,info=build(packet,choice)
    if info['status']!='saved':return info
    revision=hashlib.sha256((detail+summary).encode('utf-8')).hexdigest()
    directory=safety.local_path(ROOT/info['date']/choice,root=ROOT)
    with _LOCK:
        target=safety.local_path(directory/'revisions'/revision,root=ROOT)
        if not target.exists():
            partial=safety.local_path(directory/'revisions'/uuid4().hex,root=ROOT);partial.mkdir(parents=True)
            for name,value in (('results.csv',detail),('summary.csv',summary)):
                with (partial/name).open('x',encoding='utf-8',newline='') as stream:stream.write(value)
            safety.write_json(partial/'source_results.local.json',packet['results'],root=ROOT)
            proof=dict(info,revision=revision,created_at=datetime.now(core.JST).isoformat(),
                hashes={n:safety.digest(partial/n,root=ROOT) for n in ('results.csv','summary.csv','source_results.local.json')})
            safety.write_json(partial/'manifest.json',proof,root=ROOT)
            # Both resolved paths have been checked inside this feature-owned root.
            partial.rename(target)
        manifest=safety.read_json(target/'manifest.json',root=ROOT)
        for name in ('results.csv','summary.csv','source_results.local.json'):
            if safety.digest(target/name,root=ROOT)!=manifest['hashes'][name]:raise safety.MLFailure('artifact_invalid')
        current=directory/'current.json'
        old=safety.read_json(current,root=ROOT) if current.exists() else {}
        if old.get('revision')!=revision:
            atomic_text(directory/'results.csv',detail);atomic_text(directory/'summary.csv',summary)
            atomic_text(current,json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        return dict(info,revision=revision,directory=str(directory),
            detail_url=f'/api/next-results.csv?date={info["date"]}&candidate={choice}&kind=detail',
            summary_url=f'/api/next-results.csv?date={info["date"]}&candidate={choice}&kind=summary')


def export_status(day,choice):
    if choice not in CHOICES or day not in ('2026-10-24','2026-10-25'):raise safety.MLFailure('invalid_input')
    directory=safety.local_path(ROOT/day/choice,root=ROOT);path=directory/'current.json'
    if not path.exists():return {'status':'pending','date':day,'races':0,'rows':0}
    info=safety.read_json(path,root=ROOT)
    return dict(info,directory=str(directory),
        detail_url=f'/api/next-results.csv?date={day}&candidate={choice}&kind=detail',
        summary_url=f'/api/next-results.csv?date={day}&candidate={choice}&kind=summary')


def download(day,choice,kind):
    if kind not in ('detail','summary'):raise safety.MLFailure('invalid_input')
    info=export_status(day,choice)
    if info['status']!='saved':
        return csv_text([],DETAIL_COLUMNS if kind=='detail' else SUMMARY_COLUMNS)
    name='results.csv' if kind=='detail' else 'summary.csv'
    path=safety.local_path(ROOT/day/choice/'revisions'/info['revision']/name,root=ROOT,exists=True)
    if safety.digest(path,root=ROOT)!=info['hashes'][name]:raise safety.MLFailure('artifact_invalid')
    return path.read_bytes().decode('utf-8')


def refresh_saved(day,choice):
    """A single local writer watches result JSONs. Unchanged results need no replay."""
    if choice not in CHOICES or day not in ('2026-10-24','2026-10-25'):raise safety.MLFailure('invalid_input')
    if choice=='original':
        from . import forward as pipeline
        store=c
    else:
        from ..d39_enrichment import contracts as store,forward as pipeline
    pointer=store.ROOT/'current.json'
    if not pointer.exists():return {'status':'pending','races':0,'rows':0}
    active=store.read(pointer);directory=store.folder(active['run_id']);source=directory/'latest_results.json'
    if not source.exists():return {'status':'pending','races':0,'rows':0}
    body=store.read(source)
    if body.get('date')!=day or body.get('H_results_included') is not False or body.get('Stage7_results_read') is not False:
        raise safety.MLFailure('invalid_input')
    old=export_status(day,choice);previous_body=None
    if old.get('status')=='saved':
        path=ROOT/day/choice/'revisions'/old['revision']/'source_results.local.json'
        if safety.digest(path,root=ROOT)!=old['hashes']['source_results.local.json']:raise safety.MLFailure('artifact_invalid')
        previous_body=safety.read_json(path,root=ROOT)
        if digest(stable_body(previous_body,without_names=True))==digest(stable_body(body,without_names=True)):
            return old
    if previous_body:
        names={r['race_id']:r for r in previous_body['races']}
        for r in body['races']:
            known=names.get(r['race_id'],{})
            if known.get('source_prediction_sha256')==r.get('source_prediction_sha256'):
                r['horse_names']=known.get('horse_names',{})
    forecasts=pipeline.saved(directory,day)
    return persist({'date':day,'activation':active,'results':body,'races':forecasts},choice)


def attach_exports(packet):
    exports={}
    for choice,body in (('original',packet),('improved',packet.get('improved'))):
        if body is None:continue
        try:exports[choice]=export_status(body['date'],choice)
        except Exception:exports[choice]={'status':'pending','reason':'csv_export_unavailable'}
    packet['csv_exports']=exports
    return packet
