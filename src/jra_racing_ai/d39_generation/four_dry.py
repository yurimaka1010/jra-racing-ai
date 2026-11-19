"""Declared-entry D39 smoke prediction; never a live record or outcome evaluation."""
from collections import Counter
from datetime import date
from uuid import uuid4
import numpy as np

from ..ml import safety
from ..research import model as matrices
from . import contracts as c,addendum,four_forward,four_sources,forward
from .report import csv_file


def run(parent,session,day_text,*,bundles=None,history=None):
    plan=addendum.approved(parent)
    try:day=date.fromisoformat(day_text or plan['start_date'])
    except ValueError:raise safety.MLFailure('invalid_date') from None
    if day<c.clock().date() or not plan['start_date']<=str(day)<=plan['end_date']:
        raise safety.MLFailure('invalid_split')
    with session() as db:
        cards=forward.existing.live_cards(db,day);groups,observed=four_sources.entries(db,day)
    target=forward.mkdir(parent/'dry-four'/uuid4().hex);out=[];reasons={};races=0;model_status=Counter()
    if cards:
        if bundles is None:bundles,errors=four_forward.load_bundles(plan)
        if history is None:history,history_meta=four_sources.history(parent,session,day)
        for race,card in sorted(cards.items()):
            if card['reason'] or not card['scheduled_start']:
                key=card['reason'] or 'schedule_missing';reasons[key]=reasons.get(key,0)+1;continue
            from ..win_sim import core
            if core.instant(card['scheduled_start'])<=c.clock():
                reasons['already_started']=reasons.get('already_started',0)+1;continue
            rows=groups.get(race,[])
            if len(rows)!=len(card['roster']) or {r['horse_number'] for r in rows}!=set(card['roster']):
                reasons['entry_roster_mismatch']=reasons.get('entry_roster_mismatch',0)+1;continue
            try:frame=forward.live_features(history,rows)
            except safety.MLFailure as error:
                reasons[error.code]=reasons.get(error.code,0)+1;continue
            dummy={'race_key':list(race)}
            for name in ('old_D39','new_D39','control_D39'):
                try:
                    if bundles[name] is None:raise safety.MLFailure('invalid_model')
                    raw=four_forward.raw_predict(name,bundles[name],frame,dummy)
                    p=matrices.normalize(frame,np.array([raw[h] for h in frame.entry_key.str[-2:]]),plan['temperatures'][name])
                except Exception:
                    if name!='control_D39':raise
                    model_status['control_D39_missing']+=1
                    reasons['control_prediction_unavailable']=reasons.get('control_prediction_unavailable',0)+1;continue
                model_status[name+'_saved']+=1
                mapping=dict(zip(frame.entry_key,p));rank={k:i for i,k in enumerate(sorted(mapping,key=lambda k:(-mapping[k],k)),1)}
                generated=c.clock().isoformat()
                out.extend({'entry_key':k,'model':name,'model_id':plan['model_ids'][name],'raw_probability':raw[k[-2:]],
                    'probability':float(mapping[k]),'rank':rank[k],'temperature':plan['temperatures'][name],
                    'generated_at':generated,'source_observed_at':observed.isoformat(),'dry_run_only':True} for k in frame.entry_key)
            races+=1
    result={'status':'ok','kind':'declared_entry_D39_dry_run','date':str(day),'target_races':len(cards),'predicted_races':races,
        'rows_saved':len(out),'reason':'entries_not_yet_available' if not cards else None,'exclusions':reasons,
        'model_race_status_counts':dict(model_status),
        'results_read':False,'H':'not_attempted_before_T10','new_fits':0,'real_forward_record':False,'folder':str(target),
        'checked_at':c.clock().isoformat()}
    if out:csv_file(target/'predictions.dry.csv',out)
    c.write(target/'summary.json',result);return result
