"""Saved Oct24/25 entry/price replay only; target outcomes are never opened."""
from collections import Counter
from datetime import date,timedelta
from pathlib import Path
from uuid import uuid4
import json

from ..d39_generation import forward as entry_api
from ..stage7 import inputs
from ..win_sim import core
from ..ml import safety
from . import contracts as c,protocol,history,recorder


def verify_encoding(folder,bundles):
    import pandas as pd
    import numpy as np
    from ..research.model import normalize
    value=c.read(folder/'prediction.json')
    if value['replay_only'] is not True:raise safety.MLFailure('invalid_input')
    maximum=0.;checked=0
    for name,m in value['models'].items():
        if m['status']!='saved':continue
        path=folder/m['input_matrix_file']
        if c.sha(path)!=m['input_matrix_sha256']:raise safety.MLFailure('artifact_invalid')
        booster,meta=bundles[name]
        frame=pd.read_csv(path,dtype={k:'string' for k in meta['categories']},keep_default_na=False,na_values=[''],float_precision='round_trip')
        for field,categories in meta['categories'].items():frame[field]=pd.Categorical(frame[field],categories=categories)
        raw=booster.predict(frame,num_threads=4)
        horses=sorted(m['probability']);keys=pd.DataFrame({'entry_key':[''.join(value['race_key'])+h for h in horses]})
        p=normalize(keys,raw,meta['temperature'])
        delta=max(max(abs(float(q)-m['raw_probability'][h]) for q,h in zip(raw,horses)),
                  max(abs(float(q)-m['probability'][h]) for q,h in zip(p,horses)))
        if delta>1e-12:raise safety.MLFailure('artifact_invalid')
        maximum=max(maximum,delta);checked+=1
    return {'model_matrices_checked':checked,'maximum_probability_difference':maximum,'results_read':False}


def run(directory):
    freeze,p=protocol.load(directory);bundles,errors=recorder.load_models(freeze)
    if errors:raise safety.MLFailure('invalid_model')
    output=c.mkdir(directory/'replay'/uuid4().hex);summary={};matrices=0;maxdiff=0
    for day in (date(2026,10,24),date(2026,10,25)):
        context=history.build(directory,day,replay=True)
        source=c.RAW_PARENT/'forward-four-sealed'/str(day)/'races';counts=Counter();reasons=Counter();rosters=0
        for source_folder in sorted(source.iterdir()):
            if not source_folder.is_dir() or not (source_folder/'registered.json').exists():continue
            reg=c.read(source_folder/'registered.json')
            if not reg.get('scheduled_start') or reg.get('reason'):
                counts['source_unusable']+=1;reasons[reg.get('reason') or 'no_schedule']+=1;continue
            cutoff=core.instant(reg['scheduled_start'])-timedelta(minutes=10)
            snapshot,reason=entry_api.on_time_entries(source_folder,reg['roster'],cutoff)
            if reason:
                counts['source_entry_missing']+=1;reasons[reason]+=1;continue
            quote,qreason=inputs.select(tuple(reg['race_key']),recorder.quotes(source_folder),reg['roster'],cutoff,300,require_seen=True)
            target=c.mkdir(output/str(day)/source_folder.name)
            record=recorder.calculate(target,dict(reg,input_cutoff=cutoff.isoformat()),snapshot,quote,bundles,context,freeze,
                calculated_at=c.clock(),quote_reason=qreason,replay=True)
            record.update(interpretation='post_event_results_blocked_replay;not_genuine_pre_race_forecast',
                source_snapshot_sha256=snapshot['sha256'],master_and_model_creation_asof='models_and_static_lineage_frozen_after_replay_dates')
            receipt=recorder.write_record(target,record,replay=True)
            verified=verify_encoding(target,bundles)
            matrices+=verified['model_matrices_checked'];maxdiff=max(maxdiff,verified['maximum_probability_difference'])
            c.write(target/'verification.json',verified)
            if record['P1_prediction_side_complete']:
                counts['all3_complete']+=1;rosters+=len(reg['roster'])
            else:counts['models_incomplete']+=1
            counts['fresh_odds_linked' if quote else 'odds_missing']+=1
        summary[str(day)]={'counts':dict(counts),'unique_runners_with_all3':rosters,'missing_reasons':dict(reasons),
            'history_cutoff':context['info']['cutoff'],'source_target_results_read':False}
    result={'status':'replay_verified','artifact_id':output.name,'days':summary,'model_matrices_checked':matrices,
        'maximum_probability_difference':maxdiff,'formal_forward_records_created':0,'new_fits':0,
        'results_or_payouts_read':False,'target_model_creation_was_after_replay_dates':True}
    c.write(output/'summary.json',result);c.write(directory/'replay_summary.json',result)
    return result
