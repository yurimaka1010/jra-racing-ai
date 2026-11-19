"""Display missing targets without altering the committed forecast/evaluation cohort."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta
import re
from ..ml import safety
from ..win_sim import core
from . import contracts as c

D39=('old_D39','new_D39','control_D39','next_D39','enriched_D39')


def missing(directory,day,race,start,clock):
    if directory is not None:
        folder=directory/'forward'/day/'races'/race
        if (folder/'veto.json').exists():return 'missing','forecast_vetoed'
        if (folder/'receipt.json').exists():
            receipt=c.read(folder/'receipt.json')
            if receipt.get('status')!='saved':return 'missing',receipt.get('reason') or 'forecast_not_saved'
            return 'missing','saved_record_unavailable'
        if (folder/'prediction_error.json').exists():return 'missing','prediction_error'
    if start:
        cutoff=core.instant(start)-timedelta(minutes=10)
        if clock<cutoff:return 'pending','awaiting_T10'
        if clock<=cutoff+timedelta(seconds=60):return 'pending','prediction_in_progress'
    return 'missing','forecast_not_saved'


def attach(packet,targets,*,candidate,directory=None,fallback=None,clock=None):
    if candidate not in ('next_D39','enriched_D39') or len(targets)>36:raise safety.MLFailure('invalid_input')
    clock=c.clock() if clock is None else clock;day=packet['date'];fallback=fallback or {}
    own={r['race_id']:r for r in packet.get('races',[])}
    extra={r['race_id']:r for r in fallback.get('races',[])}
    answers={r['race_id']:r for r in (packet.get('results') or {}).get('races',[])}
    other_answers={r['race_id']:r for r in (fallback.get('results') or {}).get('races',[])}
    forecasts=[];results=[];counts=Counter();reasons=Counter();answer_count=0
    for target in targets:
        card=deepcopy(target);key=card['race_id']
        if not re.fullmatch('[0-9]{16}',key) or key[:8]!=day.replace('-',''):raise safety.MLFailure('invalid_identity')
        saved=own.get(key);prior=extra.get(key)
        if prior and candidate=='enriched_D39' and 'next_D39' in prior.get('models',{}):
            card['models']['next_D39']=deepcopy(prior['models']['next_D39'])
        if saved:
            for name in D39:
                if name in saved.get('models',{}):card['models'][name]=deepcopy(saved['models'][name])
            card['horse_names']=deepcopy(saved.get('horse_names') or card.get('horse_names',{}))
            state,reason='saved',None
        else:
            state,reason=missing(directory,day,key,card.get('scheduled_start'),clock)
            card['models'][candidate]={'status':state,'reason':reason,'rows':[],
                'model_id':packet.get('activation',{}).get('model_id'),'temperature':packet.get('activation',{}).get('temperature')}
        counts[state]+=1
        if reason:reasons[reason]+=1
        card.update(candidate_status=state,candidate_reason=reason,
            display_status='formal_saved' if state=='saved' else 'formal_pending' if state=='pending' else 'formal_missing')
        forecasts.append(card)
        source=answers.get(key) or other_answers.get(key)
        if source and source.get('source_prediction_sha256')!=card.get('source_prediction_sha256'):
            source=None
        if source:
            answer=deepcopy(source);answer['models']={n:v for n,v in answer.get('models',{}).items() if n in D39}
        else:
            answer={k:deepcopy(card.get(k)) for k in ('race_id','race_key','date','venue','race_no','scheduled_start','input_cutoff','source_prediction_sha256')}
            answer.update(models={},finishes={},result_status='pending',payout_status='pending',
                reason='formal_prediction_unavailable' if state!='saved' else 'official_result_pending')
        for name in D39:
            if name not in card['models']:continue
            if name not in answer['models']:answer['models'][name]=deepcopy(card['models'][name])
        confirmed=(state=='saved' and key in answers and answers[key].get('result_status')=='confirmed'
            and answers[key]['models'].get(candidate,{}).get('status')=='saved')
        answer_count+=bool(confirmed)
        answer.update(roster=deepcopy(card.get('roster',[])),horse_names=deepcopy(card.get('horse_names',{})),
            candidate_status=state,candidate_reason=reason,candidate_result_status='confirmed' if confirmed else 'pending',
            display_status=card['display_status'] if state!='saved' else 'result_confirmed' if confirmed else 'result_pending',
            display_only=True,H_results_included=False)
        results.append(answer)
    packet['display_races']=forecasts;packet['display_result_races']=results
    packet['display_coverage']={'target_races':len(targets),'selected_forecast_races':counts['saved'],
        'missing_forecast_races':counts['missing'],'pending_forecast_races':counts['pending'],
        'selected_result_races':answer_count,'missing_reasons':dict(reasons)}
    return packet
