"""Independent live sidecar: only on-time original inputs; H outcomes never projected."""
from collections import Counter, deque
from copy import copy, deepcopy
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4
import csv
import json
import os
import time

from ..core import RACE_KEY
from ..ml import safety
from ..win_sim import core
from ..d39_generation import forward as existing, data as history_data
from . import contracts as c, features, learning

MODELS = ('old_D39', 'new_D39', 'control_D39', 'next_D39')


def atomic(path, value):
    temp = path.with_name(uuid4().hex + '.tmp')
    c.write(temp, value)
    os.replace(temp, path)


def live_frame(builder, rows):
    import pandas as pd
    clone = copy(builder); clone.pending = deque(builder.pending); clone.start_events = deque(builder.start_events)
    if hasattr(builder, 'time_pending'):
        clone.time_pending = deque(builder.time_pending)
        clone.past_times = dict(builder.past_times)
    allowed = {k for k, v in c.prior.RAW_FIELDS.items() if v[3] in ('key', 'identity') or v[3].startswith('declared')}
    normalized = []
    for row in rows:
        value = {k: row.get(k) for k in (*RACE_KEY, *allowed)}
        weight = features.legacy.num(value.get('weight_raw'))
        value['weight_kg'] = weight / 10 if weight is not None else None
        normalized.append(value)
    x, meta = clone.transform_day(normalized, prediction=True)
    if (not x or len({r.get('horse_id') for r in normalized}) != len(normalized)
            or any(not r.get('horse_id') for r in normalized)
            or any(v[k] is None for v in x for k in ('venue', 'surface', 'distance', 'age', 'weight_kg'))):
        raise safety.MLFailure('insufficient_history')
    return pd.DataFrame([dict(v, entry_key=''.join(m['key']), date=m['date']) for v, m in zip(x, meta)])


def history(day):
    builder = features.Builder(); cutoff = day - timedelta(days=7)
    if str(cutoff) > c.read(c.PARENT / 'plan.json')['latest_final_date']:
        raise safety.MLFailure('insufficient_history')
    for rows in history_data.iter_days(c.PARENT):
        source_day = features.legacy.legacy.entry_identity(rows[0])[1]
        if source_day <= cutoff:
            builder.transform_day(rows)
    builder._available(day)
    return builder


def dry_run(directory, bundle, builder, session, *, store=None, predictor=None, frame_maker=None):
    """Fresh declared-only inputs; saved separately, never counted as formal T10."""
    store = c if store is None else store
    predictor = learning.predict if predictor is None else predictor
    frame_maker = live_frame if frame_maker is None else frame_maker
    from ..d39_generation.four_sources import entries
    from ..d39_generation.report import csv_file
    day = c.clock().date()
    with session() as db:
        cards = existing.existing.live_cards(db, day)
        groups, observed = entries(db, day)
    rows = []; reasons = Counter(); race_count = 0
    for race, raw in sorted(groups.items()):
        card = cards.get(race)
        if not card or card.get('reason') or not card.get('scheduled_start'):
            reasons['schedule_missing'] += 1; continue
        try:
            start = core.instant(card['scheduled_start'])
            if start <= c.clock():
                reasons['already_started'] += 1; continue
            if len(raw) != len(card['roster']) or {r['horse_number'] for r in raw} != set(card['roster']):
                reasons['entry_roster_mismatch'] += 1; continue
            frame = frame_maker(builder, raw)
            scores = predictor(bundle, frame, raw=True)
            from ..research.model import normalize
            p = normalize(frame, scores, bundle[1]['temperature'])
            ranking = {str(frame.entry_key.iloc[i]): rank for rank, i in enumerate(
                sorted(range(len(frame)), key=lambda i: (-float(p[i]), str(frame.entry_key.iloc[i]))), 1)}
            generated = c.clock().isoformat()
            rows.extend({'entry_key': str(k), 'model_id': bundle[1]['model_id'],
                'probability': float(prob), 'raw_probability': float(raw_p), 'rank': ranking[str(k)],
                'temperature': bundle[1]['temperature'], 'generated_at': generated,
                'input_observed_at': observed.isoformat(), 'real_forward_record': False}
                for k, prob, raw_p in zip(frame.entry_key, p, scores))
            race_count += 1
        except (safety.MLFailure, ValueError):
            reasons['declared_inputs_unavailable'] += 1
    target = directory / 'dry' / uuid4().hex; target.mkdir(parents=True)
    if rows:
        csv_file(target / 'predictions.dry.csv', rows)
    result = {'status': 'ok', 'date': str(day), 'generated_at': c.clock().isoformat(),
        'races': race_count, 'runners': len(rows), 'excluded': dict(reasons),
        'model_id': bundle[1]['model_id'], 'temperature': bundle[1]['temperature'],
        'results_read': False, 'new_fits': 0, 'real_forward_record': False}
    if rows:
        result['CSV_sha256'] = c.sha(target / 'predictions.dry.csv')
    store.write(target / 'summary.json', result)
    return result


def decide(folder, output, activated, bundle, builder, *, clock=c.clock,
           store=None, predictor=None, frame_maker=None, model_name='next_D39'):
    store = c if store is None else store
    predictor = learning.predict if predictor is None else predictor
    frame_maker = live_frame if frame_maker is None else frame_maker
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'receipt.json').exists():
        return c.read(output / 'receipt.json')
    reg = c.read(folder / 'registered.json')
    if not reg.get('scheduled_start'):
        return {'status': 'pending'}
    cutoff = core.instant(reg['scheduled_start']) - timedelta(minutes=10)
    deadline = cutoff + timedelta(seconds=60)
    now = clock()
    if now < cutoff:
        return {'status': 'pending'}
    reason = None
    if core.instant(activated['activated_at']) > cutoff:
        reason = 'model_not_active_before_cutoff'
    elif now > deadline:
        reason = 'save_deadline_missed'
    elif (folder / 'veto.json').exists():
        reason = 'original_forecast_vetoed'
    elif not (folder / 'receipt.json').exists():
        return {'status': 'pending'}
    if reason:
        value = {'status': 'missing', 'reason': reason, 'at': now.isoformat()}
        store.write(output / 'receipt.json', value); return value
    receipt = c.read(folder / 'receipt.json')
    if receipt.get('status') != 'saved':
        value = {'status': 'missing', 'reason': 'original_on_time_input_unavailable'}
        store.write(output / 'receipt.json', value); return value
    if c.sha(folder / 'prediction.json') != receipt['prediction_sha256']:
        raise safety.MLFailure('artifact_invalid')
    original = c.read(folder / 'prediction.json')
    if original.get('result_accessed') is not False or core.instant(original['input_cutoff']) != cutoff:
        raise safety.MLFailure('artifact_invalid')
    for item in (original['entry_snapshot'], original['quote_T10']):
        seen = core.instant(item['first_seen_at'])
        if seen > cutoff or (cutoff - seen).total_seconds() > 300:
            raise safety.MLFailure('forward_cutoff')
    path = safety.local_path(Path(original['entry_snapshot']['path']), root=folder, exists=True)
    if c.sha(path) != original['entry_snapshot']['sha256']:
        raise safety.MLFailure('artifact_invalid')
    entries = c.read(path)['payload']; roster = reg['roster']
    frame = frame_maker(builder, entries)
    if set(frame.entry_key.str[-2:]) != set(roster) or len(frame) != len(roster):
        raise safety.MLFailure('invalid_identity')
    raw = predictor(bundle, frame, raw=True)
    from ..research.model import normalize
    normalized = normalize(frame, raw, bundle[1]['temperature'])
    probability = dict(zip(frame.entry_key.str[-2:], map(float, normalized)))
    ranks = {h: i for i, h in enumerate(sorted(probability, key=lambda h: (-probability[h], h)), 1)}
    paper = existing.paper_decision(tuple(original['race_key']), probability, original['quote_T10'], reg,
        {'start_date': frame.date.iloc[0], 'end_date': frame.date.iloc[0]})
    generated = clock()
    rows = [{'entry_key': str(row.entry_key), 'horse_id': next(r['horse_id'] for r in entries if r['horse_number'] == str(row.entry_key)[-2:]),
        'model': model_name, 'model_id': bundle[1]['model_id'], 'raw_probability': float(raw[i]),
        'probability': probability[str(row.entry_key)[-2:]], 'rank': ranks[str(row.entry_key)[-2:]],
        'temperature': bundle[1]['temperature'], 'generated_at': generated.isoformat(),
        'input_cutoff': cutoff.isoformat(), 'odds': original['quote_T10']['odds'][str(row.entry_key)[-2:]],
        'EV': probability[str(row.entry_key)[-2:]] * original['quote_T10']['odds'][str(row.entry_key)[-2:]] - 1,
        'virtual_stake_yen': 100 if paper['ai_horse'] == str(row.entry_key)[-2:] else 0,
        'distance': float(row.distance), 'surface': str(row.surface)} for i, row in frame.iterrows()]
    record = {'version': store.VERSION, 'race_key': original['race_key'], 'date': frame.date.iloc[0],
        'model_id': bundle[1]['model_id'], 'temperature': bundle[1]['temperature'],
        'generated_at': generated.isoformat(), 'input_cutoff': cutoff.isoformat(),
        'scheduled_start': reg['scheduled_start'], 'roster': roster, 'rows': rows,
        'probabilities': probability, 'paper': paper, 'source_prediction_sha256': receipt['prediction_sha256'],
        'entry_snapshot': original['entry_snapshot'], 'result_accessed': False,
        'distance': float(frame.distance.iloc[0]), 'surface': str(frame.surface.iloc[0])}
    if clock() > deadline:
        store.write(output / 'receipt.json', {'status': 'missing', 'reason': 'prediction_deadline_missed'})
        return {'status': 'missing'}
    store.write(output / 'prediction.json', record)
    from ..d39_generation.report import csv_file
    csv_file(output / 'predictions.csv', rows)
    written = clock()
    value = {'status': 'saved' if written <= deadline else 'missing', 'written_at': written.isoformat(),
        'deadline': deadline.isoformat(), 'prediction_sha256': c.sha(output / 'prediction.json'),
        'CSV_sha256': c.sha(output / 'predictions.csv')}
    store.write(output / 'receipt.json', value)
    if clock() > deadline:
        store.write(output / 'veto.json', {'reason': 'receipt_write_late'})
    return value


def saved(directory, day, *, clock=c.clock, model_name='next_D39'):
    from ..keiba_lab_live import load_forecast
    plan = c.read(c.PARENT / 'four_addendum_plan.json')
    active = c.read(directory / 'activation.json'); freeze = c.read(directory / 'candidate_freeze.json')
    if active['model_id'] != freeze['model_id']:
        raise safety.MLFailure('artifact_invalid')
    values = []
    for path in sorted((directory / 'forward' / day / 'races').glob('*/receipt.json')):
        receipt = c.read(path)
        if receipt.get('status') != 'saved' or (path.parent / 'veto.json').exists():
            continue
        original_folder = c.PARENT / 'forward-four-sealed' / day / 'races' / path.parent.name
        base = load_forecast(original_folder, plan, clock())
        record = c.read(path.parent / 'prediction.json')
        cutoff = core.instant(record['input_cutoff'])
        if (c.sha(path.parent / 'prediction.json') != receipt['prediction_sha256']
                or c.sha(path.parent / 'predictions.csv') != receipt['CSV_sha256']
                or record['result_accessed'] is not False or record['model_id'] != freeze['model_id']
                or record['source_prediction_sha256'] != base['source_prediction_sha256']
                or not core.instant(active['activated_at']) <= cutoff <= core.instant(record['generated_at'])
                    <= core.instant(receipt['written_at']) <= core.instant(receipt['deadline']) <= cutoff + timedelta(seconds=60)
                or core.instant(receipt['written_at']) > clock() or set(record['probabilities']) != set(base['roster'])
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in record['probabilities'].values())
                or abs(sum(record['probabilities'].values()) - 1) > 1e-12
                or len(record['rows']) != len(base['roster'])
                or {r['entry_key'] for r in record['rows']} != {base['race_id'] + h for h in base['roster']}):
            raise safety.MLFailure('artifact_invalid')
        p = record['probabilities']; rank = {h: i for i, h in enumerate(sorted(p, key=lambda h: (-p[h], h)), 1)}
        if any(row['probability'] != p[row['entry_key'][-2:]] or row['rank'] != rank[row['entry_key'][-2:]] for row in record['rows']):
            raise safety.MLFailure('artifact_invalid')
        base['models'][model_name] = {'status': 'saved', 'model_id': record['model_id'],
            'temperature': record['temperature'], 'predicted_at': record['generated_at'],
            'rows': [{'horse_no': h, 'probability': p[h], 'rank': rank[h]} for h in sorted(p)]}
        base['d39_tickets'][model_name] = {'horse': record['paper']['ai_horse'], 'reason': record['paper']['reason']}
        base['next_distance'] = record['distance']; base['next_surface'] = record['surface']
        base['decision_odds'] = {r['entry_key'][-2:]: r['odds'] for r in record['rows']}
        values.append(base)
    return values


def evaluate(directory, *, once=True, session=None):
    import pandas as pd
    from ..keiba_lab_live import fetch_results
    if session is None:
        session = db_session
    day = str(c.clock().date())
    forecasts = [r for r in saved(directory, day) if core.instant(r['scheduled_start']) <= c.clock()]
    # H is removed before the first official result SELECT, not merely hidden in HTML.
    restricted = [dict(r, models={n: m for n, m in r['models'].items() if n in MODELS}) for r in forecasts]
    results, sources = fetch_results(restricted, session, names=MODELS)
    rows = list(results.values()); measurements = {}; financial = {}; diagnostics = []
    contexts = {r['race_id']: r for r in forecasts}
    from ..tuner_four_models import distance_group, surface_group
    def pop_band(rank):
        return '1' if rank == 1 else '2-3' if rank <= 3 else '4-6' if rank <= 6 else '7+'
    for r in rows:
        context = contexts[r['race_id']]
        r['distance_group'] = distance_group(context['next_distance'])
        r['surface_group'] = surface_group(context['next_surface'])
        ordered = sorted(context['decision_odds'], key=lambda h: (context['decision_odds'][h], h))
        market_ranks = {h: i for i, h in enumerate(ordered, 1)}
        for name in MODELS:
            model = r['models'].get(name, {})
            if model.get('status') == 'saved':
                pick = next(v['horse_no'] for v in model['rows'] if v['rank'] == 1)
                model['top_pick_popularity_band'] = pop_band(market_ranks[pick])
        if r['result_status'] == 'confirmed':
            winner = next(h for h, rank in r['finishes'].items() if rank == 1)
            r['winner_popularity_band'] = pop_band(market_ranks[winner])
    common = [r for r in rows if r['result_status'] == 'confirmed'
              and all(r['models'].get(n, {}).get('status') == 'saved' for n in MODELS)]
    for n in MODELS:
        matrix_rows = [dict(entry_key=r['race_id'] + v['horse_no'], date=day,
            winner_label=int(v['finish_position'] == 1), probability=v['probability'])
            for r in common for v in r['models'][n]['rows']]
        if matrix_rows:
            frame = pd.DataFrame(matrix_rows)
            measurements[n] = learning.point(frame, frame.probability.to_numpy(float))
            finishes = [r['models'][n]['top_finish'] for r in common]
            measurements[n]['top_finish_distribution'] = dict(Counter(map(str, finishes)))
            measurements[n]['top2_percent'] = 100 * sum(v == 2 for v in finishes) / len(finishes)
            measurements[n]['top3_percent'] = 100 * sum(v == 3 for v in finishes) / len(finishes)
            measurements[n]['top3_or_better_percent'] = 100 * sum(type(v) is int and v <= 3 for v in finishes) / len(finishes)
            measurements[n]['winner_prediction_mean_rank'] = sum(next(v['rank'] for v in r['models'][n]['rows'] if v['finish_position'] == 1) for r in common) / len(common)
            for axis in ('distance_group', 'surface_group', 'winner_popularity_band', 'top_pick_popularity_band'):
                tags = {r['models'][n].get(axis) if axis == 'top_pick_popularity_band' else r.get(axis) for r in common}
                for tag in sorted(t for t in tags if t is not None):
                    selected = [r for r in common if (r['models'][n].get(axis) if axis == 'top_pick_popularity_band' else r.get(axis)) == tag]
                    part = pd.DataFrame([dict(entry_key=r['race_id']+v['horse_no'], date=day,
                        winner_label=int(v['finish_position']==1), probability=v['probability'])
                        for r in selected for v in r['models'][n]['rows']])
                    measured = learning.point(part, part.probability.to_numpy(float))
                    diagnostics.append({'axis': axis, 'group': tag, 'model': n,
                        **{k: v for k, v in measured.items() if k != 'calibration_bands'}})
        money = [{'ai': r['models'][n]['paper_ticket']} for r in rows
                 if all(r['models'].get(name, {}).get('status') == 'saved' for name in MODELS)
                 and r['models'].get(n, {}).get('paper_ticket')]
        financial[n] = core.financial(money, 'ai')
    result = {'version': c.VERSION, 'date': day, 'checked_at': c.clock().isoformat(), 'races': rows,
        'common_result_races': len(common), 'metrics': measurements, 'financial': financial,
        'condition_diagnostics': diagnostics,
        'financial_common_forecast_races': sum(all(r['models'].get(n, {}).get('status') == 'saved' for n in MODELS) for r in rows),
        'pending_result_races': len(rows) - len(common),
        'popularity_source': 'fixed_T10_win_odds;horse_number_tie_break;diagnostic_not_feature_or_selection',
        'H_results_included': False, 'Stage7_results_read': False,
        'source_status': sources, 'kind': 'prospective_next_D39_exploratory'}
    target = directory / 'results'; target.mkdir(exist_ok=True)
    c.write(target / (uuid4().hex + '.json'), result)
    atomic(directory / 'latest_results.json', result)
    atomic(directory / 'results_status.json', {'status': 'ok', 'checked_at': c.clock().isoformat()})
    return {'status': 'ok', 'common_result_races': len(common), 'H_results_included': False}


def db_session():
    from ..db_connection import read_only_session
    from ..win_sim_cli import connection_config
    return read_only_session(connection_config(2000), os.environ.get('PGPASSWORD'), gate_timeout_ms=1500)


def display_names(record, card, cache):
    """Names are current SE display metadata, never a new prediction input."""
    names = card.get('horse_names', {}) if card else {}
    expected = record.get('horse_ids')
    valid = {}
    for horse in record.get('roster', card.get('roster', []) if card else []):
        name = names.get(horse)
        if not isinstance(name, str) or not name.strip():
            continue
        if expected:
            observed = cache.get(horse, {})
            if observed.get('horse_id') != expected.get(horse) or observed.get('name') != name:
                continue
        valid[horse] = name
    return valid


def public_packet(viewer, day):
    from ..keiba_lab_live import day_guard, public_forecast
    day_guard(day)
    pointer = c.ROOT / 'current.json'
    if not pointer.exists():
        return {'version': c.VERSION, 'status': 'preparing', 'races': [], 'date': day, 'H_results_included': False}
    active = c.read(pointer); directory = c.folder(active['run_id'])
    values = saved(directory, day)
    cards_body = viewer.forecasts(day)
    cards = {r['race_id']: r for r in cards_body['races']}
    name_cache = getattr(viewer, 'name_cache', {})
    name_cache = name_cache.get(day, {}) if isinstance(name_cache, dict) else {}
    for record in values:
        record['horse_names'] = display_names(record, cards.get(record['race_id']), name_cache.get(record['race_id'], {}))
    result = c.read(directory / 'latest_results.json') if (directory / 'latest_results.json').exists() else None
    if result and result['date'] != day:
        result = None
    if result:
        by_race = {r['race_id']: r for r in values}
        for record in result.get('races', []):
            forecast = by_race.get(record['race_id'], {})
            record['horse_names'] = display_names(forecast, cards.get(record['race_id']), name_cache.get(record['race_id'], {}))
    reference = {'real_forward_record': False, 'races': [], 'runners': 0, 'generated_at': None}
    candidates = []
    for path in (directory / 'dry').glob('*/summary.json'):
        summary = c.read(path)
        if summary.get('date') == day and summary.get('races', 0) and summary.get('CSV_sha256'):
            candidates.append((summary['generated_at'], path, summary))
    if candidates:
        _, path, summary = max(candidates, key=lambda v: v[0])
        csv_path = path.parent / 'predictions.dry.csv'
        if c.sha(csv_path) != summary['CSV_sha256'] or core.instant(summary['generated_at']) > c.clock():
            raise safety.MLFailure('artifact_invalid')
        from collections import defaultdict
        grouped = defaultdict(list)
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if i >= 1008 or row['real_forward_record'] != 'False' or row['model_id'] != active['model_id']:
                    raise safety.MLFailure('artifact_invalid')
                grouped[row['entry_key'][:16]].append(row)
        for key, part in sorted(grouped.items()):
            card = cards.get(key)
            if not card:
                continue
            probabilities = {r['entry_key'][-2:]: float(r['probability']) for r in part}
            if len(probabilities) != len(part) or set(probabilities) != set(card['roster']) or abs(sum(probabilities.values())-1) > 1e-12:
                raise safety.MLFailure('artifact_invalid')
            ranks = {h: i for i, h in enumerate(sorted(probabilities, key=lambda h: (-probabilities[h], h)), 1)}
            if any(not 0 <= v <= 1 for v in probabilities.values()) or any(int(r['rank']) != ranks[r['entry_key'][-2:]] for r in part):
                raise safety.MLFailure('artifact_invalid')
            reference['races'].append({**{k: card[k] for k in ('race_id','venue','race_no','scheduled_start','roster')},
                'horse_names': display_names({}, card, {}),
                'name_source': 'current_SE_display_metadata_not_prediction_input',
                'models': {'next_D39': {'model_id': active['model_id'], 'predicted_at': part[0]['generated_at'],
                    'rows': [{'horse_no': h, 'probability': v, 'rank': ranks[h]} for h, v in sorted(probabilities.items())]}},
                'reason': 'startup_reference_not_formal_T10'})
        reference.update(runners=sum(len(r['roster']) for r in reference['races']),
                         generated_at=summary['generated_at'], CSV_sha256=summary['CSV_sha256'])
    return {'version': c.VERSION, 'status': 'active', 'date': day, 'activation': active,
        'races': [public_forecast(v) for v in values], 'results': result,
        'reference': reference,
        'result_poll': c.read(directory / 'results_status.json') if (directory / 'results_status.json').exists() else {'status': 'pending'},
        '_target_races': cards_body['races'], 'existing_coverage': cards_body['coverage'], 'H_results_included': False,
        'checked_at': c.clock().isoformat()}


def preview(directory):
    """A new declared-entry reference snapshot; does not touch the live recorder."""
    c.approved(directory)
    freeze = c.read(directory / 'candidate_freeze.json'); path = Path(freeze['model_path'])
    if c.sha(path / 'model.json') != freeze['model_sha256']:
        raise safety.MLFailure('artifact_invalid')
    bundle = learning.load(path); day = c.clock().date()
    if str(day) != '2026-10-25' or str(day) <= bundle[1]['latest_training_or_calibration_date']:
        raise safety.MLFailure('invalid_split')
    return dry_run(directory, bundle, history(day), db_session)


def run(directory, *, once=False):
    c.approved(directory)
    freeze = c.read(directory / 'candidate_freeze.json')
    path = Path(freeze['model_path'])
    if c.sha(path / 'model.json') != freeze['model_sha256']:
        raise safety.MLFailure('artifact_invalid')
    bundle = learning.load(path)
    day = c.clock().date()
    if str(day) != '2026-10-25' or str(day) <= bundle[1]['latest_training_or_calibration_date']:
        raise safety.MLFailure('invalid_split')
    lock = directory / 'recorder.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid())); stream.flush()
        try:
            builder = history(day)
            dry = dry_run(directory, bundle, builder, db_session)
            if not dry['races']:
                raise safety.MLFailure('insufficient_history')
            activation = {'run_id': directory.name, 'model_id': freeze['model_id'],
                'activated_at': c.clock().isoformat(), 'date': str(day), 'model_sha256': freeze['model_sha256'],
                'temperature': bundle[1]['temperature'], 'feature_count': len(bundle[1]['features'])}
            if (directory / 'activation.json').exists():
                previous = c.read(directory / 'activation.json')
                if previous['model_id'] != activation['model_id'] or previous['date'] != activation['date']:
                    raise safety.MLFailure('artifact_invalid')
                activation = previous
            else:
                c.write(directory / 'activation.json', activation)
            atomic(c.ROOT / 'current.json', activation)
            last_result = 0; last_counts = None
            while c.clock().date() == day and c.clock().hour < 18:
                counts = Counter()
                root = c.PARENT / 'forward-four-sealed' / str(day) / 'races'
                for folder in sorted(root.iterdir()) if root.exists() else ():
                    output = directory / 'forward' / str(day) / 'races' / folder.name
                    try:
                        value = decide(folder, output, activation, bundle, builder)
                        counts[value['status']] += 1
                    except Exception as exc:
                        counts['failed'] += 1
                        if not (output / 'prediction_error.json').exists():
                            code = exc.code if isinstance(exc, safety.MLFailure) else 'internal_error'
                            c.write(output / 'prediction_error.json', {'reason': code, 'at': c.clock().isoformat()})
                if time.monotonic() - last_result > 30:
                    try:
                        evaluate(directory)
                    except Exception:
                        log = directory / 'logs'; log.mkdir(exist_ok=True)
                        c.write(log / (uuid4().hex + '.json'), {'reason': 'results_pending', 'at': c.clock().isoformat()})
                        atomic(directory / 'results_status.json', {'status': 'pending',
                            'reason': 'result_source_unavailable', 'attempted_at': c.clock().isoformat()})
                    last_result = time.monotonic()
                if counts != last_counts:
                    print(json.dumps({'status': 'running', 'counts': dict(counts), 'at': c.clock().isoformat()}), flush=True)
                    last_counts = counts
                if once:
                    return {'status': 'ok', 'counts': dict(counts)}
                time.sleep(2)
            return {'status': 'ok', 'reason': 'daily_window_finished'}
        finally:
            stream.close()
            lock.unlink()
