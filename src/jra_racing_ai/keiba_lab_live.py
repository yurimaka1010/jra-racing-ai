"""Loopback-only saved-forecast viewer with a separate four-model watch view.

This is an exploratory viewing sidecar, not the 21-week evaluator or a recorder.
No inference, fitting, edits to forecasts, or Stage 7 reads are performed here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
import argparse
import csv
import hashlib
import json
import math
import os
import re
import time

from .core import RACE_KEY
from .ml import safety
from .win_sim import core, local, settle_saved
from .keiba_lab_export import COLUMNS, VENUES, csv_text

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
PARENT = BASE / 'd39-generation/evaluations/86ed126ffc8f4e2e85c8740eacf941f0'
ROOT = BASE / 'keiba-lab-live'
WEB = Path(__file__).resolve().parents[2] / 'web/keiba-lab-live'
# Same date window as the existing signed four-model plan. This viewer does not
# extend that research, open the earlier holdout, or read H outcomes.
_VIEW_START, _VIEW_END = date(2026, 10, 24), date(2027, 3, 14)
DAYS = tuple((_VIEW_START + timedelta(days=n)).isoformat()
             for n in range((_VIEW_END - _VIEW_START).days + 1))
RESULT_DAYS = ('2026-10-24', '2026-10-25')
WATCH_START = '2026-10-31'
D39 = ('old_D39', 'new_D39', 'control_D39')
MODELS = (*D39, 'H')
LABELS = dict(zip(MODELS, ('旧D39', '新D39', 'D39対照型', '主力H')))
VERSION = 'keiba-lab-live.20261024.1'


def now():
    return datetime.now(core.JST)


def day_guard(day):
    if day not in DAYS:
        raise safety.MLFailure('invalid_date')
    return day


def result_day_guard(day):
    if day not in RESULT_DAYS:
        raise safety.MLFailure('invalid_date')
    return day


def watch_day_guard(day):
    day_guard(day)
    if day < WATCH_START or day > now().date().isoformat():
        raise safety.MLFailure('invalid_date')
    return day


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return safety.read_json(path, root=BASE)


def sha(path):
    return safety.digest(path, root=BASE)


def verify(path, expected):
    if sha(path) != expected:
        raise safety.MLFailure('artifact_invalid')


def atomic_snapshot(directory, name, value):
    """Only viewer-owned state is replaced; history and original artifacts are exclusive."""
    path = safety.local_path(directory / name, root=ROOT)
    temp = path.with_name(uuid4().hex + '.tmp')
    safety.write_json(temp, value, root=ROOT)
    os.replace(temp, path)


def side_record(folder, base, receipt, cutoff, deadline, clock):
    """A failed optional sidecar cannot veto the authoritative three-model files."""
    try:
        if not (folder / 'four_receipt.json').exists(): return None, 'fourth_sidecar_pending'
        sr = read(folder / 'four_receipt.json')
        if sr.get('status') != 'saved' or (folder / 'four_veto.json').exists():
            return None, sr.get('reason') or 'control_prediction_missing'
        verify(folder / 'four_prediction.json', sr['prediction_sha256'])
        verify(folder / 'four_predictions.csv', sr['CSV_sha256'])
        side = read(folder / 'four_prediction.json')
        if (side.get('synthetic_verification') or side.get('result_accessed') is not False
                or not cutoff <= core.instant(side['generated_at']) <= core.instant(sr['written_at']) <= deadline
                or core.instant(sr['written_at']) > clock or side['input_cutoff'] != base['input_cutoff']
                or side['base_three_prediction_sha256'] != receipt['prediction_sha256']
                or any(side['probabilities'][n] != base['probabilities'][n] for n in ('old_D39', 'new_D39', 'H'))):
            return None, 'fourth_sidecar_unverified'
        return side, None
    except Exception:
        return None, 'fourth_sidecar_unverified'


def load_forecast(folder, plan, clock):
    """Verify sealed receipts, then project only allowlisted prediction fields."""
    reg = read(folder / 'registered.json')
    race = core.key(reg['race_key'])
    start = core.instant(reg['scheduled_start']) if reg.get('scheduled_start') else None
    day_guard(f'{race[0]}-{race[1][:2]}-{race[1][2:]}')
    if folder.name != ''.join(race):
        raise safety.MLFailure('invalid_identity')
    result = {'race_id': ''.join(race), 'race_key': list(race),
        'date': f'{race[0]}-{race[1][:2]}-{race[1][2:]}', 'venue': VENUES.get(race[2], ''),
        'race_no': int(race[5]), 'scheduled_start': reg.get('scheduled_start'),
        'roster': reg.get('roster', []), 'models': {}, 'reason': 'awaiting_T10',
        'result_access': 'D39_only_separate_endpoint', 'valid_for_results': False}
    receipt_path = folder / 'receipt.json'
    if not receipt_path.exists():
        if start and clock > start - timedelta(minutes=9):
            result['reason'] = 'forecast_not_saved'
        return result
    receipt = read(receipt_path)
    if receipt.get('status') != 'saved':
        result['reason'] = receipt.get('reason', 'forecast_missing')
        return result
    verify(folder / 'prediction.json', receipt['prediction_sha256'])
    verify(folder / 'predictions.csv', receipt['CSV_sha256'])
    base = read(folder / 'prediction.json')
    cutoff, written, deadline = (core.instant(base['input_cutoff']),
        core.instant(receipt['written_at']), core.instant(receipt['deadline']))
    if (base.get('synthetic_verification') or base.get('result_accessed') is not False
            or not start or cutoff != start - timedelta(minutes=10)
            or not cutoff <= core.instant(base['generated_at']) <= written <= deadline
            or written > clock or deadline > cutoff + timedelta(seconds=60)
            or core.key(base['race_key']) != race or base['registered']['roster'] != reg['roster']):
        raise safety.MLFailure('artifact_invalid')
    entry_path = safety.local_path(Path(base['entry_snapshot']['path']), root=folder, exists=True)
    verify(entry_path, base['entry_snapshot']['sha256'])
    entries = read(entry_path)['payload']
    ids = {r['horse_number']: r['horse_id'] for r in entries}
    if set(ids) != set(reg['roster']) or len(entries) != len(ids):
        raise safety.MLFailure('invalid_identity')
    for stamp in (base['entry_snapshot']['first_seen_at'], base['quote_T10']['first_seen_at']):
        if core.instant(stamp) > cutoff:
            raise safety.MLFailure('artifact_invalid')
    # The same fixed clock and receipt checks are used for the optional fourth sidecar.
    side, side_reason = side_record(folder, base, receipt, cutoff, deadline, clock)
    with safety.local_path(folder / 'predictions.csv', exists=True, root=BASE).open(encoding='utf-8-sig', newline='') as stream:
        base_rows = list(csv.DictReader(stream))
    side_rows = []
    if side:
        with safety.local_path(folder / 'four_predictions.csv', exists=True, root=BASE).open(encoding='utf-8-sig', newline='') as stream:
            side_rows = list(csv.DictReader(stream))
    if len(base_rows) > 84 or len(side_rows) > 112: raise safety.MLFailure('input_limit')
    result.update(input_cutoff=base['input_cutoff'], predicted_at=base['generated_at'],
        odds_published_at=base['quote_T10']['published_at'],
        odds_first_seen_at=base['quote_T10']['first_seen_at'],
        odds_source=base['quote_T10'].get('source'),
        source_prediction_sha256=receipt['prediction_sha256'],
        source_four_sha256=sha(folder / 'four_prediction.json') if side else None,
        horse_ids=ids, reason=None, valid_for_results=not (folder / 'veto.json').exists())
    for name in MODELS:
        values = (side['probabilities'].get(name) if side else base['probabilities'].get(name))
        if name == 'control_D39' and side is None:
            values = None
        if values is None:
            result['models'][name] = {'status': 'missing', 'reason': side_reason, 'model_id': plan['model_ids'][name]}
            continue
        if (set(values) != set(ids) or any(type(v) not in (float, int) or not math.isfinite(v)
                or not 0 <= v <= 1 for v in values.values()) or abs(sum(values.values()) - 1) > 1e-12):
            raise safety.MLFailure('invalid_model')
        model_id = (side['model_ids'] if side else base['model_ids'])[name]
        if model_id != plan['model_ids'][name]:
            raise safety.MLFailure('invalid_model')
        rank = {h: i for i, h in enumerate(sorted(values, key=lambda h: (-values[h], h)), 1)}
        stored = [r for r in (side_rows if name == 'control_D39' else base_rows)
            if r['model'] == name and r.get('model_status', 'saved') == 'saved']
        if (len(stored) != len(ids) or {r['entry_key'] for r in stored} != {''.join(race) + h for h in ids}
                or any(float(r['probability']) != values[r['entry_key'][-2:]]
                       or int(r['rank']) != rank[r['entry_key'][-2:]] or r['model_id'] != model_id for r in stored)):
            raise safety.MLFailure('artifact_invalid')
        model = {'status': 'saved', 'model_id': model_id, 'temperature': plan['temperatures'][name],
            'predicted_at': side['generated_at'] if name == 'control_D39' else base['generated_at'],
            'rows': [{'horse_no': r['entry_key'][-2:], 'probability': float(r['probability']), 'rank': int(r['rank'])}
                for r in sorted(stored, key=lambda r: r['entry_key'])]}
        result['models'][name] = model
    if not result['valid_for_results']:
        result['reason'] = 'schedule_or_roster_changed_after_registration'
    # Only the D39 branch receives saved ticket decisions. H ledgers are not projected.
    ledgers = side['ledgers'] if side else base['ledgers']
    result['d39_tickets'] = {n: {'horse': ledgers[n]['ai_horse'], 'reason': ledgers[n]['reason']}
        for n in D39 if ledgers.get(n) is not None and result['models'][n]['status'] == 'saved'}
    return result


def result_projection(forecast, rows, outcome, *, names=D39):
    """No H probabilities, rows, labels, ledgers, scores, or results in this output."""
    if not names or any(n not in (*D39, 'next_D39', 'enriched_D39') for n in names) or len(set(names)) != len(names):
        raise safety.MLFailure('invalid_input')
    allowed = {k: forecast.get(k) for k in ('race_id', 'race_key', 'date', 'venue', 'race_no',
        'scheduled_start', 'input_cutoff', 'predicted_at', 'source_prediction_sha256', 'horse_names')}
    allowed.update(models={}, reason=None, result_status='pending', finishes={})
    roster = forecast['roster']
    if not forecast['valid_for_results']:
        allowed['reason'] = 'forecast_vetoed'; return allowed
    # Count already frozen D39 purchases as pending even before answers arrive.
    # No finish label or payout is inferred from these ticket records.
    for name in names:
        model = forecast['models'].get(name, {})
        if model.get('status') != 'saved':
            allowed['models'][name] = {'status': 'missing'}; continue
        ticket = forecast['d39_tickets'].get(name)
        allowed['models'][name] = {'status': 'saved', 'model_id': model['model_id'],
            'predicted_at': model['predicted_at'], 'top_finish': None, 'top_hit': None,
            'rows': [dict(r, finish_position=None, win_payout_100=None) for r in model['rows']],
            'paper_ticket': core.settle(ticket['horse'], None, tuple(roster)) if ticket else None,
            'paper_horse': ticket['horse'] if ticket else None}
    if len(rows) != len(roster) or {r.get('horse') for r in rows} != set(roster):
        allowed['reason'] = 'result_roster_missing_or_duplicate'; return allowed
    if any(r.get('horse_id') != forecast['horse_ids'][r['horse']] or r.get('status') not in ('5', '6', '7')
            or r.get('anomaly') not in ('0', '4') or not re.fullmatch('[0-9]{2}', r.get('rank') or '') for r in rows):
        allowed['reason'] = 'result_status_identity_or_cancellation_unconfirmed'; return allowed
    ranks = {r['horse']: int(r['rank']) if r['rank'] != '00' else 'DNF' for r in rows}
    if any((v == 'DNF' and next(r for r in rows if r['horse'] == h)['anomaly'] != '4')
            or (type(v) is int and not 1 <= v <= len(roster)) for h, v in ranks.items()):
        allowed['reason'] = 'result_rank_unconfirmed'; return allowed
    winners = {h for h, v in ranks.items() if v == 1}
    if len(winners) != 1:
        allowed['reason'] = 'dead_heat_or_unconfirmed_winner'; return allowed
    normal = core.outcome_reason(outcome, tuple(roster)) is None
    if normal and set(outcome.payouts_yen_per_100) != winners:
        allowed['reason'] = 'hr_se_winner_mismatch'; return allowed
    # Full-order flash categories are shown as confirmed only with the payout-confirmed HR.
    if any(r['status'] != '7' for r in rows) and not normal:
        allowed['reason'] = 'full_order_flash_waiting_for_confirmed_hr'; return allowed
    allowed.update(result_status='confirmed', finishes=ranks,
        result_basis='monday_final_SE7' if all(r['status'] == '7' for r in rows) else 'full_order_flash_SE5_6_with_confirmed_HR',
        payout_status='confirmed' if normal else 'pending',
        payout_reason=None if normal else (outcome.reason if outcome else 'result_unavailable'))
    for name in names:
        model = forecast['models'].get(name, {})
        if model.get('status') != 'saved':
            allowed['models'][name] = {'status': 'missing'}; continue
        values = [{**r, 'finish_position': ranks[r['horse_no']],
            'win_payout_100': outcome.payouts_yen_per_100.get(r['horse_no'], 0) if normal else None}
            for r in model['rows']]
        pick = next(r for r in values if r['rank'] == 1)
        ticket = forecast['d39_tickets'].get(name)
        allowed['models'][name] = {'status': 'saved', 'model_id': model['model_id'],
            'predicted_at': model['predicted_at'], 'rows': values,
            'top_finish': pick['finish_position'], 'top_hit': pick['finish_position'] == 1,
            'paper_ticket': core.settle(ticket['horse'], outcome, tuple(roster)) if ticket else None,
            'paper_horse': ticket['horse'] if ticket else None}
    return allowed


def fetch_results(requests, session, *, names=D39):
    """Only exact committed weekend keys; reuse existing RO adapter and HR parser."""
    if not names or any(n not in (*D39, 'next_D39', 'enriched_D39') for n in names) or len(set(names)) != len(names):
        raise safety.MLFailure('invalid_input')
    if not requests:
        return {}, []
    races = tuple(core.key(r['race_key']) for r in requests)
    if len(races) > 36 or any(result_day_guard(f'{r[0]}-{r[1][:2]}-{r[1][2:]}') not in RESULT_DAYS for r in races):
        raise safety.MLFailure('input_limit')
    grouped, decoded, sources = fetch_official(races, session)
    result = {''.join(race): result_projection(request, grouped.get(race, []), decoded.get(race), names=names)
        for race, request in zip(races, requests)}
    return result, sources


def fetch_official(races, session):
    """Shared bounded official SE/HR reader. No model, study or ledger input."""
    from .d39_generation.outcomes import reviewed_profile
    if not races or len(races) > 36: raise safety.MLFailure('input_limit')
    for race in races:
        core.key(race); day_guard(f'{race[0]}-{race[1][:2]}-{race[1][2:]}')
    with session() as db:
        budget = local.Budget(30)
        columns = local.table_info(db, 'jvd_se', budget)
        fields = {'horse': ('umaban', 2), 'horse_id': ('ketto_toroku_bango', 10),
            'rank': ('kakutei_chakujun', 2), 'status': ('data_kubun', 1), 'anomaly': ('ijo_kubun_code', 1)}
        if columns is None or any(columns.get(col, {}).get('type_is_builtin') is not True
                or columns[col]['type_name'] not in local.TEXT_TYPES for col, _ in fields.values()):
            raise safety.MLFailure('invalid_schema')
        rows = local._rows(db, budget, 'jvd_se', fields, columns, races, 1000, join_targets=True)
        try:
            profile = reviewed_profile(races)
            # HR1 = payout-confirmed flash; HR2 = Monday final. Original profile is unchanged.
            profile['allowed_final_categories'] = ['1', '2']
            hr, source, status = settle_saved.fetch_hr(db, races, max_rows=100, max_seconds=15, reviewed_source=profile)
            decoded = {o.race: o for o in settle_saved.decode_existing_hr(hr, source)} if hr else {}
        except Exception:
            decoded = {}; status = {'table': 'jvd_hr', 'status': 'pending', 'reason': 'payout_source_unavailable'}
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row['race_key'])].append(row)
    return dict(grouped), decoded, [status]


def fetch_watch_results(requests, session):
    """Only registered saved-record dates in the explicitly authorized watch scope."""
    if not requests: return {}, []
    races = tuple(core.key(r['race_key']) for r in requests)
    if len(races) > 36 or len(set(races)) != len(races): raise safety.MLFailure('input_limit')
    for race in races: watch_day_guard(f'{race[0]}-{race[1][:2]}-{race[1][2:]}')
    rows, outcomes, sources = fetch_official(races, session)
    return {''.join(race): (rows.get(race, []), outcomes.get(race)) for race in races}, sources


def public_forecast(forecast):
    return {k: v for k, v in forecast.items() if k not in ('horse_ids', 'd39_tickets', 'valid_for_results')}


def fetch_names(forecasts, session):
    """Display metadata only: no finish, anomaly, odds, or payout column is selected."""
    targets = [r for r in forecasts if r.get('race_key')]
    races = tuple(core.key(r['race_key']) for r in targets)
    if not races or len(races) > 36: return {}
    for race in races: day_guard(f'{race[0]}-{race[1][:2]}-{race[1][2:]}')
    with session() as db:
        budget = local.Budget(15); columns = local.table_info(db, 'jvd_se', budget)
        fields = {'horse': ('umaban', 2), 'horse_id': ('ketto_toroku_bango', 10), 'name': ('bamei', 256)}
        if columns is None or any(columns.get(col, {}).get('type_is_builtin') is not True
                or columns[col]['type_name'] not in local.TEXT_TYPES for col, _ in fields.values()): return {}
        rows = local._rows(db, budget, 'jvd_se', fields, columns, races, 1000, join_targets=True)
    grouped = defaultdict(list)
    for row in rows: grouped[(''.join(row['race_key']), row['horse'])].append(row)
    names = {}
    for forecast in targets:
        found = {}
        for horse in forecast['roster']:
            matches = grouped[(forecast['race_id'], horse)]
            expected = forecast.get('horse_ids', {}).get(horse)
            if len(matches) != 1 or (expected is not None and matches[0]['horse_id'] != expected): continue
            value = matches[0].get('name')
            if isinstance(value, str) and 0 < len(value.strip()) <= 120 and not re.search('[\x00-\x1f]', value):
                found[horse] = {'name': value.strip(), 'horse_id': matches[0]['horse_id']}
        names[forecast['race_id']] = found
    return names


class Viewer:
    def __init__(self, *, session=None, parent=PARENT, directory=None, clock=now, monotonic=time.monotonic):
        self.parent, self.session, self.clock, self.monotonic = parent, session, clock, monotonic
        self.plan = read(parent / 'four_addendum_plan.json')
        self.directory = directory or ROOT / uuid4().hex
        safety.local_path(self.directory, root=ROOT).mkdir(parents=True, exist_ok=False)
        safety.write_json(self.directory / 'authorization.json', {'authorized_dates': RESULT_DAYS,
            'scope': 'legacy_D39_results_endpoints_only', 'watch_authorization': 'watch-authorization.json',
            'prediction_authorized_dates': DAYS, 'created_at': clock().isoformat(),
            'D39_exploratory_result_viewing_authorized_by_user': True, 'H_prediction_only': True,
            'Stage7_results': False, 'formal_21_week_plan_unchanged': True,
            'D39_interim_views_not_blinded_final_evidence': True, 'new_fits': 0}, root=ROOT)
        safety.write_json(self.directory / 'watch-authorization.json', {
            'created_at': clock().isoformat(), 'scope': 'local_four_model_entertainment_only',
            'start_date': WATCH_START, 'end_date': DAYS[-1], 'models': list(MODELS),
            'source_predictions': str(parent / 'forward-four-sealed'),
            'H_prediction_result_join_explicitly_authorized': True,
            'H_disclosed_subset_is_not_unviewed': True, 'Stage7_study_access': False,
            'pedigree_21_week_study_access': False, 'formal_evaluation': False,
            'rule': 'watch_R1_saved_top1_win_100_yen_no_EV_filter',
            'hit_rate_denominator': 'eligible_confirmed_nonrefund_picks_per_model',
            'financial_denominator': 'settled_or_officially_refunded_picks_only', 'new_fits': 0}, root=ROOT)
        self.lock = RLock(); self.cache = {}; self.frozen = {}; self.results = {}; self.last_good = {}
        self.revision = {}; self.name_cache = {}; self.name_checked = {}; self.source_frozen = {}; self.reference_hashes = {}; self.previous_answers = {}
        self.watch_previous = {}; self.watch_revisions = {}; self.watch_disclosures = {}

    def forecasts(self, day):
        day_guard(day)
        with self.lock:
            key = ('predictions', day)
            if key in self.cache and self.monotonic() - self.cache[key][0] < 10:
                return self.cache[key][1]
            values = []; reasons = Counter()
            source = self.parent / 'forward-four-sealed' / day / 'races'
            for folder in sorted(source.iterdir()) if source.exists() else ():
                if not folder.is_dir() or not re.fullmatch('[0-9]{16}', folder.name):
                    continue
                try:
                    record = load_forecast(folder, self.plan, self.clock())
                    source_sha = record.get('source_prediction_sha256')
                    if source_sha:
                        if record['race_id'] in self.source_frozen and self.source_frozen[record['race_id']] != source_sha:
                            raise safety.MLFailure('ledger_tampered')
                        self.source_frozen[record['race_id']] = source_sha
                    for name, model in record['models'].items():
                        if model['status'] != 'saved': continue
                        identity = (record['race_id'], name)
                        fingerprint = digest(model)
                        if identity in self.frozen and self.frozen[identity] != fingerprint:
                            raise safety.MLFailure('ledger_tampered')
                        if identity not in self.frozen:
                            freeze_dir = self.directory / 'fixed-predictions'; freeze_dir.mkdir(exist_ok=True)
                            safety.write_json(freeze_dir / (record['race_id'] + '_' + name + '.json'),
                                {'prediction': model, 'sha256': fingerprint, 'input_cutoff': record['input_cutoff'],
                                 'source_sha256': record['source_prediction_sha256']}, root=ROOT)
                            self.frozen[identity] = fingerprint
                    values.append(record)
                    self.last_good[record['race_id']] = deepcopy(record)
                except Exception as exc:
                    code = exc.code if isinstance(exc, (safety.MLFailure, core.SimulationError)) else 'artifact_invalid'
                    reasons[code] += 1
                    previous = deepcopy(self.last_good.get(folder.name, {}))
                    values.append(dict(previous or {'race_id': folder.name, 'date': day, 'venue': '',
                        'race_no': int(folder.name[-2:]), 'models': {}, 'roster': [], 'reason': code,
                        'valid_for_results': False, 'scheduled_start': None}, reason=code, valid_for_results=False))
            if self.session and self.monotonic() - self.name_checked.get(day, -1000) >= 300:
                try: self.name_cache[day] = fetch_names(values, self.session)
                except Exception: pass  # Names never veto a saved forecast.
                self.name_checked[day] = self.monotonic() - (285 if not self.name_cache.get(day) else 0)
            for record in values:
                record['horse_names'] = {h: v['name'] for h, v in self.name_cache.get(day, {}).get(record['race_id'], {}).items()
                    if not record.get('horse_ids') or v['horse_id'] == record['horse_ids'].get(h)}
            saved = sum(any(m.get('status') == 'saved' for m in r['models'].values()) for r in values)
            record_root = self.parent / 'forward-four-sealed'
            available_dates = sorted((p.name for p in record_root.iterdir()
                                      if p.is_dir() and p.name in DAYS), reverse=True) if record_root.exists() else []
            body = {'version': VERSION, 'date': day, 'checked_at': self.clock().isoformat(),
                'available_dates': available_dates,
                'models': LABELS, 'races': [public_forecast(r) for r in values],
                'coverage': {'registered_races': len(values), 'prediction_races': saved,
                    'registered_runners': sum(len(r['roster']) for r in values),
                    'four_model_races': sum(all(r['models'].get(n, {}).get('status') == 'saved' for n in MODELS) for r in values),
                    'missing_reasons': dict(reasons)}, 'H_results_included': False,
                'formal_evaluation': False, 'automatic_purchase': False}
            self.results[day] = values
            self.cache[key] = (self.monotonic(), body)
            return body

    def references(self, day):
        """Display saved declared-entry inference; never a T10 receipt or scored dataset."""
        day_guard(day)
        with self.lock:
            self.forecasts(day)
            cards = {r['race_id']: r for r in self.results.get(day, [])}
            candidates = []
            root = self.parent / 'dry-four'
            for path in sorted(root.glob('*/summary.json'))[-100:]:
                doc = read(path)
                if (doc.get('date') == day and doc.get('kind') == 'declared_entry_D39_dry_run'
                        and doc.get('results_read') is False and doc.get('real_forward_record') is False
                        and doc.get('new_fits') == 0 and (path.parent / 'predictions.dry.csv').exists()
                        and core.instant(doc['checked_at']) <= self.clock()):
                    candidates.append((doc['checked_at'], path.parent, doc))
            output = []; source = None
            if candidates:
                _, folder, doc = max(candidates, key=lambda x: x[0])
                path = safety.local_path(folder / 'predictions.dry.csv', exists=True, root=BASE)
                source = {'id': folder.name, 'sha256': sha(path), 'generated_at': doc['checked_at']}
                if folder.name in self.reference_hashes and self.reference_hashes[folder.name] != source['sha256']:
                    raise safety.MLFailure('ledger_tampered')
                grouped = defaultdict(list)
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    for i, row in enumerate(csv.DictReader(stream)):
                        if i >= 3024: raise safety.MLFailure('input_limit')
                        if (not re.fullmatch('[0-9]{18}', row.get('entry_key', ''))
                                or row['entry_key'][:8] != day.replace('-', '') or row.get('dry_run_only') != 'True'
                                or row.get('model') not in D39): raise safety.MLFailure('artifact_invalid')
                        grouped[row['entry_key'][:16]].append(row)
                for race_id, rows in sorted(grouped.items()):
                    card = cards.get(race_id)
                    if card is None or card.get('reason') not in (None, 'awaiting_T10', 'forecast_not_saved'): continue
                    value = {k: card.get(k) for k in ('race_id', 'race_key', 'date', 'venue', 'race_no', 'scheduled_start', 'roster', 'horse_names')}
                    value.update(models={}, reason=None, forecast_kind='current_reference', source_prediction_sha256=source['sha256'],
                        predicted_at=max(r['generated_at'] for r in rows), source_observed_at=max(r['source_observed_at'] for r in rows))
                    for name in D39:
                        part = [r for r in rows if r['model'] == name]
                        if not part: value['models'][name] = {'status': 'missing', 'reason': 'reference_model_missing'}; continue
                        if (len(part) != len(card['roster']) or {r['entry_key'][-2:] for r in part} != set(card['roster'])
                                or any(r['model_id'] != self.plan['model_ids'][name]
                                or float(r['temperature']) != self.plan['temperatures'][name]
                                or not core.instant(r['source_observed_at']) <= core.instant(r['generated_at'])
                                    < core.instant(card['scheduled_start'])
                                or core.instant(r['generated_at']) > self.clock() for r in part)):
                            raise safety.MLFailure('artifact_invalid')
                        probabilities = {r['entry_key'][-2:]: float(r['probability']) for r in part}
                        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-12:
                            raise safety.MLFailure('invalid_model')
                        ranking = {h: i for i, h in enumerate(sorted(probabilities, key=lambda h: (-probabilities[h], h)), 1)}
                        if any(int(r['rank']) != ranking[r['entry_key'][-2:]] for r in part): raise safety.MLFailure('artifact_invalid')
                        value['models'][name] = {'status': 'saved', 'model_id': self.plan['model_ids'][name],
                            'temperature': self.plan['temperatures'][name], 'predicted_at': part[0]['generated_at'],
                            'rows': [{'horse_no': r['entry_key'][-2:], 'probability': float(r['probability']), 'rank': int(r['rank'])}
                                for r in sorted(part, key=lambda r: r['entry_key'])]}
                    output.append(value)
                if folder.name not in self.reference_hashes:
                    saved = self.directory / 'reference-predictions'; saved.mkdir(exist_ok=True)
                    safety.write_json(saved / (folder.name + '.json'), {'source': source, 'races': output,
                        'real_forward_record': False, 'results_accessed': False}, root=ROOT)
                    self.reference_hashes[folder.name] = source['sha256']
            return {'version': VERSION, 'date': day, 'checked_at': self.clock().isoformat(), 'source': source,
                'forecast_kind': 'current_reference', 'models': {n: LABELS[n] for n in D39}, 'races': output,
                'coverage': {'registered_races': len(cards), 'prediction_races': len(output),
                    'registered_runners': sum(len(r['roster']) for r in output),
                    'three_model_races': sum(all(r['models'].get(n, {}).get('status') == 'saved' for n in D39) for r in output)},
                'H_results_included': False, 'formal_evaluation': False, 'real_forward_record': False,
                'results_accessed': False, 'new_fits': 0}

    def d39_results(self, day):
        result_day_guard(day)
        with self.lock:
            self.forecasts(day)
            requests = [r for r in self.results.get(day, []) if r.get('valid_for_results')
                and any(r['models'].get(n, {}).get('status') == 'saved' for n in D39)
                and core.instant(r['scheduled_start']) <= self.clock()]
            key = ('results', day)
            if key not in self.cache or self.monotonic() - self.cache[key][0] >= 30:
                try:
                    # The DB adapter receives D39-only model projections, never H predictions.
                    restricted = [dict(r, models={n: m for n, m in r['models'].items() if n in D39}) for r in requests]
                    output, sources = fetch_results(restricted, self.session) if self.session else ({}, [])
                    reason = None if self.session else 'needs_authentication'
                    if self.session: self.previous_answers[day] = (deepcopy(output), deepcopy(sources), self.clock().isoformat())
                except Exception:
                    previous = self.previous_answers.get(day, ({}, [], None))
                    output, sources, reason = deepcopy(previous[0]), deepcopy(previous[1]), 'result_source_unavailable'
                body = {'version': VERSION, 'date': day, 'checked_at': self.clock().isoformat(),
                    'evaluation_kind': 'D39_weekend_exploratory', 'models': {n: LABELS[n] for n in D39},
                    'races': list(output.values()), 'pending_prediction_races': len(requests) - len(output),
                    'source_status': sources, 'reason': reason, 'H_results_included': False,
                    'stale': reason == 'result_source_unavailable',
                    'last_source_checked_at': self.previous_answers.get(day, ({}, [], None))[2],
                    'formal_evaluation': False, 'new_fits': 0}
                fingerprint = digest({k: v for k, v in body.items() if k != 'checked_at'})
                if fingerprint != self.revision.get(day):
                    versions = self.directory / 'result-versions'; versions.mkdir(exist_ok=True)
                    safety.write_json(versions / (uuid4().hex + '.json'), body, root=ROOT)
                    self.revision[day] = fingerprint
                self.cache[key] = (self.monotonic(), body)
            body = self.cache[key][1]
            history = self.directory / 'view-history'; history.mkdir(exist_ok=True)
            safety.write_json(history / (uuid4().hex + '.json'), {'viewed_at': self.clock().isoformat(),
                'period_start': day, 'period_end': day, 'models': list(D39),
                'race_count': len(body['races']), 'kind': 'exploratory_interim_view',
                'H_result_access': False, 'Stage7_result_access': False}, root=ROOT)
            return body

    def watch(self, day):
        """Independent R1 scoreboard. Original EV ledgers and studies stay closed."""
        from . import keiba_lab_watch as watch
        watch_day_guard(day)
        with self.lock:
            forecasts = self.forecasts(day)
            records = self.results.get(day, [])
            requests = [r for r in records if r.get('race_key') and r.get('scheduled_start')
                        and core.instant(r['scheduled_start']) <= self.clock()]
            key = ('watch', day)
            if key not in self.cache or self.monotonic() - self.cache[key][0] >= 15:
                checked = self.clock().isoformat(); reason = None
                try:
                    if not self.session: raise safety.MLFailure('needs_authentication')
                    observations, sources = fetch_watch_results(requests, self.session)
                    self.watch_previous[day] = (deepcopy(observations), deepcopy(sources), checked)
                except Exception:
                    observations, sources, checked = deepcopy(self.watch_previous.get(day, ({}, [], None)))
                    reason = 'result_source_unavailable' if self.session else 'needs_authentication'
                races, scores = watch.attach(records, observations, checked_at=checked)
                for race in races: race.pop('horse_ids', None); race.pop('d39_tickets', None); race.pop('valid_for_results', None)
                disclosed = [{'race_id': r['race_id'], 'source_sha256': r.get('source_prediction_sha256'),
                              'result_content_sha256': digest({k:v for k,v in r['watch'].items() if k != 'checked_at'})}
                             for r in races if r['watch']['podium']
                             and r['watch']['models']['H']['prediction_saved']]
                disclosure_id = digest(disclosed)
                history = ROOT / 'watch-disclosures'
                if disclosed and self.watch_disclosures.get(day) != disclosure_id:
                    history.mkdir(exist_ok=True)
                    safety.write_json(history / (uuid4().hex + '.json'), {
                        'viewed_at': self.clock().isoformat(), 'date': day, 'models': list(MODELS),
                        'kind': 'four_model_entertainment_result_disclosure', 'rule': watch.RULE,
                        'disclosed_races': disclosed, 'H_results_disclosed': True,
                        'H_disclosed_subset_is_not_unviewed': True, 'Stage7_study_access': False,
                        'pedigree_21_week_study_access': False, 'formal_evaluation': False}, root=ROOT)
                    self.watch_disclosures[day] = disclosure_id
                body = dict(forecasts, races=races, scoreboard=scores, checked_at=self.clock().isoformat(),
                    last_source_checked_at=checked, source_status=sources, reason=reason, stale=bool(reason),
                    available_dates=[d for d in forecasts['available_dates'] if WATCH_START <= d <= self.clock().date().isoformat()],
                    watch_format_version=1, watch_rule=watch.RULE, H_results_included=bool(disclosed),
                    disclosure={'H_results_disclosed': bool(disclosed), 'recorded': bool(disclosed),
                                'directory': str(history), 'is_unviewed': False if disclosed else None},
                    formal_evaluation=False, Stage7_study_access=False, pedigree_21_week_study_access=False)
                version_content = deepcopy({k: v for k, v in body.items() if k not in ('checked_at', 'last_source_checked_at')})
                for race in version_content['races']: race['watch'].pop('checked_at', None)
                revision = digest(version_content)
                if self.watch_revisions.get(day) != revision:
                    versions = self.directory / 'watch-versions'; versions.mkdir(exist_ok=True)
                    safety.write_json(versions / (uuid4().hex + '.json'), body, root=ROOT)
                    self.watch_revisions[day] = revision
                self.cache[key] = (self.monotonic(), body)
            return self.cache[key][1]


def result_csv(body):
    rows = []
    for race in body['races']:
        for name, model in race['models'].items():
            if name not in D39 or model.get('status') != 'saved': continue
            for row in model['rows']:
                rows.append(dict(race_id=race['race_id'], date=race['date'], venue=race['venue'], race_no=race['race_no'],
                    race_name='', surface='', distance='', going='', **{'class': ''}, horse_no=int(row['horse_no']),
                    horse_name=(race.get('horse_names') or {}).get(row['horse_no'], ''), model_id=name, pred_rank=row['rank'], win_prob=row['probability'],
                    finish_position=row['finish_position'], win_payout_100=row['win_payout_100'],
                    predicted_at=model['predicted_at'], start_time=race['scheduled_start'], schema_version='keiba-lab.v1.1',
                    dataset_id='local-weekend-' + body['date'], evaluation_kind='D39_weekend_exploratory',
                    source_stage='d39_forward_four', source_run_id=PARENT.name,
                    model_artifact_id=model['model_id'], source_sha256=race['source_prediction_sha256']))
    return csv_text(rows) if rows else '\ufeff' + ','.join(COLUMNS) + '\r\n'


def next_packet(viewer, day):
    from .d39_next.forward import public_packet
    from .d39_enrichment.forward import public_packet as enrichment_packet
    from .d39_next.results_csv import attach_exports
    from .d39_next.display import attach
    from .d39_next import contracts as next_contract
    from .d39_enrichment import contracts as enrichment_contract
    packet = public_packet(viewer, day)
    targets = packet.pop('_target_races', [])
    try:
        packet['improved'] = enrichment_packet(viewer, day)
    except Exception:
        packet['improved'] = None
        packet['improved_status'] = 'temporarily_unavailable'
    own_dir = next_contract.folder(packet['activation']['run_id']) if packet.get('activation') else None
    attach(packet, targets, candidate='next_D39', directory=own_dir)
    if packet.get('improved'):
        newer = packet['improved']
        new_dir = enrichment_contract.folder(newer['activation']['run_id']) if newer.get('activation') else None
        attach(newer, targets, candidate='enriched_D39', directory=new_dir, fallback=packet)
    return attach_exports(packet)


def make_handler(viewer, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_GET(self):
            attachment = None
            host = f'127.0.0.1:{port}'
            parsed = urlsplit(self.path)
            # A link from another page may open the HTML shell. Data requests
            # still require this exact loopback origin and cannot be framed.
            navigation = (parsed.path in ('/', '/results', '/research', '/next') and self.headers.get('Sec-Fetch-Mode') == 'navigate'
                and self.headers.get('Sec-Fetch-Dest') == 'document')
            if (self.headers.get('Host') != host or self.headers.get('Origin') not in (None, 'http://' + host)
                    or (self.headers.get('Sec-Fetch-Site') == 'cross-site' and not navigation)):
                self.send_error(403); return
            try:
                if parsed.path == '/api/health':
                    body, mime = json.dumps({'status': 'ok', 'version': VERSION, 'H_results_included': False,
                        'watch_H_result_viewing_enabled': True, 'Stage7_study_access': False}), 'application/json'
                elif parsed.path == '/api/watch':
                    args = parse_qs(parsed.query)
                    if set(args) != {'date'} or len(args['date']) != 1: raise safety.MLFailure('invalid_input')
                    body, mime = json.dumps(viewer.watch(watch_day_guard(args['date'][0])), ensure_ascii=False, allow_nan=False), 'application/json'
                elif parsed.path == '/api/research':
                    if parsed.query: raise safety.MLFailure('invalid_input')
                    from .multiwager.lab import load_current
                    body, mime = json.dumps(load_current(), ensure_ascii=False, allow_nan=False), 'application/json'
                elif parsed.path == '/api/next':
                    args = parse_qs(parsed.query)
                    if set(args) - {'date'}: raise safety.MLFailure('invalid_input')
                    day = day_guard(args.get('date', [''])[0])
                    packet = next_packet(viewer, day)
                    body, mime = json.dumps(packet, ensure_ascii=False, allow_nan=False), 'application/json'
                elif parsed.path == '/api/next-results.csv':
                    from .d39_next.results_csv import CHOICES, download
                    args = parse_qs(parsed.query)
                    if set(args) != {'date', 'candidate', 'kind'} or any(len(v) != 1 for v in args.values()):
                        raise safety.MLFailure('invalid_input')
                    day = day_guard(args['date'][0]); choice = args['candidate'][0]; kind = args['kind'][0]
                    if choice not in CHOICES or kind not in ('detail', 'summary'):
                        raise safety.MLFailure('invalid_input')
                    body, mime = download(day, choice, kind), 'text/csv'
                    attachment = f'next-d39-{choice}-{day}-{kind}.csv'
                elif parsed.path in ('/api/predictions', '/api/reference', '/api/d39-results', '/api/d39-results.csv'):
                    args = parse_qs(parsed.query); day = day_guard(args.get('date', [''])[0])
                    data = (viewer.forecasts(day) if parsed.path == '/api/predictions' else
                        viewer.references(day) if parsed.path == '/api/reference' else viewer.d39_results(day))
                    body = result_csv(data) if parsed.path.endswith('.csv') else json.dumps(data, ensure_ascii=False, allow_nan=False)
                    mime = 'text/csv' if parsed.path.endswith('.csv') else 'application/json'
                else:
                    asset = {'/': 'index.html', '/results': 'results.html', '/research': 'research.html',
                        '/next': 'next.html', '/next.js': 'next.js',
                        '/research.js': 'research.js', '/app.js': 'app.js',
                        '/style.css': 'style.css'}.get(parsed.path)
                    if asset is None: self.send_error(404); return
                    body = (WEB / asset).read_text(encoding='utf-8')
                    mime = 'text/html' if asset.endswith('.html') else 'text/javascript' if asset.endswith('.js') else 'text/css'
                body = body.encode('utf-8')
                self.send_response(200); self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
                if attachment: self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('X-Frame-Options', 'DENY')
                self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; form-action 'none'")
                self.end_headers(); self.wfile.write(body)
            except Exception as exc:
                code = exc.code if isinstance(exc, (safety.MLFailure, core.SimulationError)) else 'internal_error'
                body = json.dumps({'status': 'pending', 'reason': code}).encode('utf-8')
                self.send_response(400); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-store')
                self.end_headers(); self.wfile.write(body)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local KEIBA LAB saved-forecast viewer; no model fitting')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args(argv)
    try:
        if args.port != 8767: raise safety.MLFailure('invalid_input')
        from .db_connection import read_only_session
        from .win_sim_cli import connection_config
        password = os.environ.pop('PGPASSWORD', None)
        # The recorder gets the long queue budget; a busy viewer simply retries
        # at its next poll instead of competing for the one permitted DB session.
        session = (lambda: read_only_session(connection_config(2000), password, gate_timeout_ms=1500)) if password else None
        viewer = Viewer(session=session)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(viewer, args.port))
        server.daemon_threads = True
        # Suppress default traceback/HTTP line logging, including raw query strings.
        server.handle_error = lambda *args: None
        atomic_snapshot(ROOT, 'current.json', {'pid': os.getpid(), 'port': args.port,
            'directory': str(viewer.directory), 'started_at': now().isoformat(), 'version': VERSION})
        print(json.dumps({'status': 'ready', 'url': f'http://127.0.0.1:{args.port}', 'folder': str(viewer.directory)}), flush=True)
        server.serve_forever()
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'local_viewer_start_failed'}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
