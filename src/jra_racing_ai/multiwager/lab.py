"""Small saved-development adapter for the local LAB. No fitting or live outcomes.

The old single-horse CSV contract is unchanged. This sidecar keeps wager events,
rank references, official target sets and money confirmation separate.
"""
from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from ..ml import safety
from ..payout_decoder import _field_format, FieldFormat, PayoutDecodingError
from ..win_sim import local, core
from ..keiba_lab_export import VENUES
from . import contracts as c
from .outcomes import target_key, rank_reference

ROOT = c.BASE / 'keiba-lab-multiwager'
BASELINE = c.folder('881a828f4b0949388766374ddbf9a3ce')
EXPORT = c.BASE / 'keiba-lab-export/evaluations/3b0331c987dd4c1c8c344a2539289ac7'
VERSION = 'keiba-lab.multiwager.view.v1'
LAYOUTS = {'place': ('fukusho', 5, 2, 2), 'wide': ('wide', 7, 4, 3),
           'umaren': ('umaren', 3, 4, 3)}
SERIES = ('H_PL', 'raw_market_PL', 'ABCD_PL')


def digest(path):
    return safety.digest(Path(path), root=c.BASE)


def read(path):
    return safety.read_json(Path(path), root=c.BASE)


def check(path, expected):
    if not isinstance(expected, str) or digest(path) != expected:
        raise safety.MLFailure('artifact_invalid')


def event_values(values, roster, wager, paid, *, require_mass=True):
    """No rounding, renormalization, inferred pay-key or product of marginals."""
    from itertools import combinations
    if wager not in ('win', *c.WAGERS) or type(paid) is not int or paid not in (2, 3):
        raise safety.MLFailure('invalid_input')
    expected = set(roster) if wager in ('win', 'place') else {target_key(p) for p in combinations(roster, 2)}
    if (not isinstance(values, dict) or set(values) != expected
            or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in values.values())):
        raise safety.MLFailure('invalid_input')
    import math
    mass = {'win': 1, 'place': paid, 'wide': 3, 'umaren': 1}[wager]
    if require_mass and not math.isclose(math.fsum(values.values()), mass, abs_tol=1e-9):
        raise safety.MLFailure('invalid_input')
    return [{'target': key, 'probability': values[key], 'rank': i}
            for i, key in enumerate(sorted(values, key=lambda k: (-values[k], k)), 1)]


def normalize_hr(rows, record, wager, source_sha):
    """Ordinary current Monday snapshot only; preserve b verbatim, never infer yen.

    JV 4.9.0.1 PDF pp.13/33 supplies layout/flags and the special two-payee 00.
    PC physical columns are resolved from the actual schema by fetch_hr below.
    Historical correction lineage is not reconstructed. Multiple stored versions,
    cancellation/refund/special flags, unknown fields and ties remain pending.
    """
    roster = sorted(record['finishes']); paid = record['paid_positions']
    c.identity(record['race_id'], record['date'])
    if wager not in LAYOUTS: raise safety.MLFailure('invalid_input')
    if not isinstance(source_sha, str) or not re.fullmatch('[0-9a-f]{64}', source_sha):
        raise safety.MLFailure('invalid_input')
    base = {'wager': wager, 'target_status': 'pending', 'official_targets': None,
            'money_status': 'unconfirmed', 'payouts_yen_per_100': None,
            'source_sha256': source_sha, 'source': 'public.jvd_hr',
            'revision_basis': 'single_current_monday_snapshot_not_revision_history',
            'state': 'unconfirmed', 'reason': None}
    def pending(reason): return dict(base, reason=reason)
    if len(rows) != 1: return pending('hr_missing_or_multiple_versions')
    row = rows[0]
    key = record['race_id']
    expected_key = (key[:4], key[4:8], key[8:10], key[10:12], key[12:14], key[14:])
    if tuple(row.get('race_key', ())) != expected_key:
        return pending('hr_full_key_mismatch')
    if row.get('status') != '2' or row.get('record_type') != 'HR' or row.get('oversized'):
        return pending('hr_category_or_record_unconfirmed')
    if any(row.get(wager + '_' + flag) != '0' for flag in ('unestablished', 'special', 'refund')):
        return pending('wager_special_or_unknown_flags')
    # No phantom losing targets for incomplete or unsupported final rosters.
    refs, why = rank_reference(record['finishes'], paid)
    if why: return pending(why)
    if row.get('actual_runners') != f'{len(roster):02d}':
        return pending('hr_roster_count_mismatch')
    _, slots, target_width, popularity_width = LAYOUTS[wager]
    found = {}; raw_amounts = {}
    try:
        for i in range(1, slots + 1):
            a, b, pop = (row.get(f'{wager}_{i}{p}') for p in 'abc')
            forms = tuple(_field_format(v, width) for v, width in
                          ((a, target_width), (b, 9), (pop, popularity_width)))
            if forms == (FieldFormat.BLANK,) * 3: continue
            if (wager == 'place' and paid == 2 and i == 3 and a == '00'
                    and b == ' ' * 9 and pop == ' ' * 2):
                continue  # Only this exact special case is specified on PDF p.33.
            if forms != (FieldFormat.DIGITS,) * 3 or int(a) == 0 or int(b) == 0 or int(pop) == 0:
                return pending('slot_special_or_incomplete')
            parts = [a] if target_width == 2 else [a[:2], a[2:]]
            key = target_key(parts)
            if not set(parts) <= set(roster) or key in found:
                return pending('target_duplicate_or_roster_mismatch')
            found[key] = True; raw_amounts[key] = b
    except (PayoutDecodingError, safety.MLFailure):
        return pending('slot_format_unconfirmed')
    expected = {key for key, label in refs[wager].items() if label}
    if set(found) != expected:
        return pending('complete_hr_target_set_not_confirmed')
    return dict(base, target_status='confirmed', official_targets=sorted(found),
        raw_amount_fields=raw_amounts, state='normal',
        reason='pc_money_unit_per_wager_not_confirmed')


def fetch_hr(db, records):
    """One existing bounded SELECT builder, full six-part keys, no table scan."""
    if not 1 <= len(records) <= 20: raise safety.MLFailure('input_limit')
    races = []
    for r in records:
        c.identity(r['race_id'], r['date'])  # Date guard BEFORE SQL or labels.
        key = r['race_id']; races.append((key[:4], key[4:8], key[8:10], key[10:12], key[12:14], key[14:]))
    budget = local.Budget(20); columns = local.table_info(db, 'jvd_hr', budget)
    if not columns: return [], {}, {'status': 'hr_table_missing', 'rows': 0}
    fields = {'status': ('data_kubun', 1), 'record_type': ('record_id', 2),
              'actual_runners': ('shusso_tosu', 2)}
    mappings = {}; unavailable = []
    for wager, (physical, slots, a_width, c_width) in LAYOUTS.items():
        proposed = {}
        for flag, names in {'unestablished': ('fuseiritsu_flag_', 'fuseiritu_flag_'),
                            'special': ('tokubarai_flag_', 'tokuharai_flag_'),
                            'refund': ('henkan_flag_',)}.items():
            matches = [prefix + physical for prefix in names if prefix + physical in columns]
            if len(matches) == 1: proposed[wager + '_' + flag] = (matches[0], 1)
        proposed.update({f'{wager}_{i}{p}': (f'haraimodoshi_{physical}_{i}{p}', width)
                         for i in range(1, slots + 1) for p, width in zip('abc', (a_width, 9, c_width))})
        if len(proposed) != 3 + slots * 3 or any(v[0] not in columns for v in proposed.values()):
            unavailable.append(wager); continue
        fields.update(proposed); mappings[wager] = proposed
    for name, _ in fields.values():
        col = columns.get(name, {})
        if not col.get('type_is_builtin') or col.get('type_name') not in local.TEXT_TYPES:
            raise safety.MLFailure('invalid_schema')
    rows = local._rows(db, budget, 'jvd_hr', fields, columns, tuple(races),
                       max_rows=len(records) * 3, join_targets=True)
    return rows, mappings, {'status': 'bounded_read_complete', 'rows': len(rows),
                            'missing_wager_columns': unavailable, 'max_seconds': 20}


def validate_packet(packet):
    if (packet.get('schema_version') != VERSION or not 1 <= len(packet.get('races', [])) <= 20
            or packet.get('evaluation_kind') != 'reused_development_postrace_replay'
            or packet.get('new_fits') != 0 or packet.get('Stage7_results_used') is not False
            or packet.get('sealed_2026_used') is not False):
        raise safety.MLFailure('invalid_input')
    seen = set()
    for race in packet['races']:
        c.identity(race['race_id'], race['date'])
        if race['race_id'] in seen: raise safety.MLFailure('duplicate_entry')
        seen.add(race['race_id'])
        roster = race['roster']
        if len(set(roster)) != len(roster) or not 3 <= len(roster) <= 28:
            raise safety.MLFailure('invalid_identity')
        for h in roster: target_key([h])
        if set(race['finishes']) != set(roster): raise safety.MLFailure('invalid_identity')
        for model in race['win_models'].values():
            event_values(model['probabilities'], roster, 'win', race['paid_positions'])
        for model in race['wager_models'].values():
            if not model.get('probabilities') or not set(model['probabilities']) <= set(c.WAGERS):
                raise safety.MLFailure('invalid_input')
            for wager, values in model['probabilities'].items():
                event_values(values, roster, wager, race['paid_positions'],
                    require_mass=model.get('probability_status') != 'trained_direct_official_place')
    return packet


def learned_source(directory, race_ids):
    """Saved 2025 development outputs only. No new inference or model fitting."""
    from . import training
    training.approved(directory)
    done = read(directory/'training_complete.json')
    if done.get('status') != 'complete' or done.get('fits_consumed') != 2:
        raise safety.MLFailure('artifact_invalid')
    manifest = read(directory/'training_inputs.json')
    target_file = Path(manifest['official_targets_file']); check(target_file, manifest['official_HR_source_sha256'])
    targets = read(target_file)['records']; references = {str(target_file): manifest['official_HR_source_sha256']}
    output = defaultdict(dict); info = {}
    for family in c.FAMILIES:
        measured = read(directory/(family+'.evaluation.json')); model_id = measured['model']['model_id']
        meta_path = directory/'models'/model_id/'model.json'; metadata = read(meta_path)
        if metadata.get('family') != family or metadata.get('version') != c.VERSION:
            raise safety.MLFailure('invalid_model')
        tree = meta_path.parent/'model.lgb.txt'; check(tree, metadata['tree_sha256'])
        references[str(tree)] = metadata['tree_sha256']; references[str(meta_path)] = digest(meta_path)
        path = directory/(family+'.predictions.local.jsonl'); references[str(path)] = digest(path)
        info[family] = metadata
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line); c.identity(record['race_id'], record['date'])
                if not record['date'].startswith('2025-'): raise safety.MLFailure('invalid_split')
                if record['race_id'] not in race_ids: continue
                if record.get('model_id') != model_id or family in output[record['race_id']]:
                    raise safety.MLFailure('invalid_identity')
                output[record['race_id']][family] = {
                    'model_id': model_id, 'probabilities': {w: p for w,p in record['probabilities'].items() if w in c.WAGERS},
                    'source_sha256': references[str(path)], 'predicted_at': None,
                    'temperature': metadata['calibration']['temperature'],
                    'probability_status': 'trained_direct_official_place' if family == c.FAMILIES[0] else 'trained_order_PL_reference',
                    'evaluation_kind': 'reused_2025_development_after_2024_training',
                    'rank_reference_labels': {}}
    return output, targets, references, info


def load_current():
    """Only the explicit development sidecar. Never touch live or Stage 7 results."""
    pointer = ROOT / 'current.json'
    if not pointer.exists():
        return {'schema_version': VERSION, 'status': 'not_prepared', 'races': []}
    p = read(pointer)
    if not re.fullmatch('[0-9a-f]{32}', str(p.get('id', ''))): raise safety.MLFailure('invalid_identity')
    directory = safety.local_path(ROOT / p['id'], root=ROOT)
    packet = directory / 'packet.local.json'; check(packet, p['sha256'])
    return validate_packet(read(packet))


def prepare(*, session=None, export=EXPORT, baseline=BASELINE, output=ROOT, learned_run=None):
    """Re-use two accepted races, saved PL outputs, and optionally exact-key HR."""
    export = safety.local_path(export, root=c.BASE)
    em = read(export / 'manifest.json'); check(export / 'keiba-lab.csv', em['csv_sha256'])
    if em.get('evaluation_kind') != 'stage6_development': raise safety.MLFailure('invalid_split')
    wins = defaultdict(list)
    with (export / 'keiba-lab.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            c.identity(row['race_id'], row['date']); wins[row['race_id']].append(row)
    if not 1 <= len(wins) <= 20: raise safety.MLFailure('input_limit')
    dm = read(baseline / 'data_manifest.json'); check(baseline / dm['file'], dm['sha256'])
    record_map = {}
    with (baseline / dm['file']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); c.identity(row['race_id'], row['date'])
            if row['race_id'] in wins: record_map[row['race_id']] = row
            if len(record_map) == len(wins): break
    if set(record_map) != set(wins): raise safety.MLFailure('invalid_identity')
    prediction_path = baseline / 'predictions.local.jsonl'; prediction_sha = digest(prediction_path)
    saved = defaultdict(dict)
    with prediction_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); c.identity(row['race_id'], row['date'])
            if row['race_id'] in wins and row['model'] in SERIES:
                if row['model'] in saved[row['race_id']]: raise safety.MLFailure('duplicate_entry')
                saved[row['race_id']][row['model']] = row
            if sum(map(len, saved.values())) == len(wins) * len(SERIES): break
    if any(set(saved[r]) != set(SERIES) for r in wins): raise safety.MLFailure('invalid_identity')
    commit = read(baseline / 'prediction_commit.json')
    learned = {}; learned_targets = {}; learned_refs = {}; learner_info = {}
    development_comparison = None
    if learned_run is not None:
        learned, learned_targets, learned_refs, learner_info = learned_source(learned_run, set(wins))
        report_path = learned_run / 'development_comparison.json'
        if report_path.exists():
            comparison = read(report_path)
            if comparison.get('2026_rows_used') != 0 or comparison.get('Stage7_results_used') is not False:
                raise safety.MLFailure('invalid_split')
            development_comparison = {k: comparison[k] for k in ('metrics', 'coverage', 'development_period', 'interpretation')}
            learned_refs[str(report_path)] = digest(report_path)
    folder = safety.local_path(output / uuid4().hex, root=c.BASE); folder.mkdir(parents=True, exist_ok=False)
    schema = {}; source_status = {'status': 'not_requested', 'rows': 0}; hr_rows = []
    if session:
        try:
            with session() as db: hr_rows, schema, source_status = fetch_hr(db, list(record_map.values()))
        except Exception as exc:
            source_status = {'status': 'pending', 'reason': exc.code if isinstance(exc, (safety.MLFailure, core.SimulationError)) else 'source_unavailable', 'rows': 0}
    elif learned_targets:
        source_status = {'status': 'reused_verified_development_HR_cache', 'rows': 0,
                         'target_races_reused': sum(key in learned_targets for key in wins)}
    raw_path = folder / 'hr-source.local.json'
    safety.write_json(raw_path, {'rows': hr_rows, 'columns': schema, 'status': source_status}, root=c.BASE)
    raw_sha = digest(raw_path); grouped = defaultdict(list)
    for row in hr_rows: grouped[''.join(row['race_key'])].append(row)
    races = []; statuses = Counter()
    for key, rows in sorted(wins.items()):
        record = record_map[key]; roster = sorted(record['finishes']); win_models = {}
        for name in sorted({r['model_id'] for r in rows}):
            part = [r for r in rows if r['model_id'] == name]
            if (sorted(f"{int(r['horse_no']):02d}" for r in part) != roster
                    or any(int(r['finish_position']) != record['finishes'][f"{int(r['horse_no']):02d}"] for r in part)):
                raise safety.MLFailure('invalid_identity')
            values = {f"{int(r['horse_no']):02d}": float(r['win_prob']) for r in part}
            ranks = {x['target']: x['rank'] for x in event_values(values, roster, 'win', record['paid_positions'])}
            if any(ranks[f"{int(r['horse_no']):02d}"] != int(r['pred_rank']) for r in part):
                raise safety.MLFailure('invalid_identity')
            win_models[name] = {'probabilities': values, 'model_id': part[0]['model_artifact_id'],
                'source_sha256': part[0]['source_sha256'], 'predicted_at': part[0]['predicted_at'] or None,
                'probability_status': 'saved_win_probability', 'evaluation_kind': 'Stage6_development_reuse'}
        wager_models = {}
        for name in SERIES:
            row = saved[key][name]
            for wager in c.WAGERS: event_values(row['probabilities'][wager], roster, wager, record['paid_positions'])
            label = name[:-3] + '_' + record['date'][:4]
            info = commit['models'].get(label, {'model_id': 'raw_market_no_fit'})
            wager_models[name] = {'model_id': info['model_id'], 'probabilities': row['probabilities'],
                'source_sha256': prediction_sha, 'predicted_at': None,
                'probability_status': 'uncalibrated_PL_model_implied_reference',
                'evaluation_kind': row['evaluation_kind'], 'rank_reference_labels': row['rank_reference_labels']}
        official = {}
        refs, why = rank_reference(record['finishes'], record['paid_positions'])
        if why: raise safety.MLFailure('outcomes_incomplete')
        for name, model in learned.get(key, {}).items():
            model['rank_reference_labels'] = {w: refs[w] for w in model['probabilities']}
            wager_models[name] = model
        for wager in c.WAGERS:
            official[wager] = normalize_hr(grouped[key], record, wager, raw_sha)
            if key in learned_targets:
                event = learned_targets[key].get(wager, {})
                if event.get('target_status') == 'confirmed':
                    expected = {k for k,v in refs[wager].items() if v}
                    if set(event.get('official_targets', [])) != expected:
                        raise safety.MLFailure('invalid_identity')
                    official[wager] = event
            statuses[wager + '/' + official[wager]['target_status']] += 1
        first = rows[0]
        races.append({'race_id': key, 'date': record['date'], 'venue': VENUES[key[8:10]],
            'race_no': int(key[-2:]), 'roster': roster, 'paid_positions': record['paid_positions'],
            'horse_names': {f"{int(r['horse_no']):02d}": r['horse_name'] for r in rows if r['horse_name']},
            'distance': first['distance'] or None, 'surface': first['surface'] or None,
            'finishes': record['finishes'], 'win_models': win_models, 'wager_models': wager_models,
            'official': official, 'rank_reference_only': True})
    packet = validate_packet({'schema_version': VERSION, 'artifact_id': folder.name,
        'status': 'ready', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'evaluation_kind': 'reused_development_postrace_replay', 'prediction_time': 'not_original_prerace_record',
        'races': races, 'new_fits': 0, 'included_saved_research_fits': 2 if learner_info else 0,
        'sealed_2026_used': False, 'Stage7_results_used': False,
        'direct_place_model': 'trained_saved_development_only' if learner_info else 'not_trained',
        'shared_order_model': 'trained_saved_development_only' if learner_info else 'not_trained',
        'development_comparison': development_comparison,
        'assumption': 'PL_without_dead_heats;not_independent_marginal_products',
        'source_status': source_status, 'official_status_counts': dict(statuses),
        'source_references': {str(export / 'keiba-lab.csv'): em['csv_sha256'],
             str(baseline / dm['file']): dm['sha256'], str(prediction_path): prediction_sha,
             str(raw_path): raw_sha, **learned_refs}, 'ROI': None})
    safety.write_json(folder / 'packet.local.json', packet, root=c.BASE)
    # Preserve the original 22-column research event schema; never overload win_prob.
    with (folder / 'multiwager.local.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=c.COLS); writer.writeheader()
        for race in races:
            for name, model in race['wager_models'].items():
                for wager in model['probabilities']:
                    for value in event_values(model['probabilities'][wager], race['roster'], wager, race['paid_positions'],
                            require_mass=model.get('probability_status') != 'trained_direct_official_place'):
                        target = value['target']; pieces = target.split('-'); official = race['official'][wager]
                        writer.writerow(dict(schema_version=c.SCHEMA, race_id=race['race_id'], date=race['date'],
                            venue=race['venue'], race_no=race['race_no'], wager=wager, horse_no_1=pieces[0],
                            horse_no_2=pieces[1] if len(pieces)==2 else '', probability=repr(value['probability']),
                            pred_rank=value['rank'], model_id=model['model_id'], source_model=name,
                            evaluation_kind=packet['evaluation_kind'], source_stage='saved_multiwager_0fit',
                            source_sha256=model['source_sha256'], generated_at=packet['generated_at'], predicted_at='',
                            place_paid_positions=race['paid_positions'], rank_reference_hit=model['rank_reference_labels'][wager][target],
                            official_hit=int(target in official['official_targets']) if official['target_status']=='confirmed' else '',
                            official_payout_yen_per_100='', missing_reason='pc_money_unit_per_wager_not_confirmed'))
    summary = {'status': 'ok', 'artifact_id': folder.name, 'races': len(races),
        'runners': sum(len(r['roster']) for r in races), 'win_rows': sum(len(v) for v in wins.values()),
        'official_status_counts': dict(statuses), 'source_status': source_status, 'new_fits': 0,
        'ROI': None, 'H_live_results_included': False, 'sealed_2026_used': False}
    safety.write_json(folder / 'summary.json', summary, root=c.BASE)
    # Publish only a pointer to a complete, immutable, validated packet.
    pointer = safety.local_path(output / 'current.json', root=c.BASE)
    tmp = pointer.with_name('pointer-' + uuid4().hex + '.tmp')
    tmp.write_text(json.dumps({'id': folder.name, 'sha256': digest(folder/'packet.local.json')}), encoding='utf-8')
    tmp.replace(pointer)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Saved development wager viewer preparation; 0 fits')
    parser.add_argument('--read-local-db', action='store_true')
    parser.add_argument('--export', type=Path, default=EXPORT)
    parser.add_argument('--learned-run', type=Path)
    args = parser.parse_args(argv)
    try:
        import os
        session = None
        if args.read_local_db:
            from ..db_connection import read_only_session
            from ..win_sim_cli import connection_config
            password = os.environ.pop('PGPASSWORD', None)
            if not password: raise safety.MLFailure('explicit_execution_required')
            session = lambda: read_only_session(connection_config(3000), password, gate_timeout_ms=1500)
        print(json.dumps(prepare(session=session, export=args.export, learned_run=args.learned_run), ensure_ascii=False), flush=True); return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, (safety.MLFailure, core.SimulationError)) else 'preparation_failed'
        print(json.dumps({'status':'stopped','reason':code,'stage':'multiwager_lab'}), flush=True); return 1

