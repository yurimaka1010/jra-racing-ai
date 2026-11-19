"""Bounded reuse of causal ABCD caches plus exact-key official target labels."""
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import time

from ..ml import safety
from ..objective_pilot.data import read_frame, validate_roster
from ..research import contracts as features
from ..stage4 import data as market_data
from ..stage5 import place
from . import contracts as c, lab, models


def verified_part(path, expected, span):
    lab.check(path, expected)
    import pandas as pd
    identifiers = pd.read_csv(path, usecols=['entry_key', 'date'], dtype='string')
    if len(identifiers) > 130000: raise safety.MLFailure('input_limit')
    for value in identifiers.date.unique():
        day = date.fromisoformat(str(value))
        if not date(2022, 1, 1) <= day <= date(2025, 9, 30):
            raise safety.MLFailure('invalid_split')
    f = read_frame(path)
    f = f.loc[(f.date >= span[0]) & (f.date <= span[1])].copy()
    for value in f.date.unique(): c.day(str(value))
    f = f.sort_values('entry_key').reset_index(drop=True)
    validate_roster(f)
    return f


def attach_labels(frame, regimes, official):
    """Filter only explicit label/as-of gaps; never optional feature missingness."""
    retained = []; reasons = Counter(); source_races = frame.entry_key.str[:16].nunique()
    for race, f in frame.groupby(frame.entry_key.str[:16], sort=True):
        paid = regimes.get(race)
        if paid not in (2, 3): reasons['asof_place_pay_key_unavailable'] += 1; continue
        event = official.get(race, {}).get('place', {})
        if event.get('target_status') != 'confirmed':
            reasons[event.get('reason') or 'official_place_targets_unavailable'] += 1; continue
        winners = set(event['official_targets']); horses = set(f.entry_key.str[-2:])
        if not winners <= horses or len(winners) != paid:
            reasons['official_targets_roster_or_pay_key_mismatch'] += 1; continue
        f = f.copy(); f['place_paid_positions'] = paid
        f['official_place_label'] = [int(h in winners) for h in f.entry_key.str[-2:]]
        f['official_label_status'] = 'verified_ordinary_HR'
        models.validate_frame(f, c.FAMILIES[0]); retained.append(f)
    if not retained: raise safety.MLFailure('outcomes_incomplete')
    import pandas as pd
    result = pd.concat(retained, ignore_index=True).sort_values('entry_key').reset_index(drop=True)
    return result, {'source_races': int(source_races), 'retained_races': len(retained),
                    'excluded_reasons': dict(reasons), 'optional_features_used_for_filtering': False}


def prepare(directory, session):
    """At most 300 sec, <=20 complete keys per SELECT, no 2026 or live results."""
    started = time.monotonic(); plan = c.training_plan(); references = {}; frames = {}; statuses = {}
    def budget():
        if time.monotonic() - started > plan['preparation_seconds_max']:
            raise safety.MLFailure('query_timeout')
    source_manifest = c.PILOT / 'data_manifest.json'; pilot = lab.read(source_manifest)
    references[str(source_manifest)] = lab.digest(source_manifest)
    fm_path = c.S2 / 'feature_manifest.json'; fm = lab.read(fm_path)
    if fm.get('formula_version') != features.FORMULA_VERSION or fm.get('sealed_2026_rows') != 0:
        raise safety.MLFailure('mapping_not_reviewed')
    references[str(fm_path)] = lab.digest(fm_path)
    for name in ('train', 'early_stopping', 'calibration', 'evaluation'):
        item = pilot['parts'][name]; path = c.PILOT / item['file']
        frames[name] = verified_part(path, item['sha256'], plan[name]); references[str(path)] = item['sha256']
        budget()
    # Existing verified same-record O1 mapping. No head-count proxy for pay key.
    complete = lab.read(c.S5 / 'place.complete.json')
    for text, expected in complete['files'].items():
        path = Path(text)
        if path.name[:4] in ('2023', '2024', '2025'):
            lab.check(path, expected); references[text] = expected
    regimes = {}; records = {}; asof_reasons = Counter()
    for name, frame in frames.items():
        start, end = date.fromisoformat(str(frame.date.min())), date.fromisoformat(str(frame.date.max()))
        raw = place.inputs(c.S5, start, end); packets = market_data.market(place.c.S4, start, end)
        for race_id, f in frame.groupby(frame.entry_key.str[:16], sort=True):
            budget(); key = str(race_id); c.identity(key, str(f.date.iloc[0]))
            race = (key[:4], key[4:8], key[8:10], key[10:12], key[12:14], key[14:])
            entry = raw.get(race); packet = packets.get(race)
            if not entry or entry.get('read_error') or not packet or not packet.get('quotes', {}).get('10'):
                asof_reasons['saved_same_asof_O1_missing'] += 1; continue
            roster = set(f.entry_key.str[-2:])
            if set(packet['quotes']['10']['odds']) != roster:
                asof_reasons['same_asof_roster_mismatch'] += 1; continue
            values, why, paid = place.decode(race, entry['rows'], packet['quotes']['10']['odds'], packet)
            if paid not in ('2', '3') or (values is None and why != 'place_capped_or_ambiguous'):
                asof_reasons[why or 'asof_place_pay_key_unavailable'] += 1; continue
            regimes[key] = int(paid)
            records[key] = {'race_id': key, 'date': str(f.date.iloc[0]), 'paid_positions': int(paid),
                'finishes': dict(zip(f.entry_key.str[-2:], map(int, f.finish_position)))}
    budget()
    source_dir = directory / 'official-source'; source_dir.mkdir(exist_ok=False)
    official = {}; db_statuses = Counter(); all_keys = sorted(records); batch_refs = {}
    for i in range(0, len(all_keys), 20):
        budget(); keys = all_keys[i:i+20]; raw_rows = []; status = None; columns = {}
        try:
            with session() as db: raw_rows, columns, status = lab.fetch_hr(db, [records[k] for k in keys])
        except Exception as exc:
            status = {'status': 'pending', 'reason': getattr(exc, 'code', 'source_unavailable'), 'rows': 0}
        path = source_dir / f'batch_{i//20:04d}.local.json'
        c.write(source_dir, path.name, {'rows': raw_rows, 'status': status, 'columns': columns})
        source_sha = lab.digest(path); batch_refs[str(path)] = source_sha
        grouped = defaultdict(list)
        for row in raw_rows: grouped[''.join(row['race_key'])].append(row)
        db_statuses[status['status']] += 1
        for key in keys:
            official[key] = {w: lab.normalize_hr(grouped[key], records[key], w, source_sha) for w in c.WAGERS}
        # Leave the single reader role available to the time-critical recorder.
        time.sleep(.05)
    c.write(directory, 'official_targets.local.json', {'records': official, 'source_sha256': batch_refs,
        'money_status': 'unconfirmed_per_wager', '2026_rows_read': 0, 'Stage7_results_read': 0})
    target_sha = lab.digest(directory / 'official_targets.local.json'); parts = {}
    for name, frame in frames.items():
        budget(); result, statuses[name] = attach_labels(frame, regimes, official)
        cap = plan['train_row_cap'] if name == 'train' else plan['evaluation_row_cap'] if name == 'evaluation' else plan['early_and_calibration_row_cap_each']
        if len(result) > cap: raise safety.MLFailure('input_limit')
        path = directory / f'{name}.official.local.csv'
        with path.open('x', encoding='utf-8', newline='') as stream: result.to_csv(stream, index=False)
        parts[name] = {'file': str(path), 'sha256': lab.digest(path), 'rows': len(result),
            'races': int(result.entry_key.str[:16].nunique()), 'start': str(result.date.min()), 'end': str(result.date.max()),
            'feature_missing': {f: int(result[f].isna().sum()) for f in models.base_spec()['features']}}
    manifest = {'version': c.VERSION, 'formula_version': features.FORMULA_VERSION,
        'history_lag_days': 7, 'sealed_2026_used': False, 'Stage7_results_read': False,
        'label_contract': 'verified_ordinary_HR_place_targets', 'official_HR_source_sha256': target_sha,
        'official_targets_file': str(directory/'official_targets.local.json'), 'parts': parts,
        'source_references': references, 'population': statuses, 'asof_exclusions': dict(asof_reasons),
        'DB_batch_status_counts': dict(db_statuses), 'preparation_wall_seconds': time.monotonic()-started,
        'official_money_units_confirmed': False, 'optional_feature_missing_does_not_exclude_races': True}
    c.write(directory, 'training_inputs.json', manifest)
    return {'status': 'ready', 'parts': {k: {'races': p['races'], 'rows': p['rows']} for k,p in parts.items()},
            'wall_seconds': manifest['preparation_wall_seconds'], 'new_fits': 0,
            'official_payouts': 'unconfirmed', 'manifest': str(directory/'training_inputs.json')}
