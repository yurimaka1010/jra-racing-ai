"""Hash-checked cache selection and outcome-only exact joins, with no DB access."""
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

from ..ml import safety
from ..research import contracts as features
from . import contracts as c

KEY_FIELDS = ('kaisai_nen', 'kaisai_tsukihi', 'keibajo_code', 'kaisai_kai', 'kaisai_nichime', 'race_bango', 'horse_number')


def validate_roster(frame, *, with_labels=True):
    import numpy as np
    if frame.empty or frame.entry_key.duplicated().any():
        raise safety.MLFailure('duplicate_entry')
    for key, value in zip(frame.entry_key, frame.date):
        d = c.day(value)
        if (not re.fullmatch('[0-9]{18}', str(key)) or key[:8] != d.strftime('%Y%m%d')
                or not 1 <= int(key[8:10]) <= 10 or not 1 <= int(key[-2:]) <= 28):
            raise safety.MLFailure('invalid_identity')
    y = frame.winner_label.to_numpy(float)
    if not np.isin(y, [0, 1]).all():
        raise safety.MLFailure('invalid_input')
    for _, f in frame.groupby(frame.entry_key.str[:16], sort=False):
        if len(f) < 2 or f.winner_label.sum() != 1 or f.date.nunique() != 1:
            raise safety.MLFailure('invalid_identity')
        if with_labels:
            ranks = f.finish_position.to_numpy(float); n = f.actual_starters.to_numpy(float)
            if (not np.isfinite(ranks).all() or not np.isfinite(n).all()
                    or not (ranks == np.floor(ranks)).all() or not (n == len(f)).all()
                    or not ((ranks >= 1) & (ranks <= n)).all() or not (n <= 28).all()
                    or not np.array_equal((ranks == 1).astype(int), f.winner_label.to_numpy(int))):
                raise safety.MLFailure('invalid_identity')


def read_frame(path):
    import pandas as pd
    dtype = {n: 'string' for n in features.CATEGORIES}
    dtype.update(entry_key='string', date='string', start_time='string', previous_start='string')
    return pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[''], float_precision='round_trip')


def load_part(directory, name):
    if name not in c.PARTS:
        raise safety.MLFailure('invalid_input')
    manifest = c.read(directory, 'data_manifest.json'); item = manifest['parts'][name]
    path = safety.local_path(directory / item['file'], root=c.ROOT, exists=True)
    if safety.digest(path, root=c.ROOT) != item['sha256']:
        raise safety.MLFailure('artifact_invalid')
    f = read_frame(path)
    if len(f) != item['rows'] or f.entry_key.str[:16].nunique() != item['races']:
        raise safety.MLFailure('invalid_identity')
    validate_roster(f)
    return f


def prepare(directory, p):
    import pandas as pd
    began = time.monotonic(); references = {}
    def check_time():
        if time.monotonic() - began > p['budget']['preparation_wall_seconds_max']:
            raise safety.MLFailure('query_timeout')
    def source(path, expected):
        check_time(); path = safety.local_path(path, root=c.BASE, exists=True)
        if safety.digest(path, root=c.BASE) != expected:
            raise safety.MLFailure('artifact_invalid')
        references[str(path)] = expected
        return path
    for path, expected in p['input_reference_sha256'].items():
        source(Path(path), expected)
    fm = safety.read_json(c.S2 / 'feature_manifest.json', root=c.BASE)
    sm = safety.read_json(c.S2 / 'source_manifest.json', root=c.BASE)
    if (fm['features'] != list(features.ALL_FEATURES) or fm['formula_version'] != features.FORMULA_VERSION
            or fm['sealed_2026_rows'] != 0 or sm['sealed_2026_rows_read'] != 0):
        raise safety.MLFailure('invalid_input')
    from ..model_diagnostics import development_frame
    with (c.S6 / 'predictions.local.csv').open(encoding='utf-8-sig', newline='') as stream:
        old = development_frame(stream)
    old = old.loc[old.date.str[:4] == '2025'].sort_values('entry_key').reset_index(drop=True)
    targets = set(old.entry_key); races = set(old.entry_key.str[:16])
    buffers = {name: [] for name in c.PARTS}
    for entry in fm['years']:
        if entry['year'] not in (2022, 2023, 2024, 2025):
            continue
        f = read_frame(source(c.S2 / 'features' / entry['file'], entry['sha256']))
        # Date guard runs before interpreting labels; no 2026 source is opened.
        if not (f.date.str[:4] == str(entry['year'])).all():
            raise safety.MLFailure('invalid_split')
        for name, span in p['spans'].items():
            selected = f.loc[(f.date >= span['start']) & (f.date <= span['end']) & (f.eligible == 1)].copy()
            if name == 'evaluation':
                selected = selected.loc[selected.entry_key.str[:16].isin(races)].copy()
            if len(selected):
                buffers[name].append(selected)
        check_time()
    parts = {name: pd.concat(value, ignore_index=True).sort_values('entry_key').reset_index(drop=True)
             for name, value in buffers.items() if value}
    if set(parts) != set(c.PARTS) or set(parts['evaluation'].entry_key) != targets:
        raise safety.MLFailure('invalid_identity')
    wanted = set()
    for name, f in parts.items():
        span = p['spans'][name]
        if (len(f), f.entry_key.str[:16].nunique()) != (span['eligible_runners'], span['eligible_races']):
            raise safety.MLFailure('invalid_identity')
        cap = 130000 if name == 'train' else 30000 if name == 'evaluation' else 10000
        if len(f) > cap:
            raise safety.MLFailure('input_limit')
        validate_roster(f, with_labels=False); wanted.update(f.entry_key)
    wanted_months = {key[:6] for key in wanted}; found = {}
    wanted_rosters = defaultdict(set)
    for key in wanted:
        wanted_rosters[key[:16]].add(key)
    source_races = {race: set() for race in wanted_rosters}
    for entry in sm['months']:
        if entry['file'][:6] not in wanted_months:
            continue
        path = source(c.S2 / 'source' / entry['file'], entry['sha256'])
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = ''.join(row[k] for k in KEY_FIELDS)
                if key[:16] in source_races:
                    if key in source_races[key[:16]]:
                        raise safety.MLFailure('duplicate_entry')
                    source_races[key[:16]].add(key)
                if key not in wanted:
                    continue
                if key in found:
                    raise safety.MLFailure('duplicate_entry')
                rank = row['finish_position']; count = row['actual_runners']
                if not re.fullmatch('[0-9]{2}', rank) or not re.fullmatch('[0-9]{2}', count):
                    raise safety.MLFailure('invalid_input')
                found[key] = (int(rank), int(count))
        check_time()
    if set(found) != wanted or source_races != wanted_rosters:
        raise safety.MLFailure('invalid_identity')
    # No target, final odds or result fields are added to the feature specification.
    result = {'version': c.VERSION, 'feature_spec': features.spec(('A', 'B', 'C', 'D')),
              'history_lag_days': 7, 'source_references': references, 'parts': {},
              'sealed_2026_rows_read': 0, 'Stage7_results_read': 0, 'database_connections': 0,
              'label_join': 'exact_18_digit_entry_key;complete_source_races;no_dropped_target_rows'}
    for name, f in parts.items():
        f['finish_position'] = [found[k][0] for k in f.entry_key]
        f['actual_starters'] = [found[k][1] for k in f.entry_key]
        validate_roster(f)
        if name == 'evaluation':
            if not (f.winner_label.to_numpy(float) == old.winner_label.to_numpy(float)).all():
                raise safety.MLFailure('invalid_identity')
            for col in ('H', 'raw_market'):
                f[col] = old[col].to_numpy(float)
        path = safety.local_path(directory / (name + '.local.csv'), root=c.ROOT)
        with path.open('x', encoding='utf-8', newline='') as out:
            f.to_csv(out, index=False)
        result['parts'][name] = {'file': path.name, 'sha256': safety.digest(path, root=c.ROOT),
            'rows': len(f), 'races': int(f.entry_key.str[:16].nunique()),
            'start': str(f.date.min()), 'end': str(f.date.max()),
            'feature_missing': {n: int(f[n].isna().sum()) for n in features.ALL_FEATURES}}
    result['preparation_seconds'] = time.monotonic() - began
    c.write(directory, 'data_manifest.json', result)
    return {name: {'races': item['races'], 'rows': item['rows']} for name, item in result['parts'].items()}
