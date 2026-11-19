"""Rebuild all years from immutable raw caches, not already contaminated feature CSVs."""
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from uuid import uuid4
import csv
import time

from ..ml import safety
from ..d39_generation import data as old_data
from ..research import model as matrices, contracts as legacy_contract
from ..objective_pilot import data as cache
from . import contracts as c
from .features import Builder


def target_query(schema, start, end, plan):
    """Optional DB regeneration path: whole-race anti-join, never a per-horse filter.

    History preparation deliberately reads the complete raw cache and applies the
    same Python rule, so a scratched horse cannot disappear before inspection.
    """
    from ..core import RACE_KEY
    from ..db_connection import validate_audit_sql
    if 'ijo_kubun_code' not in schema['se']:
        raise safety.MLFailure('invalid_schema')
    sql, args, fields = old_data.source_query(schema, start, end, plan)
    equal = ' AND '.join('bad.' + safety.identifier(k) + '=ra.' + safety.identifier(k) for k in RACE_KEY)
    anti = (' AND NOT EXISTS (SELECT 1 FROM ONLY public.jvd_se bad WHERE ' + equal +
            " AND (bad.ijo_kubun_code::text IS DISTINCT FROM '0' OR bad.data_kubun::text IS DISTINCT FROM '7')) ")
    if sql.count(' ORDER BY ') != 1:
        raise safety.MLFailure('artifact_invalid')
    sql = sql.replace(' ORDER BY ', anti + ' ORDER BY ')
    validate_audit_sql(sql)
    return sql, args, fields


def validate_frame(frame):
    old_data.validate_frame(frame)
    if (frame.date < '2022-01-01').any() or (frame.date > '2026-10-18').any():
        raise safety.MLFailure('invalid_split')


def matrix(frame, spec, categories):
    import pandas as pd
    if spec != c.spec() or set(categories) != set(spec['categorical_features']):
        raise safety.MLFailure('invalid_model')
    base = legacy_contract.spec(('A', 'B', 'C', 'D'))
    result = matrices.matrix(frame, base, {k: categories[k] for k in base['categorical_features']})
    for name in c.EXTRA:
        if name in categories:
            cats = categories[name]
            if cats[:2] != ['__MISSING__', '__UNKNOWN__'] or len(cats) != len(set(cats)):
                raise safety.MLFailure('invalid_model')
            values = ['__MISSING__' if pd.isna(v) else 'v:' + str(v) if 'v:' + str(v) in cats else '__UNKNOWN__'
                      for v in frame[name]]
            result[name] = pd.Categorical(values, categories=cats)
        else:
            result[name] = pd.to_numeric(frame[name], errors='raise').astype('float64')
    return result.loc[:, spec['features']]


def prepare():
    import pandas as pd
    directory = safety.new_artifact('evaluations', root=c.ROOT)
    plan = c.plan()
    c.write(directory / 'plan.json', plan)
    c.write(directory / 'plan_commit.json', {'sha256': c.sha(directory / 'plan.json'),
        'fits_authorized': False, 'approval_required': 'human_user;max9;exact_plan_sha256'})
    c.write(directory / 'preservation_before.json', c.preservation())
    builder = Builder(); frames = []; stats = defaultdict(Counter)
    rejected = set(); normal = set()
    exclusions = defaultdict(Counter); source_counts = Counter(); began = time.monotonic()
    for rows in old_data.iter_days(c.PARENT):
        if time.monotonic() - began > 900:
            raise safety.MLFailure('query_timeout')
        day = date(int(rows[0]['kaisai_nen']), int(rows[0]['kaisai_tsukihi'][:2]), int(rows[0]['kaisai_tsukihi'][2:]))
        if str(day) > plan['latest_source_date']:
            raise safety.MLFailure('invalid_split')
        x, meta = builder.transform_day(rows)
        if day < date(2022, 1, 1):
            continue
        seen = set()
        for values, m in zip(x, meta):
            key = ''.join(m['key']); race = key[:16]; year = m['date'][:4]
            if race not in seen:
                seen.add(race); stats[year]['source_races'] += 1
                if m.get('exclusion_reasons'):
                    rejected.add(race)
                    stats[year]['excluded_irregular_or_invalid_races'] += 1
                    exclusions[year].update(m['exclusion_reasons'])
                elif not m['eligible']:
                    stats[year]['missing_required_declared_fields_races'] += 1
                else:
                    normal.add(race)
                    stats[year]['retained_races'] += 1
            stats[year]['source_rows'] += 1
            if m['eligible']:
                stats[year]['retained_rows'] += 1
                frames.append(dict(entry_key=key, date=m['date'], winner_label=m['winner_label'], **values))
        source_counts['raw_rows'] += len(rows)
    all_rows = pd.DataFrame(frames).sort_values('entry_key').reset_index(drop=True)
    if len(all_rows) > safety.MAX_ROWS:
        raise safety.MLFailure('input_limit')
    validate_frame(all_rows)
    admitted = set(all_rows.entry_key.str[:16])
    if admitted & rejected or not admitted <= normal:
        raise safety.MLFailure('artifact_invalid')
    parts = {}
    for name, span in plan['spans'].items():
        frame = all_rows.loc[(all_rows.date >= span['start']) & (all_rows.date <= span['end'])].copy()
        validate_frame(frame)
        path = directory / (name + '.local.csv')
        with path.open('x', encoding='utf-8', newline='') as stream:
            frame.to_csv(stream, index=False)
        parts[name] = {'file': path.name, 'sha256': c.sha(path), 'rows': len(frame),
            'races': int(frame.entry_key.str[:16].nunique()), 'start': str(frame.date.min()),
            'end': str(frame.date.max()), 'missing': {f: int(frame[f].isna().sum()) for f in c.spec()['features']}}
    # Independently re-read the raw flags, not the feature builder's own mask.
    from .features import race_reasons
    independently_seen = set(); independently_invalid = set()
    for raw_rows in old_data.iter_days(c.PARENT):
        grouped = defaultdict(list)
        for r in raw_rows:
            from ..research.features import legacy
            grouped[''.join(legacy.entry_identity(r)[0][:-1])].append(r)
        for key, group in grouped.items():
            if key not in admitted:
                continue
            independently_seen.add(key)
            if race_reasons(group):
                independently_invalid.add(key)
    if independently_seen != admitted or independently_invalid:
        raise safety.MLFailure('artifact_invalid')
    for item in parts.values():
        exported = cache.read_frame(directory / item['file'])
        if not set(exported.entry_key.str[:16]) <= admitted or len(exported) != item['rows']:
            raise safety.MLFailure('artifact_invalid')
    integrity = {'rechecked_admitted_races': len(independently_seen),
                 'irregular_race_contamination': len(independently_invalid),
                 'all_parts_reloaded': True, 'whole_race_filter_verified': True}
    c.write(directory / 'clean_race_verification.json', integrity)
    manifest = {'version': c.VERSION, 'dataset_id': uuid4().hex, 'parts': parts,
        'source_manifest_sha256': c.sha(c.PARENT / 'source_manifest.json'),
        'old_source_manifest_sha256': c.sha(c.old.S2 / 'source_manifest.json'),
        'source_snapshots_end': plan['latest_source_date'], 'feature_spec': c.spec(),
        'by_year': {y: dict(v) for y, v in stats.items()},
        'exclusion_reasons_by_year_overlapping': {y: dict(v) for y, v in exclusions.items()},
        'ordinary_rank_history_only': True, 'all_raw_years_regenerated': True,
        'actual_start_dates_include_2101_04567_without_their_abnormal_ranks': True,
        'no_Stage7_results_read': True, 'old_feature_files_reused_as_training_input': False}
    manifest['clean_race_verification'] = integrity
    c.write(directory / 'data_manifest.json', manifest)
    preservation = c.verify_preservation(c.read(directory / 'preservation_before.json'))
    result = {'status': 'prepared_awaiting_real_fit_approval', 'run_id': directory.name,
        'dataset_id': manifest['dataset_id'], 'feature_count': len(c.spec()['features']),
        'parts': {n: {k: v[k] for k in ('rows', 'races', 'start', 'end')} for n, v in parts.items()},
        'by_year': manifest['by_year'], 'exclusion_reasons': manifest['exclusion_reasons_by_year_overlapping'],
        'elapsed_seconds': time.monotonic() - began, 'real_data_fits': 0, 'preservation': preservation}
    c.write(directory / 'preparation.json', result)
    return result


def load_part(directory, name):
    p = c.read(directory / 'plan.json'); manifest = c.read(directory / 'data_manifest.json')
    if p['version'] != c.VERSION or manifest['feature_spec'] != c.spec():
        raise safety.MLFailure('artifact_invalid')
    item = manifest['parts'][name]; path = directory / item['file']
    if c.sha(path) != item['sha256']:
        raise safety.MLFailure('artifact_invalid')
    frame = cache.read_frame(path)
    if len(frame) != item['rows'] or not frame.date.between(p['spans'][name]['start'], p['spans'][name]['end']).all():
        raise safety.MLFailure('invalid_split')
    validate_frame(frame)
    return frame
