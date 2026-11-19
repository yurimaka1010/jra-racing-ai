"""No-fit, local-only importance and auxiliary metrics for frozen development assets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timezone
from html import escape
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .ml import safety
from .keiba_lab_export import BASE, S2, S6, guard_day, identity, numeric, MODEL_COLUMNS
from .stage5 import metrics
from .win_sim.core import PredictionRace

ROOT = BASE / 'model-diagnostics'
S4 = BASE / 'stage4-development/evaluations/2f26b6573d2249d38ad96b4456f89c0f'
S5 = BASE / 'stage5-development/evaluations/598c62b4dbe54142a2abe4a98b435b12'
PROJECT = Path(__file__).resolve().parents[2]
MODELS = ('H', 'raw_market', 'ABCD', 'H_style', 'H_pace')
ROLES = {'H': 'frozen_control', 'raw_market': 'no_fit_reference', 'ABCD': 'frozen_ability_reference',
         'H_style': 'control_not_promoted', 'H_pace': 'not_retained'}


def binary_metrics(labels, decisions):
    y = np.asarray(labels); selected = np.asarray(decisions)
    if (y.ndim != 1 or not len(y) or selected.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isin(selected, [0, 1]).all()):
        raise safety.MLFailure('invalid_input')
    y = y.astype(bool); selected = selected.astype(bool)
    tp = int(np.count_nonzero(y & selected)); fp = int(np.count_nonzero(~y & selected))
    tn = int(np.count_nonzero(~y & ~selected)); fn = int(np.count_nonzero(y & ~selected))
    return {'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn,
            'predicted_positive': tp + fp, 'actual_positive': tp + fn, 'runners': len(y),
            'accuracy': (tp + tn) / len(y), 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'zero_denominator': 'null_not_zero'}


def top1_decisions(keys, probabilities):
    """One per race; tie break uses horse number, never the result label."""
    p = np.asarray(probabilities, float)
    if len(keys) != len(p) or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise safety.MLFailure('invalid_input')
    groups = defaultdict(list); seen = set()
    for i, key in enumerate(keys):
        if not re.fullmatch(r'[0-9]{18}', str(key)) or key in seen:
            raise safety.MLFailure('invalid_identity')
        seen.add(key); groups[key[:16]].append(i)
    output = np.zeros(len(keys), int)
    for indices in groups.values():
        if len(indices) < 2 or abs(float(p[indices].sum()) - 1) > 1e-12:
            raise safety.MLFailure('invalid_input')
        i = min(indices, key=lambda j: (-p[j], int(keys[j][-2:])))
        output[i] = 1
    return output


def importance(booster, metadata):
    """Read saved trees only. Includes every feature; no selection or mutation."""
    names = metadata.get('features')
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or any(not isinstance(n, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,100}', n) for n in names)
            or list(booster.feature_name()) != names):
        raise safety.MLFailure('invalid_model')
    gain = np.asarray(booster.feature_importance(importance_type='gain'), float)
    split = np.asarray(booster.feature_importance(importance_type='split'), float)
    if (gain.shape != (len(names),) or split.shape != gain.shape or not np.isfinite(gain).all()
            or not np.isfinite(split).all() or (gain < 0).any() or (split < 0).any()
            or not np.equal(split, np.floor(split)).all()):
        raise safety.MLFailure('invalid_model')
    total = float(gain.sum()); order = sorted(range(len(names)), key=lambda i: (-gain[i], i))
    rank = {index: place for place, index in enumerate(order, 1)}
    return [{'feature': name, 'feature_index': i, 'gain': float(gain[i]), 'split_count': int(split[i]),
             'gain_share': float(gain[i] / total) if total > 0 else None, 'gain_rank': rank[i]}
            for i, name in enumerate(names)]


def development_frame(stream):
    reader = csv.DictReader(stream)
    required = {'entry_key', 'date', 'winner_label', 'evaluation_year', *MODEL_COLUMNS,
                *(v for v in MODEL_COLUMNS.values() if v)}
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required <= set(reader.fieldnames):
        raise safety.MLFailure('invalid_schema')
    rows = []; seen = set()
    for i, row in enumerate(reader):
        if i >= 100_000:
            raise safety.MLFailure('input_limit')
        identity(row['entry_key'], row['date'])  # Forbid sealed/forward periods before interpreting labels.
        if row['entry_key'] in seen:
            raise safety.MLFailure('duplicate_entry')
        seen.add(row['entry_key'])
        label = numeric(row['winner_label'])
        if label not in (0, 1) or row['evaluation_year'] != row['date'][:4]:
            raise safety.MLFailure('invalid_identity')
        item = {k: row[k] for k in ('entry_key', 'date')}; item['winner_label'] = int(label)
        for name in MODEL_COLUMNS:
            item[name] = numeric(row[name])
        rows.append(item)
    if not rows:
        raise safety.MLFailure('insufficient_history')
    f = pd.DataFrame(rows)
    for _, part in f.groupby(f.entry_key.str[:16], sort=False):
        if len(part) < 2 or part.winner_label.sum() != 1:
            raise safety.MLFailure('invalid_identity')
        if any(abs(float(part[n].sum()) - 1) > 1e-12 for n in MODEL_COLUMNS):
            raise safety.MLFailure('invalid_input')
    return f


def predictions(frame, column):
    output = []
    for race, part in frame.groupby(frame.entry_key.str[:16], sort=True):
        pieces = (race[:4], race[4:8], race[8:10], race[10:12], race[12:14], race[14:16])
        output.append(PredictionRace(pieces, dict(zip(part.entry_key.str[-2:], part[column].astype(float)))))
    return output


def measure(frame, *, model_names=MODELS):
    results = {}
    for name in model_names:
        p = frame[name].to_numpy(float); labels = frame.winner_label.to_numpy(int)
        before = metrics.point_only(frame, predictions(frame, name))
        results[name] = {'existing_metrics': before,
            'probability_ge_0_5': binary_metrics(labels, p >= .5),
            'race_top1_horse_number_tie_break': binary_metrics(labels, top1_decisions(frame.entry_key.tolist(), p)),
            'role': ROLES[name]}
    for value in results.values():
        value['difference_from_H'] = {k: value['existing_metrics'][k] - results['H']['existing_metrics'][k] for k in metrics.NAMES}
    return results


def assert_matches(actual, expected):
    maximum = 0.
    for name, got in actual.items():
        old = expected[name]; got = got['existing_metrics']
        for key in (*metrics.NAMES, 'races', 'runners'):
            delta = abs(got[key] - old[key]); maximum = max(maximum, delta)
            if delta > 1e-12:
                raise safety.MLFailure('artifact_invalid')
    return maximum


def load_sources(reference):
    report = safety.read_json(reference(S6 / 'development_report.json'), root=BASE)
    if (report.get('2026_data_processed') is not False or report.get('run_id') != S6.name
            or report.get('version') != 'stage6_previous_corner_field_interaction_v1'):
        raise safety.MLFailure('invalid_split')
    path = reference(S6 / 'predictions.local.csv', report['prediction_detail']['sha256'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        f = development_frame(stream)
    if (len(f), f.entry_key.str[:16].nunique()) != (87601, 6373):
        raise safety.MLFailure('invalid_identity')
    expected = safety.read_json(reference(S5 / 'baseline.json'), root=BASE)
    refs = safety.read_json(reference(S4 / 'references.json'), root=BASE)
    ability = {}; specs = []
    for year in (2023, 2024, 2025):
        for kind, label in (('market_control', 'H'), ('ability', 'ABCD')):
            path = S2 / 'trials' / f'{year}_annual_{kind}.json'
            if str(path) not in refs:
                raise safety.MLFailure('artifact_invalid')
            trial = safety.read_json(reference(path, refs[str(path)]), root=BASE)
            specs.append({'label': f'{label}_{year}', 'model_id': trial['model_id'],
                'directory': S2 / 'models' / trial['model_id'], 'metadata_sha256': trial['model_manifest_sha256'],
                'role': ROLES[label]})
        path = S2 / 'trials' / f'{year}_annual_ability.predictions.local.json'
        if str(path) not in refs:
            raise safety.MLFailure('artifact_invalid')
        document = safety.read_json(reference(path, refs[str(path)]), root=BASE)
        targets = set(f.loc[f.date.str[:4] == str(year), 'entry_key'].str[:16])
        matched = set()
        for part in document:
            for row in part:
                race = ''.join(row['race_key'])
                day = f'{race[:4]}-{race[4:6]}-{race[6:8]}'
                guard_day(day)
                if day[:4] != str(year):
                    raise safety.MLFailure('invalid_split')
                if race not in targets:
                    continue
                if race in matched:
                    raise safety.MLFailure('duplicate_entry')
                matched.add(race)
                for horse, value in row['probabilities'].items():
                    key = race + horse; identity(key, day)
                    if key in ability:
                        raise safety.MLFailure('duplicate_entry')
                    ability[key] = numeric(value)
        if matched != targets:
            raise safety.MLFailure('invalid_identity')
    if set(ability) != set(f.entry_key):
        raise safety.MLFailure('invalid_identity')
    f['ABCD'] = f.entry_key.map(ability)
    inventory = safety.read_json(reference(S6 / 'experiment_inventory.json'), root=BASE)
    for record in inventory['records']:
        if record['status'] != 'complete' or record['config'] not in ('H_style', 'H_pace') or record['year'] not in (2023, 2024, 2025):
            raise safety.MLFailure('invalid_model')
        specs.append({'label': f"{record['config']}_{record['year']}", 'model_id': record['model_id'],
            'directory': S6 / 'models' / record['model_id'], 'metadata_sha256': record['model_sha256'],
            'role': ROLES[record['config']]})
    # Older ability models are importance-only: never claim they predicted this cohort.
    for label, model_id in (('legacy_V03', '1872a2db908b417a9d982b8836377fa6'),
                            ('best242', 'a8178c2b7523499f9463c1743b41c715')):
        specs.append({'label': label, 'model_id': model_id, 'directory': BASE / 'ml-v01/models' / model_id,
                      'metadata_sha256': None, 'role': 'importance_only_no_cohort_comparison'})
    return f, specs, report, expected


def model_importance(spec, reference):
    import lightgbm as lgb
    meta = safety.read_json(reference(spec['directory'] / 'model.json', spec['metadata_sha256']), root=BASE)
    def dates(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from dates(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from dates(child)
        elif isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            yield date.fromisoformat(value)
    found = list(dates(meta.get('split', {})))
    if not found or max(found) >= date(2026, 1, 1):
        raise safety.MLFailure('invalid_split')
    path = spec['directory'] / 'win.lgb.txt'
    if path.name not in meta.get('artifacts', {}):
        raise safety.MLFailure('invalid_model')
    path = reference(path, meta['artifacts'][path.name])
    booster = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
    rows = importance(booster, meta)
    offset = meta.get('feature_spec', {}).get('mode') == 'market_offset'
    return {'label': spec['label'], 'model_id': spec['model_id'], 'role': spec['role'],
            'feature_count': len(rows), 'saved_iterations': booster.current_iteration(),
            'calibration': meta.get('calibration', {'method': 'stored_legacy_output_not_recalibrated'}),
            'external_offset': offset, 'scope': 'residual_trees_only_external_market_offset_excluded' if offset else 'saved_trees_only',
            'rows': rows}


def write_csv(path, rows, fields):
    path = safety.local_path(path, root=ROOT)
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def render(result):
    parts = ['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>保存モデル診断</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f7;color:#172535;max-width:1200px;margin:auto;padding:24px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce3e9;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.bar{display:grid;grid-template-columns:minmax(220px,2fr) 3fr 80px;align-items:center;gap:10px;margin:7px 0;font-size:13px}.track{background:#e7eef4;height:14px}.fill{background:#267f9b;height:100%}.note{color:#526577}code{overflow-wrap:anywhere}@media(max-width:650px){body{padding:10px}.bar{grid-template-columns:1fr}section{padding:14px}}</style>',
        '<h1>保存モデルの重要度と補助分類指標</h1>',
        '<p>2023〜2025年1〜9月の再利用した開発評価。6,373レース・87,601頭。再学習なし。Stage 7途中成績・封印2026年は不使用。</p>',
        '<section><h2>同じ評価対象での比較</h2><p class="note">数値は保存予測の再集計。0.5判定と1位選択は別の規則です。既存の同率按分1位的中率は変更しません。</p><div class="scroll"><table><thead><tr><th>モデル</th><th>レースLL</th><th>レース内AUC</th><th>馬Brier</th><th>既存1位的中率%</th><th>0.5陽性頭数</th><th>0.5適合率</th><th>1位選択F1</th></tr></thead><tbody>']
    def number(v, digits=5):
        return '未定義' if v is None else f'{v:.{digits}f}'
    for name in MODELS:
        row = result['metrics']['pooled'][name]; v = row['existing_metrics']; half = row['probability_ge_0_5']; top = row['race_top1_horse_number_tie_break']
        parts.append('<tr><td>' + escape(name) + '</td>' + ''.join('<td>' + number(x) + '</td>' for x in
            (v['race_log_loss'], v['within_race_auc'], v['brier_score'], v['top1_accuracy_percent']))
            + f'<td>{half["predicted_positive"]}</td><td>{number(half["precision"])}</td><td>{number(top["f1"])}</td></tr>')
    parts += ['</tbody></table></div><p class="note">H_styleは個体対照、H_paceは不採用。旧V0.3・best242は以下の重要度だけを表示し、この評価期間で予測したモデルとは扱いません。全指標・混同行列・年別値はJSON/CSVに保存。</p></section>']
    for model in result['importance']:
        parts += ['<section><h2>' + escape(model['label']) + '</h2><p><code>' + escape(model['model_id']) + '</code></p>',
            '<p class="note">' + ('木の残差部分のgain割合。外部の市場offsetの寄与は含みません。' if model['external_offset'] else '保存された木のgain割合。')
            + '重要度は因果効果・将来の有用性を保証する値ではありません。全特徴量を表示し、自動削除しません。</p>']
        for row in sorted(model['rows'], key=lambda r: r['gain_rank']):
            percent = 100 * row['gain_share'] if row['gain_share'] is not None else 0.
            parts.append('<div class="bar"><span>' + escape(row['feature']) + '</span><div class="track"><div class="fill" style="width:'
                         + f'{percent:.8f}%' + '"></div></div><span>' + (f'{percent:.2f}%' if row['gain_share'] is not None else '未定義') + '</span></div>')
        parts.append('</section>')
    return '\n'.join(parts + ['<p>ローカル閲覧専用。外部通信・CDN・スクリプトなし。</p></html>'])


def verify_protection(folder):
    from .stage4.store import task_configuration
    baseline = safety.read_json(folder / 'protection_before.json', root=ROOT)
    import hashlib
    for name, expected in baseline['existing_project_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise safety.MLFailure('artifact_invalid')
    for name, expected in baseline['protected_hashes'].items():
        if safety.digest(Path(name), root=BASE) != expected:
            raise safety.MLFailure('artifact_invalid')
    current = task_configuration(); before = baseline['tasks']
    same = current == before if current['status'] == before['status'] == 'observed' else None
    if same is False:
        raise safety.MLFailure('artifact_invalid')
    return {'unchanged_existing_project_files': len(baseline['existing_project_hashes']),
            'unchanged_protected_artifacts': len(baseline['protected_hashes']), 'task_definitions_unchanged': same,
            'Stage7_recorder_and_plan_unchanged': True, 'new_files_only': True}


def initialize():
    import hashlib
    from .stage4.store import task_configuration
    folder = safety.new_artifact('evaluations', root=ROOT)
    paths = []
    for name in ('src', 'scripts', 'tests', 'docs'):
        paths.extend(p for p in (PROJECT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in (PROJECT / n for n in ('README.md', 'requirements.txt', 'requirements-ml.txt',
                                                   'requirements-ml-lock.txt', 'pyproject.toml')) if p.is_file())
    protected = [S2 / 'models/4efe34cdef5b4da9a33c254c4586a2bf' / n for n in ('model.json', 'win.lgb.txt')]
    protected.extend(BASE / 'stage7-development/evaluations/a1b9696a04d94adfa3d42e288d45de67' / n
                     for n in ('forward_plan.json', 'plan_commit.json'))
    safety.write_json(folder / 'protection_before.json', {
        'existing_project_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'protected_hashes': {str(p): safety.digest(p, root=BASE) for p in protected}, 'tasks': task_configuration(),
        'added_files_permitted': True, 'Stage7_outcomes_read': False}, root=ROOT)
    return {'status': 'ok', 'kind': 'diagnostics_initialized', 'run_id': folder.name, 'directory': str(folder), 'fits': 0}


def run(folder):
    folder = safety.local_path(folder, root=ROOT)
    protection = verify_protection(folder)
    references = {}
    def reference(path, expected=None):
        path = safety.local_path(path, root=BASE, exists=True)
        got = safety.digest(path, root=BASE)
        if expected is not None and got != expected:
            raise safety.MLFailure('artifact_invalid')
        references[str(path)] = got
        return path
    frame, specs, source, baseline = load_sources(reference)
    pooled = measure(frame); annual = {}; maximum = 0.
    for year in (2023, 2024, 2025):
        annual[str(year)] = measure(frame.loc[frame.date.str[:4] == str(year)])
    for tag, values in [('pooled', pooled)] + list(annual.items()):
        old_scope = source['pooled'] if tag == 'pooled' else source['years'][tag]
        expected = dict(old_scope['metrics'])
        expected['ABCD'] = (baseline['pooled'] if tag == 'pooled' else baseline['years'][tag])['metrics']['ABCD']
        maximum = max(maximum, assert_matches(values, expected))
    imported = [model_importance(spec, reference) for spec in specs]
    result = {'version': 'no_fit_model_diagnostics_v1', 'run_id': folder.name,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'metrics': {'pooled': pooled, 'years': annual},
        'importance': imported, 'existing_metrics_max_difference': maximum,
        'fixed_rules': {'probability_threshold': .5, 'positive_label': 'winner', 'undefined_ratio': None,
                        'top1_ties': 'horse_number_ascending', 'existing_top1_ties': 'fractional_credit'},
        'development_reuse_not_final_test': True, 'new_real_data_fits': 0, 'new_synthetic_fits': 0,
        'sealed_2026_inputs': 0, 'Stage7_outcome_inputs': 0, 'database_operations': 0,
        'references_sha256': references, 'preservation': protection,
        'library_versions': {n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'lightgbm', 'scikit-learn')}}
    for name, expected in references.items():
        if safety.digest(Path(name), root=BASE) != expected:
            raise safety.MLFailure('artifact_invalid')
    result['preservation'] = verify_protection(folder)
    safety.write_json(folder / 'diagnostics.json', result, root=ROOT)
    rows = [{**{k: model[k] for k in ('label', 'model_id', 'role', 'scope')}, **row}
            for model in imported for row in model['rows']]
    write_csv(folder / 'feature_importance.csv', rows, tuple(rows[0]))
    rows = []
    for scope, values in [('pooled', pooled)] + list(annual.items()):
        for name, data in values.items():
            for rule in ('probability_ge_0_5', 'race_top1_horse_number_tie_break'):
                rows.append({'scope': scope, 'model': name, 'rule': rule, **data[rule]})
    write_csv(folder / 'classification_metrics.csv', rows, tuple(rows[0]))
    page = safety.local_path(folder / 'diagnostics.html', root=ROOT)
    with page.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(render(result))
    return {'status': 'ok', 'kind': 'no_fit_model_diagnostics', 'run_id': folder.name,
        'races': 6373, 'runners': 87601, 'comparison_models': len(MODELS), 'importance_models': len(imported),
        'existing_metrics_max_difference': maximum, 'new_real_data_fits': 0, 'report_directory': str(folder),
        'preservation': result['preservation'],
        'summary': {name: {k: pooled[name]['existing_metrics'][k] for k in metrics.NAMES} for name in MODELS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen-model importance and supplementary metrics. No fit or DB.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-id')
    group.add_argument('--init', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.init:
            print(json.dumps(initialize(), ensure_ascii=True, allow_nan=False))
            return 0
        if not re.fullmatch(r'[a-f0-9]{32}', args.run_id):
            raise safety.MLFailure('invalid_input')
        print(json.dumps(run(ROOT / 'evaluations' / args.run_id), ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        reason = exc.code if isinstance(exc, safety.MLFailure) else 'internal_error'
        print(json.dumps({'status': 'stopped', 'reason': reason, 'stage': 'model_diagnostics'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
