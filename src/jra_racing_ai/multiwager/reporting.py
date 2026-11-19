"""Saved predictions on per-wager common official-target cohorts; 0 extra fits."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from ..ml import safety
from ..objective_pilot.data import read_frame
from . import contracts as c, lab, training
from .scoring import Scores


def compare(directory):
    training.approved(directory)
    complete = c.read(directory, 'training_complete.json')
    if complete.get('status') != 'complete' or complete.get('fits_consumed') != 2:
        raise safety.MLFailure('artifact_invalid')
    manifest = c.read(directory, 'training_inputs.json'); part = manifest['parts']['evaluation']
    path = Path(part['file']); lab.check(path, part['sha256']); frame = read_frame(path)
    for day in frame.date.unique(): c.day(day)
    if not frame.date.str.startswith('2025-').all(): raise safety.MLFailure('invalid_split')
    races = set(frame.entry_key.str[:16]); rosters = {r: sorted(f.entry_key.str[-2:]) for r,f in frame.groupby(frame.entry_key.str[:16])}
    target_path = Path(manifest['official_targets_file'])
    lab.check(target_path, manifest['official_HR_source_sha256']); targets = lab.read(target_path)['records']
    predictions = defaultdict(dict); source_sha = {}; models = {}
    for family in c.FAMILIES:
        path = directory/(family+'.predictions.local.jsonl'); source_sha[str(path)] = lab.digest(path)
        measured = c.read(directory, family+'.evaluation.json'); model_id = measured['model']['model_id']
        models[family] = c.read(directory/'models'/model_id,'model.json')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); c.identity(row['race_id'], row['date'])
                if row['race_id'] not in races or row['model_id'] != model_id or row['race_id'] in predictions[family]:
                    raise safety.MLFailure('invalid_identity')
                predictions[family][row['race_id']] = row['probabilities']
    source = lab.BASELINE/'predictions.local.jsonl'; source_sha[str(source)] = lab.digest(source)
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); c.identity(row['race_id'], row['date'])
            if row['race_id'] in races and row['model'] in lab.SERIES:
                if row['race_id'] in predictions[row['model']]: raise safety.MLFailure('duplicate_entry')
                predictions[row['model']][row['race_id']] = row['probabilities']
    measured = {}; coverage = {}; differences = {}
    for wager in c.WAGERS:
        names = list(lab.SERIES) + ([c.FAMILIES[0]] if wager=='place' else []) + [c.FAMILIES[1]]
        scores = {name: Scores() for name in names}; reasons = Counter(); eligible = 0
        for race in sorted(races):
            event = targets.get(race, {}).get(wager, {})
            if event.get('target_status') != 'confirmed':
                reasons[event.get('reason') or 'official_targets_missing'] += 1; continue
            if any(race not in predictions[n] or wager not in predictions[n][race] for n in names):
                reasons['common_saved_prediction_missing'] += 1; continue
            sets = [set(predictions[n][race][wager]) for n in names]
            if any(s != sets[0] for s in sets) or not set(event['official_targets']) <= sets[0]:
                raise safety.MLFailure('invalid_identity')
            labels = {k: int(k in event['official_targets']) for k in sets[0]}
            for name in names: scores[name].add(predictions[name][race][wager], labels)
            eligible += 1
        coverage[wager] = {'source_evaluation_races': len(races), 'common_official_races': eligible,
            'excluded_reasons': dict(reasons), 'same_target_sets_for_all_models': True}
        measured[wager] = {name: score.result() for name,score in scores.items()}
        for result in measured[wager].values():
            result.update(label_basis='verified_normal_HR_current_snapshot_targets',
                ROI=None, ROI_reason='no_confirmed_per_wager_money_units_or_fixed_decision_time_purchase_policy')
        differences[wager] = {name: {key: measured[wager][name][key]-measured[wager]['H_PL'][key]
            for key in ('binary_log_loss','brier_score','roc_auc','top_probability_target_hit_rate_percent')}
            for name in names if name not in lab.SERIES}
    report = {'kind': 'multiwager_two_fit_development_comparison', 'run_id': directory.name,
        'training_fits': 2, 'additional_comparison_fits': 0, 'development_period': ['2025-01-01','2025-09-30'],
        'population': manifest['population'], 'asof_exclusions': manifest['asof_exclusions'],
        'coverage': coverage, 'metrics': measured, 'new_minus_H_PL': differences,
        'models': {name: {'model_id': meta['model_id'], 'features': len(meta['features']),
            'best_iteration': meta['best_iteration'], 'temperature': meta['calibration']['temperature'],
            'parameters': meta['parameters'], 'calibration': meta['calibration']}
            for name,meta in models.items()},
        'execution': complete, 'source_prediction_sha256': source_sha,
        'official_label_source_sha256': manifest['official_HR_source_sha256'],
        'ROI': None, 'automatic_adoption': False, 'formal_superiority_claim': False,
        'interpretation': 'reused_development_descriptive_comparison_not_independent_final_or_prerace_live',
        '2026_rows_used': 0, 'Stage7_results_used': False,
        'wager_AUC_values_not_ranked_against_each_other': True}
    c.write(directory,'development_comparison.json',report)
    rows=[]
    for wager,results in measured.items():
        for name,result in results.items():
            rows.append({'wager':wager,'model':name,**{k:result[k] for k in
                ('races','targets','binary_log_loss','brier_score','roc_auc','top_probability_target_hit_rate_percent','mean_probability_sum_per_race')}})
    with (directory/'development_comparison.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 複勝・ワイド・馬連：固定2fitの開発比較','',
        '2023〜2024年の学習・早期終了・較正、2025年1〜9月の再利用開発評価。独立最終評価や実際の発走前記録ではありません。',
        '通常HRの的中対象が確認できる共通レース・同一対象集合で比較。金額単位は未確認で、収益・ROIは未算出です。',
        '新規fitは複勝直接1回と共通着順1回だけ。既存H・市場・ABCDのPL参考値は再学習せず保存値を再利用。',
        '直接複勝は各馬の独立周辺確率を較正し、和を1/2/3へ強制しません。共通着順は同着なしPL仮定でペア確率を求めます。','',
        '|券種|モデル|レース|対象|馬/ペア二値LL|Brier|AUC|最上位対象の的中率|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['wager']}|{row['model']}|{row['races']}|{row['targets']}|{row['binary_log_loss']:.6f}|{row['brier_score']:.6f}|{row['roc_auc']:.6f}|{row['top_probability_target_hit_rate_percent']:.2f}%|")
    lines += ['', '確率帯別較正、券種別共通母集団・除外理由、学習時間・温度・木数・モデルIDは development_comparison.json。',
        '券種間は正例定義・標本単位が違うためAUCの大小をモデル優劣として比較しません。',
        '結果を見た追加探索・Tuner・閾値変更は行わず、2モデルを研究用に保存。自動採用はしません。',
        '公式金額・返還等の特殊精算・券種別判断時点価格と固定購入方針は未接続です。未精算を0円負けにしません。',
        '2026年・Stage 7途中結果は使っていません。']
    with (directory/'development_comparison.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return {'status':'ok','fits':2,'coverage':coverage,'models':report['models'],
            'metrics':{w:{n:{k:r[k] for k in ('binary_log_loss','brier_score','roc_auc')} for n,r in v.items()} for w,v in measured.items()},'ROI':None}
