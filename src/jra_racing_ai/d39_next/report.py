"""Anonymous diagnostics; no retrospective training window is promoted to a test."""
from collections import Counter, defaultdict
from pathlib import Path
import csv

from ..d39_generation import data as old_data, learning as old_learning
from ..tuner_pilot import experiment as tuner
from ..model_diagnostics import importance
from . import contracts as c, data, features, learning

DEFINITIONS = {
    'e_days_since_actual_start': 'last known actual start to target date; seven-day lag; abnormal start date only allowed',
    'e_previous_start_gap_days': 'gap between the last two known actual starts',
    'e_recent3_start_gap_mean_days': 'mean of at most three known past start gaps',
    'e_rest_stage': '84day rest; first/second/third/later/none; no history is missing',
    'e_rest_first_past_starts': 'clean historical first-after-rest performance count',
    'e_rest_first_smoothed_finish_percentile': '(sum clean first-after-rest percentiles+3*0.5)/(count+3)',
    'e_rest_stage_smoothed_finish_percentile': 'same smoothing for current rest-stage; only known past results',
    'e_last_start_irregular': 'last actual-start anomaly nonzero; never an ordinary finish rank',
}


def create(directory):
    manifest = c.read(directory / 'data_manifest.json')
    freeze = c.read(directory / 'candidate_freeze.json')
    plan = c.read(directory / 'plan.json')
    bundle = learning.load(Path(freeze['model_path']))
    cal = data.load_part(directory, 'calibration')
    comparison = {'next_D39': learning.point(cal, learning.predict(bundle, cal))}
    previous_cal = old_data.cache.read_frame(c.PARENT / 'calibration.local.csv')
    previous_cal = previous_cal.loc[previous_cal.entry_key.isin(cal.entry_key)].sort_values('entry_key').reset_index(drop=True)
    if previous_cal.entry_key.tolist() != cal.entry_key.tolist():
        raise c.safety.MLFailure('invalid_identity')
    parent_plan = c.read(c.PARENT / 'four_addendum_plan.json')
    for name in ('old_D39', 'new_D39', 'control_D39'):
        path = Path(parent_plan['model_paths'][name])
        if name == 'old_D39':
            old_bundle = tuner.load_model(path); p = tuner.predict(*old_bundle, previous_cal)
        else:
            old_bundle = old_learning.load(path); p = old_learning.predict(old_bundle, previous_cal)
        comparison[name] = learning.point(previous_cal, p)
    # Quantify actual admissions in the prior final CSV, independently of the new mask.
    old_final = old_data.cache.read_frame(c.PARENT / 'final_train.local.csv')
    old_keys = set(old_final.entry_key.str[:16]); old_bad = set(); old_reasons = Counter()
    for raw in old_data.iter_days(c.PARENT):
        groups = defaultdict(list)
        for row in raw:
            key = ''.join(features.legacy.legacy.entry_identity(row)[0][:-1])
            if key in old_keys:
                groups[key].append(row)
        for key, group in groups.items():
            reason = features.race_reasons(group)
            if reason:
                old_bad.add(key); old_reasons.update(reason)
    ranking = importance(*bundle)
    with (directory / 'feature_registry.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('feature', 'definition', 'gain', 'gain_rank', 'split', 'missing', 'missing_percent'))
        writer.writeheader()
        for r in ranking:
            missing = manifest['parts']['final_train']['missing'][r['feature']]
            writer.writerow({'feature': r['feature'], 'definition': DEFINITIONS.get(r['feature'], 'existing D39 formula; clean-race delayed history'),
                'gain': r['gain'], 'gain_rank': r['gain_rank'], 'split': r['split_count'], 'missing': missing,
                'missing_percent': 100 * missing / manifest['parts']['final_train']['rows']})
    result = {'status': 'ok', 'kind': 'new_clean_D39_pre_forward_report', 'run_id': directory.name,
        'model_id': freeze['model_id'], 'model_path': freeze['model_path'], 'feature_count': 47,
        'parts': manifest['parts'], 'by_year': manifest['by_year'],
        'exclusion_reasons': manifest['exclusion_reasons_by_year_overlapping'],
        'old_final_train_rechecked_races': len(old_keys), 'old_final_train_nonconforming_races': len(old_bad),
        'old_final_train_reason_counts_overlapping': dict(old_reasons),
        'new_contamination': manifest['clean_race_verification'],
        'calibration_diagnostic_same_roster': comparison,
        'comparison_limitations': ['calibration-used_not_independent_test', 'training_data_and_formulas_differ',
            'new_clean_rank_history_for_new_model_only', 'H_outcomes_never_joined'],
        'additional_candidates': {'already_present': ['days_since_previous', 'same_distance_mean_finish', 'same_course_mean_finish', 'jockey_smoothed_win_365d'],
            'implemented': list(c.EXTRA), 'deferred': {'course_direction': 'raw track categories already present; no new claimed effect',
                'going': 'final RA value is not proof of prerace observation', 'jockey_condition': 'separate full-population conditional denominator required',
                'pedigree_condition': 'parent linkage/source not confirmed', 'favorite_or_longshot_conditions': 'descriptive forward diagnostic only;no single-race rules'}},
        'preservation': c.verify_preservation(c.read(directory / 'preservation_before.json'))}
    c.write(directory / 'development_report.json', result)
    lines = ['# 次世代D39：異常レース除外・間隔特徴量・8項目有限探索', '',
        f'新モデル `{freeze["model_id"]}`。既存39項目に間隔・休養8項目を加えた47項目。既存4モデルを置換しない。', '',
        '元の全期間生データキャッシュをハッシュ検証して再処理。元DB・旧特徴量CSV・旧モデルは変更しない。',
        'JV-Data仕様2101の1〜7を含むレース、未知区分、頭数不一致、着順不正・同着を全馬除外。',
        '正常着順の履歴、C・D集計も対象レース全体を除く。実際に出走した異常馬は間隔用の日付・異常フラグだけ残す。取消・除外は実出走日としない。',
        f'旧最終学習CSVの再点検：{len(old_keys):,}レース中、今回の正常レース規則に適合しないもの{len(old_bad):,}レース。新データの混入は0件。', '',
        '## データ分割', '', '| 区分 | レース | 頭数 | 実データ期間 |', '|---|---:|---:|---|']
    for name, item in manifest['parts'].items():
        lines.append(f'| {name} | {item["races"]:,} | {item["rows"]:,} | {item["start"]}〜{item["end"]} |')
    lines += ['', '## 同一133レース・1,747頭の較正期間診断', '',
        '温度選択にも使った期間。未使用評価ではなく、改善の正式認定・モデル選択には使わない。既存モデルは元の特徴量生成値で推論。', '',
        '| モデル | レースLL | Brier | AUC | 1位的中率 |', '|---|---:|---:|---:|---:|']
    for name, m in comparison.items():
        lines.append(f'| {name} | {m["race_log_loss"]:.6f} | {m["brier_score"]:.6f} | {m["roc_auc"]:.6f} | {m["top1_accuracy_percent"]:.2f}% |')
    lines += ['', '## 凍結・前向き記録', '',
        '探索8fit＋最終再学習1fitで終了。8項目は全試行で提案し、反復上限と早期終了木数を別に保存。温度は1回だけ適用。',
        '完成後は既存記録器の結果非参照・締切前実観測入力を読み、新モデル専用のCSV・JSON・受領ハッシュを保存。',
        '有効化時点より前のT−10、保存期限超過、元入力欠損は正式予想へ追加しない。既存予想を書き換えない。',
        'KEIBA LAB `/next` で5モデルの予想、D39系4モデルだけの結果・確率品質・単勝仮想精算を表示。Hの結果はDB照合前に除く。',
        '2026年1〜9月は今回承認された新世代学習のみ。旧封印評価・Stage 7の記録や正式条件は不変。',
        '当日の実測性能は正式な発走前保存と公式結果が揃ってから集計。較正診断を改善の証明にしない。', '',
        '## 残る候補', '', '馬場の発走前時点、血統の対応元、騎手の条件別全母数が未確認。追加しない。',
        'クラスは元のカテゴリ特徴量として保持し、人為的な加点・減点や単一レースに合わせたルールは作らない。',
        'going / previous_time_seconds は従来どおり全欠損。未確認の事後値で埋めない。',
        '結果は1候補・有限探索の実測。学習済み期間を未知レースと呼ばず、今日以降の前向き記録で検証する。']
    with (directory / 'development_report.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    return {'status': 'ok', 'model_id': freeze['model_id'],
        'old_nonconforming_races': len(old_bad), 'new_contamination': 0,
        'calibration_diagnostic': {n: {k: m[k] for k in ('races','runners','race_log_loss','brier_score','roc_auc','top1_accuracy_percent')} for n,m in comparison.items()}}
