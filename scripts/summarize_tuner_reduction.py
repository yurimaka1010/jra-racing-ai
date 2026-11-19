"""Summarize completed finite runs; no new evaluation, fitting, or DB reads."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from jra_racing_ai import tuner_reduction as r
from jra_racing_ai.ml import safety
from jra_racing_ai.stage4.store import task_configuration


def csv_file(directory, name, rows):
    with safety.local_path(directory / name, root=r.ROOT).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(identifier):
    directory = r.folder(identifier)
    report = r.read(directory, 'development_report.json'); completion = r.read(directory, 'completion.json')
    freeze = r.read(directory, 'backward_candidate_freeze.json'); plan = r.read(directory, 'plan.json')
    if completion['status'] != 'complete' or not report['best_chosen_before_2025']:
        raise safety.MLFailure('artifact_invalid')
    records = report['selection_records']; c_result = report['Boruta_result']
    rows = []
    for record in records + ([c_result] if c_result else []):
        metrics = record['selection_metrics']
        rows.append({'configuration': record['name'], 'feature_count': record['feature_count'], 'model_id': record['model_id'],
                     'features': ';'.join(record['features']), 'removed_in_order': ';'.join(record['removed_features_in_order']),
                     'selection_2024_race_LL': metrics['race_log_loss'], 'selection_2024_horse_LL': metrics['horse_log_loss'],
                     'selection_2024_Brier': metrics['brier_score'], 'selection_2024_ROC_AUC': metrics['roc_auc'],
                     'selection_2024_within_race_AUC': metrics['within_race_auc'],
                     'selection_2024_top1_percent': metrics['top1_accuracy_percent'],
                     'temperature': record['temperature'], 'best_iteration': record['best_iteration'],
                     'new_actual_LGBM_fits': record['actual_train_calls'], 'new_fit_seconds': record['fit_call_seconds'],
                     'configuration_seconds': record['configuration_wall_seconds'],
                     'selected_B': record['name'] == report['best_backward'],
                     'evaluation_2025_performed': record['name'] in report['evaluation_metrics']})
    csv_file(directory, 'feature_count_results.csv', rows)
    removed = records[-1]['removed_features_in_order']; removal_rows = []
    for i, feature in enumerate(removed):
        source = None if i < 7 else records[i - 6]
        gain = None if source is None else next(v['gain'] for v in source['importance'] if v['feature'] == feature)
        removal_rows.append({'removal_order': i + 1, 'feature': feature,
                             'rule': 'user_fixed_initial_seven' if i < 7 else 'previous_model_min_Gain',
                             'resulting_count': 32 if i < 7 else 38 - i,
                             'source_model_id': source['model_id'] if source else '', 'source_model_Gain': gain})
    csv_file(directory, 'removed_feature_order.csv', removal_rows)
    evaluation_rows = []
    for name, measured in report['evaluation_metrics'].items():
        record = next(v for v in records + ([c_result] if c_result else []) if v['name'] == name)
        evaluation_rows.append({'configuration': name, 'feature_count': record['feature_count'], 'model_id': record['model_id'],
                                **{k: measured[k] for k in ('race_log_loss','horse_log_loss','brier_score','roc_auc','within_race_auc','top1_accuracy_percent')}})
    csv_file(directory, 'frozen_2025_comparison.csv', evaluation_rows)
    fits = []
    for path in sorted(directory.glob('actual_fit_*.started.json')):
        item = r.read(directory, path.name); done = directory / path.name.replace('.started.', '.complete.')
        if not done.is_file():
            raise safety.MLFailure('artifact_invalid')
        state = r.read(directory, done.name)
        fits.append(dict(item, state=state['state'], seconds=state['seconds']))
    csv_file(directory, 'actual_fit_ledger.csv', fits)
    before = r.read(directory, 'preservation_before.json')
    changed_project = sum(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha
                          for p, sha in before['project_files'].items())
    changed_private = sum(safety.digest(Path(p), root=r.BASE) != sha for p, sha in before['private_assets'].items())
    tasks = task_configuration()
    if changed_project or changed_private or tasks != before['tasks'] or tasks.get('status') != 'observed':
        raise safety.MLFailure('artifact_invalid')
    r.write(directory, 'preservation_after.json', {'old_project_files': len(before['project_files']), 'changed_old_project_files': 0,
            'protected_private_files': len(before['private_assets']), 'changed_private_files': 0,
            'task_configuration_unchanged': True, 'task_mutations': 0, 'sealed_2026_used': False, 'Stage7_results_used': False})
    if report['budget']['actual_fits'] != len(fits) or len(fits) > r.MAX_FITS:
        raise safety.MLFailure('artifact_invalid')
    r.write(directory, 'candidate_freeze.json', {'status': 'development_candidates_only_no_H_promotion',
        'A39_model_id': records[0]['model_id'], 'B_name': freeze['best_name'], 'B_model_id': freeze['best_model_id'],
        'B_features': freeze['features'], 'C_model_id': c_result['model_id'] if c_result else None,
        'C_features': c_result['features'] if c_result else [], 'B_chosen_only_2024': True,
        '2025_results_not_used_to_rechoose': True, 'further_search': False, 'production_adoption': False,
        'H_unchanged': True, 'report_sha256': safety.digest(directory / 'development_report.json', root=r.ROOT)})
    args = [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy','Bypass',
            '-File', str(PROJECT / 'scripts/render_tuner_reduction.ps1'), '-RunDirectory', str(directory)]
    proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', timeout=60, check=False)
    if proc.returncode or json.loads(proc.stdout).get('status') != 'ok':
        raise safety.MLFailure('io_failed')
    lines=['# 段階的特徴量削減＋Boruta＋LightGBM Tuner：有限比較', '',
        '主力H、Stage 7、朝7時タスクを維持。2026年封印期間・Stage 7途中成績は未使用。DB接続・外部送信・本番昇格はありません。','',
        '## 最初の39→32比較','',
        f'2024年の開発検証レースLog Loss：{records[0]["selection_metrics"]["race_log_loss"]:.6f} → {records[1]["selection_metrics"]["race_log_loss"]:.6f}。',
        '指定した7列を一括除外後、32から24項目まで当該モデルGain最小の1列を順次削除。毎回、同じ固定初期設定・Tuner4段階×2試行です。','',
        '## 全構成の2024年開発検証','',
        '|構成|項目数|レースLL|Brier|ROC-AUC|1位的中率|温度|best iteration|今回fit秒|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in rows:
        lines.append(f'|{item["configuration"]}|{item["feature_count"]}|{item["selection_2024_race_LL"]:.6f}|{item["selection_2024_Brier"]:.6f}|'
                     f'{item["selection_2024_ROC_AUC"]:.6f}|{item["selection_2024_top1_percent"]:.2f}%|{item["temperature"]}|{item["best_iteration"]}|{item["new_fit_seconds"]:.3f}|')
    lines += ['', 'A39の今回fit秒0は保存モデル再利用を意味します。元の8fit学習は7.343秒です。', '',
        '温度と特徴量数は2024年11〜12月の同じ開発検証窓で選定しました。この窓は温度当てはめにも使用したため、独立した未使用評価ではありません。',
        f'最小LLから0.002以内なら少ない項目を優先する事前規則で、B={freeze["best_name"]}（{freeze["feature_count"]}項目）を2025年評価前に固定しました。', '',
        '## 凍結後の2025年1〜9月比較','',
        f'共通{report["races"]:,}レース・{report["runners"]:,}頭。再利用済み開発評価です。未選定の他構成はこの期間で評価していません。', '',
        '|構成|項目数|レースLL|Brier|ROC-AUC|レース内AUC|1位的中率|','|---|---:|---:|---:|---:|---:|---:|']
    for item in evaluation_rows:
        lines.append(f'|{item["configuration"]}|{item["feature_count"]}|{item["race_log_loss"]:.6f}|{item["brier_score"]:.6f}|'
                     f'{item["roc_auc"]:.6f}|{item["within_race_auc"]:.6f}|{item["top1_accuracy_percent"]:.2f}%|')
    lines += ['', '差は新構成−A39で、LL/Brierは負が改善。39開催週の対応bootstrap1,000回、比較間で同じ再標本化を使用。','',
              '|比較|レースLL差|95%週クラスタ区間|','|---|---:|---|']
    for name, values in report['differences'].items():
        x=values['race_log_loss'];lo,hi=x['95_percent_ci']
        lines.append(f'|{name}−A39|{x["difference_new_minus_A39"]:+.6f}|[{lo:+.6f}, {hi:+.6f}]|')
    b_eval = report['evaluation_metrics'][report['best_backward']]
    b_delta = b_eval['race_log_loss'] - report['evaluation_metrics']['A39']['race_log_loss']
    lines += ['',f'Bの開発比較LL差は{b_delta:+.6f}。2025年の成績を見た後に特徴量数・温度・削除順を変更していません。Hの代替採用はしていません。','',
              '## Borutaとの比較','',
              f'RF実行{report["Boruta_selection_counts"]["RF_fit_calls"]}回、confirmed展開列{report["Boruta_selection_counts"]["confirmed_encoded"]}、tentative展開列{report["Boruta_selection_counts"]["tentative_encoded"]}、採用元特徴量{report["Boruta_selection_counts"]["native_features"]}項目。',
              'train-only辞書・数値中央値・one-hotで選択。confirmedの展開列が少なくとも1つあれば元特徴量を選択する規則です。tentativeは採用しません。',
              'CのLightGBMは元のnativeカテゴリ列で学習するため、A/B/C間でLightGBM入力の符号化方法は変わりません。',
              'confirmed0の場合はCを学習せず保留し、percや反復上限等を結果後に変更しません。PC-KEIBA流のパラメータ再調整は別仮説・別有限予算で検討する事項として残します。','',
              '## 予算・学習時間・校正','',
              f'実fit合計{report["budget"]["actual_fits"]} / 上限100。内訳：LGBM {report["budget"]["kind_counts"]["LGBM"]}、RF {report["budget"]["kind_counts"]["RF"]}。最終refit0、追加探索0。',
              f'fit呼出合計：LGBM {report["budget"]["fit_seconds"]["LGBM"]:.3f}秒、RF {report["budget"]["fit_seconds"]["RF"]:.3f}秒。全処理{completion["wall_seconds"]:.3f}秒。',
              f'親＋実行子孫private memoryピーク{completion["peak_parent_and_worker_private_bytes"]/1024**2:.2f}MiB。CPU4、研究ロックで直列実行、実測CPU負荷を各fit前に確認。',
              '各構成の温度候補成績と固定確率帯の校正表はdevelopment_report.jsonに保存。モデルごとに元39列契約、選択列と順序、カテゴリ辞書、式版、最適パラメータ、温度を保存しました。',
              'Gainによる削除は訓練上の寄与に基づく適応的選択です。未知データでの因果効果を示しません。既存39列の生成処理を再利用するため、生成処理の計算量削減は今回の実装範囲外です。','',
              '## ファイル','',
              '- selection_2024_race_log_loss.png：全特徴量数の2024年開発検証LL曲線。',
              '- frozen_2025_race_log_loss.png：評価前に固定した構成だけの2025年開発比較。',
              '- feature_count_results.csv：全構成、残した列、削除順、指標、較正温度、時間。',
              '- removed_feature_order.csv：1列ずつの削除順と根拠モデルGain。',
              '- frozen_2025_comparison.csv：同一2025年ロスターの比較。',
              '- actual_fit_ledger.csv：全実fit、状態、時間。',
              '- boruta_selection.json / boruta_preprocessor.json：確定・保留マスクと列対応。',
              '- candidate_freeze.json / all_candidates_frozen_before_evaluation.json：選択の固定。',
              '- preservation_after.json：旧コード、モデル、入力、タスクの保持。',
              '- selected_predictions.local.csv：個別予測を含むためPC内専用。共有しない。','',
              '再実行には新しい明示実行指示・新規計画が必要です。同一計画の実行を繰り返して追加試行はできません。','']
    with (directory / 'development_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    return {'status':'ok','report':str(directory/'development_report.md'),'feature_count_curve_PNGs':2,
            'old_project_files_unchanged':len(before['project_files']),'protected_files_unchanged':len(before['private_assets']),
            'task_configuration_unchanged':True,'actual_fits':len(fits)}


if __name__=='__main__':
    try:print(json.dumps(run(sys.argv[1]),ensure_ascii=False,allow_nan=False))
    except safety.MLFailure as error:
        print(json.dumps({'status':'stopped','reason':error.code,'stage':'summarize'}));raise SystemExit(1)
