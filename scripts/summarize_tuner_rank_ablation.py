"""Summarize one completed 36-feature ablation; no fitting or new selection."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from jra_racing_ai import tuner_rank_ablation as experiment
from jra_racing_ai import tuner_reduction as core
from jra_racing_ai.ml import safety


def csv_file(directory, name, rows):
    with (directory / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_saved_predictions(directory, report):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    frame = experiment.data.load_part(experiment.tuner.SOURCE, 'evaluation')
    saved = pd.read_csv(directory / 'selected_predictions.local.csv', dtype={'entry_key': str, 'date': str})
    if (saved.entry_key.duplicated().any() or len(saved) != 29879
            or len(set(saved.entry_key.str[:16])) != 2169
            or not np.array_equal(saved.entry_key.to_numpy(str), frame.entry_key.to_numpy(str))
            or not np.array_equal(saved.date.to_numpy(str), frame.date.to_numpy(str))
            or not np.array_equal(saved.winner_label, frame.winner_label)):
        raise safety.MLFailure('invalid_identity')
    maximum = 0.
    for record in report['selection_records']:
        name = record['name']
        if name == 'A39':
            bundle = experiment.tuner.load_model(core.OLD_RUN / 'models' / record['model_id'])
            probabilities = experiment.tuner.predict(*bundle, frame)
        else:
            bundle = core.load_model(directory / 'models' / record['model_id'])
            if bundle[1]['features'] != record['features'] or len(record['features']) != 36:
                raise safety.MLFailure('invalid_model')
            probabilities = core.predict(bundle, frame)
        maximum = max(maximum, float(np.max(np.abs(probabilities - saved[name].to_numpy(float)))))
        y = saved.winner_label.to_numpy(int)
        _, race = np.unique(saved.entry_key.str[:16], return_inverse=True)
        if not np.allclose(np.bincount(race, weights=probabilities), 1., rtol=0., atol=1e-12):
            raise safety.MLFailure('invalid_input')
        values = {'race_log_loss': float(np.mean(-np.log(probabilities[y == 1]))),
                  'horse_log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
                  'brier_score': float(brier_score_loss(y, probabilities)),
                  'roc_auc': float(roc_auc_score(y, probabilities))}
        for key, value in values.items():
            if abs(value - report['evaluation_metrics'][name][key]) > 1e-12:
                raise safety.MLFailure('artifact_invalid')
    if maximum > 1e-12:
        raise safety.MLFailure('invalid_model')
    return {'saved_model_and_CSV_prediction_max_difference': maximum,
            'independent_numpy_and_sklearn_metrics_agree': True,
            'race_probability_sums_equal_one': True, 'same_roster_and_labels': True,
            'races': 2169, 'runners': 29879, 'fits_in_validation': 0}


def run(identifier):
    directory = core.folder(identifier)
    plan = experiment.approved(directory)
    report = core.read(directory, 'development_report.json')
    done = core.read(directory, 'completion.json')
    if (done['status'] != 'complete' or report['version'] != experiment.VERSION
            or report['actual_fits'] != 8 or len(report['selection_records']) != 2
            or [x['feature_count'] for x in report['selection_records']] != [39, 36]):
        raise safety.MLFailure('artifact_invalid')
    removed = sorted(plan['removed_features'], key=lambda item: item['gain_rank'])
    if [(row['gain_rank'], row['feature']) for row in removed] != [(24, 'age'), (28, 'surface'), (30, 'horse_number')]:
        raise safety.MLFailure('invalid_mapping')
    validation = validate_saved_predictions(directory, report)
    csv_file(directory, 'removed_features.csv', removed)
    csv_file(directory, 'original_39_gain_ranking.csv', [dict(row, removed=row['gain_rank'] in experiment.DROP_GAIN_RANKS)
             for row in sorted(plan['ranking'], key=lambda item: item['gain_rank'])])
    rows = []
    bands = []
    for record in report['selection_records']:
        name = record['name']
        selection = record['selection_metrics']
        evaluation = report['evaluation_metrics'][name]
        difference = report['differences'].get(name, {}).get('race_log_loss', {})
        rows.append({'configuration': name, 'feature_count': record['feature_count'],
                     'deleted_count': 39 - record['feature_count'], 'model_id': record['model_id'],
                     'features': ';'.join(record['features']),
                     'removed_features': ';'.join(record['removed_features_in_order']),
                     'selection_2024_race_LL': selection['race_log_loss'],
                     'selection_2024_Brier': selection['brier_score'],
                     'selection_2024_AUC': selection['roc_auc'],
                     'selection_2024_top1_percent': selection['top1_accuracy_percent'],
                     **{'evaluation_2025_' + key: evaluation[key] for key in experiment.metrics.NAMES},
                     'race_LL_difference_vs_A39': difference.get('difference_new_minus_A39', 0.),
                     'race_LL_difference_CI_lower': difference.get('95_percent_ci', [0., 0.])[0],
                     'race_LL_difference_CI_upper': difference.get('95_percent_ci', [0., 0.])[1],
                     'temperature': record['temperature'], 'best_iteration': record['best_iteration'],
                     'actual_new_fits': record['actual_train_calls'],
                     'fit_call_seconds': record['fit_call_seconds'],
                     'selected_using_2024_only': name == report['overall_candidate']})
        for period, point in (('2024_selection_and_temperature', selection), ('2025_reused_development', evaluation)):
            for band in point['calibration_bands']:
                bands.append({'period': period, 'configuration': name, **band})
    csv_file(directory, 'comparison_results.csv', rows)
    csv_file(directory, 'calibration_bands.csv', bands)
    ledger = []
    for start in sorted(directory.glob('actual_fit_*.started.json')):
        attempt = core.read(directory, start.name)
        end = core.read(directory, start.name.replace('.started.', '.complete.'))
        ledger.append(dict(attempt, state=end['state'], seconds=end['seconds']))
    if len(ledger) != 8 or any(row['state'] != 'complete' or row['kind'] != 'LGBM' for row in ledger):
        raise safety.MLFailure('artifact_invalid')
    csv_file(directory, 'actual_fit_ledger.csv', ledger)
    freeze = core.read(directory, 'candidates_frozen_before_2025.json')
    core.write(directory, 'candidate_freeze.json', dict(freeze, production_adoption=False,
               old_models_unchanged=True, no_additional_search=True,
               report_sha256=safety.digest(directory / 'development_report.json', root=core.ROOT)))
    before = core.read(directory, 'preservation_before.json')
    for path, sha in before['project_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise safety.MLFailure('artifact_invalid')
    for path, sha in before['private_assets'].items():
        if safety.digest(Path(path), root=core.BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    current = experiment.task_configuration()
    if current != before['tasks'] or current.get('status') != 'observed':
        raise safety.MLFailure('artifact_invalid')
    preservation = {'old_project_files_unchanged': len(before['project_files']),
                    'protected_private_files_unchanged': len(before['private_assets']),
                    'task_configuration_unchanged': True, 'H_changed': False,
                    'sealed_2026_used': False, 'Stage7_results_used': False}
    core.write(directory, 'preservation_after.json', preservation)
    renderer = (PROJECT / 'scripts/render_tuner_reduction.ps1').read_text(encoding='utf-8-sig')
    for old, new in (
        ('Blue: A39 and backward; brown: Boruta.', 'Blue: A39 vs 36 features; remove only original Gain ranks 24, 28, 30.'),
        ('$all=@($report.selection_records)', '$all=@($report.selection_records | Sort-Object feature_count -Descending)')):
        if renderer.count(old) != 1:
            raise safety.MLFailure('artifact_invalid')
        renderer = renderer.replace(old, new)
    with (directory / 'render_rank_ablation.ps1').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(renderer)
    process = subprocess.run([r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
                '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                str(directory / 'render_rank_ablation.ps1'), '-RunDirectory', str(directory)],
                capture_output=True, text=True, encoding='utf-8', timeout=60, check=False)
    if process.returncode or json.loads(process.stdout).get('status') != 'ok':
        raise safety.MLFailure('io_failed')
    images = []
    for name in ('selection_2024_race_log_loss.png', 'frozen_2025_race_log_loss.png'):
        image = directory / name
        with image.open('rb') as stream:
            header = stream.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (2400, 1450):
            raise safety.MLFailure('artifact_invalid')
        images.append({'file': name, 'width': 2400, 'height': 1450,
                       'sha256': safety.digest(image, root=core.ROOT)})
    test_path = PROJECT / 'reports/tuner_rank_ablation_regression_tests.txt'
    test_text = test_path.read_text(encoding='utf-8-sig')
    if 'Ran 1215 tests' not in test_text or not test_text.rstrip().endswith('OK'):
        raise safety.MLFailure('artifact_invalid')
    validation.update(preservation=preservation, images=images, synthetic_tests_passed=1215,
                      synthetic_test_log_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                      removed_only_original_Gain_ranks=[24, 28, 30], remaining_features=36,
                      real_database_connections=0, external_communications=0, Boruta_fits=0,
                      old_A39_predictions_reproduced=True)
    core.write(directory, 'validation_final.json', validation)
    selected = report['overall_candidate']
    delta_2024 = rows[1]['selection_2024_race_LL'] - rows[0]['selection_2024_race_LL']
    difference = report['differences']['gain_drop_36']['race_log_loss']
    low, high = difference['95_percent_ci']
    text = ['# Gain重要度24・28・30位の3項目だけを削除：39→36特徴量', '',
            '削除対象は、元の39項目Tunerの24位 age（馬齢）、28位 surface（コース種別）、30位 horse_number（馬番）。',
            '特徴量数を24・28・30にする実験、または24・28・30項目を削除する実験ではありません。', '',
            '## 比較条件', '',
            '既存39項目モデルを読み取り専用で再利用し、36項目モデル1構成だけを新規8fitで学習。',
            '同じ初期パラメータ・Tuner手順（4段階各2試行）・最大300ラウンド・早期終了20・CPU4を使用。',
            '39項目の選択済み木やパラメータからの継ぎ足しではなく、従来と同じ探索開始条件です。',
            '学習2022-01-01〜2024-08-31、早期終了2024-09-08〜10-31、温度・構成選定2024-11-08〜12-24。',
            'カテゴリ辞書、元の列順、欠損処理、データ集合を共通化。温度候補0.75/1/1.25/2も共通です。', '',
            '## 2024年の選定結果', '',
            '|構成|レースLL|Brier|ROC-AUC|1位的中率|温度|best_iteration|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'|{row["feature_count"]}項目|{row["selection_2024_race_LL"]:.6f}|{row["selection_2024_Brier"]:.6f}|{row["selection_2024_AUC"]:.6f}|{row["selection_2024_top1_percent"]:.2f}%|{row["temperature"]}|{row["best_iteration"]}|')
    text += ['', f'差（36−39）は{delta_2024:+.6f}。最小LL+0.002以内なら少ない項目を選ぶ規則の選定結果は **{selected}**。',
             'この許容差は選定の規則であり、統計的な同等性・有意差の基準ではありません。',
             '候補を2025年の評価前に固定しました。2024年窓は温度当てはめと選定に共用した開発窓です。', '',
             '## 2025年1〜9月の共通開発評価', '',
             '同じ2,169レース・29,879頭。再利用済み開発評価であり、未使用最終テストではありません。', '',
             '|構成|レースLL|馬単位LL|Brier|ROC-AUC|レース内AUC|1位的中率|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'|{row["feature_count"]}項目|{row["evaluation_2025_race_log_loss"]:.6f}|{row["evaluation_2025_horse_log_loss"]:.6f}|{row["evaluation_2025_brier_score"]:.6f}|{row["evaluation_2025_roc_auc"]:.6f}|{row["evaluation_2025_within_race_auc"]:.6f}|{row["evaluation_2025_top1_accuracy_percent"]:.2f}%|')
    text += ['', f'レースLL差（36−39）は{difference["difference_new_minus_A39"]:+.6f}、95%週クラスタ対応bootstrap区間 [{low:+.6f}, {high:+.6f}]。',
             '39開催週、1,000反復、seed 20261019。全モデル・全指標に同じ再標本を使用。',
             '2025年は全指標の点推定が改善方向ですが、レースLL差の区間は0を含み、改善を確定していません。',
             '2024年選定では36項目が許容差を超えて悪化したため、既存39項目を維持します。2025年を見て選定を変更しません。',
             '列削除に伴うTunerの選択・木・温度の変化も含んだ比較であり、各特徴量の因果的寄与の推定ではありません。', '',
             '## テスト・予算・保持', '',
             f'新規8fit全完了。学習呼出合計{report["fit_seconds"]:.3f}秒、実行全体{done["wall_seconds"]:.3f}秒、private memoryピーク{done["peak_parent_and_worker_private_bytes"]/1024**2:.2f}MiB。',
             'Boruta・最終refit・追加探索は0。新規7件を含む1,215件の架空・モック回帰テスト成功。',
             f'元から存在するコード{preservation["old_project_files_unchanged"]}ファイル・保護対象{preservation["protected_private_files_unchanged"]}ファイルのハッシュ、予約タスク設定が不変。',
             '保存モデル再読込みと予測CSVの差は1e-12以下、NumPy/sklearnによる独立計算と主要指標が一致。',
             'H・Stage7・朝7時タスクは変更なし。封印2026年・Stage7途中成績・実DB・外部通信は未使用。', '',
             '## 出力', '',
             '- comparison_results.csv：両モデルの特徴量・指標・温度・木数・時間。',
             '- removed_features.csv / original_39_gain_ranking.csv：元順位と削除対象。',
             '- calibration_bands.csv：共通確率帯ごとの較正結果。',
             '- actual_fit_ledger.csv / candidate_freeze.json / validation_final.json：実行記録・選定・検証。',
             '- selection_2024_race_log_loss.png / frozen_2025_race_log_loss.png：数値付き比較グラフ。',
             '- selected_predictions.local.csv：PC内専用の個別予測。外部共有しない。', '']
    with (directory / 'development_report.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(text))
    return {'status': 'ok', 'run_id': identifier, 'report': str(directory / 'development_report.md'),
            'actual_fits': 8, 'selected_on_2024': selected, 'synthetic_tests_passed': 1215}


if __name__ == '__main__':
    try:
        print(json.dumps(run(sys.argv[1]), ensure_ascii=False, allow_nan=False))
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': 'summarize_rank_ablation'}))
        raise SystemExit(1)
