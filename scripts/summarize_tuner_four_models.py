"""Aggregate completed four-model fits and validate reuse; never fit a model."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from jra_racing_ai import tuner_four_models as study
from jra_racing_ai import tuner_reduction as core
from jra_racing_ai.ml import safety


def write_csv(directory, name, rows):
    with (directory / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate(directory, report):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    frame = study.data.load_part(study.tuner.SOURCE, 'evaluation')
    saved = pd.read_csv(directory / 'predictions.local.csv', dtype={'entry_key': str, 'date': str}, float_precision='round_trip')
    if (len(saved) != 29879 or saved.entry_key.duplicated().any() or saved.entry_key.str[:16].nunique() != 2169
            or not np.array_equal(saved.entry_key.to_numpy(str), frame.entry_key.to_numpy(str))
            or not np.array_equal(saved.date.to_numpy(str), frame.date.to_numpy(str))
            or not np.array_equal(saved.winner_label.to_numpy(int), frame.winner_label.to_numpy(int))):
        raise safety.MLFailure('invalid_identity')
    delta = {}
    _, codes = np.unique(saved.entry_key.str[:16], return_inverse=True)
    y = saved.winner_label.to_numpy(int)
    for record in report['selection_records']:
        name = record['name']; path = Path(record['model_path'])
        q = study.tuner.predict(*study.tuner.load_model(path), frame) if name == 'D39' else core.predict(core.load_model(path), frame)
        p = saved[name].to_numpy(float)
        delta[name] = float(np.max(np.abs(q-p)))
        if delta[name] > 1e-12 or not np.allclose(np.bincount(codes, weights=p), 1., rtol=0., atol=1e-12):
            raise safety.MLFailure('invalid_model')
        point = {'race_log_loss': float(np.mean(-np.log(p[y==1]))),
                 'horse_log_loss': float(log_loss(y, p, labels=[0,1])),
                 'brier_score': float(brier_score_loss(y, p)), 'roc_auc': float(roc_auc_score(y,p))}
        if any(abs(point[key]-report['evaluation_metrics'][name][key]) > 1e-12 for key in point):
            raise safety.MLFailure('artifact_invalid')
    for axis in ('surface','distance'):
        for name in study.NAMES:
            rows = [r for r in report['subgroup_metrics'] if r['axis']==axis and r['configuration']==name]
            if sum(r['races'] for r in rows) != 2169 or sum(r['runners'] for r in rows) != 29879:
                raise safety.MLFailure('invalid_identity')
    return {'same_roster_labels_and_race_probability_sums': True, 'independent_metrics_agree': True,
            'model_reload_prediction_max_difference': delta, 'subgroup_populations_unchanged': True,
            'validation_fits': 0, 'C_D_and_Boruta_new_fits': 0}


def run(identifier):
    directory = core.folder(identifier)
    plan = study.approved(directory); report = core.read(directory, 'development_report.json')
    done = core.read(directory, 'completion.json')
    records = report['selection_records']
    if (report['version'] != study.VERSION or done['status'] != 'complete' or report['actual_new_fits'] != 16
            or [row['name'] for row in records] != list(study.NAMES)
            or [row['feature_count'] for row in records] != [10,20,19,39]
            or [row['actual_train_calls'] for row in records] != [8,8,0,0]):
        raise safety.MLFailure('artifact_invalid')
    validation = validate(directory, report)
    frozen = core.read(directory, 'candidate_frozen_before_2025.json')
    if frozen['name'] != study.select_strict(records)['name'] or frozen['name'] != report['overall_candidate'] or frozen['tolerance'] != 0.:
        raise safety.MLFailure('artifact_invalid')
    rank = {row['feature']: row for row in plan['ranking']}
    features = []; membership = []
    for record in records:
        for order, name in enumerate(record['features'], 1):
            item = rank[name]
            features.append({'configuration': record['name'], 'feature_count': record['feature_count'],
                'model_id': record['model_id'], 'native_model_column_order': order, 'feature': name,
                'original_39_gain_rank': item['gain_rank'], 'original_39_gain': item['gain'],
                'original_39_gain_share_percent': item['gain_share']*100})
    sets = {r['name']: set(r['features']) for r in records}
    for name, item in sorted(rank.items(), key=lambda pair: pair[1]['gain_rank']):
        membership.append({'feature': name, 'original_39_gain_rank': item['gain_rank'],
            'original_39_gain': item['gain'], **{key: name in sets[key] for key in study.NAMES}})
    difference = [dict(row, relationship='common' if row['B20'] and row['C19'] else 'Gain20_only' if row['B20'] else 'Boruta19_only')
                  for row in membership if row['B20'] or row['C19']]
    write_csv(directory, 'feature_lists.csv', features)
    write_csv(directory, 'feature_membership.csv', membership)
    write_csv(directory, 'Gain20_vs_Boruta19.csv', difference)
    summary = []; calibration = []; models = []
    for record in records:
        name = record['name']; meta = safety.read_json(Path(record['model_path'])/'model.json', root=core.BASE)
        models.append({'configuration': name, 'model_id': record['model_id'], 'model_path': record['model_path'],
            'features': record['features'], 'original_gain_ranks': [rank[n]['gain_rank'] for n in record['features']],
            'parameters': record['parameters'], 'temperature': record['temperature'], 'best_iteration': record['best_iteration'],
            'base_feature_spec': plan['base_feature_spec'], 'categories': meta['categories'],
            'input_format': meta['input_format'], 'formula_version': meta['formula_version'],
            'model_json_sha256': safety.digest(Path(record['model_path'])/'model.json', root=core.BASE),
            'tree_sha256': meta['tree_sha256'], 'new_fits': record['actual_train_calls'],
            'new_fit_seconds': record['new_fit_seconds'], 'historical_fit_seconds': record['historical_fit_seconds']})
        for period, point in (('selection_2024',record['selection_metrics']), ('development_2025', report['evaluation_metrics'][name])):
            summary.append({'period': period, 'configuration': name, 'feature_count': record['feature_count'],
                'model_id': record['model_id'], **{key: point[key] for key in study.metrics.NAMES},
                'calibration_ECE_fixed_bins': study.calibration_summary(point['calibration_bands'])['fixed_bin_ECE'],
                'temperature': record['temperature'], 'best_iteration': record['best_iteration'],
                'new_fit_calls': record['actual_train_calls'], 'new_fit_seconds': record['new_fit_seconds'],
                'historical_fit_seconds': record['historical_fit_seconds'], 'selected_using_2024_only': name==frozen['name']})
            for band in point['calibration_bands']:
                calibration.append({'period': period, 'configuration': name, **band})
    write_csv(directory, 'comparison_results.csv', summary)
    write_csv(directory, 'calibration_bands.csv', calibration)
    write_csv(directory, 'condition_metrics.csv', report['subgroup_metrics'])
    write_csv(directory, 'paired_differences.csv', [{'pairing': pair, 'metric': metric, 'difference': value['difference'],
              'CI_lower': value['95_percent_ci'][0], 'CI_upper': value['95_percent_ci'][1]}
             for pair, values in report['differences'].items() for metric, value in values.items()])
    core.write(directory, 'model_index.json', {'models': models, 'parameters_were_not_shared_best_values': True,
                                              'original_common_initial_parameters': plan['parameters']})
    best = next(model for model in models if model['configuration']==frozen['name'])
    core.write(directory, 'candidate_freeze.json', dict(frozen,
               report_sha256=safety.digest(directory/'development_report.json',root=core.ROOT),
               model_json_sha256=best['model_json_sha256'], tree_sha256=best['tree_sha256'],
               H_changed=False, Stage7_changed=False, task_changes=0, adoption_scope='basic_ability_feature_candidate_only_not_all_prediction_tasks'))
    ledger = []
    for path in sorted(directory.glob('actual_fit_*.started.json')):
        start = core.read(directory,path.name)
        end = core.read(directory,path.name.replace('.started.','.complete.'))
        ledger.append(dict(start,state=end['state'],seconds=end['seconds']))
    if len(ledger)!=16 or {row['name'] for row in ledger}!={'A10','B20'} or any(row['state']!='complete' for row in ledger):
        raise safety.MLFailure('artifact_invalid')
    write_csv(directory,'actual_fit_ledger.csv',ledger)
    before = core.read(directory,'preservation_before.json')
    for path,sha in before['project_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise safety.MLFailure('artifact_invalid')
    for path,sha in before['private_assets'].items():
        if safety.digest(Path(path),root=core.BASE)!=sha:
            raise safety.MLFailure('artifact_invalid')
    current = study.task_configuration()
    if current!=before['tasks'] or current.get('status')!='observed':
        raise safety.MLFailure('artifact_invalid')
    preservation = {'old_project_files_unchanged':len(before['project_files']),
                    'protected_private_files_unchanged':len(before['private_assets']),
                    'task_configuration_unchanged':True,'H_changed':False,'Stage7_results_used':False,'sealed_2026_used':False}
    core.write(directory,'preservation_after.json',preservation)
    process = subprocess.run([r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive',
                '-ExecutionPolicy','Bypass','-File',str(PROJECT/'scripts/render_tuner_four_models.ps1'),
                '-RunDirectory',str(directory)],capture_output=True,text=True,encoding='utf-8',timeout=60)
    if process.returncode or json.loads(process.stdout).get('status')!='ok':
        raise safety.MLFailure('io_failed')
    test_path = PROJECT/'reports/tuner_four_models_regression_tests.txt'
    test_log = test_path.read_text(encoding='utf-8-sig')
    found = re.search(r'Ran (\d+) tests',test_log)
    if not found or int(found[1])!=1224 or not test_log.rstrip().endswith('OK'):
        raise safety.MLFailure('artifact_invalid')
    validation.update(preservation=preservation,synthetic_tests_passed=1224,new_tests=9,
                      source_and_test_log_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                      all_16_fits_complete=True,C19_D39_models_and_predictions_reproduced=True,
                      saved_reference_metrics_unchanged=True,selection_strict_no_tolerance=True)
    core.write(directory,'validation_final.json',validation)
    text=['# LightGBM Tuner：Gain上位10・20項目 / Boruta19項目 / 全39項目の比較','',
          f'基本特徴量候補は **{frozen["name"]}（{frozen["feature_count"]}項目）**。2024年レースLog Lossの厳密最小値で、2025年評価前に固定しました。許容差0.002は使用していません。','',
          '## 条件と学習範囲','',
          'A10・B20のみ各8fit。C19・D39は既存モデル、予測、評価結果を再利用し再学習なし。Boruta再実行なし。',
          '学習2022-01-01〜2024-08-31、早期終了2024-09-08〜10-31、温度・構成選定2024-11-08〜12-24。',
          '既存と同じ共通初期パラメータからLightGBMTunerの4段階各2試行。D39の最適パラメータからは開始していません。',
          '温度候補0.75/1/1.25/2、カテゴリ辞書・元の列順・欠損処理・対象行集合も同じ。',
          '2024年窓は温度と選定に共用。2025年も再利用済み開発評価で、未使用最終テストではありません。条件別成績を再選択へ戻していません。','',
          '## 総合性能','']
    for period,caption in (('selection_2024','2024年：較正・構成選定'),('development_2025','2025年1〜9月：同一2,169レース・29,879頭')):
        text += [f'### {caption}','','|構成|レースLL↓|Brier↓|ROC-AUC↑|レース内AUC↑|1位的中率↑|ECE↓|',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['period']==period:
                text.append(f'|{row["configuration"]}|{row["race_log_loss"]:.6f}|{row["brier_score"]:.6f}|{row["roc_auc"]:.6f}|{row["within_race_auc"]:.6f}|{row["top1_accuracy_percent"]:.2f}%|{row["calibration_ECE_fixed_bins"]:.6f}|')
        text.append('')
    text += ['ECEは既存の固定8確率帯による頭数加重の絶対誤差。較正表はcalibration_bands.csv。新しい較正器は追加していません。',
             'D39が主指標・Brier・AUC・1位的中率で最良の点推定。B20はECEがD39より小さい一方、それだけを採用根拠にはしていません。','',
             '### 2025年の対応比較','','差は左の構成−右の構成。レースLLは負が改善。39開催週、共通の対応bootstrap1,000回。',
             '|比較|レースLL差|95%区間|','|---|---:|---|']
    for pair,values in report['differences'].items():
        item=values['race_log_loss'];lo,hi=item['95_percent_ci']
        text.append(f'|{pair}|{item["difference"]:+.6f}|[{lo:+.6f}, {hi:+.6f}]|')
    text += ['','B20はC19よりレースLL・Brier・AUCが改善方向で、レースLL差の区間も負。1位的中率はC19を僅かに下回ります。',
             'B20とD39のレースLL差は小さく区間が0を含みますが、これは統計的同等性の証明ではありません。2024年の選定結果を変更しません。',
             '繰り返し使った開発期間上の有限比較であり、区間は累積した特徴量探索・構成選択バイアスを補正していません。','',
             '## B20とC19の特徴量の違い','']
    for relation,label in (('common','共通'),('Gain20_only','B20だけ'),('Boruta19_only','C19だけ')):
        chosen=[row for row in difference if row['relationship']==relation]
        text += [f'### {label}：{len(chosen)}項目','','|元のGain順位|特徴量|','|---:|---|']
        text.extend(f'|{row["original_39_gain_rank"]}|{row["feature"]}|' for row in chosen)
        text.append('')
    text += ['10・19・20・39項目すべての列名と列順・元のGain順位はfeature_lists.csv、全39項目の採否はfeature_membership.csvに保存。',
             '選択方法とTunerで得られる木・最適パラメータは同時に変わるため、各特徴量の因果的な効果の比較ではありません。','',
             '## 距離帯・芝ダート別の開発評価','',
             '条件は元の保存レース条件と完全キーで対応。同じ群内では4モデルのレース・頭数を共通化。条件欠損による行の除外なし。',
             'surfaceはJVトラックコード。ローカルJV-Data4901.pdfのPDF45頁・表2009で表示区分を確認。障害とサンドは芝・ダートへ混ぜません。',
             '|条件|レース数|頭数|A10 LL|B20 LL|C19 LL|D39 LL|LL点推定最小|','|---|---:|---:|---:|---:|---:|---:|---|']
    labels={'up_to_1400m':'〜1400m','1401_1800m':'1401〜1800m','1801_2200m':'1801〜2200m',
            '2201m_and_above':'2201m〜','turf':'芝（平地）','dirt':'ダート（平地）','jump':'障害','sand':'サンド',
            'unknown':'不明','inconsistent':'同一レース内の条件不整合'}
    for axis in ('distance','surface'):
        groups=sorted({row['group'] for row in report['subgroup_metrics'] if row['axis']==axis})
        for group in groups:
            rows={row['configuration']:row for row in report['subgroup_metrics'] if row['axis']==axis and row['group']==group}
            base=rows['D39'];winner=min(study.NAMES,key=lambda name:rows[name]['race_log_loss'])
            values='|'.join(f'{rows[name]["race_log_loss"]:.6f}' for name in study.NAMES)
            text.append(f'|{labels.get(group,group)}|{base["races"]}|{base["runners"]}|{values}|{winner}|')
    text += ['','芝1,073・ダート1,048・障害48レース。サンド・不明・不整合は0。距離区分も欠損・不整合0で全2,169レースをカバー。',
             '条件別の他指標・ECEはcondition_metrics.csv。これは小標本群も含む記述的な比較で、距離別・コース別AIの採用判定ではありません。','',
             '## 温度・学習時間・保存モデル','','|構成|温度|best_iteration|今回fit数|今回学習呼出秒|既存記録の学習呼出秒|','|---|---:|---:|---:|---:|---:|']
    for record in records:
        text.append(f'|{record["name"]}|{record["temperature"]}|{record["best_iteration"]}|{record["actual_train_calls"]}|{record["new_fit_seconds"]:.3f}|{record["historical_fit_seconds"]:.3f}|')
    text += ['',f'今回の新規学習呼出合計{report["new_fit_seconds"]:.3f}秒、処理全体{done["wall_seconds"]:.3f}秒、private memoryピーク{done["peak_parent_and_worker_private_bytes"]/1024**2:.2f}MiB。',
             'C/Dの時間は過去実行の記録です。実行負荷やキャッシュが異なるため、厳密な学習速度比とはみなしません。',
             '各モデルID・保存先・最適パラメータ・特徴量順序・カテゴリ辞書・ハッシュはmodel_index.json。','',
             '## 基本構成の判断と保持','',f'採用候補は {frozen["name"]}、モデルID `{frozen["model_id"]}`、温度{frozen["temperature"]}。',
             '現時点の基本能力モデルは全39項目を保持します。市場モデルHを置き換えたり、全券種・全条件へ一律適用したりしていません。',
             '今後の最新データ追加・前向き検証では、この構成と比較対象を別IDで扱います。今回の処理では2026年データを開いていません。',
             f'追加9件を含む1,224件の架空・モックテスト成功。旧コード{preservation["old_project_files_unchanged"]}ファイル、保護資産{preservation["protected_private_files_unchanged"]}ファイル、予約タスク設定が不変。',
             'C/Dの保存予測と旧指標を再現し、全4モデルの保存再読込み、独立したNumPy/sklearnの指標、条件別分母も検証。',
             '既存H・Stage7・朝7時タスクは変更なし。2026年封印・Stage7途中成績・実DB・外部通信・追加refit・Boruta再実行は未使用。','',
             '## グラフと再実行','',
             '![2024年比較](comparison_2024_race_log_loss.png)','',
             '![2025年比較](comparison_2025_race_log_loss.png)','',
             '![特徴量数とレースLog Loss](feature_count_race_log_loss_display.png)','',
             'ソース：src/jra_racing_ai/tuner_four_models.py、実行：scripts/run_tuner_four_models.py、',
             '集計：scripts/summarize_tuner_four_models.py、描画：scripts/render_tuner_four_models.ps1、',
             'テスト：tests/test_tuner_four_models.py、手順：docs/tuner_four_models.md。',
             'predictions.local.csvは個別データを含むためPC内専用。外部へ共有しない。','']
    feature_text=['## 4モデルの全特徴量一覧','',
                  '表示は元D39のGain順位順。実際のモデル列順は[特徴量CSV](feature_lists.csv)に保持しています。',
                  'C19はBorutaの集合で、A10/B20と同じ逐次削減経路ではありません。','']
    for record in records:
        feature_text += [f'### {record["name"]}：{record["feature_count"]}項目','','|元のGain順位|特徴量|','|---:|---|']
        feature_text.extend(f'|{rank[name]["gain_rank"]}|{name}|'
                            for name in sorted(record['features'],key=lambda name:rank[name]['gain_rank']))
        feature_text.append('')
    feature_text += ['[総合CSV](comparison_results.csv) / [20対19の相違](Gain20_vs_Boruta19.csv) / '
                     '[条件別CSV](condition_metrics.csv) / [較正表](calibration_bands.csv) / '
                     '[保存モデルと最適パラメータ](model_index.json)','']
    position=text.index('## グラフと再実行')
    text[position:position]=feature_text
    with (directory/'development_report.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(text))
    return {'status':'ok','report':str(directory/'development_report.md'),'selected':frozen['name'],
            'new_fits':16,'C_D_fits':0,'tests_passed':1224,
            'feature_overlap_counts':{relation:sum(row['relationship']==relation for row in difference)
                   for relation in ('common','Gain20_only','Boruta19_only')}}


if __name__=='__main__':
    try:
        print(json.dumps(run(sys.argv[1]),ensure_ascii=False,allow_nan=False))
    except safety.MLFailure as error:
        print(json.dumps({'status':'stopped','reason':error.code,'stage':'summarize_four_models'}))
        raise SystemExit(1)
