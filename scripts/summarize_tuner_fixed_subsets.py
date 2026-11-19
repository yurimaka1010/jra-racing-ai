"""Report the three completed independent subsets without fitting or selecting anew."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from jra_racing_ai import tuner_fixed_subsets as f
from jra_racing_ai import tuner_reduction as c
from jra_racing_ai.ml import safety


def csv_file(directory,name,rows):
    with (directory/name).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(identifier):
    p=c.folder(identifier);plan=f.approved(p);report=c.read(p,'development_report.json');done=c.read(p,'completion.json')
    if done['status']!='complete' or report['version']!=f.VERSION:
        raise safety.MLFailure('artifact_invalid')
    previous=c.read(f.PREVIOUS,'development_report.json')
    previous={x['feature_count']:x for x in previous['selection_records']}
    rows=[];bands=[]
    for record in report['selection_records']:
        name=record['name'];v=record['selection_metrics'];e=report['evaluation_metrics'][name]
        d=report['differences'].get(name,{}).get('race_log_loss',{})
        rows.append({'configuration':name,'feature_count':record['feature_count'],'model_id':record['model_id'],
            'features':';'.join(record['features']),'removed_by_initial_gain_low_to_high':';'.join(record['removed_features_in_order']),
            'selection_2024_race_LL':v['race_log_loss'],'selection_2024_Brier':v['brier_score'],
            'selection_2024_ROC_AUC':v['roc_auc'],'selection_2024_top1_percent':v['top1_accuracy_percent'],
            'previous_sequential_2024_race_LL':previous[record['feature_count']]['selection_metrics']['race_log_loss'],
            'evaluation_2025_race_LL':e['race_log_loss'],'evaluation_2025_Brier':e['brier_score'],
            'evaluation_2025_ROC_AUC':e['roc_auc'],'evaluation_2025_within_race_AUC':e['within_race_auc'],
            'evaluation_2025_top1_percent':e['top1_accuracy_percent'],
            'evaluation_race_LL_diff_A39':d.get('difference_new_minus_A39',0.),
            'evaluation_race_LL_diff_CI_lower':d.get('95_percent_ci',[0.,0.])[0],
            'evaluation_race_LL_diff_CI_upper':d.get('95_percent_ci',[0.,0.])[1],
            'temperature':record['temperature'],'best_iteration':record['best_iteration'],
            'new_fit_calls':record['actual_train_calls'],'new_fit_seconds':record['fit_call_seconds'],
            'overall_selected_on_2024':name==report['overall_candidate'],'reduced_selected_on_2024':name==report['reduced_candidate']})
        for period,values in (('2024_selection_T_fit',v),('2025_reused_development',e)):
            for b in values['calibration_bands']:bands.append({'period':period,'configuration':name,**b})
    csv_file(p,'comparison_results.csv',rows);csv_file(p,'calibration_bands.csv',bands)
    ranking=[]
    for x in sorted(plan['ranking'],key=lambda x:x['gain_rank']):
        ranking.append(dict(x,**{f'kept_in_{n}':x['gain_rank']<=n for n in f.COUNTS}))
    csv_file(p,'initial_gain_ranking_and_selection.csv',ranking)
    ledger=[]
    for start in sorted(p.glob('actual_fit_*.started.json')):
        a=c.read(p,start.name);b=c.read(p,start.name.replace('.started.','.complete.'))
        ledger.append(dict(a,**{'state':b['state'],'seconds':b['seconds']}))
    if len(ledger)!=24 or report['actual_fits']!=24:
        raise safety.MLFailure('artifact_invalid')
    csv_file(p,'actual_fit_ledger.csv',ledger)
    freeze=c.read(p,'candidates_frozen_before_2025.json')
    c.write(p,'candidate_freeze.json',dict(freeze,production_adoption=False,H_changed=False,
          additional_search=False,report_sha256=safety.digest(p/'development_report.json',root=c.ROOT)))
    before=c.read(p,'preservation_before.json')
    for path,sha in before['project_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise safety.MLFailure('artifact_invalid')
    for path,sha in before['private_assets'].items():
        if safety.digest(Path(path),root=c.BASE)!=sha:raise safety.MLFailure('artifact_invalid')
    current=f.task_configuration()
    if current!=before['tasks'] or current.get('status')!='observed':raise safety.MLFailure('artifact_invalid')
    c.write(p,'preservation_after.json',{'old_project_files_unchanged':len(before['project_files']),
            'protected_private_files_unchanged':len(before['private_assets']),'task_configuration_unchanged':True,
            'H_changed':False,'Stage7_results_used':False,'sealed_2026_used':False})
    # Reuse the native chart renderer with only the method caption changed.
    renderer=(PROJECT/'scripts/render_tuner_reduction.ps1').read_text(encoding='utf-8-sig')
    old_caption='Blue: A39 and backward; brown: Boruta.'
    if renderer.count(old_caption)!=1:raise safety.MLFailure('artifact_invalid')
    renderer=renderer.replace(old_caption,'Blue: frozen A39 and independent fixed top-Gain subsets.')
    with (p/'render_fixed_subsets.ps1').open('x',encoding='utf-8',newline='\n') as stream:stream.write(renderer)
    proc=subprocess.run([r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive',
        '-ExecutionPolicy','Bypass','-File',str(p/'render_fixed_subsets.ps1'),'-RunDirectory',str(p)],
        capture_output=True,text=True,encoding='utf-8',timeout=60,check=False)
    if proc.returncode or json.loads(proc.stdout).get('status')!='ok':raise safety.MLFailure('io_failed')
    text=['# 39項目の固定Gain順位から独立に24・28・30項目を残す比較','',
        '元の39項目TunerモデルのGain順位を一度だけ使い、上位24・28・30項目を各々独立に選択しました。途中のモデル重要度を使う前回の逐次削減とは別実験です。','',
        '列順序・カテゴリ辞書・時系列分割・欠損処理は既存処理を再利用。各構成のTunerはA39研究と同じ固定初期パラメータ、4段階×2試行から開始し、39列の木を継ぎ足してはいません。','',
        '## 2024年選定用レースLog Loss','',
        '|項目数|今回：39項目の固定順位から選択|前回：逐次削減|今回の温度|今回のfit秒|','|---|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda x:-x['feature_count']):
        text.append(f'|{row["feature_count"]}|{row["selection_2024_race_LL"]:.6f}|{row["previous_sequential_2024_race_LL"]:.6f}|{row["temperature"]}|{row["new_fit_seconds"]:.3f}|')
    text+=['',f'2024年の最小LLから0.002以内なら少ない項目を選ぶ規則で、全体候補は{report["overall_candidate"]}、削減群候補は{report["reduced_candidate"]}。全候補を2025年評価前に固定しました。',
        '同じ2024年窓を温度の当てはめと構成選択に共用しています。未使用較正評価ではありません。','',
        '## 2025年1〜9月の共通開発評価','',
        f'共通{report["races"]:,}レース・{report["runners"]:,}頭。既に利用済みの開発データであり、未使用最終テストではありません。2025年の結果で項目数を選び直していません。','',
        '|項目数|レースLL|Brier|ROC-AUC|レース内AUC|1位的中率|','|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda x:-x['feature_count']):
        text.append(f'|{row["feature_count"]}|{row["evaluation_2025_race_LL"]:.6f}|{row["evaluation_2025_Brier"]:.6f}|{row["evaluation_2025_ROC_AUC"]:.6f}|{row["evaluation_2025_within_race_AUC"]:.6f}|{row["evaluation_2025_top1_percent"]:.2f}%|')
    text+=['','差は新構成−A39。レースLLは負が改善。開催週クラスタの対応bootstrap1,000回を全比較で共有。','',
           '|項目数|レースLL差|95%区間|','|---|---:|---|']
    for row in sorted(rows[1:],key=lambda x:-x['feature_count']):
        text.append(f'|{row["feature_count"]}|{row["evaluation_race_LL_diff_A39"]:+.6f}|[{row["evaluation_race_LL_diff_CI_lower"]:+.6f}, {row["evaluation_race_LL_diff_CI_upper"]:+.6f}]|')
    text+=['','## 予算・保持・出力','',
        f'新規fitは24回、全完了。最終refit・Boruta・追加探索は0。学習呼出合計{report["fit_seconds"]:.3f}秒、実行全体{done["wall_seconds"]:.3f}秒、親と子孫のprivate memoryピーク{done["peak_parent_and_worker_private_bytes"]/1024**2:.2f}MiB。CPU4・研究ロックで直列実行。',
        f'旧コード{len(before["project_files"])}ファイルと保護対象{len(before["private_assets"])}ファイル、予約タスク設定の不変を確認しました。',
        '主力H・Stage7・朝7時タスクを変更していません。2026年封印期間・Stage7途中成績・実DB・外部通信は使用していません。',
        'Gain下位列の除外は、列サンプリングやチューニング経路も変え得ます。重要度が低いことは除外後の性能維持を保証しません。',
        'この再利用済み開発比較だけでHの代替採用や収益性を認定していません。','',
        '- comparison_results.csv：残した列、削除した列、全指標・温度・時間。',
        '- initial_gain_ranking_and_selection.csv：初期39項目のGain・Split・各構成への採否。',
        '- calibration_bands.csv：各構成の2024/2025年校正表。',
        '- actual_fit_ledger.csv：実fit台帳。',
        '- selection_2024_race_log_loss.png / frozen_2025_race_log_loss.png：各期間の性能曲線。',
        '- selected_predictions.local.csv：個別予測を含むためPC内専用。共有しない。','']
    with (p/'development_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(text))
    return {'status':'ok','report':str(p/'development_report.md'),'actual_fits':24,
            'overall_candidate':report['overall_candidate'],'reduced_candidate':report['reduced_candidate']}


if __name__=='__main__':
    try:print(json.dumps(run(sys.argv[1]),ensure_ascii=False,allow_nan=False))
    except safety.MLFailure as error:
        print(json.dumps({'status':'stopped','reason':error.code,'stage':'summarize_fixed'}));raise SystemExit(1)
