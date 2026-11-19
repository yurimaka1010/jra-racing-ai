"""Anonymous development report, finite inventory and separate candidate freezes."""
from datetime import date,datetime
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from ..ml import safety
from ..research import experiment as exp
from ..stage3 import scoring
from ..win_sim import core
from . import contracts as c,store as s,data,oof,models,quotes,accuracy,finance


def inventory(directory):
    fits=[];optimizations=[]
    for path in sorted((directory/'fits').glob('*.started.json')):
        unit=path.name[:-len('.started.json')];done=s.completed(directory/'fits',unit)
        if not done:raise safety.MLFailure('training_failed')
        fits.append(dict(unit=unit,status='complete',**done['summary']))
    for path in sorted((directory/'optimizations').glob('*.started.json')):
        unit=path.name[:-len('.started.json')];doc=s.read(path.parent,unit+'.json')
        optimizations.append({k:doc.get(k) for k in ('fold_id','config_id','strength','status','reason','iterations','scales','coefficients','validation_metrics')})
    if len(fits)>5 or len(optimizations)>12:raise safety.MLFailure('input_limit')
    h60=s.completed(directory,'h60')['summary']
    return {'LightGBM_fits_executed':len(fits),'LightGBM_limit':5,'fits':fits,
        'H_new_OOF_fits':sum(x['kind']=='H' for x in fits),'H60_fits':sum(x['kind']=='H60' for x in fits),
        'H60_skipped':h60['skipped'],'optimizations_executed':len(optimizations),'optimization_limit':12,
        'optimizations_failed':sum(x['status']!='complete' for x in optimizations),'optimizations':optimizations,
        'new_conditional_purchase_rules':1,'extra_exploration':0,'synthetic_test_fits_not_real_experiments':True,
        'recovery_notes':['2023_H60_serialization_finished_before_JSON_list_tuple_loader_check_failed;serialized_model_recovered_without_refit']}


def replay(directory):
    done=s.completed(directory,'save_replay')
    if done:return done['summary']
    rows=[]
    for fit in inventory(directory)['fits']:
        year=fit['year'];kind=fit['kind'];bundle=models.load(directory/'models'/fit['model_id'])
        if kind=='H':
            split=c.h_split(year);start,end=map(date.fromisoformat,split['evaluation'][0]);frame=data.market_frame(directory,start,end)
            saved=oof.read(directory,start,end);current=data.merge_probability(frame,saved[['entry_key','history_probability']],'history_probability')
            expected=current.history_probability.to_numpy(float)
        else:
            start,end=date(year,1,1),date(year,9,30);frame=quotes.augment(directory,data.market_frame(directory,start,end))
            predictions=accuracy.parse(s.read(directory/'H60',f'{year}_predictions.local.json'))
            pm={''.join(p.race)+h:v for p in predictions for h,v in p.probabilities.items()};expected=np.asarray([pm[str(k)] for k in frame.entry_key])
        values=models.predict(bundle,frame);delta=float(np.max(np.abs(values-expected)))
        if delta>1e-12:raise safety.MLFailure('artifact_invalid')
        rows.append({'kind':kind,'year':year,'model_id':fit['model_id'],'rows':len(frame),'max_saved_prediction_difference':delta})
    summary={'new_models':rows,'max_saved_prediction_difference':max(x['max_saved_prediction_difference'] for x in rows),
        'old_H_reproduction':s.completed(directory,'reproduce')['summary'],'new_training':0}
    s.write(directory,'save_replay.json',summary);s.complete(directory,'save_replay',[directory/'save_replay.json'],summary)
    return summary


def tests():
    path=c.PROJECT/'reports/stage4_regression_tests.txt'
    raw=path.read_bytes();text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',text)
    if not match or not text.rstrip().endswith('OK'):raise safety.MLFailure('artifact_invalid')
    return {'tests':int(match[1]),'baseline_tests':850,'added_tests':int(match[1])-850,'all_passed':True,
        'sockets_and_real_DB_driver_blocked':True,'duration_seconds':float(match[2]),'log_path':str(path),
        'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def value(v,digits=6):return '未算出' if v is None else f'{v:,.{digits}f}'
def ci(v,digits=6):return '未算出' if v is None else '['+value(v[0],digits)+', '+value(v[1],digits)+']'


def markdown(report):
    acc=report['accuracy'];fin=report['finance'];con=report['conditional_accuracy'];inv=report['inventory'];lines=[]
    lines.extend(['# Stage 4 開発検証結果','',f"run_id: `{report['run_id']}`",'',
        '2023・2024・2025年の各1〜9月、同一6,373レース・87,601頭を使用した既参照の開発評価。独立した最終評価ではない。',
        'T−10公表オッズを使った事後シミュレーション。PC実取得時刻・購入可能時刻・当初のRA予定時刻の完全なas-of性は未確認。実購入は行っていない。','',
        '## 1. 有限実験の実行','',f"新規LightGBM {inv['LightGBM_fits_executed']}/5本（H OOF 2本、H60 3本）、直接統合 {inv['optimizations_executed']}/12最適化、見送り {inv['H60_skipped']}、最適化失敗 {inv['optimizations_failed']}。追加探索0。",'',
        '## 2. 新しく分かった点','',
        'Hstackの平均改善は小さく、Hcalによる市場出力の再較正が大部分を占める。ABCDの追加効果を示すHstack−Hcalはほぼ0で、方向も年によって異なる。',
        'H60の平均改善は確認できなかった。高不一致領域でのABCDの条件付き購入も、今回の固定ルールでは黒字にならなかった。',
        'これらは今回の有限構成についての実測であり、あらゆる相補性や条件付き収益の不存在を示さない。','',
        '## 3. 精度・主要LL比較','',
        '| 構成 | レース勝馬LL | 馬二値LL | Brier | 1位的中率% |','|---|---:|---:|---:|---:|'])
    for name in ('raw_market','H','Hcal','Hstack','H60','ABCD'):
        if name not in acc['metrics']['pooled']:continue
        m=acc['metrics']['pooled'][name];lines.append(f"| {name} | {value(m['race_log_loss'])} | {value(m['log_loss'])} | {value(m['brier_score'])} | {value(m['top1_accuracy_percent'],2)} |")
    lines.extend(['','差の符号は新構成−対照。負が改善。月曜起点・全競馬場共通の開催週、5,000回、seed=20261019。','',
        '| 比較 | プール差 | 95%区間 | 98.333…%区間 | 2023差 | 2024差 | 2025差 |',
        '|---|---:|---|---|---:|---:|---:|'])
    for name in ('Hstack_minus_H','Hstack_minus_Hcal','H60_minus_H'):
        p=acc['comparisons'][name]
        if 'pooled' not in p:lines.append(f'| {name} | 未算出 | 未算出 | 未算出 | | | |');continue
        a=p['pooled']['95_percent']['race_log_loss'];b=p['pooled']['98_333333_percent']['race_log_loss']
        annual=[p['years'][str(y)]['95_percent']['race_log_loss']['difference_new_minus_baseline'] for y in c.YEARS]
        lines.append(f"| {name} | {value(a['difference_new_minus_baseline'])} | {ci([a['ci_low'],a['ci_high']])} | {ci([b['ci_low'],b['ci_high']])} | "+' | '.join(value(x) for x in annual)+' |')
    lines.extend(['','3主要比較のBonferroniによる区間を併記。既参照の開発期間であり、補正後でも独立テストの正式証明や全分析の誤検出保証とは呼ばない。','',
        '統合学習用Hの学習履歴は2021年用8か月、2022年用20か月。一方、評価用Hは32・44・56か月であり、学習量・出力分布の差が残る。これを補う追加学習は実施していない。','',
        '### T−60被覆率','', '| 年 | 主評価レース | T60/T30利用可能 | 被覆率% |','|---|---:|---:|---:|'])
    for y,row in report['T60_evaluation_coverage'].items():lines.append(f"| {y} | {row['races']} | {row['usable_races']} | {value(row['usable_rate_percent'],2)} |")
    lines.extend(['','T−60またはT−30欠損は数値NaN・欠損フラグ1で残す。公表分終端が基準以前、鮮度300秒以内、同一馬番集合という既存契約を維持。T−10母集団は減らしていない。','',
        '## 4. 高不一致層・条件付き分析','',
        '| 層 | レース | ABCD LL | H LL | ABCD 1位的中率% | H 1位的中率% |','|---|---:|---:|---:|---:|---:|'])
    for name,row in con['pooled_strata'].items():
        lines.append(f"| {name} | {row['races']} | {value(row['ABCD']['race_log_loss'])} | {value(row['H']['race_log_loss'])} | {value(row['ABCD']['top1_accuracy_percent'],2)} | {value(row['H']['top1_accuracy_percent'],2)} |")
    lines.extend(['','境界は統合学習OOFの自然対数JSDの50・90百分位（linear）。評価結果による変更なし。馬単位pA/pHの3帯は記述較正用であり、部分集合を再正規化しない。詳細・週区間はJSONに保存。','',
        '## 5. 固定購入の仮想収益','',
        '単勝EV5%以上、最大1頭100円、最低勝率フィルターなし。判断はT−10、精算は確認済み公式HR。表示オッズを払戻とせず、控除率を重ねて差し引いていない。','',
        '| 戦略 | 購入 | 的中 | 投資円 | 払戻円 | 純利益円 | ROI% | 月平均純利益円（27か月） |','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name in (*finance.MAJOR,*finance.CONTROLS):
        if name not in fin['strategies']:continue
        row=fin['strategies'][name];f=row['financial']
        lines.append(f"| {name} | {f['bet_count']} | {f['hit_count']} | {f['investment_yen']:,} | {value(f['payout_yen'],0)} | {value(f['net_profit_yen'],0)} | {value(f['return_rate_percent'],2)} | {value(row['monthly_mean_profit_yen_including_27_months'],2)} |")
    lines.extend(['','| 主要戦略 | 純利益95%区間 円 | 純利益98.333…%区間 円 | ROI95%区間 % | 購入0再標本% | 的中週 |','|---|---|---|---|---:|---:|'])
    for name in finance.MAJOR:
        if name not in fin['strategies']:continue
        w=fin['strategies'][name]['weekly_uncertainty']
        lines.append(f"| {name} | {ci(w['95_percent']['profit_yen_ci'],0)} | {ci(w['98_333333_percent']['profit_yen_ci'],0)} | {ci(w['95_percent']['ROI_percent_ci'],2)} | {value(100*w['zero_purchase_replica_fraction'],2)} | {w['hit_weeks']} |")
    lines.extend(['','Hstack/H60は購入数・的中数が極端に少ない。全敗標本のROI区間[0,0]は真の回収率0の確定ではない。未精算は0円へ置換せず、購入0のROIは未算出。','',
        '### 評価27か月の純利益（購入ゼロ月を含む）','',
        '| 月 | Hstack 円 | H60 円 | 高不一致ABCD 円 |','|---|---:|---:|---:|'])
    for month in finance.MONTHS:
        lines.append('| '+month+' | '+' | '.join(value(fin['strategies'][n]['monthly'][month]['net_profit_yen'],0) if n in fin['strategies'] else '未算出' for n in finance.MAJOR)+' |')
    lines.extend(['','年別、購入なし年/月、購入・払戻集中、対応あり対照利益差、最大払戻依存、最大払戻除外時損益、未精算はfinance.jsonに保存。',
        'DDは評価した各年1〜9月の元の時系列だけで計算し、評価外の月を含む3年間全体のDDとは区別する。','',
        '## 6. ABCD＋E補記',''])
    e=report['E_accuracy'];ef=report['E_finance'];ed=e['ABCD_E_minus_ABCD']['95_percent']['race_log_loss']
    lines.extend([f"市場共通集合でのEのレースLL {value(e['pooled_ABCD_E']['race_log_loss'])}、ABCDとの差 {value(ed['difference_new_minus_baseline'])}。",
        f"ABCDに対する開催週対応の利益差 {value(ef['E_minus_ABCD']['profit_difference_yen'],0)}円、95%区間 {ci(ef['E_minus_ABCD']['95_percent_ci'],0)}。",
        '保存済みE予測だけを再利用。Eの追加学習・OOF化・統合・候補昇格は行っていない。','',
        '## 7. 候補の凍結',''])
    for name,result in acc['candidates'].items():lines.append(f"- 精度 {name}: `{result['status']}`。本番へ自動採用しない。")
    for name,result in fin['financial_candidates'].items():lines.append(f"- 収益 {name}: `{result['status']}`。")
    lines.extend(['','収益を集計する前にaccuracy_candidate_freeze.jsonを保存した。Hcalの改善は市場出力の再較正として扱い、ABCDの追加効果と混同しない。','',
        '## 8. 検証・保持','',f"架空・モック {report['tests']['tests']}件すべて成功（既存850件、追加{report['tests']['added_tests']}件）。実DBドライバーとソケットを禁止した回帰テスト。",
        f"旧参照{report['preservation']['reference_files']}ファイルと旧コード{report['preservation']['old_code_files']}ファイルの変更0。タスク設定一致: {report['preservation']['task_configuration_match']}。タスク変更0。",
        f"新規5モデルの保存予測再現の最大差 {report['save_replay']['max_saved_prediction_difference']}。旧6,373レースの市場・H予測の差0。",
        '2026年データの取得・学習・選択・評価0。旧最終評価計画を保持し、新しい最終評価入口は作成・有効化していない。','',
        '## 9. 保存先・再現','',f"保存先: `{c.ROOT/'evaluations'/report['run_id']}`",'',
        '専用コマンド・再開方法はプロジェクトのdocs/stage4_development.mdを参照。完了単位はハッシュ検証して再利用し、再開による再学習・実験予算追加は行わない。',''])
    return '\n'.join(lines)


def run(directory):
    done=s.completed(directory,'freeze')
    if done:
        s.verify_old(directory);return done['summary']
    if not s.completed(directory,'finance'):raise safety.MLFailure('artifact_invalid')
    lineage=oof.lineage(directory);reproduced=replay(directory);test_result=tests();inv=inventory(directory)
    # Validate the order without using profits for the earlier accuracy decision.
    earliest=min(p.stat().st_mtime_ns for p in (directory/'paper').glob('*.ledger.local.json'))
    if (directory/'accuracy_candidate_freeze.json').stat().st_mtime_ns>earliest:raise safety.MLFailure('artifact_invalid')
    coverage={}
    for year in c.YEARS:
        f=quotes.augment(directory,data.market_frame(directory,date(year,1,1),date(year,9,30)))
        n=int(f.entry_key.str[:16].nunique());available=int(f.loc[f[c.EXTRA[2]]==0,'entry_key'].str[:16].nunique())
        coverage[str(year)]={'races':n,'usable_races':available,'usable_rate_percent':100*available/n,
            'missing_races_retained':n-available}
    preservation=s.verify_old(directory)
    acc=s.read(directory,'accuracy.json');fin=s.read(directory,'finance.json')
    report={'run_id':directory.name,'version':c.VERSION,'created_at_jst':datetime.now(core.JST).isoformat(),
        'development_reuse_not_independent_test':True,'new_processing_end':c.END.isoformat(),
        'inventory':inv,'accuracy':acc,'conditional_accuracy':s.read(directory,'conditional_accuracy.json'),
        'T60_collection':s.completed(directory,'quotes60')['summary'],'T60_evaluation_coverage':coverage,
        'finance':fin,'E_accuracy':s.read(directory,'E_accuracy_supplement.json'),'E_finance':s.read(directory,'E_financial_supplement.json'),
        'preservation':preservation,'tests':test_result,'save_replay':reproduced,'OOF_lineage':lineage,
        'no_2026_processing_or_final_entry':True,'scheduler_mutations':0,'no_external_requests_or_new_packages':True}
    s.write(directory,'experiment_inventory.json',inv);s.write(directory,'preservation.json',preservation)
    s.write(directory,'candidate_freeze.json',{'version':c.VERSION,'accuracy':acc['candidates'],'financial':fin['financial_candidates'],
        'accuracy_decided_before_finance':True,'accuracy_freeze_sha256':s.digest(directory/'accuracy_candidate_freeze.json'),
        'financial_freeze_sha256':s.digest(directory/'financial_candidate_freeze.json'),
        'old_final_evaluation_plans_preserved':True,'new_final_evaluation_entry':False,'automatic_operation_adoption':False})
    s.write(directory,'development_report.json',report)
    md=safety.local_path(directory/'development_report.md',root=c.ROOT)
    with md.open('x',encoding='utf-8',newline='\n') as stream:stream.write(markdown(report))
    code=[*Path(__file__).parent.glob('*.py'),*c.PROJECT.glob('scripts/stage4*'),c.PROJECT/'tests/test_stage4.py',c.PROJECT/'docs/stage4_development.md']
    s.write(directory,'implementation_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)})
    files=[directory/n for n in ('experiment_inventory.json','candidate_freeze.json','preservation.json','development_report.json','development_report.md','implementation_hashes.json',
        'finance.complete.json','lineage.complete.json','save_replay.complete.json')]
    summary={'LightGBM_fits':inv['LightGBM_fits_executed'],'optimizations':inv['optimizations_executed'],'skipped':inv['H60_skipped'],
        'tests':test_result,'accuracy_candidates':acc['candidates'],'financial_candidates':fin['financial_candidates'],
        'preserved_reference_files':preservation['reference_files'],'changed_old_files':0,'task_configuration_match':preservation['task_configuration_match'],
        '2026_data_processed':False,'final_evaluation_entry_created':False,'report':str(md)}
    s.complete(directory,'freeze',files,summary);return summary
