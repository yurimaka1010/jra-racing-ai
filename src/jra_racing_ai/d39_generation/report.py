"""Council-ready aggregates; calibration diagnostics never stand in for future tests."""
import csv
import os
from pathlib import Path

from ..tuner_pilot import experiment as tuner
from ..model_diagnostics import importance
from ..objective_pilot import data as original_data
from . import contracts as c,learning


def csv_file(path,rows,fields=None):
    if not rows and not fields:return
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())


def report(directory):
    plan=c.approved(directory);training=c.read(directory/'training_report.json');data=c.read(directory/'data_manifest.json')
    old_bundle=tuner.load_model(c.OLD_MODEL);new_bundle=learning.load(Path(training['model_path']))
    old_meta=old_bundle[1];new_meta=new_bundle[1]
    old_data=c.read(tuner.SOURCE/'data_manifest.json')['parts']['train']
    old_frame=original_data.load_part(tuner.SOURCE,'train')
    old_missing={name:int(old_frame[name].isna().sum()) for name in c.FEATURE_SPEC['features']}
    old_gain={row['feature']:row for row in importance(*old_bundle)}
    new_gain={row['feature']:row for row in importance(*new_bundle)}
    ranks=[];missing=[]
    for feature in c.FEATURE_SPEC['features']:
        a=old_gain[feature];b=new_gain[feature]
        ranks.append({'feature':feature,'old_gain_rank':a['gain_rank'],'new_gain_rank':b['gain_rank'],
            'rank_improvement':a['gain_rank']-b['gain_rank'],'old_gain':a['gain'],'new_gain':b['gain'],
            'old_gain_share_percent':100*a['gain_share'] if a['gain_share'] is not None else None,
            'new_gain_share_percent':100*b['gain_share'] if b['gain_share'] is not None else None,
            'old_split':a['split_count'],'new_split':b['split_count']})
        n=data['parts']['final_train']['rows'];count=data['parts']['final_train']['missing'][feature]
        row={'feature':feature,'old_train_missing':old_missing[feature],'old_train_missing_percent':100*old_missing[feature]/old_data['rows'],
            'new_final_train_missing':count,'new_final_train_missing_percent':100*count/n,
            'unchanged_formula_intentionally_withheld':feature in ('going','previous_time_seconds')}
        for year,item in data['by_year'].items():
            row[year+'_missing']=item['missing'][feature];row[year+'_missing_percent']=100*item['missing'][feature]/item['rows']
        missing.append(row)
    params=[]
    for key in sorted(set(old_meta['parameters'])|set(new_meta['parameters'])):
        a=old_meta['parameters'].get(key);b=new_meta['parameters'].get(key)
        params.append({'parameter':key,'old_D39':a,'new_D39':b,'changed':a!=b})
    final=data['parts']['final_train'];parts=data['parts'];source=c.read(directory/'source_manifest.json')
    preservation=c.verify(directory)
    result={'status':'ok','kind':'current_D39_council_report','run_id':directory.name,
        'models':{'old_D39':{'model_id':old_meta['model_id'],'path':str(c.OLD_MODEL),'temperature':old_meta['temperature'],
            'best_iteration':old_meta['best_iteration'],'spans':old_meta['spans'],'training_rows':old_data['rows'],'training_races':old_data['races']},
        'new_D39':{'model_id':new_meta['model_id'],'path':training['model_path'],'dataset_id':new_meta['dataset_id'],
            'temperature':new_meta['temperature'],'best_iteration':new_meta['best_iteration'],'spans':plan['spans'],
            'training_rows':final['rows'],'training_races':final['races']},
        'H':{'model_id':c.h.H_ID,'path':str(c.h.MODEL),'unchanged':True,'timing':'T10_observed_market_offset;T1'}},
        'data':{'latest_final_date':plan['latest_final_date'],'new_raw_months':len(source['new_months']),
            'new_raw_rows_fetched':source['new_rows'],'parts':parts,'eligible_by_year':data['by_year'],
            'added_final_training_races_vs_old':final['races']-old_data['races'],
            'added_final_training_rows_vs_old':final['rows']-old_data['rows'],
            'new_source_exclusions':data['new_source_exclusions']},
        'parameter_comparison':params,'importance':ranks,'missing':missing,'training':training,
        'other_changes':{'features_formulas_unchanged':True,'feature_order_unchanged':True,'history_lag_days':7,
            'Tuner_initial_parameters_and_8fit_method_unchanged':True,'category_vocabulary_refit_on_final_train_only':True,
            'independent_current_calibration':True,'final_fixed_round_refit_added':True,
            'causal_source_categories':'RA7_SE7','data_and_model_saving':'new_IDs_no_overwrite'},
        'prospective_comparison':{'status':'pending_future_results','common_races':0,'metrics':None,'ROI':None,
            'not_a_claim_of_improvement':True,'same_roster_same_T10_input_cutoff':True,'first_weekend':['2026-10-24','2026-10-25'],
            'Stage7_formal_evaluation_independent':True},'preservation':preservation,'automatic_adoption':False}
    csv_file(directory/'importance_comparison.csv',sorted(ranks,key=lambda r:r['new_gain_rank']))
    csv_file(directory/'parameter_comparison.csv',params)
    csv_file(directory/'missing_comparison.csv',missing)
    metrics=[]
    for name,value in training['calibration_diagnostic_only'].items():
        metrics.append({'model':name,'role':'2026_calibration_diagnostic_NOT_unused_test',
                        **{k:v for k,v in value.items() if k!='calibration_bands'}})
    csv_file(directory/'calibration_diagnostic.csv',metrics)
    c.write(directory/'council_report.json',result)
    lines=['# 新D39：2026年確定データ追加学習','',
        '新モデルは研究候補です。主力Hや既存タスクへの自動置換は行っていません。',
        f"最新確定日 {plan['latest_final_date']}。追加取得 {source['new_rows']:,}行／{len(source['new_months'])}か月。",
        f"旧D39学習 {old_data['races']:,}R・{old_data['rows']:,}頭 → 新D39最終学習 {final['races']:,}R・{final['rows']:,}頭。",
        f"追加 {final['races']-old_data['races']:,}R・{final['rows']-old_data['rows']:,}頭。較正・早期終了期間の行を単純合算していません。",'',
        '| 区分 | 開始 | 終了 | レース | 頭数 |','|---|---|---|---:|---:|']
    for name,item in parts.items():lines.append(f"| {name} | {item['start']} | {item['end']} | {item['races']:,} | {item['rows']:,} |")
    lines+=['',f"モデルID `{new_meta['model_id']}`、データセットID `{new_meta['dataset_id']}`。",
        f"Tuner {training['tuner_fits']}fit＋最終再学習 {training['final_refits']}fit。木数 {training['selected_rounds']}、温度 {training['temperature']}。",
        f"実train呼出合計 {training['training_seconds']:.3f}秒。保存再読込の最大確率差 {training['reload_max_probability_difference']:.3g}。",'',
        '## 診断と将来評価の区別','',
        '以下は新モデルの温度選択に使った較正データです。未使用評価・実力向上の証明には使いません。Hは同時点の市場入力が必要なため、この表へ異条件で混ぜていません。','',
        '| モデル | 較正用レースLL | Brier | ROC-AUC | 1位的中率% |','|---|---:|---:|---:|---:|']
    for row in metrics:lines.append(f"| {row['model']} | {row['race_log_loss']:.6f} | {row['brier_score']:.6f} | {row['roc_auc']:.6f} | {row['top1_accuracy_percent']:.2f} |")
    lines+=['','旧D39・新D39・Hの同一レース・同一T−10時点の前向き成績は、開催前の現在は未測定（0レース）です。ROIも未算出です。',
        '2025・2026年の学習済みレースを新モデルの未使用テストとして評価していません。Stage 7の途中成績を読み込んでいません。',
        '学習データ以外に、Tunerで選ばれるパラメータ・木数・カテゴリ辞書・温度・最終再学習の実施が変わっています。データ追加だけの効果を分離した実験とは呼びません。','',
        '## Gain順位（全39項目）','','| 特徴量 | 旧順位 | 新順位 | 新Gain比% |','|---|---:|---:|---:|']
    for row in sorted(ranks,key=lambda r:r['new_gain_rank']):lines.append(f"| {row['feature']} | {row['old_gain_rank']} | {row['new_gain_rank']} | {row['new_gain_share_percent']:.3f} |")
    lines+=['','going・previous_time_secondsは既存の因果性契約により引き続き全欠損です。値を埋めるための特徴量変更は行っていません。',
        '年別・項目別の欠損率は missing_comparison.csv、全パラメータ差は parameter_comparison.csv を参照してください。','',
        f"旧私有成果物 {preservation['old_private_files_unchanged']}件、既存コード等 {preservation['old_source_files_unchanged']}件のハッシュ一致。タスク定義も不変。",'',
        '次は独立した記録器を開催中に起動し、旧D39・新D39・Hの全予測を結果確定前に保存します。モデル昇格は評議会判断待ちです。']
    with (directory/'council_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    return {'status':'ok','run_id':directory.name,'report':str(directory/'council_report.md'),
        'new_model_id':new_meta['model_id'],'latest_final_date':plan['latest_final_date'],
        'final_training_races':final['races'],'final_training_runners':final['rows'],
        'added_races':final['races']-old_data['races'],'added_runners':final['rows']-old_data['rows'],
        'future_comparison_status':'pending_future_records','preservation':preservation}
