"""Anonymous development summaries and immutable candidate/future-protocol manifests."""
from datetime import date,datetime
import json
from pathlib import Path

from ..ml import safety
from ..win_sim import core,observed
from . import contracts as c,io,metrics


def retention(comparison,year_differences):
    """Finite development selection; this never grants a formal final-evaluation claim."""
    if len(year_differences)!=3:raise safety.MLFailure('invalid_input')
    improved=sum(v<0 for v in year_differences)
    keep=comparison['difference_new_minus_baseline']<0 and improved>=2
    return {'retain_development_candidate':keep,'improved_years':improved,
        'pooled_difference':comparison['difference_new_minus_baseline'],
        'uncertainty':comparison['interpretation'],
        'formal_improvement_certified':False,'final_evaluation_executed':False,
        'selection_rule':'negative_pooled_race_LL_difference_and_at_least_2_of_3_years_improve'}


def future_protocol(selected):
    return {'format_version':2,'status':'locked_not_executed_requires_separate_future_instruction',
        'evaluation_start':'2026-01-01','evaluation_end':'2026-09-30',
        'training_end':'2025-08-31','validation':['2025-09-08','2025-10-31'],
        'calibration':['2025-11-08','2025-12-24'],
        'history_lag_days':7,'anchor_training_start':'2020-01-01',
        'comparisons':[
            {'family':'ability','candidate':selected['ability_candidate'],
             'comparator':'same_split_retrained_legacy_features'},
            {'family':'market_anchor','candidate':selected['anchor_candidate'],
             'comparator':selected['market_control']}],
        'primary_metric':'race_winner_log_loss','difference_sign':'new_minus_comparator',
        'bootstrap_unit':'Monday_starting_race_week_all_venues','bootstrap_repetitions':1000,
        'primary_comparisons':2,'confidence_each':.975,'multiplicity':'Bonferroni',
        'formal_improvement_requires':'CI_upper_strictly_below_zero',
        'uncertain_result':'report_uncertain','same_race_and_runner_set_required':True,
        'paper_policy':{'threshold':'.05','stake_yen':100,'max_bets_per_race':1},
        'no_threshold_or_stake_tuning':True,'executable_entry_point':'scripts/stage2_final.py',
        'entry_point_currently_enabled':False}


def pooled_paper(directory,records):
    rows=[]
    for record in records:
        documents=io.read(directory/'trials',record['ledger_file'])
        rows.extend(row for document in documents for row in document['rows'])
    value=core.summarize(rows)
    value['risk']=observed.risk_metrics(rows)
    value['timing_basis']='JRA_publication_posthoc_not_PC_observation'
    value['schedule_asof_verified']=False
    value['coverage_denominator']='quote_eligible_population;see_evaluation_coverage_for_all_races'
    return value


def definitions():
    return {'format_version':2,'formula_version':c.FORMULA_VERSION,'input_format':c.INPUT_VERSION,
        'legacy_features':list(c.LEGACY_FEATURES),'legacy_definitions_unchanged':True,
        'A':{'actual_runners_source':'RA.shusso_tosu;past_only_not_registered_count',
             'a_previous_finish_percentile':'(past_actual_starters-past_rank)/(past_actual_starters-1)',
             'a_recent3_finish_percentile':'weighted_mean_of_available_last3_with_weights_1_0.5_0.25'},
        'B':{'age_specific_condition':'raw_2/3/4/5plus_condition_for_horse_age',
             'codes':'001..100,701,702,703,999;000_initial_missing;unknown_missing',
             'grade':'A..H,L;blank_named_GENERAL_OR_INITIAL;not_ordinal',
             'transitions':'past_raw_code>current_raw_code;missing_if_either_missing',
             'revision':'current_grade_or_jockey_masked_if_previous_revision_field_present'},
        'C':{'late3f':'raw_three_ASCII_digits/10_seconds;000/999_missing;obstacle_1F_not_used',
             'relative_late3f':'all_actual_starters_required;average_tied_ranks_and_full_race_mean',
             'corner4':'(past_actual_starters-past_corner4_rank)/(past_actual_starters-1)',
             'margin':'preceding_horse_gap_only;not_gap_to_winner',
             'numeric_margin':'only_official_exact_fraction/length_codes;H/K/T/Z_stay_categorical',
             'additional_margin':'presence_flag_only;disqualification/demotion_supplements_not_added'},
        'D':{'population':'all_JRA_01_10_actual_starts_in_source_not_target_horse_filtered',
             'window':'365_days_through_target_date_minus_7_days',
             'rate':'(person_wins+20*lagged_full_population_win_rate)/(person_starts+20)',
             'cancelled':'anomaly1/2/3_not_starters;DNF4_and_disqualified5_are_starters',
             'jockey':'actual_final_historical_jockey_after_7_day_lag',
             'scope_limit':'JRA_domestic_10_venues;foreign/local_racing_not_in_denominator'},
        'market':{'current':'normalized_inverse_odds_q,rank,log(odds),sum_inverse_odds',
             'history':'log(q_T10)-log(q_T30),rank_T30-rank_T10,actual_elapsed_minutes,missing_indicator',
             'offset':'Dataset.init_score=logit(q)_in_train_and_validation;prediction=sigmoid(raw_residual+logit(q));race_normalize',
             'zero_residual':'reproduces_q','temperature':1.,'best242_predictions_not_inputs':True},
        'availability':'all_history_available_only_at_past_race_date_plus_7_days',
        'snapshot_limit':'past_original_ingestion_times_and_revision_versions_unverified',
        'optional_unknowns':'remain_missing;never_filter_evaluation_population',
        'unused_legacy_all_missing':['going','previous_time_seconds']}


def trial_summary(record):
    result={k:record[k] for k in ('fold','configuration','model_id','feature_spec','split','model_manifest_sha256',
        'feature_manifest_sha256','plan_sha256','best_iteration','temperature','reload_max_difference',
        'periods','pooled','training_counts','training_missing','exploration_only')}
    if 'pooled_versus_control' in record:result['pooled_versus_control']=record['pooled_versus_control']
    return result


def cohort_report(directory,split):
    from .experiment import load_cache,key_parts
    from . import market_data
    from collections import Counter
    periods=[];missing=[]
    for start,end in split['evaluation']:
        frame=load_cache(directory,date.fromisoformat(start),date.fromisoformat(end))
        packets=market_data.load(directory,date.fromisoformat(start),date.fromisoformat(end))
        keys={key_parts(str(k))[:-1] for k in frame['entry_key']}
        good={r for r in keys if packets.get(r,{}).get('quotes',{}).get('10')}
        early={r for r in good if packets[r]['quotes']['30']}
        reasons=Counter(packets.get(r,{}).get('failures',{}).get('10','packet_missing') for r in keys-good)
        periods.append({'start':start,'end':end,'base_races':len(keys),'base_runners':len(frame),
            'market_races':len(good),'market_coverage_percent':100*len(good)/len(keys),
            'early_complete_races':len(early),'primary_exclusion_reasons':dict(reasons),
            'early_missing_races_retained_in_primary':len(good-early)})
        missing.append({'start':start,'end':end,'rows':len(frame),
            'features':{k:{'missing':int(frame[k].isna().sum()),'percent':100*int(frame[k].isna().sum())/len(frame)}
                for k in c.ALL_FEATURES}})
    return periods,missing


def main_market_comparison(directory,names,selected_control):
    from .experiment import read_trial,saved_predictions,load_cache,winners,predictions
    from . import market_data,model
    split=io.read(directory,'plan.json')['main_split'];raw=[];recal=[];labels={}
    t=io.read(directory/'trials','main_market_calibration.json')['temperature']
    for start,end in split['evaluation']:
        frame=load_cache(directory,date.fromisoformat(start),date.fromisoformat(end))
        frame=market_data.augment(frame,market_data.load(directory,date.fromisoformat(start),date.fromisoformat(end)))
        raw.extend(predictions(frame,frame['market_probability'].to_numpy()))
        recal.extend(predictions(frame,model.normalize(frame,frame['market_probability'].to_numpy(),t)))
        labels.update(winners(frame))
    control=[p for part in saved_predictions(directory,read_trial(directory,'main',selected_control)) for p in part]
    result={'raw_market':metrics.measure(raw,labels),'recalibrated_market':metrics.measure(recal,labels),
        'recalibrated_market_temperature':t,'selected_market_information_control':selected_control,'configurations':{}}
    for name in names:
        p=[p for part in saved_predictions(directory,read_trial(directory,'main',name)) for p in part]
        result['configurations'][name]={'metrics':metrics.measure(p,labels),
            'versus_selected_market_control':metrics.compare(p,control,labels),
            'versus_raw_market':metrics.compare(p,raw,labels),
            'versus_recalibrated_market':metrics.compare(p,recal,labels)}
    return result


def freeze(directory):
    from .experiment import read_trial,saved_predictions,load_cache,winners,anchor_specs
    if (directory/'candidate_freeze.json').exists():raise safety.MLFailure('artifact_invalid')
    annual=io.read(directory,'annual_results.json');selected=annual['candidates'];plan=io.read(directory,'plan.json')
    records=[]
    for path in sorted((directory/'trials').glob('*.json')):
        value=io.read(path.parent,path.name)
        if isinstance(value,dict) and 'configuration' in value and 'model_id' in value:records.append(value)
    if len(records)>28:raise safety.MLFailure('invalid_input')
    labels={};comparisons={};candidate_models={};paper={}
    for year in (2023,2024,2025):
        labels.update(winners(load_cache(directory,date(year,1,1),date(year,9,30))))
    for family,a,b in (('ability','annual_ability','annual_control'),('market_anchor','annual_anchor','annual_market_control')):
        candidates=[read_trial(directory,str(y),a) for y in (2023,2024,2025)]
        controls=[read_trial(directory,str(y),b) for y in (2023,2024,2025)]
        p=[p for rec in candidates for part in saved_predictions(directory,rec) for p in part]
        q=[p for rec in controls for part in saved_predictions(directory,rec) for p in part]
        comp=metrics.compare(p,q,labels)
        comparisons[family]={'candidate':metrics.measure(p,labels),'control':metrics.measure(q,labels),'comparison':comp,
            'selection':retention(comp['race_week_cluster'],[r[family.replace('market_anchor','anchor')+'_comparison']['race_week_cluster']['difference_new_minus_baseline'] for r in annual['years']])}
        candidate_models[family]={'candidate_models':[r['model_id'] for r in candidates],
            'control_models':[r['model_id'] for r in controls]}
        paper[family]={'candidate':pooled_paper(directory,candidates),'control':pooled_paper(directory,controls)}
    ablations=io.read(directory,'ability_full_done.json')['conditional_ablations']
    ability_names=['control','A','B','AB','C','D','ABCD',*ablations]
    anchor_names=list(anchor_specs())
    main_ability=[read_trial(directory,'main',n) for n in ability_names]
    main_anchor=[read_trial(directory,'main',n) for n in anchor_names]
    main_paper={r['configuration']:pooled_paper(directory,[r]) for r in main_ability+main_anchor}
    coverage,missing=cohort_report(directory,plan['main_split'])
    mainmarket=main_market_comparison(directory,anchor_names,selected['market_control'])
    preserved=io.verify_preserved(directory)
    source=io.read(directory,'source_manifest.json');features=io.read(directory,'feature_manifest.json')
    if source['source_end']>='2026-01-01' or source['sealed_2026_rows_read'] or features['sealed_2026_rows']:
        raise safety.MLFailure('invalid_split')
    seal={'evaluated':False,'labels_read':0,'source_max_date':source['source_end'],
        'development_accepts_2026_dates':False,'annual_years':[2023,2024,2025],
        'all_2025_and_annual_results':'development_exploration_not_new_unused_test',
        'final_entry_point':'separate_disabled_script','original_2026_artifacts_only_hashed_not_evaluated':True}
    frozen={'format_version':2,'created_at_jst':datetime.now(core.JST).isoformat(),
        'work_id':directory.name,'plan_sha256':safety.digest(directory/'plan.json',root=c.ROOT),
        'development_selection':selected,'development_judgements':comparisons,
        'ability':{'recipe':selected['ability_spec'],'selected_main_model_id':read_trial(directory,'main',selected['ability_candidate'])['model_id'],
            'active_development_baseline':'candidate' if comparisons['ability']['selection']['retain_development_candidate'] else 'legacy_control',
            **candidate_models['ability']},
        'market_anchor':{'recipe':selected['anchor_spec'],'selected_main_model_id':read_trial(directory,'main',selected['anchor_candidate'])['model_id'],
            'control_recipe':selected['market_control_spec'],
            'active_development_baseline':'candidate' if comparisons['market_anchor']['selection']['retain_development_candidate'] else selected['market_control'],
            **candidate_models['market_anchor']},
        'training_parameters':{'ability':plan['ability_parameters'],'market_offset':plan['anchor_parameters']},
        'calibration':{'ability':plan['ability_calibration'],'anchor':plan['anchor_calibration']},
        'cohorts':{'ability':'all_legacy_eligible_races_optional_missing_retained','market':'exact_same_T10_odds_runner_set',
                  'early_history_missing':'retained_primary;separate_complete_subset'},
        'paper_policy':plan['paper_policy'],'old_artifacts':preserved,'sealed_period':seal,
        'model_manifests':{r['model_id']:r['model_manifest_sha256'] for r in records},
        'no_automatic_production_adoption':True,'no_more_search_in_this_run':True}
    report={'format_version':2,'primary_metric':'race_winner_log_loss','difference_sign':'new_minus_comparator',
        'main_ability':[trial_summary(r) for r in main_ability],
        'main_market':mainmarket,'annual':annual,'annual_pooled':comparisons,
        'main_coverage':coverage,'feature_missing_evaluation':missing,
        'main_paper':main_paper,'annual_paper':paper,'old_artifacts':preserved,'sealed_period':seal,
        'reproduction':io.read(directory,'reproduction.json'),'cache_parity':io.read(directory,'cache_legacy_parity.json'),
        'source_summary':{k:v for k,v in source.items() if k!='months'},
        'market_coverage_by_year':io.read(directory,'market_manifest.json')['years'],
        'limitations':['JRA_publication_posthoc_not_PC_receipt','schedule_original_asof_version_unverified',
            'source_final_snapshot_revision_receipt_times_unverified','JRA_domestic_10_venue_person_population',
            'single_winner_complete_rank_legacy_eligibility_retained','all_evaluations_development_only'],
        'history_complete_subset_file':'trials/main_history_complete_comparison.json'}
    io.write(directory,'experiment_ledger.json',{'count':len(records),'all_trials':[trial_summary(r) for r in records],
        'no_failed_training_trials':True,'maximum_trials':28,'no_further_search':True})
    io.write(directory,'feature_definitions.v2.json',definitions())
    io.write(directory,'development_report.json',report)
    io.write(directory,'candidate_freeze.json',frozen)
    io.write(directory,'final_evaluation_plan.json',future_protocol(selected))
    io.write(directory,'old_artifacts_verified.json',preserved)
    path=safety.local_path(directory/'development_report.md',root=c.ROOT)
    with path.open('x',encoding='utf-8') as stream:stream.write(markdown(report,selected))
    freezehash=safety.digest(directory/'candidate_freeze.json',root=c.ROOT)
    io.write(directory,'freeze_checksum.json',{'candidate_freeze_sha256':freezehash,
        'final_plan_sha256':safety.digest(directory/'final_evaluation_plan.json',root=c.ROOT),
        'development_report_sha256':safety.digest(directory/'development_report.json',root=c.ROOT)})
    return {'trial_count':len(records),'ability_selection':comparisons['ability']['selection'],
        'anchor_selection':comparisons['market_anchor']['selection'],
        'old_artifacts':preserved,'sealed_2026_evaluated':False,'report':str(path),
        'candidate_freeze_sha256':freezehash}


def number(v,digits=5):
    return '未算出' if v is None else f'{v:.{digits}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])+'\n\n'


def markdown(report,selected):
    out=['# Stage 2・限定Stage 1.5 開発評価と候補凍結\n\n',
        '主指標はレース単位の勝馬Log Loss。差は「新モデル−対照」で、負値が改善です。'
        '2025年3期間と2023〜2025年の年次評価は候補選択に使用した開発評価です。'
        '2026年1〜9月の正解・性能・収益は開いていません。\n\n',
        '## 能力モデル：2025年の同一2,389レース\n\n']
    base=report['main_ability'][0]
    def difference(r):
        if 'pooled_versus_control' not in r:return '—'
        ci=r['pooled_versus_control']['race_week_cluster']
        return f"{number(ci['difference_new_minus_baseline'])} [{number(ci['ci_low'])}, {number(ci['ci_high'])}]"
    out.append(table(['構成','木数','T','レースLL','対照との差 [週95%CI]','馬LL','Brier','1位的中率%'],[
        [r['configuration'],r['best_iteration'],r['temperature'],number(r['pooled']['race_log_loss']),
         difference(r),number(r['pooled']['log_loss']),
         number(r['pooled']['brier_score']),number(r['pooled']['top1_accuracy_percent'],2)] for r in report['main_ability']]))
    out.append('## 市場アンカー：同時点・同一馬番集合\n\n')
    m=report['main_market'];rows=[]
    for label,key in [('無補正市場','raw_market'),('市場再較正','recalibrated_market')]:
        r=m[key];rows.append([label,r['races'],number(r['race_log_loss']),'—',number(r['log_loss']),number(r['brier_score'])])
    for name,row in m['configurations'].items():
        r=row['metrics'];ci=row['versus_selected_market_control']['race_week_cluster']
        rows.append([name,r['races'],number(r['race_log_loss']),f"{number(ci['difference_new_minus_baseline'])} [{number(ci['ci_low'])}, {number(ci['ci_high'])}]",number(r['log_loss']),number(r['brier_score'])])
    out.append(table(['構成','レース','レースLL','選択市場対照との差 [週95%CI]','馬LL','Brier'],rows))
    out.append(f"市場対照は `{selected['market_control']}`。市場再較正T={m['recalibrated_market_temperature']}は較正期間のみで選択。"
        'アンカーモデルのTは1.0固定で、best242のTを流用していません。残差はlogit市場offsetへ加算後、sigmoid・レース内正規化します。\n\n')
    out.append('## 年次開発評価と候補判定\n\n')
    rows=[]
    for r in report['annual']['years']:
        for family,a,b,comp in [('能力','ability_candidate','ability_control','ability_comparison'),('市場アンカー','anchor_candidate','market_control','anchor_comparison')]:
            ci=r[comp]['race_week_cluster'];rows.append([r['year'],family,r[a]['races'],number(r[b]['race_log_loss']),number(r[a]['race_log_loss']),
                f"{number(ci['difference_new_minus_baseline'])} [{number(ci['ci_low'])}, {number(ci['ci_high'])}]"])
    out.append(table(['年','比較','レース','対照LL','候補LL','差 [開催週95%CI]'],rows))
    for family,label in [('ability','能力'),('market_anchor','市場アンカー')]:
        value=report['annual_pooled'][family];ci=value['comparison']['race_week_cluster'];s=value['selection']
        out.append(f"{label}：年次プール差 {number(ci['difference_new_minus_baseline'])}、95%CI [{number(ci['ci_low'])}, {number(ci['ci_high'])}]。"
            f"改善方向は{s['improved_years']}/3年。{'開発候補として保持' if s['retain_development_candidate'] else '既存対照を基準として保持'}。正式な改善認定は行いません。\n\n")
    out.append('## 評価対象・欠損\n\n')
    out.append(table(['期間','基本レース','頭数','市場共通','被覆率%','T−30あり','早期欠損のまま保持'],[
        [r['start']+'〜'+r['end'],r['base_races'],r['base_runners'],r['market_races'],number(r['market_coverage_percent'],2),r['early_complete_races'],r['early_missing_races_retained_in_primary']] for r in report['main_coverage']]))
    out.append('全特徴量の欠損件数・率は `development_report.json.feature_missing_evaluation`、学習側は `experiment_ledger.json` に保存。'
        '任意特徴量欠損では基本母集団を変更していません。除外理由、馬番集合一致、T−30完全履歴部分集合の結果も別記録です。\n\n')
    out.append('## 固定条件の仮想収支診断\n\n')
    rows=[]
    for family,label in [('ability','能力'),('market_anchor','市場アンカー')]:
        for variant,vlabel in [('control','対照'),('candidate','候補')]:
            r=report['annual_paper'][family][variant];a=r['ai'];risk=r['risk']
            rows.append([label+vlabel,a['bet_count'],a['investment_yen'],a['payout_yen'],a['net_profit_yen'],number(a['return_rate_percent'],2),a['unsettled_count'],risk['max_drawdown_yen'],number(risk['largest_payout_share_percent'],2)])
    out.append(table(['年次3年合計','購入','投資円','払戻円','純利益円','回収率%','未精算','最大DD円','最大払戻依存%'],rows))
    out.append('未精算がある場合は全体純利益・回収率を確定せず、JSONのsettled_subsetを参照。月次収支と最大払戻を除いた純利益も保存。'
        'EV5%以上・100円・最大1点は全構成で固定し、閾値や賭け金を最適化していません。\n\n')
    out.append('## 期間別性能・校正と月次収支\n\n')
    for name in ('control',selected['ability_candidate']):
        r=next(x for x in report['main_ability'] if x['configuration']==name)
        out.append(f'### 能力 {name}\n\n')
        out.append(table(['期間','レースLL','馬LL','Brier','1位的中率%'],[
            [p['start']+'〜'+p['end'],number(p['ability_population']['race_log_loss']),number(p['ability_population']['log_loss']),number(p['ability_population']['brier_score']),number(p['ability_population']['top1_accuracy_percent'],2)] for p in r['periods']]))
        out.append(table(['確率帯','頭数','平均確率','実勝率','予測/実測'],[
            [v['probability_band'],v['count'],number(v['mean_probability']),number(v['observed_rate']),number(v['predicted_to_observed_ratio'])] for v in r['pooled']['calibration']]))
    rows=[]
    for family,label in [('ability','能力'),('market_anchor','市場アンカー')]:
        for variant,vlabel in [('control','対照'),('candidate','候補')]:
            for month,r in report['annual_paper'][family][variant]['by_month'].items():
                a=r['ai'];rows.append([label+vlabel,month,a['bet_count'],a['net_profit_yen'],number(a['return_rate_percent'],2),a['unsettled_count']])
    out.append(table(['構成','月','購入','純利益円','回収率%','未精算'],rows))
    out.append('## 保持・制約・次の操作\n\n')
    out.append(f"旧成果物{report['old_artifacts']['checked_files']}ファイルのSHA-256を開始前後で照合し、変更0件。Stage 1再現差"
        f"{report['reproduction']['stage1_reproduction_max_difference']}、旧best242予測再現差{report['cache_parity']['maximum_probability_difference']}。\n\n")
    out.append('公表時刻を使用した事後評価です。PC実取得時刻、通信遅延、発走予定時刻・訂正の当時の版は未検証。'
        '時刻変更記録ありは除外し、分精度は公表分の末尾がT−10以前・鮮度300秒以内のみを採用。'
        '人物統計の母集団は国内JRA10場です。取消・不完走・同着等の基本評価除外は旧比較条件を維持しています。\n\n')
    out.append('候補は `candidate_freeze.json`、将来の最終比較は `final_evaluation_plan.json` に固定。'
        '2つの主要比較は各97.5%区間（Bonferroni）、差の上限が0未満の場合だけ正式改善を認定します。'
        '最終評価コマンドは別入口で無効化されており、今回は実行していません。これ以上の探索を行わず、別途指示を待ちます。\n')
    return ''.join(out)
