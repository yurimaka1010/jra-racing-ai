"""Anonymous preparation report and an unapproved, bounded next-fit proposal."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import timedelta
import csv
import hashlib
import json

from . import contracts as c, data
from .features import cell, stable_id
from .preparation import raw_days, verify


def write_csv(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finalize(directory):
    prepared=c.read(directory/'preparation.json'); blood=c.read(directory/'pedigree_summary.json')
    replay=c.read(directory/'input_replay.json'); links=c.read(directory/'link_summary.json')
    controls=c.read(directory/'control_reuse.json')
    reference_model=c.read(Path(c.read(c.REFERENCE/'candidate_freeze.json')['model_path'])/'model.json')
    training_plan=c.read(c.REFERENCE/'plan.json')
    tests_path=directory/'regression.txt'
    import re
    match=re.search(r'Ran (\d+) tests',tests_path.read_text(encoding='utf-8-sig'))
    if not match or '\nOK' not in tests_path.read_text(encoding='utf-8-sig'):raise ValueError('tests_incomplete')
    # Check projected matrices against the same 39-column slice, not a different builder.
    from ..d39_enrichment import data as reference_data
    import pandas as pd
    comparisons={};cats={k:v for k,v in reference_model['categories'].items() if k!='e_rest_stage'}
    for part in ('final_train','calibration'):
        frame=reference_data.load_part(c.REFERENCE,part)
        control=data.load_part(directory,part);candidate=data.load_part(directory,part,pedigree=True)
        expected=reference_data.matrix(frame,reference_model['feature_spec'],reference_model['categories'])[data.spec()['features']]
        actual=data.matrix(control,data.spec(),cats)
        pd.testing.assert_frame_equal(expected,actual,check_exact=True)
        pd.testing.assert_frame_equal(actual,data.matrix(candidate,data.spec(True),cats)[actual.columns],check_exact=True)
        comparisons[part]={'rows':len(frame),'reference_39_encoded_columns_identical':True,
            'control_and_pedigree_common_columns_identical':True,
            'category_dictionary_identical_to_reference_subset':True,
            'entry_key_order_and_labels_identical':True}
    c.write(directory/'projected_inputs_verification.json',comparisons)
    # Target-cell counts and unique offspring, without reading a new result source.
    lineage=c.read(directory/'lineage.local.json')
    targets={part:set(data.load_part(directory,part).entry_key) for part in ('final_train','calibration')}
    ids=defaultdict(set);parent_ids=defaultdict(set)
    from ..research.features import legacy
    for rows in raw_days():
        for row in rows:
            key=''.join(legacy.entry_identity(row)[0]);condition=cell(row.get('surface'),row.get('distance')) or 'unsupported'
            for part, wanted in targets.items():
                if key not in wanted:continue
                ids[(part,condition)].add(row['horse_id'])
                for role in ('sire','damsire'):
                    parent=lineage.get(row['horse_id'],{}).get(role)
                    if parent:parent_ids[(part,condition,role)].add(parent)
    cells=[]
    for row in blood['condition_cells']:
        v=dict(row);part,condition,role=v['part'],v['cell'],v['role']
        v.update(unique_target_offspring=len(ids[(part,condition)]),distinct_parent_ids=len(parent_ids[(part,condition,role)]),
                 with_support_percent=100*v['with_support']/v['target_rows'],
                 mean_support=v['support_sum']/v['target_rows'])
        cells.append(v)
    write_csv(directory/'condition_cell_counts.csv',cells)
    registry=[{'feature':f,'meaning':{'level':'全条件の縮小成績水準','condition_deviation':'今回条件の成績偏差（正が良い）','support':'本馬以外の条件セル支持産駒数'}[f.split('_',2)[2]],
               'role':'父' if '_sire_' in f else '母父','formula_version':c.VERSION,'available_lag_days':7,
               'missing_rows_final_train':0,'missing_rows_calibration':0,'dtype':'float64',
               'current_model_input':False} for f in c.FEATURES]
    write_csv(directory/'feature_registry.csv',registry)
    # Existing WE history cannot support historical learning; observed snapshots can
    # be preserved for a later separately approved dataset, never retroactively filled.
    source=c.read(c.REFERENCE/'source_manifest.json');we=c.read(c.REFERENCE/'WE_inventory.json')
    snapshots=list((c.REFERENCE/'going-observed').glob('*/*.json'));races=set()
    for path in snapshots:
        v=c.read(path)
        if v.get('all_observed_before_T10') is True:
            for r in v['rows']:
                races.add(tuple(r[k] for k in ('kaisai_nen','kaisai_tsukihi','keibajo_code','kaisai_kai','kaisai_nichime','race_bango')))
    going={'historical_WE_rows':source['WE_rows'],'inventory':we['years'],
           'reference_training_missing_rows':196841,'reference_calibration_missing_rows':1747,
           'verified_PC_observed_snapshot_files':len(snapshots),'verified_PC_observed_races':len(races),
           'new_collection_or_tasks_started':False,'historical_final_RA_substituted':False,
           'current_model_can_learn_going_effect':False}
    c.write(directory/'going_availability.json',going)
    # Fixed zero-fit review of source/identity constraints.
    sources=c.read(directory/'identity_manifest.json')
    hn=[]
    for item in sources['parts']:
        if 'jvd_hn_' in item['file']:hn.extend(c.read(directory/item['file'])['rows'])
    aliases=defaultdict(set)
    for row in hn:
        h=stable_id(row.get('ketto_toroku_bango'))
        if h:aliases[h].add(row['hanshoku_toroku_bango'])
    alias_summary={'known_horses_with_multiple_breeding_ids':sum(len(v)>1 for v in aliases.values()),
                   'alias_canonicalization':'none;exact_referenced_ID_only;fragmentation_possible',
                   'HN_horse_registration_unknown_rows':sum(stable_id(v.get('ketto_toroku_bango')) is None for v in hn)}
    c.write(directory/'identity_limitations.json',alias_summary)
    first_week=c.clock().date()+timedelta(days=(5-c.clock().weekday())%7)
    proposal={'status':'proposal_only_requires_new_human_approval','real_fits_authorized':0,
        'created_at':c.clock().isoformat(),'reference47_model_id':reference_model['model_id'],
        'reference47_reuse':True,'reference47_refits':0,
        'new_models':[{'name':'same_condition_control39','features':data.spec()['features'],'fits':1},
                      {'name':'same_condition_control39_plus_pedigree6','features':data.spec(True)['features'],'fits':1}],
        'total_fits_max':2,'failures_consume_budget':True,'automatic_retry':False,'Optuna_or_Tuner_trials':0,
        'parameters':reference_model['parameters'],'rounds':219,'final_training':'reuse_exact_final_train_rows;early_stop_window_already_included_identically_to_reference47',
        'training_spans':training_plan['spans'],'training_actual_start':'2022-01-05','training_actual_end':'2026-09-20',
        'calibration_actual_start':'2026-10-10','calibration_actual_end':'2026-10-18',
        'calibration':{'candidates':training_plan['calibration']['grid'],'selection':'minimum_race_log_loss;separately_for_each_model;calibration_only'},
        'budget':{'CPU_threads':4,'private_bytes_max':12*1024**3,'per_fit_seconds_max':300,'whole_learning_seconds_max':900,'whole_preparation_seconds_max':900},
        'runtime_estimate':'two_219_tree_fits;seconds_to_minutes_expected;hard_cap15min;not_a_measured_2fit_runtime',
        'data_identity':{part:c.sha(c.REFERENCE/(part+'.local.csv')) for part in ('final_train','calibration')},
        'pedigree_spec_sha256':c.sha(directory/'feature_spec.json'),
        'identity_source_manifest_sha256':c.sha(directory/'identity_manifest.json'),
        'readiness_conditions':['human_approval_of_2fits_and_evaluation_rules','accept_or_resolve_static_master_revision_asof_limitation',
                              'no_new_input_inconsistency_requiring_reference47_refit'],
        'forward_evaluation':{'status':'proposed_not_started','primary':'pooled_ROC_AUC_pedigree45_minus_control39',
            'secondary':['within_race_auc','race_log_loss','brier_score','top1_hit_rate','calibration_table'],
            'rest8_comparison':'frozen47_minus_same_condition39;diagnostic_secondary',
            'difference':'new_minus_comparator;positive_AUC_better;negative_LL_Brier_better',
            'population':'same_races_same_complete_roster_formal_T10_predictions_saved_after_candidate_freeze;no_H_outcomes',
            'cutoff':'same_existing_T10;conservative_minute_end;freshness_max300s;no_changed_schedule',
            'window':'21_consecutive_calendar_weeks_from_first_full_JRA_week_after_both_candidate_freezes;no_extension_after_results',
            'earliest_possible_start':str(first_week),'earliest_possible_end':str(first_week+timedelta(weeks=20,days=1)),
            'exact_dates_fixed_at_freeze_before_any_new_results':True,
            'final_performance_openings_max':1,'interim_model_selection':False,
            'bootstrap':{'reuse':'stage5.metrics.WeekBootstrap;explicit_sample_indices_from_seed20261026','resamples':5000,'seed':20261026,'confidence':.95,'unit':'same_Monday_week;all_same_day_races_together;paired'},
            'proposed_acceptance':{'pooled_AUC_point_gain_min':.002,'pooled_AUC_CI_lower_strictly_above':0,
                'within_race_AUC_point_loss_max':.001,'race_LL_point_increase_max':.005,'Brier_point_increase_max':.0002,
                'top1_hit_rate_and_calibration':'reported;not_replacing_primary_gate'},
            'newcomer_auxiliary':'current_declared_condition_701_all_roster;official_2007;not_past_starts_zero_proxy;not_primary;no_subgroup_renormalization',
            'small_horse_subgroups':'original_probabilities_binary_metrics_only_or_full_races;never_renormalize_selected_horses',
            'exclusions':'count_and_report_missing_forecasts_invalid_identity_incomplete_roster_unknown_or_irregular_official_labels;never_filter_on_pedigree_missingness',
            'known20_and_calibration133':'excluded_from_independent_test',
            'multiple_primary_tests':False,'formal_primary_contrast_count':1,
            'if_gate_fails':'do_not_promote;freeze_this_spec;no_cell_or_shrinkage_retuning',
            'power_or_required_race_count_claim':'not_estimated_from_known20;21weeks_is_fixed_design_not_power_guarantee'},
        'production_or_task_change_authorized':False,'combined47_plus_pedigree_fit_authorized':False}
    c.write(directory/'learning_plan.proposed.json',proposal)
    c.write(directory/'learning_plan.commit.json',{'sha256':c.sha(directory/'learning_plan.proposed.json'),
        'status':'unapproved_proposal','real_fit_authorization':False})
    result={'status':'stage1_complete_zero_fit','run_id':directory.name,'new_tests':26,'all_python_tests':int(match[1]),
        'real_fits':0,'existing_predictions_reproduced':prepared['input_replay'],'lineage':links,
        'going':going,'projection_verification':comparisons,'preservation':verify(directory),
        'next_stage_fits_proposed':2,'approval_still_required':True,'no_H_or_Stage7_outcomes_read':True,
        'new_2026_holdout_evaluation':False,'new_production_or_task_changes':False}
    c.write(directory/'completion.json',result)
    # Anonymous aggregate report. Detailed input/lineage CSVs must stay local.
    rate=100*links['counts']['sire_HN_linked']/links['target_horses']
    lines=['# 次世代D39・血統 第1段階 完了（0fit）','',
        '2026-10-26。Astra独立裁定と第1段階指示に基づく準備・検証。新規fit、Optuna、温度再較正、本番接続は0。',
        '', '## 保存入力の整合性','',
        '|モデル|正式T−10レース|頭数|生確率最大差|校正後最大差|','|---|---:|---:|---:|---:|']
    for model,v in prepared['input_replay'].items():lines.append(f"|{model}|{v['races']}|{v['rows']}|{v['max_raw_probability_difference']}|{v['max_final_probability_difference']}|")
    lines += ['', '元の出馬表スナップショットと固定された履歴キャッシュを既存Builderで再生成。モデルの47列の順序・型・カテゴリ辞書・結合キーを確認し、保存済み生確率／T適用後確率を再現した。結果・払戻は読まない。',
        '', '**検証不能**：当時の加工済み47列行列は未保存。全47列の保存値との直接比較はできない。古い履歴のPC実取得時刻・訂正前全版も保証できず、予測再現を厳密な全時点as-of証明へ拡張しない。',
        '', '前走タイムは「7日以上前の正常な同距離・同一トラックの直近有効タイム」。直前の1走とは限らない。指数化は行わない。',
        '着差1は前馬との隣接差。＋／＋＋は前馬の失格・降着時の追加差で、勝馬との差ではない。コード2102の特殊値を勝馬との差へ再解釈しない。既存符号化は変更しない。',
        'declared_runnersは登録頭数のまま。過去着順パーセンタイルは別項目actual_runnersと、過去全出走馬の整合を使う。履歴7日遅延・異常レース全体除外を継承。',
        '', '## 全欠損・定数と馬場','',
        '初期47のgoing／previous_time_secondsは全欠損。タイム接続47はgoingのみ全欠損で、前走タイムは学習94,976行、今回の再現対象156/306行で有効。追加着差有無は正常レースのみの履歴では非欠損値が定数0となる。凍結列の削除は行わない。',
        f"保存済みWEは{going['historical_WE_rows']}行（2026年のみ）。学習196,841行・較正1,747行でgoingを時点検証できず全欠損。10/25のPC実観測スナップショット{len(snapshots)}件・{len(races)}レースは保存されているが、歴史的学習値の代用にしない。新規タスク・収集は開始していない。",
        '', '## 血統の成立性','',
        f"保存済み学習履歴の異なる馬{links['target_horses']:,}頭を対象。UM {links['UM'].get('usable_ids',0):,}ID、照合したHN {links['HN'].get('usable_ids',0):,}ID。SKは対象馬に対して取得0行であり、SK全表が空と断定はしない。",
        f"父・母父とも{links['counts']['sire_HN_linked']:,}頭（{rate:.4f}%）がHNへ結合。43頭は未知。母のHN父IDとの照合は{links['counts']['maternal_path_agrees']:,}頭すべて一致、矛盾0。",
        'UM/SK血統欄01aが父、05aが母父。公式PDFのUM18・HN18/19・SK13と実DB列を照合。名前・現在累積成績は不使用。現在マスタの親子IDは静的研究IDとして使用し、過去の訂正前全版は未確認。',
        f"対象参照群内では、同一血統番号に複数繁殖IDを持つ馬{alias_summary['known_horses_with_multiple_breeding_ids']}頭。名前で統合しないためID分散の留保が残る。",
        '', '## 生成した6列と支持産駒','',
        '父・母父×全条件水準／今回条件偏差／条件セル支持産駒数。頭数補正着順パーセンタイルは高いほど良い。各産駒内平均→産駒間等重み、本馬除外。セル→全条件→遅延済み全体事前平均へ10／5の事前固定強度で縮小する。',
        '全体事前平均にも未来と本馬を含めない。未知血統は事前平均・偏差0・支持数0。平地芝／ダート6セル以外の条件も対象行は保持し、偏差0・支持0とする。',
        '学習196,841行・較正1,747行を全件維持。父・母父ID不明は学習各16行、較正0行。6列はいずれも欠損0（未知を補った事前平均であり、情報が全件存在する意味ではない）。',
        '', '|条件|学習対象行|支持あり父|支持あり母父|異なる対象産駒|','|---|---:|---:|---:|---:|']
    for condition in (*c.CELLS,'unsupported'):
        a=next(v for v in cells if v['part']=='final_train' and v['role']=='sire' and v['cell']==condition)
        b=next(v for v in cells if v['part']=='final_train' and v['role']=='damsire' and v['cell']==condition)
        lines.append(f"|{condition}|{a['target_rows']:,}|{a['with_support']:,}|{b['with_support']:,}|{a['unique_target_offspring']:,}|")
    lines += ['', '支持数中央値（予測行ごとの時点集計、本馬除外）：学習 父109頭／母父86頭、較正 父103頭／母父76頭。条件セルの件数・支持5頭未満・分位はCSV/JSONに保存。異常履歴4,846レースを全体除外。',
        '', '## 同条件対照と次の学習計画（未承認）','',
        '既存D39対照型は再利用不可。学習201,651行に対してタイム接続47は196,841行。履歴・共通列値・カテゴリ辞書・パラメータが異なり、木数も215対219、温度選定方針も異なる。旧対照は保持する。',
        'タイム接続47を参照して39列を投影すれば、対象行・順序・ラベル・共通列値・カテゴリ辞書は完全一致することを検証済み。',
        '**提案：39列対照1fit＋39列対照と血統6列の45列版1fit＝最大2fit。基準47の再学習0fit。** 同じ保存パラメータ・219本、学習2022-01-05〜2026-09-20、較正2026-10-10〜18。早期終了用窓を併合済みのfinal_trainをそのまま両方で使用。温度は0.75／1／1.25／2から別々に同じ窓で選ぶ。',
        'CPU4、12GiB、各fit最大300秒、学習全体900秒。追加探索・失敗再試行・休養＋血統の合成版・本番追加は含めない。新規血統の効果量と学習時間は未測定。',
        '', '## 前向き比較の事前登録案','',
        '主比較は血統45−同条件39の全体ROC-AUC増分。候補凍結後、同一T−10・同一全出走馬集合の共通レース。既知20レースと較正133レースは独立評価から除外。H／Stage7途中成績は使用しない。',
        '提案ゲート：AUC点推定+0.002以上、週ペアードbootstrap95%区間下限>0。許容悪化はレース内AUC0.001、Race LL0.005、Brier0.0002。値は提案であり、実行前の承認で固定する。1位的中率と較正も併記。',
        '両候補凍結後の最初の完全な開催週から連続21暦週、終了後1回だけ評価。最早開始案2026-10-31、終了案2027-03-21。凍結が遅れた場合は、結果を見る前に同じ規則で確定する。21週で十分な検出力があると断定していない。',
        '新馬戦は事前指定の補助評価。馬だけを抽出して勝率を再正規化しない。主比較は1つのため、この提案に多重主比較の追加はない。未達・不確かなら昇格せず同仕様を凍結。',
        '', '## 検証・保持・承認事項','',
        f"追加26件、既存を含むPython {match[1]}件すべて成功。テスト中は実ネットワーク・実DBドライバーを遮断。新規学習0fit。",
        '既存資産12、保護コード24、追加のモデル／計画／正式予測等201ファイルのハッシュ不変。既存タスク構成不変。本番KEIBA LABは変更せず、H結果・Stage7途中結果・封印評価を未開封。',
        '**次に必要な判断**：2fit予算、提案評価ゲート／終了規則、静的血統マスタの歴史的訂正版未確認という留保を確認して別途承認。追加fit・本番接続の承認はまだ受けていない。',
        '', '## ファイル','',
        '- input_replay.json：保存予測再現・型・欠損・検証不能事項（レースキー実値なし）',
        '- condition_cell_counts.csv / pedigree_summary.json：匿名セル件数と支持数',
        '- feature_registry.csv / feature_spec.json：6列の定義',
        '- projected_inputs_verification.json / control_reuse.json：比較条件',
        '- going_availability.json / identity_limitations.json：留保',
        '- learning_plan.proposed.json / learning_plan.commit.json：承認前2fit計画とSHA256',
        '- regression.txt / completion.json：検証と保持結果',
        '- *.pedigree.local.csv / lineage.local.json / identity-source：個別の研究用データ。PC外へ共有しない。',
        '', '公式仕様参照：ローカルJV-Data4901.pdf、PDFの物理ページ12（SE）、17（UM）、20（HN/SK）、32（追加着差）、46（コード2102）。外部通信なし。']
    with (directory/'completion_report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    return {'status':'stage1_complete_zero_fit','run_id':directory.name,'real_fits':0,'tests':int(match[1]),
            'new_tests':26,'proposed_next_stage_fits':2,'preservation':result['preservation']}
