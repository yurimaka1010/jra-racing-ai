from datetime import date,datetime
from pathlib import Path
import json
import hashlib,re
from ..ml import safety
from ..research import experiment as exp,market_data,report as oldreport,contracts as oldc
from ..win_sim import core
from . import contracts as c,store as s,scoring,integrate,finance


def retain(differences,pooled):
    return pooled<0 and sum(v<0 for v in differences)>=2


def ability_results(directory):
    base={str(y):exp.saved_predictions(c.SOURCE,s.trial(str(y),'ABCD')) for y in (2023,2024,2025)}
    labels={};old={}
    for year in (2023,2024,2025):
        labels.update(exp.winners(s.frame(date(year,1,1),date(year,9,30))))
        old[str(year)]=exp.saved_predictions(c.SOURCE,s.trial(str(year),'control'))
    pooled=scoring.annual_pool(base);result={'Stage2_ABCD':scoring.measure(pooled,labels),
        'old_feature_control':scoring.measure(scoring.annual_pool(old),labels),
        'Stage2_vs_old_features':scoring.compare(pooled,scoring.annual_pool(old),labels),'candidates':{}}
    variants={}
    candidates=['expanded_temperature']
    if s.completed(directory,'E')['summary'].get('new_LightGBM_fits')==4:candidates.append('ABCD_E')
    for name in candidates:
        parts={str(y):s.predictions(directory/('reuse' if name=='expanded_temperature' else 'E_trials'),
            f'{y}_temperature.predictions.local.json' if name=='expanded_temperature' else f'{y}.predictions.local.json') for y in (2023,2024,2025)}
        p=scoring.annual_pool(parts);annual=[]
        for year in (2023,2024,2025):
            a=scoring.flatten(parts[str(year)]);b=scoring.flatten(base[str(year)])
            annual.append({'year':year,'metrics':scoring.measure(a,labels),'versus_Stage2_ABCD':scoring.compare(a,b,labels)})
        comp=scoring.compare(p,pooled,labels);delta=comp['race_log_loss']['difference_new_minus_baseline']
        result['candidates'][name]={'pooled':scoring.measure(p,labels),'versus_Stage2_ABCD':comp,
            'versus_old_features':scoring.compare(p,scoring.annual_pool(old),labels),'annual':annual,
            'retain_development_candidate':retain([x['versus_Stage2_ABCD']['race_log_loss']['difference_new_minus_baseline'] for x in annual],delta)}
        variants[name]=p
    eligible=[name for name,r in result['candidates'].items() if r['retain_development_candidate']]
    result['selected']=min(eligible,key=lambda n:(result['candidates'][n]['pooled']['race_log_loss'],n)) if eligible else 'Stage2_ABCD'
    return result,variants


def market_results(directory):
    all_years={str(y):integrate.saved(directory,y) for y in (2023,2024,2025)}
    names=set.intersection(*(set(v) for v in all_years.values()));values={};labels={}
    for year in (2023,2024,2025):labels.update(exp.winners(integrate.dataset(directory,date(year,1,1),date(year,9,30))))
    for name in sorted(names):values[name]=scoring.annual_pool({year:[parts[name]] for year,parts in all_years.items()})
    metrics={name:scoring.measure(p,labels) for name,p in values.items()}
    controls=[n for n in ('raw_market','market_history','lambda_market_history','M','MH') if n in metrics]
    smallest=min(metrics[n]['race_log_loss'] for n in controls)
    control=next(n for n in controls if abs(metrics[n]['race_log_loss']-smallest)<=1e-12)
    comparisons={name:scoring.compare(p,values[control],labels) for name,p in values.items()}
    annual=[]
    for year,v in all_years.items():
        annual.append({'year':int(year),'metrics':{name:scoring.measure(p,labels) for name,p in v.items()},
            'versus_frozen_market_control':{name:scoring.compare(p,v[control],labels) for name,p in v.items()},
            'MA_vs_M':scoring.compare(v['MA'],v['M'],labels) if 'MA' in v and 'M' in v else None,
            'MHA_vs_MH':scoring.compare(v['MHA'],v['MH'],labels) if 'MHA' in v and 'MH' in v else None})
    eligibility={}
    for name in ('MA','MHA','arithmetic','lambda_market_ABCD_history'):
        if name not in values:continue
        yearly=[r['versus_frozen_market_control'][name]['race_log_loss']['difference_new_minus_baseline'] for r in annual]
        eligibility[name]={'annual_differences':yearly,'retain_development_candidate':retain(yearly,comparisons[name]['race_log_loss']['difference_new_minus_baseline'])}
    good=[name for name,r in eligibility.items() if r['retain_development_candidate']]
    selected=min(good,key=lambda n:(metrics[n]['race_log_loss'],n)) if good else None
    result={'same_population_races':len(values[control]),'metrics':metrics,'frozen_market_control':control,
        'versus_frozen_market_control':comparisons,'annual':annual,'AI_candidate_eligibility':eligibility,
        'selected_AI_candidate':selected,
        'saved_market_history_versus_raw':scoring.compare(values['market_history'],values['raw_market'],labels),
        'market_history_interpretation':'joint_market_calibration_and_odds_movement_not_movement_alone',
        'MA_versus_M':scoring.compare(values['MA'],values['M'],labels) if 'MA' in values and 'M' in values else None,
        'MHA_versus_MH':scoring.compare(values['MHA'],values['MH'],labels) if 'MHA' in values and 'MH' in values else None,
        'lambda_AI_versus_lambda_control':scoring.compare(values['lambda_market_ABCD_history'],values['lambda_market_history'],labels)}
    return result,values,labels


def future_plan(ability,market):
    return {'version':c.VERSION,'status':'locked_not_executed','evaluation':['2026-01-01','2026-09-30'],
        'training_and_evaluation_for_2026_executed':False,'old_final_entry_enabled':False,
        'major_comparisons':[
            {'family':'ability','candidate':ability['selected'],'control':'old_feature_control' if ability['selected']=='Stage2_ABCD' else 'Stage2_ABCD'},
            {'family':'market','candidate':market['selected_AI_candidate'] or market['frozen_market_control'],
             'control':market['frozen_market_control'] if market['selected_AI_candidate'] else 'raw_market'}],
        'conditions':'same_origin_training_validation_calibration_and_race_runner_population;frozen_recipes_only',
        'annual_recipe':'train_prior_Aug31;validate_Sep08_Oct31;calibrate_Nov08_Dec24;evaluate_next_JanSep',
        'primary_metric':'race_winner_log_loss','difference_sign':'new_minus_control','primary_comparisons':2,
        'confidence_each':.975,'correction':'Bonferroni','resample':'race_week','repetitions':5000,'seed':c.SEED,
        'formal_success':'CI_upper_strictly_below_zero','no_reselection_on_2026':True,
        'activation':'separate_explicit_future_instruction_required'}


def model_dependencies(directory):
    references={}
    for path in sorted((c.SOURCE/'models').glob('*/model.json')):
        # All Stage 2 model dependencies are frozen without copying the artifacts.
        references[str(path)]=safety.digest(path,root=oldc.ROOT)
    for path in sorted((directory/'models').glob('*/model.json')):references[str(path)]=s.digest(path)
    for path in sorted((directory/'integration').glob('*.json')):
        if path.stem.endswith(('results','local','complete')):continue
        references[str(path)]=s.digest(path)
    return references


def run(directory):
    done=s.completed(directory,'freeze')
    if done:
        evidence(directory)
        return done['summary']
    for unit in ('reuse','oof','integrate','E'):
        if not s.completed(directory,unit):raise safety.MLFailure('artifact_invalid')
    ability,ability_predictions=ability_results(directory);market,variants,labels=market_results(directory)
    if (directory/'selection_before_financial.json').exists():
        selection=s.read(directory,'selection_before_financial.json')
        if selection['ability']!=ability['selected'] or selection['market']!=market['frozen_market_control'] or selection['AI']!=market['selected_AI_candidate']:
            raise safety.MLFailure('artifact_invalid')
    else:s.write(directory,'selection_before_financial.json',{'ability':ability['selected'],'market':market['frozen_market_control'],
        'AI':market['selected_AI_candidate'],'selection_metric_only':'annual_race_LL_and_year_directions','financial_results_not_computed_yet':True})
    # Add ability experiments on exactly the same quote-eligible race+horse sets for paper diagnostics.
    common={p.race for p in variants['raw_market']}
    for name,p in ability_predictions.items():variants[name]=[x for x in p if x.race in common]
    for p in variants.values():scoring.align(p,variants['raw_market'])
    packets={}
    for year in (2023,2024,2025):packets.update(market_data.load(c.SOURCE,date(year,1,1),date(year,9,30)))
    folder=directory/'paper';folder.mkdir(exist_ok=True);paper={};files=[]
    additional=set(['Stage2_ABCD',market['frozen_market_control']]+([market['selected_AI_candidate']] if market['selected_AI_candidate'] else []))
    for name,p in variants.items():
        for mode,minimum in [('primary',None)]+([('minimum_p05',.05)] if name in additional else []):
            unit=name+'_'+mode;done=s.completed(folder,unit)
            if done:
                value=s.read(folder,unit+'.json')
            else:
                value,details=finance.diagnostic(p,packets,labels,minimum=minimum)
                s.write(folder,unit+'.json',dict(s.provenance(directory,unit,'annual_2023_2025'),**value))
                s.write(folder,unit+'.ledger.local.json',details)
                s.complete(folder,unit,[folder/(unit+'.json'),folder/(unit+'.ledger.local.json')],{'name':name,'mode':mode})
            paper[unit]=value;files.append(folder/(unit+'.complete.json'))
    preservation=s.verify_old(directory)
    reuse=s.completed(directory,'reuse')['summary'];quality=s.read(directory,'E_data_quality.json') if (directory/'E_data_quality.json').exists() else s.read(directory,'E_skipped.json')
    annual_details=[s.read(directory/'integration',f'{y}_results.json') for y in (2023,2024,2025)]
    ledger={'version':c.VERSION,'run_id':directory.name,'new_LightGBM_fits':len(list((directory/'fit_units').glob('*.started.json'))),
        'new_LightGBM_fits_max':7,'small_optimizations':len(list((directory/'optimizations').glob('*.started.json'))),'small_optimizations_max':24,
        'lambda_combinations':40,'temperature_candidates':list(c.TEMPERATURE),'blend_weights':list(c.WEIGHTS),
        'no_adaptive_followup_search':True,'reuse':reuse,'integration_years':annual_details,
        'OOF':s.read(directory,'oof_origins.json'),'E':s.completed(directory,'E')['summary']}
    if ledger['new_LightGBM_fits']>7 or ledger['small_optimizations']>24:raise safety.MLFailure('input_limit')
    frozen={'version':c.VERSION,'run_id':directory.name,'created_at_jst':datetime.now(core.JST).isoformat(),
        'ability_candidate':ability['selected'],'AI_integration_candidate':market['selected_AI_candidate'],
        'market_information_control':market['frozen_market_control'],'selection_is_development_only':True,
        'source_Stage2_freeze_sha256':safety.digest(c.SOURCE/'candidate_freeze.json',root=oldc.ROOT),
        'source_feature_version':oldc.FORMULA_VERSION,'integration_version':c.VERSION,'E_version':c.E_VERSION,
        'plan_sha256':s.digest(directory/'plan.json'),'base_models_and_integration_manifests':model_dependencies(directory),
        'ability_results':ability,'market_results':market,'paper_policy':s.read(directory,'plan.json')['paper'],
        'do_not_use_financial_diagnostics_for_selection':True,'sealed_2026_read_train_evaluate':False,'old_artifacts':preservation,
        'no_automatic_production_or_shadow_task_adoption':True}
    report={'run_id':directory.name,'ability':ability,'market':market,'paper':paper,'reuse':reuse,'data_quality':quality,
        'annual_details':annual_details,'preservation':preservation,'sealed_2026_evaluated':False,
        'main_2025_kept_separate':'reuse/main_*.json and E_trials/main.json;not_in_annual_pool',
        'limitations':['all_results_development_selection_not_unused_test','publication_time_not_PC_observation',
            'schedule_and_revision_original_asof_unverified','paper_100_yen_not_real_execution'],
        'stopped_after_fixed_budget':True}
    s.write(directory,'experiment_ledger.json',ledger);s.write(directory,'development_report.json',report)
    s.write(directory,'candidate_freeze.json',frozen);s.write(directory,'final_evaluation_plan.json',future_plan(ability,market))
    path=directory/'development_report.md'
    with path.open('x',encoding='utf-8') as stream:stream.write(markdown(report))
    s.write(directory,'old_artifacts_verified.json',preservation)
    files.extend(directory/name for name in ('experiment_ledger.json','development_report.json','candidate_freeze.json','final_evaluation_plan.json','development_report.md','old_artifacts_verified.json','selection_before_financial.json'))
    summary={'ability_candidate':ability['selected'],'AI_integration_candidate':market['selected_AI_candidate'],
        'market_control':market['frozen_market_control'],'new_LightGBM_fits':ledger['new_LightGBM_fits'],'optimizations':ledger['small_optimizations'],
        'old_artifacts':preservation,'sealed_2026_evaluated':False,'report':str(path)}
    s.complete(directory,'freeze',files,summary)
    evidence(directory)
    return summary


def evidence(directory):
    """Full finite-trial inventory and verification sidecar; never rewrites frozen results."""
    done=s.completed(directory,'evidence')
    if done:return done['summary']
    r=s.read(directory,'development_report.json');inventory={}
    inventory['all_24_optimizations']=[s.read(p.parent,p.name) for p in sorted((directory/'optimizations').glob('*.json')) if not p.name.endswith('.started.json')]
    inventory['all_reuse_selections']=[s.read(p.parent,p.name) for p in sorted((directory/'reuse').glob('*.json')) if not p.name.endswith(('.local.json','.complete.json'))]
    inventory['E_trials']=[s.read(p.parent,p.name) for p in sorted((directory/'E_trials').glob('*.json')) if not p.name.endswith(('.local.json','.complete.json'))]
    inventory['fit_units']=[s.completed(directory/'fit_units',p.name[:-len('.complete.json')]) for p in sorted((directory/'fit_units').glob('*.complete.json'))]
    inventory['skipped_or_not_promoted']={
        'AI_integration':'no_candidate_beat_frozen_market_control_in_two_years_and_pooled_LL',
        'expanded_temperature':'annual_calibration_selected_original_.75_in_all_three_years',
        'arithmetic':'calibration_selected_w_zero_each_year',
        'E_into_OOF':'excluded_by_plan','temperature_plus_E':'excluded_by_plan',
        'geometric_grid':'not_added_MA_already_contains_the_constrained_form',
        'extra_lambda_temperature_or_features':'not_added_fixed_budget_exhausted','final_2026':'sealed_no_execution_or_training'}
    def interpretation(ci):
        return 'uncertain_interval_includes_zero' if ci['ci_low']<=0<=ci['ci_high'] else 'mean_improvement_supported_in_development' if ci['ci_high']<0 else 'mean_worsening_supported_in_development'
    judgements={name:interpretation(value['versus_Stage2_ABCD']['race_log_loss']) for name,value in r['ability']['candidates'].items()}
    judgements.update(MA_vs_M=interpretation(r['market']['MA_versus_M']['race_log_loss']),
        MHA_vs_MH=interpretation(r['market']['MHA_versus_MH']['race_log_loss']),
        market_history_vs_raw=interpretation(r['market']['saved_market_history_versus_raw']['race_log_loss']),
        ability_candidate_is_provisional=True,formal_unused_test_certification=False)
    # The overlapping main-2025 experiment stays separate from all annual pools.
    base=exp.saved_predictions(c.SOURCE,s.trial('main','ABCD'));labels={}
    for start,end in s.trial('main','ABCD')['split']['evaluation']:
        labels.update(exp.winners(s.frame(date.fromisoformat(start),date.fromisoformat(end))))
    main={}
    for name,folder,file in [('expanded_temperature','reuse','main_temperature.predictions.local.json'),('ABCD_E','E_trials','main.predictions.local.json')]:
        if not (directory/folder/file).exists():continue
        p=scoring.flatten(s.predictions(directory/folder,file));ref=scoring.flatten(base)
        main[name]={'metrics':scoring.measure(p,labels),'versus_Stage2_ABCD':scoring.compare(p,ref,labels),'not_added_to_annual_pool':True}
    project=Path(__file__).resolve().parents[3]
    paths=[*Path(__file__).parent.glob('*.py'),project/'scripts/stage3_local.py',project/'scripts/stage3_local.ps1',
           project/'tests/test_stage3.py',project/'docs/stage3_development.md']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    testlog=project/'reports/stage3_tests.txt';raw=testlog.read_bytes()
    testtext=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
    match=re.search(r'Ran (\d+) tests',testtext)
    tests={'total':int(match.group(1)) if match else None,'all_passed':testtext.strip().endswith('OK'),
        'log_sha256':hashlib.sha256(raw).hexdigest(),'DB_and_network_blocked':True}
    if not tests['all_passed']:raise safety.MLFailure('artifact_invalid')
    s.write(directory,'experiment_inventory.json',inventory)
    s.write(directory,'main_2025_separate_summary.json',main)
    verification={'run_id':directory.name,'tests':tests,'uncertainty_judgements':judgements,
        'source_code_sha256':hashes,'old_artifacts':s.verify_old(directory),'sealed_2026_read_train_evaluate':False,
        'new_LightGBM_fits':len(inventory['fit_units']),'optimizations':len(inventory['all_24_optimizations']),
        'main_2025_never_added_to_annual_pool':True,'frozen_candidate_sha256':s.digest(directory/'candidate_freeze.json')}
    s.write(directory,'completion_verification.json',verification)
    files=[directory/name for name in ('experiment_inventory.json','main_2025_separate_summary.json','completion_verification.json')]
    s.complete(directory,'evidence',files,{'tests':tests,'old_artifacts':verification['old_artifacts'],'uncertainty':judgements})
    return verification


def markdown(r):
    num=oldreport.number;table=oldreport.table
    out=['# Stage 3：開発評価と候補凍結\n\n','2023〜2025年の各1〜9月をプールした開発評価です。主2025年3〜11月の探索は別記録で、重複加算していません。'
        '主指標はレース単位勝馬Log Loss、差は新−対照です。2026年の最終評価とそのための再学習は行っていません。\n\n',
        '## 能力モデル（市場欠損で除外しない基本母集団）\n\n']
    a=r['ability'];rows=[]
    rows.append(['Stage2_ABCD',a['Stage2_ABCD']['races'],num(a['Stage2_ABCD']['race_log_loss']),'—'])
    for name,v in a['candidates'].items():
        ci=v['versus_Stage2_ABCD']['race_log_loss'];rows.append([name,v['pooled']['races'],num(v['pooled']['race_log_loss']),f"{num(ci['difference_new_minus_baseline'])} [{num(ci['ci_low'])}, {num(ci['ci_high'])}]"])
    out.append(table(['方式','レース','レースLL','対ABCD差 [開催週95%CI]'],rows))
    out.append(f"保持候補：`{a['selected']}`。温度改善とE群は別実験で、統合層の基底ABCDには反映していません。\n\n")
    out.append('## 市場共通母集団での比較\n\n');m=r['market'];rows=[]
    for name,v in m['metrics'].items():
        ci=m['versus_frozen_market_control'][name]['race_log_loss']
        rows.append([name,v['races'],num(v['race_log_loss']),num(v['log_loss']),num(v['brier_score']),num(v['top1_accuracy_percent'],2),f"{num(ci['difference_new_minus_baseline'])} [{num(ci['ci_low'])}, {num(ci['ci_high'])}]"])
    out.append(table(['方式','レース','レースLL','馬LL','Brier','1位的中率%','市場対照との差 [週95%CI]'],rows))
    out.append(f"凍結した市場情報だけの対照：`{m['frozen_market_control']}`。AI統合候補：`{m['selected_AI_candidate'] or 'なし（市場対照を保持）'}`。"
        '候補は年次プール差が負かつ3年中2年以上で改善方向という条件で選び、仮想収支を選択に使っていません。\n\n')
    ci=m['saved_market_history_versus_raw']['race_log_loss']
    out.append(f"既存market_history対無補正市場：差{num(ci['difference_new_minus_baseline'])}、95%CI [{num(ci['ci_low'])}, {num(ci['ci_high'])}]。"
        '市場較正とオッズ変動を合わせた効果であり、変動単独の効果とは解釈しません。\n\n')
    out.append('## 年別の性能と区間\n\n');rows=[]
    for y in m['annual']:
        for name in ('raw_market',m['frozen_market_control'],'MA','MHA','arithmetic'):
            if name not in y['metrics']:continue
            v=y['metrics'][name];ci=y['versus_frozen_market_control'][name]['race_log_loss']
            rows.append([y['year'],name,v['races'],num(v['race_log_loss']),f"{num(ci['difference_new_minus_baseline'])} [{num(ci['ci_low'])}, {num(ci['ci_high'])}]"])
    out.append(table(['年','方式','レース','レースLL','市場対照との差 [週95%CI]'],rows))
    out.append('## 時系列OOF・選択・被覆率\n\n')
    out.append('OOFは2020〜2022年に各1回だけ学習し、2023〜2025年はStage 2年次モデルを再利用。前年8月末まで学習、9月8日〜10月末で早期終了、11月8日〜12月24日で較正し、翌年を推論しました。'
        '全行の生成元モデルと履歴時点をローカルOOF CSVへ保存しています。統合層は2020年以降・前年8月末までのOOFだけで学習し、前年9〜10月で正則化を選択しています。\n\n')
    out.append(table(['年','市場共通レース','基本レース','被覆率%','T−30完備','履歴欠損のまま保持'],[
        [v['fold_id'],v['primary_races'],v['full_ability_races'],num(v['coverage_percent'],2),v['complete_history_races'],v['early_missing_retained_races']] for v in r['annual_details']]))
    rows=[]
    for year in r['annual_details']:
        for sel in year['selections']:rows.append([year['fold_id'],sel['config'],sel['strength'],', '.join(num(x) for x in sel['coefficients'])])
    out.append(table(['年','統合構成','正則化','係数（terms順）'],rows))
    out.append('損失は平均レースLL＋強度/2×市場基準係数からの二乗距離。共通切片・補助入力はありません。h欠損は0ですが、欠損状態と実発表時刻は別メタデータに保持。幾何混合はMAでa=1−w、b=wと制約した特殊形です。\n\n')
    out.append('## 確率帯別校正\n\n')
    for name in dict.fromkeys(['raw_market',m['frozen_market_control'],m['selected_AI_candidate'] or 'Stage2_ABCD']):
        v=m['metrics'][name];out.append(f'### {name}\n\n')
        out.append(table(['帯','頭数','平均確率','実勝率','予測/実測'],[[x['probability_band'],x['count'],num(x['mean_probability']),num(x['observed_rate']),num(x['predicted_to_observed_ratio'])] for x in v['calibration']]))
    out.append('## 仮想収益（選択に不使用）\n\n');rows=[]
    for name,p in r['paper'].items():
        bet=p['ai'];ci=p['weekly_uncertainty'];risk=p['risk']
        rows.append([name,bet['bet_count'],bet['investment_yen'],bet['net_profit_yen'],num(bet['return_rate_percent'],2),bet['unsettled_count'],risk['max_drawdown_yen'],
            str(ci['ROI_percent_ci']),num(ci['zero_purchase_fraction'],3)])
    out.append(table(['方式','購入','投資円','純利益円','回収率%','未精算','最大DD円','ROI週95%CI','購入0再標本割合'],rows))
    out.append('購入0件のROIは未算出。未精算を0円払戻に置換していません。100件未満の購入には記述的な少数購入注意を付けています。DDは元時系列の診断で、bootstrapしていません。'
        '同時点1番人気、月次純利益、最大払戻依存、最大払戻を除いた損益、利益の週区間はJSONへ保存。追加勝率5%条件はEV条件との積集合を先に作り、その中から最大EVを選びます。\n\n')
    out.append('## データ品質・保全・限界\n\n')
    out.append('goingとprevious_time_secondsはStage 2比較互換性のため意図的に無効化され、原列も収集対象外でした。DB元値の欠損や形式不正が原因と断定しません。'
        'E群は公式SEタイム差の符号付き0.1秒、符号付きゼロを保持。0000/9999・不正・異常時は欠損とし、当該結果は当該特徴量へ入れません。'
        '列・単位・形式件数・結合一致はE_data_quality.json、全欠損率はEモデルmanifestへ記録しています。\n\n')
    out.append(f"旧成果物{r['preservation']['checked_files']}ファイルを開始前後でハッシュ照合し、変更0件。予測再現差{r['reuse']['prediction_reproduction_max_difference']}。\n\n")
    out.append('すべて開発中の選択後の結果で、未使用テストによる証明ではありません。公表時刻による事後評価であり、PC実取得時刻、発走情報の当時版、通信遅延、実購入・約定は未検証です。'
        '指定の探索上限で終了し、2026年の最終評価入口、既存シャドータスクは有効化・変更していません。'
        '凍結した候補と依存モデルはcandidate_freeze.json、将来の主要2比較・各97.5%区間の計画はfinal_evaluation_plan.jsonに保存しています。\n')
    return ''.join(out)
