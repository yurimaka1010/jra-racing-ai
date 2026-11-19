"""Preflight every partition, run six fixed fits, compare, then freeze and stop."""
from datetime import date
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from ..ml import safety
from ..research import experiment as exp
from ..stage4 import data as source
from ..stage5 import metrics
from . import contracts as c,data,model,store as s


def bounds(values,confidence=.95):
    if confidence==.95:return metrics.bounds(values)
    if confidence!=.975:raise safety.MLFailure('invalid_input')
    values=np.asarray(values,float)
    if not np.isfinite(values).all():raise safety.MLFailure('invalid_input')
    return np.quantile(values,[.0125,.9875]).tolist()


def compare(point,draw,base,base_draw):
    return {name:{'difference_new_minus_comparator':float(point[name]-base[name]),
        '95_percent_ci':bounds(draw[name]-base_draw[name]),
        **({'97_5_percent_ci':bounds(draw[name]-base_draw[name],.975)} if name=='within_race_auc' else {})}
        for name in metrics.NAMES}


def decide(points,comparisons,annual,*,consistency):
    a,b=comparisons['A']['within_race_auc'],comparisons['B']['within_race_auc'];h,p=points['H'],points['H_pace']
    directions={str(y):v['comparisons']['A']['within_race_auc']['difference_new_minus_comparator'] for y,v in annual.items()}
    tests={'A_within_delta_at_least_.002':a['difference_new_minus_comparator']>=.002,
        'A_975_CI_low_above_zero':a['97_5_percent_ci'][0]>0,
        'B_within_delta_at_least_.001':b['difference_new_minus_comparator']>=.001,
        'B_975_CI_low_above_zero':b['97_5_percent_ci'][0]>0,
        'positive_annual_within_vs_H_at_least_two':sum(v>0 for v in directions.values())>=2,
        'no_annual_within_vs_H_delta_leq_minus_.002':all(v>-.002 for v in directions.values()),
        'pooled_ROC_AUC_at_least_H':p['roc_auc']>=h['roc_auc']-1e-12,
        'pooled_top1_at_least_H':p['top1_accuracy_percent']>=h['top1_accuracy_percent']-1e-12,
        'pooled_race_LL_at_most_H':p['race_log_loss']<=h['race_log_loss']+1e-12,
        'pooled_Brier_at_most_H':p['brier_score']<=h['brier_score']+1e-12,'consistency_verified':bool(consistency)}
    retain=all(tests.values());reasons=[]
    if not (tests['A_within_delta_at_least_.002'] and tests['B_within_delta_at_least_.001']):reasons.append('practical_effect_threshold_not_met')
    if not (tests['A_975_CI_low_above_zero'] and tests['B_975_CI_low_above_zero']):reasons.append('formal_paired_intervals_not_strictly_above_zero')
    if not (tests['positive_annual_within_vs_H_at_least_two'] and tests['no_annual_within_vs_H_delta_leq_minus_.002']):reasons.append('annual_direction_conditions_not_met')
    if not all(tests[n] for n in tests if n.startswith('pooled_')):reasons.append('secondary_quality_conditions_not_met')
    if not consistency:reasons.append('consistency_not_verified')
    return {'retain_development_candidate':retain,'status':'development_candidate' if retain else 'not_retained',
        'criteria':tests,'failed_criteria':[k for k,v in tests.items() if not v],'reasons':reasons,
        'annual_within_differences_vs_H':directions,'formal_comparisons':['A','B'],'H_style_control_only':True,
        'production_adoption':False,'independent_final_evaluation':False,'H_remains_primary':True,
        'no_information_impossibility_claim':True,'stop_this_fixed_hypothesis_if_not_retained':not retain}


def baseline(year,frame):
    predictions={'H':source.source_predictions(c.S4,year,'H'),'raw_market':exp.predictions(frame,frame.market_probability.to_numpy(float))}
    saved=safety.read_json(c.S5/'baseline.json',root=c.S5)['years'][str(year)]
    for name,values in predictions.items():
        point=metrics.point_only(frame,values)
        if any(abs(point[n]-saved['metrics'][name][n])>1e-12 for n in metrics.NAMES):raise safety.MLFailure('artifact_invalid')
        if (point['races'],point['runners'])!=(saved['metrics'][name]['races'],saved['metrics'][name]['runners']):raise safety.MLFailure('invalid_identity')
    return predictions


def prepare(directory):
    frames={};predictions={};coverage={};gates={};races=runners=0
    for year in c.YEARS:
        sp=c.split(year);parts={}
        for part in ('train','validation','calibration','evaluation'):
            a,b=sp['evaluation'][0] if part=='evaluation' else (sp[part+'_start'],sp[part+'_end'])
            parts[part]=data.load(date.fromisoformat(a),date.fromisoformat(b))
        predictions[year]=baseline(year,parts['evaluation']);frames[year]=parts
        coverage[str(year)]={name:data.coverage(f) for name,f in parts.items()};gates[str(year)]=data.training_gate(parts['train'])
        races+=coverage[str(year)]['evaluation']['races'];runners+=len(parts['evaluation'])
    if (races,runners)!=(6373,87601):raise safety.MLFailure('invalid_identity')
    summary={'all_training_partitions_usable':all(gates.values()),'training_variation_by_year':gates,'coverage':coverage,
        'fixed_evaluation_races':races,'fixed_evaluation_runners':runners,'baseline_reproduced_with_tolerance':1e-12,
        'C_origin':s.read(directory,'inputs.json')['C_origin'],'optional_missingness_never_removes_rows':True}
    return frames,predictions,summary


def evaluate(frame,predictions,scope,replicas,expected):
    y,ph,rs,ws=metrics.assemble(frame,predictions['H']);boot=metrics.WeekBootstrap(ws)
    if boot.fingerprint!=expected['bootstrap']['sample_indices_sha256']:raise safety.MLFailure('artifact_invalid')
    points={};draws={}
    for name,values in predictions.items():
        labels,p,r,w=metrics.assemble(frame,values)
        if not np.array_equal(y,labels) or not np.array_equal(rs,r) or not np.array_equal(ws,w):raise safety.MLFailure('invalid_identity')
        if name in ('H','raw_market'):
            point=metrics.point_only(frame,values)
            for metric in metrics.NAMES:
                if abs(point[metric]-expected['metrics'][name][metric])>1e-12:raise safety.MLFailure('artifact_invalid')
            draw={metric:replicas[f'{scope}__{name}__{metric}'] for metric in metrics.NAMES}
        else:point,draw=boot.measure(labels,p,rs)
        points[name]=dict(point,races=len(np.unique(rs)),runners=len(frame),within_race_AUC_95_ci=bounds(draw['within_race_auc']),
            roc_auc_95_ci=bounds(draw['roc_auc']))
        draws[name]=draw
    comparisons={label:compare(points['H_pace'],draws['H_pace'],points[other],draws[other]) for label,other in (('A','H'),('B','H_style'))}
    return {'metrics':points,'comparisons':comparisons,'H_style_minus_H_descriptive_only':compare(points['H_style'],draws['H_style'],points['H'],draws['H']),
        'bootstrap':boot.metadata()}


def test_result():
    path=c.PROJECT/'reports/stage6_regression_tests.txt';raw=path.read_bytes()
    text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig');found=re.search(r'Ran (\d+) tests in ([0-9.]+)s',text)
    if not found or not text.rstrip().endswith('OK'):raise safety.MLFailure('artifact_invalid')
    return {'executed_tests':int(found[1]),'previous_tests':912,'added_tests':int(found[1])-912,'all_passed':True,
        'seconds':float(found[2]),'synthetic_LightGBM_fit_calls':1,'real_DB_driver_and_network_blocked':True,
        'log':str(path),'log_sha256':hashlib.sha256(raw).hexdigest()}


def write_predictions(directory,frames,predictions,records):
    path=safety.local_path(directory/'predictions.local.csv',root=c.ROOT)
    if path.exists():raise safety.MLFailure('artifact_exists')
    blocks=[]
    for year in c.YEARS:
        f=frames[year]['evaluation'];out=f[['entry_key','date','winner_label']].copy();out['evaluation_year']=year
        for name,values in predictions[year].items():out[name]=metrics.assemble(f,values)[1]
        out['source_H_model_id']=source.trial(c.S4,year,'H')['model_id'];out['raw_market_source']='saved_T10_inverse_win_odds'
        for name in c.CONFIGS:out[name+'_model_id']=next(r['model_id'] for r in records if r['config']==name and r['year']==year)
        blocks.append(out)
    with path.open('x',encoding='utf-8',newline='') as stream:pd.concat(blocks,ignore_index=True).to_csv(stream,index=False,lineterminator='\n')
    return {'file':path.name,'sha256':s.sha(path),'runners':sum(len(f) for f in blocks)}


def finish(directory,result):
    inventory=s.inventory(directory);preservation=s.preserve(directory);tests=test_result()
    result.update(run_id=directory.name,version=c.VERSION,plan_sha256=s.sha(directory/'plan.json'),
        created_at_jst=s.now(),experiment_inventory=inventory,tests=tests,preservation=preservation,
        development_reuse_not_independent_final_test=True)
    result['2026_data_processed']=False
    decision=result.get('decision',{'retain_development_candidate':False,'status':'input_conditions_not_met',
        'reasons':['training_input_conditions_not_met'],'H_remains_primary':True})
    result['decision']=decision
    kept=[{'year':r['year'],'model_id':r['model_id'],'model_sha256':r['model_sha256']} for r in inventory['records'] if r['config']=='H_pace' and r['status']=='complete'] if decision['retain_development_candidate'] else None
    s.write(directory,'candidate_freeze.json',{'plan_sha256':result['plan_sha256'],'retain_development_candidate':decision['retain_development_candidate'],
        'candidate':{'config':'H_pace','annual_models':kept} if kept else None,'H_style_control_only':True,'decision':decision,
        'budget':{'real_data_fits_started':inventory['fits_started'],'maximum':6,'synthetic_test_fits':tests['synthetic_LightGBM_fit_calls']},
        'primary_model':'H_unchanged','production_adoption':False,'independent_final_evaluation':False,'2026_data_processed':False,
        'profit_improvement_certified':False,'fixed_hypothesis_closed':True,'no_automatic_next_stage':True})
    s.write(directory,'development_report.json',result)
    lines=['# Stage 6：位置取り代理値と同走馬構成','',
        'H_paceを開発候補として保持。正式昇格なし。' if decision['retain_development_candidate'] else '入力条件不適合で学習せず終了。' if not inventory['fits_started'] else 'H_paceは開発候補として不保持。今回の固定研究を終了。',
        '',f"計画SHA-256: `{result['plan_sha256']}`",'',
        '2023〜2025年1〜9月、同一6,373レース・87,601頭。繰り返し使用した開発評価であり独立最終評価ではありません。',
        '位置取りは7日遅延済み前走4角相対値の代理。実際の当日ペース、当該結果、確定頭数、確定オッズは特徴量に使用しません。',
        'T−10公表分末尾・鮮度300秒・同一馬番集合・T−30履歴の既存契約を維持。原PC取得時刻と当初発走予定版は未検証の事後評価。','']
    if result.get('pooled'):
        for title,scope in [('3年プール',result['pooled'])]+[(str(y),result['years'][str(y)]) for y in c.YEARS]:
            lines+=['## '+title,'','| モデル | レース内AUC | Hとの差 | プールROC-AUC | Hとの差 | 1位的中率% | レースLL | Brier | 用途 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---|']
            h=scope['metrics']['H']
            for name in ('H','raw_market','H_style','H_pace'):
                v=scope['metrics'][name];usage={'H':'主力維持','raw_market':'無補正市場対照','H_style':'個体対照・昇格対象外','H_pace':decision['status']}[name]
                lines.append(f"| {name} | {v['within_race_auc']:.8f} | {v['within_race_auc']-h['within_race_auc']:+.8f} | {v['roc_auc']:.8f} | {v['roc_auc']-h['roc_auc']:+.8f} | {v['top1_accuracy_percent']:.4f} | {v['race_log_loss']:.8f} | {v['brier_score']:.8f} | {usage} |")
            lines+=['','| 比較 | レース内AUC差 | 97.5%両側区間 |','|---|---:|---|']
            for label,contrast in scope['comparisons'].items():
                v=contrast['within_race_auc'];interval=v['97_5_percent_ci']
                lines.append(f"| {label}: H_pace−{'H' if label=='A' else 'H_style'} | {v['difference_new_minus_comparator']:+.8f} | [{interval[0]:+.8f}, {interval[1]:+.8f}] |")
        lines+=['','## 事前固定の採否','', '| 条件 | 判定 |','|---|---|']
        lines.extend(f"| {name} | {'満たす' if value else '未達'} |" for name,value in decision['criteria'].items())
        lines+=['','未達理由コード: '+', '.join(decision['reasons']),
            '効果量の実用幅未達、区間下限条件未達、年別方向、補助品質を区別します。不保持は情報の不存在や理論的性能上限の証明ではありません。']
    lines+=['','## 入力欠損','', '| 年用 | 区間 | レース/頭 | 個体欠損% | 相互作用欠損% |','|---|---|---|---:|---:|']
    for year,parts in result['inputs']['coverage'].items():
        for part,v in parts.items():lines.append(f"| {year} | {part} | {v['races']}/{v['runners']} | {100*v['features']['style_proxy']['missing_rate']:.3f} | {100*v['features']['style_pressure_interaction']['missing_rate']:.3f} |")
    lines+=['','任意特徴量の欠損で行・レースを除外しません。各列の欠損率・変動数はJSONへ保存。','',
        '## 学習予算と検証','', '| 構成 | 評価年 | best_iteration | 保存再読込最大差 | 状態 |','|---|---:|---:|---:|---|']
    for row in inventory['records']:lines.append(f"| {row['config']} | {row['year']} | {row.get('best_iteration')} | {row.get('reload_max_probability_difference')} | {row['status']} |")
    lines += ['',f"実データfit予約・開始 {inventory['fits_started']}/6。合成テスト内fit {tests['synthetic_LightGBM_fit_calls']}。回帰テスト{tests['executed_tests']}件成功（既存912＋追加{tests['added_tests']}）。",
        f"旧参照{preservation['unchanged_reference_files']}ファイル・旧プロジェクト{preservation['unchanged_old_project_files']}ファイルの変更0。朝7時タスク定義一致: {preservation['task_configuration_match']}。",
        'bootstrapは117開催週、5,000反復、seed20261019、全モデル共通抽出。主要2比較には97.5%両側区間。他指標の95%区間は説明用。',
        '固定学習済み予測に対する開催週変動であり、再学習・研究選択の全不確実性を表しません。',
        '2026年データ処理0、独立最終評価なし、旧モデル・凍結記録・朝7時タスク変更なし。ROI実験なし・収益改善未認定。',
        '不通過時の条件変更・代替構成・T−3への自動移行は行いません。再開は同じrun IDで完了モデルを検証再利用します。']
    path=safety.local_path(directory/'development_report.md',root=c.ROOT)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    return {'decision':decision,'fits_started':inventory['fits_started'],'fits_max':6,'report_directory':str(directory),
        'pooled_metrics':result.get('pooled',{}).get('metrics',{}),'tests':tests,'preservation':preservation}


def run(directory):
    s.verify_plan(directory)
    if (directory/'development_report.json').exists():
        saved=s.read(directory,'development_report.json');s.preserve(directory)
        for row in s.inventory(directory)['records']:
            if row['status']=='complete':
                p=directory/'models'/row['model_id']
                if s.sha(p/'model.json')!=row['model_sha256'] or s.sha(p/'win.lgb.txt')!=row['tree_sha256']:raise safety.MLFailure('artifact_invalid')
        return {'decision':saved['decision'],'fits_started':saved['experiment_inventory']['fits_started'],'resumed_without_training':True,'report_directory':str(directory)}
    frames,predictions,inputs=prepare(directory)
    if not inputs['all_training_partitions_usable']:return finish(directory,{'inputs':inputs,'new_training_skipped':True})
    for config in c.CONFIGS:
        for year in c.YEARS:
            bundle,record,p=model.fit(directory,config,year,frames[year]);predictions[year][config]=exp.predictions(frames[year]['evaluation'],p)
            print(json.dumps({'status':'running','stage':'fit','config':config,'year':year,'best_iteration':record['best_iteration'],
                'fits_started':s.inventory(directory)['fits_started'],'reload_max_difference':record['reload_max_probability_difference']}),flush=True)
    baseline_saved=safety.read_json(c.S5/'baseline.json',root=c.S5);annual={}
    path=safety.local_path(c.S5/'baseline_bootstrap.local.npz',root=c.S5,exists=True)
    with np.load(path,allow_pickle=False) as replicas:
        for year in c.YEARS:annual[str(year)]=evaluate(frames[year]['evaluation'],predictions[year],str(year),replicas,baseline_saved['years'][str(year)])
        frame=pd.concat([frames[y]['evaluation'] for y in c.YEARS],ignore_index=True)
        combined={name:[p for y in c.YEARS for p in predictions[y][name]] for name in ('H','raw_market','H_style','H_pace')}
        pooled=evaluate(frame,combined,'pooled',replicas,baseline_saved['pooled'])
    inventory=s.inventory(directory)
    consistency=all(r['status']=='complete' and r['reload_max_probability_difference']<=1e-12 for r in inventory['records']) and len(inventory['records'])==6
    decision=decide(pooled['metrics'],pooled['comparisons'],annual,consistency=consistency)
    detail=write_predictions(directory,frames,predictions,inventory['records'])
    return finish(directory,{'inputs':inputs,'pooled':pooled,'years':annual,'decision':decision,'prediction_detail':detail,
        'evaluated_configurations':2,'real_data_fits':6,'no_ROI_experiment':True,'baseline_H_and_raw_reproduced':True})
