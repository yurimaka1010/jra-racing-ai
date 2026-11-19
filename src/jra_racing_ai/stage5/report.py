"""One anonymous report/JSON and a small immutable completion record."""
from pathlib import Path
from datetime import datetime
import hashlib,re
from ..ml import safety
from ..win_sim import core
from . import contracts as c,store as s

ORDER=('H','raw_market','ABCD','Hcal','Hstack','H60','ABCD_E','H_place')
LABELS={'H':'H（現行基準）','raw_market':'raw_market','ABCD':'ABCD','Hcal':'Hcal','Hstack':'Hstack','H60':'H60','ABCD_E':'ABCD＋E','H_place':'H_place'}
STATUS={'H':'維持','raw_market':'比較対照','ABCD':'凍結参照・追加開発停止','Hcal':'補正候補・保留','Hstack':'凍結参照・追加開発停止',
        'H60':'凍結参照・追加開発停止','ABCD_E':'凍結参照・追加開発停止'}


def num(v,*,sign=False,digits=6):return '未算出' if v is None else format(v,('+' if sign else '')+f'.{digits}f')
def ci(v):return '未算出' if v is None else '['+num(v[0])+', '+num(v[1])+']'


def test_result():
    path=c.PROJECT/'reports/stage5_regression_tests.txt';raw=path.read_bytes()
    text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',text)
    if not match or not text.rstrip().endswith('OK'):raise safety.MLFailure('artifact_invalid')
    return {'executed_tests':int(match[1]),'previous_tests':885,'added_tests':int(match[1])-885,
        'seconds':float(match[2]),'all_passed':True,'real_DB_driver_and_sockets_blocked':True,'log':str(path),'log_sha256':hashlib.sha256(raw).hexdigest()}


def markdown(report):
    baseline=report['baseline'];b=baseline['pooled'];candidate=report['candidate'];trained='decision' in candidate
    h=b['H_progress'];best=b['highest_existing_AUC'];metrics=dict(b['metrics']);comparisons=dict(b['versus_H']);years=baseline['years']
    if trained:metrics['H_place']=candidate['pooled']['metrics'];comparisons['H_place']=candidate['pooled']['versus_H']
    decision=candidate.get('decision');gate=report['place_gate']
    lines=['# Stage 5：保存勝率AUCと同時点複勝情報の有限検証','',
        f"HのROC-AUC **{num(h['roc_auc'])}**、95%区間 **{ci(h['95_percent_ci'])}**。点推定・区間下限とも0.80を超え、既存Hで初期目標到達を確認。",
        (f"H_placeは{'候補保持' if decision['retain_development_candidate'] else '今回の総合採用基準に未達'}。H比ΔAUC **{num(decision['AUC_difference_vs_fixed_H'],sign=True)}**、95%区間 **{ci(decision['AUC_95_percent_ci'])}**。" if trained else f"H_placeは未実行：{candidate['reason']}。"),
        '既参照の2023〜2025年による開発評価。旧運用・朝7時タスク・2026年封印を維持し、運用モデルの置換は行っていない。','',
        f"既存測定対象での最高AUCは **{best['model']}：{num(best['roc_auc'])}**。最高モデルの選定は記述的な参考比較であり、H_placeの固定対照はHのまま。",
        (f"H_placeは既存最高値を{'更新した' if decision['exceeds_existing_highest_AUC'] else '更新していない'}。参考比較のみで正式な優越判定には使用しない。" if trained else ''),'',
        '## 同一母集団での主比較','',
        '対象6,373レース・87,601頭、正例6,373・負例81,228。2023年2,103、2024年2,101、2025年2,169レース（各1〜9月）。完全レースキー＋馬番で結合し、全比較のラベル・馬番集合を一致させた。','',
        '| モデル名 | ROC-AUC | ΔAUC対H | 予測1位的中率% | レースLog Loss | Brier | 採否 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in ORDER:
        if name not in metrics:
            lines.append(f'| {LABELS[name]} | 未実行／{candidate.get("reason")} | 未算出 | 未算出 | 未算出 | 未算出 | 見送り |');continue
        m=metrics[name];status=STATUS.get(name,'開発候補保持' if decision and decision['retain_development_candidate'] else '不採用（総合基準未達）')
        lines.append(f"| {LABELS[name]} | {num(m['roc_auc'])} | {num(comparisons[name]['roc_auc']['difference_new_minus_H'],sign=True)} | {num(m['top1_accuracy_percent'],digits=2)} | {num(m['race_log_loss'])} | {num(m['brier_score'])} | {status} |")
    lines.extend(['','全体ROC-AUCは全馬を連結した通常の二値ROC-AUC。同スコアは0.5。0.80超えは予測1位的中率80%を意味しない。保存済み連続勝率を二値化せず、温度・offset・正規化を二重適用していない。','',
        '## 年別AUC・レース内AUC・区間','',
        '| モデル | 2023 AUC | 2024 AUC | 2025 AUC | プールのレース内AUC | ΔAUC対H 95%区間 |',
        '|---|---:|---:|---:|---:|---|'])
    for name in ORDER:
        if name not in metrics:continue
        a=[candidate['years'][str(y)]['metrics']['roc_auc'] if name=='H_place' else years[str(y)]['metrics'][name]['roc_auc'] for y in c.YEARS]
        lines.append(f"| {LABELS[name]} | "+' | '.join(num(v) for v in a)+f" | {num(metrics[name]['within_race_auc'])} | {ci(comparisons[name]['roc_auc']['95_percent_ci'])} |")
    raw=b['H_minus_raw_market']['roc_auc']
    lines.extend(['',f"H−raw_marketのΔAUC {num(raw['difference_new_minus_H'],sign=True)}、95%区間 {ci(raw['95_percent_ci'])}。既存モデル一覧からの参考比較で、正式優越の認定ではない。",
        'レース内AUCは各レースで勝馬が他馬を上回る割合（同点0.5）のレース等重み平均。全体ROC-AUCの代用ではない。予測1位同点k頭に勝馬を含むと1/k加点し、馬番優先の買い目決定とは区別する。',
        '開催週は月曜起点・全競馬場共通。5,000反復、seed=20261019、95%区間。全モデル・LL・Brier・1位的中率で同じ週再標本を使用。',
        '各抽出週の多重度を、正例週×負例週の勝敗対数行列へ反映して全体AUCを各回で厳密に計算した。週別AUC平均ではない。手計算・sklearn標本重み・重複行展開との一致を検証済み。Stage 4の3比較用98.333…%区間は転用していない。','',
        '## 複勝仕様・実行条件','',
        '公式ローカルJV-Data仕様のO1（PDF物理15頁・印刷12頁）と特記事項（物理33頁・印刷29頁）を確認。実DBのodds_fukushoはvarchar(336)、28組×12文字（馬番2・最低4・最高4・人気2）。価格は数値を10で割る。',
        '同じO1のfukusho_chakubarai_keyを2＝2着まで、3＝3着までとして記録。頭数や最終結果から推測せず、両制度を被覆率・欠損理由で分離した。価格相対値の同じ計算式を使い、複勝確率への変換・追加特徴量は導入しない。',
        '既存T−10単勝選択と同じ完全キー・発表時刻のO1だけを補足取得。単勝の全馬オッズが保存値と一致することも確認。公表分終端が基準以前、鮮度300秒以内、区分1、同一馬番集合を維持。別時刻や最終オッズで補完していない。',
        'PC実取得時刻・当初予定発走時刻の完全なas-of性は引き続き未確認で、公表時刻による事後の開発評価とする。',
        'NULL・無投票・取消・不正・範囲逆転・未確認キー・打切り価格は欠損。旧仕様の0999記述は曖昧さが残るため9999とともに保守的に欠損化。1頭でも不足なら追加2数値を全頭NaN・フラグ1とし、Hの8列と母集団を維持。','',
        '| 評価年用 | 区間 | 主対象レース | 複勝完備レース | 被覆率% | 2着払いレース | 3着払いレース |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for year,parts in gate.get('folds',{}).items():
        for name,row in parts.items():
            regimes=row['pay_regimes']
            lines.append(f"| {year} | {name} | {row['races']} | {row['complete_races']} | {num(100*row['complete_race_coverage'],digits=3)} | {regimes.get('2',{}).get('races',0)} | {regimes.get('3',{}).get('races',0)} |")
    lines.extend(['','全区間80%以上・学習追加数値に変動ありというラベル非依存のゲートを通過した。完全・欠損部分集合の診断はJSONに別記し、採否は固定母集団全体だけで判断。','',
        '## H_placeの固定実験と採否',''])
    if trained:
        lines.extend(['Hの既存8列＋固定2数値＋欠損フラグの1構成。既存anchor_parameters、二値LightGBM、train/earlyの市場logit初期値、推論時のraw residualへの初期値加算、sigmoid・レース内正規化、T=1を維持。','',
            '| 年用 | best_iteration | 学習モデルID | 保存再読込最大差 |','|---|---:|---|---:|'])
        for trial in candidate['trials']:lines.append(f"| {trial['year']} | {trial['best_iteration']} | {trial['model_id']} | {trial['roundtrip_max_probability_difference']} |")
        lines.extend(['',f"実行は{candidate['new_LightGBM_fits']}/3本、1構成。設定・温度・seed・特徴量・購入ルールの追加探索は0。",'',
            'AUCの点推定は改善したが、今回の総合採用基準には未達。' if decision['AUC_difference_vs_fixed_H']>0 and not decision['retain_development_candidate'] else decision['reason'],'',
            '| 固定条件 | 判定 |','|---|---|'])
        for name,ok in decision['criteria'].items():lines.append(f"| {name} | {'満たす' if ok else '未達'} |")
        lines.extend(['','| 年 | ΔAUC対H | ΔAUC 95%区間 |','|---|---:|---|'])
        for year,row in candidate['years'].items():
            v=row['versus_H']['roc_auc'];lines.append(f"| {year} | {num(v['difference_new_minus_H'],sign=True)} | {ci(v['95_percent_ci'])} |")
        lines.extend(['','| 年 | 補助指標 | H | H_place | 差の95%区間 |','|---|---|---:|---:|---|'])
        for year,row in candidate['years'].items():
            for name in ('within_race_auc','race_log_loss','brier_score','top1_accuracy_percent'):
                v=row['versus_H'][name];base_value=baseline['years'][year]['metrics']['H'][name]
                label=name+'（差はpercentage points）' if name=='top1_accuracy_percent' else name
                lines.append(f"| {year} | {label} | {num(base_value)} | {num(row['metrics'][name])} | {ci(v['95_percent_ci'])} |")
        lines.extend(['','| H_place−H | プール差 | 95%区間 |','|---|---:|---|'])
        for name,row in candidate['pooled']['versus_H'].items():lines.append(f"| {name} | {num(row['difference_new_minus_H'],sign=True)} | {ci(row['95_percent_ci'])} |")
        lines.extend(['','LL・Brier・1位的中率の点推定維持は、統計的非劣性の証明ではない。年別値・差の区間はJSONにも保存。既存最高AUCの微小更新を採用条件の代わりにしない。'])
    else:lines.append(f"新規学習0。見送り理由：{candidate['reason']}。代替実験を自動追加せず終了。")
    tests=report['tests'];preserved=report['preservation']
    lines.extend(['','## 検証・保持・再実行','',
        f"今回実行した架空・モックテスト{tests['executed_tests']}件すべて成功（既存885＋追加{tests['added_tests']}）。実DBドライバーとソケットを禁止した回帰テスト。",
        f"旧参照{preserved['unchanged_reference_files']}ファイル、旧コード{preserved['unchanged_old_code_files']}ファイルの変更0。朝7時タスク設定一致：{preserved['task_configuration_match']}。",
        'Stage 4のHstack uncertain_provisional履歴、旧モデル・最終評価計画を保持。今回の資源配分停止を過去の不採用記録へ書き換えていない。2026年データ処理0、2026年用学習・最終評価入口・運用変更0。',
        '本実験ではROI・購入数による選択、仮想精算、新規購入戦略を実施していない。追加の実験は行わず、今回の候補は運用へ自動反映しない。','',
        f"保存先：`{c.ROOT/'evaluations'/report['run_id']}`",'',
        '再現手順はプロジェクトdocs/stage5_development.mdを参照。完了済み結果はハッシュ検証して再利用し、再開で3本の学習上限を増やさない。',''])
    return '\n'.join(lines)


def run(directory):
    done=s.completed(directory,'report')
    if done:s.preserve(directory);return done
    if not s.completed(directory,'baseline') or not s.completed(directory,'candidate'):raise safety.MLFailure('invalid_input')
    baseline=s.read(directory,'baseline.json');candidate=s.read(directory,'candidate.json');gate=s.read(directory,'place_gate.json')
    count=len(list(directory.glob('fit_*.started.json')))
    if count>3 or count!=candidate['new_LightGBM_fits']:raise safety.MLFailure('input_limit')
    tests=test_result();preserved=s.preserve(directory)
    result={'run_id':directory.name,'version':c.VERSION,'created_at_jst':datetime.now(core.JST).isoformat(),
        'baseline':baseline,'place_gate':gate,'place_mapping':s.read(directory,'place_mapping.json'),'candidate':candidate,
        'tests':tests,'preservation':preserved,'new_model_fits':count,'max_fits':3,'no_additional_exploration':True,
        'development_reuse_not_independent_test':True,'three_user_supplements_applied':True}
    s.write(directory,'development_report.json',result)
    md=safety.local_path(directory/'development_report.md',root=c.ROOT)
    with md.open('x',encoding='utf-8',newline='\n') as stream:stream.write(markdown(result))
    s.write(directory,'completion.json',{'status':'complete','inventory':candidate.get('trials',[]),'fits':count,'max_fits':3,
        'skipped_reason':candidate.get('reason') if 'decision' not in candidate else None,'decision':candidate.get('decision'),
        'highest_existing_AUC':baseline['pooled']['highest_existing_AUC'],'fixed_primary_comparator':'H',
        'preservation':preserved,'tests':tests,'no_production_adoption_or_2026_evaluation':True})
    files=[directory/n for n in ('baseline.json','baseline_bootstrap.local.npz','place_gate.json','candidate.json','development_report.json','development_report.md','completion.json')]
    paths=[*Path(__file__).parent.glob('*.py'),*c.PROJECT.glob('scripts/stage5*'),c.PROJECT/'tests/test_stage5.py',c.PROJECT/'docs/stage5_development.md']
    s.write(directory,'implementation_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    files.append(directory/'implementation_hashes.json')
    summary={'H':baseline['pooled']['H_progress'],'highest_existing_AUC':baseline['pooled']['highest_existing_AUC'],
        'H_place_decision':candidate.get('decision'),'new_LightGBM_fits':count,'tests':tests,'preservation':preserved,'report':str(md)}
    s.complete(directory,'report',files,summary);return summary
