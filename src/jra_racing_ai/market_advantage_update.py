"""Amendment 2: fixed conditions, all horses with p*O>=1, weekly ROI CI.

No fitting; closing prices are a post-decision diagnostic only. This module
never calls or alters the prospective recorder or its evaluation.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import os
import re
import time
import numpy as np

from . import market_advantage as old
from .ml import market
from .stage5.metrics import WeekBootstrap
from .win_sim import core, local, publication

VERSION='market_advantage_stage1_amendment2_v1'
ODDS_BANDS=('1〜3倍未満','3〜10倍未満','10〜30倍未満','30〜100倍未満','100倍以上')


def folder(run_id):
    if not re.fullmatch('[a-f0-9]{32}',run_id):raise ValueError('invalid_run_id')
    return old.ROOT/run_id


def plan(directory):
    value=old.read(directory/'plan.json')
    if (value.get('version')!=VERSION or value.get('fits')!=0 or
            old.digest(directory/'plan.json')!=(directory/'plan.sha256').read_text(encoding='ascii').strip() or
            [r['id'] for r in value['conditions']]!=['C1','C2','C3','C4','C5'] or
            value['purchase']['minimum_probability_times_odds']!='1.0' or
            value['purchase']['maximum_horses_per_race'] is not None):raise ValueError('plan_mismatch')
    return value


def all_value_horses(row):
    """All inclusive value candidates. No max-one, finish or closing-odds input."""
    return sorted(h['horse_number'] for h in row['runners']
                  if core.decimal(h['probability'])*core.decimal(h['decision_odds'])>=1)


def prepare(run_id):
    directory=folder(run_id);p=plan(directory);previous=folder(p['old_run_id'])
    if old.digest(previous/'selections.local.json')!=p['old_decisions_sha256']:raise ValueError('source_changed')
    if old.digest(previous/'集計結果.json')!=p['old_report_sha256']:raise ValueError('source_changed')
    original=old.read(previous/'plan.json')
    if original['conditions']!=p['conditions']:raise ValueError('conditions_changed')
    rows=old.read(previous/'selections.local.json')
    for row in rows:
        day=row['race_key'][0]+'-'+row['race_key'][1][:2]+'-'+row['race_key'][1][2:]
        if not old.in_scope(day):raise ValueError('invalid_period')
        row['selected_horses']=all_value_horses(row)
        # Preserve the original C1-C5 predicates and max-EV candidate, not its old purchase.
        row.pop('selected',None)
    old.write(directory/'purchase_decisions.local.json',rows)
    old.write(directory/'purchase_commit.json',dict(
        plan_sha256=old.digest(directory/'plan.json'), decisions_sha256=old.digest(directory/'purchase_decisions.local.json'),
        committed_at=datetime.now(core.JST).isoformat(), results_and_closing_odds_not_used=True))
    return dict(status='prepared',run_id=run_id,races=len(rows),tickets=sum(len(r['selected_horses']) for r in rows),fits=0)


def decisions(directory):
    p=plan(directory);commit=old.read(directory/'purchase_commit.json')
    if (commit['plan_sha256']!=old.digest(directory/'plan.json') or
            commit['decisions_sha256']!=old.digest(directory/'purchase_decisions.local.json')):
        raise ValueError('decision_commit_mismatch')
    return p,old.read(directory/'purchase_decisions.local.json')


def collect_final(run_id,db_session=None):
    directory=folder(run_id);p,rows=decisions(directory)
    if db_session is None:
        from .db_connection import read_only_session
        from .win_sim_cli import connection_config
        def db_session():return read_only_session(connection_config(10000),os.environ.get('PGPASSWORD'),gate_timeout_ms=3000)
    targets={tuple(r['race_key']):r for r in rows if r['selected_horses']}
    root=directory/'final-odds';root.mkdir(exist_ok=True)
    sources=[];started=time.monotonic();budget=local.Budget(180)
    with db_session() as db:
        columns=publication._columns(db,'jvd_o1',publication.O1_FIELDS,budget)
        if columns is None:raise ValueError('final_odds_table_unavailable')
        keys=sorted(targets)
        for offset in range(0,len(keys),32):
            path=root/f'{offset//32:04d}.local.json'
            if path.exists():sources.append(path);continue
            if time.monotonic()-started>180:raise ValueError('final_odds_time_budget')
            group=tuple(keys[offset:offset+32])
            raw=local._rows(db,budget,'jvd_o1',publication.O1_FIELDS,columns,group,128,join_targets=True)
            by_race=defaultdict(list)
            for value in raw:
                value['source_table']='jvd_o1';by_race[tuple(value['race_key'])].append(value)
            selected=[]
            for race in group:
                roster=tuple(h['horse_number'] for h in targets[race]['runners'])
                quote,reason=market.select_final(race,by_race[race],roster)
                if quote and quote.category not in ('4','5'):
                    quote=None;reason='confirmed_category4_5_unavailable'
                selected.append(dict(race_key=list(race),odds={h:float(v) for h,v in quote.odds.items()} if quote else None,
                                     category=quote.category if quote else None,reason=reason,source='jvd_o1'))
            old.write(path,dict(records=selected,read_at=datetime.now(core.JST).isoformat(),purchase_commit_sha256=old.digest(directory/'purchase_commit.json')))
            sources.append(path)
    old.write(directory/'final_odds_manifest.json',dict(
        source='readonly_jvd_o1_confirmed4_5_only',target_races=len(targets),batches=len(sources),
        files={str(path):old.digest(path) for path in sources},seconds=time.monotonic()-started,
        selection_preceded_final_read=True,labels_queried=False,new_fits=0))
    return dict(status='collected',run_id=run_id,races=len(targets),batches=len(sources),fits=0)


def week(race):
    d=date(int(race[0]),int(race[1][:2]),int(race[1][2:]))
    return (d-timedelta(days=d.weekday())).isoformat()


def financial(tickets,bootstrap):
    result=core.financial([{'ticket':t['settlement']} for t in tickets],'ticket')
    matrix=np.zeros((len(bootstrap.weeks),3),dtype=float)
    index={str(w):i for i,w in enumerate(bootstrap.weeks)}
    for item in tickets:
        row=matrix[index[week(item['race_key'])]];settle=item['settlement']
        row[0]+=settle['stake_yen']
        if settle['status']=='settled':row[1]+=settle['payout_yen']
        else:row[2]+=1
    draw=bootstrap.counts@matrix
    roi=np.divide(100*draw[:,1],draw[:,0],out=np.full(len(draw),np.nan),where=(draw[:,0]>0)&(draw[:,2]==0))
    valid=roi[np.isfinite(roi)]
    result['ROI95CI']=[float(v) for v in np.quantile(valid,[.025,.975])] if len(valid) and result['unsettled_count']==0 else None
    result['zero_investment_replica_fraction']=float(np.mean(draw[:,0]==0))
    result['pending_replica_fraction']=float(np.mean(draw[:,2]>0))
    result['defined_replicates']=int(len(valid));result['weeks_with_bets']=int(np.sum(matrix[:,0]>0))
    result['purchased_races']=len({tuple(t['race_key']) for t in tickets})
    return result


def settle_tickets(rows,outcomes):
    ai=[];favorite=[]
    for row in rows:
        race=tuple(row['race_key']);roster=tuple(h['horse_number'] for h in row['runners'])
        outcome=outcomes.get(race)
        for h in row['selected_horses']:
            horse=next(horse for horse in row['runners'] if horse['horse_number']==h)
            ai.append(dict(race_key=list(race),**horse,settlement=core.settle(h,outcome,roster)))
        # One favorite in every condition-eligible race, regardless of AI stake.
        favorite.append(dict(race_key=list(race),horse_number=row['favorite'],
                             settlement=core.settle(row['favorite'],outcome,roster)))
    return ai,favorite


def quality(rows,winners):
    available=[r for r in rows if tuple(r['race_key']) in winners]
    scores={}
    for field,name in (('probability','AI'),('market_probability','market')):
        predictions=[core.PredictionRace(tuple(r['race_key']),{h['horse_number']:h[field] for h in r['runners']}) for r in available]
        scores[name]=old.stage1_analysis.measure(predictions,winners)
    return scores


def odds_band(price):
    return ODDS_BANDS[next((i for i,upper in enumerate((3,10,30,100,float('inf'))) if price<upper))]


def closing_drift(tickets,finals):
    result={};reasons=Counter()
    for band in ODDS_BANDS:
        group=[t for t in tickets if odds_band(t['decision_odds'])==band];ratios=[];missing=Counter()
        for item in group:
            last=finals.get(tuple(item['race_key']))
            if not last or last.get('odds') is None:
                missing[(last or {}).get('reason') or 'final_odds_unavailable']+=1;continue
            price=last['odds'].get(item['horse_number'])
            if price is None:missing['final_horse_missing']+=1;continue
            ratios.append(float(core.decimal(price)/core.decimal(item['decision_odds'])))
        reasons.update(missing)
        result[band]={'tickets':len(group),'matched':len(ratios),'missing':sum(missing.values()),
                      'quantiles':dict(zip(('p10','p25','median','p75','p90'),[float(v) for v in np.quantile(ratios,[.1,.25,.5,.75,.9])])) if ratios else None,
                      'ratio_distribution':{'0.8未満':sum(x<.8 for x in ratios),'0.8以上1未満':sum(.8<=x<1 for x in ratios),
                                            '1以上1.2未満':sum(1<=x<1.2 for x in ratios),'1.2以上':sum(x>=1.2 for x in ratios)},
                      'missing_reasons':dict(missing)}
    return dict(bands=result,missing_reasons=dict(reasons))


def summarize(rows,winners,outcomes,finals,bootstrap):
    ai,favorite=settle_tickets(rows,outcomes)
    known=[t for t in ai if tuple(t['race_key']) in winners]
    ranks=Counter(old.rank_band(t['reference_popularity']) for t in ai)
    return dict(target_races=len(rows),target_runners=sum(len(r['runners']) for r in rows),scores=quality(rows,winners),
                AI=financial(ai,bootstrap),favorite=financial(favorite,bootstrap),
                mean_AI_probability=old.mean([t['probability'] for t in ai]),
                mean_market_probability=old.mean([t['market_probability'] for t in ai]),
                mean_probability_gap=old.mean([t['probability']-t['market_probability'] for t in ai]),
                selected_binary_LL_AI=old.mean([old.binary_loss(t['probability'],int(t['horse_number']==winners[tuple(t['race_key'])])) for t in known]),
                selected_binary_LL_market=old.mean([old.binary_loss(t['market_probability'],int(t['horse_number']==winners[tuple(t['race_key'])])) for t in known]),
                popularity={band:{'tickets':ranks[band],'percent':100*ranks[band]/len(ai) if ai else None} for band in old.BANDS},
                closing_drift=closing_drift(ai,finals))


def render(directory,result):
    def fmt(x,n=2):return '未算出' if x is None else f'{x:,.{n}f}'
    def interval(x):return '未算出' if x is None else f'[{x[0]:.2f}, {x[1]:.2f}]'
    comparisons=[];popularities=[];drifts=[]
    for key,v in result['conditions'].items():
        for period,row in [('全期間',v),*v['by_year'].items()]:
            f=row['AI'];b=row['favorite'];a=row['scores']['AI'];m=row['scores']['market'];ci=f['ROI95CI'];bi=b['ROI95CI']
            comparisons.append({'条件番号':key,'条件':old.NAMES[key],'期間':period,'対象レース数':row['target_races'],'対象頭数':row['target_runners'],
                '購入件数':f['bet_count'],'購入レース数':f['purchased_races'],'的中数':f['hit_count'],'投資額_円':f['investment_yen'],
                '払戻額_円':f['payout_yen'],'損益_円':f['net_profit_yen'],'回収率_パーセント':f['return_rate_percent'],
                '回収率95下限':ci[0] if ci else None,'回収率95上限':ci[1] if ci else None,'未精算件数':f['unsettled_count'],
                '市場本命購入件数':b['bet_count'],'市場本命回収率_パーセント':b['return_rate_percent'],
                '市場本命回収率95下限':bi[0] if bi else None,'市場本命回収率95上限':bi[1] if bi else None,
                '市場本命損益_円':b['net_profit_yen'],'AI二値LL':a.get('log_loss'),'市場二値LL':m.get('log_loss'),
                '購入馬_AI勝率平均':row['mean_AI_probability'],'購入馬_市場勝率平均':row['mean_market_probability'],
                '購入馬_勝率差平均':row['mean_probability_gap']})
        for band,x in v['popularity'].items():popularities.append({'条件番号':key,'条件':old.NAMES[key],'T10参考人気帯':band,'購入件数':x['tickets'],'構成比_パーセント':x['percent']})
        for band,x in v['closing_drift']['bands'].items():
            drifts.append({'条件番号':key,'条件':old.NAMES[key],'T10単勝オッズ帯':band,'購入件数':x['tickets'],'確定オッズ対応件数':x['matched'],
                           '欠損件数':x['missing'],**(x['quantiles'] or {k:None for k in ('p10','p25','median','p75','p90')}),**x['ratio_distribution']})
    for name,rows in (('条件比較.csv',comparisons),('人気帯.csv',popularities),('確定オッズ変化.csv',drifts)):
        old.csv_output(directory/name,list(rows[0]),rows)
    lines=['# 市場優位条件・第1段階（追補2適用）','',
        '**追加学習0fit。2023〜2025年各1〜9月の既参照開発期間による探索です。独立検証・前向き成績・市場優位の正式認定ではありません。**','',
        '## 使用予測と分析順序','',
        '予測はStage 3のABCD年次時系列OOFの1系統だけです。各年の前年までで学習・早期終了・較正済みで、保存温度T=0.75の確率を変更せず使用。'
        '現在の39列対照と同じ39項目を基礎とし、血統45列・休養47列の共通部分に最も近いため選びました。2026年まで学習した3モデル自身の過去年予測は使っていません。',
        '一致を確認したのは特徴量名と列順です。当時の生成版・欠損状況は現行モデルと異なり得るため、現行3モデルの収益性を直接証明する結果ではありません。',
        '10月31日の前向き研究とは別領域です。主力H・Stage 7の途中成績、正式21週間評価を使用していません。','',
        '追補2の受領前に、EV5%以上・最大1点で一度集計済みでした。その結果を次節に開示します。'
        '今回は条件をそのまま保持し、購入規則だけをユーザーの追補に従って変更して新規事前登録しました。','',
        '## 追補前に既に見た結果（再集計していない旧記録）','',
        '| 条件 | 旧購入件数 | 旧回収率 | 旧損益（円） |','|---|---:|---:|---:|']
    for key,v in result['already_seen']['results'].items():lines.append(f"| {v['name']} | {v['bets']:,} | {v['ROI_percent']:.2f}% | {v['profit_yen']:,} |")
    lines += ['','旧記録は全5条件が赤字でした。条件の選択をやり直さず、許可された規則での再集計であることを明記します。',
        '追補前に確認した範囲は、全体・条件別・年別の回収率／損益、二値LL、人気帯、購入馬の平均勝率差、市場本命との比較、最大払戻依存、欠損数です。'
        '元の集計結果全体を追補前集計.jsonに保存し、旧記録も上書きしていません。','',
        '## 固定条件と購入規則','',
        'T−10・鮮度300秒以内の公表オッズを使用し、分終端と同一出走馬集合を確認。発走時刻変更は既存規則で除外し、最終オッズによる補完なし。'
        'PC実取得時刻および当時の発走予定の完全な履歴は未検証なので、公表時刻基準の開催後評価です。','',
        '**各条件に該当するレースで、AI勝率×T−10単勝オッズが1.0以上の全馬を1頭100円購入。1レース1点には限定しません。**'
        '人気・血統・結果に応じた閾値変更はありません。市場本命は各条件の対象レースごとに1頭100円で、AIとは投資額が異なります。'
        '最低オッズ同率は馬番の小さい馬を選びます。','',
        'C1/C2は本命同士の一致／不一致、C3/C4/C5は**元の全馬EV最大候補**によるレース条件のままです。'
        '例えばC3に該当するレースでは、購入全馬が4〜6番人気とは限りません。追補後に条件を馬単位へ変更していません。条件間は重複し、合算できません。','']
    for v in result['condition_definitions']:lines.append(f"- {v['id']}：{v['name']}。{v['predicate']}。理由：{v['reason']}。")
    lines += ['','## 被覆率と除外','',
        f"対象OOFは{result['coverage']['target_races']:,}レース・{result['coverage']['target_runners']:,}頭。"
        f"T−10の利用条件を満たす{result['coverage']['usable_races']:,}レース・{result['coverage']['usable_runners']:,}頭"
        f"（{result['coverage']['coverage_percent']:.2f}%）で比較しました。",'',
        '利用不可591レースは、発走時刻変更の当時情報を再現できないためです。単なるオッズ欠落とは区別し、最終オッズで埋めていません。'
        'C5はT−30不足の417レースで判定不能。C4/C5の判定不能を基準の全対象からは落としていません。','',
        '## 条件別収益と95%区間','',
        '| 条件 | 対象レース | 購入件数 | 投資（円） | 払戻（円） | 損益（円） | 回収率［95%区間］ | 市場本命回収率［95%区間］ | 未精算 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in result['conditions'].values():
        a=v['AI'];b=v['favorite']
        lines.append(f"| {v['name']} | {v['target_races']:,} | {a['bet_count']:,} | {a['investment_yen']:,} | {fmt(a['payout_yen'],0)} | {fmt(a['net_profit_yen'],0)} | {fmt(a['return_rate_percent'])}% {interval(a['ROI95CI'])} | {fmt(b['return_rate_percent'])}% {interval(b['ROI95CI'])} | {a['unsettled_count']} |")
    lines += ['','区間は既存WeekBootstrapの週再標本を再利用し、月曜始まり・全競馬場共通の週ごとの投資と公式払戻を合計してから比率を計算しました。'
              '5,000回・seed20261019・全条件で同じ再標本です。週別回収率の単純平均ではありません。未精算があれば全体回収率・区間は未算出、購入0件も未算出です。','',
        '区間は探索上の名目95%で、多重比較の補正はしていません。同じ開発データを繰り返し参照した選択バイアスは、この区間によって解消されません。','',
        '払戻は保存済みの検証済み公式HR（100円あたり円額）。T−10オッズは購入判断だけに使用します。','',
        '## 確率品質（低いほど良い）と購入馬の市場との差','',
        '| 条件 | AI二値LL・全馬 | 市場二値LL・全馬 | 購入馬の平均AI勝率 | 購入馬の平均市場勝率 | 平均勝率差（ポイント） |',
        '|---|---:|---:|---:|---:|---:|']
    for v in result['conditions'].values():
        lines.append(f"| {v['name']} | {fmt(v['scores']['AI'].get('log_loss'),6)} | {fmt(v['scores']['market'].get('log_loss'),6)} | {fmt(100*v['mean_AI_probability'] if v['mean_AI_probability'] is not None else None)}% | {fmt(100*v['mean_market_probability'] if v['mean_market_probability'] is not None else None)}% | {fmt(100*v['mean_probability_gap'] if v['mean_probability_gap'] is not None else None)} |")
    lines += ['','二値LLは各条件の同じレース・全馬集合で比較し、購入馬だけで再正規化しません。'
              '人気帯別の購入件数・割合は人気帯.csv、年別は条件比較.csvに保存しています。','',
        '## 購入後のオッズ変化','',
        '確定オッズ÷T−10オッズを購入馬だけで集計。1未満なら確定までにオッズが下がり、1超なら上がったことを表します。'
        '確定区分4/5のみを使い、区分3しかないケースや集合不一致は欠損扱い。選択ファイルのハッシュ固定後に取得し、購入に影響させていません。','',
        '| T−10オッズ帯（全対象基準） | 購入件数 | 確定オッズ対応 | 欠損 | 比の中央値 | 10〜90百分位 |',
        '|---|---:|---:|---:|---:|---:|']
    for band,v in result['conditions']['ALL']['closing_drift']['bands'].items():
        q=v['quantiles'] or {};lines.append(f"| {band} | {v['tickets']:,} | {v['matched']:,} | {v['missing']:,} | {fmt(q.get('median'),3)} | {fmt(q.get('p10'),3)}〜{fmt(q.get('p90'),3)} |")
    lines += ['','全5条件の中央値・10/25/75/90百分位・分布は確定オッズ変化.csvへ保存。欠損理由は集計結果.jsonを参照。','',
        '## 次に検証する条件','']
    candidates=[v for k,v in result['conditions'].items() if k!='ALL' and v['AI']['return_rate_percent'] is not None]
    if candidates:
        best=max(candidates,key=lambda v:v['AI']['return_rate_percent']);a=best['AI']
        lines += [f"探索上の最高回収率は **{best['name']}：{a['return_rate_percent']:.2f}%**。"
                  f"年別は"+'、'.join(f"{y}年{v['AI']['return_rate_percent']:.2f}%" for y,v in best['by_year'].items() if v['AI']['return_rate_percent'] is not None)+'。',
                  '候補としてはこの条件を最初に別期間で確認できますが、5条件の中から選んだ結果です。収益の符号と区間、年別の再現性を確認するまでは優位とは認定しません。']
        if a['return_rate_percent']<=100:lines += ['今回の最高値も100%に届かず、黒字の市場優位条件は発見できませんでした。']
    lines += ['','新しい条件・閾値・学習へ進まず、第1段階で終了します。'
              '本結果による39/45/47列モデル・正式21週計画・H/Stage 7・本番タスク・KEIBA LABの変更はありません。', '',
        f"研究ID：`{directory.name}`。計画SHA-256：`{result['plan_sha256']}`。追加学習0fit。保存済み予測・元データのハッシュ不変を確認。",'']
    with (directory/'研究レポート.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))


def analyze(run_id):
    directory=folder(run_id);p,rows=decisions(directory);previous=folder(p['old_run_id'])
    source=old.read(previous/'集計結果.json')
    if any(old.digest(Path(path))!=sha for path,sha in source['source_hashes'].items()):raise ValueError('source_changed')
    predictions={tuple(r['race_key']):{h['horse_number']:h['probability'] for h in r['runners']} for r in rows}
    inputs=old.Inputs();winners,outcomes,answer_reasons=inputs.answers(predictions)
    finals={};final_manifest=old.read(directory/'final_odds_manifest.json') if (directory/'final_odds_manifest.json').exists() else None
    if final_manifest:
        for path,sha in final_manifest['files'].items():
            if old.digest(Path(path))!=sha:raise ValueError('closing_odds_changed')
            for record in old.read(path)['records']:finals[tuple(record['race_key'])]=record
    bootstrap=WeekBootstrap([week(r['race_key']) for r in rows],repetitions=5000)
    values={}
    for key in old.NAMES:
        selected=rows if key=='ALL' else [r for r in rows if r['conditions'][key] is True]
        v=summarize(selected,winners,outcomes,finals,bootstrap);v['name']=old.NAMES[key]
        v['unassessable_races']=0 if key=='ALL' else sum(r['conditions'][key] is None for r in rows)
        v['by_year']={str(year):summarize([r for r in selected if r['year']==str(year)],winners,outcomes,finals,bootstrap) for year in old.YEARS}
        values[key]=v
    # Keep weekly totals and ticket-level results local, independently reproducible.
    tickets,favorites=settle_tickets(rows,outcomes)
    old.write(directory/'tickets.local.json',dict(AI=tickets,favorite=favorites))
    metadata=bootstrap.metadata();metadata.pop('AUC_method',None);metadata['calculation']='ratio_of_resampled_weekly_total_payout_to_investment'
    result=dict(version=VERSION,created_at=datetime.now(core.JST).isoformat(),plan_sha256=old.digest(directory/'plan.json'),fits=0,
                single_predictor=source['predictor'],models=source['model_metadata'],condition_definitions=p['conditions'],
                already_seen=p['already_seen_before_amendment2'],conditions=values,bootstrap=metadata,
                coverage={'target_races':sum(v['OOF_races'] for v in source['availability'].values()),
                          'target_runners':sum(v['OOF_runners'] for v in source['availability'].values()),
                          'usable_races':len(rows),'usable_runners':sum(len(r['runners']) for r in rows),
                          'coverage_percent':100*len(rows)/sum(v['OOF_races'] for v in source['availability'].values()),
                          'by_year':source['availability'],'exclusions':source['exclusion_reasons']},
                answer_reasons=answer_reasons,final_odds_manifest_sha256=old.digest(directory/'final_odds_manifest.json') if final_manifest else None,
                saved_sources_unchanged=all(old.digest(Path(path))==sha for path,sha in source['source_hashes'].items()),
                old_decisions_unchanged=old.digest(previous/'selections.local.json')==p['old_decisions_sha256'],
                old_report_unchanged=old.digest(previous/'集計結果.json')==p['old_report_sha256'])
    if not all(result[k] for k in ('saved_sources_unchanged','old_decisions_unchanged','old_report_unchanged')):raise ValueError('source_changed')
    old.write(directory/'集計結果.json',result);render(directory,result)
    return dict(status='ok',run_id=run_id,path=str(directory/'研究レポート.md'),fits=0,
                rows=[dict(condition=k,target_races=v['target_races'],bets=v['AI']['bet_count'],ROI=v['AI']['return_rate_percent'],
                           ROI95CI=v['AI']['ROI95CI'],profit_yen=v['AI']['net_profit_yen'],favorite_ROI=v['favorite']['return_rate_percent']) for k,v in values.items()])
