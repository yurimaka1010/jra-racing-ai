"""Japanese, self-contained local report. No network, fitting or source prediction edits."""
from html import escape
from pathlib import Path
import json
from . import pedigree_replay_review as r

COLORS={'control39':'#287575','pedigree45':'#8554b8','reference47':'#c77c30','market':'#64748b','market_favorite':'#596f8a'}
DISPLAY_LABELS=dict(r.LABELS,market_favorite='市場1番人気（T−10）')
FIELDS=tuple(r.METRICS)


def number(value,digits=6):return '未算出' if value is None else f'{value:,.{digits}f}'
def yen(value):return '未精算' if value is None else f'{value:,.0f}円'
def pct(value):return '未算出' if value is None else f'{value:.2f}％'


def change(field,value):
    if value is None:return '未算出'
    if abs(value)<=1e-12:return '同値'
    return '改善方向' if (value>0)==r.METRICS[field][1] else '悪化方向'


def start(width,height,title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title,quote=True)}">',
        '<style>text{font-family:"Yu Gothic","Meiryo",sans-serif;fill:#263449} .small{font-size:13px}.label{font-size:15px}.title{font-size:20px;font-weight:700}</style>',
        f'<rect width="{width}" height="{height}" rx="16" fill="#fffdf9"/>',
        f'<text x="26" y="34" class="title">{escape(title)}</text>']


def metric_svg(scope,market=False):
    names=(*r.NAMES,'market') if market else r.NAMES;population='P2' if market else 'P1'
    data=scope[population];h=60+len(FIELDS)*176
    out=start(1080,h,'同じレース・全出走馬での精度比較'+('（市場との共通集合）' if market else '（異常レース除外後）'))
    for j,field in enumerate(FIELDS):
        title,higher,meaning=r.METRICS[field];y=67+j*176
        out.append(f'<text x="26" y="{y}" class="label">{title}：{"高いほど良い" if higher else "低いほど良い"}</text>')
        available=[data[n][field] for n in names if data.get(n)]
        maximum=(100 if field=='top1_accuracy_percent' else 1 if field.endswith('auc') else max(available+[.001])*1.12)
        x=220;w=680
        for tick in (0,.5,1):
            xx=x+w*tick;out.append(f'<line x1="{xx}" x2="{xx}" y1="{y+10}" y2="{y+121}" stroke="#e5e8e9"/>')
            out.append(f'<text x="{xx}" y="{y+140}" text-anchor="middle" class="small">{maximum*tick:.3f}</text>')
        for i,n in enumerate(names):
            yy=y+13+i*27;v=data[n][field] if data.get(n) else None
            out.append(f'<text x="26" y="{yy+16}" class="label">{r.LABELS[n]}</text>')
            if v is not None:out.append(f'<rect x="{x}" y="{yy}" width="{w*v/maximum:.4f}" height="19" rx="3" fill="{COLORS[n]}"/>')
            text=number(v,2 if field=='top1_accuracy_percent' else 6)+('％' if field=='top1_accuracy_percent' and v is not None else '')
            out.append(f'<text x="{x+w+18}" y="{yy+16}" class="label">{text}</text>')
    out.append('</svg>');return ''.join(out)


def delta_svg(scope):
    out=start(1080,340,'血統45列 − 39列対照：点推定の差（採否判定ではありません）')
    a=scope['P1'].get('pedigree45');b=scope['P1'].get('control39')
    for i,k in enumerate(FIELDS):
        y=78+i*44;value=a[k]-b[k] if a and b else None;state=change(k,value)
        limit={'within_race_auc':.02,'roc_auc':.02,'race_log_loss':.1,'brier_score':.005,'top1_accuracy_percent':10}[k]
        color='#287575' if state=='改善方向' else '#bc5147' if state=='悪化方向' else '#94a3b8'
        out.append(f'<text x="24" y="{y+16}" class="label">{r.METRICS[k][0]}</text>')
        out.append(f'<line x1="550" x2="550" y1="{y-5}" y2="{y+24}" stroke="#768190"/>')
        out.append(f'<text x="246" y="{y+16}" class="small">横軸 ±{limit:g}</text>')
        if value is not None:
            width=min(190,190*abs(value)/limit)
            out.append(f'<rect x="{550 if value>=0 else 550-width}" y="{y}" width="{width}" height="19" fill="{color}" rx="3"/>')
        unit='ポイント' if k=='top1_accuracy_percent' else ''
        text='未算出' if value is None else f'{value:+.6f}{unit} ／ {state}'
        out.append(f'<text x="750" y="{y+16}" class="label">{text}</text>')
    out.extend(['<text x="26" y="318" class="small">中央が差0。色は良い方向。行ごとに別尺度（範囲外は端で表示）。異なる指標間で長さは比較できません。</text>','</svg>'])
    return ''.join(out)


def distribution_svg(details):
    out=start(1080,260,'本命馬の着順：2着・3着を分けて確認')
    colors=('#287575','#7c88c6','#c5a75a','#dce1e4');labels=('1着','2着','3着','4着以下')
    items=[d for d in details if d['P1']];total=len(items)
    for i,n in enumerate(r.NAMES):
        y=74+i*43;x=215
        counts=[sum(d['models'][n]['finish']==k for d in items) for k in (1,2,3)]
        counts.append(total-sum(counts));out.append(f'<text x="26" y="{y+21}" class="label">{r.LABELS[n]}</text>')
        for count,color in zip(counts,colors):
            width=710*count/total if total else 0
            out.append(f'<rect x="{x}" y="{y}" width="{width}" height="28" fill="{color}"/>')
            if width>25:out.append(f'<text x="{x+width/2}" y="{y+20}" text-anchor="middle" style="fill:{"#fff" if color in colors[:2] else "#263449"};font-size:14px">{count}</text>')
            x+=width
        out.append(f'<text x="942" y="{y+20}" class="label">計{total}レース</text>')
    for i,(label,color) in enumerate(zip(labels,colors)):
        x=220+i*170;out.extend([f'<rect x="{x}" y="218" width="17" height="17" fill="{color}"/>',f'<text x="{x+25}" y="232" class="small">{label}</text>'])
    out.append('</svg>');return ''.join(out)


def money_svg(scope,market_scope=None):
    data=scope.get('finance_common',scope['finance']);count=data['control39']['bet_count'];names=r.NAMES
    if market_scope is not None:data=market_scope['finance_on_three_common'];names=(*r.NAMES,'market_favorite')
    height=250+48*(len(names)-3)
    title=f'本命単勝100円の仮想損益（精算済み共通{count}レース）' if market_scope is None else f'単勝100円：3モデル＋市場本命（元の共通{count}レース）'
    out=start(1080,height,title)
    values=[abs(m['net_profit_yen']) for m in data.values() if m['net_profit_yen'] is not None];maximum=max(values+[100])
    center=540;scale=260/maximum
    out.append(f'<line x1="540" x2="540" y1="58" y2="{205+48*(len(names)-3)}" stroke="#768190"/>')
    for i,n in enumerate(names):
        y=73+i*48;value=data[n]['net_profit_yen'];width=abs(value)*scale if value is not None else 0
        out.append(f'<text x="26" y="{y+21}" class="label">{DISPLAY_LABELS[n]}</text>')
        if value is not None:out.append(f'<rect x="{center if value>=0 else center-width}" y="{y}" width="{width}" height="27" rx="4" fill="{"#287575" if value>=0 else "#bc5147"}"/>')
        out.append(f'<text x="830" y="{y+20}" class="label">{yen(value)} ／ 回収率{pct(data[n]["return_rate_percent"])}</text>')
    out.extend([f'<text x="26" y="{height-20}" class="small">実購入ではありません。払戻は公式単勝払戻を使用。未精算が残る場合、総損益と回収率は未算出。</text>','</svg>'])
    return ''.join(out)


def popularity_svg(scope,names):
    import math
    from .pedigree_popularity import BANDS
    out=start(1080,430,'本命馬のT−10参考人気分布（共通のP3レース）')
    total=scope['P3_races']
    out.append(f'<text x="26" y="60" class="small">対象{total}レース。P2の精度比較集合とは別。公式の最終人気ではありません。</text>')
    maximum=max(5,5*math.ceil(max([b['count'] for n in names for b in scope['models'][n]['bands']]+[0])/5))
    baseline=306;height=208
    for fraction in (0,.5,1):
        y=baseline-height*fraction
        out.extend([f'<line x1="66" x2="1030" y1="{y}" y2="{y}" stroke="#dde2e4"/>',
            f'<text x="53" y="{y+5}" text-anchor="end" class="small">{maximum*fraction:g}</text>'])
    out.append('<text x="26" y="87" class="small">件数</text>')
    for i,(_,label) in enumerate(BANDS):
        middle=185+i*240;width=51;gap=10;span=len(names)*width+(len(names)-1)*gap
        for j,n in enumerate(names):
            b=scope['models'][n]['bands'][i];x=middle-span/2+j*(width+gap)
            bar=height*b['count']/maximum;y=baseline-bar
            share=pct(b['share_percent']);title=f'{r.LABELS[n]}／{label}：{b["count"]}件、{share}。分母はP3の{total}レース。'
            out.append(f'<g data-popularity-model="{n}"><title>{escape(title)}</title><rect x="{x}" y="{y}" width="{width}" height="{bar}" rx="3" fill="{COLORS[n]}"/>')
            out.append(f'<text x="{x+width/2}" y="{y-22}" text-anchor="middle" class="small">{b["count"]}件</text>')
            out.append(f'<text x="{x+width/2}" y="{y-6}" text-anchor="middle" style="font-size:12px">{share}</text></g>')
        out.append(f'<text x="{middle}" y="337" text-anchor="middle" class="label">{label}</text>')
    for j,n in enumerate(names):
        x=35+j*345;m=scope['models'][n]
        mean='未算出' if m['mean_popularity_rank'] is None else number(m['mean_popularity_rank'],2)+'位'
        out.extend([f'<rect x="{x}" y="365" width="15" height="15" rx="3" fill="{COLORS[n]}"/>',
            f'<text x="{x+23}" y="378" class="label">{r.LABELS[n]}</text>',
            f'<text x="{x+23}" y="400" class="small">平均参考人気 {mean}</text>'])
    out.append('</svg>');return ''.join(out)


def assessment(report):
    delta=report['pedigree_minus_control']
    if not delta:return ['照合可能な通常レースがないため、精度差は算出できません。']
    good=[r.METRICS[k][0] for k in FIELDS if change(k,delta[k])=='改善方向']
    bad=[r.METRICS[k][0] for k in FIELDS if change(k,delta[k])=='悪化方向']
    a=report['periods']['all']['P1']['pedigree45'];b=report['periods']['all']['P1']['control39']
    pair=report['pair_disagreements']['pedigree45_vs_control39']
    return [f'血統45列は39列対照に対し、{("・".join(good)+"が改善方向") if good else "改善方向の指標はありませんでした"}。'
            +(f'一方、{"・".join(bad)}は悪化方向でした。' if bad else '悪化方向の指標はありませんでした。'),
        f'本命1着は血統45列{a["top1_hit_count"]}回、39列対照{b["top1_hit_count"]}回（同じ{a["races"]}レース）。',
        f'本命が異なる{pair["different_races"]}レースでは、血統45列の本命が先着{pair["first_finishes_ahead"]}回、対照が先着{pair["second_finishes_ahead"]}回。',
        f'血統45列の本命2着は{report["top_finish_distribution"]["pedigree45"].get("2",0)}回、3着は{report["top_finish_distribution"]["pedigree45"].get("3",0)}回。対照はそれぞれ{report["top_finish_distribution"]["control39"].get("2",0)}回・{report["top_finish_distribution"]["control39"].get("3",0)}回でした。',
        '保存温度は血統45列が1.0、39列対照が1.25。この比較は温度を含む保存モデル全体の比較で、血統6列だけの因果効果を分離したものではありません。',
        '2日間だけの開催後再生です。正式な採用判断・有意差判定・特徴量の再選択には使用しません。']


def save(directory,report):
    from .pedigree_replay_review import csv_file
    views={}
    for key,scope in report['scopes'].items():
        day,venue=key.split('|');details=[d for d in report['details'] if (day=='all' or d['date']==day) and (venue=='all' or d['venue_code']==venue)]
        common=[d for d in details if d['P3'] and all(d['models'][n]['ticket']['status'] in ('settled','refunded') for n in r.NAMES)]
        scope['finance_common']={n:r.finance([d['models'][n]['ticket'] for d in common]) for n in r.NAMES}
        market_scope=report.get('market_favorite',{}).get('scopes',{}).get(key)
        views[key]={'metrics':metric_svg(scope),'market':metric_svg(scope,True),'delta':delta_svg(scope),
            'distribution':distribution_svg(details),'money':money_svg(scope,market_scope)}
        if report.get('popularity'):
            pop=report['popularity']['scopes'][key]
            views[key]['popularity']={'all':popularity_svg(pop,r.NAMES),
                **{n:popularity_svg(pop,(n,)) for n in r.NAMES}}
    file_map={'metrics':'総合精度比較.svg','market':'市場との共通比較.svg','delta':'血統追加の差.svg',
              'distribution':'本命着順分布.svg','money':'本命単勝の仮想損益.svg'}
    for key,name in file_map.items():
        with (directory/name).open('x',encoding='utf-8') as stream:stream.write(views['all|all'][key])
    if report.get('popularity'):
        with (directory/'本命のT10参考人気分布.svg').open('x',encoding='utf-8') as stream:
            stream.write(views['all|all']['popularity']['all'])
    tables=[]
    for period,scope in report['periods'].items():
        for population in ('P1','P2'):
            for n,m in scope[population].items():
                if not m:continue
                tables.append({'対象日':'2日合計' if period=='all' else period,'評価集合':'精度評価（異常除外）' if population=='P1' else '市場との共通集合',
                    'モデル':r.LABELS[n],'内部モデル名':n,'モデルID':report['model_ids'].get(n),'レース数':m['races'],'頭数':m['runners'],
                    **{r.METRICS[k][0]:m[k] for k in FIELDS},'本命的中数':m['top1_hit_count']})
    csv_file(directory/'総合性能比較.csv',tables)
    delta_rows=[{'指標':r.METRICS[k][0],'良い方向':'高いほど良い' if r.METRICS[k][1] else '低いほど良い',
        '血統45列から39列対照を引いた差':v,'点推定の方向':change(k,v)} for k,v in report['pedigree_minus_control'].items()]
    csv_file(directory/'血統追加の差.csv',delta_rows)
    race_rows=[]
    for d in report['details']:
        for n,m in d['models'].items():
            t=m['ticket'];entry={'開催日':d['date'],'競馬場':d['venue'],'レース番号':d['race_no'],'レースID':d['race_id'],
                'モデル':r.LABELS[n],'内部モデル名':n,'モデルID':m['model_id'],'本命馬番':m['horse_number'],'本命馬名':m['horse_name'],
                '保存予測勝率':m['probability'],'公式着順':m['finish'],'異常区分':m['anomaly'],'勝馬の保存予測順位':m['winner_predicted_rank'],
                '精度評価対象':d['P1'],'市場比較対象':d['P2'],'仮想購入対象':d['P3'],'3モデルの本命不一致':d['top_disagreement'],
                '仮想精算状態':t['status'],'仮想投資額_円':100 if d['P3'] else 0,'仮想払戻額_円':t.get('payout_yen'),
                '予測生成日時':d['predicted_at'],'判断締切':d['input_cutoff'],'オッズ発表日時':d['odds_published_at'],
                'オッズ観測日時':d['odds_first_seen_at'],'除外理由':'・'.join(r.REASONS.get(x,x) for x in d['exclusion_reasons'])}
            if report.get('popularity'):
                rank=m['T10_popularity_rank']
                entry.update(T10参考人気順位=rank,T10参考人気='人気不明' if rank is None else f'{rank}番人気'+('（同率）' if m['T10_popularity_tied'] else ''),
                    T10単勝オッズ_倍=m['T10_win_odds'],T10人気帯=m['T10_popularity_band'])
            race_rows.append(entry)
    csv_file(directory/'レース別本命比較.local.csv',race_rows)
    finance_rows=[{'集計対象':population,'モデル':r.LABELS[n],'モデルID':report['model_ids'][n],'仮想購入数':m['bet_count'],
        '的中率_％':m['hit_rate_percent'],'投資額_円':m['investment_yen'],'払戻額_円':m['payout_yen'],'損益_円':m['net_profit_yen'],
        '回収率_％':m['return_rate_percent'],'的中数':m['hit_count'],'未精算数':m['unsettled_count'],'返還数':m['refunded_count']}
        for population,values in (('全仮想購入（未精算を含む）',report['finance_P3']),('精算済み共通レース',report['finance_common_settled'])) for n,m in values.items()]
    if report.get('market_favorite'):
        market=report['market_favorite']['scopes']['all|all']
        for population,m in (('全仮想購入（未精算を含む）',market['finance_P3']),('精算済み共通レース',market['finance_on_three_common']['market_favorite'])):
            finance_rows.append({'集計対象':population,'モデル':DISPLAY_LABELS['market_favorite'],'モデルID':None,
                '仮想購入数':m['bet_count'],'的中率_％':m['hit_rate_percent'],'投資額_円':m['investment_yen'],'払戻額_円':m['payout_yen'],
                '損益_円':m['net_profit_yen'],'回収率_％':m['return_rate_percent'],'的中数':m['hit_count'],
                '未精算数':m['unsettled_count'],'返還数':m['refunded_count']})
    csv_file(directory/'本命単勝100円の仮想収益.csv',finance_rows)
    if report.get('market_favorite'):
        market_rows=[]
        for d in report['details']:
            m=d['market_favorite'];t=m['ticket'];market_rows.append({'開催日':d['date'],'競馬場':d['venue'],
                'レース番号':d['race_no'],'レースID':d['race_id'],'市場本命馬番':m['horse_number'],'市場本命馬名':m['horse_name'],
                'T10参考人気':m['T10_popularity_rank'],'T10単勝オッズ_倍':m['T10_win_odds'],
                '最低オッズ同率':m['T10_popularity_tied'],'同率頭数':m['tied_lowest_count'],
                '同率候補馬番':'・'.join(m['tied_lowest_horse_numbers']),'同率選択規則':'馬番昇順',
                '公式着順':m['finish'],'異常区分':m['anomaly'],'仮想購入対象_P3':d['P3'],
                '投資額_円':100 if d['P3'] else 0,'払戻額_円':t.get('payout_yen'),
                '精算状態':{'settled':'精算済み','refunded':'返還','unsettled':'未精算','no_bet':'対象外'}.get(t['status'],t['status']),
                '精算理由コード':t.get('reason'),'オッズ発表日時':d['odds_published_at'],'オッズ観測日時':d['odds_first_seen_at']})
        csv_file(directory/'市場1番人気の仮想購入.local.csv',market_rows)
    bands=[]
    for n,m in report['periods']['all']['P1'].items():
        for b in (m or {}).get('calibration_bands',[]):bands.append({'モデル':r.LABELS[n],
            '確率帯の下限':b['lower_inclusive'],'確率帯の上限':b['upper'],'頭数':b['runners'],
            '平均予測勝率':b['mean_prediction'],'実際の勝率':b['observed_win_rate']})
    csv_file(directory/'確率帯別の較正.csv',bands)
    if report.get('popularity'):
        pop_rows=[];performance=[]
        for key,scope in report['popularity']['scopes'].items():
            day,venue=key.split('|')
            for n,m in scope['models'].items():
                common={'対象日':'2日合計' if day=='all' else day,'競馬場':'すべて' if venue=='all' else r.VENUES.get(venue,venue),
                    'モデル':r.LABELS[n],'内部モデル名':n,'モデルID':report['model_ids'][n],'母集団':'P3',
                    '共通対象レース数':scope['P3_races']}
                pop_rows.append({**common,'平均参考人気順位':m['mean_popularity_rank'],'対象外の人気不明数':m['unknown_outside_P3'],
                    **{b['label']+'件数':b['count'] for b in m['bands']},**{b['label']+'割合_％':b['share_percent'] for b in m['bands']}})
                for b in m['bands']:
                    t=b['finance'];part=b['settled_subset']
                    performance.append({**common,'T10参考人気帯':b['label'],'選択件数':b['count'],'選択割合_％':b['share_percent'],
                        '結果確認済み本命数':b['known_result_count'],'1着数':b['win_count'],'的中率_％':b['hit_rate_percent'],
                        '結果未確認または返還数':b['result_unconfirmed_or_refunded_count'],
                        '投資額_円':t['investment_yen'],'全件払戻額_円':t['payout_yen'],'全件回収率_％':t['return_rate_percent'],
                        '精算済み件数':part['bet_count'],'精算済み部分投資額_円':part['investment_yen'],
                        '精算済み部分払戻額_円':part['payout_yen'],'精算済み部分回収率_％':part['return_rate_percent'],
                        '未精算数':t['unsettled_count'],'注意':b['note']})
        csv_file(directory/'T10参考人気分布.csv',pop_rows)
        csv_file(directory/'T10人気帯別成績.csv',performance)
    lines=['# 血統45列は予想をどう変えたか：3モデル開催後再生の参考評価','',
        '対象：2026年10月24日・25日。2026年10月26日に作成・固定した3モデルによる開催後再生です。真正な発走前予測・独立テストではありません。',
        '正式な21週間評価には一切算入しません。新規学習0fit、保存確率・順位・温度の変更なし。H・Stage 7の結果・最終ホールドアウトは不使用。','',
        '## 対象数と集計の分母','',
        f'- 保存予測：{report["target_races"]}レース・{report["target_runners"]}頭。',
        f'- 精度評価（P1）：{report["P1_eligible_races"]}レース・{report["periods"]["all"]["P1_runners"]}頭。異常・不整合・未確認による除外{report["excluded_P1_races"]}レース。',
        f'- 市場との共通比較（P2）：{report["P2_eligible_races"]}レース。全3モデルと市場を同じ全出走馬集合で比較。',
        f'- 仮想収益（P3）：{report["P3_eligible_races"]}レース。異常レースも購入時点で結果不明のため含める。オッズ欠損・鮮度不合格{report["odds_missing_races"]}レースでは購入しない。','',
        '## 指標の意味','']
    for k,(title,higher,meaning) in r.METRICS.items():lines.append(f'- {title}：{meaning}。**{"高いほど良い" if higher else "低いほど良い"}**。')
    lines+=['','本命の同率時は保存順位（勝率降順、馬番昇順）を使用。ブライア得点は馬単位の平均二乗誤差。レースLLとレース内AUCはレースを等重みで平均。','',
        '## 3モデルの精度比較（異常レース除外後）','',
        '|モデル|レース数|レース内AUC ↑|全体AUC ↑|レースLL ↓|ブライア得点 ↓|本命1着率 ↑|的中数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,m in report['periods']['all']['P1'].items():
        if m:lines.append(f'|{r.LABELS[n]}|{m["races"]}|{number(m["within_race_auc"])}|{number(m["roc_auc"])}|{number(m["race_log_loss"])}|{number(m["brier_score"])}|{pct(m["top1_accuracy_percent"])}|{m["top1_hit_count"]}|')
    lines+=['','## 血統45列 − 39列対照','', '|指標|差|点推定の方向|','|---|---:|---|']
    for k,v in report['pedigree_minus_control'].items():lines.append(f'|{r.METRICS[k][0]}|{v:+.6f}{"ポイント" if k=="top1_accuracy_percent" else ""}|{change(k,v)}|')
    lines+=['','## 市場との比較（同じレース・同じ馬集合）','',
        '|モデル|レース数|レース内AUC ↑|全体AUC ↑|レースLL ↓|ブライア得点 ↓|本命1着率 ↑|','|---|---:|---:|---:|---:|---:|---:|']
    for n,m in report['periods']['all']['P2'].items():
        if m:lines.append(f'|{r.LABELS[n]}|{m["races"]}|{number(m["within_race_auc"])}|{number(m["roc_auc"])}|{number(m["race_log_loss"])}|{number(m["brier_score"])}|{pct(m["top1_accuracy_percent"])}|')
    lines+=['','市場確率＝単勝オッズの逆数を同じ全出走馬集合で正規化。T−10以前に公表・観測済み、分の終端が締切以前、鮮度300秒以内の保存オッズのみ。事後の最終オッズは未使用。','',
        '## 本命単勝100円の仮想収益（R1のみ）','',
        '|モデル|購入数|的中数|的中率|投資額|払戻額|損益|回収率|未精算|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,m in report['finance_common_settled'].items():lines.append(f'|{r.LABELS[n]}|{m["bet_count"]}|{m["hit_count"]}|{pct(m["hit_rate_percent"])}|{yen(m["investment_yen"])}|{yen(m["payout_yen"])}|{yen(m["net_profit_yen"])}|{pct(m["return_rate_percent"])}|{m["unsettled_count"]}|')
    if report.get('market_favorite'):
        market=report['market_favorite']['scopes']['all|all'];m=market['finance_on_three_common']['market_favorite']
        lines.append(f'|{DISPLAY_LABELS["market_favorite"]}|{m["bet_count"]}|{m["hit_count"]}|{pct(m["hit_rate_percent"])}|{yen(m["investment_yen"])}|{yen(m["payout_yen"])}|{yen(m["net_profit_yen"])}|{pct(m["return_rate_percent"])}|{m["unsettled_count"]}|')
    pending=report['finance_P3']['control39']
    lines+=['',f'上表は全3モデルで精算済みの共通レースのみ。全購入対象は各{pending["bet_count"]}件・{yen(pending["investment_yen"])}で、未精算{pending["unsettled_count"]}件を含む全件の払戻総額・損益・回収率は未確定。未精算分を0円としていない。',
        'オッズ条件を満たすレースだけで各モデルの本命を100円仮想購入。期待値フィルターなし。払戻額は確認済みの公式単勝100円払戻を使用し、T−10オッズからは算定しない。','',
        '## 本命が違ったレース','',
        f'3モデルの本命が一致しなかった有効レースは{report["top_disagreement_P1_races"]}レース。レース別の馬名・馬番・保存勝率・公式着順はローカルHTMLおよびレース別CSVを参照。','',
        '|比較|本命不一致|前者の本命1着|後者の本命1着|前者が先着|後者が先着|','|---|---:|---:|---:|---:|---:|']
    for pair,value in report['pair_disagreements'].items():
        a,b=pair.split('_vs_');lines.append(f'|{r.LABELS[a]} 対 {r.LABELS[b]}|{value["different_races"]}|{value["first_top_win_count"]}|{value["second_top_win_count"]}|{value["first_finishes_ahead"]}|{value["second_finishes_ahead"]}|')
    lines+=['','## 本命馬の着順分布（精度評価と同じレース）','',
        '|モデル|1着|2着|3着|3着以内|4着以下|','|---|---:|---:|---:|---:|---:|']
    for n,dist in report['top_finish_distribution'].items():
        counts=[dist.get(str(k),0) for k in (1,2,3)];total=sum(dist.values())
        lines.append(f'|{r.LABELS[n]}|{counts[0]}|{counts[1]}|{counts[2]}|{sum(counts)}/{total}（{pct(100*sum(counts)/total if total else None)}）|{total-sum(counts)}|')
    lines+=['','## 血統追加の良かった点・悪かった点','']+['- '+s for s in assessment(report)]
    lines+=['','## 除外・留保','']
    if report['exclusion_reasons']:
        lines += [f'- {r.REASONS.get(k,k)}：{v}レース（理由は重複あり）。' for k,v in report['exclusion_reasons'].items()]
    else:lines.append('- 異常・結果不整合による除外なし。')
    lines += ['- '+s for s in report.get('caveats',[])]
    if report.get('result_source_note'):lines.append('- '+report['result_source_note'])
    lines += ['','既存モデル・保存予測・事前登録のハッシュ一致と、既存タスク設定の一致を確認。実馬券購入・本番KEIBA LAB接続・自動タスク変更はありません。','',
        '## モデルと保存情報','', '|表示名|内部識別子|モデルID|温度|','|---|---|---|---:|']
    for n in r.NAMES:lines.append(f'|{r.LABELS[n]}|{n}|{report["model_ids"][n]}|{report["temperatures"][n]}|')
    if report.get('popularity'):
        scope=report['popularity']['scopes']['all|all']
        lines+=['','## T−10参考人気（追加機能）','',
            f'人気分布と人気帯別収益の母集団はP3の共通{scope["P3_races"]}レース。P2の市場精度比較とは別。対象外{scope["outside_P3_races"]}レースは人気不明と表示する。',
            '保存された完全で新鮮なT−10単勝オッズを低い順に並べ、同一オッズは同順位・次順位を飛ばす（1・1・3位）。公式最終人気ではない。','',
            '|モデル|平均参考人気|1番人気|2〜3番人気|4〜6番人気|7番人気以下|','|---|---:|---:|---:|---:|---:|']
        for n,m in scope['models'].items():
            cells='|'.join(f'{b["count"]}件（{pct(b["share_percent"])}）' for b in m['bands'])
            lines.append(f'|{r.LABELS[n]}|{number(m["mean_popularity_rank"],2)}位|{cells}|')
        lines+=['','人気帯別の的中率は、結果確認済みの本命が分母（取消・返還・結果未確認を除く）。',
            '人気帯別の回収率は、P3内の精算済み部分を投資額とともに参考表示。未精算を含む帯の全件回収率は未確定として別列へ保存。10件未満は少数例と明記。',
            'レース別表には本命ごとの参考人気・単勝オッズを追加し、人気帯・順位範囲の絞り込みと人気順の並べ替えに対応。','']
    if report.get('market_favorite'):
        m=report['market_favorite']['scopes']['all|all'];cash=m['finance_P3']
        lines+=['','## 市場1番人気の単勝100円（追加基準）','',
            '保存されたT−10単勝オッズが最小の馬を1頭選ぶ。最低オッズが同率なら馬番昇順で選択し、同率候補も記録。結果・最終オッズは選択に使わない。',
            f'P3対象は{m["P3_races"]}レース。最低オッズ同率は{m["market_tied_lowest_races"]}レース。',
            f'上の収益比較表は、以前からの3モデル共通{m["old_three_common_races"]}レースを維持。市場側だけ未精算のケースは{m["market_only_unsettled_on_three_common"]}件。',
            f'市場のP3全件投資は{yen(cash["investment_yen"])}、未精算{cash["unsettled_count"]}件、返還{cash["refunded_count"]}件。未精算が残れば全件回収率は確定しない。',
            '市場本命の馬名・馬番・参考人気・判断オッズ・着順・払戻は、レース別表と「市場1番人気の仮想購入.local.csv」を参照。市場は学習モデルではない。','']
    lines+=['',f'生成日時：{report["created_at"]}',f'研究ID：`{report["artifact_id"]}`','',
        'HTMLは「比較レポート.html」をダブルクリック。ネット接続・サーバー・Python起動は不要。日付・競馬場の絞り込みは表とグラフ、モデル選択は表へ反映。','']
    with (directory/'比較レポート.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    payload={'report':report,'views':views,'labels':DISPLAY_LABELS,'metrics':r.METRICS,'assessment':assessment(report),
             'reasons':r.REASONS,'colors':COLORS}
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=Path(__file__).with_name('pedigree_replay_review.html').read_text(encoding='utf-8')
    with (directory/'比較レポート.html').open('x',encoding='utf-8') as stream:
        stream.write(template.replace('__REVIEW_DATA__',data).replace('__RESULT_SOURCE_NOTE__',escape(report.get('result_source_note','架空データによる表示確認'))))
