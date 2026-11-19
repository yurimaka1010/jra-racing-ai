"""Independent, zero-fit exploration of five predeclared market conditions.

Only saved 2023--2025 development OOF, T10/T30 packets and official HR are
read. No model loader, DB connection, scheduler or live-recorder dependency.
"""
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

from . import stage1_analysis
from .ml import market
from .win_sim import core

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
S2 = BASE / 'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S3 = BASE / 'stage3-development/evaluations/d70f15d205a14cf48e4f2bff3827f8cb'
REFS = BASE / 'stage4-development/evaluations/2f26b6573d2249d38ad96b4456f89c0f/references.json'
ROOT = BASE / 'market-advantage/evaluations'
YEARS = (2023, 2024, 2025)
NAMES = {
    'ALL': '全対象・固定EVルール（基準）',
    'C1': 'AI本命と市場本命が一致',
    'C2': 'AI本命と市場本命が不一致',
    'C3': 'EV最大候補が4〜6番人気',
    'C4': 'EV最大候補の過去出走3走以上',
    'C5': 'EV最大候補の市場支持が上昇',
}
BANDS = ('1番人気', '2〜3番人気', '4〜6番人気', '7番人気以下')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def in_scope(day):
    d = date.fromisoformat(day)
    return d.year in YEARS and d.month <= 9


def race_key(text):
    if not isinstance(text, str) or not re.fullmatch(r'\d{18}', text, re.ASCII):
        raise ValueError('invalid_entry_key')
    return core.key((text[:4], text[4:8], text[8:10], text[10:12], text[12:14], text[14:16]))


def rank_band(rank):
    return BANDS[0] if rank == 1 else BANDS[1] if rank <= 3 else BANDS[2] if rank <= 6 else BANDS[3]


def checked_quote(race, packet, roster, schedule, minutes):
    """Reuse the existing conservative minute/same-roster validator, no fallback."""
    value = packet.get('quotes', {}).get(str(minutes))
    if not value:
        return None, packet.get('failures', {}).get(str(minutes)) or 'odds_unavailable'
    if (tuple(value.get('race_key', ())) != race or value.get('category') != '1'
            or value.get('basis') != 'pre_race_publication_time'
            or value.get('source') != 'apd_sokuho_o1'):
        return None, 'odds_unverified'
    if packet.get('scheduled_start') != schedule.scheduled_start.isoformat():
        return None, 'schedule_conflict'
    try:
        cutoff = schedule.scheduled_start - timedelta(minutes=minutes)
        snap = core.Snapshot(race, core.instant(value['published_at']), None,
                             value['odds'], value['source'], True, publication_precision='minute')
        selected, reason = core.choose_snapshot(race, [snap], cutoff, roster,
                                               timing_basis='jra_publication_time')
        if reason:
            return None, reason
        if (cutoff - selected.published_at).total_seconds() > 300:
            return None, 'odds_stale'
        return selected, None
    except (KeyError, TypeError, core.SimulationError):
        return None, 'odds_invalid'


def decide(race, probabilities, features, packet):
    """Pure selection: no labels, finish order or payout arguments are accepted."""
    probs, _ = core.probabilities(probabilities, core.WinPolicy(probability_method='identity'))
    if set(probs) != set(features):
        return None, 'feature_roster_mismatch'
    clocks = {(row.get('start_time'), row.get('previous_start')) for row in features.values()}
    if len(clocks) != 1:
        return None, 'schedule_conflict'
    clock, previous = next(iter(clocks))
    schedule, reason = stage1_analysis.reference_schedule(race, [dict(
        record_type='RA', status='7', start=clock, previous_start=previous, oversized=False)], tuple(probs))
    if reason:
        return None, reason
    if packet is None:
        return None, 'odds_unavailable'
    snap, reason = checked_quote(race, packet, tuple(probs), schedule, 10)
    if reason:
        return None, reason
    pm, _ = market.implied_probabilities(snap.odds)
    ev = {h: probs[h] * snap.odds[h] - 1 for h in probs}
    candidate = min(ev, key=lambda h: (-ev[h], h))
    favorite = min(snap.odds, key=lambda h: (snap.odds[h], h))
    ai_top = min(probs, key=lambda h: (-probs[h], h))
    ranks = {h: 1 + sum(o < snap.odds[h] for o in snap.odds.values()) for h in probs}
    early, early_reason = checked_quote(race, packet, tuple(probs), schedule, 30)
    change = None
    if early:
        pe, _ = market.implied_probabilities(early.odds)
        change = math.log(float(pm[candidate])) - math.log(float(pe[candidate]))
    starts = features[candidate].get('past_starts')
    try:
        starts = float(starts)
        starts = starts if math.isfinite(starts) and starts >= 0 else None
    except (ValueError, TypeError):
        starts = None
    conditions = {'C1': ai_top == favorite, 'C2': ai_top != favorite,
                  'C3': 4 <= ranks[candidate] <= 6,
                  'C4': starts >= 3 if starts is not None else None,
                  'C5': change > 0 if change is not None else None}
    return dict(race_key=list(race), year=race[0], month=race[0]+'-'+race[1][:2],
                scheduled_start=schedule.scheduled_start.isoformat(),
                decision_at=(schedule.scheduled_start-timedelta(minutes=10)).isoformat(),
                odds_published_at=snap.published_at.isoformat(),
                age_seconds=(schedule.scheduled_start-timedelta(minutes=10)-snap.published_at).total_seconds(),
                early_published_at=early.published_at.isoformat() if early else None,
                early_reason=early_reason, history_log_change=change,
                ai_top=ai_top, favorite=favorite, candidate=candidate,
                candidate_ev=float(ev[candidate]),
                selected=candidate if ev[candidate] >= core.decimal('0.05') else None,
                candidate_rank=ranks[candidate], candidate_past_starts=starts, conditions=conditions,
                runners=[dict(horse_number=h, probability=float(probs[h]), market_probability=float(pm[h]),
                              decision_odds=float(snap.odds[h]), reference_popularity=ranks[h]) for h in sorted(probs)]), None


class Inputs:
    """Only exact development source files; no discovery under protected outputs."""
    def __init__(self):
        self.hashes = {}
        self.references = self.json(REFS)
        self.metadata = []

    def track(self, path, expected=None):
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('saved_source_hash_mismatch')
        self.hashes[str(path)] = sha
        return path

    def json(self, path, expected=None):
        return read(self.track(path, expected))

    def saved(self, path):
        expected = self.references.get(str(path))
        if expected is None:
            raise ValueError('source_not_in_saved_references')
        return self.track(path, expected)

    def pre_race(self):
        predictions, features, packets, availability = {}, {}, {}, {}
        for year in YEARS:
            meta_path = S3 / 'oof' / f'{year}.complete.json'
            meta = self.json(meta_path)
            summary = meta['summary']
            for field in ('train_end', 'validation_end', 'calibration_end'):
                if summary['split'][field] >= f'{year}-01-01':
                    raise ValueError('non_chronological_OOF')
            self.metadata.append({k: summary[k] for k in ('model_id', 'split', 'temperature', 'rows', 'races')})
            path = S3 / 'oof' / f'{year}.csv'
            expected = meta['files'].get(str(path), meta['files'].get(path.name))
            if expected is None:
                raise ValueError('missing_OOF_manifest_hash')
            with self.track(path, expected).open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['date'][:4] != str(year):
                        raise ValueError('invalid_OOF_year')
                    if not in_scope(row['date']):
                        continue
                    key = row['entry_key']; race = race_key(key)
                    if key[:8] != row['date'].replace('-', ''):
                        raise ValueError('date_identity_mismatch')
                    cutoff = (date.fromisoformat(row['date'])-timedelta(days=7)).isoformat()
                    if row['feature_history_cutoff'] != cutoff or row['source_model_id'] != summary['model_id']:
                        raise ValueError('OOF_provenance_mismatch')
                    target = predictions.setdefault(race, {})
                    if key[-2:] in target:
                        raise ValueError('duplicate_prediction')
                    target[key[-2:]] = float(row['ability_probability'])
            path = self.saved(S2 / 'features' / f'{year}.csv')
            with path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['date'][:4] != str(year):
                        raise ValueError('invalid_feature_year')
                    if not in_scope(row['date']) or row['eligible'] != '1':
                        continue
                    race = race_key(row['entry_key'])
                    target = features.setdefault(race, {})
                    h = row['entry_key'][-2:]
                    if h in target:
                        raise ValueError('duplicate_feature')
                    # Deliberately omit the saved label until decisions are committed.
                    target[h] = {k: row[k] for k in ('start_time', 'previous_start', 'past_starts')}
            paths = sorted((S2 / 'market').glob(f'{year}-*.local.json'))
            for path in paths:
                self.saved(path)
                for row in read(path)['records']:
                    race = core.key(row['race_key'])
                    day = race[0]+'-'+race[1][:2]+'-'+race[1][2:]
                    if race[0] != str(year):
                        raise ValueError('invalid_market_year')
                    if not in_scope(day):
                        continue
                    if race in packets:
                        raise ValueError('duplicate_packet')
                    packets[race] = {k: row[k] for k in ('scheduled_start', 'quotes', 'failures')}
            selected = [r for r in predictions if r[0] == str(year)]
            availability[str(year)] = dict(OOF_races=len(selected), OOF_runners=sum(len(predictions[r]) for r in selected),
                                          cache_files=len(paths), cache_races=sum(r[0] == str(year) for r in packets))
        if set(predictions) != set(features):
            raise ValueError('feature_cohort_mismatch')
        return predictions, features, packets, availability

    def answers(self, predictions):
        labels, outcomes, reasons = {}, {}, Counter()
        for year in YEARS:
            with (S2 / 'features' / f'{year}.csv').open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if not in_scope(row['date']) or row['eligible'] != '1':
                        continue
                    race = race_key(row['entry_key'])
                    if race in predictions:
                        value = float(row['winner_label'])
                        if value not in (0., 1.):
                            raise ValueError('invalid_binary_label')
                        labels.setdefault(race, {})[row['entry_key'][-2:]] = int(value)
            for path in sorted((S2 / 'market').glob(f'{year}-*.local.json')):
                for row in read(path)['records']:
                    race = tuple(row['race_key'])
                    if race in predictions:
                        item = row.get('outcome')
                        if item:
                            outcomes[race] = core.Outcome(race, item['status'], item['payouts_yen_per_100'],
                                                          item['verified'], item['complete_winner_set'], item['reason'])
        winners = {}
        for race, p in predictions.items():
            y = labels.get(race, {})
            if set(y) == set(p) and set(y.values()) <= {0, 1} and sum(y.values()) == 1:
                winners[race] = next(h for h in y if y[h] == 1)
            else:
                reasons['label_incomplete_or_not_single_winner'] += 1
            outcome = outcomes.get(race)
            if (outcome and outcome.status == 'normal_final' and race in winners
                    and set(outcome.payouts_yen_per_100) != {winners[race]}):
                outcomes[race] = core.Outcome(race, 'pending', reason='result_unconfirmed')
                reasons['HR_label_disagreement'] += 1
        return winners, outcomes, dict(reasons)

    def verify_unchanged(self):
        return all(digest(Path(path)) == expected for path, expected in self.hashes.items())


def answer(rows, winners, outcomes):
    result = []
    for row in rows:
        race = tuple(row['race_key']); roster = tuple(r['horse_number'] for r in row['runners'])
        outcome = outcomes.get(race)
        result.append(dict(row, winner=winners.get(race),
                           ai=core.settle(row['selected'], outcome, roster),
                           # Matched purchase races, identical investment to AI.
                           favorite_bet=core.settle(row['favorite'] if row['selected'] else None, outcome, roster),
                           settlement_reason=core.outcome_reason(outcome, roster)))
    return result


def mean(values):
    return statistics.fmean(values) if values else None


def binary_loss(p, y):
    p = max(1e-15, min(1-1e-15, p))
    return -y*math.log(p)-(1-y)*math.log1p(-p)


def describe(rows):
    available = [r for r in rows if r['winner'] is not None]
    winners = {tuple(r['race_key']): r['winner'] for r in available}
    scores = {}
    for field, name in (('probability', 'AI'), ('market_probability', 'market')):
        pred = [core.PredictionRace(tuple(r['race_key']), {h['horse_number']: h[field] for h in r['runners']}) for r in available]
        scores[name] = stage1_analysis.measure(pred, winners)
    bought = [r for r in rows if r['selected']]
    selected = [(r, next(h for h in r['runners'] if h['horse_number'] == r['selected'])) for r in bought]
    known = [(r, h) for r, h in selected if r['winner'] is not None]
    band_counts = Counter(rank_band(h['reference_popularity']) for _, h in selected)
    ai, favorite = core.financial(rows, 'ai'), core.financial(rows, 'favorite_bet')
    settled = [r for r in rows if r['ai']['status'] == 'settled']
    largest = max((r['ai']['payout_yen'] for r in settled), default=0)
    return dict(target_races=len(rows), runners=sum(len(r['runners']) for r in rows),
                label_races=len(available), label_missing_races=len(rows)-len(available),
                financial=ai, favorite_same_purchases=favorite, scores=scores,
                popularity_bands={b: {'count': band_counts[b], 'percent': 100*band_counts[b]/len(bought) if bought else None} for b in BANDS},
                mean_selected_rank=mean([h['reference_popularity'] for _, h in selected]),
                mean_AI_probability=mean([h['probability'] for _, h in selected]),
                mean_market_probability=mean([h['market_probability'] for _, h in selected]),
                mean_probability_gap=mean([h['probability']-h['market_probability'] for _, h in selected]),
                mean_probability_ratio=mean([h['probability']/h['market_probability'] for _, h in selected]),
                selected_binary_LL_AI=mean([binary_loss(h['probability'], int(r['selected'] == r['winner'])) for r, h in known]),
                selected_binary_LL_market=mean([binary_loss(h['market_probability'], int(r['selected'] == r['winner'])) for r, h in known]),
                mean_selected_EV=mean([r['candidate_ev'] for r in bought]),
                largest_payout_yen=largest,
                largest_payout_fraction=largest/ai['settled_subset']['payout_yen'] if ai['settled_subset']['payout_yen'] else None,
                fewer_than_100_bets=len(bought) < 100,
                profit_without_largest_payout_yen=ai['net_profit_yen']-largest if ai['net_profit_yen'] is not None else None)


def aggregate(rows):
    output = {}
    for name in NAMES:
        selected = rows if name == 'ALL' else [r for r in rows if r['conditions'][name] is True]
        result = describe(selected)
        result['condition'] = name; result['name'] = NAMES[name]
        result['unassessable_races'] = 0 if name == 'ALL' else sum(r['conditions'][name] is None for r in rows)
        result['by_year'] = {str(y): describe([r for r in selected if r['year'] == str(y)]) for y in YEARS}
        result['by_month'] = {f'{y}-{m:02d}': {
            'AI': core.financial([r for r in selected if r['month'] == f'{y}-{m:02d}'], 'ai'),
            'market_favorite': core.financial([r for r in selected if r['month'] == f'{y}-{m:02d}'], 'favorite_bet')}
            for y in YEARS for m in range(1, 10)}
        output[name] = result
    # Missing early prices are an availability subgroup, not a sixth condition.
    output['C5']['history_available_reference'] = describe([
        r for r in rows if r['conditions']['C5'] is not None])
    return output


def csv_output(path, fields, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def render(directory, report):
    fields = ['条件番号', '条件', '対象レース数', '対象頭数', '条件判定不能数', '期間', '購入件数', '的中数',
              '投資額_円', '払戻額_円', '損益_円', '回収率_パーセント', '未精算件数', '精算済みのみ回収率_パーセント',
              '市場本命_同一購入レース損益_円', '市場本命_同一購入レース回収率_パーセント',
              'AI二値LL_全馬', '市場二値LL_全馬', '二値LL差_AI引く市場', 'AIレースLL', '市場レースLL',
              '購入馬_AI勝率平均', '購入馬_市場勝率平均', '購入馬_勝率差平均', '購入馬_参考人気平均',
              '購入馬_期待利益率平均', '最大払戻依存割合', '最大払戻を除いた損益_円']
    comparison, bands, monthly = [], [], []
    for name, result in report['conditions'].items():
        for period, v in [('全期間', result), *result['by_year'].items()]:
            f = v['financial']; m = v['favorite_same_purchases']; a = v['scores']['AI']; b = v['scores']['market']
            comparison.append(dict(zip(fields, [name, NAMES[name], v['target_races'], v['runners'], result['unassessable_races'] if period == '全期間' else None,
                period, f['bet_count'], f['hit_count'], f['investment_yen'], f['payout_yen'], f['net_profit_yen'], f['return_rate_percent'],
                f['unsettled_count'], f['settled_subset']['return_rate_percent'], m['net_profit_yen'], m['return_rate_percent'],
                a.get('log_loss'), b.get('log_loss'), a['log_loss']-b['log_loss'] if a.get('log_loss') is not None else None,
                a.get('race_log_loss'), b.get('race_log_loss'), v['mean_AI_probability'], v['mean_market_probability'], v['mean_probability_gap'],
                v['mean_selected_rank'], v['mean_selected_EV'], v['largest_payout_fraction'], v['profit_without_largest_payout_yen']])))
        for band, v in result['popularity_bands'].items():
            bands.append({'条件番号': name, '条件': NAMES[name], '購入馬人気帯': band, '購入件数': v['count'], '構成比_パーセント': v['percent']})
        for month, v in result['by_month'].items():
            monthly.append({'条件番号': name, '条件': NAMES[name], '年月': month, '購入件数': v['AI']['bet_count'],
                            '損益_円': v['AI']['net_profit_yen'], '回収率_パーセント': v['AI']['return_rate_percent'],
                            '未精算件数': v['AI']['unsettled_count'], '市場本命_同一購入レース損益_円': v['market_favorite']['net_profit_yen']})
    csv_output(directory/'条件比較.csv', fields, comparison)
    csv_output(directory/'人気帯.csv', list(bands[0]), bands)
    csv_output(directory/'月次損益.csv', list(monthly[0]), monthly)
    def fmt(v, precision=2):
        return '未算出' if v is None else f'{v:,.{precision}f}'
    lines = ['# 市場優位条件の探索研究・第1段階', '',
        '**追加学習0fit。2023〜2025年各1〜9月の既参照開発期間による探索であり、独立検証でも前向き成績でもありません。**', '',
        '## 使用データと固定条件', '',
        'Stage 3のABCD能力モデル時系列OOFを再利用しました。各年の前年までの学習・早期終了・較正を確認しています。'
        '新しい39・45・47列モデルを過去年へ適用したものではありません。保存済み確率に温度を再適用していません。', '',
        '単勝100円・期待利益率5%以上・1レース最大1点。全馬からEV最大候補を選んでから条件で絞り、条件内で別馬を選び直しません。'
        '比較の市場本命も、AIが購入した同じレースに100円ずつ購入します。これは市場本命を全レース買う戦略の成績とは異なります。', '',
        '中間O1のT−10公表オッズのみ。分終端が判断時刻以前、鮮度300秒以内、同一馬番集合を確認。'
        '最終オッズによる代替なし。発走時刻変更は既存規則で除外。PC実取得時刻と当時の発走予定時刻の利用可能性は未検証のため、発表時刻基準の事後評価です。', '',
        '人気はT−10オッズの低い順（同率同順位、次順位を飛ばす）。条件は重複し、合計してはいけません。'
        'C4の履歴欠損・C5の早期オッズ欠損は当該条件のみ判定不能で、基準母集団を変えません。', '',
        '**「本命一致／不一致」はAIと市場の予測1位同士の比較です。仮想購入はEV最大馬なので、本命を買うとは限りません。**', '',
        '## データ被覆率', '', '| 年 | OOFレース | 頭数 | T−10利用可 | 被覆率 | T−30も利用可 | 通常公式払戻確認 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for year, v in report['availability'].items():
        lines.append(f"| {year} | {v['OOF_races']:,} | {v['OOF_runners']:,} | {v['usable_races']:,} | {fmt(v['coverage_percent'])}% | {v['early_usable_races']:,} | {v['official_settleable_races']:,} |")
    lines += ['', '除外理由（T−10オッズ・時刻・集合）：`'+json.dumps(report['exclusion_reasons'], ensure_ascii=False)+'`', '',
              'schedule_changed_asof_unavailable＝発走予定時刻の変更があり、判断当時の時刻を再現できないため除外。オッズ自体の欠落とは区別します。', '',
              '正解・払戻留保：`'+json.dumps(report['answer_reasons'], ensure_ascii=False)+'`', '',
              '## 条件別の仮想収益', '',
              '| 条件 | 対象レース | 購入 | 的中 | 投資額（円） | 払戻額（円） | 損益（円） | 回収率 | 市場本命回収率・同一購入レース | 未精算 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in report['conditions'].values():
        f=v['financial']; m=v['favorite_same_purchases']
        lines.append(f"| {v['name']} | {v['target_races']:,} | {f['bet_count']:,} | {f['hit_count']} | {f['investment_yen']:,} | {fmt(f['payout_yen'],0)} | {fmt(f['net_profit_yen'],0)} | {fmt(f['return_rate_percent'])}% | {fmt(m['return_rate_percent'])}% | {f['unsettled_count']} |")
    lines += ['', '払戻は確認済みHRの100円あたり円額を使用。T−10表示オッズによる払戻保証はありません。'
              '未精算がある全体収益は未算出とし、CSVには精算済み部分の回収率も別欄で保存しています。購入0件の回収率も未算出です。', '',
              '## 確率品質と市場との差', '',
              '二値LL＝馬単位の勝敗確率の対数損失（低いほど良い）。各条件に合うレースの**全出走馬**でAIと市場を同じ集合で比較し、部分集合で再正規化しません。'
              '購入馬のみの二値LLは集計JSONに補足保存。勝率差は購入馬の「AI−市場」、単位はポイントです。', '',
              '| 条件 | AI二値LL | 市場二値LL | 差（負ならAI良好） | 購入馬の平均参考人気 | 購入馬の平均勝率差 | 条件判定不能レース |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for v in report['conditions'].values():
        a=v['scores']['AI'].get('log_loss'); b=v['scores']['market'].get('log_loss')
        lines.append(f"| {v['name']} | {fmt(a,6)} | {fmt(b,6)} | {fmt(a-b if a is not None else None,6)} | {fmt(v['mean_selected_rank'])} | {fmt(100*v['mean_probability_gap'] if v['mean_probability_gap'] is not None else None)} | {v['unassessable_races']} |")
    lines += ['', '## 購入馬のT−10参考人気帯', '',
              '| 条件 | 1番人気 | 2〜3番人気 | 4〜6番人気 | 7番人気以下 |',
              '|---|---:|---:|---:|---:|']
    for v in report['conditions'].values():
        parts=[f"{v['popularity_bands'][band]['count']:,}件（{fmt(v['popularity_bands'][band]['percent'])}%）" for band in BANDS]
        lines.append('| '+v['name']+' | '+' | '.join(parts)+' |')
    lines += ['', '## 年別損益と次の検証候補', '', '| 条件 | 2023年損益（円） | 2024年損益（円） | 2025年損益（円） | 最大払戻依存率 |', '|---|---:|---:|---:|---:|']
    for v in report['conditions'].values():
        annual=[v['by_year'][str(y)]['financial']['net_profit_yen'] for y in YEARS]
        lines.append('| '+v['name']+' | '+' | '.join(fmt(x,0) for x in annual)+' | '+fmt(100*v['largest_payout_fraction'] if v['largest_payout_fraction'] is not None else None)+'% |')
    positives=[v for k,v in report['conditions'].items() if k != 'ALL' and v['financial']['net_profit_yen'] is not None and v['financial']['net_profit_yen'] > 0]
    if positives:
        lines += ['', '探索上の黒字条件：'+ '、'.join(v['name'] for v in positives)+'。'
                  '年別の方向、購入件数、最大払戻依存を含めて別期間で再検証すべき候補であり、市場優位の認定ではありません。']
    else:
        lines += ['', '**今回の5条件から、黒字の市場優位条件は確認できませんでした。**'
                  '追加の閾値探索は行いません。次に検証するなら、収益が相対的に良い条件について、十分な購入件数と年別の再現性を確かめる独立の計画が必要です。']
    c5=report['conditions']['C5']; history=c5['history_available_reference']
    lines += ['', '次の検証を1つに絞るなら、**C5「EV最大候補の市場支持が上昇」**を、候補として記録します。'
              f"今回の回収率{fmt(c5['financial']['return_rate_percent'])}%、損益{fmt(c5['financial']['net_profit_yen'],0)}円で、黒字条件ではありません。"
              '市場本命との同一購入レース比較、年別結果、人気帯構成を併せて判断する必要があります。採用・閾値変更・自動運用は行いません。', '',
              f"C5を判定できるT−30併存母集団は{history['target_races']:,}レース。ここへ条件を掛けない固定EVルールの購入は"
              f"{history['financial']['bet_count']:,}件、回収率{fmt(history['financial']['return_rate_percent'])}%、"
              f"損益{fmt(history['financial']['net_profit_yen'],0)}円です。早期オッズ欠損のあるレースとの構成差にも留意してください。", '',
              'C5の市場支持上昇は、単に単勝オッズが下がったかではなく、レース内で正規化した市場確率の対数変化が正かで判定します。'
              '市場側の支持も得られた条件という仮説を次回確認する余地はありますが、今回の差を因果効果とは解釈しません。']
    lines += ['', '5条件を同じ開発期間で探索したため、最良値には選択バイアスがあります。'
              '単一の良い年・高額配当・市場本命との差だけでは採用しません。今回の固定条件はplan.json / plan.md、人気帯と月次内訳はCSVで確認できます。', '',
              '## 保持・実行記録', '',
              f"- 研究ID：`{directory.name}`。計画SHA-256：`{report['plan_sha256']}`。",
              f"- 保存済み入力{len(report['source_hashes'])}ファイルの開始・終了ハッシュ一致：{report['sources_unchanged']}。",
              '- DB接続0、追加学習0fit、2026年データ読込0。凍結モデル・正式21週間評価・H/Stage 7・タスク・KEIBA LABの変更なし。',
              '- 予測選択をselections.local.jsonへ固定保存した後、正解ラベル・公式払戻を集計に対応付けた。',
              '- ローカル詳細台帳はrecords.local.json。個別データを外部送信しない。', '']
    with (directory/'研究レポート.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def run(run_id):
    if not re.fullmatch('[a-f0-9]{32}', run_id):
        raise ValueError('invalid_run_id')
    directory = ROOT / run_id
    plan = read(directory/'plan.json'); sha = digest(directory/'plan.json')
    if (directory/'plan.sha256').read_text(encoding='ascii').strip() != sha:
        raise ValueError('plan_hash_mismatch')
    if (plan.get('version') != 'market_advantage_stage1_v1' or plan.get('fits') != 0
            or plan.get('years') != list(YEARS) or plan.get('months_each_year') != list(range(1,10))
            or [v['id'] for v in plan['conditions']] != ['C1','C2','C3','C4','C5']):
        raise ValueError('plan_mismatch')
    if (directory/'selections.local.json').exists():
        raise ValueError('run_already_started_no_overwrite')
    inputs = Inputs()
    predictions, features, packets, availability = inputs.pre_race()
    choices, exclusions = [], Counter()
    for race, p in sorted(predictions.items()):
        choice, reason = decide(race, p, features[race], packets.get(race))
        if reason:
            exclusions[(race[0], reason)] += 1
        else:
            choices.append(choice)
    write(directory/'selections.local.json', choices)
    write(directory/'selection_commit.json', dict(at=datetime.now(core.JST).isoformat(), plan_sha256=sha,
          selections_sha256=digest(directory/'selections.local.json'), outcomes_not_used_for_selection=True))
    winners, outcomes, answer_reasons = inputs.answers(predictions)
    rows = answer(choices, winners, outcomes)
    for year, value in availability.items():
        chosen = [r for r in rows if r['year'] == year]
        value.update(usable_races=len(chosen), coverage_percent=100*len(chosen)/value['OOF_races'],
                     early_usable_races=sum(r['early_published_at'] is not None for r in chosen),
                     official_settleable_races=sum(r['settlement_reason'] is None for r in chosen),
                     exclusions={why: count for (y,why),count in sorted(exclusions.items()) if y == year})
    report = dict(version='market_advantage_stage1_v1', run_id=run_id, created_at=datetime.now(core.JST).isoformat(),
                  plan_sha256=sha, fits=0, DB_connections=0, labels_2026_read=0, exploratory_only=True,
                  predictor='ABCD annual chronological OOF, saved T=0.75', model_metadata=inputs.metadata,
                  availability=availability, exclusion_reasons=dict(Counter({reason: sum(n for (_,r),n in exclusions.items() if r == reason) for _,reason in exclusions})),
                  answer_reasons=answer_reasons, conditions=aggregate(rows), source_hashes=inputs.hashes,
                  sources_unchanged=inputs.verify_unchanged())
    if not report['sources_unchanged']:
        raise ValueError('source_changed_during_analysis')
    write(directory/'records.local.json', rows)
    write(directory/'集計結果.json', report)
    render(directory, report)
    return dict(status='ok', kind='market_advantage_stage1', run_id=run_id, fits=0,
                target_races=sum(v['OOF_races'] for v in availability.values()), odds_usable_races=len(rows),
                report_path=str(directory/'研究レポート.md'), sources_unchanged=True,
                results=[dict(condition=k, name=v['name'], target_races=v['target_races'],
                              financial=v['financial'], binary_LL_AI=v['scores']['AI'].get('log_loss'),
                              binary_LL_market=v['scores']['market'].get('log_loss')) for k,v in report['conditions'].items()])


def publish(run_id):
    """Add readable presentation; keep the initial selections/results untouched."""
    if not re.fullmatch('[a-f0-9]{32}', run_id):
        raise ValueError('invalid_run_id')
    original = ROOT/run_id
    report = read(original/'集計結果.json')
    rows = read(original/'records.local.json')
    commit = read(original/'selection_commit.json')
    if digest(original/'selections.local.json') != commit['selections_sha256']:
        raise ValueError('selection_commit_mismatch')
    report['conditions']['C5']['history_available_reference'] = describe([
        r for r in rows if r['conditions']['C5'] is not None])
    target = original/'日本語報告'
    target.mkdir(exist_ok=False)
    render(target, report)
    write(target/'集計結果.json', report)
    write(original/'publication.json', dict(path=str(target), plan_sha256=report['plan_sha256'],
          original_report_sha256=digest(original/'集計結果.json'),
          selections_unchanged=True, new_conditions=0, new_fits=0))
    return dict(status='published', path=str(target/'研究レポート.md'), fits=0)
