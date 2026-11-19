"""Local Stage 1 measurements; no fitting of the frozen LightGBM model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import csv
import gc
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from .core import RACE_KEY, SafeAuditError
from .db_connection import read_only_session
from .inspect_schema import COLUMNS_SQL
from .ml import data, features, market, model, safety, v03
from .win_sim import core, formats, local, publication, settle_saved
from .win_sim_cli import connection_config
from . import stage1_analysis as analysis

ROOT = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\stage1-market')
MODEL_ID = 'a8178c2b7523499f9463c1743b41c715'
PRIOR = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\post-step2\evaluations\2d8fd55592234b4a891974d03ee98ffc')
PERIODS = {'development': ('2025-03-01', '2025-05-31'),
           'confirmation': ('2025-06-01', '2025-08-31'),
           'reused_final': ('2025-09-01', '2025-11-30')}


def now():
    return datetime.now(core.JST)


def write(directory, name, value):
    safety.write_json(directory / name, value, root=ROOT)


def read(directory, name):
    return safety.read_json(directory / name, root=ROOT)


def session():
    if not os.environ.get('PGPASSWORD'):
        raise safety.MLFailure('needs_authentication')
    return read_only_session(connection_config(60000), os.environ['PGPASSWORD'])


def checked_columns(db, table, fields):
    """A physical table is used only after catalog/type validation, including new O1 tables."""
    safety.identifier(table)
    from .ml.data import TABLE_SQL
    info = db.aggregate('stage1.table', TABLE_SQL, ('public', table), max_rows=1)
    if not info:
        return None
    if (len(info) != 1 or info[0]['kind'] != 'r' or info[0]['access_method'] != 'heap'
            or any(info[0][k] for k in ('is_partition', 'descendants', 'rls', 'forced_rls'))):
        raise safety.MLFailure('invalid_schema')
    columns = db.aggregate('stage1.columns', COLUMNS_SQL, ('public', table), max_rows=1000)
    by_name = {c['name']: c for c in columns}
    for name in (*RACE_KEY, *(v[0] for v in fields.values())):
        c = by_name.get(name, {})
        if not c.get('type_is_builtin') or c.get('type_name') not in ('varchar', 'bpchar', 'text'):
            raise safety.MLFailure('invalid_schema')
    return columns


def inventory(directory):
    """Count local O1 snapshots by year/category, without returning any application row."""
    result = {'read_at_jst': now().isoformat(), 'tables': [], 'clock_semantics': {
        'happyo_tsukihi_jifun': 'JRA_publication_MMDDhhmm_not_PC_ingestion',
        'data_sakusei_nengappi': 'source_creation_date_not_availability_time',
        'actual_PC_ingestion_time': 'not_inferred_from_source_creation_or_import_time'}}
    with session() as db:
        names = db.aggregate('stage1.discover',
            "SELECT c.relname AS name, c.reltuples::bigint AS estimated_rows "
            "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relkind='r' "
            "AND (c.relname LIKE '%%o1%%' OR c.relname LIKE '%%odds%%' OR c.relname LIKE '%%jikeiretsu%%') "
            "ORDER BY c.relname", max_rows=200)
        for item in names:
            table = item['name']
            columns = db.aggregate('stage1.discover_columns', COLUMNS_SQL, ('public', table), max_rows=1000)
            fields = {c['name'] for c in columns}
            entry = dict(item, columns=columns, yearly=[], status='not_O1_mapping')
            needed = set(RACE_KEY) | {v[0] for v in publication.O1_FIELDS.values()}
            if needed <= fields:
                checked_columns(db, table, publication.O1_FIELDS)
                entry['status'] = 'O1_schema_observed'
                for year in range(2020, now().year + 1):
                    started = time.monotonic()
                    rows = db.aggregate('stage1.inventory',
                        'SELECT data_kubun::text AS category,count(*) AS snapshots,'
                        'count(DISTINCT (kaisai_tsukihi,keibajo_code,kaisai_kai,kaisai_nichime,race_bango)) AS races,'
                        'min(kaisai_tsukihi::text) AS first_month_day,max(kaisai_tsukihi::text) AS last_month_day '
                        'FROM ONLY public.' + safety.identifier(table) +
                        ' WHERE kaisai_nen::text=%s AND keibajo_code::text IN '
                        "('01','02','03','04','05','06','07','08','09','10') GROUP BY 1 ORDER BY 1",
                        (str(year),), max_rows=30)
                    entry['yearly'].append({'year': year, 'categories': rows,
                                            'elapsed_seconds': round(time.monotonic()-started, 3)})
                entry['clock_columns'] = [c['name'] for c in columns if re.search(
                    'time|date|jifun|jikoku|sakusei', c['name'])]
            result['tables'].append(entry)
    write(directory, 'inventory.json', result)
    return {'tables': [{k: v for k, v in item.items() if k != 'columns'} for item in result['tables']],
            'clock_semantics': result['clock_semantics']}


def prepare(directory):
    """Freeze the protocol before producing new metrics; never train or edit an old artifact."""
    from . import post_step2
    model_dir = safety.ML_ROOT/'models'/MODEL_ID
    model_hash = safety.digest(model_dir/'model.json')
    plan = {'created_at_jst': now().isoformat(), 'model_id': MODEL_ID, 'model_temperature': .75,
        'model_manifest_sha256': model_hash, 'model_file_sha256': safety.digest(model_dir/'win.lgb.txt'),
        'periods': PERIODS, 'period_roles': {'development': 'select_market_temperature_and_weights',
            'confirmation': 'evaluation_after_selection', 'reused_final': 'evaluation_after_selection_already_seen_in_prior_work'},
        'no_new_independent_test_claim': True, 'held_out_2026': '2026-01-01..2026-09-30_not_scored_or_tuned',
        'market_temperature_candidates': analysis.TEMPERATURES, 'ai_weight_candidates': analysis.WEIGHTS,
        'primary_metric': 'horse_binary_log_loss', 'threshold': '.05', 'stake_yen': 100,
        'max_bets_per_race': 1, 'minutes_before_start': analysis.OFFSETS, 'primary_minutes': 10,
        'maximum_quote_age_seconds': 300, 'minute_precision_policy': 'publication_minute_end_must_be_at_or_before_cutoff',
        'schedule_basis': 'saved_RA_no_recorded_start_change_reference_asof_version_unverified',
        'schedule_asof_verified': False, 'actual_PC_observation_time_verified': False,
        'interpretation': 'provisional_posthoc_publication_replay_not_executable_forward_results',
        'payout_policy': 'existing_confirmed_JPY_per_100_HR_normal_single_snapshot_only',
        'bootstrap': {'repetitions': 1000, 'seed': 20261018, 'unit': 'race_cluster_paired'},
        'recent_coverage_only': 'saved_2026-10-17_and_2026-10-18_prediction_rosters_no_new_2026_labels'}
    write(directory, 'plan.json', plan)
    bundle = model.load_models(model_dir, required_targets=('win',))
    predictions, originals = {}, {}
    pool, _ = data.read_dataset(post_step2.POOL)
    cohort, winners, excluded = post_step2.eligible_cohort(pool, *PERIODS['development'])
    raw, _ = post_step2.predict(bundle, cohort, pool)
    predictions['development'] = {'predictions': analysis.normalized_ai(raw), 'winners': winners,
                                  'exclusions': excluded}
    originals['development'] = analysis.measure(predictions['development']['predictions'], winners)
    del pool, cohort, raw
    gc.collect()
    rows, _ = data.read_dataset(PRIOR/'evaluation'/'runners.csv', root=PRIOR.parents[1])
    history, _ = data.read_dataset(PRIOR/'evaluation_history'/'runners.csv', root=PRIOR.parents[1])
    for period in ('confirmation', 'reused_final'):
        cohort, winners, excluded = post_step2.eligible_cohort(rows, *PERIODS[period])
        raw, _ = post_step2.predict(bundle, cohort, history)
        predictions[period] = {'predictions': analysis.normalized_ai(raw), 'winners': winners,
                               'exclusions': excluded}
        originals[period] = analysis.measure(predictions[period]['predictions'], winners)
    records = []
    for period, part in predictions.items():
        for p in part['predictions']:
            records.append({'period': period, 'race_key': p.race, 'probabilities': p.probabilities,
                            'winner': part['winners'][p.race], 'probabilities_already_T075': True})
    # Only rosters, not labels or old-model probabilities, are used for recent coverage.
    for day, artifact in (('20261017', '9b2dc7a96aa24008a0dae3a0c34f31e2'),
                          ('20261018', 'c32ac7a4a2ae404fa7e4ec4342856ade')):
        root = ROOT.parent/('trial-'+day)
        path = safety.local_path(root/'evaluations'/artifact/'predictions.csv', root=root, exists=True)
        originals['frozen_'+day+'_sha256'] = safety.digest(path, root=root)
        rosters = defaultdict(dict)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                race = core.key([row[k] for k in RACE_KEY])
                h = core.horse(row['horse_number'])
                if h in rosters[race]:
                    raise safety.MLFailure('duplicate_entry')
                rosters[race][h] = 1.
        records.extend({'period': 'recent_coverage', 'race_key': race,
                        'roster': sorted(roster), 'winner': None} for race, roster in rosters.items())
    write(directory, 'predictions.local.json', records)
    write(directory, 'original_cohort_metrics.json', originals)
    return {'cohort_race_counts': dict(Counter(r['period'] for r in records)),
            'model_id': MODEL_ID, 'temperature': .75, 'retraining': False, '2026_labels_read': False}


def _quote_document(quote):
    if quote is None:
        return None
    return {'race_key': quote.race, 'odds': {h: float(v) for h, v in quote.odds.items()},
            'source': quote.source, 'category': quote.category, 'basis': quote.basis,
            'published_at': quote.published_at.isoformat() if quote.published_at else None,
            'minutes_before_start': quote.minutes_before_start,
            'cancelled_outside_roster': quote.cancelled_outside_roster}


def _quote(value):
    return market.MarketRace(tuple(value['race_key']), {h: core.decimal(o) for h, o in value['odds'].items()},
        value['source'], value['category'], value['basis'],
        core.instant(value['published_at']) if value['published_at'] else None,
        value['minutes_before_start'], value['cancelled_outside_roster'])


def collect(directory):
    """Read only bounded target-key batches; retain only selected snapshots locally."""
    from . import post_step2
    plan = read(directory, 'plan.json')
    records = read(directory, 'predictions.local.json')
    inventory_value = read(directory, 'inventory.json')
    tables = [t['name'] for t in inventory_value['tables'] if t['status'] == 'O1_schema_observed'
              and any(y['categories'] for y in t['yearly'])]
    if not tables or any(t not in formats.ODDS_TABLES for t in tables):
        raise safety.MLFailure('mapping_not_reviewed')
    before = now().isoformat()
    counts, outcomes_status = Counter(), Counter()
    for offset in range(0, len(records), 32):
        batch = records[offset:offset+32]
        races = tuple(tuple(r['race_key']) for r in batch)
        with session() as db:
            budget = local.Budget(300)
            columns = publication._columns(db, 'jvd_ra', publication.RA_FIELDS, budget)
            if columns is None:
                raise safety.MLFailure('invalid_schema')
            ra = local._rows(db, budget, 'jvd_ra', publication.RA_FIELDS, columns, races, 100, join_targets=True)
            odds = []
            for table in tables:
                columns = publication._columns(db, table, publication.O1_FIELDS, budget)
                packet = local._rows(db, budget, table, publication.O1_FIELDS, columns, races, 20000, join_targets=True)
                for row in packet:
                    row['source_table'] = table
                    counts[table+'.'+str(row.get('status'))] += 1
                odds.extend(packet)
            # HR is needed only for already evaluated 2025 periods, never for the sealed 2026 period.
            settling = tuple(tuple(r['race_key']) for r in batch if r['period'] != 'recent_coverage')
            if settling:
                source = post_step2.confirmed_hr_source(settling)
                hr, source, _ = settle_saved.fetch_hr(db, settling, max_rows=200, reviewed_source=source)
                outcomes = settle_saved.decode_existing_hr(hr, source)
            else:
                hr, outcomes = [], []
        grouped, ra_groups = defaultdict(list), defaultdict(list)
        for row in odds:
            grouped[tuple(row['race_key'])].append(row)
        for row in ra:
            ra_groups[tuple(row['race_key'])].append(row)
        packets = []
        for record in batch:
            race = tuple(record['race_key'])
            roster = tuple(record.get('roster', record.get('probabilities', {})))
            schedule, reason = analysis.reference_schedule(race, ra_groups[race], roster)
            item = {'race_key': race, 'period': record['period'], 'schedule_reason': reason,
                'scheduled_start': schedule.scheduled_start.isoformat() if schedule else None,
                'schedule_asof_verified': False, 'quotes': {}, 'failures': {}, 'selected_raw_O1': {}}
            for minutes in analysis.OFFSETS:
                q, why = (analysis.select_quote(race, grouped[race], roster, schedule, minutes,
                    plan['maximum_quote_age_seconds']) if schedule else (None, reason))
                item['quotes'][str(minutes)] = _quote_document(q)
                item['failures'][str(minutes)] = why
                if q:
                    selected = [r for r in grouped[race] if r.get('status') == '1' and
                        r['source_table'] == q.source and formats.parse_time(r.get('published'), 'mmddhhmm', race) == q.published_at]
                    item['selected_raw_O1'][str(minutes)] = selected
            final, why = market.select_final(race, grouped[race], roster)
            item['final_quote'], item['final_reason'] = _quote_document(final), why
            packets.append(item)
        for outcome in outcomes:
            outcomes_status[outcome.reason or outcome.status] += 1
        write(directory, f'inputs-{offset//32:04d}.local.json', {'records': packets,
            'hr_rows': hr, 'outcomes': [{'race_key': o.race, 'status': o.status,
                'payouts_yen_per_100': o.payouts_yen_per_100, 'verified': o.verified,
                'complete_winner_set': o.complete_winner_set, 'reason': o.reason} for o in outcomes],
            'read_completed_at_jst': now().isoformat(), 'schedule_asof_verified': False})
        if offset % 320 == 0 or offset+32 >= len(records):
            print(json.dumps({'status': 'running', 'stage': 'collect', 'completed_races': min(offset+32, len(records)),
                              'total_races': len(records)}), flush=True)
    result = {'started_at_jst': before, 'finished_at_jst': now().isoformat(),
              'target_races': len(records), 'source_counts': dict(counts), 'hr_status_counts': dict(outcomes_status)}
    write(directory, 'collection.json', result)
    return result


def load_inputs(directory):
    packets, outcomes = {}, {}
    for path in sorted(directory.glob('inputs-*.local.json')):
        document = read(directory, path.name)
        for row in document['records']:
            race = tuple(row['race_key'])
            if race in packets:
                raise safety.MLFailure('duplicate_entry')
            packets[race] = row
        for row in document['outcomes']:
            race = tuple(row['race_key'])
            if race in outcomes:
                raise safety.MLFailure('duplicate_entry')
            outcomes[race] = core.Outcome(race, row['status'], row['payouts_yen_per_100'],
                row['verified'], row['complete_winner_set'], row['reason'])
    return packets, outcomes


def coverage_summary(records, packets):
    result = {}
    for period in sorted({r['period'] for r in records}):
        group = [r for r in records if r['period'] == period]
        timings = {}
        for minutes in analysis.OFFSETS:
            selected, ages, failures = [], [], Counter()
            for row in group:
                item = packets[tuple(row['race_key'])]
                q = item['quotes'][str(minutes)]
                if q:
                    selected.append(q)
                    start = core.instant(item['scheduled_start'])
                    ages.append((start-timedelta(minutes=minutes)-core.instant(q['published_at'])).total_seconds())
                else:
                    failures[item['failures'][str(minutes)]] += 1
            timings[str(minutes)] = {'usable_races': len(selected), 'total_races': len(group),
                'coverage_percent': 100*len(selected)/len(group) if group else None,
                'excluded': dict(failures), 'publication_age_seconds': core.distribution(ages),
                'publication_minute_end_not_after_cutoff': True, 'schedule_asof_verified': False}
        result[period] = timings
    return result


def evaluate(directory):
    records = read(directory, 'predictions.local.json')
    packets, outcomes = load_inputs(directory)
    outcomes, mismatch_count = analysis.match_outcomes(outcomes,
        {tuple(r['race_key']):r['winner'] for r in records if r['winner'] is not None})
    if set(packets) != {tuple(r['race_key']) for r in records}:
        raise safety.MLFailure('artifact_invalid')
    directory = safety.local_path(directory/'analyses'/uuid4().hex, root=ROOT)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = coverage_summary(records, packets)
    write(directory, 'coverage.json', coverage)
    cohorts = {}
    for period in PERIODS:
        eligible = [r for r in records if r['period'] == period and packets[tuple(r['race_key'])]['quotes']['10']]
        cohorts[period] = {'ai': [core.PredictionRace(tuple(r['race_key']), r['probabilities']) for r in eligible],
            'quotes': [_quote(packets[tuple(r['race_key'])]['quotes']['10']) for r in eligible],
            'winners': {tuple(r['race_key']): r['winner'] for r in eligible}}
    dev = cohorts['development']
    if not dev['ai']:
        write(directory, 'results.json', {'status': 'unavailable', 'reason': 'no_development_pre_race_quotes',
                                         'coverage': coverage})
        return {'status': 'unavailable', 'reason': 'no_development_pre_race_quotes', 'coverage': coverage}
    selection = analysis.fit_development(dev['ai'], dev['quotes'], dev['winners'],
        development_end=PERIODS['development'][1], evaluation_start=PERIODS['confirmation'][0])
    # Exclusive write completes before any held-period mixed probability or score is generated.
    write(directory, 'selection.json', selection)
    results, simulation_documents = {}, []
    for period, group in cohorts.items():
        if not group['ai']:
            results[period] = {'status': 'unavailable'}
            continue
        variants = analysis.variants(group['ai'], group['quotes'], selection)
        metrics = {name: analysis.measure(p, group['winners']) for name, p in variants.items()}
        intervals = {name: {'versus_raw_market': analysis.paired(p, variants['market_raw'], group['winners']),
                           'versus_recalibrated_market': analysis.paired(p, variants['market_recalibrated'], group['winners'])}
                     for name, p in variants.items() if name in ('best242', 'linear_raw', 'linear_recalibrated', 'geometric_recalibrated')}
        simulations = {}
        for name, predictions in variants.items():
            schedules = [core.Schedule(p.race, core.instant(packets[p.race]['scheduled_start']), None,
                tuple(p.probabilities), True, 'reviewed_historical_schedule') for p in predictions]
            snapshots = [core.Snapshot(q.race, q.published_at, None, q.odds, q.source, True,
                                      publication_precision='minute') for q in group['quotes']]
            policy = core.WinPolicy(probability_method='identity', timing_basis='jra_publication_time',
                evaluation_start=PERIODS[period][0], evaluation_end=PERIODS[period][1],
                model_fit_end='2024-12-28')
            ledger = core.run_simulation(predictions, schedules, snapshots,
                [outcomes[p.race] for p in predictions if p.race in outcomes], policy=policy)
            simulations[name] = ledger['summary']
            simulations[name]['selected_bands'] = selected_bands(ledger['rows'])
            simulation_documents.append({'period': period, 'variant': name, 'ledger': ledger,
                                          'schedule_asof_verified': False})
        finals = {p.race: _quote(packets[p.race]['final_quote']) for p in group['ai'] if packets[p.race]['final_quote']}
        results[period] = {'status': 'ok', 'selection_period': period == 'development',
            'previously_evaluated_period': True, 'metrics': metrics, 'paired_bootstrap': intervals,
            'top1_agreement': analysis.disagreement(group['ai'], variants['market_raw'], group['winners']),
            'simulations': simulations, 'price_drift': analysis.price_drift(group['quotes'], finals)}
    value = {'status': 'ok', 'model_id': MODEL_ID, 'model_temperature': .75, 'coverage': coverage,
        'selection': selection, 'periods': results, 'schedule_asof_verified': False,
        'hr_se_winner_mismatch_count': mismatch_count,
        'interpretation': 'provisional_historical_publication_replay_not_forward_execution',
        'held_out_2026': '2026-01-01..2026-09-30_not_scored_or_tuned'}
    write(directory, 'simulations.local.json', simulation_documents)
    write(directory, 'results.json', value)
    return {'analysis_id': directory.name, 'coverage': coverage, 'selection': {k: selection[k] for k in ('market_temperature', 'ai_weights')},
        'periods': {name: {'metrics': {k: {f: v for f, v in m.items() if f != 'calibration'}
                                      for k, m in item.get('metrics', {}).items()},
            'top1_agreement': item.get('top1_agreement')} for name, item in results.items()}}


def selected_bands(rows):
    specs = {'probability': (0., .01, .02, .05, .1, .2, 1.00001),
             'decision_odds': (1., 5., 10., 20., 50., 100., float('inf')),
             'expected_return': (.05, .1, .25, .5, 1., 2., 5., 10., float('inf'))}
    result = {}
    for field, edges in specs.items():
        bands = []
        for low, high in zip(edges, edges[1:]):
            selected = []
            for row in rows:
                runner = next((r for r in row['runners'] if r['horse_number'] == row['ai_horse']), None)
                if runner and low <= runner[field] < high:
                    selected.append(row)
            bands.append({'lower_inclusive': low, 'upper_exclusive': None if high == float('inf') else high,
                          **core.financial(selected, 'ai')})
        result[field] = bands
    return result


def report(directory):
    source_directory = directory
    inventory_value = read(directory, 'inventory.json')
    plan = read(directory, 'plan.json')
    preserved = verify_preserved(source_directory, plan)
    analyses = list((directory/'analyses').glob('*/results.json'))
    if not analyses:
        raise safety.MLFailure('insufficient_history')
    directory = max(analyses, key=lambda p: p.stat().st_mtime_ns).parent
    result = read(directory, 'results.json')
    for table in inventory_value['tables']:
        table['clock_columns'] = [c['name'] for c in table['columns'] if re.search(
            'time|date|jifun|jikoku|sakusei', c['name'])]
        table['PC_ingestion_clock_present'] = any(c['type_name'] in ('timestamp','timestamptz')
                                                 for c in table['columns'])
    write(directory, 'inventory.reviewed.json', inventory_value)
    lines = ['# Stage 1 市場・AI相補性診断', '', f"作成日時: {now().isoformat()}", '',
        f'固定モデル: {MODEL_ID} / T=0.75。再学習・特徴量変更・購入条件の最適化は行っていません。', '',
        '## 評価上の条件', '',
        '- 発表時刻基準の暫定的な事後シミュレーションです。PC実取得時刻・購入可能時刻は未検証です。',
        '- 保存RAの発走予定時刻を基準とし、変更前時刻が記録されたレースは除外。当時のRA版は保存されておらず、変更履歴がないことの完全な保証はありません。',
        '- 発表時刻は分精度のため、その分の末尾が判断時刻以前となる記録だけを採用。完全一致の分は採用しません。鮮度上限300秒を事前固定。',
        '- 2025年3～5月で市場再較正・混合重みを選び、6～8月と9～11月では固定。いずれも以前の研究で評価済みで、新しい未使用テストとは主張しません。',
        '- 2026年1～9月の性能・収益を今回開いていません。10月17～18日は保存済み出走馬集合で被覆率のみ確認。',
        '- 仮想購入は単勝・EV5%以上・最大1点・100円。発走前表示オッズは払戻を保証せず、精算は既存HRデコーダーによる確認済み通常払戻のみです。',
        '- 既存評価の完全着順・単独勝馬・基本特徴量充足の母集団を保持。中止・取消・同着等を含む全運用母集団の成績ではありません。',
        '- 精度表の1位的中率は既存の同率按分規則。一致/不一致表と仮想購入は既存の馬番最小規則。',
        '- bootstrapはレース単位ペアード1000回。開催日相関・多重比較の補正は行っていません。', '',
        '## 収録状況', '', '| テーブル | 年 | 区分 | スナップショット | レース |', '|---|---:|---|---:|---:|']
    for table in inventory_value['tables']:
        for year in table['yearly']:
            for item in year['categories']:
                lines.append(f"| {table['name']} | {year['year']} | {item['category']} | {item['snapshots']} | {item['races']} |")
    lines += ['', 'O1公式区分: 1=中間、2=前日売最終、3=最終、4=確定、5=月曜確定、9=中止、0=削除。中間以外は事前オッズとして採用しません。',
              '単勝・複勝は同じO1レコード内の別列。発表月日時分はJRAの発表時刻で、データ作成日や今回の登録・読込時刻を当時の取得時刻には変換しません。', '',
              '## 被覆率', '', '| 期間 | 発走何分前 | 対象 | 利用可能 | 被覆率 | 除外理由 |', '|---|---:|---:|---:|---:|---|']
    for period, timings in result['coverage'].items():
        for minutes, value in timings.items():
            lines.append(f"| {period} | {minutes} | {value['total_races']} | {value['usable_races']} | {value['coverage_percent']:.2f}% | {json.dumps(value['excluded'], ensure_ascii=False)} |")
    if result['status'] == 'ok':
        lines += ['', '## 開発期間で固定した設定', '', f"市場温度: {result['selection']['market_temperature']}",
            f"AI混合重み: {json.dumps(result['selection']['ai_weights'])}", '',
            '市場再較正による改善とAI追加による改善を分けるため、混合はraw市場・再較正済み市場の双方と比較します。', '',
            '## 同一レース・同一出走馬の予測性能', '',
            '| 期間 | 予測器 | R | 馬単位LL | Brier | 勝馬LL/R | 1位的中率 |', '|---|---|---:|---:|---:|---:|---:|']
        for period, item in result['periods'].items():
            for name, m in item.get('metrics', {}).items():
                lines.append(f"| {period} | {name} | {m['races']} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['race_log_loss']:.6f} | {m['top1_accuracy_percent']:.2f}% |")
        lines += ['', '## AI追加効果の区間推定', '',
            '| 期間 | 混合 | 比較基準 | 馬単位LL差 | 95%下限 | 95%上限 |', '|---|---|---|---:|---:|---:|']
        for period, item in result['periods'].items():
            if period == 'development':
                continue
            for name, comparisons in item.get('paired_bootstrap', {}).items():
                for basis, values in comparisons.items():
                    v = values['log_loss']
                    lines.append(f"| {period} | {name} | {basis} | {v['difference_new_minus_baseline']:.7f} | {v['ci95_low']:.7f} | {v['ci95_high']:.7f} |")
        held = [result['periods'][p] for p in ('confirmation', 'reused_final') if result['periods'][p]['status']=='ok']
        supported = any(item['paired_bootstrap'][name]['versus_recalibrated_market']['log_loss']['ci95_high'] < 0
                        for item in held for name in ('linear_recalibrated', 'geometric_recalibrated'))
        lines += ['', 'AIの追加情報: '+('改善を支持する区間があるため期間間の再現性を要確認。' if supported else
            '設定選択後の評価で市場単独に対する明確な改善を確認できません。現行モデルの混合採用・収益化の根拠とはしません。'),
            '開発期間の微小な改善だけでは追加価値があると認定しません。判定はStage 2の特徴量追加を中止する理由にはしません。', '',
            '## AI1位と市場1位の一致・不一致', '',
            '| 期間 | 条件 | レース数 | AI的中 | 市場的中 |', '|---|---|---:|---:|---:|']
        for period, item in result['periods'].items():
            if 'top1_agreement' not in item:
                continue
            for condition in ('agreement', 'disagreement'):
                c = item['top1_agreement'][condition]
                lines.append(f"| {period} | {condition} | {c['races']} | {c['ai_hits']} | {c['market_hits']} |")
        lines += ['', '## 仮想収益（参考）', '',
            '| 期間 | 予測器 | 購入 | 的中 | 投資円 | 払戻円 | 純利益円 | 回収率 | 未精算 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for period, item in result['periods'].items():
            if 'simulations' not in item:
                continue
            for name, s in item.get('simulations', {}).items():
                f = s['ai']
                roi = f"{f['return_rate_percent']:.2f}%" if f['return_rate_percent'] is not None else '未算出'
                lines.append(f"| {period} | {name} | {f['bet_count']} | {f['hit_count']} | {f['investment_yen']} | {f['payout_yen']} | {f['net_profit_yen']} | {roi} | {f['unsettled_count']} |")
            f = item['simulations']['best242']['favorite']
            roi = f"{f['return_rate_percent']:.2f}%" if f['return_rate_percent'] is not None else '未算出'
            lines.append(f"| {period} | 同時点1番人気 | {f['bet_count']} | {f['hit_count']} | {f['investment_yen']} | {f['payout_yen']} | {f['net_profit_yen']} | {roi} | {f['unsettled_count']} |")
        lines += ['', '## 月別参考収益', '', '| 月 | 戦略 | 購入 | 投資円 | 払戻円 | 純利益円 | 回収率 |', '|---|---|---:|---:|---:|---:|---:|']
        for item in result['periods'].values():
            if 'simulations' not in item:
                continue
            for name in ('best242', 'linear_recalibrated', 'geometric_recalibrated'):
                for month, summary in item['simulations'][name]['by_month'].items():
                    f = summary['ai']
                    roi = f"{f['return_rate_percent']:.2f}%" if f['return_rate_percent'] is not None else '未算出'
                    lines.append(f"| {month} | {name} | {f['bet_count']} | {f['investment_yen']} | {f['payout_yen']} | {f['net_profit_yen']} | {roi} |")
        lines += ['', '月別収支・選択馬の勝率帯/オッズ帯/EV帯・校正帯・発走前と確定オッズの乖離は同じフォルダーのresults.jsonへ匿名集計で保存しています。', '',
            '## Stage 2の候補', '',
            '1. 既存着順・過去レース頭数・日付を使う頭数正規化と近走重視。新規デコーダーを増やさず開始しやすい。',
            '2. RA grade_code、kyoso_joken_code等のクラス・条件。コード対応を確認してカテゴリとして追加。',
            '3. SE chakusa_code_1～3、kohan_3f、corner_1～4の過去走情報。原形式と特殊値を確認し、当該レース結果を入れない。',
            '4. SE kishu_code、chokyoshi_code。過去成績は全騎乗・全出走を母数とし、対象馬限定履歴から勝率を作らない。',
            '', '相補性が確認できなくても、現行15特徴量モデルの基準線として保持します。Stage 2/3の実装は今回行いません。']
    path = safety.local_path(directory/'report.md', root=ROOT)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    write(directory, 'verification.json', preserved)
    return {'report': str(path), 'status': result['status'], **preserved}


def verify_preserved(directory, plan):
    if (safety.digest(safety.ML_ROOT/'models'/MODEL_ID/'model.json') != plan['model_manifest_sha256']
            or safety.digest(safety.ML_ROOT/'models'/MODEL_ID/'win.lgb.txt') != plan['model_file_sha256']):
        raise safety.MLFailure('artifact_invalid')
    originals = read(directory, 'original_cohort_metrics.json')
    for day, artifact in (('20261017', '9b2dc7a96aa24008a0dae3a0c34f31e2'),
                          ('20261018', 'c32ac7a4a2ae404fa7e4ec4342856ade')):
        root = ROOT.parent/('trial-'+day)
        path = root/'evaluations'/artifact/'predictions.csv'
        if safety.digest(path, root=root) != originals['frozen_'+day+'_sha256']:
            raise safety.MLFailure('artifact_invalid')
    def previous(name):
        return safety.read_json(PRIOR/name, root=PRIOR.parents[1])
    references = {'development':previous('feature_results.json')['conditions']['baseline']['metrics']['model'],
        'confirmation':previous('independent.json')['confirmation']['metrics']['best242']['model'],
        'reused_final':previous('final_evaluation.json')['metrics']['best242']['model']}
    maximum = max(abs(originals[p][f]-references[p][f]) for p in PERIODS
                  for f in ('log_loss','brier_score','race_log_loss','top1_accuracy_percent'))
    if maximum > 1e-12:
        raise safety.MLFailure('artifact_invalid')
    return {'old_model_unchanged': True, 'frozen_20261017_18_predictions_unchanged': True,
            'old_full_cohort_metrics_max_difference': maximum}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inventory', 'prepare', 'collect', 'evaluate', 'report'), default='inventory')
    parser.add_argument('--work-id', default='')
    args = parser.parse_args(argv)
    stage = args.stage
    try:
        if args.work_id:
            if not re.fullmatch('[a-f0-9]{32}', args.work_id):
                raise safety.MLFailure('invalid_input')
            directory = safety.local_path(ROOT/'evaluations'/args.work_id, root=ROOT)
            if not directory.is_dir():
                raise safety.MLFailure('invalid_input')
        else:
            directory = safety.new_artifact('evaluations', root=ROOT)
        result = globals()[stage](directory)
        print(json.dumps(dict(stage=stage, work_id=directory.name, **{'status':'ok', **result}), ensure_ascii=True))
        return 0
    except SafeAuditError as exc:
        print(json.dumps(dict(status='stopped', stage=stage, reason=exc.code)))
        return 2
    except Exception as exc:
        import traceback
        frames = traceback.extract_tb(exc.__traceback__)
        location = {'file':Path(frames[-1].filename).name, 'line':frames[-1].lineno} if frames else {}
        print(json.dumps(dict(status='stopped', stage=stage, reason='internal_error',
                              exception_type=type(exc).__name__, location=location)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
