"""Read-only Stage 6 saved-results adapter. No DB, fitting, or Stage 7 input.

Missing display metadata stays blank. The Stage 6 receiver revision explicitly
accepts those blanks; numerical identity and historical provenance stay strict.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import re
import stat

from .core import RACE_KEY
from .ml import safety

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
S2 = BASE / 'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S6 = BASE / 'stage6-development/evaluations/719a0d3915194bc59b6e55eba2659576'
OUTPUT = BASE / 'keiba-lab-export'
SCHEMA = 'keiba-lab.v1.1'
RECEIVER_REVISION = '2026-10-24.stage6-missing-display'
COLUMNS = (
    'race_id', 'date', 'venue', 'race_no', 'race_name', 'surface', 'distance',
    'going', 'class', 'horse_no', 'horse_name', 'model_id', 'pred_rank', 'win_prob',
    'finish_position', 'win_payout_100', 'predicted_at', 'start_time',
    'schema_version', 'dataset_id', 'evaluation_kind', 'source_stage',
    'source_run_id', 'model_artifact_id', 'source_sha256',
)
MODEL_COLUMNS = {'H': 'source_H_model_id', 'raw_market': None,
                 'H_style': 'H_style_model_id', 'H_pace': 'H_pace_model_id'}
# Same confirmed venue dictionary as duel/replay.py; do not import its DB pipeline.
VENUES = dict(zip(('01', '02', '03', '04', '05', '06', '07', '08', '09', '10'),
                  ('札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉')))
IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,119}\Z')
MAX_ROWS, MAX_BYTES = 20_000, 8 * 1024 * 1024


def guard_day(text):
    try:
        d = date.fromisoformat(text)
    except (ValueError, TypeError):
        raise safety.MLFailure('invalid_date') from None
    if str(d) != text or d.year not in (2023, 2024, 2025) or d.month > 9:
        raise safety.MLFailure('invalid_split')
    return d


def identity(key, day):
    guard_day(day)  # Before reading any label/probability/result.
    if (not isinstance(key, str) or not re.fullmatch(r'[0-9]{18}', key)
            or key[:8] != day.replace('-', '') or key[8:10] not in VENUES
            or not 1 <= int(key[10:12]) <= 99 or not 1 <= int(key[12:14]) <= 99
            or not 1 <= int(key[14:16]) <= 12 or not 1 <= int(key[16:]) <= 28):
        raise safety.MLFailure('invalid_identity')
    return key[:16]


def numeric(text):
    try:
        value = float(text)
    except (ValueError, TypeError, OverflowError):
        raise safety.MLFailure('invalid_input') from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise safety.MLFailure('invalid_input')
    return value


def select_rows(stream, start, end, limit):
    """First N chronological races, selected without looking at their outcomes."""
    guard_day(start); guard_day(end)
    if start > end or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise safety.MLFailure('invalid_split')
    reader = csv.DictReader(stream)
    required = {'entry_key', 'date', 'winner_label', 'evaluation_year',
                'raw_market_source', *MODEL_COLUMNS,
                *(v for v in MODEL_COLUMNS.values() if v)}
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
        raise safety.MLFailure('invalid_schema')
    selected = {}; previous = ''; seen = set()
    for index, row in enumerate(reader):
        if index >= 100_000:
            raise safety.MLFailure('input_limit')
        if None in row or any(value is None for value in row.values()):
            raise safety.MLFailure('invalid_schema')
        race = identity(row['entry_key'], row['date'])
        if row['entry_key'] <= previous:
            raise safety.MLFailure('duplicate_entry' if row['entry_key'] == previous else 'invalid_identity')
        previous = row['entry_key']
        if not start <= row['date'] <= end:
            continue
        if race not in selected and len(selected) == limit:
            break
        if row['entry_key'] in seen:
            raise safety.MLFailure('duplicate_entry')
        seen.add(row['entry_key'])
        if row['evaluation_year'] != row['date'][:4] or numeric(row['winner_label']) not in (0., 1.):
            raise safety.MLFailure('invalid_identity')
        for model, column in MODEL_COLUMNS.items():
            numeric(row[model])
            if column and not IDENTIFIER.fullmatch(row[column]):
                raise safety.MLFailure('invalid_model')
        if row['raw_market_source'] != 'saved_T10_inverse_win_odds':
            raise safety.MLFailure('invalid_mapping')
        selected.setdefault(race, []).append(row)
    if not selected:
        raise safety.MLFailure('insufficient_history')
    for rows in selected.values():
        if len(rows) < 2 or sum(numeric(r['winner_label']) for r in rows) != 1:
            raise safety.MLFailure('invalid_identity')
        for model, column in MODEL_COLUMNS.items():
            if abs(math.fsum(numeric(r[model]) for r in rows) - 1) > 1e-12:
                raise safety.MLFailure('invalid_input')
            if column and len({r[column] for r in rows}) != 1:
                raise safety.MLFailure('invalid_model')
    if sum(map(len, selected.values())) * len(MODEL_COLUMNS) > MAX_ROWS:
        raise safety.MLFailure('input_limit')
    return selected


def source_rows(stream, selected):
    """Read the saved monthly RA/SE cache, retain only the exact selected keys."""
    reader = csv.DictReader(stream)
    required = {*RACE_KEY, 'horse_number', 'finish_position', 'distance', 'surface'}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise safety.MLFailure('invalid_schema')
    found = defaultdict(dict)
    for row in reader:
        race = ''.join(row[k] for k in RACE_KEY)
        # Monthly input is selected by a development-only YYYYMM before opening.
        if race not in selected:
            continue
        key = race + row['horse_number']
        identity(key, selected[race][0]['date'])
        if key in found[race]:
            raise safety.MLFailure('duplicate_entry')
        found[race][key] = {k: row.get(k, '') for k in (
            'finish_position', 'distance', 'surface', 'horse_name', 'race_name', 'going', 'class')}
    return dict(found)


def official_finishes(rows, source):
    if set(source) != {r['entry_key'] for r in rows}:
        raise safety.MLFailure('invalid_identity')
    raw = {key: item['finish_position'] for key, item in source.items()}
    if (not all(re.fullmatch(r'[0-9]{2}', value or '') for value in raw.values())
            or sorted(int(v) for v in raw.values()) != list(range(1, len(rows) + 1))):
        # No invented 2..N, DNF/DSQ translation, dead-heat resolution or 0-as-last.
        return {}, 'unsupported_or_incomplete_official_result'
    result = {key: int(value) for key, value in raw.items()}
    if any(int(numeric(r['winner_label'])) != int(result[r['entry_key']] == 1) for r in rows):
        raise safety.MLFailure('invalid_identity')
    return result, None


def convert(selected, metadata, dataset_id, prediction_sha):
    if not IDENTIFIER.fullmatch(dataset_id) or not re.fullmatch(r'[a-f0-9]{64}', prediction_sha):
        raise safety.MLFailure('invalid_input')
    output = []; result_reasons = Counter()
    for race, rows in selected.items():
        source = metadata.get(race, {})
        finishes, reason = official_finishes(rows, source)
        if reason:
            result_reasons[reason] += 1
        for field in ('distance', 'surface', 'race_name', 'going', 'class'):
            if len({v[field] for v in source.values()}) != 1:
                raise safety.MLFailure('invalid_identity')
        for model, column in MODEL_COLUMNS.items():
            order = sorted(rows, key=lambda r: (-numeric(r[model]), int(r['entry_key'][-2:])))
            ranks = {r['entry_key']: i for i, r in enumerate(order, 1)}
            for row in rows:
                key = row['entry_key']; extra = source[key]
                distance = extra['distance']
                if distance and (not re.fullmatch(r'[0-9]{3,4}', distance) or not 600 <= int(distance) <= 6000):
                    distance = ''
                # source.surface is an untranslated JV track code, not 芝/ダート.
                # Do not guess a display category from an unversioned lookup.
                surface = extra['surface'] if extra['surface'] in ('芝', 'ダート', '障害') else ''
                going = extra['going'] if extra['going'] in ('良', '稍重', '重', '不良') else ''
                item = dict.fromkeys(COLUMNS, '')
                item.update(race_id=race, date=row['date'], venue=VENUES[race[8:10]], race_no=int(race[-2:]),
                    race_name=extra['race_name'], surface=surface, distance=int(distance) if distance else '',
                    going=going, **{'class': extra['class']}, horse_no=int(key[-2:]), horse_name=extra['horse_name'],
                    model_id=model, pred_rank=ranks[key], win_prob=repr(numeric(row[model])),
                    finish_position=finishes.get(key, ''), schema_version=SCHEMA, dataset_id=dataset_id,
                    evaluation_kind='stage6_development', source_stage='stage6', source_run_id=S6.name,
                    model_artifact_id=row[column] if column else 'no_fit:raw_market:T10', source_sha256=prediction_sha)
                for value in item.values():
                    if isinstance(value, str) and (re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', value)
                                                  or value.startswith(('=', '+', '-', '@', '\t', '\r', '\n'))):
                        raise safety.MLFailure('invalid_input')
                output.append(item)
    missing = {field: sum(r[field] == '' for r in output) for field in COLUMNS}
    warnings = ['missing_display_' + k for k in ('race_name', 'surface', 'distance', 'going', 'class', 'horse_name') if missing[k]]
    return output, {'missing_cells_by_column': missing, 'result_pending_races': dict(result_reasons),
                    'keiba_lab_v1_1_import_ready': True, 'intake_blocker_codes': [],
                    'receiver_contract_revision': RECEIVER_REVISION,
                    'missing_display_warning_codes': warnings,
                    'legacy_strict_display_import_ready': not warnings}


def csv_text(rows):
    stream = StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\r\n')
    writer.writeheader(); writer.writerows(rows)
    value = '\ufeff' + stream.getvalue()
    if len(rows) > MAX_ROWS or len(value.encode('utf-8')) > MAX_BYTES:
        raise safety.MLFailure('input_limit')
    return value


def reconcile(selected, exported, metadata):
    """Round-trip values and re-use Stage 6's actual evaluation primitives."""
    import pandas as pd
    from .stage5 import metrics
    from .win_sim.core import PredictionRace

    by_model = defaultdict(dict); seen = set()
    for row in exported:
        if set(row) != set(COLUMNS):
            raise safety.MLFailure('invalid_schema')
        race = identity(row['race_id'] + f"{int(row['horse_no']):02d}", row['date'])
        model = row['model_id']; key = race + f"{int(row['horse_no']):02d}"
        if race not in selected or model not in MODEL_COLUMNS or (model, key) in seen:
            raise safety.MLFailure('invalid_identity')
        seen.add((model, key)); by_model[model][key] = row
    count = sum(map(len, selected.values()))
    if len(seen) != count * len(MODEL_COLUMNS):
        raise safety.MLFailure('invalid_identity')
    points = {}; maximum = 0.; pending = 0
    for model, column in MODEL_COLUMNS.items():
        before = []; after = []; frame = []; restored = []
        for race, rows in selected.items():
            expected_finishes, _ = official_finishes(rows, metadata.get(race, {}))
            order = sorted(rows, key=lambda r: (-numeric(r[model]), int(r['entry_key'][-2:])))
            ranks = {r['entry_key']: i for i, r in enumerate(order, 1)}
            full_results = True; original = {}; round_trip = {}
            for row in rows:
                key = row['entry_key']; got = by_model[model].get(key)
                if got is None:
                    raise safety.MLFailure('invalid_identity')
                prob = numeric(got['win_prob'])
                expected_model = row[column] if column else 'no_fit:raw_market:T10'
                if (prob != numeric(row[model]) or int(got['pred_rank']) != ranks[key]
                        or got['model_artifact_id'] != expected_model or got['source_run_id'] != S6.name
                        or got['evaluation_kind'] != 'stage6_development' or got['source_stage'] != 'stage6'
                        or got['schema_version'] != SCHEMA or got['predicted_at'] or got['start_time']
                        or got['win_payout_100']
                        or str(got['finish_position']) != str(expected_finishes.get(key, ''))):
                    raise safety.MLFailure('artifact_invalid')
                original[key[-2:]] = numeric(row[model]); round_trip[key[-2:]] = prob
                full_results &= got['finish_position'] != ''
            if not full_results:
                if any(by_model[model][r['entry_key']]['finish_position'] != '' for r in rows):
                    raise safety.MLFailure('invalid_identity')
                pending += int(model == 'H')
                continue
            finishes = [int(by_model[model][r['entry_key']]['finish_position']) for r in rows]
            if sorted(finishes) != list(range(1, len(rows) + 1)):
                raise safety.MLFailure('invalid_identity')
            parts = (race[:4], race[4:8], race[8:10], race[10:12], race[12:14], race[14:16])
            before.append(PredictionRace(parts, original)); after.append(PredictionRace(parts, round_trip))
            for row, finish in zip(rows, finishes):
                label = int(numeric(row['winner_label']))
                if label != int(finish == 1):
                    raise safety.MLFailure('invalid_identity')
                frame.append({'entry_key': row['entry_key'], 'date': row['date'], 'winner_label': label})
                restored.append({'entry_key': row['entry_key'], 'date': row['date'], 'winner_label': int(finish == 1)})
        if frame:
            a = metrics.point_only(pd.DataFrame(frame), before)
            b = metrics.point_only(pd.DataFrame(restored), after)
            diff = max(abs(a[k] - b[k]) for k in a)
            if diff > 1e-12:
                raise safety.MLFailure('artifact_invalid')
            maximum = max(maximum, diff)
            points[model] = {'source': a, 'csv_round_trip': b, 'maximum_metric_difference': diff}
    return {'all_probabilities_ranks_model_ids_and_winner_labels_match': True,
            'maximum_metric_difference': maximum, 'verified_models': len(points), 'pending_result_races': pending,
            'metrics': points, 'rank_rule': 'probability_descending_then_horse_number_ascending',
            'metric_top1_rule': 'existing_Stage6_fractional_credit_for_exact_probability_ties',
            'lab_Brier_note': 'LAB=race_mean_sum_of_squared_errors;Stage6=horse_mean_squared_error',
            'lab_top1_note': 'LAB=one_ordinal_top;Stage6=fractional_tie_credit;not_always_equal'}


def export(*, start='2023-01-01', end='2025-09-30', limit=1):
    guard_day(start); guard_day(end)
    references = {}

    def reference(path, expected=None):
        path = safety.local_path(path, root=BASE, exists=True)
        actual = safety.digest(path, root=BASE)
        if expected is not None and actual != expected:
            raise safety.MLFailure('artifact_invalid')
        references[str(path)] = actual
        return path

    report_path = reference(S6 / 'development_report.json')
    report = safety.read_json(report_path, root=BASE)
    if (report.get('run_id') != S6.name or report.get('version') != 'stage6_previous_corner_field_interaction_v1'
            or report.get('2026_data_processed') is not False
            or report.get('development_reuse_not_independent_final_test') is not True):
        raise safety.MLFailure('invalid_split')
    path = reference(S6 / 'predictions.local.csv', report['prediction_detail']['sha256'])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        selected = select_rows(stream, start, end, limit)
    manifest = safety.read_json(reference(S2 / 'source_manifest.json'), root=BASE)
    metadata = {}
    for month in sorted({r[:6] for r in selected}):
        guard_day(f'{month[:4]}-{month[4:]}-01')
        matches = [p for p in manifest['months'] if p.get('file') == month + '.csv']
        if len(matches) != 1:
            raise safety.MLFailure('artifact_invalid')
        source = reference(S2 / 'source' / (month + '.csv'), matches[0]['sha256'])
        with source.open(encoding='utf-8-sig', newline='') as stream:
            metadata.update(source_rows(stream, selected))
    directory = safety.new_artifact('evaluations', root=OUTPUT)
    dataset_id = 'stage6-export-' + directory.name
    rows, availability = convert(selected, metadata, dataset_id, references[str(path)])
    csv_path = safety.local_path(directory / 'keiba-lab.csv', root=OUTPUT)
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        stream.write(csv_text(rows)); stream.flush()
        import os
        os.fsync(stream.fileno())
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != COLUMNS:
            raise safety.MLFailure('invalid_schema')
        validation = reconcile(selected, list(reader), metadata)
    for name, expected in references.items():
        if safety.digest(Path(name), root=BASE) != expected:
            raise safety.MLFailure('artifact_invalid')
    value = {'schema_version': SCHEMA, 'dataset_id': dataset_id, 'source_stage': 'stage6',
        'source_run_id': S6.name, 'evaluation_kind': 'stage6_development',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'races': len(selected),
        'runners': sum(map(len, selected.values())), 'models': list(MODEL_COLUMNS), 'csv_rows': len(rows),
        'csv_sha256': safety.digest(csv_path, root=OUTPUT), 'references_sha256': references,
        'preservation': {'all_used_source_files_unchanged': True, 'DB_operations': 0, 'fits': 0,
                         'Stage7_inputs': 0, 'sealed_2026_inputs': 0, 'task_mutations': 0},
        'availability': availability, 'validation': validation,
        'model_roles': {'H': 'frozen_research_control', 'raw_market': 'no_fit_reference',
                        'H_style': 'individual_style_control_not_promoted', 'H_pace': 'not_retained'},
        'field_notes': {'finish_position': 'saved_RA_SE_source_actual_finish_not_winner_label_inference',
            'surface': 'untranslated_track_code_is_not_a_display_surface;blank_until_confirmed',
            'class': 'raw_grade_condition_codes_not_guessed_as_display_class',
            'predicted_at,start_time': 'both_blank_no_authentic_original_prediction_timestamp',
            'win_payout_100': 'blank_Stage6_revenue_out_of_scope',
            'raw_market_model_artifact_id': 'no_fit:raw_market:T10_is_a_method_identifier_not_a_trained_model_id'},
        'interpretation': 'reused_development_postrace_evaluation_not_authentic_prerace_or_independent_final_test'}
    safety.write_json(directory / 'manifest.json', value, root=OUTPUT)
    text = '\n'.join([
        '# KEIBA LAB：保存済みStage 6評価のCSV照合', '',
        f'対象 {value["races"]}レース・{value["runners"]}頭・4系列、CSV {len(rows)}行。',
        '開催後の開発評価の参考再集計。真正な発走前保存予測、独立最終評価、収益評価ではありません。',
        '予測値は再計算・較正・丸めをせず保存値を使用。実着順は保存済みRA/SEと完全キー結合。',
        f'確率・順位・モデルID・勝敗対応：一致。既存指標との差の最大 {validation["maximum_metric_difference"]}。',
        f'確定結果で照合できた系列 {validation["verified_models"]}、結果保留 {validation["pending_result_races"]}レース。',
        'KEIBA LAB v1.1研究用欠損対応版と列互換：はい（実画面取込は別途確認）。',
        '受入規則の版：' + RECEIVER_REVISION,
        '取込阻害コード：' + ', '.join(availability['intake_blocker_codes']),
        '欠損を仮の名称・馬場・クラスで埋めていません。受入画面の「未取得」は表示ラベルで、元値は空欄を保持します。',
        'BrierはKEIBA LABがレース内誤差合計のレース平均、既存評価が馬平均。同率トップの評価規則も異なります。',
        'CSVの序数順位は同確率時に馬番昇順。既存指標の同率按分は変更しません。',
        'DB接続・学習・タスク操作なし。封印2026年とStage 7保存領域は入力に含めません。',
        '入力ハッシュ、欠損件数、モデル別照合値はmanifest.json。既存成果物の上書きなし。', ''])
    report_path = safety.local_path(directory / 'report.md', root=OUTPUT)
    with report_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    csv_path.chmod(stat.S_IREAD)
    return {'status': 'ok', 'kind': 'keiba_lab_saved_development_export', 'artifact_id': directory.name,
            'races': value['races'], 'runners': value['runners'], 'models': len(MODEL_COLUMNS),
            'csv_rows': len(rows), 'csv_path': str(csv_path), 'report_path': str(report_path),
            'maximum_metric_difference': validation['maximum_metric_difference'],
            'verified_models': validation['verified_models'], **availability}


def main(argv=None):
    parser = argparse.ArgumentParser(description='ローカル保存済みStage 6の閲覧用CSV。DB・学習・Stage 7不使用。')
    parser.add_argument('--from', dest='start', default='2023-01-01')
    parser.add_argument('--to', dest='end', default='2025-09-30')
    parser.add_argument('--race-limit', dest='limit', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = export(**vars(args))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, safety.MLFailure) else 'internal_error'
        print(json.dumps({'status': 'stopped', 'reason': code, 'stage': 'keiba_lab_export'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
