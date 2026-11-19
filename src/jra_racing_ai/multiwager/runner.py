"""Local saved-prediction baselines and a separately gated two-family learner."""
from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ..ml import safety
from . import contracts as c
from .outcomes import rank_reference
from .probabilities import plackett_luce
from .scoring import Scores


def sample_rows(record, name, probabilities, labels, generated, source_hash, model_id=None):
    """A new schema: never overload KEIBA LAB's single-horse win_prob field."""
    for wager in c.WAGERS:
        if wager not in probabilities: continue
        values = probabilities[wager]
        order = sorted(values, key=lambda key: (-values[key], key))
        for rank, target in enumerate(order, 1):
            pieces = target.split('-')
            yield dict(zip(c.COLS, (c.SCHEMA, record['race_id'], record['date'], record['race_id'][8:10],
                record['race_id'][14:16], wager, pieces[0], pieces[1] if len(pieces) == 2 else '', values[target],
                rank, model_id or name, name, 'retrospective_saved_prediction_replay', 'saved_Stage2_5_6_and_objective_pilot',
                source_hash, generated, '', record['paid_positions'], labels[wager][target], '', '',
                'official_HR_flags_winning_targets_units_versions_unconfirmed')))


def baseline(directory):
    from .data import prepare
    if not (directory / 'data_manifest.json').exists(): prepare(directory)
    manifest = c.read(directory, 'data_manifest.json'); path = directory / manifest['file']
    if safety.digest(path, root=c.ROOT) != manifest['sha256']: raise safety.MLFailure('artifact_invalid')
    scores = defaultdict(Scores); excluded = Counter(); counts = Counter(); generated = datetime.now(timezone.utc).isoformat()
    first = None; started = time.monotonic(); row_count = 0
    info = c.read(directory, 'prediction_commit.json')['models']
    with path.open(encoding='utf-8') as stream, (directory / 'predictions.local.jsonl').open('x', encoding='utf-8', newline='\n') as detail, \
            (directory / 'keiba_lab_sample.local.csv').open('x', encoding='utf-8-sig', newline='') as sample:
        writer = csv.DictWriter(sample, fieldnames=c.COLS); writer.writeheader()
        for line in stream:
            record = json.loads(line); c.identity(record['race_id'], record['date']); counts['all_races'] += 1
            k = record['paid_positions']
            if k not in (2, 3): excluded['asof_place_paid_positions_missing'] += 1; continue
            labels, reason = rank_reference(record['finishes'], k)
            if reason: excluded[reason] += 1; continue
            counts['rank_reference_races'] += 1; counts['pay_' + str(k)] += 1
            year = record['date'][:4]; outputs = {}
            for name, strengths in record['predictions'].items():
                outputs[name] = plackett_luce(strengths, k)
            if record['legacy_place_proxy'] is not None:
                outputs['legacy_place_proxy'] = {'place': record['legacy_place_proxy']}
            for name, result in outputs.items():
                if not result: continue
                scopes = [('year_' + year, name)]
                if name in ('H_PL', 'raw_market_PL', 'ABCD_PL'):
                    scopes.append(('2023_2025_common', name))
                if year == '2025' and name != 'legacy_place_proxy': scopes.append(('2025_objectives_common', name))
                if year == '2025' and record['date'] > '2025-01-19' and 'place' in result:
                    scopes.append(('2025_legacy_place_after_fit_common', name))
                for wager in c.WAGERS:
                    if wager not in result: continue
                    for scope, model in scopes:
                        if scope == '2025_legacy_place_after_fit_common' and wager != 'place': continue
                        scores[(scope, model, wager)].add(result[wager], labels[wager])
                if first is None: first = record['race_id']
                if record['race_id'] == first:
                    base = name[:-3] if name.endswith('_PL') else name
                    model_id = info.get(base + '_' + year, info.get(base, {})).get('model_id', 'raw_market_no_fit')
                    for row in sample_rows(record, name, result, labels, generated, manifest['sha256'], model_id):
                        writer.writerow(row); row_count += 1
                safe_result = {w: result[w] for w in c.WAGERS if w in result}
                detail.write(json.dumps({'race_id': record['race_id'], 'date': record['date'], 'model': name,
                    'place_paid_positions': k, 'probabilities': safe_result, 'rank_reference_labels': labels,
                    'official_labels': None, 'label_status': 'conditional_rank_reference_only',
                    'evaluation_kind': 'retrospective_saved_prediction_replay', 'source_sha256': manifest['sha256']},
                    ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            if counts['all_races'] % 1000 == 0:
                print(json.dumps({'status': 'running', 'stage': 'no_fit_probability_baselines',
                                  'races_processed': counts['all_races'], 'real_fits': 0}), flush=True)
    measured = {}
    for (scope, model, wager), accumulator in scores.items():
        measured.setdefault(scope, {}).setdefault(model, {})[wager] = accumulator.result()
    report = {'version': c.VERSION, 'kind': 'multiwager_no_fit_baselines', 'status': 'complete',
        'counts': dict(counts), 'excluded_rank_reference_reasons': dict(excluded), 'scores': measured,
        'label_status': 'unique_top3_rank_reference;not_verified_official_HR_ticket_labels',
        'model_assumption': 'Plackett_Luce_no_dead_heats;shared_joint_distribution_not_independent_products',
        'source_prediction_count': manifest['counts']['races'], 'data_manifest_sha256': safety.digest(directory / 'data_manifest.json', root=c.ROOT),
        'new_real_fits': 0, 'sealed_2026_used': False, 'Stage7_results_used': False,
        'ROI': None, 'ROI_reason': 'official_place_wide_umaren_payouts_flags_units_revisions_not_connected',
        'price_midpoint_not_used': True, 'purchase_policy_changes': 0,
        'sample_export': {'schema': c.SCHEMA, 'rows': row_count, 'path': 'keiba_lab_sample.local.csv',
                         'existing_KEIBA_LAB_v1_1_import_ready': False},
        'wall_seconds': time.monotonic() - started}
    c.write(directory, 'baseline_report.json', report)
    lines = ['# 複勝・ワイド・馬連：保存予測による0fit基準線', '',
             f'対象{counts["all_races"]:,}レース、上位同着等を分離した順位参考評価{counts["rank_reference_races"]:,}レース。',
             'これは同時点O1の2/3着払いと確定着順による条件付きの順位参考評価です。公式HR対象・フラグ未接続のため、公式的中ラベル・仮想収益の検証完了とは扱いません。',
             '保存単勝確率からPL共通着順分布を定義。複勝価格の逆数・中央値や、独立確率の積は使いません。', '',
             '## 2023〜2025年の同一母集団', '', '|基準線|券種|レース|Log Loss|Brier|AUC|最高確率対象の的中率|', '|---|---|---:|---:|---:|---:|---:|']
    for name, wagers in measured.get('2023_2025_common', {}).items():
        for wager, m in wagers.items():
            lines.append(f'|{name}|{wager}|{m["races"]}|{m["binary_log_loss"]:.6f}|{m["brier_score"]:.6f}|{m["roc_auc"]:.6f}|{m["top_probability_target_hit_rate_percent"]:.2f}%|')
    lines += ['', '## 利用可能性と次のゲート', '', '既存H・市場・ABCDは年別の保存予測を使用。今朝の4方式は2025年だけで別集計。旧複勝モデルは2025-01-19より後の参考比較に限定。',
              '複勝直接モデルと共通着順モデルは別モジュールとして実装。実データの新規学習は最大2fit計画の確認・公式複勝ラベル接続後。',
              '払戻・返還・特払・発売なし・訂正削除は通常ラベルと混在させず、未確認は保留。実ROIは未算出。',
              'KEIBA LAB用サンプルは閲覧研究用の新スキーマで、既存の単勝CSVへ組合せを詰め込まない。公式ラベル・払戻の空欄を0で補完しない。',
              '年別・確率帯別・2025年4方式比較・旧複勝モデル共通母集団比較はbaseline_report.json。',
              '個別予測・組合せ・着順・サンプルCSVはローカル専用。チャットへ貼り付けない。']
    with (directory / 'baseline_report.md').open('x', encoding='utf-8', newline='\n') as out: out.write('\n'.join(lines) + '\n')
    return {'status': 'ok', 'kind': report['kind'], 'run_id': directory.name, 'counts': dict(counts),
            'excluded': dict(excluded), 'new_real_fits': 0, 'ROI': None}


def preserve(directory, name='preservation_after.json'):
    from ..stage4.store import task_configuration
    before = c.read(directory, 'preservation_before.json')
    changed = sum(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in before['project_files'].items())
    protected = sum(safety.digest(Path(p), root=c.BASE) != sha for p, sha in before['protected_assets'].items())
    tasks = task_configuration()
    value = {'old_project_files': len(before['project_files']), 'changed_old_files': changed,
        'changed_protected_assets': protected, 'tasks_unchanged': tasks == before['tasks'] and tasks.get('status') == 'observed',
        'sealed_2026_used': False, 'Stage7_results_used': False, 'real_data_fits': 0}
    c.write(directory, name, value)
    if changed or protected or not value['tasks_unchanged']: raise safety.MLFailure('artifact_invalid')
    return value


def plan(directory):
    path = directory / 'training_plan.json'
    if not path.exists(): c.write(directory, path.name, c.training_plan())
    sha = safety.digest(path, root=c.ROOT)
    return {'status': 'not_started', 'plan_sha256': sha, 'new_families_max': 2, 'real_fits_max': 2,
            'official_place_training_labels_available': False, 'requires_explicit_fit_confirmation': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Multiwager local research; no purchases or frozen model changes')
    parser.add_argument('command', choices=('plan', 'baseline', 'verify', 'prepare-inputs', 'train', '_worker'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--training-manifest', type=Path)
    parser.add_argument('--family', choices=c.FAMILIES)
    args = parser.parse_args(argv)
    try:
        directory = c.folder(args.run_id)
        if args.command == 'plan': result = plan(directory)
        elif args.command == 'baseline': result = baseline(directory)
        elif args.command == 'verify': result = {'status': 'ok', **preserve(directory)}
        elif args.command == 'prepare-inputs':
            from . import training, preparation
            from ..db_connection import read_only_session
            from ..win_sim_cli import connection_config
            import os
            training.approved(directory)
            password = os.environ.pop('PGPASSWORD', None)
            if not password: raise safety.MLFailure('needs_authentication')
            session = lambda: read_only_session(connection_config(3000), password, gate_timeout_ms=1500)
            result = preparation.prepare(directory, session)
        else:
            from . import training
            if args.command == 'train': result = training.execute(directory, args.training_manifest)
            else:
                if args.training_manifest is None or args.family is None: raise safety.MLFailure('invalid_input')
                result = {'status': 'ok', 'result': training.worker(directory, args.training_manifest, args.family)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': args.command}), flush=True)
        return 1
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': args.command}), flush=True)
        return 1
