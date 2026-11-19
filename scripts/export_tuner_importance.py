"""Read frozen saved trees only; export all 39 importances without fits or rows."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from jra_racing_ai.ml import safety
from jra_racing_ai.model_diagnostics import importance
from jra_racing_ai.objective_pilot import model as control
from jra_racing_ai.tuner_pilot import experiment as tuner

RUN_ID = '2ef007c34914490995b991a308333980'
ROOT = tuner.BASE / 'model-diagnostics'


def write_csv(path, rows):
    path = safety.local_path(path, root=ROOT)
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotate(booster, metadata, train):
    rows = importance(booster, metadata)
    if len(rows) != 39 or booster.current_iteration() != metadata['best_iteration']:
        raise safety.MLFailure('invalid_model')
    order = sorted(rows, key=lambda r: (-r['split_count'], r['feature_index']))
    split_ranks = {r['feature']: i + 1 for i, r in enumerate(order)}
    for row in rows:
        missing = train['feature_missing'][row['feature']]
        if type(missing) is not int or not 0 <= missing <= train['rows']:
            raise safety.MLFailure('artifact_invalid')
        row.update(model_id=metadata['model_id'],
                   gain_share_percent=row.pop('gain_share') * 100,
                   split_rank=split_ranks[row['feature']], train_missing_count=missing,
                   train_missing_percent=100 * missing / train['rows'],
                   train_all_missing=missing == train['rows'],
                   gain_zero=row['gain'] == 0, split_zero=row['split_count'] == 0)
        flags = []
        if row['train_all_missing']:
            flags.append('訓練全欠損')
        if row['gain_zero'] and row['split_zero']:
            flags.append('Gain/Split=0')
        row['status'] = ' / '.join(flags) or '使用あり'
    return sorted(rows, key=lambda r: r['gain_rank'])


def run():
    source = tuner.run_folder(RUN_ID)
    plan = tuner.read(source, 'plan.json')
    report = tuner.read(source, 'development_report.json')
    freeze = tuner.read(source, 'candidate_freeze.json')
    if safety.digest(source / 'development_report.json', root=tuner.ROOT) != freeze['report_sha256']:
        raise safety.MLFailure('artifact_invalid')
    manifest_path = tuner.SOURCE / 'data_manifest.json'
    if safety.digest(manifest_path, root=tuner.BASE) != plan['source_references'][str(manifest_path)]:
        raise safety.MLFailure('artifact_invalid')
    manifest = safety.read_json(manifest_path, root=tuner.BASE)
    train = manifest['parts']['train']
    tuned_path = source / 'models' / report['model_id']
    control_path = tuner.SOURCE / 'models' / plan['comparison']['control_model_id']
    protected = [source / n for n in ('plan.json', 'development_report.json', 'candidate_freeze.json')]
    protected += [p / n for p in (tuned_path, control_path) for n in ('model.json', 'booster.lgb.txt')]
    protected.append(manifest_path)
    before = {str(p): safety.digest(p, root=tuner.BASE) for p in protected}
    booster, metadata = tuner.load_model(tuned_path)
    old = control.load(control_path)
    if (metadata['features'] != old['metadata']['features']
            or safety.digest(tuned_path / 'model.json', root=tuner.ROOT) != freeze['model_json_sha256']):
        raise safety.MLFailure('invalid_model')
    destination = safety.new_artifact('evaluations', root=ROOT)
    models = []
    for key, title, tree, meta in (
            ('tuner', 'LightGBM Tuner 適用モデル', booster, metadata),
            ('binary_control', '従来の39特徴量 二値対照モデル', old['booster'], old['metadata'])):
        rows = annotate(tree, meta, train)
        total = sum(r['gain'] for r in rows)
        if total <= 0 or abs(sum(r['gain_share_percent'] for r in rows) - 100) > 1e-10:
            raise safety.MLFailure('artifact_invalid')
        write_csv(destination / f'{key}_feature_importance.csv', rows)
        models.append({'key': key, 'title': title, 'model_id': meta['model_id'],
                       'best_iteration': meta['best_iteration'], 'rows': rows,
                       'total_gain': total, 'total_splits': sum(r['split_count'] for r in rows)})
    old_rows = {r['feature']: r for r in models[1]['rows']}
    compared = []
    for new in models[0]['rows']:
        previous = old_rows[new['feature']]
        compared.append({'feature': new['feature'], 'tuner_gain_rank': new['gain_rank'],
                         'control_gain_rank': previous['gain_rank'],
                         'gain_rank_improvement': previous['gain_rank'] - new['gain_rank'],
                         'tuner_gain': new['gain'], 'control_gain': previous['gain'],
                         'tuner_gain_share_percent': new['gain_share_percent'],
                         'control_gain_share_percent': previous['gain_share_percent'],
                         'gain_share_change_percentage_points': new['gain_share_percent'] - previous['gain_share_percent'],
                         'tuner_split': new['split_count'], 'control_split': previous['split_count'],
                         'tuner_split_rank': new['split_rank'], 'control_split_rank': previous['split_rank'],
                         'train_all_missing': new['train_all_missing'], 'status': new['status']})
    write_csv(destination / 'importance_comparison.csv', compared)
    bundle = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'feature_count': 39,
              'models': models, 'source_files_sha256': before,
              'actual_new_fits': 0, 'Boruta_executed': False, 'database_connections': 0,
              'individual_rows_or_predictions_read': False,
              'interpretation': 'Saved-tree training importance, not causal or held-out predictive contribution; calibration changes no trees.',
              'comparison_caveat': 'Gain totals and split counts depend on tree size/rounds; compare shares and ranks with this limitation.'}
    safety.write_json(destination / 'importance_summary.json', bundle, root=ROOT)
    args = [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '-NoProfile',
            '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
            str(PROJECT / 'scripts/render_tuner_importance.ps1'), '-BundlePath',
            str(destination / 'importance_summary.json')]
    process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding='utf-8', timeout=90, check=False)
    if process.returncode or json.loads(process.stdout).get('status') != 'ok':
        raise safety.MLFailure('io_failed')
    for model in models:
        with (destination / f'{model["key"]}_feature_importance.csv').open(encoding='utf-8-sig', newline='') as stream:
            back = list(csv.DictReader(stream))
        if len(back) != 39 or any(float(a['gain']) != b['gain'] or int(a['split_count']) != b['split_count']
                                  or float(a['gain_share_percent']) != b['gain_share_percent']
                                  for a, b in zip(back, model['rows'])):
            raise safety.MLFailure('artifact_invalid')
    if any(safety.digest(Path(p), root=tuner.BASE) != sha for p, sha in before.items()):
        raise safety.MLFailure('artifact_invalid')
    validation = {'status': 'ok', 'all_39_features_present_each_model': True,
                  'csv_full_precision_round_trip': True, 'source_files_unchanged': len(before),
                  'actual_new_fits': 0, 'Boruta_executed': False,
                  'files': {p.name: safety.digest(p, root=ROOT) for p in destination.iterdir() if p.is_file()}}
    safety.write_json(destination / 'validation.json', validation, root=ROOT)
    with (destination / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# 保存済み39特徴量モデルの重要度\n\n'
                     'tuner_gain.png / tuner_split.png が今回のTunerモデル、binary_control_gain.png / binary_control_split.png が従来対照です。\n\n'
                     '両モデルとも39項目すべてを表示。Gain順・Split順は別々に降順、同値は保存時の特徴量順です。'
                     'CSVはUTF-8 BOM付きで、元の特徴量名と丸め前の数値を保持します。importance_comparison.csv で順位・構成比変化を比較できます。\n\n'
                     '訓練全欠損は既存の訓練集計によるフラグです。ゼロ重要度を独立して表示し、全欠損と混同しません。\n\n'
                     'Gainは保存木での分割による訓練損失減少の合計、Splitは分割回数です。因果効果や未使用データでの改善量ではありません。'
                     '相関する特徴量では重要度が分散し、全欠損以外の0重要度も無価値の証明にはなりません。'
                     'モデル間で木数・木の大きさが異なるため、絶対Gain・Splitの単純比較に注意し、構成比と順位を併記しています。'
                     '温度較正は木の重要度を変更しません。新規学習・Boruta・DB接続・個別予測読込みは0です。\n')
    return {'status': 'ok', 'output_dir': str(destination), 'model_ids': [m['model_id'] for m in models],
            'features_per_model': 39, 'PNG_count': 4, 'CSV_count': 3, 'new_fits': 0,
            'model_files_unchanged': True}


if __name__ == '__main__':
    try:
        print(json.dumps(run(), ensure_ascii=False, allow_nan=False))
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': 'importance_visualization'}))
        raise SystemExit(1)
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': 'importance_visualization'}))
        raise SystemExit(1)
