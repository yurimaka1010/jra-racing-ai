"""One 36-column ablation deleting only frozen Gain ranks 24, 28, and 30."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import tuner_reduction as core
from .ml import safety
from .model_diagnostics import importance, predictions
from .tuner_pilot import experiment as tuner
from .objective_pilot import data
from .research import model as matrices
from .stage5 import metrics
from .stage4.store import task_configuration

VERSION = 'fixed_39_drop_gain_ranks24_28_30_keep36_20261022_v1'
COUNTS = (36,)
DROP_GAIN_RANKS = (24, 28, 30)
MAX_FITS = 8
PREVIOUS = core.folder('3218d64c9bcb4c66bac5c4aed1eb2b07')


def remove_gain_ranks(ranking, original):
    if (original != list(core.features.ALL_FEATURES) or len(ranking) != len(original)
            or {row.get('feature') for row in ranking} != set(original)
            or any(type(row.get('gain_rank')) is not int for row in ranking)
            or sorted(row['gain_rank'] for row in ranking) != list(range(1, len(original) + 1))):
        raise safety.MLFailure('invalid_mapping')
    remove = {row['feature'] for row in ranking if row['gain_rank'] in DROP_GAIN_RANKS}
    names = core.subset([name for name in original if name not in remove])
    if len(remove) != 3 or len(names) != 36:
        raise safety.MLFailure('invalid_mapping')
    return names


class FixedBudget(core.Budget):
    def reserve(self, kind, name):
        if kind != 'LGBM' or self.calls >= MAX_FITS:
            raise safety.MLFailure('input_limit')
        return super().reserve(kind, name)


def prepare():
    source_plan = tuner.read(core.OLD_RUN, 'plan.json')
    source_report = tuner.read(core.OLD_RUN, 'development_report.json')
    frozen = tuner.read(core.OLD_RUN, 'candidate_freeze.json')
    model_path = core.OLD_RUN / 'models' / source_report['model_id']
    if (safety.digest(core.OLD_RUN / 'development_report.json', root=core.BASE) != frozen['report_sha256']
            or safety.digest(model_path / 'model.json', root=core.BASE) != frozen['model_json_sha256']):
        raise safety.MLFailure('artifact_invalid')
    booster, metadata = tuner.load_model(model_path)
    ranking = importance(booster, metadata)
    names = remove_gain_ranks(ranking, metadata['features'])
    removed = [x for x in sorted(ranking, key=lambda x: x['gain_rank']) if x['gain_rank'] in DROP_GAIN_RANKS]
    specs = [{'name': 'gain_drop_36', 'feature_count': len(names), 'features': names}]
    previous_protection = core.read(PREVIOUS, 'preservation_before.json')
    protected = dict(previous_protection['private_assets'])
    for path in PREVIOUS.rglob('*'):
        if path.is_file():
            protected[str(path)] = safety.digest(path, root=core.BASE)
    refs = dict(source_plan['source_references'])
    for name in ('plan.json', 'development_report.json', 'candidate_freeze.json'):
        path = core.OLD_RUN / name; refs[str(path)] = safety.digest(path, root=core.BASE)
    for name in ('model.json', 'booster.lgb.txt'):
        path = model_path / name; refs[str(path)] = safety.digest(path, root=core.BASE)
    for path, sha in refs.items():
        if safety.digest(Path(path), root=core.BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    protected.update(refs)
    directory = safety.new_artifact('evaluations', root=core.ROOT)
    plan = {'version': VERSION, 'source_model_id': metadata['model_id'], 'ranking': ranking,
        'configurations': specs, 'initial_39_gain_ranking_fixed': True, 'sequential_reduction': False,
        'removed_gain_ranks': list(DROP_GAIN_RANKS), 'removed_features': removed,
        'remaining_count': 36, 'ablation_configurations': 1,
        'base_feature_spec': source_plan['feature_spec'], 'parameters': source_plan['parameters'],
        'parameter_initialization': 'same_original_fixed_parameters_as_A39_search;no_booster_warm_start',
        'spans': source_plan['spans'], 'calibration': source_plan['calibration'], 'steps': source_plan['steps'],
        'source_references': refs, 'versions': tuner.versions(),
        'selection': {'period': '2024_calibration_development_window', 'near_equivalent_tolerance': .002,
            'authorization': 'latest_human_message_allows_0.002_tolerance',
            'rule': 'smallest_count_within_minimum_LL_plus_0.002',
            'reduced_candidate_pool': list(COUNTS), 'overall_candidate_pool': [39, *COUNTS],
            'temperature_and_count_share_2024_window': True, '2025_not_used_for_selection': True},
        'budget': {'actual_fits_max': MAX_FITS, 'fits_each_max': 8, 'rounds_max': 300, 'early_stopping_rounds': 20,
            'CPU_threads': 4, 'private_bytes_max': 12 * 1024**3, 'whole_seconds': 900,
            'final_refit': 0, 'automatic_retry': False, 'RF_fits': 0},
        'comparison': {'evaluation': '2025_01_09_reused_development', 'races': 2169, 'runners': 29879,
                       'single_rank_ablation_fixed_before_fit': True, 'week_bootstrap_repetitions': 1000,
                       'previous_sequential_results_reference_only': True},
        'database_access': False, 'external_communication': False, 'sealed_2026_used': False,
        'Stage7_results_used': False, 'H_changed': False, 'task_mutations': 0}
    core.write(directory, 'plan.json', plan)
    sha = safety.digest(directory / 'plan.json', root=core.ROOT)
    core.write(directory, 'approval.json', {'source': 'human_user', 'allow_execution': True,
        'plan_sha256': sha, 'actual_fits_max': MAX_FITS,
        'authorization': 'human_explicit_remove_only_Gain_rank24_28_30_from_saved39_Tuner_model'})
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('src', 'scripts', 'tests', 'docs')
             for p in (core.PROJECT / sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    core.write(directory, 'preservation_before.json', {'project_files': files, 'private_assets': protected,
                                                     'tasks': task_configuration()})
    return {'status': 'prepared', 'run_id': directory.name, 'actual_fit_max': MAX_FITS, 'plan_sha256': sha}


def approved(directory):
    plan = core.read(directory, 'plan.json'); approval = core.read(directory, 'approval.json')
    if (plan['version'] != VERSION or approval.get('source') != 'human_user'
            or approval.get('allow_execution') is not True or approval.get('actual_fits_max') != MAX_FITS
            or approval.get('plan_sha256') != safety.digest(directory / 'plan.json', root=core.ROOT)
            or tuple(x['feature_count'] for x in plan['configurations']) != COUNTS
            or plan.get('removed_gain_ranks') != list(DROP_GAIN_RANKS)
            or plan['configurations'][0]['name'] != 'gain_drop_36'
            or plan['configurations'][0]['features'] != remove_gain_ranks(plan['ranking'], plan['base_feature_spec']['features'])):
        raise safety.MLFailure('explicit_execution_required')
    return plan


def worker(directory):
    import numpy as np
    plan = approved(directory)
    lock = safety.read_json(core.BASE / 'research-training.lock', root=core.BASE)
    if lock.get('run_id') != directory.name or not (directory / 'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    core.write(directory, 'worker_entered.json', {'retry_forbidden': True})
    if tuner.versions() != plan['versions']:
        raise safety.MLFailure('missing_library')
    for path, sha in plan['source_references'].items():
        if safety.digest(Path(path), root=core.BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    budget = FixedBudget(directory, time.monotonic() + 880)
    frames = {name: data.load_part(tuner.SOURCE, name) for name in ('train', 'early_stopping', 'calibration')}
    categories = matrices.schema(frames['train'], plan['base_feature_spec'])
    full = {name: matrices.matrix(f, plan['base_feature_spec'], categories) for name, f in frames.items()}
    a_bundle = tuner.load_model(core.OLD_RUN / 'models' / plan['source_model_id'])
    source_report = tuner.read(core.OLD_RUN, 'development_report.json')
    records = [{'name': 'A39', 'feature_count': 39, 'model_id': plan['source_model_id'],
                'features': a_bundle[1]['features'], 'removed_features_in_order': [],
                'selection_metrics': core.point(frames['calibration'], tuner.predict(*a_bundle, frames['calibration'])),
                'temperature': a_bundle[1]['temperature'], 'best_iteration': a_bundle[1]['best_iteration'],
                'actual_train_calls': 0, 'fit_call_seconds': 0., 'configuration_wall_seconds': 0.}]
    for config in plan['configurations']:
        names = config['features']
        removed = [x['feature'] for x in sorted(plan['ranking'], key=lambda x: x['gain_rank'], reverse=True)
                   if x['feature'] not in names]
        records.append(core.train_one(directory, plan, config['name'], names, removed,
                                      frames, full, categories, budget))
    tolerance = plan['selection']['near_equivalent_tolerance']
    best = core.choose(records, tolerance); reduced = core.choose(records[1:], tolerance)
    core.write(directory, 'candidates_frozen_before_2025.json', {
        'overall_name': best['name'], 'overall_model_id': best['model_id'], 'overall_features': best['features'],
        'reduced_name': reduced['name'], 'reduced_model_id': reduced['model_id'], 'reduced_features': reduced['features'],
        'tolerance': tolerance, 'selection_only_2024': True, '2025_not_read': True})
    f = data.load_part(tuner.SOURCE, 'evaluation')
    evaluation = f[['entry_key', 'date', 'winner_label']].copy(); points = {}; draws = {}; boot = None
    max_reload = 0.
    for record in records:
        name = record['name']
        q = tuner.predict(*a_bundle, f) if name == 'A39' else core.predict(core.load_model(directory / 'models' / record['model_id']), f)
        evaluation[name] = q
        y, p, codes, weeks = metrics.assemble(evaluation, predictions(evaluation, name))
        if boot is None:
            boot = metrics.WeekBootstrap(weeks, repetitions=1000)
        points[name], draws[name] = boot.measure(y, p, codes)
        points[name]['calibration_bands'] = core.calibration_table(y, p)
        max_reload = max(max_reload, record.get('reload_probability_max_difference', 0.))
    for key in metrics.NAMES:
        if abs(points['A39'][key] - source_report['metrics']['tuned'][key]) > 1e-12:
            raise safety.MLFailure('artifact_invalid')
    with (directory / 'selected_predictions.local.csv').open('x', encoding='utf-8', newline='') as stream:
        evaluation.to_csv(stream, index=False)
    differences = {name: {key: {'difference_new_minus_A39': points[name][key] - points['A39'][key],
                        '95_percent_ci': metrics.bounds(draws[name][key] - draws['A39'][key])}
                  for key in metrics.NAMES} for name in points if name != 'A39'}
    report = {'version': VERSION, 'status': 'complete', 'selection_records': records, 'Boruta_result': None,
        'races': len(set(f.entry_key.str[:16])), 'runners': len(f), 'evaluation_metrics': points,
        'differences': differences, 'bootstrap': boot.metadata(), 'overall_candidate': best['name'],
        'reduced_candidate': reduced['name'], 'tolerance': tolerance,
        'actual_fits': budget.calls, 'fit_seconds': budget.fit_seconds['LGBM'], 'CPU_percent_samples': budget.cpu.observed,
        'max_reload_difference': max_reload, 'best_chosen_before_2025': True, '2025_reused_development': True,
        'initial_ranking_only': True, 'no_sequential_drops': True, 'Boruta_executed': False,
        'production_adoption': False, 'H_changed': False, 'sealed_2026_used': False, 'Stage7_results_used': False}
    core.write(directory, 'development_report.json', report)
    return {'status': 'ok', 'actual_fits': budget.calls, 'overall_candidate': best['name'], 'reduced_candidate': reduced['name']}


def execute(directory):
    plan = approved(directory)
    with tuner.training_lock(directory):
        core.write(core.ROOT, 'execution_' + safety.digest(directory / 'plan.json', root=core.ROOT) + '.json',
                   {'run_id': directory.name, 'approval_consumed': True})
        core.write(directory, 'execution_started.json', {'utc': datetime.now(timezone.utc).isoformat(),
                   'fit_max': MAX_FITS, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        args = [sys.executable, '-B', str(core.PROJECT / 'scripts/run_tuner_rank_ablation.py'), '_worker', '--run-id', directory.name]
        started = time.monotonic(); peak = 0; reason = None
        with subprocess.Popen(args, cwd=core.PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8') as child:
            while child.poll() is None:
                used = core.tree_memory(child.pid); own = core.memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:
                        break
                    reason = 'input_limit'
                else:
                    peak = max(peak, used + own)
                    if used + own > plan['budget']['private_bytes_max']:
                        reason = 'input_limit'
                if time.monotonic() - started > plan['budget']['whole_seconds']:
                    reason = 'query_timeout'
                if reason:
                    core.terminate_worker(child); break
                time.sleep(.25)
            output, _ = child.communicate(timeout=10)
        try:
            result = json.loads(output)
        except ValueError:
            result = {'status': 'stopped', 'reason': 'training_failed'}
        complete = {'status': 'complete' if child.returncode == 0 and result.get('status') == 'ok' and not reason else 'stopped',
                    'reason': reason or result.get('reason'), 'wall_seconds': time.monotonic() - started,
                    'peak_parent_and_worker_private_bytes': peak,
                    'actual_fit_attempts': len(list(directory.glob('actual_fit_*.started.json'))), 'automatic_retry': False}
        core.write(directory, 'completion.json', complete)
        return complete


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute', '_worker')); parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.command == 'prepare' else (execute(core.folder(args.run_id)) if args.command == 'execute' else worker(core.folder(args.run_id)))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result.get('status') in ('prepared', 'complete', 'ok') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': args.command}), flush=True); return 1
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': args.command}), flush=True); return 1
