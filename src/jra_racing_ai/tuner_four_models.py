"""Finite top10/top20 vs saved Boruta19/all39 development comparison."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import tuner_reduction as core
from .tuner_fixed_subsets import top_features
from .tuner_pilot import experiment as tuner
from .ml import safety
from .model_diagnostics import importance, predictions
from .objective_pilot import data
from .research import model as matrices
from .stage4.store import task_configuration
from .stage5 import metrics

VERSION = 'four_models_Gain10_Gain20_savedBoruta19_saved39_20261023_v1'
NAMES = ('A10', 'B20', 'C19', 'D39')
NEW_COUNTS = {'A10': 10, 'B20': 20}
MAX_FITS = 16
BORUTA = core.folder('552f89f1716643a39742ce84acb31cff')
PREVIOUS = core.folder('0650575a0e92404c94daad03e60d9b7b')
OFFICIAL_PDF = core.BASE / 'references/JRA-VAN/JV-Data4901.pdf'


def select_strict(records):
    """2024 race LL only; the former .002 tolerance is deliberately unused."""
    if not records or any(not math.isfinite(r['selection_metrics']['race_log_loss']) for r in records):
        raise safety.MLFailure('invalid_input')
    return min(records, key=lambda row: (row['selection_metrics']['race_log_loss'], row['name']))


def surface_group(value):
    """JV-Data4901 PDF p45, table2009; sand and jumps stay separate."""
    code = str(value)
    if code in {str(i) for i in range(10, 23)}:
        return 'turf'
    if code in ('23', '24', '25', '26', '29'):
        return 'dirt'
    if code in ('27', '28'):
        return 'sand'
    if code in {str(i) for i in range(51, 60)}:
        return 'jump'
    return 'unknown'


def distance_group(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer() or not 100 <= number <= 10000:
        return 'unknown'
    return 'up_to_1400m' if number <= 1400 else '1401_1800m' if number <= 1800 else '1801_2200m' if number <= 2200 else '2201m_and_above'


def calibration_summary(bands):
    count = sum(row['runners'] for row in bands)
    if not count:
        raise safety.MLFailure('invalid_input')
    return {'fixed_bin_ECE': sum(row['runners'] * abs(row['mean_prediction'] - row['observed_win_rate'])
                for row in bands if row['runners']) / count,
            'runners': count, 'role': 'descriptive_same_existing_probability_bins_not_new_calibration'}


class FourModelBudget(core.Budget):
    def reserve(self, kind, name):
        if kind != 'LGBM' or name not in NEW_COUNTS or self.calls >= MAX_FITS:
            raise safety.MLFailure('input_limit')
        return super().reserve(kind, name)


def prepare():
    source_plan = tuner.read(core.OLD_RUN, 'plan.json')
    old_report = tuner.read(core.OLD_RUN, 'development_report.json')
    freeze = tuner.read(core.OLD_RUN, 'candidate_freeze.json')
    if safety.digest(core.OLD_RUN / 'development_report.json', root=core.BASE) != freeze['report_sha256']:
        raise safety.MLFailure('artifact_invalid')
    d_path = core.OLD_RUN / 'models' / old_report['model_id']
    if safety.digest(d_path / 'model.json', root=core.BASE) != freeze['model_json_sha256']:
        raise safety.MLFailure('artifact_invalid')
    d_booster, d_meta = tuner.load_model(d_path)
    boruta_report = core.read(BORUTA, 'development_report.json')
    c_record = boruta_report['Boruta_result']
    c_path = BORUTA / 'models' / c_record['model_id']
    _, c_meta = core.load_model(c_path)
    if (len(c_meta['features']) != 19 or c_meta['features'] != c_record['features']
            or c_meta['spans'] != source_plan['spans'] or d_meta['spans'] != source_plan['spans']
            or c_meta['base_feature_spec'] != source_plan['feature_spec']
            or c_record['actual_train_calls'] != 8 or d_meta['actual_train_calls'] != 8):
        raise safety.MLFailure('artifact_invalid')
    ranking = importance(d_booster, d_meta)
    specs = [{'name': name, 'feature_count': count, 'features': top_features(ranking, count, d_meta['features'])}
             for name, count in NEW_COUNTS.items()]
    protected = dict(core.read(PREVIOUS, 'preservation_before.json')['private_assets'])
    for old in (PREVIOUS, BORUTA):
        for path in old.rglob('*'):
            if path.is_file():
                protected[str(path)] = safety.digest(path, root=core.BASE)
    refs = dict(source_plan['source_references'])
    for path in (core.OLD_RUN / 'plan.json', core.OLD_RUN / 'development_report.json',
                 core.OLD_RUN / 'candidate_freeze.json', d_path / 'model.json', d_path / 'booster.lgb.txt',
                 BORUTA / 'development_report.json', BORUTA / 'selected_predictions.local.csv',
                 c_path / 'model.json', c_path / 'booster.lgb.txt', OFFICIAL_PDF):
        refs[str(path)] = safety.digest(path, root=core.BASE)
    for path, sha in refs.items():
        if safety.digest(Path(path), root=core.BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    protected.update(refs)
    directory = safety.new_artifact('evaluations', root=core.ROOT)
    plan = {'version': VERSION, 'new_configurations': specs,
            'saved_models': {'C19': {'model_id': c_meta['model_id'], 'path': str(c_path), 'features': c_meta['features'],
                                    'saved_prediction_column': 'Boruta_C'},
                             'D39': {'model_id': d_meta['model_id'], 'path': str(d_path), 'features': d_meta['features'],
                                    'saved_prediction_column': 'A39'}},
            'ranking': ranking, 'base_feature_spec': source_plan['feature_spec'],
            'parameters': source_plan['parameters'], 'steps': source_plan['steps'],
            'parameter_initialization': 'original_common_initial_parameters;not_saved_best_parameters;no_warm_start',
            'spans': source_plan['spans'], 'calibration': source_plan['calibration'], 'versions': tuner.versions(),
            'source_references': refs, 'selection': {'metric': 'race_log_loss', 'period': '2024_calibration_development',
                'tolerance': 0., 'rule': 'strict_smallest_race_LL;exact_ties_use_fixed_model_name',
                'shared_temperature_and_selection_window': True, '2025_not_used_for_selection': True,
                'old_2025_aggregate_results_already_known': True},
            'budget': {'actual_fits_max': 16, 'each_Tuner_actual_fits_max': 8, 'rounds_max': 300,
                'early_stopping_rounds': 20, 'CPU_threads': 4, 'private_bytes_max': 12 * 1024**3,
                'whole_seconds': 900, 'each_Tuner_seconds': 600, 'final_refit': 0, 'automatic_retry': False,
                'C19_fits': 0, 'D39_fits': 0, 'RF_fits': 0},
            'comparison': {'period': '2025_01_09_reused_development', 'races': 2169, 'runners': 29879,
                'week_bootstrap_repetitions': 1000, 'difference': 'new_minus_reference',
                'pairings': [['A10', 'D39'], ['B20', 'D39'], ['C19', 'D39'], ['B20', 'C19']],
                'subgroups_are_descriptive_not_selection': True},
            'subgroups': {'surface_mapping_source': str(OFFICIAL_PDF), 'PDF_page': 45, 'table': '2009',
                'turf_codes': list(range(10, 23)), 'dirt_codes': [23, 24, 25, 26, 29],
                'sand_codes': [27, 28], 'jump_codes': list(range(51, 60)), 'unknown': 'retain_as_unknown',
                'distance_edges_m': [1400, 1800, 2200], 'inconsistent_race_condition': 'retain_as_inconsistent',
                'all_four_models_use_same_races_within_each_group': True},
            'database_access': False, 'external_communication': False, 'sealed_2026_used': False,
            'Stage7_results_used': False, 'H_changed': False, 'task_mutations': 0, 'ROI_computed': False}
    core.write(directory, 'plan.json', plan)
    sha = safety.digest(directory / 'plan.json', root=core.ROOT)
    core.write(directory, 'approval.json', {'source': 'human_user', 'allow_execution': True,
        'plan_sha256': sha, 'actual_fits_max': MAX_FITS, 'authorization': 'explicit_four_model_top10_top20_saved19_saved39_max16fit_request'})
    files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for sub in ('src', 'scripts', 'tests', 'docs')
             for path in (core.PROJECT / sub).rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    core.write(directory, 'preservation_before.json', {'project_files': files, 'private_assets': protected,
                                                      'tasks': task_configuration()})
    return {'status': 'prepared', 'run_id': directory.name, 'actual_fit_max': 16, 'C_D_fits': 0, 'plan_sha256': sha}


def approved(directory):
    p = core.read(directory, 'plan.json')
    a = core.read(directory, 'approval.json')
    if (p['version'] != VERSION or a.get('source') != 'human_user' or a.get('allow_execution') is not True
            or a.get('actual_fits_max') != MAX_FITS or a.get('plan_sha256') != safety.digest(directory / 'plan.json', root=core.ROOT)
            or p['selection']['tolerance'] != 0. or [x['name'] for x in p['new_configurations']] != list(NEW_COUNTS)):
        raise safety.MLFailure('explicit_execution_required')
    for spec in p['new_configurations']:
        count = NEW_COUNTS[spec['name']]
        if spec['feature_count'] != count or spec['features'] != top_features(p['ranking'], count, p['base_feature_spec']['features']):
            raise safety.MLFailure('invalid_mapping')
    return p


def frozen_probabilities(frame):
    import numpy as np
    import pandas as pd
    saved = pd.read_csv(BORUTA / 'selected_predictions.local.csv', dtype={'entry_key': str, 'date': str}, float_precision='round_trip')
    if saved.entry_key.duplicated().any() or set(saved.entry_key) != set(frame.entry_key):
        raise safety.MLFailure('invalid_identity')
    saved = saved.set_index('entry_key').loc[frame.entry_key]
    if not np.array_equal(saved.date.to_numpy(str), frame.date.to_numpy(str)) or not np.array_equal(saved.winner_label.to_numpy(int), frame.winner_label.to_numpy(int)):
        raise safety.MLFailure('invalid_identity')
    return {'C19': saved.Boruta_C.to_numpy(float), 'D39': saved.A39.to_numpy(float)}


def grouped_metrics(frame, predictions_frame):
    rows = []
    for axis, column, classifier in (('distance', 'distance', distance_group), ('surface', 'surface', surface_group)):
        groups = {}
        for race, part in frame.groupby(frame.entry_key.str[:16], sort=False):
            values = {classifier(value) for value in part[column]}
            groups[race] = next(iter(values)) if len(values) == 1 else 'inconsistent'
        tags = frame.entry_key.str[:16].map(groups)
        for group in sorted(set(tags)):
            mask = tags == group
            selected = frame.loc[mask]
            for name in NAMES:
                point = core.point(selected, predictions_frame.loc[mask, name].to_numpy(float))
                rows.append({'axis': axis, 'group': group, 'configuration': name,
                    'races': int(selected.entry_key.str[:16].nunique()), 'runners': len(selected),
                    **{key: point[key] for key in metrics.NAMES},
                    'calibration_ECE': calibration_summary(point['calibration_bands'])['fixed_bin_ECE']})
    return rows


def worker(directory):
    import numpy as np
    plan = approved(directory)
    lock = safety.read_json(core.BASE / 'research-training.lock', root=core.BASE)
    if lock.get('run_id') != directory.name or not (directory / 'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    core.write(directory, 'worker_entered.json', {'automatic_retry': False, 'C19_fits': 0, 'D39_fits': 0})
    if tuner.versions() != plan['versions']:
        raise safety.MLFailure('missing_library')
    for path, sha in plan['source_references'].items():
        if safety.digest(Path(path), root=core.BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    budget = FourModelBudget(directory, time.monotonic() + 880)
    frames = {name: data.load_part(tuner.SOURCE, name) for name in ('train', 'early_stopping', 'calibration')}
    categories = matrices.schema(frames['train'], plan['base_feature_spec'])
    full = {name: matrices.matrix(frame, plan['base_feature_spec'], categories) for name, frame in frames.items()}
    previous = core.read(BORUTA, 'development_report.json')
    original = tuner.read(core.OLD_RUN, 'development_report.json')
    c_record = dict(previous['Boruta_result'], name='C19', actual_train_calls=0,
                    historical_actual_train_calls=8, historical_fit_seconds=previous['Boruta_result']['fit_call_seconds'],
                    new_fit_seconds=0., reused_saved_model=True, model_path=plan['saved_models']['C19']['path'])
    d_bundle = tuner.load_model(Path(plan['saved_models']['D39']['path']))
    d_meta = d_bundle[1]
    d_record = dict(previous['selection_records'][0], name='D39', actual_train_calls=0,
                    historical_actual_train_calls=8, historical_fit_seconds=original['timing']['actual_lgb_train_call_seconds_total'],
                    new_fit_seconds=0., parameters=d_meta['parameters'], reused_saved_model=True,
                    model_path=plan['saved_models']['D39']['path'])
    for record in (c_record, d_record):
        q = core.predict(core.load_model(Path(record['model_path'])), frames['calibration']) if record['name'] == 'C19' else tuner.predict(*d_bundle, frames['calibration'])
        check = core.point(frames['calibration'], q)
        if any(abs(check[key] - record['selection_metrics'][key]) > 1e-12 for key in metrics.NAMES):
            raise safety.MLFailure('artifact_invalid')
    records = []
    for spec in plan['new_configurations']:
        removed = [row['feature'] for row in sorted(plan['ranking'], key=lambda row: -row['gain_rank']) if row['feature'] not in spec['features']]
        record = core.train_one(directory, plan, spec['name'], spec['features'], removed, frames, full, categories, budget)
        record.update(new_fit_seconds=record['fit_call_seconds'], historical_actual_train_calls=0,
                      historical_fit_seconds=0., reused_saved_model=False,
                      model_path=str(directory / 'models' / record['model_id']))
        records.append(record)
    records += [c_record, d_record]
    best = select_strict(records)
    core.write(directory, 'candidate_frozen_before_2025.json', {'name': best['name'], 'model_id': best['model_id'],
        'model_path': best['model_path'], 'features': best['features'], 'feature_count': best['feature_count'],
        'parameters': best['parameters'], 'temperature': best['temperature'], 'best_iteration': best['best_iteration'],
        'selection_race_log_loss': best['selection_metrics']['race_log_loss'], 'tolerance': 0.,
        'selection_uses_2024_only': True, '2025_labels_not_read_by_this_run_before_freeze': True,
        'production_adoption': False, '2026_data_used': False})
    frame = data.load_part(tuner.SOURCE, 'evaluation')
    saved = frozen_probabilities(frame)
    evaluation = frame[['entry_key', 'date', 'winner_label']].copy()
    points = {}; draws = {}; bootstrap = None; reload_delta = {}
    for record in records:
        name = record['name']
        if name in saved:
            q = saved[name]
            check = tuner.predict(*d_bundle, frame) if name == 'D39' else core.predict(core.load_model(Path(record['model_path'])), frame)
            reload_delta[name] = float(np.max(np.abs(check - q)))
            if reload_delta[name] > 1e-12:
                raise safety.MLFailure('invalid_model')
        else:
            q = core.predict(core.load_model(Path(record['model_path'])), frame)
            reload_delta[name] = record['reload_probability_max_difference']
        evaluation[name] = q
        y, p, codes, weeks = metrics.assemble(evaluation, predictions(evaluation, name))
        if bootstrap is None:
            bootstrap = metrics.WeekBootstrap(weeks, repetitions=1000)
        points[name], draws[name] = bootstrap.measure(y, p, codes)
        points[name]['calibration_bands'] = core.calibration_table(y, p)
        points[name]['calibration_summary'] = calibration_summary(points[name]['calibration_bands'])
    for name, old in (('C19', 'Boruta_C'), ('D39', 'A39')):
        if any(abs(points[name][key] - previous['evaluation_metrics'][old][key]) > 1e-12 for key in metrics.NAMES):
            raise safety.MLFailure('artifact_invalid')
    with (directory / 'predictions.local.csv').open('x', encoding='utf-8', newline='') as stream:
        evaluation.to_csv(stream, index=False)
    pairs = {a + '_minus_' + b: {key: {'difference': points[a][key] - points[b][key],
                '95_percent_ci': metrics.bounds(draws[a][key] - draws[b][key])} for key in metrics.NAMES}
             for a, b in plan['comparison']['pairings']}
    report = {'version': VERSION, 'status': 'complete', 'selection_records': records,
        'evaluation_metrics': points, 'differences': pairs, 'bootstrap': bootstrap.metadata(),
        'subgroup_metrics': grouped_metrics(frame, evaluation), 'races': 2169, 'runners': 29879,
        'overall_candidate': best['name'], 'selection_tolerance': 0., 'actual_new_fits': budget.calls,
        'new_fit_seconds': budget.fit_seconds['LGBM'], 'CPU_percent_samples': budget.cpu.observed,
        'saved_prediction_reload_differences': reload_delta, 'saved_C_D_metrics_reproduced': True,
        'C19_new_fits': 0, 'D39_new_fits': 0, 'Boruta_new_fits': 0, 'additional_refit': 0,
        '2025_reused_development_not_unused_test': True, '2025_used_for_selection': False,
        'production_adoption': False, 'H_changed': False, 'Stage7_results_used': False,
        'sealed_2026_used': False, 'database_access': False, 'external_communication': False}
    core.write(directory, 'development_report.json', report)
    return {'status': 'ok', 'actual_new_fits': budget.calls, 'selected_on_2024': best['name'], 'C_D_new_fits': 0}


def execute(directory):
    plan = approved(directory)
    with tuner.training_lock(directory):
        core.write(core.ROOT, 'execution_' + safety.digest(directory / 'plan.json', root=core.ROOT) + '.json',
                   {'run_id': directory.name, 'approval_consumed': True})
        core.write(directory, 'execution_started.json', {'utc': datetime.now(timezone.utc).isoformat(),
            'actual_fit_max': MAX_FITS, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        args = [sys.executable, '-B', str(core.PROJECT / 'scripts/run_tuner_four_models.py'), '_worker', '--run-id', directory.name]
        began = time.monotonic(); peak = 0; reason = None
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
                if time.monotonic() - began > plan['budget']['whole_seconds']:
                    reason = 'query_timeout'
                if reason:
                    core.terminate_worker(child); break
                time.sleep(.25)
            output, _ = child.communicate(timeout=10)
        try:
            result = json.loads(output)
        except ValueError:
            result = {'status': 'stopped', 'reason': 'training_failed'}
        done = {'status': 'complete' if child.returncode == 0 and result.get('status') == 'ok' and not reason else 'stopped',
            'reason': reason or result.get('reason'), 'wall_seconds': time.monotonic() - began,
            'peak_parent_and_worker_private_bytes': peak,
            'actual_new_fit_attempts': len(list(directory.glob('actual_fit_*.started.json'))), 'automatic_retry': False}
        core.write(directory, 'completion.json', done)
        return done


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute', '_worker'))
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.command == 'prepare' else (execute(core.folder(args.run_id)) if args.command == 'execute' else worker(core.folder(args.run_id)))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] in ('prepared', 'complete', 'ok') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': args.command}), flush=True)
        return 1
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': args.command}), flush=True)
        return 1
