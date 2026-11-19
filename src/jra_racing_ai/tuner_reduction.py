"""Finite backward reduction and training-only Boruta; frozen assets are read-only."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch
from uuid import uuid4

from .ml import safety
from . import boruta_pilot
from .tuner_pilot import experiment as tuner
from .objective_pilot import data, model as adapters
from .objective_pilot.runner import calibration_table, tree_memory, memory_bytes, terminate_worker
from .research import contracts as features, model as matrices

BASE = tuner.BASE
ROOT = BASE / 'tuner-reduction'
PROJECT = tuner.PROJECT
OLD_RUN = tuner.run_folder('2ef007c34914490995b991a308333980')
VERSION = 'finite_native_feature_reduction_20261022_v1'
REMOVE = ('b_previous_condition', 'b_grade', 'b_previous_grade', 'b_weight_type',
          'going', 'previous_time_seconds', 'c_previous_additional_margin_present')
COUNTS = tuple(range(32, 23, -1))
MAX_FITS = 100


def write(directory, name, value):
    safety.write_json(directory / name, value, root=ROOT)


def read(directory, name):
    return safety.read_json(directory / name, root=ROOT)


def folder(identifier):
    if not re.fullmatch('[0-9a-f]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def subset(names):
    original = list(features.ALL_FEATURES)
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or names != [n for n in original if n in names]):
        raise safety.MLFailure('invalid_mapping')
    return names


def next_remove(names, gains):
    subset(names)
    if set(gains) != set(names) or any(not isinstance(v, (int, float)) or v < 0 or not __import__('math').isfinite(v) for v in gains.values()):
        raise safety.MLFailure('invalid_input')
    return min(names, key=lambda n: (gains[n], names.index(n)))


def choose(records, tolerance=.002):
    if not records or tolerance < 0:
        raise safety.MLFailure('invalid_input')
    minimum = min(r['selection_metrics']['race_log_loss'] for r in records)
    eligible = [r for r in records if r['selection_metrics']['race_log_loss'] <= minimum + tolerance]
    return min(eligible, key=lambda r: (r['feature_count'], r['selection_metrics']['race_log_loss'], r['name']))


def prepare():
    from .stage4.store import task_configuration
    source_plan = tuner.read(OLD_RUN, 'plan.json')
    freeze = tuner.read(OLD_RUN, 'candidate_freeze.json')
    report = tuner.read(OLD_RUN, 'development_report.json')
    if (safety.digest(OLD_RUN / 'development_report.json', root=BASE) != freeze['report_sha256']
            or safety.digest(OLD_RUN / 'models' / report['model_id'] / 'model.json', root=BASE) != freeze['model_json_sha256']):
        raise safety.MLFailure('artifact_invalid')
    refs = dict(source_plan['source_references'])
    for path in (OLD_RUN / 'plan.json', OLD_RUN / 'candidate_freeze.json', OLD_RUN / 'development_report.json',
                 *list((OLD_RUN / 'models' / report['model_id']).iterdir())):
        if path.is_file():
            refs[str(path)] = safety.digest(path, root=BASE)
    for path, sha in refs.items():
        if safety.digest(Path(path), root=BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    directory = safety.new_artifact('evaluations', root=ROOT)
    plan = {'version': VERSION, 'source_tuner_run': str(OLD_RUN), 'source_references': refs,
            'base_feature_spec': features.spec(('A', 'B', 'C', 'D')), 'initial_remove_in_order': list(REMOVE),
            'backward_counts': list(COUNTS), 'minimum_feature_count': 24,
            'parameters': source_plan['parameters'], 'spans': source_plan['spans'], 'steps': source_plan['steps'],
            'categorical_policy': 'original_train_only_dictionary_and_native_LightGBM_categories;subset_after_shared_matrix',
            'calibration': source_plan['calibration'],
            'selection': {'period': 'existing_2024_calibration_development_window', 'temperature_and_feature_count_share_window': True,
                          'not_independent_calibration_assessment': True, 'primary': 'race_log_loss', 'near_equivalent_tolerance': .002,
                          'rule': 'smallest_feature_count_within_0.002_of_minimum_including_frozen_A39',
                          'evaluation_2025_not_read_until_count_and_Boruta_candidate_frozen': True,
                          'backward_drop': 'saved_tree_lowest_Gain_then_original_feature_order;no_outcome_based_drop'},
            'Boruta': {'max_forest_fits': 20, 'max_iter': 21, 'n_estimators': 64, 'max_depth': 5, 'n_jobs': 4,
                       'random_state': 42, 'perc': 100, 'alpha': .05, 'two_step': True,
                       'encoding': 'reuse_train_only_one_hot_and_numeric_medians;exclude_all_missing',
                       'native_feature_adoption': 'at_least_one_confirmed_encoded_column_selects_original_source_feature;then_native_LGBM',
                       'tentative_promoted': False, 'if_zero': 'no_C_fit;record_pending;no_parameter_retry',
                       'parameter_adjustments_after_results': False},
            'budget': {'LGBM_backward_max': 72, 'RF_max': 20, 'LGBM_Boruta_max': 8, 'actual_fits_max': MAX_FITS,
                       'each_Tuner_actual_fit_max': 8, 'rounds_max': 300, 'early_stopping_rounds': 20,
                       'CPU_threads': 4, 'private_bytes_max': 12 * 1024 ** 3, 'whole_seconds': 1800,
                       'each_Tuner_seconds': 600, 'Boruta_seconds': 600, 'final_refit': 0,
                       'failed_attempt_counts': True, 'automatic_retry': False,
                       'CPU_policy': 'measured_GetSystemTimes;before_each_fit_wait_if_above_85_percent;max90seconds',
                       'exclusive_training_lock': str(BASE / 'research-training.lock')},
            'evaluation': {'period': '2025_01_09_reused_development_not_unused_test',
                           'only_models': ['A39', 'mandatory_32', 'frozen_best_B', 'Boruta_C_if_available'],
                           'main_comparison': 'B_minus_A_and_C_minus_A', 'week_bootstrap_repetitions': 1000,
                           'no_feedback_to_selection': True, 'no_ROI': True},
            'versions': dict(tuner.versions(), Boruta=__import__('importlib.metadata', fromlist=['version']).version('Boruta')),
            'sealed_2026_used': False, 'Stage7_results_used': False, 'H_changed': False, 'task_changes': 0,
            'database_access': False, 'network_access': False}
    write(directory, 'plan.json', plan)
    sha = safety.digest(directory / 'plan.json', root=ROOT)
    write(directory, 'approval.json', {'source': 'human_user', 'authorization': 'latest_explicit_finite_backward_reduction_then_Boruta_execution_request',
                                      'plan_sha256': sha, 'actual_fits_max': MAX_FITS, 'allow_execution': True})
    old_snapshot = tuner.read(OLD_RUN, 'preservation_before.json')
    protected = dict(old_snapshot['private_assets'])
    protected.update(refs)
    project_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('src', 'scripts', 'tests', 'docs')
                     for p in (PROJECT / sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(directory, 'preservation_before.json', {'project_files': project_files, 'private_assets': protected,
                                                'tasks': task_configuration()})
    return {'status': 'prepared', 'run_id': directory.name, 'plan_sha256': sha, 'actual_fit_max': MAX_FITS}


def approved(directory):
    p = read(directory, 'plan.json'); a = read(directory, 'approval.json')
    if (p['version'] != VERSION or a.get('source') != 'human_user' or a.get('allow_execution') is not True
            or a.get('actual_fits_max') != MAX_FITS or a['plan_sha256'] != safety.digest(directory / 'plan.json', root=ROOT)):
        raise safety.MLFailure('explicit_execution_required')
    return p


class CPU:
    def __init__(self):
        self.previous = self.sample(); self.observed = []

    @staticmethod
    def sample():
        import ctypes as ct
        values = [ct.c_ulonglong() for _ in range(3)]
        if not ct.windll.kernel32.GetSystemTimes(*(ct.byref(v) for v in values)):
            raise safety.MLFailure('input_limit')
        return tuple(v.value for v in values)

    def percent(self):
        now = self.sample(); old = self.previous; self.previous = now
        total = (now[1] - old[1]) + (now[2] - old[2])
        used = 100 * (1 - (now[0] - old[0]) / total) if total > 0 else 0.
        value = max(0., min(100., used)); self.observed.append(value)
        return value


class Budget:
    def __init__(self, directory, deadline):
        self.directory = directory; self.deadline = deadline; self.calls = 0; self.cpu = CPU()
        self.kind_counts = {'LGBM': 0, 'RF': 0}; self.fit_seconds = {'LGBM': 0., 'RF': 0.}

    def check(self):
        if time.monotonic() >= self.deadline:
            raise safety.MLFailure('query_timeout')

    def reserve(self, kind, name):
        self.check(); waited = time.monotonic()
        while self.cpu.percent() > 85:
            if time.monotonic() - waited > 90:
                raise safety.MLFailure('input_limit')
            time.sleep(2); self.check()
        limit = 20 if kind == 'RF' else 80
        if self.calls >= MAX_FITS or self.kind_counts[kind] >= limit:
            raise safety.MLFailure('input_limit')
        self.calls += 1; self.kind_counts[kind] += 1
        number = self.calls
        write(self.directory, f'actual_fit_{number:03d}.started.json',
              {'number': number, 'kind': kind, 'name': name, 'failed_attempt_consumes_budget': True})
        return number

    def finished(self, number, kind, state, seconds):
        self.fit_seconds[kind] += seconds
        write(self.directory, f'actual_fit_{number:03d}.{state}.json',
              {'number': number, 'kind': kind, 'state': state, 'seconds': seconds})


def save_model(directory, booster, metadata):
    path = directory / 'models' / uuid4().hex
    safety.local_path(path, root=ROOT).mkdir(parents=True, exist_ok=False)
    tree = path / 'booster.lgb.txt'
    with tree.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(booster.model_to_string(num_iteration=booster.best_iteration))
    metadata.update(version=VERSION, model_id=path.name, tree_sha256=safety.digest(tree, root=ROOT))
    write(path, 'model.json', metadata)
    return path


def load_model(path):
    import lightgbm as lgb
    m = read(path, 'model.json'); subset(m['features'])
    if (m['version'] != VERSION or m['base_feature_spec'] != features.spec(('A', 'B', 'C', 'D'))
            or m['temperature'] not in (.75, 1., 1.25, 2.)
            or safety.digest(path / 'booster.lgb.txt', root=ROOT) != m['tree_sha256']):
        raise safety.MLFailure('invalid_model')
    b = lgb.Booster(model_str=(path / 'booster.lgb.txt').read_text(encoding='utf-8'))
    if b.feature_name() != m['features'] or b.current_iteration() != m['best_iteration']:
        raise safety.MLFailure('invalid_model')
    return b, m


def predict(bundle, frame):
    b, m = bundle
    x = matrices.matrix(frame, m['base_feature_spec'], m['categories']).loc[:, m['features']]
    return matrices.normalize(frame, b.predict(x, num_threads=4), m['temperature'])


def point(frame, probabilities):
    from .stage5 import metrics
    if any(not '2022-01-01' <= str(d) <= '2025-09-30' for d in frame.date):
        raise safety.MLFailure('invalid_split')
    f = frame[['entry_key', 'date', 'winner_label']].copy(); f['p'] = probabilities
    # The 2024 selection window is deliberately not routed through 2025 evaluation guards.
    import numpy as np
    y = f.winner_label.to_numpy(int); p = f.p.to_numpy(float)
    _, codes = np.unique(f.entry_key.str[:16].to_numpy(str), return_inverse=True)
    stats = metrics.statistics(y, p, codes, np.zeros(len(f), dtype=int), 1)[0]
    return {'race_log_loss': float(stats[2] / stats[0]), 'horse_log_loss': float(stats[3] / stats[1]),
            'brier_score': float(stats[4] / stats[1]), 'roc_auc': metrics.auc(y, p),
            'within_race_auc': float(stats[6] / stats[0]), 'top1_accuracy_percent': float(100 * stats[5] / stats[0]),
            'calibration_bands': calibration_table(y, p)}


def train_one(directory, plan, name, names, removed, frames, full_matrices, categories, budget):
    import lightgbm as lgb
    import numpy as np
    subset(names)
    out = directory / 'configurations' / name
    safety.local_path(out, root=ROOT).mkdir(parents=True, exist_ok=False)
    cat_names = [n for n in names if n in features.CATEGORIES]
    training = lgb.Dataset(full_matrices['train'].loc[:, names], label=frames['train'].winner_label.to_numpy(int), categorical_feature=cat_names)
    early = lgb.Dataset(full_matrices['early_stopping'].loc[:, names], label=frames['early_stopping'].winner_label.to_numpy(int),
                        reference=training, categorical_feature=cat_names)
    started = time.monotonic(); reservations = {}
    def ledger(number, state, details):
        if state == 'started':
            reservations[number] = budget.reserve('LGBM', name)
        else:
            budget.finished(reservations[number], 'LGBM', state, details['actual_train_call_seconds'])
        write(out, f'trial_{number:02d}.{state}.json', details)
    counter = tuner.FitCounter(8, min(budget.deadline, time.monotonic() + 600), ledger)
    b, params, trials = tuner.tune(training, early, plan['parameters'], counter=counter)
    raw = b.predict(full_matrices['calibration'].loc[:, names], num_threads=4)
    temperature, grid = adapters.calibrate(frames['calibration'], raw,
                   {'objective': 'binary', 'temperature_candidates': plan['calibration']['grid']})
    meta = {'name': name, 'features': names, 'base_feature_spec': plan['base_feature_spec'], 'categories': categories,
            'input_format': features.INPUT_VERSION, 'formula_version': features.FORMULA_VERSION,
            'removed_features_in_order': removed, 'parameters': params, 'best_iteration': int(b.best_iteration),
            'temperature': temperature, 'temperature_grid': grid, 'spans': plan['spans'],
            'calibration_and_count_selection': '2024_development_selection_not_fresh_test',
            'automatic_adoption': False, 'actual_train_calls': counter.calls}
    model_path = save_model(directory, b, meta)
    bundle = load_model(model_path)
    probabilities = predict(bundle, frames['calibration'])
    delta = float(np.max(np.abs(probabilities - matrices.normalize(frames['calibration'], raw, temperature))))
    if delta > 1e-12:
        raise safety.MLFailure('invalid_model')
    gain = b.feature_importance(importance_type='gain'); split = b.feature_importance(importance_type='split')
    measured = point(frames['calibration'], probabilities)
    result = {'name': name, 'model_id': model_path.name, 'feature_count': len(names), 'features': names,
              'removed_features_in_order': removed, 'temperature': temperature, 'temperature_grid': grid,
              'best_iteration': meta['best_iteration'], 'parameters': params, 'actual_train_calls': counter.calls,
              'fit_call_seconds': counter.training_seconds, 'configuration_wall_seconds': time.monotonic() - started,
              'selection_metrics': measured, 'reload_probability_max_difference': delta, 'trials': trials,
              'importance': [{'feature': n, 'gain': float(g), 'split_count': int(s)} for n, g, s in zip(names, gain, split)]}
    write(out, 'result.json', result)
    return result


def boruta_select(directory, train, budget):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    boruta_pilot.compatibility()
    preprocessor = boruta_pilot.make_preprocessor(train)
    x = boruta_pilot.transform(train, preprocessor).to_numpy()
    forest = RandomForestClassifier(n_estimators=64, max_depth=5, n_jobs=4, random_state=42)
    selector = BorutaPy(forest, n_estimators=64, max_iter=21, perc=100, alpha=.05, two_step=True, random_state=42, verbose=0)
    original = forest.fit; calls = 0; began = time.monotonic()
    def counted(*args, **kwargs):
        nonlocal calls
        budget.check()
        if calls >= 20 or time.monotonic() - began >= 600:
            raise safety.MLFailure('input_limit')
        number = budget.reserve('RF', 'Boruta'); calls += 1; start = time.monotonic(); state = 'failed'
        try:
            result = original(*args, **kwargs)
            if time.monotonic() - began >= 600:
                raise safety.MLFailure('query_timeout')
            state = 'complete'; return result
        finally:
            budget.finished(number, 'RF', state, time.monotonic() - start)
    with patch.object(forest, 'fit', counted):
        selector.fit(x, train.winner_label.to_numpy(int))
    selection = boruta_pilot.result_masks(selector, preprocessor)
    confirmed = set(selection['confirmed_source_features'])
    names = [n for n in features.ALL_FEATURES if n in confirmed]
    selection.update(actual_forest_fits=calls, selection_seconds=time.monotonic() - began,
                     selected_native_features=names, weak_promoted=False,
                     native_mapping='whole_source_feature_if_any_encoded_level_confirmed',
                     if_empty='no_C_training;parameter_adjustment_requires_new_fixed_experiment')
    write(directory, 'boruta_preprocessor.json', preprocessor)
    write(directory, 'boruta_selection.json', selection)
    return names, selection


def worker(directory):
    import numpy as np
    import pandas as pd
    from .model_diagnostics import predictions
    from .stage5 import metrics
    plan = approved(directory)
    lock = safety.read_json(BASE / 'research-training.lock', root=BASE)
    if lock.get('run_id') != directory.name or not (directory / 'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    write(directory, 'worker_entered.json', {'automatic_retry': False})
    for path, sha in plan['source_references'].items():
        if safety.digest(Path(path), root=BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    budget = Budget(directory, time.monotonic() + 1780)
    frames = {name: data.load_part(tuner.SOURCE, name) for name in ('train', 'early_stopping', 'calibration')}
    if not max(frames['train'].date) < min(frames['early_stopping'].date) < min(frames['calibration'].date):
        raise safety.MLFailure('invalid_split')
    categories = matrices.schema(frames['train'], plan['base_feature_spec'])
    full = {name: matrices.matrix(f, plan['base_feature_spec'], categories) for name, f in frames.items()}
    old_report = tuner.read(OLD_RUN, 'development_report.json')
    a_bundle = tuner.load_model(OLD_RUN / 'models' / old_report['model_id'])
    a_prob = tuner.predict(*a_bundle, frames['calibration'])
    a_result = {'name': 'A39', 'model_id': old_report['model_id'], 'feature_count': 39,
                'features': list(features.ALL_FEATURES), 'removed_features_in_order': [],
                'selection_metrics': point(frames['calibration'], a_prob), 'temperature': a_bundle[1]['temperature'],
                'best_iteration': a_bundle[1]['best_iteration'], 'actual_train_calls': 0,
                'fit_call_seconds': 0., 'configuration_wall_seconds': 0., 'previously_frozen_model_reused': True}
    records = [a_result]; removed = list(REMOVE); names = [n for n in features.ALL_FEATURES if n not in removed]
    for count in COUNTS:
        budget.check()
        if len(names) != count:
            raise safety.MLFailure('invalid_identity')
        result = train_one(directory, plan, f'backward_{count}', names.copy(), removed.copy(), frames, full, categories, budget)
        records.append(result)
        if count == 32:
            write(directory, 'first_39_to_32.json', {'status': 'complete', 'period': '2024_selection_window',
                  'A39': a_result['selection_metrics'], 'B32': result['selection_metrics'],
                  'difference_B32_minus_A39': result['selection_metrics']['race_log_loss'] - a_result['selection_metrics']['race_log_loss'],
                  'actual_new_fits': result['actual_train_calls'], '2025_evaluation_not_read': True})
        if count > min(COUNTS):
            drop = next_remove(names, {r['feature']: r['gain'] for r in result['importance']})
            removed.append(drop); names.remove(drop)
    best = choose(records, plan['selection']['near_equivalent_tolerance'])
    write(directory, 'backward_candidate_freeze.json', {'best_name': best['name'], 'best_model_id': best['model_id'],
          'features': best['features'], 'feature_count': best['feature_count'],
          'selection_race_log_loss': best['selection_metrics']['race_log_loss'],
          'selection_only_2024': True, 'evaluation_2025_not_read': True, 'tolerance': .002})
    c_names, selection = boruta_select(directory, frames['train'], budget)
    c_result = None
    if c_names:
        c_result = train_one(directory, plan, 'Boruta_C', c_names, [n for n in features.ALL_FEATURES if n not in c_names],
                             frames, full, categories, budget)
    write(directory, 'all_candidates_frozen_before_evaluation.json', {'A': a_result['model_id'], 'B': best['model_id'],
          'mandatory_32': records[1]['model_id'], 'C': c_result['model_id'] if c_result else None,
          'C_features': c_names, 'evaluation_2025_not_read': True, 'no_later_selection': True})
    # First and only read of evaluation labels occurs after all candidate selection.
    f = data.load_part(tuner.SOURCE, 'evaluation')
    selected = {r['name']: r for r in (a_result, records[1], best, c_result) if r is not None}
    measured = {}; draws = {}; boot = None; evaluation = f[['entry_key', 'date', 'winner_label']].copy()
    for name, record in selected.items():
        p = tuner.predict(*a_bundle, f) if name == 'A39' else predict(load_model(directory / 'models' / record['model_id']), f)
        evaluation[name] = p
        y, q, codes, weeks = metrics.assemble(evaluation, predictions(evaluation, name))
        if boot is None:
            boot = metrics.WeekBootstrap(weeks, repetitions=1000)
        measured[name], draws[name] = boot.measure(y, q, codes)
        measured[name]['calibration_bands'] = calibration_table(y, q)
    with (directory / 'selected_predictions.local.csv').open('x', encoding='utf-8', newline='') as stream:
        evaluation.to_csv(stream, index=False)
    differences = {name: {key: {'difference_new_minus_A39': measured[name][key] - measured['A39'][key],
                        '95_percent_ci': metrics.bounds(draws[name][key] - draws['A39'][key])}
                  for key in metrics.NAMES} for name in selected if name != 'A39'}
    report = {'version': VERSION, 'status': 'complete', 'races': int(f.entry_key.str[:16].nunique()), 'runners': len(f),
              'selection_records': records, 'best_backward': best['name'], 'Boruta_result': c_result,
              'Boruta_selection_counts': {'confirmed_encoded': len(selection['confirmed_columns']),
                'tentative_encoded': len(selection['tentative_columns']), 'native_features': len(c_names),
                'RF_fit_calls': selection['actual_forest_fits']},
              'evaluation_metrics': measured, 'differences': differences, 'bootstrap': boot.metadata(),
              'budget': {'actual_fits': budget.calls, 'kind_counts': budget.kind_counts,
                         'fit_seconds': budget.fit_seconds, 'CPU_percent_samples': budget.cpu.observed},
              'best_chosen_before_2025': True, '2025_reused_development': True, 'production_adoption': False,
              'H_changed': False, 'sealed_2026_used': False, 'Stage7_results_used': False, 'ROI_computed': False}
    write(directory, 'development_report.json', report)
    return {'status': 'ok', 'actual_fits': budget.calls, 'best_backward': best['name'], 'C_feature_count': len(c_names)}


def execute(directory):
    p = approved(directory)
    with tuner.training_lock(directory):
        write(ROOT, 'execution_' + safety.digest(directory / 'plan.json', root=ROOT) + '.json',
              {'run_id': directory.name, 'approval_consumed': True, 'automatic_retry': False})
        write(directory, 'execution_started.json', {'utc': datetime.now(timezone.utc).isoformat(),
              'actual_fit_max': MAX_FITS, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        command = [sys.executable, '-B', str(PROJECT / 'scripts/run_tuner_reduction.py'), '_worker', '--run-id', directory.name]
        began = time.monotonic(); peak = 0; reason = None
        with subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8') as child:
            while child.poll() is None:
                used = tree_memory(child.pid); own = memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:
                        break
                    reason = 'input_limit'
                else:
                    peak = max(peak, used + own)
                    if used + own > p['budget']['private_bytes_max']:
                        reason = 'input_limit'
                if time.monotonic() - began > p['budget']['whole_seconds']:
                    reason = 'query_timeout'
                if reason:
                    terminate_worker(child); break
                time.sleep(.25)
            output, _ = child.communicate(timeout=10)
        try:
            result = json.loads(output)
        except ValueError:
            result = {'status': 'stopped', 'reason': 'training_failed'}
        completion = {'status': 'complete' if child.returncode == 0 and result.get('status') == 'ok' and not reason else 'stopped',
                      'reason': reason or result.get('reason'), 'wall_seconds': time.monotonic() - began,
                      'peak_parent_and_worker_private_bytes': peak,
                      'actual_fit_attempts': len(list(directory.glob('actual_fit_*.started.json'))), 'automatic_retry': False}
        write(directory, 'completion.json', completion)
        return completion


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute', '_worker'))
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.command == 'prepare' else (
            execute(folder(args.run_id)) if args.command == 'execute' else worker(folder(args.run_id)))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result.get('status') in ('prepared', 'complete', 'ok') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': args.command}), flush=True)
        return 1
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': args.command}), flush=True)
        return 1
