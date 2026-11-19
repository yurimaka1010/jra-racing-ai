"""Prepare freely; real-data fitting requires a separately approved, hashed plan."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4
from unittest.mock import patch

from ..ml import safety
from ..objective_pilot import contracts as old, data as cache, model as adapters
from ..research import contracts as feature_contract, model as feature_model

BASE = old.BASE
ROOT = BASE / 'tuner-pilot'
SOURCE = BASE / 'objective-pilot/evaluations/108a75ebe25d48a9a3421516f4b0c033'
PROJECT = Path(__file__).resolve().parents[3]
STEPS = (('tune_feature_fraction', 2), ('tune_num_leaves', 2), ('tune_bagging', 2), ('tune_regularization_factors', 2))
VERSION = 'finite_official_lightgbm_tuner_20261022_v1'
LIBRARIES = ('optuna', 'optuna-integration', 'lightgbm', 'numpy', 'pandas', 'scikit-learn')


def write(directory, name, value):
    safety.write_json(directory / name, value, root=ROOT)


def read(directory, name):
    return safety.read_json(directory / name, root=ROOT)


def run_folder(identifier):
    if not re.fullmatch('[0-9a-f]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def versions():
    try:
        return {name: importlib.metadata.version(name) for name in LIBRARIES}
    except importlib.metadata.PackageNotFoundError:
        raise safety.MLFailure('missing_library') from None


def compatibility():
    from optuna_integration.lightgbm import LightGBMTuner
    import inspect
    methods = {name: 'n_trials' in inspect.signature(getattr(LightGBMTuner, name)).parameters for name, _ in STEPS}
    if not all(methods.values()):
        raise safety.MLFailure('missing_library')
    return {'status': 'ok', 'versions': versions(), 'public_methods_support_n_trials': methods,
            'implementation': 'optuna_integration.lightgbm.LightGBMTuner', 'real_data_fits': 0}


def prepare():
    compatibility()
    manifest = safety.read_json(SOURCE / 'data_manifest.json', root=BASE)
    before = safety.read_json(SOURCE / 'authorization.json', root=BASE)
    if before['proposal_sha256'] != old.PROPOSAL_HASH or manifest['sealed_2026_rows_read'] != 0:
        raise safety.MLFailure('artifact_invalid')
    p = old.plan(); refs = {}
    for part, item in manifest['parts'].items():
        if part not in old.PARTS or item['start'] < p['spans'][part]['start'] or item['end'] > p['spans'][part]['end']:
            raise safety.MLFailure('invalid_split')
        old.day(item['start']); old.day(item['end'])
        if (item['rows'], item['races']) != (p['spans'][part]['eligible_runners'], p['spans'][part]['eligible_races']):
            raise safety.MLFailure('invalid_identity')
        path = safety.local_path(SOURCE / item['file'], root=BASE, exists=True)
        if safety.digest(path, root=BASE) != item['sha256']:
            raise safety.MLFailure('artifact_invalid')
        refs[str(path)] = item['sha256']
    if set(manifest['parts']) != set(old.PARTS):
        raise safety.MLFailure('invalid_split')
    for name in ('data_manifest.json', 'fit_paired_binary_control.result.json', 'paired_binary_control.predictions.local.csv'):
        path = safety.local_path(SOURCE / name, root=BASE, exists=True)
        refs[str(path)] = safety.digest(path, root=BASE)
    result = safety.read_json(SOURCE / 'fit_paired_binary_control.result.json', root=BASE)
    if refs[str(SOURCE / 'paired_binary_control.predictions.local.csv')] != result['prediction_sha256']:
        raise safety.MLFailure('artifact_invalid')
    directory = safety.new_artifact('evaluations', root=ROOT)
    plan = {'version': VERSION, 'status': 'prepared_awaiting_separate_real_fit_approval',
        'source': str(SOURCE), 'source_references': refs, 'feature_spec': feature_contract.spec(('A', 'B', 'C', 'D')),
        'parameters': dict(p['parameters'], objective='binary', metric='binary_logloss'), 'spans': p['spans'],
        'steps': [{'method': name, 'n_trials': n} for name, n in STEPS],
        'budget': {'actual_train_calls_max': 8, 'each_train_attempt_consumes_budget': True,
                   'failed_and_pruned_attempts_count': True, 'final_refit': 0, 'rounds_max': 300,
                   'early_stopping_rounds': 20, 'tuning_seconds': 600, 'whole_worker_seconds': 1200,
                   'CPU_threads': 4, 'parent_and_worker_private_bytes_max': 12 * 1024 ** 3,
                   'automatic_retry': False, 'new_dependency_installs': 0},
        'calibration': {'grid': [.75, 1., 1.25, 2.], 'source': 'calibration_only', 'metric': 'race_log_loss'},
        'selection_metric': 'binary_logloss_on_existing_early_stopping_window',
        'comparison': {'control': 'paired_binary_control', 'control_model_id': result['model_id'],
                       'primary_metric': 'race_log_loss', 'difference': 'tuned_minus_paired_binary',
                       'reference': 'saved_H_same_roster;different_training_window', 'week_bootstrap_repetitions': 1000,
                       'development_reuse_not_independent_test': True, 'ROI_optimization': False},
        'versions': versions(), 'sealed_2026_used': False, 'Stage7_results_used': False,
        'H_changed': False, 'task_changes': 0, 'database_access': False, 'network_access': False,
        'exclusive_training_lock': str(BASE / 'research-training.lock')}
    write(directory, 'plan.json', plan)
    fingerprint = safety.digest(directory / 'plan.json', root=ROOT)
    write(directory, 'plan_commit.json', {'plan_sha256': fingerprint, 'fit_allowed': False,
          'approval_required': {'filename': 'approval.json', 'allow_real_fits': True,
                                'plan_sha256': fingerprint, 'actual_train_calls_max': 8, 'source': 'human_user'}})
    write(directory, 'preparation.json', {'status': 'ready_awaiting_approval', 'real_data_fits': 0,
          'plan_sha256': fingerprint, 'compatibility': compatibility()})
    return {'status': 'prepared', 'run_id': directory.name, 'plan_sha256': fingerprint, 'real_data_fits': 0,
            'max_actual_train_calls_if_approved': 8}


def approved(directory):
    plan = read(directory, 'plan.json'); commit = read(directory, 'plan_commit.json')
    fingerprint = safety.digest(directory / 'plan.json', root=ROOT)
    if commit['plan_sha256'] != fingerprint or plan['version'] != VERSION:
        raise safety.MLFailure('artifact_invalid')
    if not (directory / 'approval.json').is_file():
        raise safety.MLFailure('explicit_execution_required')
    approval = read(directory, 'approval.json')
    if (approval.get('allow_real_fits') is not True or approval.get('source') != 'human_user'
            or approval.get('plan_sha256') != fingerprint or approval.get('actual_train_calls_max') != 8):
        raise safety.MLFailure('explicit_execution_required')
    return plan, fingerprint


@contextmanager
def training_lock(directory):
    lock = safety.local_path(BASE / 'research-training.lock', root=BASE)
    token = uuid4().hex
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise safety.MLFailure('input_limit') from None
    try:
        json.dump({'owner': token, 'pid': os.getpid(), 'run_id': directory.name, 'purpose': VERSION}, stream)
        stream.flush(); os.fsync(stream.fileno())
        yield
    finally:
        stream.close()
        # Never steal or remove somebody else's lock.
        try:
            if json.loads(lock.read_text(encoding='utf-8'))['owner'] == token:
                lock.unlink()
        except (OSError, ValueError, KeyError):
            pass


class FitCounter:
    """Wrap the actual library train(), not Optuna trial numbers or callbacks."""
    def __init__(self, limit, deadline, record):
        self.limit = limit; self.deadline = deadline; self.record = record; self.calls = 0
        self.training_seconds = 0.

    def check(self, environment=None):
        if time.monotonic() >= self.deadline:
            raise safety.MLFailure('query_timeout')

    def wrap(self, original):
        def counted(*args, **kwargs):
            self.check()
            if self.calls >= self.limit:
                raise safety.MLFailure('input_limit')
            self.calls += 1; number = self.calls
            self.record(number, 'started', {'counts_as_attempt': True})
            state = 'failed'; began = time.monotonic()
            try:
                callbacks = list(kwargs.get('callbacks') or []) + [self.check]
                result = original(*args, **dict(kwargs, callbacks=callbacks))
                state = 'complete'
                return result
            finally:
                elapsed = time.monotonic() - began; self.training_seconds += elapsed
                self.record(number, state, {'counts_as_attempt': True, 'actual_train_call_seconds': elapsed})
        return counted


def tune(train_set, valid_set, params, *, counter, steps=STEPS, rounds=300, seconds=600):
    import lightgbm as lgb
    import optuna
    from optuna_integration.lightgbm import LightGBMTuner
    # Trial failures must not print arbitrary library exception text or row values.
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    optuna.logging.disable_default_handler()
    tuner = LightGBMTuner(dict(params), train_set, num_boost_round=rounds, valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(20, verbose=False)], time_budget=seconds,
                show_progress_bar=False, optuna_seed=42)
    # optimize.py resolves lgb.train on this same module object in installed 5.0.0.
    with patch.object(lgb, 'train', counter.wrap(lgb.train)):
        for method, n in steps:
            counter.check()
            if (method, n) not in STEPS:
                raise safety.MLFailure('invalid_input')
            getattr(tuner, method)(n_trials=n)
    return tuner.get_best_booster(), tuner.best_params, [
        {'number': t.number, 'state': t.state.name, 'value': t.value, 'parameters': t.params} for t in tuner.study.trials]


def load_model(directory):
    import lightgbm as lgb
    metadata = read(directory, 'model.json')
    tree = safety.local_path(directory / 'booster.lgb.txt', root=ROOT, exists=True)
    if (metadata.get('version') != VERSION or metadata['feature_spec'] != feature_contract.spec(('A', 'B', 'C', 'D'))
            or metadata['features'] != metadata['feature_spec']['features']
            or metadata['temperature'] not in (.75, 1., 1.25, 2.)
            or safety.digest(tree, root=ROOT) != metadata['tree_sha256']):
        raise safety.MLFailure('invalid_model')
    booster = lgb.Booster(model_str=tree.read_text(encoding='utf-8'))
    if list(booster.feature_name()) != metadata['features']:
        raise safety.MLFailure('invalid_model')
    return booster, metadata


def predict(booster, metadata, frame):
    values = booster.predict(feature_model.matrix(frame, metadata['feature_spec'], metadata['categories']), num_threads=4)
    return feature_model.normalize(frame, values, metadata['temperature'])


def worker(directory):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from ..model_diagnostics import predictions
    from ..stage5 import metrics
    plan, fingerprint = approved(directory)
    if not (directory / 'execution_started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    claim = read(ROOT, 'execution_' + fingerprint + '.json')
    lock = safety.read_json(BASE / 'research-training.lock', root=BASE)
    if claim.get('run_id') != directory.name or lock.get('run_id') != directory.name or lock.get('purpose') != VERSION:
        raise safety.MLFailure('explicit_execution_required')
    write(directory, 'worker_entered.json', {'plan_sha256': fingerprint, 'automatic_retry': False})
    if versions() != plan['versions']:
        raise safety.MLFailure('missing_library')
    for path, sha in plan['source_references'].items():
        if safety.digest(Path(path), root=BASE) != sha:
            raise safety.MLFailure('artifact_invalid')
    train = cache.load_part(SOURCE, 'train'); early = cache.load_part(SOURCE, 'early_stopping')
    cal = cache.load_part(SOURCE, 'calibration')
    if not max(train.date) < min(early.date) < min(cal.date):
        raise safety.MLFailure('invalid_split')
    spec = plan['feature_spec']; categories = feature_model.schema(train, spec)
    training = lgb.Dataset(feature_model.matrix(train, spec, categories), label=train.winner_label.to_numpy(int),
                           categorical_feature=spec['categorical_features'])
    validation = lgb.Dataset(feature_model.matrix(early, spec, categories), label=early.winner_label.to_numpy(int),
                             reference=training, categorical_feature=spec['categorical_features'])
    def ledger(number, state, details):
        write(directory, f'fit_{number:02d}.{state}.json', dict(number=number, state=state, **details))
    counter = FitCounter(8, time.monotonic() + 600, ledger)
    search_began = time.monotonic()
    booster, best, trials = tune(training, validation, plan['parameters'], counter=counter)
    search_seconds = time.monotonic() - search_began
    if not 1 <= counter.calls <= 8:
        raise safety.MLFailure('training_failed')
    raw = booster.predict(feature_model.matrix(cal, spec, categories), num_threads=4)
    configuration = {'objective': 'binary', 'temperature_candidates': plan['calibration']['grid']}
    temperature, grid = adapters.calibrate(cal, raw, configuration)
    path = safety.local_path(directory / 'models' / uuid4().hex, root=ROOT)
    path.mkdir(parents=True, exist_ok=False)
    tree = path / 'booster.lgb.txt'
    with tree.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(booster.model_to_string(num_iteration=booster.best_iteration))
    metadata = {'version': VERSION, 'model_id': path.name, 'feature_spec': spec, 'features': spec['features'],
        'categories': categories, 'input_format': feature_contract.INPUT_VERSION,
        'formula_version': feature_contract.FORMULA_VERSION, 'temperature': temperature,
        'calibration': grid, 'calibration_fixed_before_evaluation': True, 'spans': plan['spans'],
        'plan_sha256': fingerprint, 'parameters': best, 'versions': versions(),
        'best_iteration': int(booster.best_iteration), 'actual_train_calls': counter.calls, 'final_refit': 0,
        'tree_sha256': safety.digest(tree, root=ROOT), 'H_unchanged': True, 'production_adoption': False}
    write(path, 'model.json', metadata)
    loaded, meta = load_model(path)
    max_delta = float(np.max(np.abs(predict(loaded, meta, cal) - adapters.probability(cal, raw, 'binary', temperature))))
    if max_delta > 1e-12:
        raise safety.MLFailure('invalid_model')
    f = cache.load_part(SOURCE, 'evaluation')
    if min(f.date) <= max(cal.date):
        raise safety.MLFailure('invalid_split')
    baseline = pd.read_csv(SOURCE / 'paired_binary_control.predictions.local.csv', dtype={'entry_key':'string','date':'string'},float_precision='round_trip')
    if (not np.array_equal(f.entry_key, baseline.entry_key) or not np.array_equal(f.date, baseline.date)
            or not np.array_equal(f.winner_label.to_numpy(float), baseline.winner_label.to_numpy(float))):
        raise safety.MLFailure('invalid_identity')
    f['tuned'] = predict(loaded, meta, f); f['paired_binary_control'] = baseline.probability.to_numpy(float)
    with (directory / 'predictions.local.csv').open('x', encoding='utf-8', newline='') as stream:
        f[['entry_key','date','winner_label','tuned','paired_binary_control','H']].to_csv(stream,index=False)
    from ..objective_pilot.runner import calibration_table
    measured = {}; draws = {}; calibration_bands = {}; boot = None
    for name in ('paired_binary_control','tuned','H'):
        y, p, race_codes, weeks = metrics.assemble(f, predictions(f,name))
        if boot is None: boot = metrics.WeekBootstrap(weeks,repetitions=1000)
        measured[name], draws[name] = boot.measure(y,p,race_codes)
        calibration_bands[name] = calibration_table(y,p)
    differences = {name:{'tuned_minus_binary':measured['tuned'][name]-measured['paired_binary_control'][name],
                    '95_percent_ci':metrics.bounds(draws['tuned'][name]-draws['paired_binary_control'][name])} for name in metrics.NAMES}
    baseline_timing_path = SOURCE / 'fit_paired_binary_control.worker.json'
    baseline_worker = safety.read_json(baseline_timing_path, root=BASE)
    report = {'version':VERSION,'status':'complete','model_id':path.name,'plan_sha256':fingerprint,
        'actual_train_calls':counter.calls,'final_refit':0,'trials':trials,'temperature':temperature,
        'reload_probability_max_difference':max_delta,'races':int(f.entry_key.str[:16].nunique()),'runners':len(f),
        'metrics':measured,'comparison':differences,'bootstrap':boot.metadata(),
        'calibration_bands':calibration_bands,
        'timing':{'actual_lgb_train_call_seconds_total':counter.training_seconds,'search_wall_seconds':search_seconds,
                  'actual_train_scope':'lgb.train_calls_including_in_call_dataset_construction;CSV_preparation_calibration_bootstrap_excluded',
                  'search_scope':'four_tuner_stages_including_sampling_and_all_actual_train_calls',
                  'baseline_worker_wall_seconds':baseline_worker['wall_seconds'],
                  'baseline_worker_scope':'CSV_loading_feature_matrix_training_calibration_model_reload_evaluation_prediction',
                  'baseline_timing_sha256':safety.digest(baseline_timing_path,root=BASE),
                  'direct_pure_training_speed_comparison_possible':False},
        'development_reuse_not_independent_test':True,'H_is_unmatched_training_reference':True,
        'ROI_computed':False,'sealed_2026_used':False,'Stage7_results_used':False,'automatic_adoption':False}
    write(directory,'development_report.json',report)
    return {'status':'ok','actual_train_calls':counter.calls,'model_id':path.name,'real_fit_report_complete':True}


def execute(directory):
    _, fingerprint = approved(directory)  # Before lock acquisition or opening outcome data.
    from ..objective_pilot.runner import tree_memory, memory_bytes, terminate_worker
    with training_lock(directory):
        write(ROOT, 'execution_' + fingerprint + '.json', {'run_id':directory.name,'actual_train_calls_max':8,
              'same_approval_reuse_forbidden':True})
        write(directory, 'execution_started.json', {'utc':datetime.now(timezone.utc).isoformat(),'automatic_retry':False,
            'implementation_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
        started = time.monotonic(); peak = 0
        args = [sys.executable,'-B',str(PROJECT/'scripts/run_tuner_pilot.py'),'_worker','--run-id',directory.name]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with subprocess.Popen(args,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8') as child:
            reason = None
            while child.poll() is None:
                used = tree_memory(child.pid); own = memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None: break
                    reason = 'input_limit'
                else:
                    peak = max(peak,used+own)
                    if used+own > 12*1024**3: reason = 'input_limit'
                if time.monotonic()-started >= 1200: reason = 'query_timeout'
                if reason:
                    terminate_worker(child); break
                time.sleep(.25)
            output,_ = child.communicate(timeout=10)
        try:
            response = json.loads(output.strip())
        except ValueError:
            response = {'status':'stopped','reason':'training_failed'}
        result = {'status':'complete' if child.returncode==0 and response.get('status')=='ok' and not reason else 'stopped',
            'reason':reason or response.get('reason'), 'actual_train_calls':len(list(directory.glob('fit_*.started.json'))),
            'wall_seconds':time.monotonic()-started,'peak_parent_and_worker_private_bytes':peak,'automatic_retry':False}
        write(directory,'completion.json',result)
        return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Approval-gated, maximum eight actual-fit LightGBM Tuner pilot')
    parser.add_argument('command',choices=('prepare','compatibility','execute','_worker'))
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    try:
        result = prepare() if args.command=='prepare' else compatibility() if args.command=='compatibility' else (
                 execute(run_folder(args.run_id)) if args.command=='execute' else worker(run_folder(args.run_id)))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if result.get('status') in ('ok','prepared','complete') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status':'stopped','reason':error.code,'stage':args.command}),flush=True)
        return 1
    except Exception:
        print(json.dumps({'status':'stopped','reason':'internal_error','stage':args.command}),flush=True)
        return 1
