"""Approval-bound Optuna search of all eight parameters, followed by one refit."""
from pathlib import Path
from uuid import uuid4
import json
import os
import subprocess
import sys
import time

from ..ml import safety
from ..research import model as matrices
from ..objective_pilot import model as adapters
from ..tuner_pilot import experiment as prior_tuner
from ..d39_generation.learning import point
from . import contracts as c, data


def trial_parameters(trial, plan):
    params = dict(plan['parameters'])
    # LightGBM aliases must not silently override the tuned value.
    for key in ('min_data_in_leaf', 'min_data', 'num_iterations', 'num_boost_round', 'n_estimators'):
        params.pop(key, None)
    space = plan['search']
    for key in ('lambda_l1', 'lambda_l2'):
        params[key] = trial.suggest_float(key, *space[key], log=True)
    for key in ('num_leaves', 'bagging_freq', 'min_child_samples'):
        params[key] = trial.suggest_int(key, *space[key])
    for key in ('feature_fraction', 'bagging_fraction'):
        params[key] = trial.suggest_float(key, *space[key])
    low, high, step = space['num_iterations']
    rounds = trial.suggest_int('num_iterations', low, high, step=step)
    return params, rounds


def load(path):
    import lightgbm as lgb
    m = c.read(path / 'model.json')
    if (m.get('version') != c.VERSION or m.get('feature_spec') != c.spec()
            or m.get('features') != c.spec()['features']
            or m.get('temperature') not in (.75, 1., 1.25, 2.)
            or c.sha(path / 'booster.lgb.txt') != m['tree_sha256']):
        raise safety.MLFailure('invalid_model')
    booster = lgb.Booster(model_str=(path / 'booster.lgb.txt').read_text(encoding='utf-8'))
    if booster.feature_name() != m['features'] or booster.current_iteration() != m['best_iteration']:
        raise safety.MLFailure('invalid_model')
    return booster, m


def predict(bundle, frame, *, raw=False):
    booster, meta = bundle
    values = booster.predict(data.matrix(frame, meta['feature_spec'], meta['categories']), num_threads=4)
    return values if raw else matrices.normalize(frame, values, meta['temperature'])


def worker(directory):
    import lightgbm as lgb
    import numpy as np
    import optuna
    from .. import tuner_reduction as resource
    p = c.approved(directory)
    if (not (directory / 'execution_started.json').exists()
            or c.read(c.BASE / 'research-training.lock').get('run_id') != directory.name):
        raise safety.MLFailure('explicit_execution_required')
    c.write(directory / 'worker_entered.json', {'started_at': c.clock().isoformat(), 'automatic_retry': False})
    if prior_tuner.versions() != p['versions']:
        raise safety.MLFailure('missing_library')
    frames = {n: data.load_part(directory, n) for n in ('train', 'early_stopping', 'final_train', 'calibration')}
    if not max(frames['train'].date) < min(frames['early_stopping'].date) <= max(frames['final_train'].date) < min(frames['calibration'].date):
        raise safety.MLFailure('invalid_split')
    spec = p['feature_spec']; cats = matrices.schema(frames['train'], spec)
    sets = [lgb.Dataset(data.matrix(frames[n], spec, cats), label=frames[n].winner_label.to_numpy(int),
                        categorical_feature=spec['categorical_features'], free_raw_data=False)
            for n in ('train', 'early_stopping')]
    sets[1].set_reference(sets[0])
    def ledger(number, state, details):
        c.write(directory / f'fit_{number:02d}.{state}.json', details)
    deadline = time.monotonic() + p['budget']['whole_seconds'] - 20
    counter = prior_tuner.FitCounter(8, min(deadline, time.monotonic() + 900), ledger)
    cpu = resource.CPU()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    boosters = {}; configs = {}
    def objective(trial):
        counter.check()
        waited = time.monotonic()
        while cpu.percent() > 80:
            if time.monotonic() - waited > 90:
                raise safety.MLFailure('query_timeout')
            counter.check(); time.sleep(2)
        params, rounds = trial_parameters(trial, p)
        booster = counter.wrap(lgb.train)(params, sets[0], num_boost_round=rounds,
            valid_sets=[sets[1]], callbacks=[lgb.early_stopping(20, verbose=False)])
        boosters[trial.number] = booster
        configs[trial.number] = (params, rounds)
        trial.set_user_attr('selected_rounds', int(booster.best_iteration))
        return float(booster.best_score['valid_0']['binary_logloss'])
    study.optimize(objective, n_trials=8, timeout=900, catch=())
    if not 1 <= counter.calls <= 8:
        raise safety.MLFailure('training_failed')
    best = study.best_trial; params, cap = configs[best.number]
    rounds = int(boosters[best.number].best_iteration)
    trials = [{'number': t.number, 'parameters': t.params, 'value': t.value, 'state': t.state.name,
               'selected_rounds': t.user_attrs.get('selected_rounds')} for t in study.trials]
    c.write(directory / 'tuning_result.json', {'parameters': params, 'round_cap': cap, 'selected_rounds': rounds,
        'trials': trials, 'actual_search_fits': counter.calls, 'all_eight_parameters_suggested': True})
    cats = matrices.schema(frames['final_train'], spec)
    final_data = lgb.Dataset(data.matrix(frames['final_train'], spec, cats),
        label=frames['final_train'].winner_label.to_numpy(int), categorical_feature=spec['categorical_features'])
    def refit_ledger(number, state, details):
        c.write(directory / f'fit_09.{state}.json', details)
    refit = prior_tuner.FitCounter(1, deadline, refit_ledger)
    final = refit.wrap(lgb.train)(params, final_data, num_boost_round=rounds)
    raw = final.predict(data.matrix(frames['calibration'], spec, cats), num_threads=4)
    temperature, grid = adapters.calibrate(frames['calibration'], raw,
        {'objective': 'binary', 'temperature_candidates': p['calibration']['grid']})
    path = safety.local_path(directory / 'models' / uuid4().hex, root=c.ROOT)
    path.mkdir(parents=True)
    tree = path / 'booster.lgb.txt'
    with tree.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(final.model_to_string())
    manifest = c.read(directory / 'data_manifest.json')
    meta = {'version': c.VERSION, 'model_id': path.name, 'features': spec['features'], 'feature_spec': spec,
        'categories': cats, 'parameters': params, 'selected_round_cap': cap,
        'best_iteration': int(final.current_iteration()), 'temperature': temperature, 'calibration': grid,
        'dataset_id': manifest['dataset_id'], 'data_manifest_sha256': c.sha(directory / 'data_manifest.json'),
        'plan_sha256': c.sha(directory / 'plan.json'), 'spans': p['spans'],
        'created_at': c.clock().isoformat(), 'latest_training_or_calibration_date': p['latest_source_date'],
        'tree_sha256': c.sha(tree), 'versions': p['versions'], 'actual_train_calls': counter.calls + 1,
        'production_replacement': False, 'Stage7_results_used': False}
    c.write(path / 'model.json', meta)
    loaded = load(path); predicted = predict(loaded, frames['calibration'])
    delta = float(np.max(np.abs(predicted - matrices.normalize(frames['calibration'], raw, temperature))))
    if delta > 1e-12:
        raise safety.MLFailure('invalid_model')
    report = {'status': 'ok', 'model_id': path.name, 'model_path': str(path),
        'actual_train_calls': counter.calls + 1, 'selected_rounds': rounds, 'temperature': temperature,
        'calibration_grid': grid, 'parameters': params, 'selected_round_cap': cap,
        'reload_max_difference': delta, 'calibration_diagnostic_only': point(frames['calibration'], predicted),
        'not_an_unused_test': True, 'prospective_evaluation_races': 0,
        'training_seconds': counter.training_seconds + refit.training_seconds}
    c.write(directory / 'training_report.json', report)
    c.write(directory / 'candidate_freeze.json', {'model_id': path.name, 'model_path': str(path),
        'model_sha256': c.sha(path / 'model.json'), 'tree_sha256': c.sha(tree),
        'plan_sha256': c.sha(directory / 'plan.json'), 'created_at': c.clock().isoformat(),
        'live_participation_authorized': True, 'replace_existing_models': False})
    return {k: report[k] for k in ('status', 'model_id', 'actual_train_calls', 'selected_rounds', 'temperature')}


def execute(directory):
    from ..objective_pilot.runner import tree_memory, memory_bytes, terminate_worker
    c.approved(directory)
    with prior_tuner.training_lock(directory):
        c.write(directory / 'execution_started.json', {'at': c.clock().isoformat(), 'automatic_retry': False})
        env = {k: v for k, v in os.environ.items() if k not in ('PGPASSWORD', 'PGPASSFILE')}
        env.update(PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        command = [sys.executable, '-B', str(c.old.PROJECT / 'scripts/d39_next.py'), '_worker', '--run-id', directory.name]
        began = time.monotonic(); peak = 0; reason = None
        options = {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == 'nt' else {}
        with subprocess.Popen(command, cwd=c.old.PROJECT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, encoding='utf-8', **options) as child:
            while child.poll() is None:
                used, own = tree_memory(child.pid), memory_bytes(os.getpid())
                if used is None or own is None:
                    if child.poll() is not None:
                        break
                    reason = 'input_limit'
                else:
                    peak = max(peak, used + own)
                    if used + own > 12 * 1024 ** 3:
                        reason = 'input_limit'
                if time.monotonic() - began > 1800:
                    reason = 'query_timeout'
                if reason:
                    terminate_worker(child); break
                time.sleep(.25)
            output, _ = child.communicate(timeout=10)
        try:
            response = json.loads(output.strip())
        except ValueError:
            response = {'status': 'stopped', 'reason': 'training_failed'}
        value = {'status': 'complete' if not reason and child.returncode == 0 and response.get('status') == 'ok' else 'stopped',
            'reason': reason or response.get('reason'), 'actual_train_calls': len(list(directory.glob('fit_*.started.json'))),
            'wall_seconds': time.monotonic() - began, 'peak_private_bytes': peak, 'worker': response,
            'preservation': c.verify_preservation(c.read(directory / 'preservation_before.json'))}
        c.write(directory / 'completion.json', value)
        return value
