"""Four objective adapters sharing existing feature matrices; no H/legacy edits."""
from datetime import datetime, timezone
import math
from uuid import uuid4

from ..ml import safety
from ..research import contracts as features, model as existing
from . import contracts as c
from .data import load_part


def groups(frame):
    """LambdaRank groups must be contiguous, complete races, never random rows."""
    import numpy as np
    keys = frame.entry_key.to_numpy(str)
    if len(keys) < 2 or len(set(keys)) != len(keys) or list(keys) != sorted(keys):
        raise safety.MLFailure('invalid_identity')
    _, sizes = np.unique(frame.entry_key.str[:16].to_numpy(str), return_counts=True)
    if (sizes < 2).any():
        raise safety.MLFailure('invalid_identity')
    return sizes.tolist()


def targets(frame, objective):
    import numpy as np
    r = frame.finish_position.to_numpy(float); n = frame.actual_starters.to_numpy(float)
    if (not np.isfinite(r).all() or not np.isfinite(n).all() or not (r == np.floor(r)).all()
            or not (n == np.floor(n)).all() or not ((n >= 2) & (n <= 28) & (r >= 1) & (r <= n)).all()):
        raise safety.MLFailure('invalid_input')
    if objective == 'binary':
        return (r == 1).astype(int)
    if objective == 'regression':
        return (n - r) / (n - 1)
    if objective in ('multiclass', 'lambdarank'):
        return np.where(r == 1, 3, np.where(r <= 3, 2, np.where(r <= 5, 1, 0))).astype(int)
    raise safety.MLFailure('invalid_input')


def probability(frame, scores, objective, temperature):
    import numpy as np
    values = np.asarray(scores, dtype=float)
    if not math.isfinite(temperature) or temperature <= 0 or not np.isfinite(values).all():
        raise safety.MLFailure('invalid_input')
    if objective == 'multiclass':
        if (values.shape != (len(frame), 4) or ((values < 0) | (values > 1)).any()
                or not np.allclose(values.sum(axis=1), 1., atol=1e-12, rtol=0.)):
            raise safety.MLFailure('invalid_model')
        return existing.normalize(frame, values[:, 3], temperature)
    if values.shape != (len(frame),):
        raise safety.MLFailure('invalid_model')
    if objective == 'binary':
        return existing.normalize(frame, values, temperature)
    if objective not in ('regression', 'lambdarank'):
        raise safety.MLFailure('invalid_input')
    _, codes = np.unique(frame.entry_key.str[:16].to_numpy(str), return_inverse=True)
    maxes = np.full(codes.max() + 1, -np.inf); np.maximum.at(maxes, codes, values)
    e = np.exp((values - maxes[codes]) / temperature)
    return e / np.bincount(codes, weights=e)[codes]


def race_loss(frame, probabilities):
    import numpy as np
    p = np.asarray(probabilities, float); y = frame.winner_label.to_numpy(float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise safety.MLFailure('invalid_input')
    return float(-np.log(np.clip(p[y == 1], 1e-15, 1)).mean())


def calibrate(frame, scores, configuration):
    results = [{'temperature': t, 'race_log_loss': race_loss(frame, probability(frame, scores,
                configuration['objective'], t))} for t in configuration['temperature_candidates']]
    # Ties retain the preregistered grid order, never evaluation performance.
    return min(results, key=lambda r: r['race_log_loss'])['temperature'], results


def parameters(p, configuration):
    out = dict(p['parameters'], objective=configuration['objective'], metric=configuration['metric'])
    for key in ('num_class', 'ndcg_eval_at'):
        if key in configuration:
            out[key] = configuration[key]
    return out


def fit(directory, p, name):
    """Called once in an isolated worker, after the parent exclusively reserves fit."""
    import lightgbm as lgb
    import numpy as np
    import importlib.metadata
    configuration = next(x for x in p['model_configurations'] if x['name'] == name)
    if not (directory / f'fit_{name}.started.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    c.write(directory, f'fit_{name}.entered.json', {'name': name, 'retry_forbidden': True})
    train = load_part(directory, 'train'); early = load_part(directory, 'early_stopping')
    cal = load_part(directory, 'calibration')
    if not max(train.date) < min(early.date) < min(cal.date):
        raise safety.MLFailure('invalid_split')
    spec = features.spec(('A', 'B', 'C', 'D')); categories = existing.schema(train, spec)
    data = []
    for f in (train, early):
        g = groups(f)
        data.append(lgb.Dataset(existing.matrix(f, spec, categories), label=targets(f, configuration['objective']),
                    group=g if configuration['objective'] == 'lambdarank' else None,
                    reference=data[0] if data else None, categorical_feature=spec['categorical_features']))
    history = {}; params = parameters(p, configuration)
    booster = lgb.train(params, data[0], num_boost_round=p['budget']['max_boosting_rounds_per_fit'],
                        valid_sets=[data[1]], callbacks=[lgb.early_stopping(20, verbose=False),
                                                       lgb.record_evaluation(history)])
    if booster.num_trees() > (1200 if configuration['objective'] == 'multiclass' else 300):
        raise safety.MLFailure('input_limit')
    raw = booster.predict(existing.matrix(cal, spec, categories), num_threads=4)
    temperature, grid = calibrate(cal, raw, configuration)
    path = safety.local_path(directory / 'models' / uuid4().hex, root=c.ROOT)
    path.mkdir(parents=True, exist_ok=False)
    tree = path / 'booster.lgb.txt'
    with tree.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(booster.model_to_string(num_iteration=booster.best_iteration))
    metadata = {'format_version': c.VERSION, 'model_id': path.name, 'configuration': configuration,
        'feature_spec': spec, 'features': spec['features'], 'categories': categories,
        'formula_version': features.FORMULA_VERSION, 'input_format': features.INPUT_VERSION,
        'calibration': {'temperature': temperature, 'grid': grid, 'source': 'calibration_only',
                        'chosen_before_evaluation': True}, 'parameters': params, 'spans': p['spans'],
        'best_iteration': int(booster.best_iteration), 'trees': booster.num_trees(),
        'max_rounds': 300, 'early_stopping_rounds': 20,
        'tree_sha256': safety.digest(tree, root=c.ROOT), 'data_manifest_sha256': safety.digest(directory / 'data_manifest.json', root=c.ROOT),
        'proposal_sha256': c.PROPOSAL_HASH, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'library_versions': {n: importlib.metadata.version(n) for n in ('lightgbm', 'numpy', 'pandas', 'scikit-learn')},
        'production_adoption': False, 'H_changed': False, 'sealed_2026_used': False}
    c.write(path, 'model.json', metadata)  # freeze calibration before evaluation cache is read
    loaded = load(path)
    reloaded = predict(loaded, cal)
    delta = float(np.max(np.abs(reloaded - probability(cal, raw, configuration['objective'], temperature))))
    if delta > 1e-12:
        raise safety.MLFailure('invalid_model')
    evaluation = load_part(directory, 'evaluation')
    if min(evaluation.date) <= max(cal.date):
        raise safety.MLFailure('invalid_split')
    predictions = predict(loaded, evaluation)
    out = directory / f'{name}.predictions.local.csv'
    with out.open('x', encoding='utf-8', newline='') as stream:
        evaluation[['entry_key', 'date', 'winner_label']].assign(probability=predictions).to_csv(stream, index=False)
    result = {'name': name, 'status': 'complete', 'model_id': path.name,
              'best_iteration': metadata['best_iteration'], 'trees': metadata['trees'],
              'temperature': temperature, 'calibration': grid, 'reload_max_difference': delta,
              'prediction_sha256': safety.digest(out, root=c.ROOT), 'model_sha256': safety.digest(path / 'model.json', root=c.ROOT)}
    c.write(directory, f'fit_{name}.result.json', result)
    return {k: result[k] for k in ('name', 'status', 'best_iteration', 'trees', 'temperature')}


def load(path):
    import lightgbm as lgb
    m = c.read(path, 'model.json')
    if (m['format_version'] != c.VERSION or m['proposal_sha256'] != c.PROPOSAL_HASH
            or m['feature_spec'] != features.spec(('A', 'B', 'C', 'D')) or m['features'] != m['feature_spec']['features']
            or m['configuration'] not in c.plan()['model_configurations']
            or m['calibration']['temperature'] not in m['configuration']['temperature_candidates']):
        raise safety.MLFailure('invalid_model')
    tree = safety.local_path(path / 'booster.lgb.txt', root=c.ROOT, exists=True)
    if safety.digest(tree, root=c.ROOT) != m['tree_sha256']:
        raise safety.MLFailure('artifact_invalid')
    booster = lgb.Booster(model_str=tree.read_text(encoding='utf-8'))
    if list(booster.feature_name()) != m['features'] or booster.num_trees() != m['trees']:
        raise safety.MLFailure('invalid_model')
    return {'metadata': m, 'booster': booster}


def predict(bundle, frame):
    # Results can be present in a joined frame, but only the fixed 39 names reach the model.
    m = bundle['metadata']
    values = bundle['booster'].predict(existing.matrix(frame, m['feature_spec'], m['categories']), num_threads=4)
    return probability(frame, values, m['configuration']['objective'], m['calibration']['temperature'])
