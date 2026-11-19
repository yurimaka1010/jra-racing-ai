"""Two independent research families, sharing the existing causal ABCD matrix."""
from datetime import datetime, timezone
import math
from uuid import uuid4

from ..ml import safety
from ..research import contracts as existing_contracts, model as existing
from ..objective_pilot import model as pilot
from . import contracts as c
from .outcomes import rank_reference


def base_spec():
    return existing_contracts.spec(('A', 'B', 'C', 'D'))


def matrix(frame, categories, family):
    import pandas as pd
    x = existing.matrix(frame, base_spec(), categories)
    if family == c.FAMILIES[0]:
        k = frame.place_paid_positions.to_numpy(float)
        if not all(v in (2, 3) for v in k): raise safety.MLFailure('invalid_input')
        x['place_paid_positions'] = pd.Categorical([str(int(v)) for v in k], categories=['2', '3'])
    elif family != c.FAMILIES[1]:
        raise safety.MLFailure('invalid_input')
    return x


def validate_frame(frame, family):
    import numpy as np
    required = {'entry_key', 'date', 'finish_position', 'actual_starters', 'place_paid_positions', *base_spec()['features']}
    if family not in c.FAMILIES: raise safety.MLFailure('invalid_input')
    if family == c.FAMILIES[0]: required.update(('official_label_status', 'official_place_label'))
    if not required <= set(frame.columns): raise safety.MLFailure('invalid_schema')
    if frame.empty or frame.entry_key.duplicated().any() or list(frame.entry_key) != sorted(frame.entry_key):
        raise safety.MLFailure('invalid_identity')
    for race, f in frame.groupby(frame.entry_key.str[:16], sort=False):
        c.identity(str(race), str(f.date.iloc[0]))
        import re
        if any(not re.fullmatch('[0-9]{18}', str(key)) for key in f.entry_key):
            raise safety.MLFailure('invalid_identity')
        if not all(c.day(v) == c.day(f.date.iloc[0]) for v in f.date): raise safety.MLFailure('invalid_identity')
        ranks = f.finish_position.to_numpy(float); n = f.actual_starters.to_numpy(float)
        if (not np.isfinite(ranks).all() or not (ranks == np.floor(ranks)).all()
                or not (n == len(f)).all() or not ((ranks >= 1) & (ranks <= n)).all()):
            raise safety.MLFailure('invalid_identity')
        paid = f.place_paid_positions.to_numpy(float)
        if not np.isin(paid, (2, 3)).all() or len(set(paid)) != 1:
            raise safety.MLFailure('invalid_input')
        labels, why = rank_reference(dict(zip(f.entry_key.str[-2:], map(int, ranks))), int(paid[0]))
        if why: raise safety.MLFailure('outcomes_incomplete')
        if family == c.FAMILIES[0]:
            if (not (f.official_label_status == 'verified_ordinary_HR').all()
                    or not np.isin(f.official_place_label.to_numpy(), (0, 1)).all()
                    or not np.array_equal(f.official_place_label.to_numpy(int),
                                          np.array([labels['place'][k[-2:]] for k in f.entry_key]))):
                raise safety.MLFailure('mapping_not_reviewed')


def place_probability(raw, temperature=1.):
    import numpy as np
    p = np.asarray(raw, float)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0):
        raise safety.MLFailure('invalid_input')
    if temperature == 1: return p
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return existing.sigmoid((np.log(q) - np.log1p(-q)) / temperature)


def strengths(scores, temperature=1.):
    import numpy as np
    r = np.asarray(scores, float)
    if (r.ndim != 1 or not len(r) or not np.isfinite(r).all()
            or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0):
        raise safety.MLFailure('invalid_input')
    # Relative strength only. Never interpret these scores as standalone probabilities.
    out = np.exp((r - r.max()) / temperature)
    if not (out > 0).all(): raise safety.MLFailure('invalid_model')
    return out


def top3_loss(frame, scores, temperature):
    import numpy as np
    losses = []; offset = 0
    for _, f in frame.groupby(frame.entry_key.str[:16], sort=True):
        w = strengths(scores[offset:offset + len(f)], temperature); offset += len(f)
        ranks = f.finish_position.to_numpy(int); loss = 0.; remaining = np.ones(len(f), bool)
        for position in (1, 2, 3):
            indices = np.flatnonzero(ranks == position)
            if len(indices) != 1: raise safety.MLFailure('outcomes_incomplete')
            index = indices[0]; loss -= math.log(float(w[index] / w[remaining].sum())); remaining[index] = False
        losses.append(loss)
    return float(np.mean(losses))


def fit_family(directory, family, train, early, calibration, *, approved_plan_sha256, reserve_fit):
    """One real fit per family. Caller owns approval, process budget and serial lock."""
    from .training import approved
    if approved(directory) != approved_plan_sha256:
        raise safety.MLFailure('explicit_execution_required')
    ownership = safety.read_json(c.TRAINING_LOCK, root=c.BASE)
    if ownership.get('run_id') != directory.name or ownership.get('purpose') != c.VERSION:
        raise safety.MLFailure('explicit_execution_required')
    import lightgbm as lgb
    import numpy as np
    from ..ml.model import probability_metrics, versions
    if family not in c.FAMILIES or approved_plan_sha256 is None:
        raise safety.MLFailure('explicit_execution_required')
    for f in (train, early, calibration): validate_frame(f, family)
    if not max(train.date) < min(early.date) < min(calibration.date) or max(calibration.date) >= '2025-01-01':
        raise safety.MLFailure('invalid_split')
    categories = existing.schema(train, base_spec()); categorical = base_spec()['categorical_features'][:]
    if family == c.FAMILIES[0]: categorical.append('place_paid_positions')
    params = dict(learning_rate=.05, num_leaves=15, max_depth=5, min_data_in_leaf=20,
                  lambda_l1=0., lambda_l2=0., feature_pre_filter=False, num_threads=4,
                  seed=42, deterministic=True, force_col_wise=True, verbosity=-1)
    direct = family == c.FAMILIES[0]
    params.update(objective='binary' if direct else 'lambdarank', metric='binary_logloss' if direct else 'ndcg')
    if not direct: params['ndcg_eval_at'] = [1, 3]
    datasets = []
    for f in (train, early):
        y = f.official_place_label.to_numpy(int) if direct else pilot.targets(f, 'lambdarank')
        datasets.append(lgb.Dataset(matrix(f, categories, family), label=y,
            group=pilot.groups(f) if not direct else None, reference=datasets[0] if datasets else None,
            categorical_feature=categorical))
    reserve_fit(family)  # An exception from train still consumes the reservation.
    booster = lgb.train(params, datasets[0], num_boost_round=300, valid_sets=[datasets[1]],
                        callbacks=[lgb.early_stopping(20, verbose=False)])
    raw = booster.predict(matrix(calibration, categories, family), num_threads=4)
    grid = []
    for t in ((.75, 1., 1.25, 2.) if direct else (.25, .5, 1., 2.)):
        loss = (probability_metrics(calibration.official_place_label.tolist(), place_probability(raw, t).tolist())['log_loss']
                if direct else top3_loss(calibration, raw, t))
        grid.append({'temperature': t, 'calibration_loss': loss})
    temperature = min(grid, key=lambda item: item['calibration_loss'])['temperature']
    path = safety.local_path(directory / 'models' / uuid4().hex, root=c.ROOT); path.mkdir(parents=True, exist_ok=False)
    tree = path / 'model.lgb.txt'
    with tree.open('x', encoding='utf-8') as stream: stream.write(booster.model_to_string())
    metadata = {'version': c.VERSION, 'family': family, 'model_id': path.name, 'feature_spec': base_spec(),
        'features': base_spec()['features'] + (['place_paid_positions'] if direct else []), 'categories': categories,
        'extra_categories': {'place_paid_positions': ['2', '3']} if direct else {},
        'input_format': existing_contracts.INPUT_VERSION, 'formula_version': existing_contracts.FORMULA_VERSION,
        'calibration': {'temperature': temperature, 'grid': grid, 'period_only': True},
        'best_iteration': booster.best_iteration, 'parameters': params, 'library_versions': versions(),
        'tree_sha256': safety.digest(tree, root=c.ROOT), 'approved_plan_sha256': approved_plan_sha256,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'production_adoption': False,
        'official_place_target_required': direct, 'shared_order_assumption': None if direct else 'plackett_luce_without_dead_heats'}
    c.write(path, 'model.json', metadata)
    restored = load(path); check = restored['booster'].predict(matrix(calibration, categories, family), num_threads=4)
    if not np.allclose(raw, check, atol=1e-12, rtol=0): raise safety.MLFailure('invalid_model')
    return {'model_id': path.name, 'family': family, 'temperature': temperature, 'best_iteration': booster.best_iteration}


def load(path):
    import lightgbm as lgb
    m = c.read(path, 'model.json')
    if (m['version'] != c.VERSION or m['family'] not in c.FAMILIES or m['feature_spec'] != base_spec()
            or m['features'] != base_spec()['features'] + (['place_paid_positions'] if m['family'] == c.FAMILIES[0] else [])
            or m['calibration']['temperature'] not in ((.75, 1., 1.25, 2.) if m['family'] == c.FAMILIES[0] else (.25, .5, 1., 2.))):
        raise safety.MLFailure('invalid_model')
    tree = safety.local_path(path / 'model.lgb.txt', root=c.ROOT, exists=True)
    if safety.digest(tree, root=c.ROOT) != m['tree_sha256']: raise safety.MLFailure('artifact_invalid')
    booster = lgb.Booster(model_str=tree.read_text(encoding='utf-8'))
    if booster.feature_name() != m['features']: raise safety.MLFailure('invalid_model')
    return {'metadata': m, 'booster': booster}


def predict(bundle, frame):
    from .probabilities import plackett_luce
    pilot.groups(frame)
    m = bundle['metadata']; family = m['family']; t = m['calibration']['temperature']; output = {}
    raw = bundle['booster'].predict(matrix(frame, m['categories'], family), num_threads=4); offset = 0
    for race, f in frame.groupby(frame.entry_key.str[:16], sort=False):
        k = set(f.place_paid_positions.to_numpy(float))
        if len(k) != 1 or next(iter(k)) not in (2., 3.): raise safety.MLFailure('invalid_input')
        r = raw[offset:offset + len(f)]; offset += len(f); horses = f.entry_key.str[-2:]
        output[str(race)] = ({'place': dict(zip(horses, map(float, place_probability(r, t))))}
            if family == c.FAMILIES[0] else plackett_luce(dict(zip(horses, map(float, strengths(r, t)))), int(next(iter(k)))))
    return output
