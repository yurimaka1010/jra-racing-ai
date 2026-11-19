"""Two small CPU LightGBM models, persisted as text plus a fixed category schema."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path

from .features import (CATEGORY_FEATURES, FEATURES, build_features, chronological_split,
                       entry_identity, rank_labels, validate_rows)
from .safety import MLFailure, digest, local_path, read_json, write_json


def libraries():
    try:
        import pandas as pd
        import lightgbm as lgb
        return pd, lgb
    except ImportError:
        raise MLFailure("missing_library") from None


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "lightgbm", "scikit-learn")}


def category_schema(features):
    return {name: ["__MISSING__", "__UNKNOWN__"] + sorted({"v:" + str(row[name]) for row in features
                                                        if row[name] is not None})
            for name in CATEGORY_FEATURES}


def frame(features, schema):
    pd, _ = libraries()
    if set(schema) != set(CATEGORY_FEATURES):
        raise MLFailure("invalid_model")
    data = pd.DataFrame(features, columns=FEATURES)
    for name in CATEGORY_FEATURES:
        categories = schema[name]
        if (not isinstance(categories, list) or categories[:2] != ["__MISSING__", "__UNKNOWN__"]
                or len(set(categories)) != len(categories)
                or any(not isinstance(item, str) for item in categories)):
            raise MLFailure("invalid_model")
        allowed = set(categories)
        values = ["__MISSING__" if row[name] is None else
                  "v:" + str(row[name]) if "v:" + str(row[name]) in allowed else "__UNKNOWN__"
                  for row in features]
        data[name] = pd.Categorical(values, dtype=pd.CategoricalDtype(categories=categories))
    for name in FEATURES[len(CATEGORY_FEATURES):]:
        data[name] = pd.to_numeric(data[name], errors="raise").astype("float64")
    return data


def probability_metrics(labels, probabilities):
    if len(labels) != len(probabilities) or not labels:
        raise MLFailure("invalid_input")
    bins = [{"count": 0, "probability_sum": 0., "observed_sum": 0} for _ in range(10)]
    loss, brier = 0., 0.
    for label, probability in zip(labels, probabilities):
        p = float(probability)
        if label not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise MLFailure("invalid_input")
        clipped = max(1e-15, min(1 - 1e-15, p))
        loss -= label * math.log(clipped) + (1 - label) * math.log(1 - clipped)
        brier += (p - label) ** 2
        item = bins[min(9, int(p * 10))]
        item["count"] += 1
        item["probability_sum"] += p
        item["observed_sum"] += label
    calibration, ece = [], 0.
    for index, item in enumerate(bins):
        count = item["count"]
        mean = item["probability_sum"] / count if count else None
        observed = item["observed_sum"] / count if count else None
        if count:
            ece += count * abs(mean - observed) / len(labels)
        calibration.append({"bin": index, "count": count, "mean_probability": mean, "observed_rate": observed})
    return {"count": len(labels), "log_loss": loss / len(labels), "brier_score": brier / len(labels),
            "calibration_ece": ece, "calibration_bins": calibration}


def fit_models(rows, *, history_lag_days=7, train_end=None, valid_end=None,
               test_end=None, rounds=180, policy=None, history=(), targets=("win", "place"),
               calibration_start=None, calibration_end=None, test_start=None,
               _prepared_features=None):
    validate_rows(rows)
    if type(rounds) is not int or not 5 <= rounds <= 1000:
        raise MLFailure("invalid_input")
    pd, lgb = libraries()
    # The split and purchase rule are fixed before either final-test evaluation.
    split = chronological_split(rows, train_end=train_end, valid_end=valid_end, test_end=test_end)
    if targets not in (("win",), ("win", "place")):
        raise MLFailure("invalid_input")
    if calibration_end is not None:
        from datetime import date
        if (not all(isinstance(day, date) for day in (calibration_start, calibration_end, test_start, valid_end, test_end))
                or not valid_end < calibration_start <= calibration_end < test_start <= test_end):
            raise MLFailure("invalid_split")
        later = split["test"]
        split["calibration"] = [i for i in later if calibration_start <= entry_identity(rows[i])[1] <= calibration_end]
        split["test"] = [i for i in later if test_start <= entry_identity(rows[i])[1] <= test_end]
        if not split["calibration"] or not split["test"]:
            raise MLFailure("invalid_split")
    elif calibration_start is not None or test_start is not None:
        raise MLFailure("invalid_split")
    from .paper import PurchasePolicy
    policy = PurchasePolicy() if policy is None else policy
    policy.validate()
    # Internal cache for controlled learning-curve experiments. Its values are
    # produced by the same causal generator before slicing the nested windows.
    if _prepared_features is None:
        features = build_features(rows, history=history, history_lag_days=history_lag_days)
    else:
        if (not isinstance(_prepared_features, list) or len(_prepared_features) != len(rows)
                or any(not isinstance(item, dict) or set(item) != set(FEATURES) for item in _prepared_features)):
            raise MLFailure("invalid_input")
        features = _prepared_features
    labels, label_reasons = rank_labels(rows)
    schema = category_schema([features[i] for i in split["train"]])
    matrix = frame(features, schema)
    boosters, predictions, metrics, priors, training_rounds = {}, {}, {}, {}, {}
    race_counts = {}
    for row in rows:
        key, _ = entry_identity(row)
        race_counts[key[:-1]] = race_counts.get(key[:-1], 0) + 1
    for target in targets:
        indexes = {part: [i for i in indices if labels[i][target] is not None] for part, indices in split.items()}
        if (any(not values for values in indexes.values())
                or set(labels[i][target] for i in indexes["train"]) != {0, 1}):
            raise MLFailure("insufficient_history")
        train_y = [labels[i][target] for i in indexes["train"]]
        prior = sum(train_y) / len(train_y)
        priors[target] = prior
        train_data = lgb.Dataset(matrix.iloc[indexes["train"]], label=train_y,
                                 categorical_feature=list(CATEGORY_FEATURES), free_raw_data=True)
        validation = lgb.Dataset(matrix.iloc[indexes["validation"]],
                                 label=[labels[i][target] for i in indexes["validation"]], reference=train_data)
        params = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
                      num_leaves=15, max_depth=5, min_data_in_leaf=20, feature_pre_filter=False,
                      num_threads=4, seed=42, deterministic=True, force_col_wise=True,
                      verbosity=-1, bagging_fraction=1., feature_fraction=1.)
        try:
            evaluation_history = {}
            booster = lgb.train(params, train_data, num_boost_round=rounds, valid_sets=[validation],
                                callbacks=[lgb.early_stopping(20, verbose=False), lgb.record_evaluation(evaluation_history)])
            probabilities = booster.predict(matrix, num_iteration=booster.best_iteration, num_threads=4)
        except Exception:
            raise MLFailure("training_failed") from None
        boosters[target] = booster
        executed = len(evaluation_history.get("valid_0", {}).get("binary_logloss", []))
        training_rounds[target] = {"max_rounds": rounds, "early_stopping_patience": 20,
            "rounds_executed": executed, "best_iteration": booster.best_iteration,
            "early_stopped": executed < rounds, "reached_round_limit": executed == rounds}
        predictions[target] = [float(value) for value in probabilities]
        metrics[target] = {}
        for part, indices in indexes.items():
            y = [labels[i][target] for i in indices]
            probabilities_part = [predictions[target][i] for i in indices]
            uniform = []
            for i in indices:
                n = race_counts[entry_identity(rows[i])[0][:-1]]
                uniform.append((1 if target == "win" else (3 if n >= 8 else 2)) / n)
            metrics[target][part] = {
                "model": probability_metrics(y, probabilities_part),
                "train_prior_baseline": probability_metrics(y, [prior] * len(y)),
                "uniform_race_baseline": probability_metrics(y, uniform),
            }
    summary = {part: {"rows": len(indices),
                      "races": len({entry_identity(rows[i])[0][:-1] for i in indices}),
                      "start": min(entry_identity(rows[i])[1] for i in indices).isoformat(),
                      "end": max(entry_identity(rows[i])[1] for i in indices).isoformat()}
               for part, indices in split.items()}
    metadata = {"format_version": 1, "model_kind": "lightgbm_rank_proxy_v01",
                "targets": list(targets),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "features": list(FEATURES), "categories": schema,
                "history_lag_days": history_lag_days, "split": summary, "metrics": metrics,
                "label_reasons": label_reasons, "label_contract": "rank_proxy_not_official_ticket_labels",
                "availability": "retrospective_snapshot_not_verified_point_in_time",
                "production_hr_settlement_enabled": False, "library_versions": versions(),
                "baseline_train_priors": priors, "purchase_policy": policy.as_dict(),
                "best_iterations": {k: b.best_iteration for k, b in boosters.items()}}
    metadata.update(training_parameters=params, training_rounds=training_rounds)
    metadata["missing_features"] = {part: {name: sum(features[i][name] is None for i in indices)
                                            for name in FEATURES} for part, indices in split.items()}
    metadata["missing_feature_reasons"] = ["unmapped_or_unreviewed_optional_column",
                                            "invalid_optional_numeric", "no_eligible_past_run"]
    return {"boosters": boosters, "metadata": metadata, "predictions": predictions,
            "labels": labels, "split": split}


def save_models(bundle, directory: Path, *, root=None):
    directory = local_path(directory, root=root)
    artifacts = {}
    for target, booster in bundle["boosters"].items():
        path = local_path(directory / (target + ".lgb.txt"), root=root)
        try:
            # Native LightGBM tree offsets are byte-based. Windows CRLF
            # translation would corrupt a model_to_string() serialization.
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(booster.model_to_string(num_iteration=booster.best_iteration))
        except Exception:
            raise MLFailure("io_failed") from None
        artifacts[path.name] = digest(path, root=root)
    metadata = dict(bundle["metadata"], artifacts=artifacts)
    write_json(directory / "model.json", metadata, root=root)
    return metadata


def load_models(directory: Path, *, root=None, required_targets=("win", "place")):
    _, lgb = libraries()
    metadata = read_json(directory / "model.json", root=root)
    if (metadata.get("format_version") != 1 or metadata.get("features") != list(FEATURES)
            or metadata.get("model_kind") != "lightgbm_rank_proxy_v01"
            or metadata.get("production_hr_settlement_enabled") is not False):
        raise MLFailure("invalid_model")
    boosters = {}
    try:
        declared = metadata.get("targets", ["win", "place"])
        if (declared not in (["win"], ["win", "place"]) or not set(required_targets) <= set(declared)):
            raise MLFailure("invalid_model")
        for target in declared:
            path = local_path(directory / (target + ".lgb.txt"), exists=True, root=root)
            if digest(path, root=root) != metadata["artifacts"][path.name]:
                raise MLFailure("artifact_invalid")
            boosters[target] = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        return {"boosters": boosters, "metadata": metadata}
    except MLFailure:
        raise
    except Exception:
        raise MLFailure("invalid_model") from None


def predict(bundle, rows, *, history=()):
    metadata = bundle["metadata"]
    features = build_features(rows, history=history, history_lag_days=metadata["history_lag_days"],
                              feature_profile=metadata.get('feature_profile'))
    matrix = frame(features, metadata["categories"])
    return {target: [float(p) for p in booster.predict(matrix, num_threads=4)]
            for target, booster in bundle["boosters"].items()}
