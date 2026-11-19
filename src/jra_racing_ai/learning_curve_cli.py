"""STEP 2 V2 runner over one bounded local snapshot; no DB or purchase writes."""
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
import argparse
import csv
import json
from pathlib import Path
import time

from . import morning_trial as trial
from .core import SafeAuditError
from .ml import data, features, learning_curve as curve, market, model, safety, v03
from .ml_v03_cli import save_policy, save_predictions, old_history, freeze
from .win_sim import core, local

ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\ml-step2")
COLLECTION = ROOT / "evaluations" / "4b6339b48f954dcca0d12567bf3dbacd"


def day_of(race):
    return date(int(race[0]), int(race[1][:2]), int(race[1][2:]))


def part_predictions(rows, indices, values):
    return v03.prediction_races([rows[i] for i in indices], [values[i] for i in indices])


def report(directory, result):
    plan = result["plan"]
    lines = ["# STEP 2 V2 学習量拡大の比較", "",
        "旧モデル・温度・10月17日の保存予測・購入/精算台帳は保持。新モデルを予約タスクに反映していません。", "",
        "主比較Bは同じ特徴量・同じ学習終了時点・同じ検証/校正/評価・同じ履歴開始日で、教師あり学習のレース数だけを変えた入れ子窓です。",
        "比較Aの旧V0.3は2025年1月の144レースで学習済み。新窓の終了は2024年11月であり、古いモデルという名前が学習日も古いことを意味しません。Aには期間と履歴範囲の違いが含まれます。", "",
        "## 取得範囲と固定した分割", "",
        f"共通取得: {result['pool_rows']:,}馬行・{result['pool_races']:,}レース。履歴参照開始{plan['history_start']}。",
        "DBはより古いデータも収録していますが、今回は既存の結合特徴量上限50万行に収まる最大の月初始まりの窓（見積48万行以下）に限定しました。DB全量で学習したという意味ではありません。LIMIT超過時は失敗させ、途中切捨てはしていません。", "",
        "| 用途 | 開始 | 終了 | 有効レース | 頭数 |", "|---|---|---|---:|---:|"]
    for name, part in result["partitions"].items():
        lines.append(f"| {name} | {part['start']} | {part['end']} | {part['races']} | {part['rows']} |")
    lines += ["", "最初の学習日まで365日を確保。小窓も同じ共通履歴を使うため、同じ馬・同じ日時の特徴量は窓間で一致します。履歴専用期間を学習ラベルに混ぜません。過去走定義・日付+7日遅延は変更していません。評価内の先行開催結果も7日経過後のみ後続開催の履歴に使用します。",
        "ラベル・除外は既存rank proxyと単独勝馬の評価条件を使い、全頭の既知着順・必要な基本特徴量が揃うレースを全窓で共通に採用します。取消/除外/中止による着順不明を0の負例へ変えません。",
        "主評価は今回初めて測定する共通の未学習期間。参考の2025年2月15～16日は既に結果を見た期間です。候補をこの主評価で選ぶ場合、その後に別期間での確認が必要です。", "",
        "## 主評価：馬単位二値指標", "",
        "| 系列 | 学習レース | T | Log Loss | Brier | 1位的中率% | 勝馬平均順位 | best/max |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    baseline = result["saved_v03"]["primary"]["model"]
    lines.append(f"| 保存V0.3 | 144 | 2.0 | {baseline['log_loss']:.6f} | {baseline['brier_score']:.6f} | {baseline['top1_accuracy_percent']:.3f} | {baseline['winner_mean_rank']:.3f} | {result['saved_v03']['best_iteration']}/100 |")
    for entry in result["models"]:
        rounds = entry["training_rounds"]
        for series, item in entry["series"].items():
            m = item["primary"]["model"]
            lines.append(f"| {entry['experiment']}/{series} | {entry['training_races']} | {item['temperature']} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['top1_accuracy_percent']:.3f} | {m['winner_mean_rank']:.3f} | {rounds['best_iteration']}/{rounds['max_rounds']} |")
    uniform = result["saved_v03"]["primary"]["uniform"]
    lines += ["", f"一様予測: Log Loss={uniform['log_loss']:.6f}, Brier={uniform['brier_score']:.6f}。",
        "系列fixedはT=2.0を固定。recalibratedは独立校正期間のレースLog Lossで既存候補0.75/1/1.25/2から選択（テストを温度選択に使っていない）。全候補・両系列をrun.jsonに保存しました。",
        "max_rounds_only_referenceは上限100到達を条件に、最大窓で上限のみ300へ広げた別実験です。学習量だけを変える主比較には混ぜません。", "",
        "## ペアード・ブートストラップ", "",
        "1000回、固定seed=20261018。レースを丸ごと再標本化し、各標本で馬単位Log Loss/Brierを再計算。同一レース・同一馬集合のペア差です。95%はパーセンタイル区間で、多重比較補正は行っていません。", "",
        "| モデル/系列 | 基準 | ΔLog Loss | 95%区間 | ΔBrier | Δ1位的中率pp |", "|---|---|---:|---|---:|---:|"]
    for entry in result["models"]:
        for series, item in entry["series"].items():
            for label, ci in item["comparisons"].items():
                ll = ci["log_loss"]
                lines.append(f"| {entry['training_races']}/{entry['experiment']}/{series} | {label} | {ll['difference_new_minus_baseline']:.6f} | [{ll['ci95_low']:.6f}, {ll['ci95_high']:.6f}] | {ci['brier_score']['difference_new_minus_baseline']:.6f} | {ci['top1_accuracy_percentage_points']['difference_new_minus_baseline']:.3f} |")
    lines += ["", "Brier差・的中率差それぞれの95%区間もrun.jsonへ保存。差は新モデル−基準。Log Lossの区間上限<0なら改善を支持、0を含めば不確実です。", "",
        "## 市場との同一集合比較（確定オッズ・事後参考）", "",
        "市場は1/oddsをレース内正規化。区分別に別集計し、発走前オッズや収益評価に流用しません。欠損は補完しません。", "",
        "| モデル/系列 | O1区分 | レース | 頭数 | AI LL | 市場LL | AI−市場 |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, comparisons in result["market_comparisons"].items():
        for basis, item in comparisons.items():
            ai, mr = item["metrics"]["v03_calibrated"], item["metrics"]["market"]
            lines.append(f"| {label} | {basis} | {item['race_count']} | {item['runner_count']} | {ai['log_loss']:.6f} | {mr['log_loss']:.6f} | {ai['log_loss']-mr['log_loss']:.6f} |")
    lines += ["", "## 欠損・再現性・残る境界", "",
        "窓別・検証/校正/主評価別の欠損率と履歴なし率、確率帯別の頭数・平均予測・実勝率、温度前後の分布はrun.jsonに保存。馬場状態・過去走破タイムは元のV0.3で無効のままです。",
        "既存モデルの再学習再現の最大出力差: " + str(result["reproduction"].get("max_absolute_prediction_difference")) + "。",
        "旧66レースの保存予測と現在スナップショットからの再計算差: " + str(result["saved_v03"]["reference_saved_prediction_check"]) + "。",
        "RA/SEの最終保存版による遡及評価であり、全特徴量の事前公開・訂正履歴を完全再現したとは認定しません。今回、購入方針・市場ブレンド・無人オッズ取得は変更していません。", ""]
    with safety.local_path(directory / "report.md", root=ROOT).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def run(collection, directory, progress):
    started = time.monotonic()
    plan = safety.read_json(collection / "plan.json", root=ROOT)
    days = curve.validate_plan(plan)
    source = safety.read_json(collection / "collection.json", root=ROOT)
    dataset = safety.local_path(Path(source["dataset_path"]), exists=True)
    saved, mapping, old, source_policy, frozen = trial.load_source(date(2026, 10, 17))
    frozen += [(p, root, safety.digest(p, root=root)) for p, root in (
        (dataset, safety.ML_ROOT), (dataset.parent / "dataset.json", safety.ML_ROOT),
        (collection / "plan.json", ROOT), (collection / "market_inputs.local.json", ROOT),
        (trial.ROOT / "evaluations" / "9b2dc7a96aa24008a0dae3a0c34f31e2" / "predictions.csv", trial.ROOT))]
    def stage(name, **values):
        progress.update(stage=name, **values)
        print(json.dumps(dict(status="running", **progress), ensure_ascii=True), flush=True)
    stage("load_shared_dataset")
    rows, metadata = data.read_dataset(dataset)
    stage("build_shared_causal_features", pool_rows=len(rows))
    matrix = features.build_features(rows, history_lag_days=7)
    parts, groups, winners, exclusions = curve.prepare_cohorts(rows, matrix, plan)
    windows = curve.nested_windows(parts["train"])
    partition_summary = {}
    for name, races in parts.items():
        partition_summary[name] = {"start": min(day_of(r) for r in races).isoformat(),
            "end": max(day_of(r) for r in races).isoformat(), "races": len(races),
            "rows": sum(len(groups[r]) for r in races)}
    stage("partitions_fixed", partitions=partition_summary, maximum_training_races=len(parts["train"]))
    primary_indices = [i for race in parts["test"] for i in groups[race]]
    reference_indices = [i for race in parts["reference"] for i in groups[race]]
    reference_keys = set(parts["reference"])
    baseline_indices = primary_indices + reference_indices
    base_rows = [rows[i] for i in baseline_indices]
    stage("saved_v03_predict")
    original_history = [row for row in rows if features.entry_identity(row)[1] >= date(2024, 1, 1)]
    old_features = features.build_features(base_rows, history=original_history, history_lag_days=7)
    old_values = old["boosters"]["win"].predict(model.frame(old_features, old["metadata"]["categories"]), num_threads=4)
    old_predictions = v03.prediction_races(base_rows, old_values)
    old_by_race = {p.race: p for p in old_predictions}
    old_test = [old_by_race[r] for r in parts["test"]]
    old_reference = [old_by_race[r] for r in parts["reference"]]
    base_policy = replace(source_policy, evaluation_start=plan["test_start"], evaluation_end=plan["test_end"])
    old_scores = curve.race_scores(old_test, winners, base_policy)
    saved_predictions, _ = local.load_predictions(local.ROOT / "predictions" / saved["heldout_predictions_id"])
    saved_reference = {p.race: p for p in saved_predictions}
    comparable = [r for r in parts["reference"] if r in saved_reference and set(saved_reference[r].probabilities) == set(old_by_race[r].probabilities)]
    reference_check = {"common_races": len(comparable), "max_raw_probability_difference": max(
        abs(float(old_by_race[r].probabilities[h])-float(saved_reference[r].probabilities[h])) for r in comparable for h in old_by_race[r].probabilities)}
    old_prediction_dir = save_predictions(old_predictions, safety.digest(safety.ML_ROOT / "models" / saved["model_id"] / "model.json"), dataset, purpose="step2_saved_v03_reused")
    del old_features, original_history
    stage("reproduce_original_144")
    old_rows, _ = data.read_dataset(safety.ML_ROOT / "datasets" / saved["dataset_id"] / "runners.csv")
    old_past, _ = old_history({"metadata": dict(old["metadata"], dataset_sha256=old["metadata"]["history_dataset_sha256"])})
    reproduction_bundle = model.fit_models(old_rows, history=old_past, targets=("win",), rounds=100,
        train_end=date(2025, 1, 18), valid_end=date(2025, 1, 19), test_end=date(2025, 2, 16),
        calibration_start=date(2025, 2, 1), calibration_end=date(2025, 2, 2), test_start=date(2025, 2, 15))
    reproduced = part_predictions(old_rows, reproduction_bundle["split"]["test"], reproduction_bundle["predictions"]["win"])
    reproduction_difference = max(abs(float(p.probabilities[h])-float(saved_reference[p.race].probabilities[h])) for p in reproduced for h in p.probabilities)
    reproduction_bundle["metadata"].update(pipeline_version="0.3_reproduction", dataset_sha256=old["metadata"]["dataset_sha256"])
    rep_dir = safety.new_artifact("models")
    model.save_models(reproduction_bundle, rep_dir)
    freeze(rep_dir)
    reproduction = {"model_id": rep_dir.name, "max_absolute_prediction_difference": reproduction_difference,
        "training_rounds": reproduction_bundle["metadata"]["training_rounds"]["win"], "original_model_unchanged": True}
    del old_past, old_rows, reproduction_bundle
    raw_market = safety.read_json(collection / "market_inputs.local.json", root=ROOT)
    market_rows = defaultdict(list)
    for record in raw_market["odds_rows"]:
        market_rows[core.key(record["race_key"])].append(record)
    quotes, market_exclusions = defaultdict(list), Counter()
    for race in parts["test"]:
        quote, reason = market.select_final(race, market_rows[race], tuple(old_by_race[race].probabilities))
        if reason:
            market_exclusions[reason] += 1
        else:
            quotes[quote.basis].append(quote)
    result = {"kind": "step2_learning_curve", "plan": plan, "pool_rows": len(rows), "pool_races": metadata["races"],
        "dataset_id": dataset.parent.name, "partitions": partition_summary, "pool_exclusion_counts": exclusions,
        "source_collection": str(collection), "reproduction": reproduction, "models": [],
        "saved_v03": {"model_id": saved["model_id"], "prediction_id": old_prediction_dir.name,
            "best_iteration": old["metadata"]["best_iterations"]["win"],
            "primary": v03.probability_evaluation(old_test, winners, base_policy),
            "known_reference": v03.probability_evaluation(old_reference, winners, base_policy),
            "reference_saved_prediction_check": reference_check},
        "market_exclusions": dict(market_exclusions),
        "market_comparisons": {"saved_v03_T2": {basis: market.measure(old_test, winners, items, base_policy) for basis, items in quotes.items()}},
        "disabled_features": [name for name in features.FEATURES if name in ("going", "previous_time_seconds")],
        "live_purchase_enabled": False, "old_artifacts_unchanged": None}
    controlled_scores = {}
    def fit_window(size, selected, rounds, experiment):
        stage("fit_window", train_races=size, rounds=rounds, experiment=experiment)
        indices = sorted(i for name, races in parts.items() for race in (selected if name == "train" else races) for i in groups[race])
        selected_rows = [rows[i] for i in indices]
        prepared = [matrix[i] for i in indices]
        coverage = {name: v03.coverage([matrix[i] for race in (selected if name == "train" else races) for i in groups[race]]) for name, races in parts.items()}
        v03.check_training_coverage(coverage["train"])
        bundle = model.fit_models(selected_rows, rounds=rounds, targets=("win",), train_end=days["train_end"], valid_end=days["validation_end"],
            test_end=days["test_end"], calibration_start=days["calibration_start"], calibration_end=days["calibration_end"],
            test_start=days["test_start"], _prepared_features=prepared)
        if bundle["metadata"]["split"]["train"]["races"] != size:
            raise safety.MLFailure("invalid_split")
        bundle["metadata"].update(pipeline_version="0.3_step2", dataset_sha256=metadata["csv_sha256"],
            history_dataset_sha256=metadata["csv_sha256"], history_start=plan["history_start"],
            mapping_sha256=old["metadata"]["mapping_sha256"], source_plan_sha256=safety.digest(collection / "plan.json", root=ROOT),
            training_key_selection="last_k_complete_races_at_common_end", feature_cache="shared_unchanged_build_features",
            feature_coverage=coverage, experiment=experiment)
        model_directory = safety.new_artifact("models")
        selection_path = model_directory / "training_selection.local.json"
        safety.write_json(selection_path, {"race_keys": [list(r) for r in selected], "dataset_id": dataset.parent.name})
        selection_path.chmod(0o444)
        bundle["metadata"]["training_selection_sha256"] = safety.digest(selection_path)
        saved_meta = model.save_models(bundle, model_directory)
        freeze(model_directory)
        def partition(name):
            if name == "reference":
                indexes = [i for i, row in enumerate(selected_rows) if features.entry_identity(row)[0][:-1] in reference_keys]
            else:
                indexes = bundle["split"][name]
            return part_predictions(selected_rows, indexes, bundle["predictions"]["win"])
        calibration_predictions = partition("calibration")
        calibration = core.fit_temperature(calibration_predictions, winners, model_fit_end=plan["validation_end"], test_start=plan["test_start"])
        test_predictions, reference_predictions = partition("test"), partition("reference")
        model_hash = safety.digest(model_directory / "model.json")
        predictions_directory = save_predictions(test_predictions + reference_predictions, model_hash, dataset, purpose="step2_common_untrained_cohort")
        entry = {"experiment": experiment, "training_races": size, "model_id": model_directory.name,
            "prediction_id": predictions_directory.name, "training_rounds": saved_meta["training_rounds"]["win"],
            "training_parameters": saved_meta["training_parameters"], "training_rows": saved_meta["split"]["train"]["rows"],
            "train_start": saved_meta["split"]["train"]["start"], "train_end": saved_meta["split"]["train"]["end"],
            "feature_coverage": coverage, "calibration": calibration, "series": {},
            "before_temperature": v03.probability_evaluation(test_predictions, winners, replace(base_policy, temperature=1.0)),
            "raw_race_normalized_probability_distribution": curve.probability_distribution(test_predictions, replace(base_policy, temperature=1.0))}
        for series, temperature in (("fixed", 2.), ("recalibrated", calibration["temperature"])):
            policy = replace(base_policy, temperature=temperature, model_fit_end=plan["validation_end"],
                calibration_end=plan["calibration_end"] if series == "recalibrated" else None,
                calibration_source="independent_validation" if series == "recalibrated" else "fixed_not_fitted")
            policy_dir, _ = save_policy(policy, model_hash, calibration if series == "recalibrated" else {"temperature": 2., "origin": "predeclared_fixed"})
            scores = curve.race_scores(test_predictions, winners, policy)
            if size == windows[0][0] and experiment == "controlled_learning_size":
                controlled_scores[series] = scores
            item = {"temperature": temperature, "policy_id": policy_dir.name,
                "primary": v03.probability_evaluation(test_predictions, winners, policy),
                "known_reference": v03.probability_evaluation(reference_predictions, winners, policy),
                "calibration_metrics": v03.probability_evaluation(calibration_predictions, winners, policy),
                "probability_distribution": curve.probability_distribution(test_predictions, policy),
                "comparisons": {"saved_v03_T2": curve.paired_bootstrap(scores, old_scores),
                    "controlled_small_same_series": curve.paired_bootstrap(scores, controlled_scores[series])}}
            entry["series"][series] = item
            label = f"{size}_{experiment}_{series}"
            result["market_comparisons"][label] = {basis: market.measure(test_predictions, winners, items, policy) for basis, items in quotes.items()}
        checkpoint = directory / f"model_{size}_{experiment}.json"
        safety.write_json(checkpoint, entry, root=ROOT)
        checkpoint.chmod(0o444)
        result["models"].append(entry)
        del bundle, prepared, selected_rows
        stage("window_complete", train_races=size, model_id=model_directory.name,
            best_iteration=entry["training_rounds"]["best_iteration"], recalibrated_temperature=calibration["temperature"])
        return entry
    for size, selected in windows:
        fit_window(size, selected, 100, "controlled_learning_size")
    maximum = result["models"][-1]
    if maximum["training_rounds"]["reached_round_limit"]:
        fit_window(windows[-1][0], windows[-1][1], 300, "max_rounds_only_reference")
    result.update(status="ok", elapsed_seconds=time.monotonic() - started,
        old_artifacts_unchanged=all(safety.digest(path, root=root) == digest for path, root, digest in frozen),
        primary_cohort_consistent=all(entry["series"]["fixed"]["primary"]["races"] == len(parts["test"]) for entry in result["models"]))
    if not result["old_artifacts_unchanged"] or not result["primary_cohort_consistent"]:
        raise safety.MLFailure("artifact_invalid")
    safety.write_json(directory / "input_provenance.json", {"frozen_hashes": [{"path": str(p), "sha256": digest} for p, _, digest in frozen]}, root=ROOT)
    safety.write_json(directory / "run.json", result, root=ROOT)
    report(directory, result)
    with safety.local_path(directory / "learning_curve.csv", root=ROOT).open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ("experiment", "training_races", "model_id", "series", "temperature", "log_loss", "brier_score", "top1_accuracy_percent", "best_iteration", "max_rounds")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in result["models"]:
            for series, item in entry["series"].items():
                writer.writerow(dict(experiment=entry["experiment"], training_races=entry["training_races"], model_id=entry["model_id"], series=series,
                    temperature=item["temperature"], **{k: item["primary"]["model"][k] for k in ("log_loss", "brier_score", "top1_accuracy_percent")},
                    **{k: entry["training_rounds"][k] for k in ("best_iteration", "max_rounds")}))
    for path in directory.iterdir():
        if path.is_file():
            path.chmod(0o444)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline STEP 2 V2 controlled learning curves")
    parser.add_argument("--collection", type=Path, default=COLLECTION)
    args = parser.parse_args(argv)
    directory = safety.new_artifact("evaluations", root=ROOT)
    progress = {"evaluation_directory": str(directory)}
    try:
        result = run(args.collection, directory, progress)
        compact = {"status": "ok", "kind": result["kind"], "evaluation_directory": str(directory),
            "partitions": result["partitions"], "old_artifacts_unchanged": result["old_artifacts_unchanged"],
            "models": [{"training_races": m["training_races"], "experiment": m["experiment"], "model_id": m["model_id"],
                "training_rounds": m["training_rounds"], "temperatures": {k: s["temperature"] for k, s in m["series"].items()},
                "metrics": {k: s["primary"]["model"] for k, s in m["series"].items()}} for m in result["models"]]}
        print(json.dumps(compact, ensure_ascii=True), flush=True)
        return 0
    except SafeAuditError as exc:
        progress.update(status="stopped", reason=exc.code)
    except Exception:
        progress.update(status="stopped", reason="internal_error")
    safety.write_json(directory / "failure.json", progress, root=ROOT)
    print(json.dumps(progress, ensure_ascii=True), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
