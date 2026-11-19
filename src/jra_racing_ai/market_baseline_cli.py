"""User-operated market measurement against frozen Version 0.3 predictions."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

from .core import SafeAuditError
from .db_connection import read_only_session
from .ml import data, features, market, safety, v03
from .ml_v03_cli import ROOT
from .win_sim import core, local, publication
from .win_sim_cli import connection_config, password

DEFAULT_RUN = ROOT / "evaluations" / "c62408217a6146b1bf91f88e6f902631" / "run.json"


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise safety.MLFailure("invalid_input")


def parser():
    result = Parser(description="Measure market probabilities; no training or betting")
    result.add_argument("--run", type=Path, default=DEFAULT_RUN)
    result.add_argument("--execute-local", action="store_true")
    result.add_argument("--max-rows", type=int, default=50000)
    result.add_argument("--max-seconds", type=int, default=180)
    result.add_argument("--statement-timeout-ms", type=int, default=60000)
    return result


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise safety.MLFailure("artifact_invalid")
    return value


def load_frozen(path):
    saved = safety.read_json(path, root=ROOT)
    if saved.get("kind") != "v03_first_run" or saved.get("status") != "ok":
        raise safety.MLFailure("artifact_invalid")
    model_directory = safety.ML_ROOT / "models" / _id(saved["model_id"])
    model_info, model_hash = local.model_info(model_directory)
    if model_info.get("features") != list(features.FEATURES):
        raise safety.MLFailure("artifact_invalid")
    win_path = model_directory / "win.lgb.txt"
    if safety.digest(win_path) != model_info["artifacts"]["win.lgb.txt"]:
        raise safety.MLFailure("artifact_invalid")
    dataset = safety.ML_ROOT / "datasets" / _id(saved["dataset_id"]) / "runners.csv"
    rows, metadata = data.read_dataset(dataset)
    if metadata["csv_sha256"] != model_info.get("dataset_sha256"):
        raise safety.MLFailure("artifact_invalid")
    predictions_directory = local.ROOT / "predictions" / _id(saved["heldout_predictions_id"])
    predictions, manifest = local.load_predictions(predictions_directory, expected_model_hash=model_hash)
    if (manifest.get("purpose") != "heldout_probability_test" or manifest.get("pipeline_version") != "0.3"
            or manifest.get("entries_sha256") != metadata["csv_sha256"]):
        raise safety.MLFailure("artifact_invalid")
    part = saved["split"]["test"]
    start, end = date.fromisoformat(part["start"]), date.fromisoformat(part["end"])
    targets = [row for row in rows if start <= features.entry_identity(row)[1] <= end]
    expected = {}
    for row in targets:
        identity, _ = features.entry_identity(row)
        expected.setdefault(identity[:-1], set()).add(identity[-1])
    if (len(targets) != part["rows"] or len(expected) != part["races"]
            or {p.race: set(p.probabilities) for p in predictions} != expected):
        raise safety.MLFailure("artifact_invalid")
    policy_path = local.ROOT / "policies" / _id(saved["policy_id"]) / "policy.json"
    policy, document = local.load_policy(policy_path)
    if (document["model_manifest_sha256"] != model_hash or document.get("calibration") != saved["calibration"]
            or policy.temperature != saved["calibration"]["temperature"]):
        raise safety.MLFailure("artifact_invalid")
    policy = replace(policy, evaluation_start=start.isoformat(), evaluation_end=end.isoformat()).validate()
    winners = v03.single_winners(targets)
    eligible = [p for p in predictions if p.race in winners]
    full = v03.probability_evaluation(eligible, winners, policy)
    original = saved["heldout_test"]["v03_calibrated"]
    if (full.get("status") != "ok" or full["races"] != original["races"]
            or full["model"]["count"] != original["model"]["count"]
            or any(abs(full["model"][key] - original["model"][key]) > 1e-10 for key in
                   ("log_loss", "brier_score", "top1_accuracy_percent", "winner_mean_rank"))):
        raise safety.MLFailure("artifact_invalid")
    paths = [(path, ROOT), (model_directory / "model.json", safety.ML_ROOT), (win_path, safety.ML_ROOT),
             (dataset, safety.ML_ROOT), (dataset.parent / "dataset.json", safety.ML_ROOT),
             (predictions_directory / "predictions.json", local.ROOT), (predictions_directory / "manifest.json", local.ROOT),
             (policy_path, local.ROOT)]
    frozen = [(p, root, safety.digest(p, root=root)) for p, root in paths]
    schema = {name: model_info.get(name) for name in ("features", "categories", "history_lag_days")}
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return {"saved": saved, "predictions": eligible, "winners": winners, "policy": policy,
            "full": full, "frozen": frozen, "provenance": {"model_id": saved["model_id"],
                "model_manifest_sha256": model_hash, "feature_schema_sha256": schema_hash,
                "model_training_period": model_info["split"]["train"], "temperature": policy.temperature,
                "prediction_artifact_id": saved["heldout_predictions_id"], "dataset_id": saved["dataset_id"],
                "test_period": part, "source_run_sha256": safety.digest(path, root=ROOT)}}


def fetch(db, races, *, max_rows, max_seconds):
    """Existing SELECT adapter, restricted to the exact evaluation race keys."""
    budget, rows, statuses = local.Budget(max_seconds), [], []
    for table in ("apd_sokuho_o1", "jvd_o1"):
        columns = publication._columns(db, table, publication.O1_FIELDS, budget)
        found = []
        if columns is not None:
            found = local._rows(db, budget, table, publication.O1_FIELDS, columns, races, max_rows - len(rows), join_targets=True)
        for row in found:
            row["source_table"] = table
        rows.extend(found)
        statuses.append({"table": table, "status": "bounded_read_complete" if columns is not None else "not_found",
                         "rows": len(found), "category_counts": dict(Counter(row["status"] if row.get("status") in tuple("0123459")
                                                                           else "unknown" for row in found))})
        if len(rows) >= max_rows:
            raise safety.MLFailure("input_limit")
    ra_rows = []
    if any(row.get("status") == "1" for row in rows):
        columns = publication._columns(db, "jvd_ra", publication.RA_FIELDS, budget)
        if columns is not None:
            ra_rows = local._rows(db, budget, "jvd_ra", publication.RA_FIELDS, columns, races, max_rows - len(rows), join_targets=True)
        statuses.append({"table": "jvd_ra", "status": "bounded_read_complete" if columns is not None else "not_found", "rows": len(ra_rows)})
    return rows, ra_rows, statuses


def review_format(directory, rows):
    examples = []
    for table in ("apd_sokuho_o1", "jvd_o1"):
        candidates = [r for r in rows if r["source_table"] == table and r.get("status") in tuple("1345")
                      and isinstance(r.get("payload"), str) and len(r["payload"]) == 224]
        if candidates:
            row = candidates[0]
            examples.append({"source_table": table, "data_category": row["status"], "raw_win_block": row["payload"]})
    if examples:
        path = directory / "raw_format_review.local.json"
        safety.write_json(path, {"field_mapping": "public.<source_table>.odds_tansho -> raw_win_block",
            "examples": examples, "expected": "28 slots * 8 chars; horse 2 / odds 4 / popularity 2; raw odds / 10"}, root=ROOT)
        publication.confirm_raw_format(path)
        path.chmod(0o444)


def _show(value, *, percent=False):
    return "算出不能" if value is None else str(round(value * (100 if percent else 1), 6))


def write_report(directory, result):
    full = result["original_evaluation"]
    lines = ["# V0.3と市場確率の固定データ比較", "",
        "市場単独の実測です。モデル・温度・購入条件を変更せず、保存済みの予測と正解ラベルを再利用しました。", "",
        f"元の評価対象: {result['target_race_count']}レース・{result['target_runner_count']}頭。",
        "馬単位二値Log Loss / Brierです。市場確率は(1/odds)をレース内で正規化しています。", "",
        "| 元の全対象 | Log Loss | Brier | 予測1位的中率% | 勝馬平均順位 |", "|---|---:|---:|---:|---:|"]
    for name, value in (("V0.3校正後", full["model"]), ("一様予測", full["uniform"])):
        lines.append("| " + name + " | " + " | ".join(_show(value[k]) for k in
            ("log_loss", "brier_score", "top1_accuracy_percent", "winner_mean_rank")) + " |")
    for name, cohort in result["cohorts"].items():
        lines += ["", "## " + name, ""]
        if cohort["status"] != "ok":
            lines.append("利用できる価格がなく未算出です。補完していません。")
            continue
        timing = cohort["timing"]
        lines += [f"同じ{cohort['race_count']}レース・{cohort['runner_count']}頭で比較。元の全対象とは別の部分集合になる場合があります。",
            f"発走前は当初予定{result['provenance']['minutes_before_start']}分前までに発表分が終了した中間オッズ。実取得時刻・購入可能時刻は未検証です。"
            if name == "pre_race_publication_time" else
            "確定・最終オッズの事後参考比較です。発走前に利用可能だった価格や収益実績を意味しません。正確な発表時計は最終系レコードに定義されていないため推定しません。",
            "使用表件数: " + json.dumps(timing["source_table_counts"], ensure_ascii=False) + "、区分件数: " + json.dumps(timing["category_counts"]), "",
            "| 比較対象 | Log Loss | Brier | 予測1位的中率% | 勝馬平均順位 |", "|---|---:|---:|---:|---:|"]
        for label, metric in cohort["metrics"].items():
            lines.append("| " + label + " | " + " | ".join(_show(metric[k]) for k in
                ("log_loss", "brier_score", "top1_accuracy_percent", "winner_mean_rank")) + " |")
        lines += ["", "| 確率帯 | 予測 | 頭数 | 勝数 | 平均予測% | 実勝率% | 予測/実勝率 |", "|---|---|---:|---:|---:|---:|---:|"]
        for index in range(6):
            for predictor, bins in cohort["calibration"].items():
                item = bins[index]
                lines.append("| " + item["probability_band"] + " | " + predictor + " | " +
                    str(item["count"]) + " | " + str(item["wins"]) + " | " + _show(item["mean_probability"], percent=True) + " | " +
                    _show(item["observed_rate"], percent=True) + " | " + _show(item["predicted_to_observed_ratio"]) + " |")
        relation = cohort["rank_relationship"]
        lines += ["", "Spearmanはレース単位の平均順位（同率は平均順位）で計算し、レースを等重みで平均しています。",
            "Spearman平均: " + _show(relation["spearman_per_race"].get("mean")) +
            "、算出不能レース数: " + str(relation["undefined_spearman_races"]) + "。",
            "AIと市場の予測1位一致率: " + _show(relation["top1_agreement_percent"]) +
            "%（同率1位は両者とも最小馬番を採用）。的中率は既存評価と同じ同率配分です。",
            "Claudeの未測定概算0.21との差（実測−概算）: " + _show(cohort["comparison_with_unmeasured_estimate"]["difference"]) + "。",
            "市場−V0.3のLog Loss差: " + _show(cohort["market_minus_model_log_loss"]) +
            "（負なら市場の損失が小さい）。"]
    lines += ["", "## 欠損と解釈", "", json.dumps(result["exclusions"], ensure_ascii=False), "",
        "発走前と最終系、最終系の区分3・4・5は別集計です。欠損した馬を落として比較集合を変えず、価格が揃わないレース全体を除外します。",
        "評価対象外の馬番に明示された取消・除外の非価格記号だけを取り除き、対象出走馬集合は固定しています。",
        "各確率帯への所属は各予測ごとに決まります。勝数0の帯の比は算出不能です。",
        "約0.21は許容範囲の定義がない概算なので、合否判定に使いません。値と差を実測として報告します。",
        "小規模な期間の点推定です。市場より良くても収益優位を証明せず、悪くても全条件で情報がないとは断定しません。",
        "次の検討は市場との確率帯・順位の差を確認した後に行います。この実行ではブレンド、重み最適化、購入条件変更、再学習、再校正、収支計算を行いません。",
        "個別の価格・レース情報はselected_inputs.local.jsonとraw_format_review.local.jsonに限定します。チャットや外部へ共有しないでください。", ""]
    path = safety.local_path(directory / "report.md", root=ROOT)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    path.chmod(0o444)


def run(args, progress):
    if not args.execute_local:
        raise safety.MLFailure("explicit_execution_required")
    if not (100 <= args.max_rows <= 100000 and 10 <= args.max_seconds <= 600 and 1000 <= args.statement_timeout_ms <= 60000):
        raise safety.MLFailure("input_limit")
    progress["stage"] = "load_frozen_evaluation"
    frozen = load_frozen(args.run)
    races = tuple(p.race for p in frozen["predictions"])
    if not 1 <= len(races) <= 200:
        raise safety.MLFailure("input_limit")
    directory = safety.new_artifact("evaluations", root=ROOT)
    config, secret = connection_config(args.statement_timeout_ms), password()
    try:
        progress["stage"] = "read_odds"
        with read_only_session(config, secret) as db:
            rows, ra_rows, sources = fetch(db, races, max_rows=args.max_rows, max_seconds=args.max_seconds)
    finally:
        secret = None
    progress["stage"] = "format_confirmation"
    review_format(directory, rows)
    progress["stage"] = "market_measurement"
    cohorts, exclusions = market.select_cohorts(frozen["predictions"], rows, ra_rows, frozen["policy"])
    results = {name: market.measure(frozen["predictions"], frozen["winners"], quotes, frozen["policy"]) for name, quotes in cohorts.items()}
    private = [{"basis": name, "race_key": list(q.race), "source_table": q.source, "data_category": q.category,
        "published_at": q.published_at.isoformat() if q.published_at is not None else None,
        "minutes_before_start": q.minutes_before_start, "odds": {h: str(o) for h, o in q.odds.items()}}
        for name, quotes in cohorts.items() for q in quotes]
    input_path = directory / "selected_inputs.local.json"
    safety.write_json(input_path, {"local_only": True, "records": private}, root=ROOT)
    input_path.chmod(0o444)
    for path, root, digest in frozen["frozen"]:
        if safety.digest(path, root=root) != digest:
            raise safety.MLFailure("artifact_invalid")
    full = frozen["full"]
    largest = max(c["race_count"] for c in results.values())
    result = {"status": "ok", "kind": "v03_market_baseline", "artifact_id": directory.name,
        "measurement_status": "complete" if largest == full["races"] else "partial" if largest else "odds_unavailable",
        "target_race_count": full["races"], "target_runner_count": full["model"]["count"],
        "existing_label_selection": {"source_test_races": frozen["saved"]["split"]["test"]["races"],
            "source_test_runners": frozen["saved"]["split"]["test"]["rows"],
            "previously_excluded_races": frozen["saved"]["heldout_test"]["v03_calibrated"]["excluded_races"]},
        "original_evaluation": full, "cohorts": results, "exclusions": exclusions, "db_sources": sources,
        "provenance": dict(frozen["provenance"], selected_inputs_sha256=safety.digest(input_path, root=ROOT),
            odds_columns={k: v[0] for k, v in publication.O1_FIELDS.items()},
            query_adapter="win_sim.local.build_rows_query exact six-key parameterized SELECT JOIN VALUES",
            final_selection="category 5 then 4 then 3; separate cohorts; same-category duplicates/conflicts excluded",
            pre_race_selection="latest publication minute fully before the configured scheduled-start cutoff",
            minutes_before_start=frozen["policy"].minutes_before_start,
            final_timestamps="not_defined_not_inferred", raw_price_scale="1/10", market_formula="(1/odds) / sum(1/odds)",
            metric_definition="horse_level_binary_log_loss_natural_log_and_brier",
            source_document="JRA競馬予想AI_V0.3_市場確率ベースライン検証レポート.docx section 5; blending excluded"),
        "source_artifacts_unchanged": True, "model_and_strategy_changed": False, "report_directory": str(directory)}
    safety.write_json(directory / "summary.json", result, root=ROOT)
    (directory / "summary.json").chmod(0o444)
    write_report(directory, result)
    return result


def main(argv=None):
    progress = {"stage": "preflight"}
    try:
        result = run(parser().parse_args(argv), progress)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (SafeAuditError, core.SimulationError, safety.MLFailure) as exc:
        print(json.dumps(dict(status="stopped", reason=exc.code, **progress), ensure_ascii=True))
        return 2
    except KeyboardInterrupt:
        print(json.dumps(dict(status="stopped", reason="interrupted", **progress)))
        return 130
    except Exception:
        print(json.dumps(dict(status="stopped", reason="internal_error", **progress)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
