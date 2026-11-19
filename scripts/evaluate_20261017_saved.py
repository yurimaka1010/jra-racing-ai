"""Offline evaluation of frozen morning predictions and owner-local result snapshots.

No database access, inference, fitting, calibration fitting or purchase execution.
All individual inputs and outputs remain in the existing private trial directory.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai import morning_trial as trial
from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.ml import features, market, safety, v03
from jra_racing_ai.win_sim import core, formats, local, settle_saved

DAY = date(2026, 10, 17)
ORIGINAL = trial.ROOT / "evaluations" / "9b2dc7a96aa24008a0dae3a0c34f31e2"
COLLECTION = trial.ROOT / "evaluations" / "cedfa962cd754b3d9eadc3a5756eeaed"
PRIOR_HR = local.ROOT / "simulations" / "b465a4023292420ba075385876d8d71c"


def load_predictions(policy):
    run = safety.read_json(ORIGINAL / "run.json", root=trial.ROOT)
    path = safety.local_path(ORIGINAL / "predictions.csv", exists=True, root=trial.ROOT)
    if (run.get("status") != "ok" or run.get("kind") != "morning_predictions"
            or run.get("temperature") != 2.0 or run.get("trial_date") != DAY.isoformat()
            or safety.digest(path, root=trial.ROOT) != run["predictions_sha256"]):
        raise safety.MLFailure("artifact_invalid")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    raw, calibrated, identities = defaultdict(dict), defaultdict(dict), {}
    for row in rows:
        identity, day = features.entry_identity(row)
        race, horse = identity[:-1], identity[-1]
        if identity in identities or day != DAY or row["model_id"] != run["model_id"] or row["temperature"] != "2.0":
            raise safety.MLFailure("artifact_invalid")
        if core.instant(row["predicted_at_jst"]) >= core.instant(row["scheduled_start_jst"]):
            raise safety.MLFailure("forward_cutoff")
        raw[race][horse] = core.decimal(row["raw_win_score"])
        calibrated[race][horse] = core.decimal(row["win_probability"])
        identities[identity] = row["horse_id"]
    if len(rows) != run["runner_count"] or len(raw) != run["race_count"]:
        raise safety.MLFailure("artifact_invalid")
    # Recompute only as an integrity check. Stored probabilities stay unchanged.
    for race in raw:
        checked, _ = core.probabilities(raw[race], policy)
        if (abs(sum(calibrated[race].values()) - 1) > core.decimal("1e-12")
                or any(abs(checked[h] - calibrated[race][h]) > core.decimal("1e-12") for h in checked)):
            raise safety.MLFailure("invalid_probability")
    return run, raw, calibrated, identities


def verified_outcomes(payload, races):
    manifest = local.read(PRIOR_HR / "manifest.json")
    if (local.sha(PRIOR_HR / "hr_payout_source.json") != manifest["provenance"]["hr_source_sha256"]
            or manifest["provenance"].get("hr_source_confirmed") is not True
            or manifest["provenance"].get("odds_raw_format_and_scale_human_confirmed") is not True):
        raise safety.MLFailure("artifact_invalid")
    source = deepcopy(formats.validate_payout_source(local.read(PRIOR_HR / "hr_payout_source.json")))
    if source["numerator"] != 1 or source["denominator"] != 1:
        raise safety.MLFailure("mapping_not_reviewed")
    # Reuse the owner's checked physical columns and JPY/100 normal-win units.
    # This evaluation's scope is new; no old profile or decision is overwritten.
    source.update(years=[DAY.year], venues=sorted({r[2] for r in races}), race_keys=[list(r) for r in races])
    return settle_saved.decode_existing_hr(payload["hr_rows"], source)


def top_horse(probabilities):
    return min(probabilities, key=lambda h: (-probabilities[h], h))


def evaluate(payload, anomalies, raw, calibrated, identities, policy):
    outcomes = {o.race: o for o in verified_outcomes(payload, tuple(sorted(raw)))}
    se, odds = defaultdict(dict), defaultdict(list)
    for row in payload["se_rows"]:
        identity, _ = features.entry_identity(row)
        if (identity not in identities or identities[identity] != row["horse_id"]
                or identity[-1] in se[identity[:-1]]):
            raise safety.MLFailure("invalid_identity")
        se[identity[:-1]][identity[-1]] = row
    anomaly = {}
    for row in anomalies["rows"]:
        identity, _ = features.entry_identity(row)
        if identity not in identities or identity in anomaly or row["anomaly_code"] not in tuple("1234567"):
            raise safety.MLFailure("invalid_input")
        anomaly[identity] = row["anomaly_code"]
    for row in payload["odds_rows"]:
        odds[core.key(row["race_key"])].append(row)
    winners, exclusions, valid, quotes, price_failures = {}, Counter(), {}, {}, Counter()
    for race, probabilities in calibrated.items():
        roster = tuple(probabilities)
        outcome = outcomes.get(race)
        if core.outcome_reason(outcome, roster):
            exclusions["hr_not_ordinary_confirmed"] += 1
            continue
        official = set(outcome.payouts_yen_per_100)
        se_winners = {h for h, r in se[race].items() if features.number(r["finish_position"]) == 1}
        if (set(se[race]) != set(roster) or official != se_winners
                or any(r["status"] not in ("5", "6", "7") for r in se[race].values())):
            exclusions["se_hr_alignment_unconfirmed"] += 1
            continue
        valid[race] = outcome
        quote, reason = market.select_final(race, odds[race], roster)
        if reason:
            price_failures[reason] += 1
        else:
            quotes[race] = quote
        if any(anomaly.get((*race, h), "0") in ("1", "2", "3") for h in roster):
            # Do not erase a morning runner or renormalize frozen probabilities.
            exclusions["nonstarter_race_fixed_probability_cohort"] += 1
            continue
        if len(official) != 1:
            exclusions["dead_heat_single_winner_metrics"] += 1
            continue
        unknown = [h for h, r in se[race].items() if features.number(r["finish_position"], low=1, high=28) is None
                   and anomaly.get((*race, h)) not in ("4", "5")]
        if unknown:
            exclusions["rank_or_nonfinish_status_unconfirmed"] += 1
            continue
        winners[race] = next(iter(official))
    predictions = [core.PredictionRace(race, raw[race]) for race in sorted(winners)]
    full = v03.probability_evaluation(predictions, winners, policy)
    by_basis = defaultdict(list)
    for race in winners:
        if race in quotes:
            by_basis[quotes[race].basis].append(quotes[race])
    cohorts = {basis: market.measure(predictions, winners, items, policy) for basis, items in by_basis.items()}
    # Independent sklearn check uses the SAVED calibrated probabilities, not T again.
    from sklearn.metrics import log_loss, brier_score_loss
    labels = [int(h == winners[r]) for r in sorted(winners) for h in sorted(calibrated[r])]
    probabilities = [float(calibrated[r][h]) for r in sorted(winners) for h in sorted(calibrated[r])]
    independent = {"log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
                   "brier_score": float(brier_score_loss(labels, probabilities))}
    if any(abs(independent[k] - full["model"][k]) > 1e-12 for k in independent):
        raise safety.MLFailure("artifact_invalid")
    top_rows, reference_rows, private = [], [], []
    cancelled_selections, ties = 0, 0
    for race in sorted(valid):
        p, outcome = calibrated[race], valid[race]
        chosen = top_horse(p)
        ties += sum(value == max(p.values()) for value in p.values()) > 1
        cancelled = anomaly.get((*race, chosen)) in ("1", "2", "3")
        cancelled_selections += cancelled
        ai_outcome = core.Outcome(race, "pending", reason="refund_pending") if cancelled else outcome
        quote = quotes.get(race)
        favorite = min(quote.odds, key=lambda h: (quote.odds[h], h)) if quote else None
        row = {"ai": core.settle(chosen, ai_outcome, tuple(p)), "favorite": core.settle(favorite, outcome, tuple(p))}
        top_rows.append(row)
        details = {"race_key": list(race), "morning_top_horse": chosen, "top1": row, "mode": "postrace_reference_only"}
        if race in winners and quote is not None:
            ev = {h: p[h] * core.decimal(quote.odds[h]) - 1 for h in p}
            candidates = sorted((h for h in p if ev[h] >= core.decimal(policy.threshold)), key=lambda h: (-ev[h], h))
            selected = candidates[0] if candidates else None
            evrow = {"ai": core.settle(selected, outcome, tuple(p)), "favorite": core.settle(favorite, outcome, tuple(p))}
            reference_rows.append(evrow)
            details.update(final_odds_ai_horse=selected, final_odds_favorite_horse=favorite,
                           selected_ev=float(ev[selected]) if selected else None, final_odds_reference=evrow,
                           final_odds_basis=quote.basis, decision_at=None, observed_at=None)
        private.append(details)
    common = [row for row in top_rows if row["favorite"]["status"] == "settled" and row["ai"]["status"] == "settled"]
    selected_ev = [d["selected_ev"] for d in private if d.get("selected_ev") is not None]
    result = {"target_race_count": len(raw), "target_runner_count": len(identities),
        "probability_metrics": full, "probability_exclusion_reasons": dict(exclusions),
        "market_cohorts": cohorts, "final_odds_available_race_count": len(quotes),
        "final_odds_acquisition_rate_percent": 100 * len(quotes) / len(raw), "final_odds_exclusions": dict(price_failures),
        "forward_ev_evaluation": {"status": "not_available", "reason": "no_frozen_purchase_decisions_and_no_pre_race_odds",
            "pre_race_odds_row_count": sum(r.get("status") == "1" for r in payload["odds_rows"]),
            "official_purchase_execution_count": 0},
        "top1_flat_100_reference": {"evaluation_type": "postrace_hypothetical_top1_not_saved_purchase_decisions",
            "ai": core.financial(top_rows, "ai"), "cancelled_ai_selections_held": cancelled_selections,
            "same_races_comparison": {"race_count": len(common), "ai": core.financial(common, "ai"),
                                      "final_favorite": core.financial(common, "favorite")}},
        "final_odds_ev_reference": {"evaluation_type": "postrace_final_odds_not_forward_ev_performance",
            "threshold": policy.threshold, "stake_yen": 100, "max_bets_per_race": 1, "race_count": len(reference_rows),
            "ai": core.financial(reference_rows, "ai"), "final_favorite": core.financial(reference_rows, "favorite"),
            "selected_ev_distribution": core.distribution(selected_ev)},
        "anomaly_counts": dict(Counter(anomaly.values())), "top1_tie_race_count": ties,
        "tie_break_for_virtual_tickets": "lowest_horse_number",
        "sp_comparison": {"status": "out_of_scope_by_user_instruction"},
        "verification": {"sklearn_independent_metrics": independent, "double_calibration_applied": False,
            "saved_probabilities_unchanged": True, "se_hr_winners_checked": len(valid)}}
    return result, private


def number(value, digits=4):
    return "未算出" if value is None else f"{value:.{digits}f}"


def write_report(directory, result):
    full = result["probability_metrics"]
    top = result["top1_flat_100_reference"]
    ev = result["final_odds_ev_reference"]
    lines = ["# 2026年10月17日 V0.3無人試験運用の結果検証", "",
        "朝7時の予測保存は成功。V0.3モデル・温度T=2.0・保存済み予測・予約タスクは変更していません。",
        "Windowsタスク最終実行：" + result["windows_task"]["last_run_time"] + "、終了コード：" + str(result["windows_task"]["last_task_result"]) + "。予測保存：" + result["prediction_saved_at_jst"] + "。1回実行済みで次回予定はありません。", "",
        "対象は36レース・472頭。SE区分6とHR区分1を完全キーと馬番・馬IDで照合しました。HRは払戻金確定速報の現在保存版です。月曜成績や将来訂正の検証済み認定ではありません。",
        "単勝の不成立・特払・返還フラグは全36レースで0。金額は利用者が既に確認したPC列のJPY/100円対応と既存HRデコーダー・精算関数を再利用しました。最終オッズから払戻金を作っていません。", "",
        "出走取消1頭と競走中止1頭を区別しました（SE異常区分1/4、JV-Dataコード表2101）。取消を通常の不的中として確率評価に混ぜず、該当1レース・朝の9頭集合を除外しています。保存勝率の削除・再正規化はしていません。競走中止馬は出走済みの非勝馬として残しています。", "",
        f"## 予測性能：同じ{full['races']}レース・{full['model']['count']}頭", "",
        "Log LossとBrierは従来と同じ馬単位二値指標。予測1位が同率の場合、従来の確率指標は同率按分、仮想購入は最小馬番です。今回の同率1位レース数：" + str(result["top1_tie_race_count"]) + "。", "",
        "| 予測 | Log Loss | Brier | 予測1位的中率% | 勝馬平均順位 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in (("V0.3 T=2.0", full["model"]), ("一様予測", full["uniform"])):
        lines.append("| " + name + " | " + " | ".join(number(metrics[k], 6) for k in
            ("log_loss", "brier_score", "top1_accuracy_percent", "winner_mean_rank")) + " |")
    for basis, cohort in result["market_cohorts"].items():
        metrics = cohort["metrics"]["market"]
        lines.append("| 市場 " + basis + " | " + " | ".join(number(metrics[k], 6) for k in
            ("log_loss", "brier_score", "top1_accuracy_percent", "winner_mean_rank")) + " |")
        lines += ["", "市場は確定O1区分4の1/オッズをレース内で正規化した事後参考比較です。発走前の市場を意味しません。発走前履歴は0件であり、確定レコードの発表時計を推定していません。", "",
            "| 確率帯 | 予測 | 頭数 | 平均予測% | 実勝率% |", "|---|---|---:|---:|---:|"]
        for key in ("v03_calibrated", "market"):
            for item in cohort["calibration"][key]:
                lines.append("| " + item["probability_band"] + " | " + key + " | " + str(item["count"]) + " | " +
                    number(item["mean_probability"] * 100 if item["mean_probability"] is not None else None) + " | " +
                    number(item["observed_rate"] * 100 if item["observed_rate"] is not None else None) + " |")
        lines += ["", "レース内Spearman順位相関の平均：" + number(cohort["rank_relationship"]["spearman_per_race"]["mean"]) +
                  "、AI・市場の1位一致率：" + number(cohort["rank_relationship"]["top1_agreement_percent"]) + "% 。"]
    lines += ["", "## 仮想収支（いずれも今回追加した事後参考集計）", "",
        "朝の予測処理はEV・仮想購入判断を保存していません。当日事前に記録済みの購入推奨や約定・実購入の実績はありません。予測1位100円の集計も、今回定義した参考方針です。", "",
        "| 参考方針 | 購入 | 的中 | 投資円 | 払戻円 | 純利益円 | 回収率% | 未精算 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    comparisons = top["same_races_comparison"]
    values = [("朝予測1位・全36", top["ai"]),
        ("朝予測1位・共通" + str(comparisons["race_count"]), comparisons["ai"]),
        ("確定1番人気・同じ共通集合", comparisons["final_favorite"]),
        ("確定オッズEV>=5%参考", ev["ai"]), ("確定1番人気・EV参考対象集合", ev["final_favorite"])]
    for name, metric in values:
        lines.append("| " + name + " | " + " | ".join(str(metric[k]) if metric[k] is not None else "未算出" for k in
            ("bet_count", "hit_count", "investment_yen", "payout_yen", "net_profit_yen", "return_rate_percent", "unsettled_count")) + " |")
    lines += ["", "EV参考は固定済み朝勝率×確定オッズ−1、従来閾値5%以上、最大1点100円、最大EV選択です。確定オッズは購入時点に利用できない事後情報であり、この収支を発走10分前EVの運用実績とは扱えません。決済には公式HR額を使っています。",
        "選択馬の推定EV中央値は" + number(ev["selected_ev_distribution"]["median"] * 100, 2) + "%です。最小でも" + number(ev["selected_ev_distribution"]["min"] * 100, 2) + "%で対象35レースすべてが購入候補となっています。大きな推定EVをモデルの正しさとは解釈しません。高回収率は2的中の結果に依存します。",
        "朝予測1位の参考集計で取消馬を選択していた場合は返還未確認として保留します。今回の選択取消頭数：" + str(top["cancelled_ai_selections_held"]) + "。",
        "購入0件の回収率は未算出。未精算があれば全体純利益は確定せず、通常ケースだけを別集計します。", "",
        "## 優先課題", "",
        "1. 小規模・短期の学習範囲を広げ、時系列で学習・早期終了・校正・最終評価を分離する。今回の1日を見て閾値や温度を後付け変更しない。",
        "2. 実取得時刻付きの発走前オッズと、変更不可の購入・見送り判断を保存する。取消を反映した入力集合の管理も必要。現状は前向き予測だけ成功し、前向きの収益検証は未実施。",
        "3. 確率精度・市場との差を複数日/月で確認してから、EV選択と月間純利益・回収率・月間変動を評価する。1日の収支やEV表示の大きさを成功条件にしない。",
        "4. 朝の実行ログで馬場状態・過去タイムが全件欠損。意図的な未取込項目と欠損を区別し、単位と事前利用可能性を確認後に段階的改善する。モデルは今回変更していない。", "",
        "SP指数比較は利用者の指示により対象外。外部指数への代用はしていません。",
        "馬単位のLog Loss/Brierはscikit-learnでも独立再計算して一致を確認しました。個別台帳・元入力はローカル限定です。",
        "公式区分の根拠：[JV-Data 4.9.0.1（ローカルPDF）](C:/Users/acbd5/AppData/Local/JraRacingAI/references/JRA-VAN/JV-Data4901.pdf)、PDF12ページと46ページ・コード表2101。",
        "元の朝予測：[run.json](" + str(ORIGINAL / "run.json").replace("\\", "/") + ")、[predictions.csv](" + str(ORIGINAL / "predictions.csv").replace("\\", "/") + ")。",
        "正式な集計は本レポートとsummary.jsonです。収集フォルダーのpreliminary_summary.jsonは取消レース除外前の試算であり、本集計に置き換わります。", ""]
    path = safety.local_path(directory / "report.md", root=trial.ROOT)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def main():
    result = {"status": "stopped", "kind": "trial_20261017_postrace_evaluation"}
    try:
        _, _, _, policy, frozen = trial.load_source(DAY)
        run, raw, calibrated, identities = load_predictions(policy)
        inputs = COLLECTION / "results_input.local.json"
        anomaly_file = COLLECTION / "anomaly_entries.local.json"
        task_file = COLLECTION / "task_observation.json"
        frozen += [(p, trial.ROOT, safety.digest(p, root=trial.ROOT)) for p in
            (inputs, anomaly_file, task_file, ORIGINAL / "run.json", ORIGINAL / "predictions.csv")]
        payload = safety.read_json(inputs, root=trial.ROOT)
        anomalies = safety.read_json(anomaly_file, root=trial.ROOT)
        task = safety.read_json(task_file, root=trial.ROOT)
        summary, private = evaluate(payload, anomalies, raw, calibrated, identities, policy)
        result.update(summary, status="ok", evaluated_at_jst=trial.now().isoformat(), temperature=2.0,
            model_id=run["model_id"], prediction_saved_at_jst=run["predicted_at_jst"],
            windows_task=task,
            model_training_race_count=144, training_count_source="existing_v03_training_run",
            interpretation="single_day_forward_predictions_with_postrace_reference_analysis",
            source_run_directory=str(ORIGINAL), source_collection_directory=str(COLLECTION))
        if not all(safety.digest(p, root=root) == digest for p, root, digest in frozen):
            raise safety.MLFailure("artifact_invalid")
        result["verification"]["frozen_source_hashes_unchanged"] = True
        directory = safety.new_artifact("evaluations", root=trial.ROOT)
        result["report_directory"] = str(directory)
        safety.write_json(directory / "summary.json", result, root=trial.ROOT)
        safety.write_json(directory / "reference_ledger.local.json", {"records": private, "real_purchase": False}, root=trial.ROOT)
        safety.write_json(directory / "input_provenance.json", {"hashes": [{"path": str(p), "sha256": d}
            for p, _, d in frozen], "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
            "hr_unit_basis": "owner_confirmed_20261004_physical_columns_JPY_per_100_reused_for_ordinary_20261017_only",
            "preliminary_summary_is_superseded": True}, root=trial.ROOT)
        write_report(directory, result)
        for path in directory.iterdir():
            if path.is_file():
                path.chmod(0o444)
    except SafeAuditError as exc:
        result.update(status="stopped", reason=exc.code)
    except Exception:
        result.update(status="stopped", reason="internal_error")
    compact = {k: v for k, v in result.items() if k not in ("market_cohorts", "probability_metrics")}
    if result["status"] == "ok":
        compact["metrics"] = {"races": result["probability_metrics"]["races"],
            "v03": result["probability_metrics"]["model"], "uniform": result["probability_metrics"]["uniform"],
            "market": {k: v["metrics"]["market"] for k, v in result["market_cohorts"].items()}}
    print(json.dumps(compact, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
