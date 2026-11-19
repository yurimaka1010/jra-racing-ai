"""Save today's pre-start predictions using the frozen STEP 2 candidate only."""
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jra_racing_ai import morning_trial as trial
from jra_racing_ai.core import RACE_KEY, SafeAuditError
from jra_racing_ai.db_connection import read_only_session
from jra_racing_ai.ml import data, features, model, safety, v03
from jra_racing_ai.win_sim import core, formats, local, publication
from jra_racing_ai.win_sim_cli import connection_config

DAY = date(2026, 10, 18)
ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\trial-20261018")
MODEL_ID = "a8178c2b7523499f9463c1743b41c715"
POLICY_ID = "921ebbb9f33a49bbb513fe104e1e375b"
MAPPING_ID = "0543279477dd44f3a84fc36bcc61841b"
FIELDS = (*RACE_KEY, "race_date", "scheduled_start_jst", "horse_number", "horse_id", "horse_name",
          "win_probability", "prediction_rank", "raw_win_score", "model_id", "model_version",
          "temperature", "predicted_at_jst")


def load_source():
    model_dir = safety.ML_ROOT / "models" / MODEL_ID
    mapping_path = safety.ML_ROOT / "schema" / MAPPING_ID / "mapping.reviewed.json"
    policy_path = local.ROOT / "policies" / POLICY_ID / "policy.json"
    bundle = model.load_models(model_dir, required_targets=("win",))
    mapping = data.validate_mapping(safety.read_json(mapping_path))
    policy, document = local.load_policy(policy_path)
    metadata = bundle["metadata"]
    if (metadata.get("pipeline_version") != "0.3_step2"
            or metadata.get("best_iterations", {}).get("win") != 242
            or bundle["boosters"]["win"].current_iteration() != 242
            or policy.temperature != .75 or policy.probability_method != "race_normalize"
            or document.get("calibration", {}).get("temperature") != .75
            or metadata.get("history_lag_days") != 7 or mapping["history_lag_days"] != 7
            or metadata.get("history_start") != "2015-07-02"
            or metadata.get("mapping_sha256") != safety.digest(mapping_path)
            or document.get("model_manifest_sha256") != safety.digest(model_dir / "model.json")):
        raise safety.MLFailure("artifact_invalid")
    frozen = [(p, root, safety.digest(p, root=root)) for p, root in (
        (model_dir / "model.json", safety.ML_ROOT), (model_dir / "win.lgb.txt", safety.ML_ROOT),
        (mapping_path, safety.ML_ROOT), (policy_path, local.ROOT))]
    return bundle, mapping, policy, frozen


class ScopedReader:
    """Reuse the existing exporter, limiting entry status or history to today's horses."""
    def __init__(self, db, mapping, *, card=False, horses=None):
        self.db, self.mapping, self.card, self.horses = db, mapping, card, horses

    def iter_batches(self, operation, sql, params, **kwargs):
        if operation != "ml.export" or sql.count(" ORDER BY ") != 1:
            raise safety.MLFailure("invalid_input")
        columns = self.mapping["columns"]
        if self.card:
            clause = (" AND ra." + safety.identifier(columns["race_status"]) + "::text = %s"
                      " AND se." + safety.identifier(columns["horse_status"]) + "::text = %s")
            extra = ("2", "2")
        else:
            if not self.horses or any(not isinstance(h, str) or not h for h in self.horses):
                raise safety.MLFailure("invalid_input")
            clause = " AND se." + safety.identifier(columns["horse_id"]) + "::text = ANY(%s)"
            extra = (sorted(self.horses),)
        sql = sql.replace(" ORDER BY ", clause + " ORDER BY ", 1)
        return self.db.iter_batches(operation, sql, (*params[:-1], *extra, params[-1]), **kwargs)


def export(db, mapping, catalog, directory, start, end, *, card=False, horses=None):
    directory.mkdir()
    data.export_dataset(ScopedReader(db, mapping, card=card, horses=horses), mapping, directory,
        start=start, end=end, max_rows=5000 if card else 100000, max_seconds=240,
        root=ROOT, prediction_only=card, catalog=catalog)
    return data.read_dataset(directory / "runners.csv", root=ROOT, allow_prediction_snapshot=card)


def eligible(entries, ra_rows, current):
    grouped, schedules = defaultdict(list), defaultdict(list)
    for row in entries:
        grouped[features.entry_identity(row)[0][:-1]].append(row)
    for row in ra_rows:
        schedules[row["race_key"]].append(row)
    valid, starts, skipped = [], {}, {}
    for race, rows in grouped.items():
        versions = schedules[race]
        reason, start = None, None
        if len(versions) != 1:
            reason = "schedule_unavailable_or_conflicting"
        else:
            record = versions[0]
            if record.get("oversized") or record.get("record_type") != "RA" or record.get("status") != "2":
                reason = "entry_data_incomplete"
            else:
                try:
                    start = formats.parse_time(record.get("start"), "hhmm_on_race_date", race)
                except core.SimulationError:
                    reason = "schedule_unavailable_or_conflicting"
        if not reason and current >= start:
            reason = "already_started"
        if not reason:
            basic = features.build_features(rows, history_lag_days=7)
            sizes = {features.number(row.get("declared_runners"), low=2, high=28) for row in rows}
            if (sizes != {float(len(rows))} or any(not row.get("horse_id") for row in rows)
                    or any(item[name] is None for item in basic for name in v03.PRIORITY_FEATURES)):
                reason = "entry_data_incomplete"
        if reason:
            skipped[race] = reason
        else:
            valid.extend(rows)
            starts[race] = start
    return valid, starts, skipped


def calibrated_rows(predictions, identities, starts, policy, predicted_at, skipped):
    output = []
    for prediction in predictions:
        if predicted_at >= starts[prediction.race]:
            skipped[prediction.race] = "already_started"
            continue
        # predict_win supplies raw binary model scores. Apply the saved T=.75 exactly once here.
        probabilities, _ = core.probabilities(prediction.probabilities, policy)
        ordered = sorted(probabilities, key=lambda horse: (-probabilities[horse], int(horse)))
        if abs(sum(float(value) for value in probabilities.values()) - 1.) > 1e-10:
            raise safety.MLFailure("invalid_input")
        for rank, horse in enumerate(ordered, 1):
            output.append(dict(zip(RACE_KEY, prediction.race), race_date=DAY.isoformat(),
                scheduled_start_jst=starts[prediction.race].isoformat(), horse_number=horse,
                horse_id=identities[(*prediction.race, horse)], horse_name="",
                win_probability=float(probabilities[horse]), prediction_rank=rank,
                raw_win_score=float(prediction.probabilities[horse]), model_id=MODEL_ID,
                model_version="STEP2_best242_independent_evaluation_pending", temperature=policy.temperature,
                predicted_at_jst=predicted_at.isoformat()))
    return sorted(output, key=lambda row: (row["scheduled_start_jst"], row["keibajo_code"], row["prediction_rank"]))


def save_csv(path, rows):
    path = safety.local_path(path, root=ROOT)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def execute(directory, result):
    result["stage"] = "model_load"
    bundle, mapping, policy, frozen = load_source()
    if trial.now().date() != DAY:
        raise safety.MLFailure("invalid_date")
    secret = os.environ.get("PGPASSWORD")
    if not secret:
        raise safety.MLFailure("needs_authentication")
    result["stage"] = "entries_read"
    with read_only_session(connection_config(60000), secret) as db:
        catalog, _ = data.inspect_sources(db, include_years=False)
        entries, metadata = export(db, mapping, catalog, directory / "entries", DAY, DAY, card=True)
        races = tuple(sorted({features.entry_identity(row)[0][:-1] for row in entries}))
        result.update(input_race_count=len(races), input_runner_count=len(entries), source_status_counts=metadata["source_status_counts"])
        budget = local.Budget(30)
        fields = {name: publication.RA_FIELDS[name] for name in ("record_type", "status", "start")}
        columns = publication._columns(db, "jvd_ra", fields, budget)
        if columns is None:
            raise safety.MLFailure("invalid_schema")
        ra_rows = local._rows(db, budget, "jvd_ra", fields, columns, races, 100, join_targets=True)
        entries, starts, skipped = eligible(entries, ra_rows, trial.now())
        if not entries:
            raise safety.MLFailure("insufficient_history")
        result["stage"] = "prediction_history_read"
        history, history_meta = export(db, mapping, catalog, directory / "history",
            date.fromisoformat(bundle["metadata"]["history_start"]), DAY - timedelta(days=7),
            horses={row["horse_id"] for row in entries})
    secret = None
    result["data_read_completed_at_jst"] = trial.now().isoformat()
    result["stage"] = "predict"
    predictions, coverage = v03.predict_win(bundle, entries, history)
    predicted_at = trial.now()
    identities = {features.entry_identity(row)[0]: row["horse_id"] for row in entries}
    output = calibrated_rows(predictions, identities, starts, policy, predicted_at, skipped)
    for path, root, digest in frozen:
        if safety.digest(path, root=root) != digest:
            raise safety.MLFailure("artifact_invalid")
    save_started = trial.now()
    late = {tuple(row[k] for k in RACE_KEY) for row in output
            if save_started >= starts[tuple(row[k] for k in RACE_KEY)]}
    for race in late:
        skipped[race] = "already_started_before_save"
    output = [row for row in output if tuple(row[k] for k in RACE_KEY) not in late]
    if not output:
        raise safety.MLFailure("insufficient_history")
    result["stage"] = "save_predictions"
    save_csv(directory / "predictions.csv", output)
    saved_at = trial.now()
    result.update(status="ok", stage="predictions_saved", model_id=MODEL_ID, policy_id=POLICY_ID,
        temperature=.75, calibration_applications=1, best_iteration=242,
        first_prediction_at_jst=predicted_at.isoformat(), last_prediction_at_jst=predicted_at.isoformat(),
        predictions_saved_at_jst=saved_at.isoformat(), earliest_scheduled_start_jst=min(row["scheduled_start_jst"] for row in output),
        race_count=len({tuple(row[k] for k in RACE_KEY) for row in output}), runner_count=len(output),
        skipped_race_reasons=dict(Counter(skipped.values())), history_rows=len(history),
        history_start="2015-07-02", history_end="2026-10-11", history_scope="only_target_horses_same_training_history_start",
        feature_coverage=coverage, predictions_csv=str(directory / "predictions.csv"),
        predictions_sha256=safety.digest(directory / "predictions.csv", root=ROOT),
        current_results_excluded=True, odds_used=False, source_artifacts_unchanged=True,
        rank_tie_break="lowest_horse_number", horse_names="not_requested_to_prioritize_first_save")
    print(json.dumps({k: result[k] for k in ("status", "stage", "race_count", "runner_count", "temperature",
        "first_prediction_at_jst", "predictions_saved_at_jst", "predictions_csv")}, ensure_ascii=True), flush=True)
    # Only after the full prediction CSV is durable, write convenience files.
    top = [row for row in output if row["prediction_rank"] == 1]
    save_csv(directory / "top_predictions.csv", top)
    safety.write_json(directory / "skipped_races.local.json", [dict(zip(RACE_KEY, race), reason=reason)
        for race, reason in sorted(skipped.items())], root=ROOT)
    report = safety.local_path(directory / "top_predictions.md", root=ROOT)
    with report.open("x", encoding="utf-8-sig", newline="\n") as stream:
        stream.write("# 2026-10-18 STEP 2 best242 試験予測\n\n独立評価前の候補モデル。T=0.75を1回適用。EV購入判断なし。\n\n")
        stream.write("| 発走予定(JST) | 競馬場コード | R | 予測1位馬番 | 勝率 |\n|---|---|---:|---:|---:|\n")
        for row in top:
            stream.write(f"| {row['scheduled_start_jst'][11:16]} | {row['keibajo_code']} | {int(row['race_bango'])} | {row['horse_number']} | {100*row['win_probability']:.2f}% |\n")
    result["top_predictions_markdown"] = str(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = {"status": "stopped", "kind": "20261018_step2_predictions", "stage": "preflight",
              "started_at_jst": trial.now().isoformat(), "live_purchase_enabled": False}
    directory = None
    try:
        if args.preflight:
            load_source()
            result.update(status="ok", model_id=MODEL_ID, temperature=.75, best_iteration=242, database_accessed=False)
        else:
            directory = safety.new_artifact("evaluations", root=ROOT)
            execute(directory, result)
    except SafeAuditError as exc:
        result.update(status="stopped", reason=exc.code)
    except Exception:
        result.update(status="stopped", reason="internal_error")
    finally:
        result["finished_at_jst"] = trial.now().isoformat()
        if directory:
            result["run_directory"] = str(directory)
            try:
                safety.write_json(directory / "run.json", result, root=ROOT)
            except Exception:
                result.update(status="stopped", reason="io_failed")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
