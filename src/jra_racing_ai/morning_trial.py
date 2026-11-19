"""One-day morning prediction only; reuses the frozen V0.3 feature/model path."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta
import csv
import json
import os
from pathlib import Path

from .core import RACE_KEY, SafeAuditError
from .db_connection import read_only_session
from .ml import data, features, model, safety, v03
from .ml_v03_cli import ROOT as V03_ROOT
from .win_sim import core, local, publication
from .win_sim_cli import connection_config

ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\trial-20261017")
SOURCE_RUN = V03_ROOT / "evaluations" / "c62408217a6146b1bf91f88e6f902631" / "run.json"
CSV_FIELDS = (*RACE_KEY, "horse_number", "horse_id", "win_probability", "raw_win_score", "model_id",
              "model_version", "temperature", "predicted_at_jst", "data_read_completed_at_jst", "scheduled_start_jst")


def now():
    return datetime.now(core.JST)


def load_source(day):
    saved = safety.read_json(SOURCE_RUN, root=V03_ROOT)
    if saved.get("kind") != "v03_first_run" or saved.get("status") != "ok":
        raise safety.MLFailure("artifact_invalid")
    for name in ("model_id", "mapping_id", "policy_id"):
        if not isinstance(saved.get(name), str) or len(saved[name]) != 32 or any(c not in "0123456789abcdef" for c in saved[name]):
            raise safety.MLFailure("artifact_invalid")
    model_path = safety.ML_ROOT / "models" / saved["model_id"]
    mapping_path = safety.ML_ROOT / "schema" / saved["mapping_id"] / "mapping.reviewed.json"
    policy_path = local.ROOT / "policies" / saved["policy_id"] / "policy.json"
    mapping = data.validate_mapping(safety.read_json(mapping_path))
    bundle = model.load_models(model_path, required_targets=("win",))
    policy, document = local.load_policy(policy_path)
    model_hash = safety.digest(model_path / "model.json")
    if (bundle["metadata"].get("pipeline_version") != "0.3" or policy.temperature != 2.0
            or mapping["history_lag_days"] != 7 or policy.probability_method != "race_normalize"
            or document["model_manifest_sha256"] != model_hash
            or safety.digest(mapping_path) != bundle["metadata"].get("mapping_sha256")
            or document.get("calibration") != saved.get("calibration")):
        raise safety.MLFailure("artifact_invalid")
    policy = replace(policy, evaluation_start=day.isoformat(), evaluation_end=day.isoformat()).validate()
    frozen = [(path, root, safety.digest(path, root=root)) for path, root in (
        (SOURCE_RUN, V03_ROOT), (model_path / "model.json", safety.ML_ROOT),
        (model_path / "win.lgb.txt", safety.ML_ROOT), (mapping_path, safety.ML_ROOT), (policy_path, local.ROOT))]
    return saved, mapping, bundle, policy, frozen


def probe(db, mapping, day, catalog, *, report=None, allow_missing=False):
    data.check_mapping_schema(mapping, catalog["tables"])
    sql, params = data.export_query(mapping, start=day, end=day, max_rows=5000, prediction_only=True)
    fields = ", ".join('"' + key + '"' for key in RACE_KEY)
    counts = db.aggregate("trial.entries_probe", "SELECT count(*) AS runners, count(DISTINCT (" + fields +
        ")) AS races, count(*) FILTER (WHERE _ra_status <> '2' OR _se_status <> '2') AS non_card_rows FROM (" +
        sql + ") AS entries", params, max_rows=1)
    if (len(counts) == 1 and all(type(counts[0].get(k)) is int and counts[0][k] >= 0
                              for k in ("runners", "races", "non_card_rows")) and report is not None):
        # Preserve only anonymous counts even when the readiness check fails.
        report["available_card_counts"] = {k: counts[0][k] for k in ("runners", "races", "non_card_rows")}
        if counts[0]["runners"] == 0:
            scope = (" WHERE kaisai_nen::text = %s AND kaisai_tsukihi::text = %s"
                     " AND keibajo_code::text IN (" + ", ".join(["%s"] * 10) + ")")
            source_params = (str(day.year), day.strftime("%m%d"), *(f"{i:02d}" for i in range(1, 11)))
            source_counts = db.aggregate("trial.empty_entries", "SELECT (SELECT count(*) FROM ONLY public.jvd_ra" +
                scope + ") AS ra_rows, (SELECT count(*) FROM ONLY public.jvd_se" + scope + ") AS se_rows",
                source_params * 2, max_rows=1)
            if (len(source_counts) != 1 or any(type(source_counts[0].get(k)) is not int or source_counts[0][k] < 0
                                              for k in ("ra_rows", "se_rows"))):
                raise safety.MLFailure("invalid_schema")
            report["source_date_counts"] = source_counts[0]
            if source_counts[0] == {"ra_rows": 0, "se_rows": 0}:
                report["data_status"] = "ra_se_target_date_missing"
                # Setup may precede PC-KEIBA's card import. Only the count-only
                # probe can succeed as pending; the actual prediction still fails.
                if allow_missing and counts[0] == {"runners": 0, "races": 0, "non_card_rows": 0}:
                    return counts[0]
    if (len(counts) != 1 or not all(type(counts[0].get(k)) is int for k in ("runners", "races", "non_card_rows"))
            or not 1 <= counts[0]["runners"] <= 5000 or not 1 <= counts[0]["races"] <= 50
            or counts[0]["non_card_rows"]):
        raise safety.MLFailure("insufficient_history")
    return counts[0]


def export(db, mapping, catalog, directory, start, end, *, prediction=False):
    directory.mkdir()
    data.export_dataset(db, mapping, directory, start=start, end=end,
        max_rows=5000 if prediction else 250000, max_seconds=240, root=ROOT,
        prediction_only=prediction, catalog=catalog)
    return data.read_dataset(directory / "runners.csv", root=ROOT, allow_prediction_snapshot=prediction)


def eligible_rows(rows, schedules, policy, current):
    grouped, schedule_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[features.entry_identity(row)[0][:-1]].append(row)
    for schedule in schedules:
        schedule_groups[schedule.race].append(schedule)
    valid, starts, skipped = [], {}, Counter()
    for race, entries in grouped.items():
        schedule, cutoff, reason = core.choose_schedule(race, schedule_groups[race], policy)
        if reason:
            skipped[reason] += 1
            continue
        if current >= cutoff:
            skipped["prediction_deadline_passed"] += 1
            continue
        counts = {features.number(row.get("declared_runners"), low=2, high=28) for row in entries}
        basic = features.build_features(entries, history_lag_days=7)
        if (counts != {float(len(entries))} or any(not row.get("horse_id") for row in entries)
                or any(value[name] is None for value in basic for name in v03.PRIORITY_FEATURES)):
            skipped["entry_data_incomplete"] += 1
            continue
        valid.extend(entries)
        starts[race] = schedule.scheduled_start
    return valid, starts, skipped


def execute(args, directory, result):
    day = date.fromisoformat(args.race_date)
    result.update(trial_date=day.isoformat(), model_version="0.3", temperature=2.0,
                  live_purchase_enabled=False, ev_status="not_requested_morning_prediction_only",
                  ingestion_update_status="not_verified_from_read_only_snapshot")
    saved, mapping, bundle, policy, frozen = load_source(day)
    result.update(model_id=saved["model_id"], model_sha256=safety.digest(safety.ML_ROOT / "models" / saved["model_id"] / "model.json"))
    if args.preflight:
        result.update(kind="offline_preflight", status="ok", database_accessed=False)
        return
    if now().date() != day:
        raise core.SimulationError("invalid_period")
    secret = os.environ.get("PGPASSWORD")
    if not secret:
        raise core.SimulationError("needs_authentication")
    config = connection_config(60000)
    result["stage"] = "database_probe" if args.probe else "data_export"
    with read_only_session(config, secret) as db:
        catalog, _ = data.inspect_sources(db, include_years=False)
        counts = probe(db, mapping, day, catalog, report=result, allow_missing=args.probe)
        result.update(available_card_counts=counts, data_read_started_at_jst=now().isoformat())
        if args.probe:
            result.update(kind="database_probe", status="ok", database_accessed=True,
                          ready_for_prediction=bool(counts["runners"]),
                          data_read_completed_at_jst=now().isoformat())
            return
        history, history_meta = export(db, mapping, catalog, directory / "history", date(2024, 1, 1),
                                       day - timedelta(days=mapping["history_lag_days"]))
        entries, entry_meta = export(db, mapping, catalog, directory / "entries", day, day, prediction=True)
        if entry_meta["source_status_counts"] != {"ra_races": {"2": entry_meta["races"]}, "se_runners": {"2": len(entries)}}:
            raise safety.MLFailure("insufficient_history")
        races = tuple(sorted({features.entry_identity(row)[0][:-1] for row in entries}))
        budget = local.Budget(30)
        columns = publication._columns(db, "jvd_ra", publication.RA_FIELDS, budget)
        if columns is None:
            raise safety.MLFailure("invalid_schema")
        ra = local._rows(db, budget, "jvd_ra", publication.RA_FIELDS, columns, races, 100, join_targets=True)
    secret = None
    result["data_read_completed_at_jst"] = now().isoformat()
    rosters = [core.PredictionRace(race, {row["horse_number"]: .1 for row in entries if features.entry_identity(row)[0][:-1] == race}) for race in races]
    schedules, _ = publication.schedules_from_ra(ra, rosters)
    entries, starts, skipped = eligible_rows(entries, schedules, policy, now())
    result.update(skipped_race_reasons=dict(skipped), input_race_count=len(races))
    if not entries:
        raise safety.MLFailure("insufficient_history")
    result["stage"] = "predict"
    predictions, coverage = v03.predict_win(bundle, entries, history)
    predicted_at = now()
    output, identities = [], {features.entry_identity(row)[0]: row["horse_id"] for row in entries}
    for prediction in predictions:
        if predicted_at >= starts[prediction.race] - timedelta(minutes=policy.minutes_before_start):
            skipped["prediction_deadline_passed"] += 1
            continue
        probabilities, _ = core.probabilities(prediction.probabilities, policy)
        for horse, probability in probabilities.items():
            output.append(dict(zip(RACE_KEY, prediction.race), horse_number=horse,
                horse_id=identities[(*prediction.race, horse)], win_probability=float(probability),
                raw_win_score=float(prediction.probabilities[horse]), model_id=saved["model_id"], model_version="0.3",
                temperature=2.0, predicted_at_jst=predicted_at.isoformat(),
                data_read_completed_at_jst=result["data_read_completed_at_jst"], scheduled_start_jst=starts[prediction.race].isoformat()))
    if not output:
        raise safety.MLFailure("insufficient_history")
    for path, root, digest in frozen:
        if safety.digest(path, root=root) != digest:
            raise safety.MLFailure("artifact_invalid")
    result["stage"] = "save_predictions"
    path = safety.local_path(directory / "predictions.csv", root=ROOT)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    path.chmod(0o444)
    result.update(status="ok", kind="morning_predictions", predicted_at_jst=predicted_at.isoformat(),
        race_count=len({tuple(row[k] for k in RACE_KEY) for row in output}), runner_count=len(output),
        skipped_race_reasons=dict(skipped), history_rows=len(history), feature_coverage=coverage,
        predictions_sha256=safety.digest(path, root=ROOT), predictions_csv=str(path),
        source_artifacts_unchanged=True, history_end_date=(day-timedelta(days=7)).isoformat(),
        prediction_timing="actual_run_time_no_backdating", data_source="current_local_pre_race_card_snapshot")


def main(argv=None):
    result = {"status": "stopped", "kind": "morning_trial", "stage": "preflight", "started_at_jst": now().isoformat()}
    directory = None
    try:
        parser = argparse.ArgumentParser(description="One-day V0.3 morning predictions only")
        parser.add_argument("--race-date", choices=("2026-10-17",), default="2026-10-17")
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument("--preflight", action="store_true")
        modes.add_argument("--probe", action="store_true")
        args = parser.parse_args(argv)
        directory = safety.new_artifact("evaluations", root=ROOT)
        execute(args, directory, result)
    except SafeAuditError as exc:
        result.update(status="stopped", reason=exc.code)
    except Exception:
        result.update(status="stopped", reason="internal_error")
    finally:
        result["finished_at_jst"] = now().isoformat()
        if directory:
            result["run_directory"] = str(directory)
            try:
                safety.write_json(directory / "run.json", result, root=ROOT)
                (directory / "run.json").chmod(0o444)
                for path in directory.rglob("*"):
                    if path.is_file():
                        path.chmod(0o444)
            except Exception:
                result.update(status="stopped", reason="io_failed")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
