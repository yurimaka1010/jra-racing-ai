"""Join existing HR payouts to frozen virtual tickets; reuse the existing decoder."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
import sys

from . import core, formats, local

SETTLEMENT_FIELDS = {"ai", "favorite", "common_settlement_eligible", "settlement_reason"}
FLAG_COLUMNS = {
    "status": ("data_kubun",),
    "unestablished": ("fuseiritsu_flag_tansho", "fuseiritu_flag_tansho"),
    "special": ("tokubarai_flag_tansho", "tokuharai_flag_tansho"),
    "refund": ("henkan_flag_tansho",),
}


def decision_digest(ledger):
    """The only mutable row fields are the two settlements and result eligibility."""
    rows = [{k: v for k, v in row.items() if k not in SETTLEMENT_FIELDS} for row in ledger["rows"]]
    try:
        return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        raise core.SimulationError("artifact_invalid") from None


def load_saved(simulation_id, *, max_races=50):
    if not isinstance(simulation_id, str) or not re.fullmatch(r"[a-f0-9]{32}", simulation_id):
        raise core.SimulationError("invalid_input")
    directory = local.ROOT / "simulations" / simulation_id
    manifest = local.read(directory / "manifest.json")
    if (manifest.get("format_version") != 2 or manifest.get("mode") != "hypothetical_backtest"
            or manifest.get("wager") != "win" or manifest.get("live_purchase_enabled") is not False):
        raise core.SimulationError("artifact_invalid")
    for name in ("ledger.json", "policy.json", "summary.json"):
        if manifest.get("hashes", {}).get(name) != local.sha(directory / name):
            raise core.SimulationError("artifact_invalid")
    ledger = local.read(directory / "ledger.json")
    policy, policy_document = local.load_policy(directory / "policy.json")
    if (ledger.get("format_version") != 2 or ledger.get("mode") != "hypothetical_backtest"
            or ledger.get("wager") != "win" or ledger.get("timing_basis") != policy.timing_basis
            or not isinstance(ledger.get("rows"), list) or not 1 <= len(ledger["rows"]) <= max_races):
        raise core.SimulationError("artifact_invalid")
    races = []
    for row in ledger["rows"]:
        race = core.key(row.get("race_key"))
        if race in races or row.get("month") != race[0] + "-" + race[1][:2]:
            raise core.SimulationError("artifact_invalid")
        races.append(race)
        roster = tuple(core.horse(item["horse_number"]) for item in row.get("runners", []))
        if len(roster) != len(set(roster)):
            raise core.SimulationError("artifact_invalid")
        for strategy in ("ai", "favorite"):
            selection = row.get(strategy + "_horse")
            if selection is not None and (selection not in roster or row.get("simulation_eligible") is not True):
                raise core.SimulationError("artifact_invalid")
            if row.get(strategy, {}).get("stake_yen") != (100 if selection is not None else 0):
                raise core.SimulationError("artifact_invalid")
    return directory, ledger, policy_document, tuple(races), manifest


def scoped_source(races, columns, reviewed_source=None):
    """Exact physical names only; aliases are shown locally for human confirmation."""
    if reviewed_source is not None:
        source = deepcopy(formats.validate_payout_source(reviewed_source))
    else:
        mapping = {}
        for alias, candidates in FLAG_COLUMNS.items():
            matches = [name for name in candidates if name in columns]
            if len(matches) != 1:
                raise core.SimulationError("mapping_unconfirmed")
            mapping[alias] = matches[0]
        source = {"table": "jvd_hr", "reviewed_normal_conditions": False,
                  "currency": "JPY", "base_stake_yen": 100, "final_category": "2",
                  "allowed_final_categories": ["1", "2"],
                  "revision_policy": "single_reviewed_final", "semantics_reviewed": False,
                  "official_reference": str(local.SPEC_PDF),
                  "local_reconciliation_reference": "pending_current_run_local_confirmation",
                  "years": sorted({int(race[0]) for race in races}),
                  "venues": sorted({race[2] for race in races}),
                  "race_keys": [list(race) for race in races],
                  "numerator": 1, "denominator": 1, "columns": mapping}
    if "race_keys" in source:
        scope = source["race_keys"]
        if (not isinstance(scope, list) or not 1 <= len(scope) <= 50
                or not set(races).issubset({core.key(race) for race in scope})):
            raise core.SimulationError("outside_scope")
    allowed = source.get("allowed_final_categories", [source["final_category"]])
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
            or any(category not in ("1", "2") for category in allowed)):
        raise core.SimulationError("invalid_mapping")
    source["allowed_final_categories"] = allowed
    return source


def fetch_hr(db, races, *, max_rows=200, max_seconds=30, reviewed_source=None):
    """Reuse the existing fixed SELECT builder/cursor; no year-wide scan or new parser."""
    budget = local.Budget(max_seconds)
    columns = local.table_info(db, "jvd_hr", budget)
    if columns is None:
        return [], None, {"table": "jvd_hr", "status": "not_found", "rows": 0}
    source = scoped_source(races, columns, reviewed_source)
    fields = {alias: (physical, 1) for alias, physical in source["columns"].items()}
    fields["record_type"] = ("record_id", 2)
    fields.update({f"slot_{index}{part}": (f"haraimodoshi_tansho_{index}{part}", width)
                   for index in range(1, 4) for part, width in zip("abc", (2, 9, 2))})
    for physical, _ in fields.values():
        col = columns.get(physical, {})
        if col.get("type_is_builtin") is not True or col.get("type_name") not in local.TEXT_TYPES:
            raise core.SimulationError("invalid_schema")
    rows = local._rows(db, budget, "jvd_hr", fields, columns, races, max_rows, join_targets=True)
    budget.check()
    status = {"table": "jvd_hr", "status": "bounded_read_complete", "rows": len(rows)}
    if not rows:
        # One date-bounded count only after a failed join. No records, date-range
        # scan, alternate join or outcome substitution is introduced.
        dates = sorted({race[:2] for race in races})
        where = " OR ".join('(t.kaisai_nen::text = %s AND t.kaisai_tsukihi::text = %s)' for _ in dates)
        sql = ('SELECT count(*) AS target_date_rows, count(*) FILTER (WHERE t.keibajo_code::text '
               "IN ('01','02','03','04','05','06','07','08','09','10')) AS jra_target_date_rows "
               'FROM ONLY public.jvd_hr t WHERE (' + where + ')')
        result = budget.aggregate(db, "v02.hr.target_date_count", sql,
                                  tuple(value for day in dates for value in day), max_rows=1)
        if (len(result) != 1 or any(type(result[0].get(field)) is not int or result[0][field] < 0
                                  for field in ("target_date_rows", "jra_target_date_rows"))):
            raise core.SimulationError("invalid_schema")
        counts = result[0]
        reason = ("hr_target_date_not_found" if counts["target_date_rows"] == 0 else
                  "hr_jra_target_date_not_found" if counts["jra_target_date_rows"] == 0 else
                  "hr_on_date_but_target_keys_unmatched")
        status["target_date_availability"] = {"target_date_rows": counts["target_date_rows"],
                                              "jra_target_date_rows": counts["jra_target_date_rows"],
                                              "reason": reason}
    return rows, source, status


def confirm_source(review_path):
    print("Local-only HR settlement review: " + str(review_path), file=sys.stderr)
    print("hr_rowsのslot_1a/1b/1cはharaimodoshi_tansho_1a/1b/1cの元値です（2・3枠も同じ対応）。"
          "proposed_sourceは確認前の候補です。", file=sys.stderr)
    print("列対応、通常払戻金の整数値が100円馬券あたりの円額であること、"
          "同一キー1件の現在スナップショットを使うことを今回の対象範囲で確認できた場合だけ "
          "CONFIRM_HR_JPY_PER_100 と入力してください。値をチャットへ貼る必要はありません。", file=sys.stderr)
    try:
        if input() != "CONFIRM_HR_JPY_PER_100":
            raise core.SimulationError("mapping_unconfirmed")
    except EOFError:
        raise core.SimulationError("mapping_unconfirmed") from None


def decode_existing_hr(rows, source):
    """Group by the full key before calling the existing decoder, including versions."""
    grouped, outcomes = defaultdict(list), []
    for record in rows:
        row = dict(record)
        if row.get("record_type") != "HR":
            row["oversized"] = True
        grouped[core.key(row["race_key"])].append(row)
    for versions in grouped.values():
        category = versions[0].get("status")
        selected = dict(source)
        if len(versions) == 1 and category in source["allowed_final_categories"]:
            selected["final_category"] = category
        else:
            selected["final_category"] = source["final_category"]
            # A category outside the reviewed list cannot use the default category.
            if len(versions) == 1 and category not in source["allowed_final_categories"]:
                versions = [dict(versions[0], status=None)]
        outcomes.extend(formats.decode_hr_rows(versions, selected))
    return outcomes


def settle_frozen(ledger, outcomes):
    """Only replace settlement fields. Never run inference, calibration or selection."""
    before = decision_digest(ledger)
    result, grouped = deepcopy(ledger), defaultdict(list)
    for outcome in outcomes:
        grouped[core.key(outcome.race)].append(outcome)
    for row in result["rows"]:
        matches = grouped[core.key(row["race_key"])]
        outcome = matches[0] if len(matches) == 1 else None
        if len(matches) > 1:
            outcome = core.Outcome(core.key(row["race_key"]), "pending", reason="result_duplicate")
        roster = tuple(item["horse_number"] for item in row["runners"])
        reason = core.outcome_reason(outcome, roster)
        row["common_settlement_eligible"] = row["simulation_eligible"] and reason is None
        row["settlement_reason"] = reason
        for strategy in ("ai", "favorite"):
            row[strategy] = core.settle(row[strategy + "_horse"], outcome, roster)
    if before != decision_digest(result):
        raise core.SimulationError("artifact_invalid")
    summary = core.summarize(result["rows"])
    summary["timing_basis"] = ledger["timing_basis"]
    summary["odds_coverage_basis"] = ("published_snapshots_not_actual_observations"
                                      if ledger["timing_basis"] == "jra_publication_time"
                                      else "verified_observed_snapshots")
    result["summary"] = summary
    return result


def run(args, *, connection_config, password, read_only_session):
    if not args.execute_local:
        raise core.SimulationError("explicit_execution_required")
    if (not 1 <= args.max_races <= 50 or not 1 <= args.max_rows <= 1000
            or not 1 <= args.max_seconds <= 120 or not 1000 <= args.query_timeout_ms <= 30000):
        raise core.SimulationError("input_limit")
    parent, ledger, policy_document, races, manifest = load_saved(args.simulation_id, max_races=args.max_races)
    reviewed_source = local.read(args.payout_source) if args.payout_source else None
    if reviewed_source is not None:
        scoped_source(races, {}, reviewed_source)
    config = connection_config(min(args.query_timeout_ms, args.max_seconds * 1000))
    secret = password()
    try:
        with read_only_session(config, secret) as db:
            rows, source, db_status = fetch_hr(db, races, max_rows=args.max_rows,
                max_seconds=args.max_seconds, reviewed_source=reviewed_source)
    finally:
        secret = None
    directory = local.artifact("simulations")
    review_path = directory / "hr_settlement_review.json"
    local.write(review_path, {"format_version": 2, "data_source": "local_jra_van_do_not_share",
                "parent_simulation_id": args.simulation_id, "hr_rows": rows, "proposed_source": source,
                "field_mapping": {f"slot_{index}{part}": f"haraimodoshi_tansho_{index}{part}"
                                  for index in range(1, 4) for part in "abc"},
                "interpretation_status": "reviewed_profile_supplied" if reviewed_source else "pending_local_confirmation",
                "note": "proposed_source is not unit evidence; raw monetary strings are preserved"})
    review_path.chmod(0o444)
    if rows and reviewed_source is None:
        confirm_source(review_path)
        source["local_reconciliation_reference"] = str(review_path)
        source["reviewed_normal_conditions"] = True
        source["semantics_reviewed"] = True
    outcomes = decode_existing_hr(rows, source) if rows else []
    simulation = settle_frozen(ledger, outcomes)
    source_path = directory / "hr_payout_source.json"
    local.write(source_path, source)
    source_path.chmod(0o444)
    # The original ledger and policy remain immutable. Each run creates a new artifact.
    provenance = dict(manifest.get("provenance", {}))
    provenance.update(source="local_inputs", settlement_source="public.jvd_hr_existing_decoder",
                      parent_simulation_id=args.simulation_id,
                      parent_ledger_sha256=local.sha(parent / "ledger.json"),
                      parent_manifest_sha256=local.sha(parent / "manifest.json"),
                      parent_policy_sha256=local.sha(parent / "policy.json"),
                      frozen_decisions_sha256=decision_digest(ledger), decisions_unchanged=True,
                      hr_review_sha256=local.sha(review_path), hr_source_sha256=local.sha(source_path),
                      hr_source_confirmed=bool(rows),
                      hr_source_confirmation_basis="reviewed_profile" if reviewed_source else "current_run_local_confirmation",
                      db_sources=[db_status],
                      settlement_scope="current_single_hr_snapshot_normal_win_only_no_global_revision_guarantee")
    local.write_report(directory, simulation, policy_payload=policy_document, provenance=provenance)
    summary = simulation["summary"]
    compact = {name: summary[name] for name in ("target_race_count", "odds_acquired_race_count",
              "odds_acquisition_rate_percent", "simulation_race_count", "race_reasons", "ai", "favorite", "comparisons")}
    comparison_count = summary["comparisons"]["all_common_eligible_races"]["race_count"]
    settlement_status = ("pending_no_matching_hr" if not rows else
                         "complete" if summary["ai"]["unsettled_count"] == 0 and summary["favorite"]["unsettled_count"] == 0 else
                         "partial" if comparison_count else "pending")
    return {"status": "ok", "kind": "v02_hr_settlement", "artifact_id": directory.name,
            "settlement_status": settlement_status,
            "parent_simulation_id": args.simulation_id, "decisions_unchanged": True,
            "timing_basis": ledger["timing_basis"], "db_sources": [db_status],
            "hr_category_counts": dict(Counter(row["status"] if row.get("status") in ("0", "1", "2", "9")
                                               else "unknown" for row in rows)),
            "summary": compact, "notice": simulation["notice"]}
