"""Frozen ticket settlement using synthetic HR rows and mocked local DB access."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import win_sim_cli as cli
from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai.win_sim import core, formats, local, settle_saved
from test_win_sim_v02 import example, hr_row, policy, race, simulate


def native_hr(r=None, **changes):
    return dict(hr_row(r), record_type="HR", **changes)


def columns():
    names = set(RACE_KEY) | {"record_id", *(v[0] for v in settle_saved.FLAG_COLUMNS.values())}
    names.update(f"haraimodoshi_tansho_{i}{part}" for i in range(1, 4) for part in "abc")
    return {name: {"name": name, "type_is_builtin": True, "type_name": "varchar"} for name in names}


def source():
    config = settle_saved.scoped_source((race(),), columns())
    config.update(reviewed_normal_conditions=True, semantics_reviewed=True,
                  local_reconciliation_reference="SYNTHETIC_REVIEW")
    return config


class FrozenSettlementTests(unittest.TestCase):
    def test_existing_decoder_settles_both_strategies_without_reselecting(self):
        ledger = simulate([example(), example(number="02")])
        rows = [native_hr(), native_hr(race(number="02"))]
        rows[1].update(slot_1a="01", slot_1b="000000180", slot_1c="01")
        before = settle_saved.decision_digest(ledger)
        with patch.object(core, "run_simulation", side_effect=AssertionError("selection_must_not_run")):
            outcomes = settle_saved.decode_existing_hr(rows, source())
            result = settle_saved.settle_frozen(ledger, outcomes)
        self.assertEqual(before, settle_saved.decision_digest(result))
        comparison = result["summary"]["comparisons"]["all_common_eligible_races"]
        self.assertEqual(comparison["race_count"], 2)
        self.assertEqual(comparison["ai"]["hit_count"], 1)
        self.assertEqual(comparison["ai"]["hit_rate_percent"], 50)
        self.assertEqual(comparison["ai"]["investment_yen"], 200)
        self.assertEqual(comparison["ai"]["payout_yen"], 820)
        self.assertEqual(comparison["ai"]["net_profit_yen"], 620)
        self.assertEqual(comparison["ai"]["return_rate_percent"], 410)
        self.assertEqual(comparison["favorite"]["net_profit_yen"], -20)
        self.assertEqual(comparison["favorite"]["return_rate_percent"], 90)

    def test_only_unresolved_races_are_held_and_common_set_is_identical(self):
        ledger = simulate([example(), example(number="02"), example(number="03")])
        rows = [native_hr(), native_hr(race(number="02"), refund="1")]
        result = settle_saved.settle_frozen(ledger, settle_saved.decode_existing_hr(rows, source()))
        self.assertEqual(result["summary"]["target_race_count"], 3)
        self.assertEqual(result["summary"]["ai"]["unsettled_count"], 2)
        self.assertIsNone(result["summary"]["ai"]["net_profit_yen"])
        common = result["summary"]["comparisons"]["all_common_eligible_races"]
        self.assertEqual(common["race_count"], 1)
        self.assertEqual(common["ai"]["net_profit_yen"], 720)
        self.assertEqual(common["favorite"]["net_profit_yen"], -100)

    def test_all_six_key_parts_are_used_for_result_matching(self):
        ledger = simulate()
        for index, replacement in enumerate(("2031", "0106", "06", "02", "02", "02")):
            different = list(race())
            different[index] = replacement
            outcome = core.Outcome(tuple(different), "normal_final", {"03": 820}, True, True)
            result = settle_saved.settle_frozen(ledger, [outcome])
            self.assertEqual(result["summary"]["ai"]["unsettled_count"], 1)

    def test_hr1_is_payout_final_but_mixed_versions_are_never_selected(self):
        rows = [native_hr(status="1")]
        self.assertEqual(settle_saved.decode_existing_hr(rows, source())[0].status, "normal_final")
        outcomes = settle_saved.decode_existing_hr(rows + [native_hr()], source())
        self.assertEqual(outcomes[0].reason, "result_duplicate")

    def test_profile_restricted_to_hr2_does_not_adopt_hr1(self):
        config = source()
        config["allowed_final_categories"] = ["2"]
        self.assertEqual(settle_saved.decode_existing_hr([native_hr(status="1")], config)[0].status, "pending")

    def test_existing_parser_holds_special_invalid_deleted_and_duplicate_cases(self):
        invalid = [{"special": "1"}, {"unestablished": "1"}, {"status": "0"}, {"status": "9"},
                   {"status": None}, {"slot_1a": "00"}, {"slot_1b": None}, {"slot_1b": "0000000820"},
                   {"slot_2a": "03", "slot_2b": "000000820", "slot_2c": "03"}, {"record_type": "XX"}]
        for changes in invalid:
            rows = [native_hr(), native_hr(race(number="02"))]
            rows[1].update(changes)
            outcomes = settle_saved.decode_existing_hr(rows, source())
            self.assertEqual([item.status for item in outcomes], ["normal_final", "pending"])

    def test_dead_heat_keeps_all_official_payouts(self):
        row = native_hr()
        row.update(slot_2a="01", slot_2b="000000260", slot_2c="01")
        result = settle_saved.settle_frozen(simulate(), settle_saved.decode_existing_hr([row], source()))
        self.assertEqual(result["summary"]["ai"]["payout_yen"], 820)
        self.assertEqual(result["summary"]["favorite"]["payout_yen"], 260)

    def test_ambiguous_flag_columns_are_not_guessed(self):
        metadata = columns()
        metadata["tokuharai_flag_tansho"] = {"type_is_builtin": True, "type_name": "varchar"}
        with self.assertRaises(core.SimulationError):
            settle_saved.scoped_source((race(),), metadata)

    def test_local_confirmation_never_echoes_private_input(self):
        with patch("builtins.input", return_value="SYNTHETIC_PRIVATE"), patch("sys.stderr", io.StringIO()) as output:
            with self.assertRaises(core.SimulationError):
                settle_saved.confirm_source(Path("synthetic-review.json"))
        self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())


class SavedSettlementLocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "v02"
        self.root.mkdir()
        self.root_patch = patch.object(local, "ROOT", self.root)
        self.root_patch.start()
        self.parent = local.artifact("simulations")
        self.ledger = simulate([example(), example(number="02")])
        local.write_report(self.parent, self.ledger,
            policy_payload=local.policy_document(policy(), model_hash="SYNTHETIC_MODEL"),
            provenance={"source": "synthetic"})
        self.args = cli.parser().parse_args(["settle-hr", "--execute-local", "--simulation-id", self.parent.name])

    def tearDown(self):
        self.root_patch.stop()
        for path in Path(self.temp.name).rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def run_mock(self, *, rows=None, confirm=None):
        rows = [native_hr(), native_hr(race(number="02"))] if rows is None else rows
        config = (source() if self.args.payout_source else
                  settle_saved.scoped_source((race(), race(number="02")), columns()))
        with patch.object(cli, "password", return_value="SYNTHETIC_PASSWORD"), patch.object(cli, "read_only_session"), \
                patch.object(settle_saved, "fetch_hr", return_value=(rows, config, {"table": "jvd_hr", "rows": len(rows)})), \
                patch.object(settle_saved, "confirm_source", side_effect=confirm) as confirmation:
            result = cli.run(self.args)
        return result, confirmation

    def test_complete_user_path_preserves_old_files_and_stdout_is_aggregate_only(self):
        previous = local.sha(self.parent / "ledger.json")
        result, confirmation = self.run_mock()
        confirmation.assert_called_once()
        self.assertTrue(result["decisions_unchanged"])
        self.assertEqual(result["settlement_status"], "complete")
        self.assertEqual(result["summary"]["comparisons"]["all_common_eligible_races"]["race_count"], 2)
        self.assertEqual(result["summary"]["ai"]["net_profit_yen"], 1440)
        self.assertEqual(previous, local.sha(self.parent / "ledger.json"))
        self.assertNotEqual(result["artifact_id"], self.parent.name)
        output = json.dumps(result)
        for private in ("race_key", "horse_number", "slot_1b", "000000820", "SYNTHETIC_PASSWORD"):
            self.assertNotIn(private, output)
        manifest = local.read(self.root / "simulations" / result["artifact_id"] / "manifest.json")
        self.assertEqual(manifest["provenance"]["settlement_source"], "public.jvd_hr_existing_decoder")

    def test_confirmation_precedes_money_decoding_and_raw_value_is_preserved(self):
        with patch.object(formats, "decode_hr_rows", wraps=formats.decode_hr_rows) as decoder:
            def review(path):
                decoder.assert_not_called()
                data = local.read(path)
                self.assertEqual(data["hr_rows"][0]["slot_1b"], "000000820")
                self.assertEqual(data["field_mapping"]["slot_1b"], "haraimodoshi_tansho_1b")
                self.assertFalse(data["proposed_source"]["semantics_reviewed"])
                self.assertFalse((path.parent / "ledger.json").exists())
            self.run_mock(confirm=review)
        self.assertTrue(decoder.called)

    def test_unconfirmed_unit_does_not_create_a_settled_ledger(self):
        with self.assertRaises(core.SimulationError):
            self.run_mock(confirm=core.SimulationError("mapping_unconfirmed"))
        ledgers = list((self.root / "simulations").rglob("ledger.json"))
        self.assertEqual(ledgers, [self.parent / "ledger.json"])

    def test_missing_hr_does_not_invent_zero_losses_or_prompt_for_units(self):
        result, confirmation = self.run_mock(rows=[])
        confirmation.assert_not_called()
        self.assertEqual(result["settlement_status"], "pending_no_matching_hr")
        self.assertIsNone(result["summary"]["ai"]["net_profit_yen"])
        self.assertEqual(result["summary"]["ai"]["unsettled_count"], 2)

    def test_reviewed_profile_reuses_existing_mapping_without_another_prompt(self):
        profile = self.root / "source.json"
        config = source()
        config["race_keys"].append(list(race(number="02")))
        local.write(profile, config)
        self.args.payout_source = profile
        result, confirmation = self.run_mock()
        confirmation.assert_not_called()
        self.assertEqual(result["summary"]["ai"]["unsettled_count"], 0)

    def test_tampered_parent_stops_before_authentication(self):
        path = self.parent / "ledger.json"
        path.chmod(0o666)
        path.write_text("{}", encoding="utf-8")
        with patch.object(cli, "password") as secret:
            with self.assertRaises(core.SimulationError):
                cli.run(self.args)
        secret.assert_not_called()

    def test_limits_path_and_execution_flag_are_checked_before_authentication(self):
        for changes in ({"execute_local": False}, {"max_races": 1}, {"max_rows": 1001},
                        {"max_seconds": 121}, {"query_timeout_ms": 0}, {"simulation_id": "../OneDrive"}):
            args = deepcopy(self.args)
            for name, value in changes.items():
                setattr(args, name, value)
            with patch.object(cli, "password") as secret:
                with self.assertRaises(core.SimulationError):
                    cli.run(args)
            secret.assert_not_called()

    def test_reuse_of_an_unreviewed_profile_stops_before_authentication(self):
        for field, value in (("semantics_reviewed", False), ("race_keys", [list(race())])):
            config = source()
            config[field] = value
            path = self.root / (field + ".json")
            local.write(path, config)
            self.args.payout_source = path
            with patch.object(cli, "password") as secret:
                with self.assertRaises(core.SimulationError):
                    cli.run(self.args)
            secret.assert_not_called()

    def test_existing_sql_reads_only_requested_hr_keys_and_preserves_raw_strings(self):
        db = MagicMock()
        row = native_hr()
        raw = {**dict(zip(RACE_KEY, race())), **{k: v for k, v in row.items() if k != "race_key"}}
        db.iter_batches.return_value = iter(([raw],))
        with patch.object(local, "table_info", return_value=columns()):
            rows, _, status = settle_saved.fetch_hr(db, (race(),), max_rows=5)
        query_id, sql, params = db.iter_batches.call_args.args
        validate_audit_sql(sql)
        self.assertEqual(query_id, "v02.rows.jvd_hr")
        self.assertIn('FROM ONLY public."jvd_hr"', sql)
        self.assertIn("JOIN (VALUES", sql)
        self.assertEqual(sql.count("= requested."), 6)
        self.assertNotIn("jvd_o1", sql)
        self.assertNotIn("jvd_ra", sql)
        self.assertEqual(params, (*race(), 6))
        self.assertEqual(rows[0]["slot_1b"], "000000820")
        self.assertEqual(status["rows"], 1)

    def test_unsupported_storage_type_stops_before_hr_rows(self):
        db = MagicMock()
        metadata = columns()
        metadata["haraimodoshi_tansho_1b"]["type_name"] = "numeric"
        with patch.object(local, "table_info", return_value=metadata):
            with self.assertRaises(core.SimulationError):
                settle_saved.fetch_hr(db, (race(),))
        db.iter_batches.assert_not_called()

    def test_zero_join_adds_one_date_bounded_count_without_looser_matching(self):
        for counts, reason in (({"target_date_rows": 0, "jra_target_date_rows": 0}, "hr_target_date_not_found"),
                               ({"target_date_rows": 2, "jra_target_date_rows": 0}, "hr_jra_target_date_not_found"),
                               ({"target_date_rows": 3, "jra_target_date_rows": 3}, "hr_on_date_but_target_keys_unmatched")):
            db = MagicMock()
            db.iter_batches.return_value = iter(())
            db.aggregate.return_value = [counts]
            with patch.object(local, "table_info", return_value=columns()):
                rows, _, status = settle_saved.fetch_hr(db, (race(),))
            self.assertEqual(rows, [])
            self.assertEqual(status["target_date_availability"]["reason"], reason)
            db.iter_batches.assert_called_once()
            db.aggregate.assert_called_once()
            query_id, sql, params = db.aggregate.call_args.args
            validate_audit_sql(sql)
            self.assertEqual(query_id, "v02.hr.target_date_count")
            self.assertEqual(params, race()[:2])
            self.assertNotIn("min(", sql)
            self.assertNotIn("max(", sql)

    def test_zero_join_count_does_not_emit_unexpected_database_values(self):
        db = MagicMock()
        db.iter_batches.return_value = iter(())
        db.aggregate.return_value = [{"target_date_rows": "SYNTHETIC_PRIVATE", "jra_target_date_rows": 0}]
        with patch.object(local, "table_info", return_value=columns()):
            with self.assertRaises(core.SimulationError) as failure:
                settle_saved.fetch_hr(db, (race(),))
        self.assertEqual(failure.exception.code, "invalid_schema")
        db.iter_batches.return_value = iter(())
        db.aggregate.return_value = [{"target_date_rows": 0, "jra_target_date_rows": 0,
                                      "unexpected_field": "SYNTHETIC_PRIVATE"}]
        with patch.object(local, "table_info", return_value=columns()):
            _, _, status = settle_saved.fetch_hr(db, (race(),))
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(status))

    def test_unknown_failure_is_sanitized_by_the_existing_cli(self):
        output = io.StringIO()
        with patch.object(settle_saved, "run", side_effect=RuntimeError("SYNTHETIC_PASSWORD_AND_RECORD")), \
                redirect_stdout(output):
            result = cli.main(["settle-hr", "--execute-local", "--simulation-id", self.parent.name])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(output.getvalue()), {"status": "stopped", "reason": "internal_error"})
