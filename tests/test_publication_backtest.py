"""Publication-only replay with synthetic O1/RA rows and no database access."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai import win_sim_cli as cli
from jra_racing_ai.ml import data, safety
from jra_racing_ai.win_sim import core, formats, local, publication
from test_win_sim_v02 import race, example, policy


def payload(prices=("0018", "0040", "0070")):
    return "".join(f"{index + 1:02d}" + value + f"{index + 1:02d}" for index, value in enumerate(prices)) + " " * ((28 - len(prices)) * 8)


def odds_row(clock="01051448", **kwargs):
    return dict({"race_key": race(), "record_type": "O1", "status": "1", "published": clock,
                 "sale": "7", "payload": payload(), "oversized": False}, **kwargs)


def ra_row(**kwargs):
    return dict({"race_key": race(), "record_type": "RA", "status": "2", "created": "20300104",
                 "start": "1500", "previous_start": "0000", "oversized": False}, **kwargs)


def pub_policy():
    return replace(policy(), timing_basis=publication.TIMING_BASIS)


def native_case(rows=None):
    p, _, _, _ = example()
    schedules, _ = publication.schedules_from_ra([ra_row()], [p])
    snaps, failures, _, _ = publication.decode_odds([odds_row()] if rows is None else rows, format_reviewed=True)
    return p, schedules, snaps, failures


class PublicationTimeTests(unittest.TestCase):
    def test_observation_mode_stays_strict_and_publication_mode_keeps_null_observation(self):
        p, schedules, snaps, failures = native_case()
        strict = core.run_simulation([p], schedules, snaps, policy=policy())
        self.assertFalse(strict["rows"][0]["simulation_eligible"])
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy(), odds_failures=failures)
        row = result["rows"][0]
        self.assertTrue(row["simulation_eligible"])
        self.assertIsNone(row["odds_observed_at"])
        self.assertIsNone(row["schedule_observed_at"])
        self.assertFalse(row["actual_observation_time_verified"])
        self.assertFalse(row["purchase_availability_verified"])
        self.assertIn("暫定的な事後", result["notice"])
        self.assertEqual(result["summary"]["ai"]["unsettled_count"], 1)

    def test_minute_precision_at_decision_is_held(self):
        p, schedules, snaps, _ = native_case([odds_row("01051450")])
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy())
        self.assertEqual(result["rows"][0]["reason"], "odds_time_precision")
        self.assertEqual(result["summary"]["ai"]["bet_count"], 0)

    def test_last_full_minute_before_decision_is_selected_and_future_is_ignored(self):
        p, schedules, snaps, _ = native_case([odds_row("01051448"), odds_row("01051449"),
                                             odds_row("01051450", payload=payload(("9900",) * 3))])
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy())
        self.assertIn("14:49:00", result["rows"][0]["odds_published_at"])
        self.assertEqual(result["rows"][0]["ai_horse"], "03")

    def test_bad_future_payload_does_not_poison_earlier_price(self):
        p, schedules, snaps, failures = native_case([odds_row(), odds_row("01051455", payload="BAD")])
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy(), odds_failures=failures)
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 1)

    def test_latest_invalid_price_prevents_stale_fallback(self):
        p, schedules, snaps, failures = native_case([odds_row("01051447"), odds_row(payload="BAD")])
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy(), odds_failures=failures)
        self.assertEqual(result["rows"][0]["reason"], "odds_invalid")

    def test_outcomes_do_not_change_either_strategy_choice(self):
        p, schedules, snaps, _ = native_case()
        first = core.run_simulation([p], schedules, snaps, policy=pub_policy())
        second = core.run_simulation([p], schedules, snaps,
            [core.Outcome(race(), "normal_final", {"01": 110}, True, True)], policy=pub_policy())
        for field in ("ai_horse", "favorite_horse", "runners", "decision_at"):
            self.assertEqual(first["rows"][0][field], second["rows"][0][field])

    def test_mixed_clock_precision_is_not_silently_ordered(self):
        p, schedules, snaps, _ = native_case()
        snaps.append(replace(snaps[0], published_at=snaps[0].published_at + timedelta(seconds=20), publication_precision="second"))
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy())
        self.assertEqual(result["rows"][0]["reason"], "odds_conflict")

    def test_native_packed_format_preserves_leading_zeros_and_decodes_by_explicit_scale(self):
        snaps, _, stats, examples = publication.decode_odds([odds_row()])
        self.assertEqual(str(snaps[0].odds["01"]), "1.8")
        self.assertEqual(examples[0]["raw_win_block"], payload())
        self.assertFalse(snaps[0].verified)
        self.assertIsNone(snaps[0].observed_at)
        self.assertEqual(stats["structurally_valid_intermediate_snapshots"], 1)

    def test_local_format_review_prefers_priced_examples_over_early_empty_snapshots(self):
        rows = [odds_row("01050900", payload=" " * 224), odds_row("01050901", payload="BAD"), odds_row()]
        _, _, _, examples = publication.decode_odds(rows)
        self.assertEqual(len(examples), 1)
        self.assertTrue(examples[0]["normal_priced_candidate"])
        self.assertEqual(examples[0]["raw_win_block"], payload())

    def test_blank_unregistered_slot_is_safe_but_expected_horse_missing_excludes_race(self):
        p, schedules, snaps, _ = native_case([odds_row(payload=payload()[:24] + "04" + " " * 6 + " " * (24 * 8))])
        self.assertIsNone(snaps[0].reason)
        p = replace(p, probabilities=dict(p.probabilities, **{"04": .1}))
        schedules = [replace(schedules[0], horses=("01", "02", "03", "04"))]
        result = core.run_simulation([p], schedules, snaps, policy=pub_policy())
        self.assertEqual(result["rows"][0]["reason"], "odds_incomplete")

    def test_bad_lengths_characters_positions_popularity_and_cap_are_rejected(self):
        values = (None, "", " " * 223, "０" * 224, "02" + payload()[2:],
                  payload()[:6] + "??" + payload()[8:], payload(("0000", "0040", "0070")))
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                snaps, _, _, _ = publication.decode_odds([odds_row(payload=value)])
                self.assertEqual(snaps[0].reason, "odds_invalid")
        snaps, _, _, _ = publication.decode_odds([odds_row(payload=payload(("9999", "0040", "0070")))])
        self.assertEqual(snaps[0].reason, "odds_capped")

    def test_final_records_without_publication_clock_are_never_decision_prices(self):
        rows = [odds_row()] + [odds_row(None, status=status) for status in ("2", "3", "4", "5")]
        snaps, failures, stats, _ = publication.decode_odds(rows, format_reviewed=True)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(failures, {})
        self.assertEqual(stats["category_5"], 1)

    def test_deleted_canceled_unknown_or_missing_clock_require_confirmation(self):
        for row in (odds_row(status="0"), odds_row(status="9"), odds_row(status="UNKNOWN"), odds_row(None)):
            _, failures, _, _ = publication.decode_odds([row])
            self.assertEqual(failures, {race(): "odds_unverified"})

    def test_duplicate_snapshots_and_no_sale_are_not_normalized_to_usable_prices(self):
        snaps, _, _, _ = publication.decode_odds([odds_row(), odds_row()])
        self.assertEqual(snaps[0].reason, "odds_conflict")
        for flag in ("0", "1", "3", None):
            snaps, _, _, _ = publication.decode_odds([odds_row(sale=flag)])
            self.assertEqual(snaps[0].reason, "unsupported_status")

    def test_advance_card_schedule_has_evidence_but_no_fabricated_observation(self):
        p, _, _, _ = example()
        schedules, counts = publication.schedules_from_ra([ra_row()], [p])
        self.assertIsNone(schedules[0].observed_at)
        self.assertEqual(schedules[0].availability_basis, "ra_advance_card")
        self.assertEqual(counts, {"advance_card_schedule": 1})

    def test_creation_date_alone_final_ra_changes_and_missing_schedule_are_not_accepted(self):
        p, _, _, _ = example()
        invalid = ([ra_row(status="7")], [ra_row(created="20300105")], [ra_row(created="20300106")],
                   [ra_row(previous_start="1450")], [ra_row(start="0000")], [ra_row(created=None)],
                   [ra_row(), ra_row()])
        for rows in invalid:
            schedules, _ = publication.schedules_from_ra(rows, [p])
            chosen, _, _ = core.choose_schedule(race(), schedules, pub_policy())
            self.assertIsNone(chosen)
        schedules, _ = publication.schedules_from_ra([], [p])
        self.assertEqual(schedules, [])

    def test_local_format_confirmation_never_echoes_input(self):
        with patch("builtins.input", return_value="SYNTHETIC_PRIVATE"), patch("sys.stderr", io.StringIO()) as output:
            with self.assertRaises(core.SimulationError):
                publication.confirm_raw_format(Path("synthetic-review.json"))
        self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
        self.assertIn("raw_win_block", output.getvalue())
        self.assertIn("odds_tansho", output.getvalue())


class PublicationLocalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.legacy, self.root = self.base / "v01", self.base / "v02"
        self.legacy.mkdir()
        self.root.mkdir()
        self.patches = [patch.object(safety, "ML_ROOT", self.legacy), patch.object(local, "ROOT", self.root)]
        for item in self.patches:
            item.start()
        self.p, _, _, _ = example()
        self.entries = safety.new_artifact("datasets") / "runners.csv"
        with self.entries.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data.DATA_FIELDS)
            writer.writeheader()
            writer.writerows({**dict(zip(data.RACE_KEY, race())), "horse_number": h} for h in self.p.probabilities)
        self.metadata = {"format_version": 1, "complete": True, "source": data.ENTRY_SOURCE,
                         "purpose": "prediction_only", "current_results_excluded": True, "rows": 3,
                         "csv_sha256": safety.digest(self.entries),
                         "source_status_counts": {"ra_races": {"2": 1}, "se_runners": {"2": 3}}}
        safety.write_json(self.entries.parent / "dataset.json", self.metadata)
        self.predictions = local.artifact("predictions")
        local.write(self.predictions / "predictions.json", {"format_version": 2, "mode": "hypothetical_backtest",
            "records": [{"entry_key": list(race()) + [h], "wager": "win", "probability": float(p)} for h, p in self.p.probabilities.items()]})
        self.manifest = {"format_version": 2, "mode": "hypothetical_backtest", "model_manifest_sha256": "SYNTHETIC_MODEL",
                         "predictions_sha256": local.sha(self.predictions / "predictions.json"),
                         "entries_source": data.ENTRY_SOURCE, "entries_sha256": safety.digest(self.entries)}
        local.write(self.predictions / "manifest.json", self.manifest)
        self.policy_path = self.root / "policy.json"
        local.write(self.policy_path, local.policy_document(policy(), model_hash="SYNTHETIC_MODEL"))
        self.args = cli.parser().parse_args(["publication-run", "--execute-local", "--policy", str(self.policy_path),
                                            "--predictions", str(self.predictions), "--entries", str(self.entries)])

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temporary.cleanup()

    def run_mock(self, *, odds=None, confirmation=None):
        rows = [odds_row()] if odds is None else odds
        with patch.object(cli, "password", return_value="SYNTHETIC_PASSWORD"), patch.object(cli, "read_only_session"), \
                patch.object(publication, "fetch", return_value=([ra_row()], rows, [])), \
                patch.object(publication, "confirm_raw_format", side_effect=confirmation) as confirm:
            result = cli.run(self.args)
        return result, confirm

    def test_full_local_replay_saves_pending_choices_and_does_not_mutate_previous_policy(self):
        previous = local.sha(self.policy_path)
        result, confirm = self.run_mock()
        confirm.assert_called_once()
        self.assertEqual(result["summary"]["ai"]["bet_count"], 1)
        self.assertEqual(result["summary"]["favorite"]["bet_count"], 1)
        self.assertIsNone(result["summary"]["ai"]["net_profit_yen"])
        self.assertIsNone(result["summary"]["favorite"]["return_rate_percent"])
        self.assertEqual(previous, local.sha(self.policy_path))
        self.assertNotIn('"race_key"', json.dumps(result))
        folder = self.root / "simulations" / result["artifact_id"]
        review = local.read(folder / "raw_format_review.json")
        self.assertEqual(review["field_mapping"], {"odds_tansho": "examples[].raw_win_block"})
        self.assertEqual(review["examples"][0]["raw_win_block"], payload())
        ledger = local.read(folder / "ledger.json")
        self.assertIsNone(ledger["rows"][0]["odds_observed_at"])
        self.assertEqual(ledger["timing_basis"], publication.TIMING_BASIS)
        self.assertIn("暫定的な事後", (folder / "report.md").read_text(encoding="utf-8"))
        self.assertIn("1番人気購入", (folder / "report.md").read_text(encoding="utf-8"))

    def test_no_odds_returns_original_denominator_without_confirmation_or_fake_roi(self):
        result, confirm = self.run_mock(odds=[])
        confirm.assert_not_called()
        self.assertEqual(result["summary"]["target_race_count"], 1)
        self.assertEqual(result["summary"]["odds_acquisition_rate_percent"], 0)
        self.assertIsNone(result["summary"]["ai"]["return_rate_percent"])

    def test_unconfirmed_raw_format_creates_no_completed_simulation(self):
        with self.assertRaises(core.SimulationError):
            self.run_mock(confirmation=core.SimulationError("mapping_unconfirmed"))
        self.assertFalse(list((self.root / "simulations").rglob("ledger.json")))
        self.assertTrue(list((self.root / "simulations").rglob("raw_format_review.json")))

    def test_verified_payouts_allow_normal_comparison_without_enabling_hr_conversion(self):
        path = self.root / "payouts.json"
        local.write(path, {"format_version": 2, "source": "synthetic", "records": [{"race_key": list(race()),
            "status": "normal_final", "reviewed": True, "currency": "JPY", "base_stake_yen": 100,
            "evidence_basis": "official_published_payout", "complete_winner_set": True,
            "payouts": [{"horse_number": "03", "yen_per_100": 620}]}]})
        self.args.payouts = path
        result, _ = self.run_mock()
        self.assertEqual(result["summary"]["ai"]["net_profit_yen"], 520)
        self.assertEqual(result["summary"]["comparisons"]["all_common_eligible_races"]["race_count"], 1)

    def test_final_entry_categories_and_hash_mismatch_stop_before_authentication(self):
        for change in ("categories", "hash"):
            with self.subTest(change=change):
                metadata = deepcopy(self.metadata)
                manifest = deepcopy(self.manifest)
                if change == "categories":
                    metadata["source_status_counts"]["se_runners"] = {"6": 3}
                else:
                    manifest["entries_sha256"] = "WRONG"
                (self.entries.parent / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
                (self.predictions / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with patch.object(cli, "password") as secret, patch.object(cli, "read_only_session") as connect:
                    with self.assertRaises(core.SimulationError):
                        cli.run(self.args)
                secret.assert_not_called()
                connect.assert_not_called()

    def test_missing_execution_flag_and_limits_are_checked_before_authentication(self):
        for changes in ({"execute_local": False}, {"max_races": 0}, {"max_rows": 50001}, {"max_seconds": 121}):
            args = deepcopy(self.args)
            for name, value in changes.items():
                setattr(args, name, value)
            with patch.object(cli, "password") as secret, patch.object(cli, "read_only_session") as connect:
                with self.assertRaises(core.SimulationError):
                    cli.run(args)
            secret.assert_not_called()
            connect.assert_not_called()

    def test_saved_inputs_replay_with_null_times_and_minute_precision(self):
        result, _ = self.run_mock()
        folder = self.root / "simulations" / result["artifact_id"]
        schedules = formats.parse_schedules(local.read(folder / "input-schedules.json"))
        snapshots = formats.parse_snapshots(local.read(folder / "input-odds.json"))
        self.assertIsNone(snapshots[0].observed_at)
        self.assertEqual(snapshots[0].publication_precision, "minute")
        replay = core.run_simulation([self.p], schedules, snapshots, policy=pub_policy())
        self.assertEqual(replay["summary"]["ai"]["bet_count"], result["summary"]["ai"]["bet_count"])

    def test_fetch_reuses_total_row_budget_and_rejects_custom_types(self):
        races = (race(),)
        with patch.object(local, "table_info", return_value={name: {"type_is_builtin": True, "type_name": "varchar"}
                for name in {p for fields in (publication.RA_FIELDS, publication.O1_FIELDS) for p, _ in fields.values()}}), \
                patch.object(local, "_rows", side_effect=[[ra_row()], [odds_row()]]) as read:
            publication.fetch(MagicMock(), races, max_rows=10)
        self.assertEqual(read.call_args_list[1].args[-1], 9)
        with patch.object(local, "table_info", return_value={}):
            with self.assertRaises(core.SimulationError):
                publication.fetch(MagicMock(), races)
