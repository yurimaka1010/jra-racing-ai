"""Version 0.2: fictional odds/payments and DB mocks only."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.ml import safety as old_io
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim import core, formats, local
from jra_racing_ai import win_sim_cli as cli


def race(day="0105", number="01"):
    return ("2030", day, "05", "01", "01", number)


def example(day="0105", number="01"):
    r = race(day, number)
    start = datetime(2030, int(day[:2]), int(day[2:]), 15, tzinfo=core.JST)
    prediction = core.PredictionRace(r, {"01": .5, "02": .3, "03": .2})
    schedule = core.Schedule(r, start, start.replace(hour=9), ("01", "02", "03"), True)
    snapshot = core.Snapshot(r, start-timedelta(minutes=12), start-timedelta(minutes=11),
                             {"01": "1.8", "02": "4.0", "03": "7.0"}, verified=True)
    outcome = core.Outcome(r, "normal_final", {"03": 820}, True, True)
    return prediction, schedule, snapshot, outcome


def policy(**kwargs):
    return core.WinPolicy(evaluation_start="2030-01-01", evaluation_end="2030-12-31", model_fit_end="2029-12-31", **kwargs)


def simulate(cases=None, **kwargs):
    cases = [example()] if cases is None else cases
    values = [[item[index] for item in cases if item[index] is not None] for index in range(4)]
    return core.run_simulation(*values, policy=policy(), **kwargs)


def source(layout="per_horse"):
    return {"table": "apd_sokuho_o1", "enabled": True, "layout": layout,
            "columns": {"status": "fake_status", "published": "fake_published", "observed": "fake_observed",
                        "horse": "fake_horse" if layout == "per_horse" else None,
                        "price": "fake_price" if layout == "per_horse" else None,
                        "payload": "fake_payload" if layout == "packed" else None},
            "time_formats": {"published": "iso", "observed": "iso"}, "allowed_statuses": ["1"],
            "price_encoding": {"format": "digits", "width": 4, "numerator": 1, "denominator": 10,
                               "unavailable_tokens": ["0000", "----"]},
            "packed": {"slots": 4, "stride": 8, "horse_offset": 0, "horse_width": 2,
                       "price_offset": 2, "price_width": 4, "empty_horse_codes": ["00"]},
            "confirmed": {field: True for field in ("official_format", "pc_keiba_mapping", "observed_time_semantics",
                                                    "odds_unit", "revision_policy")},
            "official_reference": "SYNTHETIC_SPEC", "local_mapping_reference": "SYNTHETIC_MAPPING"}


def hr_source():
    return {"table": "jvd_hr", "reviewed_normal_conditions": True, "currency": "JPY", "base_stake_yen": 100,
            "final_category": "2", "revision_policy": "single_reviewed_final", "semantics_reviewed": True,
            "official_reference": "SYNTHETIC_SPEC", "local_reconciliation_reference": "SYNTHETIC_EVIDENCE",
            "years": [2030], "venues": ["05"], "numerator": 1, "denominator": 1,
            "columns": {"status": "fake_status", "unestablished": "fake_unestablished", "special": "fake_special", "refund": "fake_refund"}}


def hr_row(r=None):
    row = {"race_key": race() if r is None else r, "status": "2", "unestablished": "0", "special": "0", "refund": "0"}
    for index in range(1, 4):
        row.update({f"slot_{index}a": "  ", f"slot_{index}b": " "*9, f"slot_{index}c": "  "})
    row.update(slot_1a="03", slot_1b="000000820", slot_1c="03")
    return row


def odds_rows():
    return [{"race_key": race(), "status": "1", "published": "2030-01-05T14:48:00+09:00",
             "observed": "2030-01-05T14:49:00+09:00", "horse": h, "price": p}
            for h, p in (("01", "0018"), ("02", "0040"), ("03", "0070"))]


class AsOfOddsTests(unittest.TestCase):
    def setUp(self):
        self.prediction, self.schedule, self.snapshot, self.outcome = example()
        self.decision = self.schedule.scheduled_start-timedelta(minutes=10)

    def test_latest_before_cutoff(self):
        older = replace(self.snapshot, observed_at=self.snapshot.observed_at-timedelta(minutes=1))
        chosen, reason = core.choose_snapshot(race(), [self.snapshot, older], self.decision, self.schedule.horses)
        self.assertIsNone(reason)
        self.assertEqual(chosen.observed_at, self.snapshot.observed_at)

    def test_future_quote_is_excluded(self):
        future = replace(self.snapshot, published_at=self.decision+timedelta(seconds=1),
                         observed_at=self.decision+timedelta(seconds=2), odds={"01": "999", "02": "999", "03": "999"})
        result = core.run_simulation([self.prediction], [self.schedule], [self.snapshot, future], [self.outcome], policy=policy())
        self.assertEqual(result["rows"][0]["ai_horse"], "03")
        self.assertEqual(result["rows"][0]["runners"][0]["decision_odds"], 1.8)

    def test_exact_cutoff_is_eligible(self):
        item = replace(self.snapshot, published_at=self.decision, observed_at=self.decision)
        self.assertIsNotNone(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[0])

    def test_publication_earlier_but_observation_late_is_excluded(self):
        item = replace(self.snapshot, observed_at=self.decision+timedelta(seconds=1))
        self.assertEqual(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[1], "odds_after_decision")

    def test_publication_after_observation_is_invalid(self):
        item = replace(self.snapshot, published_at=self.decision, observed_at=self.decision-timedelta(seconds=1))
        self.assertEqual(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[1], "odds_invalid")

    def test_latest_incomplete_blocks_stale_fallback(self):
        item = replace(self.snapshot, observed_at=self.decision, odds={"01": "2"})
        self.assertEqual(core.choose_snapshot(race(), [self.snapshot, item], self.decision, self.schedule.horses)[1], "odds_incomplete")

    def test_latest_invalid_does_not_remove_bad_horse(self):
        item = replace(self.snapshot, odds={"01": "2", "02": "0", "03": "5"})
        self.assertEqual(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[1], "odds_invalid")

    def test_capped_quote_is_not_an_exact_scalar(self):
        item = replace(self.snapshot, odds={"01": "2", "02": "999.9", "03": "5"})
        self.assertEqual(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[1], "odds_capped")

    def test_unknown_observation_semantics_are_unusable(self):
        item = replace(self.snapshot, verified=False)
        self.assertEqual(core.choose_snapshot(race(), [item], self.decision, self.schedule.horses)[1], "odds_unverified")

    def test_duplicate_snapshot_is_ambiguous(self):
        self.assertEqual(core.choose_snapshot(race(), [self.snapshot, self.snapshot], self.decision, self.schedule.horses)[1], "odds_conflict")

    def test_naive_time_is_rejected(self):
        with self.assertRaises(core.SimulationError):
            core.instant("2030-01-05T14:50:00")

    def test_utc_is_equivalent_to_jst(self):
        self.assertEqual(core.instant(self.decision.astimezone(timezone.utc)), self.decision)

    def test_delayed_schedule_arriving_after_original_decision_is_ignored(self):
        late = replace(self.schedule, scheduled_start=self.schedule.scheduled_start+timedelta(hours=1),
                       observed_at=self.decision+timedelta(minutes=1))
        chosen, cutoff, reason = core.choose_schedule(race(), [late, self.schedule], policy())
        self.assertIsNone(reason)
        self.assertEqual(cutoff, self.decision)

    def test_delay_known_before_decision_updates_schedule(self):
        early = replace(self.schedule, scheduled_start=self.schedule.scheduled_start+timedelta(hours=1),
                        observed_at=self.decision-timedelta(minutes=1))
        chosen, cutoff, reason = core.choose_schedule(race(), [early, self.schedule], policy())
        self.assertEqual(cutoff, self.decision+timedelta(hours=1))

    def test_schedule_conflict_is_held(self):
        conflicting = replace(self.schedule, scheduled_start=self.schedule.scheduled_start+timedelta(minutes=10))
        self.assertEqual(core.choose_schedule(race(), [self.schedule, conflicting], policy())[2], "schedule_conflict")

    def test_schedule_first_observed_after_cutoff_is_held(self):
        item = replace(self.schedule, observed_at=self.decision+timedelta(seconds=1))
        self.assertEqual(core.choose_schedule(race(), [item], policy())[2], "schedule_after_decision")

    def test_unverified_future_schedule_does_not_change_past_decision(self):
        late = replace(self.schedule, observed_at=self.decision+timedelta(minutes=2), verified=False)
        self.assertEqual(core.choose_schedule(race(), [self.schedule, late], policy())[1], self.decision)

    def test_roster_must_match_prediction_and_odds(self):
        item = replace(self.prediction, probabilities={"01": .7, "02": .3})
        result = core.run_simulation([item], [self.schedule], [self.snapshot], policy=policy())
        self.assertEqual(result["rows"][0]["reason"], "roster_mismatch")
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 1)

    def test_configurable_cutoff(self):
        self.assertEqual(core.choose_schedule(race(), [self.schedule], policy(minutes_before_start=20))[1],
                         self.schedule.scheduled_start-timedelta(minutes=20))

    def test_unknown_timestamp_failure_holds_only_affected_race(self):
        cases = [example(), example("0106")]
        result = simulate(cases, odds_failures={race(): "odds_unverified"})
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 1)


class DecisionSettlementReportTests(unittest.TestCase):
    def test_probabilities_checked_even_without_odds(self):
        prediction, _, _, _ = example()
        result = core.run_simulation([prediction], policy=policy())
        self.assertEqual(result["summary"]["raw_probability_sum_distribution"]["count"], 1)
        self.assertEqual(result["summary"]["probability_status_counts"], {"valid": 1})
        bad = replace(prediction, probabilities={"01": -1, "02": 2})
        result = core.run_simulation([bad], policy=policy())
        self.assertEqual(result["summary"]["probability_status_counts"], {"invalid_probability": 1})
    def test_ev_selects_one_largest_and_fixed_100(self):
        result = simulate()
        row = result["rows"][0]
        self.assertEqual(row["ai_horse"], "03")
        self.assertEqual(row["favorite_horse"], "01")
        self.assertEqual(row["ai"]["stake_yen"], 100)
        self.assertAlmostEqual(row["runners"][2]["expected_return"], .4)

    def test_five_percent_boundary_is_inclusive(self):
        prediction, schedule, snapshot, outcome = example()
        prediction = replace(prediction, probabilities={"01": .35, "02": .65})
        schedule = replace(schedule, horses=("01", "02"))
        snapshot = replace(snapshot, odds={"01": "3.0", "02": "1.0"})
        result = core.run_simulation([prediction], [schedule], [snapshot], policy=policy())
        self.assertEqual(result["rows"][0]["ai_horse"], "01")

    def test_below_threshold_skips(self):
        p, s, o, r = example()
        o = replace(o, odds={"01": "1.1", "02": "1.1", "03": "1.1"})
        result = core.run_simulation([p], [s], [o], [r], policy=policy())
        self.assertIsNone(result["rows"][0]["ai_horse"])
        self.assertEqual(result["summary"]["ai"]["bet_count"], 0)
        self.assertIsNone(result["summary"]["ai"]["return_rate_percent"])

    def test_favorite_tie_uses_lowest_horse_number(self):
        p, s, o, r = example()
        o = replace(o, odds={"01": "2", "02": "2", "03": "3"})
        result = core.run_simulation([p], [s], [o], [r], policy=policy())
        self.assertEqual(result["rows"][0]["favorite_horse"], "01")

    def test_probability_sum_is_checked_and_normalized_before_ev(self):
        values, total = core.probabilities({"01": .9, "02": .9}, policy())
        self.assertEqual(total, core.decimal("1.8"))
        self.assertEqual(values["01"], core.decimal(".5"))
        self.assertEqual(sum(values.values()), 1)

    def test_identity_does_not_silently_fix_sum(self):
        with self.assertRaises(core.SimulationError):
            core.probabilities({"01": .2, "02": .3}, policy(probability_method="identity"))

    def test_invalid_probabilities_are_rejected(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), "SYNTHETIC_PRIVATE"):
            with self.subTest(value=value), self.assertRaises(core.SimulationError):
                core.probabilities({"01": value, "02": .5}, policy())

    def test_all_zero_probability_is_held(self):
        with self.assertRaises(core.SimulationError):
            core.probabilities({"01": 0, "02": 0}, policy())

    def test_official_payout_not_decision_odds(self):
        row = simulate()["rows"][0]
        self.assertEqual(row["runners"][2]["decision_odds"], 7)
        self.assertEqual(row["ai"]["payout_yen"], 820)
        self.assertEqual(row["ai"]["net_profit_yen"], 720)

    def test_official_payment_change_cannot_change_decision(self):
        p, s, o, r = example()
        first = core.run_simulation([p], [s], [o], [r], policy=policy())["rows"][0]
        second = core.run_simulation([p], [s], [o], [replace(r, payouts_yen_per_100={"01": 110})], policy=policy())["rows"][0]
        self.assertEqual(first["ai_horse"], second["ai_horse"])
        self.assertEqual(second["ai"]["payout_yen"], 0)

    def test_dead_heat_preserves_each_official_payment(self):
        outcome = core.Outcome(race(), "normal_final", {"01": 110, "03": 490}, True, True)
        self.assertEqual(core.settle("03", outcome, ("01", "02", "03"))["payout_yen"], 490)
        self.assertEqual(core.settle("01", outcome, ("01", "02", "03"))["payout_yen"], 110)

    def test_plus_ten_amount_is_not_recomputed(self):
        outcome = core.Outcome(race(), "normal_final", {"01": 110}, True, True)
        self.assertEqual(core.settle("01", outcome, ("01", "02"))["net_profit_yen"], 10)

    def test_incomplete_winner_set_never_produces_loser_zero(self):
        result = core.settle("02", core.Outcome(race(), "normal_final", {"01": 300}, True, False), ("01", "02"))
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["payout_yen"])

    def test_special_refund_canceled_no_sale_are_pending(self):
        for status in ("refund", "special", "canceled", "no_sale", "unestablished", "pending"):
            result = core.settle("01", core.Outcome(race(), status), ("01", "02"))
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["net_profit_yen"])

    def test_mixed_normal_and_unconfirmed_not_blanket_blocked(self):
        cases = [example(), example("0106")]
        p, s, o, r = cases[1]
        cases[1] = p, s, o, replace(r, verified=False)
        summary = simulate(cases)["summary"]
        self.assertEqual(summary["ai"]["unsettled_count"], 1)
        self.assertEqual(summary["ai"]["settled_bet_count"], 1)
        self.assertIsNone(summary["ai"]["net_profit_yen"])
        self.assertEqual(summary["ai"]["settled_subset"]["net_profit_yen"], 720)
        self.assertEqual(summary["comparisons"]["all_common_eligible_races"]["race_count"], 1)

    def test_duplicate_final_versions_are_pending(self):
        p, s, o, r = example()
        result = core.run_simulation([p], [s], [o], [r, r], policy=policy())
        self.assertEqual(result["rows"][0]["ai"]["reason"], "result_duplicate")

    def test_original_population_is_odds_coverage_denominator(self):
        cases = [example(), example("0106"), example("0107")]
        p, s, o, r = cases[2]
        cases[2] = p, s, None, r
        summary = simulate(cases)["summary"]
        self.assertEqual(summary["target_race_count"], 3)
        self.assertEqual(summary["odds_acquired_race_count"], 2)
        self.assertAlmostEqual(summary["odds_acquisition_rate_percent"], 200/3)
        self.assertEqual(summary["ai"]["investment_yen"], 200)

    def test_schedule_missing_still_counts_in_original_population(self):
        p, s, o, r = example()
        summary = core.run_simulation([p], [], [o], [r], policy=policy())["summary"]
        self.assertEqual(summary["target_race_count"], 1)
        self.assertEqual(summary["odds_acquisition_rate_percent"], 0)

    def test_monthly_and_favorite_roi_computed_from_payments(self):
        first = example()
        p, s, o, r = example("0205")
        second = p, s, o, replace(r, payouts_yen_per_100={"01": 110})
        summary = simulate([first, second])["summary"]
        self.assertEqual(summary["ai"]["investment_yen"], 200)
        self.assertEqual(summary["ai"]["net_profit_yen"], 620)
        self.assertEqual(summary["ai"]["return_rate_percent"], 410)
        self.assertEqual(summary["favorite"]["return_rate_percent"], 55)
        self.assertEqual(summary["by_month"]["2030-02"]["ai"]["net_profit_yen"], -100)
        self.assertEqual(summary["monthly_stability"]["worst_month_yen"], -100)

    def test_common_race_and_ai_bought_race_comparisons(self):
        cases = [example(), example("0106")]
        p, s, o, r = cases[1]
        cases[1] = p, s, replace(o, odds={"01": "1.1", "02": "1.1", "03": "1.1"}), r
        comparison = simulate(cases)["summary"]["comparisons"]
        self.assertEqual(comparison["all_common_eligible_races"]["favorite"]["bet_count"], 2)
        self.assertEqual(comparison["ai_purchased_common_races"]["favorite"]["bet_count"], 1)

    def test_outside_test_window_cannot_be_silently_scored(self):
        p, s, o, r = example()
        with self.assertRaises(core.SimulationError):
            core.run_simulation([p], [s], [o], [r], policy=replace(policy(), evaluation_start="2030-02-01"))

    def test_calibration_train_test_periods_separated(self):
        p, _, _, _ = example()
        result = core.fit_temperature([p], {race(): "01"}, model_fit_end="2029-12-31", test_start="2030-02-01")
        self.assertEqual(result["calibration_end"], "2030-01-05")
        with self.assertRaises(core.SimulationError):
            core.fit_temperature([p], {race(): "01"}, model_fit_end="2029-12-31", test_start="2030-01-05")

    def test_policy_rejects_calibration_or_fit_touching_test(self):
        for kwargs in ({"calibration_end": "2030-01-01"}, {"model_fit_end": "2030-01-01"},
                       {"threshold_selection_end": "2030-02-01"}):
            with self.assertRaises(core.SimulationError):
                replace(policy(), **kwargs).validate()


class FormatAdapterTests(unittest.TestCase):
    def test_bad_normalized_odds_only_holds_one_race(self):
        _, _, snapshot, _ = example()
        good = {"race_key": list(race()), "published_at": snapshot.published_at.isoformat(),
                "observed_at": snapshot.observed_at.isoformat(),
                "odds": [{"horse_number": h, "value": v} for h, v in snapshot.odds.items()],
                "availability_verified": True, "unit_verified": True}
        bad = deepcopy(good)
        bad["race_key"] = list(race("0106"))
        bad["odds"][0]["value"] = None
        parsed = formats.parse_snapshots({"format_version": 2, "source": "synthetic", "records": [good, bad]})
        cases = [example(), example("0106")]
        result = core.run_simulation([item[0] for item in cases], [item[1] for item in cases], parsed,
                                     [item[3] for item in cases], policy=policy())
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 1)

    def test_bad_schedule_only_holds_one_race(self):
        cases = [example(), example("0106")]
        good = {"race_key": list(race()), "scheduled_start": cases[0][1].scheduled_start.isoformat(),
                "observed_at": cases[0][1].observed_at.isoformat(), "horses": ["01", "02", "03"], "availability_verified": True}
        bad = dict(good, race_key=list(race("0106")), scheduled_start=None)
        schedules = formats.parse_schedules({"format_version": 2, "source": "synthetic", "records": [good, bad]})
        result = core.run_simulation([item[0] for item in cases], schedules, [item[2] for item in cases],
                                     [item[3] for item in cases], policy=policy())
        self.assertEqual(result["summary"]["simulation_race_count"], 1)
    def test_per_horse_scaled_digits_and_leading_zeros(self):
        snapshots, failed = formats.decode_odds_rows(odds_rows(), source())
        self.assertEqual(failed, {})
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].odds["01"], core.decimal("1.8"))

    def test_fraction_unit_is_not_assumed(self):
        config = source()
        config["price_encoding"]["denominator"] = 100
        snapshots, _ = formats.decode_odds_rows(odds_rows(), config)
        self.assertEqual(snapshots[0].reason, "odds_invalid")

    def test_duplicate_horse_marks_whole_snapshot_bad(self):
        rows = odds_rows()
        rows.append(rows[0])
        snapshots, _ = formats.decode_odds_rows(rows, source())
        self.assertEqual(snapshots[0].reason, "odds_conflict")

    def test_unknown_status_is_not_filtered_out_to_reveal_older_quote(self):
        rows = odds_rows()
        rows[0]["status"] = "5"
        snapshots, _ = formats.decode_odds_rows(rows, source())
        self.assertEqual(snapshots[0].reason, "unsupported_status")

    def test_unavailable_tokens_and_bad_width_mark_snapshot_invalid(self):
        for value in (None, "0000", "----", "020", "００２０", "00 2", "-020"):
            rows = odds_rows()
            rows[0]["price"] = value
            parsed, _ = formats.decode_odds_rows(rows, source())
            self.assertEqual(parsed[0].reason, "odds_invalid")

    def test_packed_fixture_fields_no_official_layout_assumption(self):
        row = odds_rows()[0]
        row["payload"] = "01001801" + "02004002" + "03007003" + "00      "
        parsed, _ = formats.decode_odds_rows([row], source("packed"))
        self.assertEqual(parsed[0].odds, {"01": core.decimal("1.8"), "02": core.decimal("4"), "03": core.decimal("7")})

    def test_packed_length_never_padded(self):
        row = odds_rows()[0]
        row["payload"] = "01001801"
        parsed, _ = formats.decode_odds_rows([row], source("packed"))
        self.assertEqual(parsed[0].reason, "odds_invalid")

    def test_unknown_timestamp_has_no_creation_date_fallback(self):
        rows = odds_rows()
        rows[0]["observed"] = None
        parsed, failed = formats.decode_odds_rows(rows, source())
        self.assertEqual(failed, {race(): "odds_unverified"})

    def test_local_time_encodings_explicit(self):
        self.assertEqual(formats.parse_time("01051449", "mmddhhmm", race()).hour, 14)
        self.assertEqual(formats.parse_time("1449", "hhmm_on_race_date", race()).minute, 49)
        self.assertEqual(formats.parse_time("20300105144900", "yyyymmddhhmmss", race()).year, 2030)
        self.assertIsNotNone(formats.parse_time(datetime(2030, 1, 5, 14), "timestamp_jst", race()).tzinfo)

    def test_year_rollover_is_not_guessed(self):
        parsed = formats.parse_time("12312300", "mmddhhmm", race("0101"))
        self.assertEqual(parsed.year, 2030)
        self.assertGreater(parsed, datetime(2030, 1, 1, 15, tzinfo=core.JST))

    def test_missing_confirmation_gates_db_conversion(self):
        for field in source()["confirmed"]:
            config = source()
            config["confirmed"][field] = False
            with self.assertRaises(core.SimulationError):
                formats.validate_source(config)

    def test_final_odds_code_cannot_be_enabled(self):
        for status in ("0", "3", "4", "5", "9"):
            config = source()
            config["allowed_statuses"] = [status]
            with self.assertRaises(core.SimulationError):
                formats.validate_source(config)

    def test_official_raw_capped_value_is_not_converted_to_exact_price(self):
        rows = odds_rows()
        rows[0]["price"] = "9999"
        parsed, _ = formats.decode_odds_rows(rows, source())
        self.assertEqual(parsed[0].reason, "odds_capped")

    def test_later_final_record_without_publication_time_does_not_poison_history(self):
        rows = odds_rows()
        rows.append(dict(rows[0], status="5", published="00000000", observed="2030-01-05T17:00:00+09:00"))
        parsed, failed = formats.decode_odds_rows(rows, source())
        prediction, schedule, _, outcome = example()
        result = core.run_simulation([prediction], [schedule], parsed, [outcome], policy=policy(), odds_failures=failed)
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 1)

    def test_official_profile_is_a_reference_not_an_enabled_db_mapping(self):
        draft = local.draft_config()
        for item in draft["odds_sources"]:
            self.assertFalse(item["enabled"])
            self.assertFalse(item["confirmed"]["pc_keiba_mapping"])
            self.assertEqual(item["official_definition"]["slots"], 28)
            self.assertEqual(item["official_definition"]["stride"], 8)

    def test_column_injection_rejected_before_sql(self):
        config = source()
        config["columns"]["price"] = 'x; DELETE FROM y'
        with self.assertRaises(core.SimulationError):
            formats.validate_source(config)

    def test_hr_ordinary_explicit_scoped_conversion(self):
        result = formats.decode_hr_rows([hr_row()], hr_source())
        self.assertEqual(result[0].payouts_yen_per_100, {"03": 820})
        self.assertTrue(result[0].verified)

    def test_hr_money_and_scope_require_confirmation(self):
        config = hr_source()
        config["semantics_reviewed"] = False
        with self.assertRaises(core.SimulationError):
            formats.decode_hr_rows([hr_row()], config)
        config = hr_source()
        config["years"] = [2029]
        self.assertEqual(formats.decode_hr_rows([hr_row()], config)[0].reason, "outside_scope")

    def test_hr_width_null_special_initials_pending(self):
        for field, value in (("slot_1a", "00"), ("slot_1b", None), ("slot_1b", "000000000"),
                             ("slot_1b", "0000000820"), ("slot_1c", "00")):
            row = hr_row()
            row[field] = value
            result = formats.decode_hr_rows([row], hr_source())[0]
            self.assertEqual(result.status, "pending")

    def test_hr_revisions_duplicates_refunds_specials(self):
        row = hr_row()
        self.assertEqual(formats.decode_hr_rows([row, row], hr_source())[0].reason, "result_duplicate")
        for field, value, expected in (("status", "0", "result_revision_unconfirmed"),
                                       ("refund", "1", "refund_pending"), ("special", "1", "special_pending")):
            altered = dict(row, **{field: value})
            self.assertEqual(formats.decode_hr_rows([altered], hr_source())[0].reason, expected)

    def test_invalid_hr_case_does_not_disable_other_normal_race(self):
        bad = hr_row(race("0106"))
        bad["slot_1b"] = None
        result = formats.decode_hr_rows([hr_row(), bad], hr_source())
        self.assertEqual([outcome.status for outcome in result], ["normal_final", "pending"])

    def test_hr_fractional_yen_not_rounded(self):
        config = hr_source()
        config["denominator"] = 3
        self.assertEqual(formats.decode_hr_rows([hr_row()], config)[0].status, "pending")

    def test_manual_official_file_review_and_base_stake(self):
        row = {"race_key": list(race()), "status": "normal_final", "payouts": [{"horse_number": "03", "yen_per_100": 820}],
               "currency": "JPY", "base_stake_yen": 100, "reviewed": True,
               "evidence_basis": "official_published_payout", "complete_winner_set": True}
        doc = {"format_version": 2, "source": "local_reviewed", "records": [row]}
        self.assertTrue(formats.parse_outcomes(doc)[0].verified)
        row["base_stake_yen"] = 1000
        self.assertFalse(formats.parse_outcomes(doc)[0].verified)

    def test_bad_manual_payment_only_holds_that_race(self):
        good = {"race_key": list(race()), "status": "normal_final", "payouts": [{"horse_number": "03", "yen_per_100": 820}],
                "currency": "JPY", "base_stake_yen": 100, "reviewed": True,
                "evidence_basis": "official_published_payout", "complete_winner_set": True}
        bad = deepcopy(good)
        bad["race_key"] = list(race("0106"))
        bad["payouts"][0]["yen_per_100"] = "SYNTHETIC_SECRET"
        result = formats.parse_outcomes({"format_version": 2, "source": "local_reviewed", "records": [good, bad]})
        self.assertEqual([item.status for item in result], ["normal_final", "pending"])


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.legacy = self.base/"v02", self.base/"v01"
        self.root.mkdir()
        self.legacy.mkdir()
        self.patch_root = patch.object(local, "ROOT", self.root)
        self.patch_legacy = patch.object(old_io, "ML_ROOT", self.legacy)
        self.patch_root.start()
        self.patch_legacy.start()

    def tearDown(self):
        self.patch_legacy.stop()
        self.patch_root.stop()
        for path in self.base.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        self.temp.cleanup()

    def _predictions(self, model_hash="SYNTHETIC_MODEL", include_place=True):
        path = self.legacy/"predictions"/"synthetic"
        path.mkdir(parents=True)
        records = [{"entry_key": list(race())+[h], "wager": wager, "probability": p}
                   for wager in (("win", "place") if include_place else ("win",))
                   for h, p in (("01", .5), ("02", .3), ("03", .2))]
        old_io.write_json(path/"predictions.json", {"format_version": 1, "mode": "backtest", "records": records}, root=self.legacy)
        old_io.write_json(path/"manifest.json", {"format_version": 1, "mode": "backtest", "model_manifest_sha256": model_hash,
                            "predictions_sha256": old_io.digest(path/"predictions.json", root=self.legacy)}, root=self.legacy)
        return path

    def test_v01_ledger_reused_win_only_untouched(self):
        directory = self._predictions()
        before = old_io.digest(directory/"predictions.json", root=self.legacy)
        predictions, _ = local.load_predictions(directory, expected_model_hash="SYNTHETIC_MODEL")
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0].probabilities), 3)
        self.assertEqual(before, old_io.digest(directory/"predictions.json", root=self.legacy))

    def test_model_hash_mismatch_rejected(self):
        with self.assertRaises(core.SimulationError):
            local.load_predictions(self._predictions(), expected_model_hash="OTHER")

    def test_legacy_tamper_rejected(self):
        directory = self._predictions()
        with (directory/"predictions.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaises(core.SimulationError):
            local.load_predictions(directory)

    def test_sql_full_key_and_no_payload_text_cast(self):
        from jra_racing_ai.db_connection import validate_audit_sql
        columns = {name: {"type_is_builtin": True, "type_name": "bpchar"} for name in (*RACE_KEY, "fake_payload")}
        sql, params = local.build_rows_query("apd_sokuho_o1", {"payload": ("fake_payload", 1792)}, columns, [race()], 100)
        validate_audit_sql(sql)
        for name in RACE_KEY:
            self.assertIn('t."'+name+'" = %s', sql)
        self.assertIn('ELSE t."fake_payload" END', sql)
        self.assertNotIn('"fake_payload"::text', sql)
        self.assertEqual(params[-1], 101)

    def test_sql_unknown_table_custom_type_and_limit_rejected(self):
        columns = {name: {"type_is_builtin": True, "type_name": "bpchar"} for name in (*RACE_KEY, "fake")}
        for table, cap in (("unknown", 100), ("jvd_o1", 200001)):
            with self.assertRaises(core.SimulationError):
                local.build_rows_query(table, {"price": ("fake", 9)}, columns, [race()], cap)
        columns["fake"]["type_is_builtin"] = False
        with self.assertRaises(core.SimulationError):
            local.build_rows_query("jvd_o1", {"price": ("fake", 9)}, columns, [race()], 100)

    def test_disabled_mappings_do_not_read_real_rows(self):
        db = MagicMock()
        parsed, outcomes, failures, status = local.fetch_inputs(db, local.draft_config(), [race()])
        self.assertEqual(parsed, [])
        db.iter_batches.assert_not_called()
        db.aggregate.assert_not_called()

    def test_missing_odds_table_is_unavailable_not_fatal(self):
        config = {"format_version": 2, "odds_sources": [source()], "payout_source": None}
        with patch.object(local, "table_info", return_value=None):
            snapshots, outcomes, failures, status = local.fetch_inputs(MagicMock(), config, [race()])
        self.assertEqual(status[0]["status"], "not_found")
        self.assertEqual(snapshots, [])

    def test_db_mock_rows_feed_full_simulation(self):
        config = {"format_version": 2, "odds_sources": [source()], "payout_source": hr_source()}
        with patch.object(local, "table_info", return_value={}), patch.object(local, "_rows", side_effect=[odds_rows(), [hr_row()]]):
            snapshots, outcomes, failures, status = local.fetch_inputs(MagicMock(), config, [race()])
        p, s, _, _ = example()
        result = core.run_simulation([p], [s], snapshots, outcomes, policy=policy())
        self.assertEqual(result["summary"]["ai"]["net_profit_yen"], 720)

    def test_deadline_stops_before_query(self):
        clock = MagicMock(side_effect=[0, 11])
        budget = local.Budget(10, clock=clock)
        db = MagicMock()
        with self.assertRaises(core.SimulationError):
            budget.aggregate(db, "v02.fake", "SELECT 1")
        db.aggregate.assert_not_called()

    def test_one_drive_output_rejected(self):
        with self.assertRaises(core.SimulationError):
            local.artifact("demo", root=self.base/"OneDrive"/"v02")

    def test_project_or_arbitrary_input_rejected(self):
        with self.assertRaises(core.SimulationError):
            local.read(self.base/"outside.json")

    def test_report_has_aggregates_only_and_no_overwrite(self):
        result = simulate()
        directory = local.artifact("simulations")
        local.write_report(directory, result, policy_payload=local.policy_document(policy(), model_hash="synthetic"),
                           provenance={"source": "synthetic"})
        text = (directory/"report.md").read_text(encoding="utf-8")
        self.assertIn("判断時点の表示オッズは払戻額を保証しない", text)
        self.assertIn("架空データのみ", text)
        self.assertNotIn('"race_key"', text)
        self.assertNotIn('"horse_number"', text)
        with self.assertRaises(core.SimulationError):
            local.write(directory/"summary.json", {})

    def test_offline_cli_simulation_from_legacy_predictions(self):
        directory = self._predictions()
        policy_path = self.root/"policy.json"
        local.write(policy_path, local.policy_document(policy(), model_hash="SYNTHETIC_MODEL"))
        _, s, o, r = example()
        doc = {"format_version": 2, "source": "synthetic", "records": [
            {"race_key": list(race()), "scheduled_start": s.scheduled_start.isoformat(), "observed_at": s.observed_at.isoformat(),
             "horses": list(s.horses), "availability_verified": True}]}
        local.write(self.root/"schedules.json", doc)
        doc["records"] = [{"race_key": list(race()), "published_at": o.published_at.isoformat(), "observed_at": o.observed_at.isoformat(),
                           "odds": [{"horse_number": h, "value": value} for h, value in o.odds.items()],
                           "availability_verified": True, "unit_verified": True}]
        local.write(self.root/"odds.json", doc)
        doc["records"] = [{"race_key": list(race()), "status": "normal_final", "payouts": [{"horse_number": "03", "yen_per_100": 820}],
                           "currency": "JPY", "base_stake_yen": 100, "reviewed": True,
                           "evidence_basis": "official_published_payout", "complete_winner_set": True}]
        local.write(self.root/"payouts.json", doc)
        args = cli.parser().parse_args(["simulate", "--policy", str(policy_path), "--predictions", str(directory),
            "--schedules", str(self.root/"schedules.json"), "--odds", str(self.root/"odds.json"), "--payouts", str(self.root/"payouts.json")])
        with patch.object(cli, "read_only_session") as connect:
            result = cli.run(args)
        connect.assert_not_called()
        self.assertEqual(result["summary"]["ai"]["payout_yen"], 820)
        self.assertEqual(result["summary"]["favorite"]["payout_yen"], 0)

    def test_cli_exceptions_never_echo_private_values(self):
        stream = io.StringIO()
        with patch.object(cli, "run", side_effect=RuntimeError("SYNTHETIC_PASSWORD")), redirect_stdout(stream):
            self.assertEqual(cli.main(["demo"]), 2)
        self.assertNotIn("PASSWORD", stream.getvalue())

    def test_stdout_json_is_codepage_independent(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(cli.main(["demo"]), 0)
        self.assertTrue(stream.getvalue().isascii())
        self.assertIn("表示オッズ", json.loads(stream.getvalue())["notice"])

    def test_db_execution_needs_explicit_flag(self):
        with patch.object(cli, "read_only_session") as connect:
            with self.assertRaises(core.SimulationError):
                cli.run(cli.parser().parse_args(["availability"]))
        connect.assert_not_called()

    def test_non_loopback_or_admin_connection_rejected(self):
        for name, value in (("PGHOST", "example.com"), ("PGUSER", "postgres")):
            with patch.dict("os.environ", {name: value}, clear=True):
                with self.assertRaises(Exception):
                    cli.connection_config(5000)

    def test_demo_runs_without_db_and_place_simulation(self):
        from jra_racing_ai.win_sim.demo import run_demo
        with patch.object(cli, "read_only_session") as connect:
            result = cli.run(cli.parser().parse_args(["demo"]))
        connect.assert_not_called()
        self.assertEqual(result["summary"]["target_race_count"], 12)
        self.assertEqual(result["summary"]["odds_acquired_race_count"], 9)
        self.assertNotIn("place", result["summary"])


class ModelCompatibilityTests(unittest.TestCase):
    def test_v01_lightgbm_model_reused_and_place_preserved(self):
        from jra_racing_ai.ml import data, model
        from jra_racing_ai.ml.demo import synthetic_rows
        try:
            model.libraries()
        except MLFailure:
            self.skipTest("optional_ml_libraries_not_installed")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            legacy, current = base/"v01", base/"v02"
            legacy.mkdir()
            current.mkdir()
            with patch.object(old_io, "ML_ROOT", legacy), patch.object(local, "ROOT", current):
                rows = synthetic_rows(20)
                bundle = model.fit_models(rows, rounds=10)
                model_dir = old_io.new_artifact("models")
                model.save_models(bundle, model_dir)
                target = old_io.new_artifact("datasets")
                entries = data.write_synthetic_dataset(rows[-8:], target)
                history_dir = old_io.new_artifact("datasets")
                history = data.write_synthetic_dataset(rows[:-8], history_dir)
                before = old_io.digest(model_dir/"place.lgb.txt")
                loaded = model.load_models(model_dir)
                with patch.object(model, "load_models", return_value=loaded), \
                        patch.object(loaded["boosters"]["place"], "predict", side_effect=AssertionError("place_inference_forbidden")):
                    args = cli.parser().parse_args(["predict-win", "--model", str(model_dir), "--entries", str(entries), "--history", str(history)])
                    result = cli.run(args)
                predicted, _ = local.load_predictions(current/"predictions"/result["artifact_id"])
                self.assertEqual(result["runner_count"], 8)
                self.assertEqual(len(predicted), 1)
                self.assertEqual(before, old_io.digest(model_dir/"place.lgb.txt"))
            for path in base.rglob("*"):
                if path.is_file():
                    path.chmod(0o666)
