"""Postrace checks use synthetic rows only; no database or model inference."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai.win_sim import core
from test_ml_v03 import race_rows

_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_20261017_saved.py"
_spec = importlib.util.spec_from_file_location("trial_postrace_test_module", _path)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


class TrialPostraceTests(unittest.TestCase):
    def fixture(self, extra=None):
        rows = race_rows(date(2026, 10, 17))
        if extra:
            rows += [dict(row, race_bango="02") for row in race_rows(date(2026, 10, 17))]
        raw, cal, ids, se, outcomes = {}, {}, {}, [], []
        for row in rows:
            identity, _ = evaluation.features.entry_identity(row)
            race, h = identity[:-1], identity[-1]
            raw.setdefault(race, {})[h] = Decimal(".81" if h == "01" else ".01")
            cal.setdefault(race, {})[h] = Decimal(".5625" if h == "01" else ".0625")
            ids[identity] = row["horse_id"]
            se.append(dict(row, status="6", finish_position=h))
        for race in raw:
            outcomes.append(core.Outcome(race, "normal_final", {"01": 1230}, True, True))
        anomalies = {"rows": []}
        if extra == "cancel":
            target = next(row for row in se if row["race_bango"] == "02" and row["horse_number"] == "01")
            target["finish_position"] = "00"
            next(row for row in se if row["race_bango"] == "02" and row["horse_number"] == "02")["finish_position"] = "01"
            outcomes[-1] = core.Outcome(outcomes[-1].race, "normal_final", {"02": 500}, True, True)
            anomalies["rows"].append(dict(target, anomaly_code="1"))
        elif extra == "dnf":
            target = next(row for row in se if row["race_bango"] == "02" and row["horse_number"] == "02")
            target["finish_position"] = "00"
            anomalies["rows"].append(dict(target, anomaly_code="4"))
        return {"se_rows": se, "hr_rows": [], "odds_rows": []}, anomalies, raw, cal, ids, outcomes

    def evaluate(self, extra=None):
        payload, anomalies, raw, cal, ids, outcomes = self.fixture(extra)
        before = deepcopy(cal)
        with patch.object(evaluation, "verified_outcomes", return_value=outcomes):
            summary, private = evaluation.evaluate(payload, anomalies, raw, cal, ids, core.WinPolicy(temperature=2))
        self.assertEqual(cal, before)
        return summary, private

    def test_saved_calibrated_probability_not_temperature_adjusted_twice(self):
        summary, _ = self.evaluate()
        expected = (-math.log(.5625) - 7 * math.log(1 - .0625)) / 8
        self.assertAlmostEqual(summary["probability_metrics"]["model"]["log_loss"], expected)
        self.assertFalse(summary["verification"]["double_calibration_applied"])
        self.assertIsNone(summary["final_odds_ev_reference"]["ai"]["return_rate_percent"])

    def test_cancelled_selection_is_pending_and_whole_race_excluded_from_metrics(self):
        summary, _ = self.evaluate("cancel")
        self.assertEqual(summary["probability_metrics"]["races"], 1)
        self.assertEqual(summary["probability_exclusion_reasons"]["nonstarter_race_fixed_probability_cohort"], 1)
        money = summary["top1_flat_100_reference"]["ai"]
        self.assertEqual(money["unsettled_count"], 1)
        self.assertIsNone(money["net_profit_yen"])
        self.assertEqual(money["pending_reasons"], {"refund_pending": 1})

    def test_official_dnf_is_nonwinner_not_a_nonstarter(self):
        summary, _ = self.evaluate("dnf")
        self.assertEqual(summary["probability_metrics"]["races"], 2)
        self.assertEqual(summary["probability_metrics"]["model"]["count"], 16)
        self.assertEqual(summary["top1_flat_100_reference"]["ai"]["unsettled_count"], 0)
