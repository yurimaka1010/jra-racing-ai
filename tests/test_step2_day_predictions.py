"""Synthetic checks for the one-day STEP 2 prediction adapter."""
from datetime import datetime
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from jra_racing_ai.win_sim import core
from test_ml_v03 import race_rows

SPEC = importlib.util.spec_from_file_location('step2_day', Path(__file__).resolve().parents[1] / 'scripts' / 'predict_20261018_step2.py')
day_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(day_run)


class DayPredictionTests(unittest.TestCase):
    def setUp(self):
        self.rows = race_rows(day_run.DAY)
        self.race = tuple(self.rows[0][key] for key in day_run.RACE_KEY)
        self.clock = datetime(2026, 10, 18, 9, 30, tzinfo=core.JST)
        self.ra = [dict(race_key=self.race, record_type='RA', status='2', start='0945')]

    def test_complete_upcoming_entries_accepted_and_started_race_excluded(self):
        rows, starts, skipped = day_run.eligible(self.rows, self.ra, self.clock)
        self.assertEqual(rows, self.rows)
        self.assertFalse(skipped)
        rows, _, skipped = day_run.eligible(self.rows, self.ra, self.clock.replace(minute=45))
        self.assertFalse(rows)
        self.assertEqual(skipped[self.race], 'already_started')

    def test_incomplete_card_is_not_imputed(self):
        rows, _, skipped = day_run.eligible(self.rows[:-1], self.ra, self.clock)
        self.assertFalse(rows)
        self.assertEqual(skipped[self.race], 'entry_data_incomplete')

    def test_raw_scores_receive_temperature_once_and_rank_deterministically(self):
        horses = [row['horse_number'] for row in self.rows]
        raw = {horse: (index + 1) / 100 for index, horse in enumerate(horses)}
        prediction = core.PredictionRace(self.race, raw)
        identities = {(*self.race, h): 'SYNTHETIC' for h in horses}
        policy = core.WinPolicy(temperature=.75)
        output = day_run.calibrated_rows([prediction], identities, {self.race: self.clock.replace(hour=12)}, policy, self.clock, {})
        denominator = sum(p ** (1 / .75) for p in raw.values())
        self.assertEqual(output[0]['prediction_rank'], 1)
        self.assertEqual(output[0]['horse_number'], horses[-1])
        self.assertAlmostEqual(sum(row['win_probability'] for row in output), 1.)
        for row in output:
            self.assertAlmostEqual(row['win_probability'], raw[row['horse_number']] ** (1/.75) / denominator)
            self.assertEqual(row['temperature'], .75)

    def test_scoped_reader_adds_bound_parameters_before_limit(self):
        db = MagicMock()
        mapping = {'columns': {'race_status': 'data_kubun', 'horse_status': 'data_kubun', 'horse_id': 'ketto_toroku_bango'}}
        sql = 'SELECT 1 FROM ONLY public.jvd_se se ORDER BY se.umaban LIMIT %s'
        reader = day_run.ScopedReader(db, mapping, horses={'SYNTHETIC'})
        reader.iter_batches('ml.export', sql, (100,), max_rows=99)
        args = db.iter_batches.call_args.args
        self.assertIn('= ANY(%s)', args[1])
        self.assertEqual(args[2], (['SYNTHETIC'], 100))
        self.assertNotIn('SYNTHETIC', args[1])
