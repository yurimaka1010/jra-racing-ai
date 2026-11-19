"""Mock-only launch probe; never opens a real DB or starts the recorder."""
from datetime import date
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('four_day_probe',Path(__file__).resolve().parents[1]/'scripts/d39_four_day_probe.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProbeTests(unittest.TestCase):
    def test_entries_enable_recording_using_only_today_declared_count(self):
        db=Mock();db.aggregate.return_value=[{'entry_races':24}]
        d=date(2026,10,31)
        self.assertEqual(m.probe(db,d,today=d)['status'],'ready')
        args=db.aggregate.call_args.args
        self.assertEqual(args[2],('2026','1031'))
        for word in ('jvd_se','jvd_hr','chakujun','haraimodoshi'):
            self.assertNotIn(word,args[1])

    def test_no_entries_exits_without_inference(self):
        db=Mock();db.aggregate.return_value=[{'entry_races':0}]
        self.assertEqual(m.probe(db,date(2026,11,2),today=date(2026,11,2))['status'],'no_entries')

    def test_other_day_is_rejected_before_sql(self):
        db=Mock()
        with self.assertRaises(ValueError):m.probe(db,date(2026,1,1),today=date(2026,10,31))
        db.aggregate.assert_not_called()

    def test_database_failure_is_not_misreported_as_non_racing_day(self):
        db=Mock();db.aggregate.side_effect=RuntimeError('mock failure')
        with self.assertRaises(RuntimeError):m.probe(db,date(2026,10,31),today=date(2026,10,31))

    def test_bad_count_is_rejected(self):
        for rows in ([],[{'entry_races':None}],[{'entry_races':-1}]):
            db=Mock();db.aggregate.return_value=rows
            with self.assertRaises(ValueError):m.probe(db,date(2026,10,31),today=date(2026,10,31))
