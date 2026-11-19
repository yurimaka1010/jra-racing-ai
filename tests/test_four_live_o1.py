"""Live transport regression: no source/time forgery or change to fixed H."""
from datetime import datetime
import unittest
from jra_racing_ai.d39_generation import four_sources as m
from jra_racing_ai.stage7 import inputs
from jra_racing_ai.win_sim import core
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.core import RACE_KEY

RACE=('2026','1024','05','04','05','04')
START='2026-10-24T11:00:00+09:00'


class CurrentO1TransportTests(unittest.TestCase):
    def test_only_live_table_changes_all_clock_parameters_identical(self):
        old,args=inputs.query([RACE],{RACE:START},(30,10),historical=False)
        new,actual=m.current_odds_query([RACE],{RACE:START})
        self.assertEqual(args,actual)
        self.assertEqual(new,old.replace('public.apd_sokuho_o1','public.jvd_o1'))
        self.assertNotIn('kakutei_chakujun',new);self.assertNotIn('haraimodoshi',new)
        self.assertIn('public.apd_sokuho_o1',inputs.query([RACE],{RACE:START},(30,10),historical=False)[0])

    def test_bad_date_does_not_open_the_sealed_year(self):
        r=('2026','0124','05','01','01','04')
        with self.assertRaises(MLFailure):m.current_odds_query([r],{r:'2026-01-24T11:00:00+09:00'})

    def test_grouping_records_real_source_without_rewriting_raw_payload(self):
        raw=dict(zip(RACE_KEY,RACE),minutes=10,record_type='O1',status='1',published='10241049',
                 sale='7',payload='opaque-exact-source',unsafe_version=False)
        rows=m.current_odds_grouped([raw])[(RACE,10)]
        self.assertEqual(rows[0]['source_table'],'jvd_o1');self.assertEqual(rows[0]['payload'],'opaque-exact-source')
        self.assertNotIn('first_seen_at',rows[0])

    def test_future_first_seen_still_rejected_not_backdated(self):
        row={'race_key':list(RACE),'status':'1','published':'10241049','source_table':'jvd_o1',
             'first_seen_at':'2026-10-24T10:50:01+09:00'}
        value,reason=inputs.select(RACE,[row],['01','02','03'],core.instant('2026-10-24T10:50:00+09:00'),300,require_seen=True)
        self.assertIsNone(value);self.assertEqual(reason,'no_on_time_observation')

    def test_last_publication_minute_must_end_before_cutoff(self):
        row={'race_key':list(RACE),'status':'1','published':'10241050','source_table':'jvd_o1',
             'first_seen_at':'2026-10-24T10:49:50+09:00'}
        value,reason=inputs.select(RACE,[row],['01','02','03'],core.instant('2026-10-24T10:50:00+09:00'),300,require_seen=True)
        self.assertIsNone(value);self.assertEqual(reason,'no_on_time_observation')
