from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
from jra_racing_ai.d39_next import display as m
from jra_racing_ai.d39_next import results_csv
from jra_racing_ai.win_sim.core import JST
import test_next_results_csv as fixture


def setup():
    p=fixture.packet();first=deepcopy(p['races'][0]);first.update(date=p['date'],venue='テスト',race_no=1,
        roster=['01','02'],scheduled_start='2026-10-25T09:40:00+09:00')
    first['models']={n:v for n,v in first['models'].items() if n in ('old_D39','new_D39','control_D39')}
    first['models']['H']={'status':'saved','rows':[{'horse_no':'01','rank':1,'probability':.8},{'horse_no':'02','rank':2,'probability':.2}]}
    second=deepcopy(first);second.update(race_id='2026102505010102',race_no=2,scheduled_start='2026-10-25T10:10:00+09:00')
    return p,[first,second]


class MissingDisplayTests(unittest.TestCase):
    def test_all_targets_remain_visible_without_changing_recorded_population(self):
        p,targets=setup();before=deepcopy(p)
        m.attach(p,targets,candidate='enriched_D39',clock=datetime(2026,10,25,17,tzinfo=JST))
        self.assertEqual(len(p['display_races']),2);self.assertEqual(len(p['display_result_races']),2)
        self.assertEqual(p['races'],before['races']);self.assertEqual(p['results'],before['results'])
        self.assertEqual(p['display_coverage']['selected_forecast_races'],1)
        self.assertEqual(p['display_coverage']['missing_forecast_races'],1)

    def test_missing_prediction_has_no_invented_probability_rank_or_result(self):
        p,targets=setup();p['reference']={'races':[{'models':{'enriched_D39':{'probability':1}}}]}
        m.attach(p,targets,candidate='enriched_D39',clock=datetime(2026,10,25,17,tzinfo=JST))
        r=p['display_races'][1]
        self.assertEqual(r['models']['enriched_D39']['rows'],[])
        self.assertEqual(r['candidate_status'],'missing')
        self.assertEqual(p['display_result_races'][1]['finishes'],{})
        self.assertEqual(p['display_result_races'][1]['roster'],['01','02'])

    def test_other_model_predictions_and_approved_results_are_retained(self):
        p,targets=setup();fallback=deepcopy(p)
        second=deepcopy(fallback['races'][0]);second['race_id']=targets[1]['race_id'];fallback['races'].append(second)
        answer=deepcopy(fallback['results']['races'][0]);answer['race_id']=second['race_id']
        answer['models'].pop('enriched_D39');fallback['results']['races'].append(answer)
        m.attach(p,targets,candidate='enriched_D39',fallback=fallback,clock=datetime(2026,10,25,17,tzinfo=JST))
        r=p['display_result_races'][1]
        self.assertEqual(r['models']['next_D39']['rows'],answer['models']['next_D39']['rows'])
        self.assertEqual(r['models']['enriched_D39']['status'],'missing')
        self.assertNotIn('H',r['models']);self.assertFalse(r['H_results_included'])

    def test_pre_cutoff_missing_record_is_waiting_not_failed(self):
        p,targets=setup();p['races']=[];p['results']=None
        m.attach(p,targets,candidate='enriched_D39',clock=datetime(2026,10,25,8,tzinfo=JST))
        self.assertEqual(p['display_coverage']['pending_forecast_races'],2)
        self.assertEqual(p['display_coverage']['missing_forecast_races'],0)

    def test_saved_missing_reason_is_preserved(self):
        p,targets=setup()
        with TemporaryDirectory() as temp:
            directory=Path(temp);folder=directory/'forward'/p['date']/'races'/targets[1]['race_id'];folder.mkdir(parents=True)
            (folder/'receipt.json').write_text(json.dumps({'status':'missing','reason':'model_not_active_before_cutoff'}))
            with patch.object(m.c,'read',side_effect=lambda path:json.loads(path.read_text())):
                m.attach(p,targets,candidate='enriched_D39',directory=directory,clock=datetime(2026,10,25,17,tzinfo=JST))
        self.assertEqual(p['display_races'][1]['candidate_reason'],'model_not_active_before_cutoff')

    def test_display_placeholders_do_not_change_result_csv_or_metrics(self):
        p,targets=setup();before=results_csv.build(p,'improved')
        m.attach(p,targets,candidate='enriched_D39',clock=datetime(2026,10,25,17,tzinfo=JST))
        after=results_csv.build(p,'improved')
        self.assertEqual(before,after)

    def test_different_forecast_source_is_not_joined_to_fallback_result(self):
        p,targets=setup();fallback=deepcopy(p);p['races']=[];p['results']=None
        targets[0]['source_prediction_sha256']='other'
        m.attach(p,targets,candidate='enriched_D39',fallback=fallback,clock=datetime(2026,10,25,17,tzinfo=JST))
        self.assertEqual(p['display_result_races'][0]['finishes'],{})


if __name__=='__main__':unittest.main()
