"""Synthetic receipts only: no real DB or forecast generation."""
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
import json
import unittest
import numpy as np

from jra_racing_ai.d39_next import contracts as c, forward as f, features
from jra_racing_ai import keiba_lab_live as lab
from jra_racing_ai.win_sim import core
from test_research_stage2 import source
from test_keiba_lab_live import fixture


class NewForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / 'parent'; self.parent.mkdir()
        self.patch = patch.object(lab, 'BASE', self.root); self.patch.start(); self.addCleanup(self.patch.stop)
        def read(path):
            return json.loads(Path(path).read_text(encoding='utf-8'))
        def write(path, value):
            with path.open('x', encoding='utf-8') as stream: json.dump(value, stream)
        import hashlib
        self.read = patch.object(c, 'read', side_effect=read); self.read.start(); self.addCleanup(self.read.stop)
        self.write = patch.object(c, 'write', side_effect=write); self.write.start(); self.addCleanup(self.write.stop)
        self.sha = patch.object(c, 'sha', side_effect=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
        self.sha.start(); self.addCleanup(self.sha.stop)

    def test_existing_default_result_projection_rejects_H_even_if_explicitly_requested(self):
        with self.assertRaisesRegex(c.safety.MLFailure, 'invalid_input'):
            lab.result_projection({}, [], None, names=('H', 'next_D39'))
        with self.assertRaisesRegex(c.safety.MLFailure, 'invalid_input'):
            lab.fetch_results([], MagicMock(), names=('H',))

    def test_saved_original_3_projection_unchanged_by_new_names_support(self):
        folder, plan = fixture(self.parent)
        roster = ['01', '02', '03']
        forecast = lab.load_forecast(folder, plan, datetime(2026,10,24,9,40,tzinfo=core.JST))
        rows = [{'horse':h,'horse_id':'horse-'+h,'status':'7','anomaly':'0','rank':f'{i:02d}'} for i,h in enumerate(roster,1)]
        a = lab.result_projection(forecast, rows, None)
        b = lab.result_projection(forecast, rows, None, names=lab.D39)
        self.assertEqual(a, b)
        self.assertNotIn('H', a['models'])
        forecast['models']['next_D39'] = deepcopy(forecast['models']['new_D39'])
        out = lab.result_projection(forecast, rows, None, names=f.MODELS)
        self.assertIn('next_D39', out['models']); self.assertNotIn('H', out['models'])

    def test_activation_after_cutoff_cannot_backfill(self):
        folder = self.root / 'case'; folder.mkdir()
        (folder/'registered.json').write_text(json.dumps({'scheduled_start':'2026-10-25T09:45:00+09:00'}))
        output = self.root / 'out'
        result = f.decide(folder, output, {'activated_at':'2026-10-25T09:35:01+09:00'}, None, None,
                          clock=lambda:datetime(2026,10,25,9,35,2,tzinfo=core.JST))
        self.assertEqual(result['reason'], 'model_not_active_before_cutoff')
        self.assertFalse((output/'prediction.json').exists())

    def test_after_deadline_cannot_generate_formal_forecast(self):
        folder = self.root / 'case'; folder.mkdir()
        (folder/'registered.json').write_text(json.dumps({'scheduled_start':'2026-10-25T09:45:00+09:00'}))
        result = f.decide(folder, self.root/'out', {'activated_at':'2026-10-25T08:00:00+09:00'}, None, None,
                          clock=lambda:datetime(2026,10,25,9,36,1,tzinfo=core.JST))
        self.assertEqual(result['reason'], 'save_deadline_missed')

    def test_before_cutoff_does_not_generate_or_read_results(self):
        folder = self.root / 'case'; folder.mkdir()
        (folder/'registered.json').write_text(json.dumps({'scheduled_start':'2026-10-25T09:45:00+09:00'}))
        with patch.object(f.learning, 'predict') as predict:
            value = f.decide(folder, self.root/'out', {}, None, None,
                clock=lambda:datetime(2026,10,25,8,0,tzinfo=core.JST))
        self.assertEqual(value, {'status':'pending'}); predict.assert_not_called()

    def test_missing_original_snapshot_remains_pending(self):
        folder = self.root / 'case'; folder.mkdir()
        (folder/'registered.json').write_text(json.dumps({'scheduled_start':'2026-10-25T09:45:00+09:00'}))
        value = f.decide(folder, self.root/'out', {'activated_at':'2026-10-25T08:00:00+09:00'}, None, None,
            clock=lambda:datetime(2026,10,25,9,35,2,tzinfo=core.JST))
        self.assertEqual(value, {'status':'pending'})

    def test_live_features_share_clean_builder_and_strip_current_labels(self):
        past = source(date(2024,1,1)); target = source(date(2024,4,15))
        builder = features.Builder(); builder.transform_day(past); builder._available(date(2024,4,15))
        a = f.live_frame(builder, target)
        changed = deepcopy(target)
        for row in changed: row.update(finish_position='00', anomaly='7', final_odds='1000')
        b = f.live_frame(builder, changed)
        self.assertTrue(a.equals(b))
        expected = deepcopy(builder).transform_day(target, prediction=True)[0]
        for name in c.spec()['features']:
            for x, y in zip(a[name], [r[name] for r in expected]):
                import pandas as pd
                self.assertTrue((pd.isna(x) and y is None) or x == y, name)

    def test_new_model_gets_exactly_one_temperature_application(self):
        import pandas as pd
        frame = pd.DataFrame({'entry_key':['202610250501010101','202610250501010102']})
        meta = {'feature_spec':c.spec(),'categories':{},'temperature':2.}
        booster = MagicMock();booster.predict.return_value=np.array([.81,.04])
        with patch.object(f.learning.data,'matrix',return_value='fake'):
            raw = f.learning.predict((booster,meta),frame,raw=True)
            p = f.learning.predict((booster,meta),frame)
        np.testing.assert_array_equal(raw,[.81,.04])
        np.testing.assert_allclose(p,[.9/1.1,.2/1.1],rtol=0,atol=1e-15)

    def test_dry_run_gets_schedule_from_card_not_declared_entry_rows(self):
        from contextlib import nullcontext
        import pandas as pd
        day=date(2026,10,25);raw=source(day)
        for row in raw: row.pop('start_time',None)
        key=tuple(raw[0][k] for k in f.RACE_KEY)
        card={key:{'scheduled_start':'2026-10-25T15:00:00+09:00','reason':None,'roster':['01','02','03']}}
        builder=features.Builder()
        with patch.object(c,'clock',return_value=datetime(2026,10,25,8,0,tzinfo=core.JST)), \
             patch.object(f.existing.existing,'live_cards',return_value=card), \
             patch('jra_racing_ai.d39_generation.four_sources.entries',return_value=({key:raw},c.clock())), \
             patch.object(f.learning,'predict',return_value=np.array([.6,.3,.1])):
            result=f.dry_run(self.root,(None,{'model_id':'fake','temperature':1.}),builder,lambda:nullcontext(MagicMock()))
        self.assertEqual(result['races'],1);self.assertEqual(result['runners'],3)
        self.assertFalse(result['results_read']);self.assertFalse(result['real_forward_record'])

    def test_on_time_sidecar_saves_original_inputs_and_never_edits_base(self):
        folder, _ = fixture(self.parent)
        reg = json.loads((folder/'registered.json').read_text())
        reg['registered_at'] = '2026-10-24T08:00:00+09:00'
        (folder/'registered.json').write_text(json.dumps(reg))
        original = json.loads((folder/'prediction.json').read_text())
        entry = Path(original['entry_snapshot']['path'])
        target = source(date(2026,10,24))
        entry.write_text(json.dumps({'payload': target}))
        original['entry_snapshot']['sha256'] = c.sha(entry)
        original['quote_T10']['odds'] = {'01':2.,'02':5.,'03':10.}
        (folder/'prediction.json').write_text(json.dumps(original))
        receipt = json.loads((folder/'receipt.json').read_text())
        receipt['prediction_sha256'] = c.sha(folder/'prediction.json')
        (folder/'receipt.json').write_text(json.dumps(receipt))
        before = {p.name:c.sha(p) for p in folder.glob('*') if p.is_file()}
        builder = features.Builder(); builder.transform_day(source(date(2026,1,1)));builder._available(date(2026,10,24))
        output = self.root/'out'
        meta = {'model_id':'fake-next','temperature':1.25}
        with patch.object(f.learning,'predict',return_value=np.array([.6,.3,.1])):
            result=f.decide(folder,output,{'activated_at':'2026-10-24T08:00:00+09:00'},(None,meta),builder,
                clock=lambda:datetime(2026,10,24,9,35,40,tzinfo=core.JST))
        self.assertEqual(result['status'],'saved')
        value=json.loads((output/'prediction.json').read_text())
        self.assertFalse(value['result_accessed'])
        self.assertEqual(value['source_prediction_sha256'],receipt['prediction_sha256'])
        self.assertAlmostEqual(sum(value['probabilities'].values()),1.)
        self.assertEqual(before,{p.name:c.sha(p) for p in folder.glob('*') if p.is_file()})
        self.assertTrue((output/'predictions.csv').is_file())

    def test_result_query_projection_never_contains_H_and_computes_four_D39_scores(self):
        folder, plan = fixture(self.parent)
        clock=datetime(2026,10,24,10,0,tzinfo=core.JST)
        base=lab.load_forecast(folder,plan,clock)
        base['models']['next_D39']=deepcopy(base['models']['new_D39'])
        base['d39_tickets']['next_D39']={'horse':'01','reason':'ev_selected'}
        base.update(next_distance=1600,next_surface='11',decision_odds={'01':2.,'02':5.,'03':10.})
        official=[{'horse':h,'horse_id':'horse-'+h,'status':'7','anomaly':'0','rank':f'{i:02}'} for i,h in enumerate(base['roster'],1)]
        def answers(requests,session,*,names):
            self.assertEqual(names,f.MODELS)
            self.assertNotIn('H',requests[0]['models'])
            projection=lab.result_projection(requests[0],official,None,names=names)
            return {base['race_id']:projection},[]
        with patch.object(f,'saved',return_value=[base]),patch.object(c,'clock',return_value=clock), \
             patch.object(lab,'fetch_results',side_effect=answers):
            result=f.evaluate(self.root,session=MagicMock())
        self.assertEqual(result['common_result_races'],1)
        value=json.loads((self.root/'latest_results.json').read_text())
        self.assertEqual(set(value['metrics']),set(f.MODELS))
        self.assertTrue(value['condition_diagnostics'])
        self.assertFalse(value['H_results_included'])
        self.assertEqual(value['financial_common_forecast_races'],1)

    def test_reference_predictions_stay_separate_from_formal_saved_list(self):
        import csv
        folder, plan=fixture(self.parent)
        clock=datetime(2026,10,24,10,0,tzinfo=core.JST)
        base=lab.load_forecast(folder,plan,clock)
        run=self.root/'run';run.mkdir();dry=run/'dry'/'fake';dry.mkdir(parents=True)
        active={'run_id':'a'*32,'model_id':'fake-next','temperature':1.}
        (self.root/'current.json').write_text(json.dumps(active))
        csv_path=dry/'predictions.dry.csv'
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['entry_key','real_forward_record','model_id','probability','rank','generated_at'])
            writer.writeheader()
            writer.writerows({'entry_key':base['race_id']+h,'real_forward_record':False,'model_id':'fake-next',
                'probability':v,'rank':i,'generated_at':'2026-10-24T08:00:00+09:00'}
                for i,(h,v) in enumerate(zip(base['roster'],[.6,.3,.1]),1))
        (dry/'summary.json').write_text(json.dumps({'date':'2026-10-24','races':1,
            'generated_at':'2026-10-24T08:00:00+09:00','CSV_sha256':c.sha(csv_path)}))
        viewer=MagicMock();viewer.forecasts.return_value={'races':[base],'coverage':{}}
        base['horse_names']={'01':'Fictional One','02':'Fictional Two','03':'Fictional Three'}
        original_hash=c.sha(csv_path)
        with patch.object(c,'ROOT',self.root),patch.object(c,'folder',return_value=run), \
             patch.object(c,'clock',return_value=clock),patch.object(f,'saved',return_value=[base]):
            result=f.public_packet(viewer,'2026-10-24')
        self.assertEqual(len(result['races']),1)
        self.assertEqual(len(result['reference']['races']),1)
        self.assertFalse(result['reference']['real_forward_record'])
        self.assertNotIn('horse_ids',result['races'][0])
        self.assertEqual(result['reference']['races'][0]['horse_names'],base['horse_names'])
        self.assertEqual(c.sha(csv_path),original_hash)

    def test_formal_display_name_requires_matching_recorded_horse_identity(self):
        record={'roster':['01','02'],'horse_ids':{'01':'horse-A','02':'horse-B'}}
        card={'horse_names':{'01':'Fictional A','02':'Fictional B'}}
        cache={'01':{'horse_id':'horse-A','name':'Fictional A'},
               '02':{'horse_id':'different-horse','name':'Fictional B'}}
        self.assertEqual(f.display_names(record,card,cache),{'01':'Fictional A'})

    def test_display_name_does_not_invent_missing_metadata(self):
        self.assertEqual(f.display_names({'roster':['01']},{'horse_names':{}},{}),{})


if __name__ == '__main__':
    unittest.main()
