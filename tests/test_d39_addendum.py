"""Fictional zero-fit/raw/fallback tests; no private data or real DB."""
from contextlib import ExitStack
from datetime import datetime,date,timedelta
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
from jra_racing_ai.d39_generation import addendum,contracts as c,forward as legacy,four_forward as f,four_final,four_sources,four_dry
from jra_racing_ai.research import model as matrices
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim import core


class AddendumTests(unittest.TestCase):
    def p(self):
        return {'model_ids':{n:'fake_'+n for n in legacy.FOUR_NAMES},'temperatures':{'old_D39':.75,'new_D39':1.,'control_D39':.75,'H':1.},
                'references':{},'model_paths':{n:'fake/'+n for n in legacy.FOUR_NAMES},'start_date':'2026-10-24','end_date':'2027-03-14',
                'availability_deadline':'2026-10-24T07:00:00+09:00'}

    def test_fixed_twenty_one_monday_slots(self):
        start=date(2026,10,24);monday=start-timedelta(days=start.weekday());end=monday+timedelta(weeks=21,days=-1)
        self.assertEqual(str(end),'2027-03-14');self.assertEqual((end-monday).days,146)

    def test_interim_evaluation_never_opens_record_or_DB(self):
        session=MagicMock()
        for day in (datetime(2026,10,26,tzinfo=core.JST),datetime(2027,3,14,23,59,tzinfo=core.JST)):
            with patch.object(addendum,'approved',return_value=self.p()),patch.object(c,'clock',return_value=day),patch.object(four_final,'eligible') as record:
                with self.assertRaises(MLFailure):four_final.run(Path('fake'),session)
            record.assert_not_called()
        session.assert_not_called()

    def test_window_opens_only_after_last_day(self):
        four_final.require_closed(self.p(),datetime(2027,3,15,tzinfo=core.JST))

    def test_three_signed_contrasts_and_Bonferroni_primary_intervals(self):
        points={n:{m:i*.1 for m in four_final.metrics.NAMES} for i,n in enumerate(legacy.FOUR_NAMES)}
        draws={n:{m:np.full(20,v) for m,v in p.items()} for n,p in points.items()}
        values=four_final.contrasts(points,draws,1-.05/3)
        self.assertEqual(set(values),{'new_D39_minus_control_D39','control_D39_minus_old_D39','new_D39_minus_old_D39'})
        self.assertAlmostEqual(values['new_D39_minus_control_D39']['metrics']['race_log_loss']['difference'],-.1)
        self.assertAlmostEqual(values['control_D39_minus_old_D39']['metrics']['race_log_loss']['difference'],.2)
        self.assertEqual(values['new_D39_minus_old_D39']['metrics']['race_log_loss']['confidence'],1-.05/3)

    def test_only_late_control_availability_cannot_override_deadline(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp:
            parent=Path(temp);(parent/'availability-four').mkdir()
            values=[{'observed_at':'2026-10-24T06:59:00+09:00','control_available':False},
                    {'observed_at':'2026-10-24T07:01:00+09:00','control_available':True}]
            for i,v in enumerate(values):(parent/'availability-four'/f'{i}.json').write_text(json.dumps(v))
            with patch.object(c,'read',side_effect=lambda p:json.loads(p.read_text())):
                value=f.startup_mode(parent,self.p(),True,clock=lambda:datetime(2026,10,24,8,tzinfo=core.JST))
            self.assertEqual(value['mode'],'three_fallback');self.assertFalse(value['control_enabled'])

    def test_current_control_failure_automatically_selects_three(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp:
            parent=Path(temp);(parent/'availability-four').mkdir();(parent/'availability-four/a.json').write_text('{}')
            with patch.object(c,'read',return_value={'observed_at':'2026-10-23T20:00:00+09:00','control_available':True}):
                self.assertEqual(f.startup_mode(parent,self.p(),False,clock=lambda:datetime(2026,10,24,8,tzinfo=core.JST))['mode'],'three_fallback')
                self.assertEqual(f.startup_mode(parent,self.p(),True,clock=lambda:datetime(2026,10,24,8,tzinfo=core.JST))['mode'],'four')

    def test_dry_run_rejects_calibration_and_past_result_dates(self):
        session=MagicMock()
        with patch.object(addendum,'approved',return_value=self.p()),patch.object(c,'clock',return_value=datetime(2026,10,23,tzinfo=core.JST)):
            for day in ('2026-10-17','2026-10-18','2026-09-30'):
                with self.assertRaises(MLFailure):four_dry.run(Path('fake'),session,day)
        session.assert_not_called()

    def test_declared_entry_query_excludes_results_odds_and_payouts(self):
        schema={t:{v[1]:{} for v in c.features.RAW_FIELDS.values() if v[0]==t} for t in ('ra','se')}
        sql,args=four_sources.entry_query(schema,date(2026,10,24))
        for name in ('kakutei_chakujun','kohan_3f','chakusa_code_1','haraimodoshi','odds_tansho'):
            self.assertNotIn(name,sql)
        self.assertEqual(args,('2026','1024'));self.assertIn('LIMIT 3001',sql)

    def test_no_fit_in_diagnostics_raw_save_or_dry_run(self):
        for function in (addendum.diagnostics,addendum.verify_raw,f.sidecar,four_dry.run):
            text=inspect.getsource(function)
            self.assertNotIn('lgb.train(',text);self.assertNotIn('tuner.tune(',text);self.assertNotIn('.save_model(',text)

    def test_control_dry_prediction_failure_keeps_old_and_new_outputs(self):
        now=datetime(2026,10,24,8,tzinfo=core.JST);race=('2026','1024','05','01','01','01')
        card={'reason':None,'roster':['01','02'],'scheduled_start':'2026-10-24T10:00:00+09:00'}
        rows=[{'horse_number':'01'},{'horse_number':'02'}]
        frame=pd.DataFrame({'entry_key':[''.join(race)+h for h in card['roster']]})
        def raw(name,*args):
            if name=='control_D39':raise RuntimeError('SECRET_DO_NOT_LOG')
            return {'01':.3,'02':.7}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            parent=Path(temp)
            def mkdir(p):p.mkdir(parents=True,exist_ok=True);return p
            stack.enter_context(patch.object(addendum,'approved',return_value=self.p()))
            stack.enter_context(patch.object(c,'clock',return_value=now))
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v))))
            stack.enter_context(patch.object(legacy,'mkdir',side_effect=mkdir))
            stack.enter_context(patch.object(legacy.existing,'live_cards',return_value={race:card}))
            stack.enter_context(patch.object(four_sources,'entries',return_value=({race:rows},now)))
            stack.enter_context(patch.object(legacy,'live_features',return_value=frame))
            stack.enter_context(patch.object(f,'raw_predict',side_effect=raw))
            result=four_dry.run(parent,MagicMock(),'2026-10-24',bundles={n:object() for n in legacy.FOUR_NAMES},history=object())
            self.assertEqual(result['model_race_status_counts'],{'old_D39_saved':1,'new_D39_saved':1,'control_D39_missing':1})
            self.assertEqual(result['rows_saved'],4);self.assertFalse(result['results_read'])
            self.assertNotIn('SECRET_DO_NOT_LOG',(Path(result['folder'])/'summary.json').read_text())

    def test_history_startup_failure_logs_fixed_code_and_releases_lock(self):
        now=datetime(2026,10,24,8,tzinfo=core.JST)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            parent=Path(temp)
            def mkdir(p):p.mkdir(parents=True,exist_ok=True);return p
            stack.enter_context(patch.object(f,'preflight',return_value={'status':'ready'}))
            stack.enter_context(patch.object(addendum,'approved',return_value=self.p()))
            stack.enter_context(patch.object(c,'clock',return_value=now))
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v))))
            stack.enter_context(patch.object(legacy,'mkdir',side_effect=mkdir))
            stack.enter_context(patch.object(f,'load_bundles',return_value=({'control_D39':object()},{})))
            stack.enter_context(patch.object(f,'startup_mode',return_value={'control_enabled':True,'mode':'four'}))
            stack.enter_context(patch.object(four_sources,'history',side_effect=RuntimeError('SECRET_DO_NOT_LOG')))
            with self.assertRaises(MLFailure) as caught:f.run(parent,MagicMock(),once=True)
            self.assertEqual(caught.exception.code,'internal_error')
            self.assertFalse((parent/'four-addendum-recorder.lock').exists())
            logs=list((parent/addendum.OUTPUT/'2026-10-24/logs').glob('*.json'))
            self.assertEqual(len(logs),1);self.assertNotIn('SECRET_DO_NOT_LOG',logs[0].read_text())
            self.assertEqual(json.loads(logs[0].read_text())['reason'],'internal_error')

    def exercise(self,fail_control=False,late=False):
        now=datetime(2026,10,24,9,50,5,tzinfo=core.JST);race=('2026','1024','05','01','01','01');plan=self.p()
        reg={'race_key':list(race),'roster':['01','02'],'scheduled_start':'2026-10-24T10:00:00+09:00',
             'registered_at':'2026-10-24T09:00:00+09:00','reason':None}
        quote={'published_at':'2026-10-24T09:49:00+09:00','first_seen_at':'2026-10-24T09:49:20+09:00','odds':{'01':4.,'02':1.5}}
        frame=pd.DataFrame({'entry_key':[''.join(race)+'01',''.join(race)+'02'],'date':['2026-10-24']*2,'distance':[1600.,1600.],'surface':['11','11']})
        raw={'old_D39':{'01':.2,'02':.1},'new_D39':{'01':.4,'02':.3},'H':{'01':.3,'02':.7},'control_D39':{'01':.1,'02':.3}}
        ps={n:matrices.normalize(frame,np.array(list(r.values())),plan['temperatures'][n]) for n,r in raw.items()}
        baseplan={'model_ids':{n:plan['model_ids'][n] for n in legacy.NAMES},'temperature':{n:plan['temperatures'][n] for n in legacy.NAMES},
                  'references':{},'start_date':plan['start_date'],'end_date':plan['end_date']}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            parent=Path(temp);(parent/'forward_plan.json').write_text(json.dumps(baseplan))
            entry_path=parent/'entry.json';entries=[{'horse_number':'01','horse_id':'FAKE1'},{'horse_number':'02','horse_id':'FAKE2'}]
            entry_path.write_text(json.dumps({'payload':entries,'first_seen_at':'2026-10-24T09:49:40+09:00'}))
            stack.enter_context(patch.object(c,'read',side_effect=lambda p:json.loads(p.read_text(encoding='utf-8'))))
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')))
            stack.enter_context(patch.object(c,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()))
            stack.enter_context(patch.object(legacy.inputs,'select',return_value=(quote,None)))
            stack.enter_context(patch.object(legacy,'on_time_entries',return_value=({'payload':entries,'path':str(entry_path),
                'sha256':hashlib.sha256(entry_path.read_bytes()).hexdigest(),'first_seen_at':'2026-10-24T09:49:40+09:00'},None)))
            stack.enter_context(patch.object(legacy,'live_features',return_value=frame))
            stack.enter_context(patch.object(legacy.tuner,'predict',return_value=ps['old_D39']))
            stack.enter_context(patch.object(legacy.learning,'predict',return_value=ps['new_D39']))
            stack.enter_context(patch.object(legacy.inputs,'predict',return_value=({'H':dict(zip(['01','02'],ps['H']))},None)))
            before=parent/'before';after=parent/'after';before.mkdir();after.mkdir();bundles={'old_D39':(None,None),'new_D39':None,'H':None,'control_D39':None}
            for folder in (before,after):legacy.decide(folder,reg,reg,baseplan,bundles,None,clock=lambda:now)
            original={n:(before/n).read_bytes() for n in ('predictions.csv','prediction.json','receipt.json')}
            def predict(name,*args):
                if name=='control_D39' and fail_control:raise RuntimeError('SECRET_DO_NOT_LOG')
                return raw[name]
            stack.enter_context(patch.object(f,'raw_predict',side_effect=predict))
            result=f.sidecar(after,parent,plan,{**bundles,'control_D39':object()},None,
                             clock=lambda:now+timedelta(seconds=60) if late else now)
            for n,v in original.items():self.assertEqual(v,(after/n).read_bytes())
            if late:self.assertEqual(result['status'],'missing');return
            rows=pd.read_csv(after/'four_predictions.csv');saved=json.loads((after/'four_prediction.json').read_text())
            self.assertEqual(len(rows),8)
            if fail_control:
                bad=rows.loc[rows.model=='control_D39'];self.assertTrue(bad.probability.isna().all());self.assertTrue(bad.raw_probability.isna().all())
                self.assertEqual(set(bad.missing_reason),{'control_prediction_failed'});self.assertEqual(result['control_status'],'missing')
                self.assertNotIn('SECRET_DO_NOT_LOG',(after/'four_prediction.json').read_text())
            else:
                self.assertFalse(rows.raw_probability.isna().any())
                for name,part in rows.groupby('model'):
                    q=part.raw_probability.to_numpy(float)**(1/plan['temperatures'][name]);q/=q.sum()
                    np.testing.assert_allclose(q,part.probability,rtol=0,atol=1e-12)
            for name in legacy.NAMES:self.assertEqual(saved['probabilities'][name],json.loads(original['prediction.json'])['probabilities'][name])

    def test_raw_save_preserves_all_three_model_files_byte_for_byte(self):self.exercise()
    def test_control_failure_records_missing_and_keeps_other_predictions(self):self.exercise(fail_control=True)
    def test_sidecar_deadline_miss_cannot_change_or_veto_base_predictions(self):self.exercise(late=True)
