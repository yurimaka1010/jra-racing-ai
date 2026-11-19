"""Fictional fourth-model fits/records only. No private data or actual fitting."""
from contextlib import ExitStack
from datetime import datetime,date
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch,MagicMock

import numpy as np
import pandas as pd

from jra_racing_ai.d39_generation import contracts as c,control,learning,forward,outcomes
from jra_racing_ai.research import model as matrices
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.win_sim import core
from test_research_stage2 import source


class FourthD39Tests(unittest.TestCase):
    def plan(self,four=True):
        names=forward.FOUR_NAMES if four else forward.NAMES
        return {'models':list(names),'model_ids':{n:'fake_'+n for n in names},'temperature':{n:.75 if n!='H' else 1. for n in names},
            'references':{},'start_date':'2026-10-24','end_date':'2026-10-25'}

    def test_existing_three_model_names_and_output_path_remain_default(self):
        self.assertEqual(forward.NAMES,('old_D39','new_D39','H'))
        self.assertEqual(forward.model_names(self.plan(False)),forward.NAMES)
        self.assertEqual(forward.record_subdir(self.plan(False)),'forward')
        self.assertEqual(forward.record_subdir(self.plan()),'forward-four')

    def test_four_profile_never_silently_runs_only_three(self):
        plan=self.plan();plan['model_ids'].pop('control_D39')
        with self.assertRaises(MLFailure):forward.model_names(plan)
        plan=self.plan();plan['temperature'].pop('control_D39')
        with self.assertRaises(MLFailure):forward.model_names(plan)

    def test_control_folder_is_bounded_and_rejects_traversal(self):
        for identity in ('../other','bad','0'*31,'z'*32):
            with self.assertRaises(MLFailure):control.folder(Path('synthetic'),identity)

    def test_fitcounter_forbids_second_control_fit(self):
        counter=control.tuner.FitCounter(1,time.monotonic()+10,lambda *a:None)
        called=MagicMock(return_value=object());fit=counter.wrap(called)
        fit()
        with self.assertRaises(MLFailure):fit()
        self.assertEqual(called.call_count,1)

    def test_training_source_has_no_Tuner_call_or_temperature_selection(self):
        text=inspect.getsource(control.worker)
        self.assertNotIn('tuner.tune(',text);self.assertNotIn('.calibrate(',text)
        self.assertNotIn('early_stopping(',text)
        self.assertIn("num_boost_round=plan['rounds']",text)
        self.assertIn("data.load_part(parent,'final_train')",text)

    def test_frozen_old_parameters_rounds_and_temperature_are_rechecked_before_fit(self):
        parent=Path('fake_parent');directory=parent/'control'
        old={'parameters':{'feature_fraction':.4},'best_iteration':215,'temperature':.75}
        original={'version':control.VERSION,'parent_run_id':parent.name,'feature_spec':c.FEATURE_SPEC,
            'parameters':old['parameters'],'rounds':215,'temperature':.75,'references':{},
            'budget':{'actual_train_calls_max':1,'Tuner_fits':0}}
        authorization={'source':'human_user','max_fits':1,'plan_sha256':'hash'}
        commit={'control_id':directory.name,'plan_sha256':'hash','max_total_control_fits_for_parent':1}
        for changed in ({'rounds':216},{'temperature':1.},{'parameters':{'feature_fraction':1.}}):
            plan=dict(original,**changed)
            def read(path):
                if path==directory/'plan.json':return plan
                if path==directory/'authorization.json':return authorization
                if path==parent/'control_plan_commit.json':return commit
                return old
            with patch.object(c,'approved'),patch.object(c,'read',side_effect=read),patch.object(c,'sha',return_value='hash'):
                with self.assertRaises(MLFailure):control.approved(parent,directory)

    def test_repeated_execution_cannot_start_a_second_worker(self):
        with patch.object(control,'approved'),patch.object(control.tuner,'training_lock'), \
             patch.object(c,'write',side_effect=MLFailure('artifact_exists')),patch.object(control.subprocess,'Popen') as process:
            with self.assertRaises(MLFailure):control.execute(Path('fake_parent'),Path('fake_control'))
        process.assert_not_called()

    def test_control_worker_calls_train_once_with_frozen_parameters_and_temperature(self):
        import lightgbm as lgb
        plan={'parameters':{'objective':'binary','num_threads':4,'learning_rate':.05},'rounds':215,'temperature':.75,
            'feature_spec':c.FEATURE_SPEC,'features':c.FEATURE_SPEC['features'],'categories':{},'dataset_id':'fake_data',
            'final_train':{'sha256':'same_bytes','rows':3,'races':1,'end':'2026-09-20'},'versions':{},
            'source_old_model_id':'old_fake','source_new_model_id':'new_fake','spans':{}}
        frame=pd.DataFrame({'entry_key':['202609200501010101','202609200501010102','202609200501010103'],
                            'date':['2026-09-20']*3,'winner_label':[1,0,0]})
        booster=MagicMock();booster.current_iteration.return_value=215;booster.model_to_string.return_value='fake_tree'
        booster.predict.return_value=np.array([.4,.3,.3])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            parent=Path(temp);directory=parent/'control';directory.mkdir();(directory/'execution_started.json').write_text('{}')
            stack.enter_context(patch.object(control,'approved',return_value=plan))
            stack.enter_context(patch.object(c,'read',return_value={'run_id':directory.name}))
            stack.enter_context(patch.object(c,'sha',return_value='fake_hash'))
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')))
            stack.enter_context(patch.object(control.tuner,'versions',return_value={}))
            stack.enter_context(patch.object(control.tuner,'tune'))
            stack.enter_context(patch.object(control.data,'load_part',return_value=frame))
            stack.enter_context(patch.object(matrices,'schema',return_value={}))
            stack.enter_context(patch.object(matrices,'matrix',return_value=frame[['winner_label']]))
            stack.enter_context(patch.object(control.safety,'local_path',side_effect=lambda p,**kw:p))
            cpu=MagicMock();cpu.percent.return_value=0;cpu.observed=[0.]
            stack.enter_context(patch.object(control.prior,'CPU',return_value=cpu))
            stack.enter_context(patch.object(control,'verify',return_value={'tasks_unchanged':True}))
            stack.enter_context(patch.object(learning,'load',return_value=(booster,{})))
            stack.enter_context(patch.object(lgb,'Dataset'))
            train=stack.enter_context(patch.object(lgb,'train',return_value=booster))
            result=control.worker(parent,directory)
            self.assertEqual(train.call_count,1);self.assertEqual(train.call_args.args[0],plan['parameters'])
            self.assertEqual(train.call_args.kwargs['num_boost_round'],215)
            saved=json.loads((directory/'models'/result['model_id']/'model.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['temperature'],.75);self.assertEqual(saved['actual_train_calls'],1)
            self.assertEqual(saved['same_final_train_sha256'],'same_bytes');self.assertEqual(saved['Tuner_fits'],0)

    def test_four_realistic_fake_predictions_share_roster_cutoff_timestamp_and_model_ids(self):
        now=datetime(2026,10,24,9,50,5,tzinfo=core.JST);race=('2026','1024','05','01','01','01');plan=self.plan()
        reg={'race_key':list(race),'roster':['01','02'],'scheduled_start':'2026-10-24T10:00:00+09:00',
             'registered_at':'2026-10-24T09:00:00+09:00','reason':None}
        quote={'published_at':'2026-10-24T09:49:00+09:00','first_seen_at':'2026-10-24T09:49:20+09:00','odds':{'01':4.,'02':1.5}}
        frame=pd.DataFrame({'entry_key':[''.join(race)+'01',''.join(race)+'02'],'date':['2026-10-24']*2,
                            'distance':[1600.,1600.],'surface':['11','11']})
        entries=[{'horse_number':'01','horse_id':'FAKE1'},{'horse_number':'02','horse_id':'FAKE2'}]
        results={}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            stack.enter_context(patch.object(c,'write',side_effect=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')))
            stack.enter_context(patch.object(c,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()))
            stack.enter_context(patch.object(forward.inputs,'select',return_value=(quote,None)))
            stack.enter_context(patch.object(forward,'on_time_entries',return_value=({'payload':entries,'path':'fake','sha256':'mock','first_seen_at':'2026-10-24T09:49:40+09:00'},None)))
            stack.enter_context(patch.object(forward,'live_features',return_value=frame))
            stack.enter_context(patch.object(forward.tuner,'predict',return_value=np.array([.4,.6])))
            stack.enter_context(patch.object(learning,'predict',side_effect=lambda bundle,f:np.array([.7,.3]) if bundle=='control' else np.array([.5,.5])))
            stack.enter_context(patch.object(forward.inputs,'predict',return_value=({'H':{'01':.6,'02':.4}},None)))
            for four in (False,True):
                folder=Path(temp)/str(four);folder.mkdir()
                bundles={'old_D39':(None,None),'new_D39':'new','H':None,'control_D39':'control'}
                receipt=forward.decide(folder,reg,reg,self.plan(four),bundles,None,clock=lambda:now)
                self.assertEqual(receipt['status'],'saved')
                results[four]=json.loads((folder/'prediction.json').read_text(encoding='utf-8'))
                csv=pd.read_csv(folder/'predictions.csv',dtype={'entry_key':str})
                self.assertEqual(len(csv),8 if four else 6)
                self.assertEqual(set(csv.generated_at),{now.isoformat()})
                self.assertEqual(set(csv.input_cutoff),{'2026-10-24T09:50:00+09:00'})
                for name,part in csv.groupby('model'):
                    self.assertEqual(set(part.entry_key),set(frame.entry_key));self.assertAlmostEqual(part.probability.sum(),1.)
                    self.assertEqual(set(part.model_id),{self.plan(four)['model_ids'][name]})
            for name in forward.NAMES:
                self.assertEqual(results[False]['probabilities'][name],results[True]['probabilities'][name])
                self.assertEqual(results[False]['ledgers'][name],results[True]['ledgers'][name])

    def test_four_score_path_blocks_race_day_before_any_result_query(self):
        session=MagicMock()
        with patch.object(c,'approved'),patch.object(c,'read',return_value=self.plan()), \
             patch.object(c,'clock',return_value=datetime(2026,10,24,19,tzinfo=core.JST)):
            with self.assertRaises(MLFailure):outcomes.evaluate(Path('fake'),session,'2026-10-24',four=True)
        session.assert_not_called()

    def test_synthetic_verification_is_not_scoreable_as_a_live_forecast(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as temp,ExitStack() as stack:
            folder=Path(temp);(folder/'receipt.json').write_text('{}')
            record={'synthetic_verification':True,'input_cutoff':'2026-10-24T09:50:00+09:00'}
            receipt={'status':'saved','prediction_sha256':'fake','CSV_sha256':'fake','deadline':'2026-10-24T09:51:00+09:00'}
            stack.enter_context(patch.object(c,'sha',return_value='fake'))
            stack.enter_context(patch.object(c,'read',side_effect=lambda p:receipt if p.name=='receipt.json' else record))
            self.assertIsNone(outcomes.committed(folder,self.plan()))


if __name__=='__main__':unittest.main()
