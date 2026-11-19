"""Mocked official tuner routing and fabricated inputs; zero real-data/real LightGBM fits."""
from pathlib import Path
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from jra_racing_ai.ml import safety
from jra_racing_ai.research import contracts as features, model as matrix_model
from jra_racing_ai.tuner_pilot import experiment as t


class FitAccountingTests(unittest.TestCase):
    def test_actual_train_call_is_reserved_before_call(self):
        records = []
        def original(*args, **kwargs):
            self.assertEqual(records[0][:2], (1,'started'))
            self.assertEqual(len(kwargs['callbacks']),1)
            return 'booster'
        counter = t.FitCounter(2,time.monotonic()+60,lambda *v:records.append(v))
        self.assertEqual(counter.wrap(original)({},object()),'booster')
        self.assertEqual(counter.calls,1); self.assertEqual(records[-1][:2],(1,'complete'))
        self.assertGreaterEqual(counter.training_seconds,0)
        self.assertEqual(counter.training_seconds,records[-1][2]['actual_train_call_seconds'])

    def test_failed_actual_train_consumes_budget(self):
        records = []; original = MagicMock(side_effect=RuntimeError('synthetic failure'))
        counter = t.FitCounter(1,time.monotonic()+60,lambda *v:records.append(v))
        with self.assertRaises(RuntimeError):counter.wrap(original)()
        with self.assertRaises(safety.MLFailure):counter.wrap(original)()
        self.assertEqual(counter.calls,1);original.assert_called_once()
        self.assertEqual([v[1] for v in records],['started','failed'])

    def test_absolute_cap_counts_calls_not_trial_ids(self):
        counter = t.FitCounter(8,time.monotonic()+60,lambda *v:None)
        original = MagicMock(return_value='mock')
        call=counter.wrap(original)
        for _ in range(8):call()
        with self.assertRaises(safety.MLFailure):call()
        self.assertEqual(counter.calls,8);self.assertEqual(original.call_count,8)

    def test_deadline_prevents_new_actual_fit(self):
        original=MagicMock();counter=t.FitCounter(8,time.monotonic()-1,lambda *v:None)
        with self.assertRaises(safety.MLFailure):counter.wrap(original)()
        self.assertEqual(counter.calls,0);original.assert_not_called()

    def test_round_callback_stops_deadline_and_records_failed_attempt(self):
        records=[];counter=t.FitCounter(1,time.monotonic()+60,lambda *v:records.append(v))
        def original(*args,**kwargs):
            counter.deadline=time.monotonic()-1
            kwargs['callbacks'][-1](object())
        with self.assertRaises(safety.MLFailure):counter.wrap(original)()
        self.assertEqual(counter.calls,1);self.assertEqual(records[-1][1],'failed')

    def test_official_API_calls_are_all_counted_without_real_fit(self):
        import lightgbm as lgb
        from optuna_integration.lightgbm import LightGBMTuner
        data=lgb.Dataset(np.arange(40).reshape(20,2),label=np.arange(20)%2)
        validation=lgb.Dataset(np.arange(20).reshape(10,2),label=np.arange(10)%2,reference=data)
        booster=MagicMock();booster.best_score={'valid_0':{'binary_logloss':.5}}
        booster.current_iteration.return_value=1
        counter=t.FitCounter(2,time.monotonic()+30,lambda *v:None)
        with patch.object(lgb,'train',return_value=booster) as actual, \
                patch.object(LightGBMTuner,'run',side_effect=AssertionError('unbounded_run_forbidden')):
            best,params,trials=t.tune(data,validation,{'objective':'binary','metric':'binary_logloss',
               'verbosity':-1,'num_threads':1,'feature_pre_filter':False},counter=counter,
               steps=(('tune_feature_fraction',2),),rounds=5,seconds=20)
        self.assertIs(best,booster);self.assertEqual(counter.calls,2)
        self.assertEqual(actual.call_count,2);self.assertEqual(len(trials),2)

    def test_unknown_or_expanded_step_rejected_without_training(self):
        import lightgbm as lgb
        dataset=lgb.Dataset(np.arange(20).reshape(10,2),label=np.arange(10)%2)
        counter=t.FitCounter(8,time.monotonic()+60,lambda *v:None)
        with patch.object(lgb,'train') as actual,self.assertRaises(safety.MLFailure):
            t.tune(dataset,dataset,{'objective':'binary','metric':'binary_logloss','verbosity':-1},
               counter=counter,steps=(('tune_feature_fraction',99),),rounds=5)
        actual.assert_not_called();self.assertEqual(counter.calls,0)


class ApprovalAndCompatibilityTests(unittest.TestCase):
    def test_api_compatibility_is_zero_fit(self):
        import lightgbm as lgb
        with patch.object(lgb,'train',side_effect=AssertionError('fit_forbidden')):
            result=t.compatibility()
        self.assertEqual(result['real_data_fits'],0)
        self.assertTrue(all(result['public_methods_support_n_trials'].values()))

    def test_execute_without_approval_stops_before_lock_or_data(self):
        with patch.object(t,'approved',side_effect=safety.MLFailure('explicit_execution_required')), \
                patch.object(t,'training_lock') as lock,patch.object(t.cache,'load_part') as load:
            with self.assertRaises(safety.MLFailure):t.execute(Path('fabricated'))
        lock.assert_not_called();load.assert_not_called()

    def test_worker_without_approval_stops_before_data_or_fit(self):
        with patch.object(t,'approved',side_effect=safety.MLFailure('explicit_execution_required')), \
                patch.object(t.cache,'load_part') as load,patch.object(t,'tune') as tune:
            with self.assertRaises(safety.MLFailure):t.worker(Path('fabricated'))
        load.assert_not_called();tune.assert_not_called()

    def test_plan_hash_mismatch_blocks_execution(self):
        with patch.object(t,'read',side_effect=[{'version':t.VERSION},{'plan_sha256':'before'}]), \
                patch.object(t.safety,'digest',return_value='after'):
            with self.assertRaises(safety.MLFailure):t.approved(Path('fabricated'))

    def test_invalid_approval_cannot_enable_fits(self):
        for approval in ({},{'allow_real_fits':True},{'allow_real_fits':True,'source':'human_user','plan_sha256':'ok','actual_train_calls_max':9}):
            with self.subTest(approval=approval),patch.object(Path,'is_file',return_value=True), \
                    patch.object(t,'read',side_effect=[{'version':t.VERSION},{'plan_sha256':'ok'},approval]), \
                    patch.object(t.safety,'digest',return_value='ok'):
                with self.assertRaises(safety.MLFailure):t.approved(Path('fabricated'))

    def test_identifier_rejects_escape(self):
        for name in ('..','A'*32,'a'*32+'/child',None):
            with self.subTest(name=name),self.assertRaises(safety.MLFailure):t.run_folder(name)

    def test_fixed_error_does_not_disclose_records_or_credentials(self):
        with patch.object(t,'compatibility',side_effect=RuntimeError('PASSWORD_AND_PRIVATE_RECORD')),patch('builtins.print') as emit:
            self.assertEqual(t.main(['compatibility']),1)
        message=emit.call_args.args[0]
        self.assertIn('internal_error',message);self.assertNotIn('PASSWORD',message)


class FeatureAndPersistenceTests(unittest.TestCase):
    def test_results_and_payoffs_never_enter_fixed_39_columns(self):
        spec=features.spec(('A','B','C','D'))
        frame=pd.DataFrame([{n:None for n in features.ALL_FEATURES} for _ in range(2)])
        frame['entry_key']=['202501050501010101','202501050501010102']
        categories=matrix_model.schema(frame,spec); before=matrix_model.matrix(frame,spec,categories)
        frame['winner_label']=[1,0];frame['finish_position']=[1,2];frame['final_odds']=[2.,99.]
        pd.testing.assert_frame_equal(before,matrix_model.matrix(frame,spec,categories))
        self.assertEqual(len(before.columns),39)

    def test_prediction_uses_saved_temperature_once(self):
        spec=features.spec(('A','B','C','D'))
        f=pd.DataFrame([{n:None for n in features.ALL_FEATURES} for _ in range(2)])
        f['entry_key']=['202501050501010101','202501050501010102']
        booster=MagicMock();booster.predict.return_value=np.array([.8,.2])
        metadata={'feature_spec':spec,'categories':matrix_model.schema(f,spec),'temperature':2.}
        np.testing.assert_allclose(t.predict(booster,metadata,f),[2/3,1/3])

    def test_model_read_rejects_wrong_format_before_tree_loading(self):
        with patch.object(t,'read',return_value={'version':'wrong'}), \
                patch.object(t.safety,'local_path',return_value=Path('fabricated')):
            with self.assertRaises(safety.MLFailure):t.load_model(Path('fabricated'))


if __name__=='__main__':unittest.main()
