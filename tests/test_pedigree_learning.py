"""Synthetic/mocked only: the two real fits are reserved for the approved study."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai.pedigree_research import learning as l, contracts as c
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.research.model import normalize


class PedigreeLearningTests(unittest.TestCase):
    def fixture(self):
        p={'version':l.VERSION,'rounds':219,'parameters':{'num_threads':4,'seed':42},
           'budget':{'actual_train_calls_max':2,'per_model_actual_train_calls_max':1,'CPU_threads':4,
                     'per_fit_seconds_max':300,'whole_learning_seconds_max':900,'private_bytes_max':12*1024**3,
                     'automatic_retry':False,'Optuna_or_Tuner_trials':0},
           'configurations':[{'name':n,'feature_spec':l.model_spec(n),'fits_max':1} for n in l.NAMES],
           'production_connected':False,'promotion_authorized':False,
           'future_acceptance_criteria':'deferred_pending_user_decision','source_preparation_id':l.SOURCE_ID,
           'references':{},'reference47_model_path':'C:/test/model','versions':{'lightgbm':'test'},
           'categories':{},'calibration':{'candidates':[.75,1,1.25,2]}}
        a={'source':'human_user','allow_real_fits':True,'training_plan_sha256':'sha',
           'actual_train_calls_max':2,'per_model_actual_train_calls_max':1}
        return p,a

    def validate(self,p,a):
        def read(path):
            if path.name=='training_plan.json':return p
            if path.name=='authorization.json':return a
            if path.name=='learning_plan.proposed.json':return {'calibration':p['calibration']}
            return {'parameters':p['parameters'],'categories':p['categories']}
        with patch.object(c,'read',side_effect=read),patch.object(c,'sha',return_value='sha'), \
             patch.object(l.budget,'versions',return_value=p['versions']):
            return l.approved(Path('C:/test'))

    def test_fixed_configurations(self):
        self.assertEqual(len(l.model_spec('control39')['features']),39)
        self.assertEqual(len(l.model_spec('pedigree45')['features']),45)
        self.assertEqual(l.model_spec('pedigree45')['features'][:39],l.model_spec('control39')['features'])
        with self.assertRaises(MLFailure):l.model_spec('another_model')

    def test_authorized_exact_plan(self):
        p,a=self.fixture();self.assertEqual(self.validate(p,a),p)

    def test_unapproved_fits_refused(self):
        p,a=self.fixture();a['allow_real_fits']=False
        with self.assertRaises(MLFailure):self.validate(p,a)

    def test_third_fit_or_changed_rounds_refused(self):
        for change in ('limit','rounds','threads','search','retry','memory','deadline'):
            p,a=self.fixture()
            if change=='limit':p['budget']['actual_train_calls_max']=3
            if change=='rounds':p['rounds']=220
            if change=='threads':p['parameters']['num_threads']=8
            if change=='search':p['budget']['Optuna_or_Tuner_trials']=1
            if change=='retry':p['budget']['automatic_retry']=True
            if change=='memory':p['budget']['private_bytes_max']+=1
            if change=='deadline':p['budget']['whole_learning_seconds_max']+=1
            with self.subTest(change=change),self.assertRaises(MLFailure):self.validate(p,a)

    def test_promotion_and_final_evaluation_not_authorized(self):
        for field,value in (('production_connected',True),('promotion_authorized',True),
                            ('future_acceptance_criteria','approved')):
            p,a=self.fixture();p[field]=value
            with self.subTest(field=field),self.assertRaises(MLFailure):self.validate(p,a)

    def test_changed_column_order_refused(self):
        p,a=self.fixture();p['configurations'][0]['feature_spec']['features'].reverse()
        with self.assertRaises(MLFailure):self.validate(p,a)

    def test_predict_uses_saved_temperature_once(self):
        import pandas as pd
        import numpy as np
        class Booster:
            def predict(self,x,**kwargs):return np.array([.2,.8])
        frame=pd.DataFrame({'entry_key':['202401010501010101','202401010501010102']})
        bundle=(Booster(),{'feature_spec':l.model_spec('control39'),'categories':{},'temperature':2})
        with patch.object(l.data,'matrix',return_value=frame):
            raw=l.predict(bundle,frame,raw=True);p=l.predict(bundle,frame)
        np.testing.assert_array_equal(raw,[.2,.8])
        np.testing.assert_allclose(p,normalize(frame,raw,2))
        self.assertAlmostEqual(p.sum(),1)

    def test_reload_checks_tree_integrity_without_training(self):
        meta={'configuration':'control39','version':l.VERSION,'feature_spec':l.model_spec('control39'),
              'features':l.model_spec('control39')['features'],'best_iteration':219,'temperature':1.25,
              'actual_train_calls':1,'production_connected':False,'tree_sha256':'expected'}
        with patch.object(l.safety,'local_path',return_value=Path('C:/model')), \
             patch.object(c,'read',return_value=meta),patch.object(c,'sha',return_value='different'):
            with self.assertRaises(MLFailure):l.load(Path('C:/model'))

    def test_common_matrix_hash_sensitive_to_order_and_values(self):
        import pandas as pd
        a=pd.DataFrame({'a':[1.,2.],'b':[3.,4.]})
        self.assertNotEqual(l.fingerprint(a),l.fingerprint(a[['b','a']]))
        b=a.copy();b.loc[0,'a']=1.1
        self.assertNotEqual(l.fingerprint(a),l.fingerprint(b))


if __name__=='__main__':unittest.main()
