"""Fictional rosters and one tiny synthetic fit. No DB, network, or private data."""
from datetime import date,timedelta
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
from jra_racing_ai.ml import safety
from jra_racing_ai.stage5 import metrics
from jra_racing_ai.stage6 import contracts as c,data,model,store as s,experiment as e
from jra_racing_ai.research import model as old_model


def frame(values=(1.,0.,2/3),day=date(2023,1,7)):
    base=[];styles=[];q=np.arange(len(values),0,-1,dtype=float);q/=q.sum()
    for i,(style,p) in enumerate(zip(values,q),1):
        key=f'{day:%Y%m%d}05010101{i:02d}'
        row={n:0. for n in c.H_FEATURES};row.update(entry_key=key,date=str(day),winner_label=int(i==1),market_probability=p)
        base.append(row);styles.append({'entry_key':key,'date':str(day),c.SOURCE_COLUMN:style})
    return pd.DataFrame(base),pd.DataFrame(styles)


def initialized(directory):
    s.write(directory,'inputs.json',{})
    s.write(directory,'plan.json',dict(c.plan(),inputs_sha256=s.sha(directory/'inputs.json')))
    s.inventory(directory,{'plan_sha256':s.sha(directory/'plan.json'),'fits_started':0,'fits_max':6,'records':[]})


class CompositionTests(unittest.TestCase):
    def test_known_three_horses_exclude_self_and_preserve_source(self):
        f,st=frame();before=f.copy(deep=True);out=data.augment(f,st)
        np.testing.assert_allclose(out.others_front_fraction,[.5,1.,.5])
        np.testing.assert_allclose(out.others_style_mean,[1/3,5/6,.5])
        np.testing.assert_allclose(out.style_pressure_interaction,[.25,-.5,1/12])
        np.testing.assert_array_equal(out.others_style_known_fraction,[1,1,1])
        pd.testing.assert_frame_equal(f,before)

    def test_two_horses_and_exact_two_thirds_threshold(self):
        f,st=frame((2/3,0.));out=data.augment(f,st)
        np.testing.assert_array_equal(out.others_front_fraction,[0.,1.])
        np.testing.assert_allclose(out.others_style_mean,[0.,2/3])

    def test_unknown_others_still_use_total_roster_denominator(self):
        f,st=frame((1.,1.,np.nan,0.));out=data.augment(f,st)
        self.assertEqual(out.others_front_fraction.iloc[0],1/3)
        self.assertEqual(out.others_style_known_fraction.iloc[0],2/3)
        self.assertEqual(out.others_style_mean.iloc[0],.5)
        self.assertEqual(out.others_front_fraction.iloc[2],2/3)
        self.assertTrue(np.isnan(out.style_pressure_interaction.iloc[2]));self.assertEqual(out.style_missing.iloc[2],1.)
        self.assertEqual(len(out),4)

    def test_all_others_missing_nan_means_and_zero_known_fraction(self):
        f,st=frame((1.,np.nan,np.nan));out=data.augment(f,st)
        self.assertTrue(np.isnan(out.others_front_fraction.iloc[0]));self.assertTrue(np.isnan(out.others_style_mean.iloc[0]))
        self.assertTrue(np.isnan(out.style_pressure_interaction.iloc[0]));self.assertEqual(out.others_style_known_fraction.iloc[0],0.)
        self.assertEqual(out.others_front_fraction.iloc[1],.5)

    def test_all_missing_rows_not_removed(self):
        f,st=frame((np.nan,np.nan,np.nan));out=data.augment(f,st)
        self.assertEqual(len(out),3);self.assertTrue(out.style_missing.eq(1.).all())
        self.assertTrue(out.others_style_known_fraction.eq(0.).all());self.assertTrue(out.style_pressure_interaction.isna().all())

    def test_horse_order_does_not_change_features(self):
        f,st=frame((1.,.7,np.nan,0.))
        a=data.augment(f,st).set_index('entry_key').sort_index()
        b=data.augment(f.sample(frac=1,random_state=7),st.iloc[::-1]).set_index('entry_key').sort_index()
        pd.testing.assert_frame_equal(a,b)

    def test_current_results_final_odds_and_final_headcount_are_ignored(self):
        f,st=frame();a=data.augment(f,st)
        f['winner_label']=1-f.winner_label;f['corner_4']=99;f['actual_runners']=99;f['final_odds']=9999;f['soha_time']='9999'
        b=data.augment(f,st)
        pd.testing.assert_frame_equal(a[list(c.STYLE+c.PACE)],b[list(c.STYLE+c.PACE)])
        self.assertFalse(set(('winner_label','actual_runners','final_odds','corner_4'))&set(model.matrix(b,'H_pace')))

    def test_nonmissing_invalid_values_rejected_without_clipping(self):
        for value in (-.01,1.01,np.inf,-np.inf,'bad'):
            with self.subTest(value=value):
                f,st=frame((value,0.))
                with self.assertRaises(safety.MLFailure):data.augment(f,st)

    def test_duplicate_keys_on_either_side_rejected(self):
        f,st=frame()
        for left,right in ((pd.concat([f,f.iloc[:1]]),st),(f,pd.concat([st,st.iloc[:1]]))):
            with self.assertRaises(safety.MLFailure):data.augment(left,right)

    def test_date_and_horse_roster_mismatch_rejected(self):
        f,st=frame();bad=st.copy();bad.loc[0,'date']='2023-01-08'
        with self.assertRaises(safety.MLFailure):data.augment(f,bad)
        bad=st.copy();bad.loc[0,'entry_key']=bad.entry_key.iloc[0][:-2]+'04'
        with self.assertRaises(safety.MLFailure):data.augment(f,bad)
        with self.assertRaises(safety.MLFailure):data.augment(f.iloc[:1],st.iloc[:1])

    def test_style_column_must_exist_not_silently_all_missing(self):
        f,st=frame()
        with self.assertRaises(safety.MLFailure):data.augment(f,st.drop(columns=c.SOURCE_COLUMN))

    def test_reader_stops_before_post_boundary_value_and_ignores_other_columns(self):
        with TemporaryDirectory() as td,patch.object(c,'S2',Path(td)),patch.object(data.old_store,'check_reference'):
            folder=Path(td)/'features';folder.mkdir();p=folder/'2025.csv'
            p.write_text('entry_key,date,c_previous_corner4_percentile,finish_position\n'
                '202509270501010101,2025-09-27,0.75,NOT_PARSED\n'
                '202510040501010101,2025-10-04,NOT_PARSED,NOT_PARSED\n',encoding='utf-8')
            out=data.read_style(date(2025,9,1),date(2025,9,30));self.assertEqual(len(out),1)
            self.assertEqual(out[c.SOURCE_COLUMN].iloc[0],.75);self.assertNotIn('finish_position',out)

    def test_forbidden_period_refused_before_file_or_market_read(self):
        with patch.object(data.old,'market_frame') as reader:
            for start,end in ((date(2026,1,1),date(2026,9,30)),(date(2025,9,1),date(2025,10,1)),(date(2019,1,1),date(2020,1,1))):
                with self.assertRaises(safety.MLFailure):data.load(start,end)
            reader.assert_not_called()

    def test_training_variation_gate_rejects_constant_or_all_missing(self):
        f,st=frame((np.nan,np.nan,np.nan));self.assertFalse(data.training_gate(data.augment(f,st)))
        f,st=frame((1.,1.,1.));self.assertFalse(data.training_gate(data.augment(f,st)))
        f,st=frame();self.assertTrue(data.training_gate(data.augment(f,st)))


class ModelAndBudgetTests(unittest.TestCase):
    def test_fixed_feature_orders_offset_and_six_unit_budget(self):
        self.assertEqual(len(c.CONFIGS['H_style']),10);self.assertEqual(len(c.CONFIGS['H_pace']),14)
        self.assertEqual(c.CONFIGS['H_pace'][:10],c.CONFIGS['H_style'])
        self.assertEqual(c.plan()['fits_max'],6);self.assertEqual(c.plan()['boosting_rounds_max'],1800)
        self.assertEqual(c.PARAMETERS['metric'],'binary_logloss');self.assertFalse(c.PARAMETERS['boost_from_average'])

    def test_zero_residual_reproduces_market_and_uses_no_label(self):
        f,st=frame();f=data.augment(f,st);booster=MagicMock();booster.predict.return_value=np.zeros(len(f))
        p=model.predict({'metadata':{'config':'H_pace'},'boosters':{'win':booster}},f)
        np.testing.assert_allclose(p,f.market_probability,atol=1e-15);self.assertAlmostEqual(p.sum(),1.)
        self.assertTrue(booster.predict.call_args.kwargs['raw_score']);self.assertNotIn('winner_label',booster.predict.call_args.args[0])

    def test_all_2026_inference_rejected_before_booster(self):
        f,st=frame(day=date(2026,10,18));booster=MagicMock()
        with self.assertRaises(safety.MLFailure):model.predict({'metadata':{'config':'H_style'},'boosters':{'win':booster}},f)
        booster.predict.assert_not_called()

    def test_exclusive_reservations_six_units_and_no_repeat(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(c,'BASE',Path(td).parent):
            d=Path(td);initialized(d)
            for config in c.CONFIGS:
                for year in c.YEARS:self.assertTrue(model.reserve(d,config,year)[1])
            before=s.inventory(d);self.assertEqual(before['fits_started'],6)
            self.assertFalse(model.reserve(d,'H_pace',2025)[1]);self.assertEqual(before,s.inventory(d))
            invalid=deepcopy(before);invalid['records'].append(invalid['records'][0]);invalid['fits_started']=7
            with self.assertRaises(safety.MLFailure):s.validate_inventory(invalid)

    def test_started_without_complete_model_is_not_retrained(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(c,'BASE',Path(td).parent),patch('lightgbm.train') as train:
            d=Path(td);initialized(d);model.reserve(d,'H_style',2023)
            parts={name:data.augment(*frame(day=date.fromisoformat(c.split(2023)[name+'_start']))) for name in ('train','validation','calibration')}
            parts['evaluation']=data.augment(*frame())
            with self.assertRaises(safety.MLFailure):model.fit(d,'H_style',2023,parts)
            train.assert_not_called();self.assertEqual(s.inventory(d)['records'][0]['status'],'failed')
            self.assertEqual(s.inventory(d)['fits_started'],1)

    def test_plan_mutation_refused(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(c,'BASE',Path(td).parent):
            d=Path(td);initialized(d);p=d/'plan.json';value=s.read(d,'plan.json');value['rounds_max']=301
            p.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(safety.MLFailure):s.verify_plan(d)

    def test_one_synthetic_fit_serialization_reload_and_resume(self):
        def fake(day,days):
            return pd.concat([data.augment(*frame((1.,0.,.5),day+timedelta(days=i))) for i in range(days)],ignore_index=True)
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(c,'BASE',Path(td).parent):
            d=Path(td);initialized(d);parts={'train':fake(date(2022,1,1),120),'validation':fake(date(2022,9,8),5),
                'calibration':fake(date(2022,11,8),3),'evaluation':fake(date(2023,1,7),3)}
            bundle,row,p=model.fit(d,'H_pace',2023,parts);self.assertEqual(s.inventory(d)['fits_started'],1)
            loaded=model.load(d/'models'/row['model_id']);np.testing.assert_array_equal(p,model.predict(loaded,parts['evaluation']))
            with patch('lightgbm.train',side_effect=AssertionError('must_not_refit')):model.fit(d,'H_pace',2023,parts)
            self.assertEqual(s.inventory(d)['fits_started'],1)
            with self.assertRaises(safety.MLFailure):old_model.load(d/'models'/row['model_id'],root=d)
            path=d/'models'/row['model_id']/'model.json';meta=json.loads(path.read_text(encoding='utf-8'));meta['features'].reverse()
            path.write_text(json.dumps(meta),encoding='utf-8')
            with self.assertRaises(safety.MLFailure):model.load(path.parent)


class DecisionTests(unittest.TestCase):
    def example(self):
        h={'within_race_auc':.8,'roc_auc':.83,'race_log_loss':1.9,'horse_log_loss':.2,'brier_score':.06,'top1_accuracy_percent':33.}
        p=dict(h,within_race_auc=.802,roc_auc=.831,race_log_loss=1.89,brier_score=.059,top1_accuracy_percent=34.)
        comparisons={name:{'within_race_auc':{'difference_new_minus_comparator':delta,'97_5_percent_ci':[.0001,.003]}}
            for name,delta in (('A',.002),('B',.001))}
        annual={str(y):{'comparisons':deepcopy(comparisons)} for y in c.YEARS}
        return {'H':h,'H_pace':p},comparisons,annual

    def test_all_boundary_effect_thresholds_pass_without_rounding(self):
        points,comp,annual=self.example();result=e.decide(points,comp,annual,consistency=True)
        self.assertTrue(result['retain_development_candidate']);self.assertTrue(result['H_style_control_only'])
        self.assertFalse(result['production_adoption']);self.assertTrue(result['H_remains_primary'])

    def test_each_practical_or_CI_boundary_independently_vetoes(self):
        for label,mode in (('A','delta'),('B','delta'),('A','CI'),('B','CI')):
            points,comp,annual=self.example();item=comp[label]['within_race_auc']
            if mode=='delta':item['difference_new_minus_comparator']=np.nextafter(item['difference_new_minus_comparator'],-np.inf)
            else:item['97_5_percent_ci'][0]=0.
            self.assertFalse(e.decide(points,comp,annual,consistency=True)['retain_development_candidate'])

    def test_annual_veto_and_two_positive_years_are_both_required(self):
        for mode in ('veto','one_positive'):
            points,comp,annual=self.example();annual['2023']['comparisons']['A']['within_race_auc']['difference_new_minus_comparator']=-.002 if mode=='veto' else 0.
            if mode=='one_positive':annual['2024']['comparisons']['A']['within_race_auc']['difference_new_minus_comparator']=0.
            self.assertFalse(e.decide(points,comp,annual,consistency=True)['retain_development_candidate'])

    def test_each_secondary_quality_and_consistency_failure_vetoes(self):
        for key,direction in (('roc_auc',-1),('top1_accuracy_percent',-1),('race_log_loss',1),('brier_score',1),('consistency',0)):
            points,comp,annual=self.example()
            if key!='consistency':points['H_pace'][key]=points['H'][key]+direction*2e-12
            self.assertFalse(e.decide(points,comp,annual,consistency=key!='consistency')['retain_development_candidate'])

    def test_paired_975_interval_uses_specified_quantiles(self):
        np.testing.assert_allclose(e.bounds(np.arange(100),.975),[1.2375,97.7625])
        self.assertGreater(e.bounds(np.arange(100),.95)[0],e.bounds(np.arange(100),.975)[0])

    def test_existing_ties_and_equal_race_weights_preserved(self):
        f1,_=frame((.5,.5));f2,_=frame((.1,.2,.3,.4),date(2023,1,14));f=pd.concat([f1,f2],ignore_index=True)
        pred=e.exp.predictions(f,[.5,.5,.1,.2,.3,.4]);point=metrics.point_only(f,pred)
        self.assertEqual(point['within_race_auc'],.25);self.assertEqual(point['top1_accuracy_percent'],25.)


if __name__=='__main__':unittest.main()
