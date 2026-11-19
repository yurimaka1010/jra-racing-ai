"""Fictional scores, O1 rows and mocks only; no real database or network."""
from datetime import date,datetime,timedelta
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import math,unittest
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
from jra_racing_ai.ml import safety
from jra_racing_ai.stage5 import contracts as c,metrics,place,experiment,store,candidate,model
from jra_racing_ai.win_sim import core
from jra_racing_ai.db_connection import validate_audit_sql
from jra_racing_ai import stage1,stage1_analysis


RACE=('2025','0301','05','01','01','01');ROSTER=('01','02');START=datetime(2025,3,1,15,tzinfo=core.JST)


def small():
    frame=[];pred=[]
    for j,(prob,d) in enumerate(((.51,date(2025,3,1)),(.1,date(2025,3,8)))):
        race=(str(d.year),d.strftime('%m%d'),'05','01','01','01')
        pred.append(core.PredictionRace(race,{'01':prob,'02':1-prob}))
        for h in ('01','02'):frame.append(dict(entry_key=''.join(race)+h,date=d.isoformat(),winner_label=int(h=='01')))
    return pd.DataFrame(frame),pred


def winrow(clock='03011449'):
    return dict(record_type='O1',status='1',published=clock,sale='7',payload='0100200102004002'+' '*8*26,
        place_sale='7',pay_key='3',place_payload='010012001801020018003202'+' '*12*26)


def packet():
    sc=core.Schedule(RACE,START,None,ROSTER,True,'reviewed_historical_schedule')
    row=winrow();q,why=stage1_analysis.select_quote(RACE,[dict(row,source_table='apd_sokuho_o1',oversized=False)],ROSTER,sc,10,300)
    assert why is None
    return {'scheduled_start':START.isoformat(),'quotes':{'10':stage1._quote_document(q)}}


class AUCTests(unittest.TestCase):
    def test_perfect_reversed_and_all_ties(self):
        y=np.array([0,0,1,1]);self.assertEqual(metrics.auc(y,[.1,.2,.8,.9]),1.)
        self.assertEqual(metrics.auc(y,[.9,.8,.2,.1]),0.);self.assertEqual(metrics.auc(y,[.5]*4),.5)

    def test_one_class_invalid_not_filled_with_half(self):
        for y,p in (([0,0],[.1,.2]),([1,1],[.3,.4]),([0,1],[.2,np.nan])):
            with self.assertRaises(safety.MLFailure):metrics.auc(y,p)

    def test_hand_computed_global_and_within_race_AUC(self):
        f,pred=small();y,p,rs,weeks=metrics.assemble(f,pred);point,_=metrics.WeekBootstrap(weeks,repetitions=10).measure(y,p,rs)
        self.assertEqual(point['roc_auc'],.25);self.assertEqual(point['within_race_auc'],.5);self.assertEqual(point['top1_accuracy_percent'],50.)

    def test_tie_within_race_half_and_fractional_top1(self):
        f,pred=small();pred=[core.PredictionRace(p.race,{'01':.5,'02':.5}) for p in pred]
        y,p,rs,weeks=metrics.assemble(f,pred);point,_=metrics.WeekBootstrap(weeks,repetitions=10).measure(y,p,rs)
        self.assertEqual(point['roc_auc'],.5);self.assertEqual(point['within_race_auc'],.5);self.assertEqual(point['top1_accuracy_percent'],50.)

    def test_week_multiplicity_equals_expansion_and_sklearn_weights(self):
        f,pred=small();y,p,rs,weeks=metrics.assemble(f,pred);b=metrics.WeekBootstrap(weeks,sample_indices=[[0,1],[0,0],[1,1]])
        _,draw=b.measure(y,p,rs)
        for i,counts in enumerate(b.counts):
            weights=counts[b.codes];expanded=np.repeat(np.arange(len(y)),weights.astype(int))
            self.assertAlmostEqual(draw['roc_auc'][i],metrics.auc(y,p,weights))
            self.assertAlmostEqual(draw['roc_auc'][i],metrics.auc(y[expanded],p[expanded]))
        np.testing.assert_allclose(draw['roc_auc'],[.25,1.,0.])

    def test_same_scores_have_zero_paired_differences(self):
        f,pred=small();y,p,rs,weeks=metrics.assemble(f,pred);b=metrics.WeekBootstrap(weeks,repetitions=50)
        a,draw=b.measure(y,p,rs);v=metrics.compare(a,draw,a,draw)
        for name in metrics.NAMES:self.assertEqual(v[name]['difference_new_minus_H'],0.);self.assertEqual(v[name]['95_percent_ci'],[0.,0.])

    def test_shuffled_rows_join_by_entry_key_not_position(self):
        f,pred=small();q=f.sample(frac=1,random_state=3);pred2=list(reversed(pred))
        y,p,r,w=metrics.assemble(f,pred);a,_=metrics.WeekBootstrap(w,repetitions=10).measure(y,p,r)
        y,p,r,w=metrics.assemble(q,pred2);b,_=metrics.WeekBootstrap(w,repetitions=10).measure(y,p,r)
        self.assertEqual(a,b)

    def test_duplicate_wrong_horse_and_multiple_positive_rejected(self):
        f,pred=small()
        with self.assertRaises(safety.MLFailure):metrics.assemble(pd.concat([f,f.iloc[:1]]),pred)
        wrong=[core.PredictionRace(p.race,{'01':.51,'03':.49}) for p in pred]
        with self.assertRaises(safety.MLFailure):metrics.assemble(f,wrong)
        f.loc[:1,'winner_label']=1
        with self.assertRaises(safety.MLFailure):metrics.assemble(f,pred)

    def test_invalid_score_sums_not_silently_normalized(self):
        f,p=small();p[0]=core.PredictionRace(p[0].race,{'01':.7,'02':.7})
        with self.assertRaises(safety.MLFailure):metrics.assemble(f,p)

    def test_saved_final_score_not_transformed_again(self):
        f,pred=small();expected=np.array([.51,.49,.1,.9]);y,p,_,_=metrics.assemble(f,pred)
        np.testing.assert_array_equal(p,expected)

    def test_2026_rejected_before_any_metric_processing(self):
        f,pred=small();f.date='2026-01-01'
        with self.assertRaises(safety.MLFailure):metrics.assemble(f,pred)

    def test_model_split_endpoints_strictly_prior(self):
        for year in c.YEARS:experiment.timing(year,c.split(year))
        sp=c.split(2023);sp['calibration_end']='2023-01-01'
        with self.assertRaises(safety.MLFailure):experiment.timing(2023,sp)

    def test_label_cohorts_must_match_even_with_same_key_set(self):
        f,_=small();metrics.same_labels(f,f.iloc[::-1])
        bad=f.copy();bad.winner_label=1-bad.winner_label
        with self.assertRaises(safety.MLFailure):metrics.same_labels(f,bad)


class PlaceTests(unittest.TestCase):
    def test_2_and_3_payees_explicit_same_price_formula_no_probability(self):
        base=None
        for key in ('2','3'):
            row=winrow();row['pay_key']=key;v,why,pay=place.decode(RACE,[row],ROSTER,packet())
            self.assertIsNone(why);self.assertEqual(pay,key)
            self.assertAlmostEqual(v['01']['place_relative_support'],math.log(2)-(math.log(1.2)+math.log(1.8))/2)
            self.assertAlmostEqual(v['01']['place_log_range'],math.log(1.8)-math.log(1.2))
            if base is not None:self.assertEqual(v,base)
            base=v

    def test_unknown_pay_key_and_no_sale_stay_missing(self):
        for key in (None,'0','1','X'):
            row=winrow();row['pay_key']=key;self.assertIsNone(place.decode(RACE,[row],ROSTER,packet())[0])
        row=winrow();row.update(place_sale='0',pay_key='0')
        self.assertEqual(place.decode(RACE,[row],ROSTER,packet())[1],'place_not_sold')

    def test_payload_initial_special_width_and_order_are_rejected(self):
        for payload in (None,'',' '*336,'x'*336,winrow()['place_payload'][:-1],
                        '010000001801'+winrow()['place_payload'][12:],
                        '010020001001'+winrow()['place_payload'][12:],
                        '010999099901'+winrow()['place_payload'][12:]):
            self.assertIsNone(place.numeric(payload,ROSTER)[0])

    def test_future_same_minute_stale_and_final_category(self):
        for change in ({'published':'03011450'},{'published':'03011451'},{'published':'03011444'},{'status':'5'}):
            row=dict(winrow(),**change);self.assertIsNone(place.decode(RACE,[row],ROSTER,packet())[0])

    def test_changed_win_record_and_duplicate_not_used(self):
        row=winrow();row['payload']='0100300102004002'+' '*8*26
        self.assertEqual(place.decode(RACE,[row],ROSTER,packet())[1],'saved_T10_win_record_changed')
        self.assertEqual(place.decode(RACE,[winrow(),winrow()],ROSTER,packet())[1],'same_O1_duplicate')

    def test_any_runner_missing_marks_whole_race_without_dropping(self):
        f=pd.DataFrame([dict(entry_key=''.join(RACE)+h,date='2025-03-01',market_probability=p,**{n:0. for n in c.H_FEATURES if n!='market_probability'})
            for h,p in zip(ROSTER,(.6,.4))])
        bad=winrow();bad['place_payload']=bad['place_payload'][:12]+' '*12+bad['place_payload'][24:]
        raw={RACE:{'rows':[bad],'read_error':None}}
        with patch.object(place.old_data,'market',return_value={RACE:packet()}),patch.object(place,'inputs',return_value=raw):
            augmented,stats=place.augment(Path('unused'),f)
        self.assertEqual(len(augmented),len(f));self.assertEqual(stats['complete_races'],0)
        self.assertTrue((augmented.place_missing==1).all());self.assertTrue(augmented.place_relative_support.isna().all())
        pd.testing.assert_frame_equal(f,augmented[list(f.columns)])

    def test_query_uses_exact_saved_timestamp_and_is_read_only(self):
        sql,args=place.query([RACE],{RACE:packet()});validate_audit_sql(sql)
        self.assertIn('t.happyo_tsukihi_jifun::text=w.published',sql);self.assertEqual(args[-1],'03011449')
        with self.assertRaises(safety.MLFailure):place.query([RACE]*33,{RACE:packet()})

    def test_2026_place_query_rejected_before_SQL(self):
        race=('2026',*RACE[1:])
        with self.assertRaises(safety.MLFailure):place.query([race],{race:packet()})


class CandidateTests(unittest.TestCase):
    def example(self):
        base={'roc_auc':.83,'within_race_auc':.82,'top1_accuracy_percent':33.,'race_log_loss':1.94,'brier_score':.058}
        new=dict(base,roc_auc=.833,within_race_auc=.821,top1_accuracy_percent=33.1,race_log_loss=1.93,brier_score=.057)
        compare={'roc_auc':{'95_percent_ci':[.001,.004]}}
        annual={str(y):{'roc_auc':{'difference_new_minus_H':.003}} for y in c.YEARS}
        return new,base,compare,annual

    def test_AUC_improvement_distinct_from_overall_rejection(self):
        n,b,ci,years=self.example();n['brier_score']=.059
        result=candidate.decide(n,b,ci,years)
        self.assertFalse(result['retain_development_candidate']);self.assertIn('AUC_point_estimate_improved',result['reason'])
        self.assertIn('Brier_at_most_H',result['failed_criteria'])

    def test_minimum_delta_CI_and_annual_veto_not_relaxed(self):
        for mode in ('small_delta','CI_zero','annual_veto'):
            n,b,ci,years=self.example()
            if mode=='small_delta':n['roc_auc']=.8319
            if mode=='CI_zero':ci['roc_auc']['95_percent_ci'][0]=0.
            if mode=='annual_veto':years['2023']['roc_auc']['difference_new_minus_H']=-.002
            self.assertFalse(candidate.decide(n,b,ci,years)['retain_development_candidate'])

    def test_passes_only_all_criteria_and_highest_model_is_descriptive(self):
        n,b,ci,years=self.example();result=candidate.decide(n,b,ci,years,highest_existing={'model':'other','roc_auc':.9})
        self.assertTrue(result['retain_development_candidate']);self.assertFalse(result['exceeds_existing_highest_AUC'])
        self.assertTrue(result['H_is_fixed_formal_comparator'])

    def test_zero_residual_reproduces_market_and_input_excludes_labels(self):
        f,_=small()
        for n in c.FEATURES:f[n]=0.
        f.market_probability=[.6,.4,.2,.8]
        booster=MagicMock();booster.predict.return_value=np.zeros(len(f));bundle={'boosters':{'win':booster}}
        np.testing.assert_allclose(model.predict(bundle,f),f.market_probability,atol=1e-15)
        self.assertNotIn('winner_label',booster.predict.call_args.args[0].columns)
        self.assertTrue(booster.predict.call_args.kwargs['raw_score'])

    def test_fit_gate_denial_prevents_learning(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch('lightgbm.train') as train:
            store.write(Path(td),'place_gate.json',{'fit_allowed':False})
            with self.assertRaises(safety.MLFailure):model.fit(Path(td),2023,[],{})
            train.assert_not_called()

    def test_model_roundtrip_preserves_offset_and_old_loader_stays_strict(self):
        from jra_racing_ai.research import contracts as oldc,model as oldmodel
        def fake(d,days):
            rows=[]
            for j in range(days):
                day=d+timedelta(days=j)
                for i,p in enumerate((.6,.4),1):
                    row={f:float(i+j*.01) for f in c.FEATURES};row.update(entry_key=f'{day:%Y%m%d}05010101{i:02d}',date=day.isoformat(),winner_label=int(i==1),market_probability=p)
                    rows.append(row)
            return pd.DataFrame(rows)
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            directory=Path(td)
            for name,obj in (('place_gate.json',{'fit_allowed':True}),('training_plan.json',{}),('place_mapping.json',{})):store.write(directory,name,obj)
            parts=[fake(date(2022,1,1),40),fake(date(2022,9,8),4),fake(date(2022,11,8),4)]
            bundle,meta=model.fit(directory,2023,parts,oldc.plan()['anchor_parameters'])
            after=model.load(directory/'models'/meta['model_id'])
            np.testing.assert_array_equal(model.predict(bundle,parts[1]),model.predict(after,parts[1]))
            with patch('lightgbm.train',side_effect=AssertionError('must_not_refit')):model.fit(directory,2023,parts,{})
            with self.assertRaises(safety.MLFailure):oldmodel.load(directory/'models'/meta['model_id'],root=directory)


if __name__=='__main__':unittest.main()
