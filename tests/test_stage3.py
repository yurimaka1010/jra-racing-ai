"""Fictional inputs only: OOF chronology, integration calculus and paper money."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from datetime import date,datetime,timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json,math,unittest
from unittest.mock import MagicMock,patch
import numpy as np
import pandas as pd
from jra_racing_ai.ml import safety,model as legacy
from jra_racing_ai.research import contracts as oldc,model as oldm
from jra_racing_ai.stage3 import contracts as c,store as s,softmax,scoring,oof,reuse,supplement as e,finance,report,cli,integrate
from jra_racing_ai.win_sim import core


def frame(start=date(2022,1,1),days=4):
    rows=[]
    for j in range(days):
        day=start+timedelta(days=j)
        for i,(q,a) in enumerate(zip((.6,.3,.1),(.5,.2,.3)),1):
            row={k:np.nan for k in oldc.ALL_FEATURES}
            row.update(entry_key=f'{day.year}{day:%m%d}05010101{i:02d}',date=day.isoformat(),winner_label=int(i==1+j%2),
                market_probability=q,ability_probability=a,market_log_probability_change=(i-2)*.03,
                market_history_missing=0,market_rank=i,market_log_odds=math.log(1.25/q),market_inverse_sum=1.25,
                source_model_id='FICTITIOUS_MODEL',feature_history_cutoff=(day-timedelta(days=7)).isoformat(),
                **{k:.1*j for k in c.E_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


class SoftmaxTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_stay_finite(self):
        p,logs=softmax.normalized_scores(np.array([10000,9999,-10000,70000,70001]),np.array([0,0,0,1,1]))
        self.assertTrue(np.isfinite(p).all());self.assertTrue(np.isfinite(logs).all())
        self.assertAlmostEqual(p[:3].sum(),1.);self.assertAlmostEqual(p[3:].sum(),1.)

    def test_same_race_constant_invariance(self):
        x=np.array([2.,3.,-4.,1.]);codes=np.array([0,1,0,1])
        p,_=softmax.normalized_scores(x,codes);q,_=softmax.normalized_scores(x+np.array([100.,-999.,100.,-999.]),codes)
        self.assertTrue(np.allclose(p,q,atol=1e-14))

    def test_market_reference_coefficients_reproduce_market(self):
        f=frame()
        for name in c.TERMS:
            d={'version':c.VERSION,'config_id':name,'terms':c.TERMS[name],'coefficients':[1.]+[0.]*(len(c.TERMS[name])-1)}
            self.assertTrue(np.allclose(softmax.predict(f,d),f.market_probability,atol=1e-15))

    def test_analytic_gradient_matches_finite_differences(self):
        for config in c.TERMS:
            x,codes,y=softmax.design(frame(),config);theta=np.linspace(.9,.15,x.shape[1])
            for strength in c.STRENGTH:
                _,gradient=softmax.loss_gradient(theta,x,codes,y,strength)
                for i in range(len(theta)):
                    a=theta.copy();b=theta.copy();a[i]+=1e-6;b[i]-=1e-6
                    numeric=(softmax.loss_gradient(a,x,codes,y,strength)[0]-softmax.loss_gradient(b,x,codes,y,strength)[0])/2e-6
                    self.assertAlmostEqual(gradient[i],numeric,places=7)

    def test_loss_is_mean_race_not_mean_runner(self):
        f=frame(days=1);x,codes,y=softmax.design(f,'M')
        loss,_=softmax.loss_gradient(np.array([1.]),x,codes,y,.1)
        self.assertAlmostEqual(loss,-math.log(.6))

    def test_regularization_scale_is_half_strength_squared(self):
        x,codes,y=softmax.design(frame(),'M');theta=np.array([2.])
        no,_=softmax.loss_gradient(theta,x,codes,y,0.);yes,_=softmax.loss_gradient(theta,x,codes,y,.1)
        self.assertAlmostEqual(yes-no,.05)

    def test_terms_have_no_intercept_or_auxiliary_features(self):
        self.assertEqual(c.TERMS,{'M':('q',),'MH':('q','h'),'MA':('q','a'),'MHA':('q','a','h')})

    def test_missing_early_history_keeps_rows_and_h_zero(self):
        f=frame();f.loc[:2,'market_log_probability_change']=np.nan;f.loc[:2,'market_history_missing']=1
        x,codes,y=softmax.design(f,'MHA')
        self.assertEqual(len(x),len(f));self.assertTrue((x[:3,2]==0).all());self.assertTrue((f.market_history_missing[:3]==1).all())

    def test_invalid_probabilities_stop(self):
        for value in (0.,1.,np.nan,np.inf):
            f=frame();f.loc[0,'ability_probability']=value
            with self.assertRaises(safety.MLFailure):softmax.design(f,'MA')

    def test_duplicate_winner_label_stops(self):
        f=frame();f.loc[:2,'winner_label']=1
        with self.assertRaises(safety.MLFailure):softmax.design(f,'M')

    def test_model_order_or_version_change_stops(self):
        d={'version':'unknown','config_id':'MA','terms':['q','a'],'coefficients':[1.,.1]}
        with self.assertRaises(safety.MLFailure):softmax.predict(frame(),d)
        d.update(version=c.VERSION,terms=['a','q'])
        with self.assertRaises(safety.MLFailure):softmax.predict(frame(),d)

    def test_save_reload_coefficients_reproduces_predictions(self):
        d={'version':c.VERSION,'config_id':'MHA','terms':['q','a','h'],'coefficients':[1.1,.05,-.2]}
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            s.write(Path(td),'model.json',d);loaded=s.read(Path(td),'model.json')
            self.assertTrue(np.array_equal(softmax.predict(frame(),d),softmax.predict(frame(),loaded)))

    def test_future_OOF_is_rejected_before_optimization(self):
        sp=c.integration_split(2023);train=frame(date(2023,1,1));valid=frame(date(2022,9,8))
        with patch('scipy.optimize.minimize') as optimize,self.assertRaises(safety.MLFailure):
            softmax.fit(Path('unused'),'M','2023',train,valid,.01,sp)
        optimize.assert_not_called()


class ReuseAndChronologyTests(unittest.TestCase):
    def test_lambda_zero_and_one_endpoints(self):
        f=frame(days=1);spec=oldc.spec(anchor=True,ability=False);booster=MagicMock()
        booster.predict.return_value=np.array([.3,-.2,.1])
        bundle={'metadata':{'feature_spec':spec,'categories':{},'calibration':{'temperature':1.}},'boosters':{'win':booster}}
        self.assertTrue(np.allclose(reuse.shrink(f,bundle,0.),f.market_probability,atol=1e-15))
        self.assertTrue(np.allclose(reuse.shrink(f,bundle,1.),oldm.predict(bundle,f),atol=1e-15))
        self.assertTrue(booster.predict.call_args.kwargs['raw_score'])

    def test_temperature_uses_uncalibrated_input_once(self):
        f=frame(days=1);raw=np.array([.2,.3,.5]);p=oldm.normalize(f,raw,.5)
        self.assertTrue(np.allclose(p,raw**2/(raw**2).sum()))
        old=oldm.normalize(f,raw,.75);wrong=oldm.normalize(f,old,.5)
        self.assertFalse(np.allclose(p,wrong))

    def test_calibration_ties_are_predeclared(self):
        rows=[{'race_log_loss':1.,'coefficient':l} for l in c.LAMBDA]
        self.assertEqual(scoring.choose(rows,lambda r:r['coefficient'])['coefficient'],0.)
        rows=[{'race_log_loss':1.,'temperature':t} for t in c.TEMPERATURE]
        self.assertEqual(scoring.choose(rows,lambda r:(abs(r['temperature']-.75),r['temperature']))['temperature'],.75)

    def test_all_OOF_origins_are_strictly_prior(self):
        for year in range(2020,2026):oof.verify_timing(c.split(year),f'{year}-01-01')
        sp=c.split(2023);sp['calibration_end']='2023-01-01'
        with self.assertRaises(safety.MLFailure):oof.verify_timing(sp,'2023-01-01')

    def test_OOF_provenance_is_checked(self):
        f=frame(date(2023,1,1));summary={'split':c.split(2023),'model_id':'FICTITIOUS_MODEL'}
        oof.validate_rows(f,summary,2023)
        f.loc[0,'source_model_id']='WRONG'
        with self.assertRaises(safety.MLFailure):oof.validate_rows(f,summary,2023)

    def test_future_rows_cannot_masquerade_as_old_OOF(self):
        f=frame(date(2024,1,1));summary={'split':c.split(2023),'model_id':'FICTITIOUS_MODEL'}
        with self.assertRaises(safety.MLFailure):oof.validate_rows(f,summary,2023)

    def test_future_or_wrong_history_cutoff_is_rejected(self):
        f=frame(date(2023,1,1));f.loc[0,'feature_history_cutoff']='2023-01-01'
        with self.assertRaises(safety.MLFailure):oof.validate_rows(f,{'split':c.split(2023),'model_id':'FICTITIOUS_MODEL'},2023)

    def test_2026_rejected_before_cache_or_SQL(self):
        with self.assertRaises(safety.MLFailure):e.query(date(2026,1,1),date(2026,9,30))
        with self.assertRaises(safety.MLFailure):integrate.dataset(Path('unused'),date(2026,1,1),date(2026,9,30))
        with self.assertRaises(safety.MLFailure):c.split(2026)

    def test_main_2025_cannot_be_double_counted(self):
        with self.assertRaises(safety.MLFailure):scoring.annual_pool({'main':[],'2023':[],'2024':[],'2025':[]})

    def test_race_keys_and_rosters_must_match(self):
        r=('2025','0301','05','01','01','01')
        a=[core.PredictionRace(r,{'01':.6,'02':.4})];b=[core.PredictionRace(r,{'01':.6,'03':.4})]
        with self.assertRaises(safety.MLFailure):scoring.align(a,b)

    def test_runner_order_does_not_change_alignment(self):
        r=('2025','0301','05','01','01','01')
        a=[core.PredictionRace(r,{'01':.6,'02':.4})];b=[core.PredictionRace(r,{'02':.4,'01':.6})]
        self.assertEqual(scoring.compare(a,b,{r:'01'})['race_log_loss']['ci_high'],0.)

    def test_week_bootstrap_uses_5000_replicates(self):
        r=('2025','0301','05','01','01','01');p=[core.PredictionRace(r,{'01':.6,'02':.4})]
        result=scoring.compare(p,p,{r:'01'})
        self.assertEqual(result['repetitions'],5000);self.assertEqual(result['race_log_loss']['ci_high'],0.)

    def test_incomplete_unit_is_not_reused_and_tampering_stops(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td);s.write(folder,'partial.json',{'status':'started'})
            self.assertIsNone(s.completed(folder,'unit'))
            s.complete(folder,'unit',[folder/'partial.json'],{'ok':True})
            self.assertIsNotNone(s.completed(folder,'unit'))
            (folder/'partial.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(safety.MLFailure):s.completed(folder,'unit')

    def test_nested_completion_checks_the_actual_child_data(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td);s.write(folder,'child.json',{'value':1})
            s.complete(folder,'child',[folder/'child.json'],{})
            s.complete(folder,'parent',[folder/'child.complete.json'],{})
            (folder/'child.json').write_text('{"value":2}',encoding='utf-8')
            with self.assertRaises(safety.MLFailure):s.completed(folder,'parent')

    def test_verified_resume_never_overwrites_saved_timestamp(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td);s.write(folder,'artifact.json',{'value':1,'created_at_jst':'first'})
            before=(folder/'artifact.json').read_bytes()
            s.write(folder,'artifact.json',{'value':1,'created_at_jst':'retry'})
            self.assertEqual(before,(folder/'artifact.json').read_bytes())
            with self.assertRaises(safety.MLFailure):s.write(folder,'artifact.json',{'value':2})

    def test_checkpoint_has_actual_run_identity(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td)/'evaluations'/('b'*32)/'unit';folder.mkdir(parents=True)
            s.complete(folder,'example',[],{})
            self.assertEqual(s.read(folder,'example.complete.json')['run_id'],'b'*32)

    def test_initial_checkpoint_scope_read_is_compatible_without_rewrite(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td)/'evaluations'/('b'*32)/'unit';folder.mkdir(parents=True)
            s.write(folder,'example.complete.json',{'status':'complete','run_id':'unit','files':{},'summary':{}})
            before=(folder/'example.complete.json').read_bytes()
            self.assertEqual(s.completed(folder,'example')['run_id'],'b'*32)
            self.assertEqual(before,(folder/'example.complete.json').read_bytes())

    def test_eighth_real_fit_is_blocked_before_training(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            folder=Path(td)/'fit_units';folder.mkdir()
            for i in range(7):(folder/f'other_{i}.started.json').write_text('{}',encoding='utf-8')
            with patch.object(oldm,'fit') as train,self.assertRaises(safety.MLFailure):
                oof.fit_once(Path(td),'OOF_ABCD',2020,c.split(2020),{})
            train.assert_not_called()

    def test_completed_E_CLI_status_is_safe_and_noninteractive(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            identity='a'*32;folder=Path(td)/'evaluations'/identity;folder.mkdir(parents=True)
            with patch.object(e,'run',return_value={'status':'complete','new_LightGBM_fits':4}),redirect_stdout(StringIO()) as output:
                self.assertEqual(cli.main(['--stage','e','--run-id',identity]),0)
            value=json.loads(output.getvalue());self.assertEqual(value['status'],'ok');self.assertEqual(value['unit_status'],'complete')


def Erow(day,raw='-010',rank='01'):
    return dict(kaisai_nen=str(day.year),kaisai_tsukihi=day.strftime('%m%d'),keibajo_code='05',kaisai_kai='01',
        kaisai_nichime='01',race_bango='01',horse_number='01',horse_id='FAKE_HORSE',anomaly='0',finish_position=rank,time_sa=raw)


class SignedTimeTests(unittest.TestCase):
    def test_units_and_sign_not_forced_to_zero(self):
        self.assertEqual(e.parse('-015','01','0')[0],-1.5);self.assertEqual(e.parse('+015','02','0')[0],1.5)

    def test_signed_zeros_are_valid_and_sign_preserved(self):
        a,reason=e.parse('-000','01','0');b,_=e.parse('+000','02','0')
        self.assertEqual(reason,'signed_zero_valid');self.assertEqual(a,0.);self.assertEqual(b,0.)
        self.assertLess(math.copysign(1.,a),0);self.assertGreater(math.copysign(1.,b),0)

    def test_initial_sentinel_null_are_missing(self):
        for raw in (None,'','    ','0000','9999'):self.assertIsNone(e.parse(raw,'01','0')[0])

    def test_strict_ASCII_width_and_sign(self):
        for raw in ('+01','+0000','0015',' 015','＋015','+０１５','+nan'):
            self.assertIsNone(e.parse(raw,'02','0')[0])

    def test_conflicting_sign_and_anomaly_are_missing(self):
        self.assertIsNone(e.parse('+015','01','0')[0]);self.assertIsNone(e.parse('-015','02','0')[0])
        self.assertIsNone(e.parse('+015','02','4')[0])

    def test_last_three_weights_are_not_searched(self):
        b=e.History()
        for day,raw in [(date(2024,1,1),'-010'),(date(2024,1,8),'-020'),(date(2024,1,15),'-030')]:b.day([Erow(day,raw)])
        result=b.day([Erow(date(2024,1,22))])[0]
        self.assertEqual(result[c.E_FEATURES[0]],-3.)
        self.assertAlmostEqual(result[c.E_FEATURES[1]],(-3.-1.-.25)/1.75)

    def test_seven_day_embargo_and_current_result_block(self):
        day=date(2024,1,1)
        b=e.History();b.day([Erow(day,'-015')]);a=b.day([Erow(day+timedelta(days=6),'-999')])[0]
        self.assertIsNone(a[c.E_FEATURES[0]])
        b=e.History();b.day([Erow(day,'-015')]);a=b.day([Erow(day+timedelta(days=7),'-999')])[0]
        self.assertEqual(a[c.E_FEATURES[0]],-1.5)

    def test_future_result_addition_does_not_rewrite_past_features(self):
        b=e.History();b.day([Erow(date(2024,1,1))]);past=b.day([Erow(date(2024,1,8))]);saved=deepcopy(past)
        b.day([Erow(date(2024,2,1),'-999')]);self.assertEqual(past,saved)

    def test_extra_query_is_read_only_and_bounded(self):
        sql,params=e.query(date(2025,1,1),date(2025,1,31))
        self.assertIn('time_sa',sql);self.assertIn('LIMIT',sql);self.assertEqual(params[-1],20001)
        self.assertNotIn('soha_time',sql);self.assertNotIn('UPDATE',sql)

    def test_E_model_saved_reload_matches_predictions(self):
        parts={'train':frame(date(2024,6,1),20),'validation':frame(date(2024,9,8),5),'calibration':frame(date(2024,11,8),5)}
        split=c.split(2025);bundle=e.fit_model(parts,split,oldc.plan()['ability_parameters'])
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            legacy.save_models(bundle,Path(td),root=Path(td));loaded=e.load_model(Path(td))
            self.assertTrue(np.array_equal(e.predict(bundle,parts['calibration']),e.predict(loaded,parts['calibration'])))


class FinancialAndFreezeTests(unittest.TestCase):
    def rows(self,bet):
        return [{'race_key':['2025','0301','05','01','01','01'],'ai':bet}]

    def test_zero_purchase_ROI_stays_undefined(self):
        value=finance.weekly_financial(self.rows({'status':'no_bet'}))
        self.assertIsNone(value['ROI_percent_ci']);self.assertEqual(value['zero_purchase_fraction'],1.)
        self.assertEqual(value['profit_yen_ci'],[0.,0.])

    def test_pending_is_not_zero_payout(self):
        value=finance.weekly_financial(self.rows({'status':'pending','payout_yen':None}))
        self.assertIsNone(value['ROI_percent_ci']);self.assertIsNone(value['profit_yen_ci'])
        self.assertEqual(value['unsettled_replica_fraction'],1.)

    def test_zero_purchase_resamples_excluded_only_from_ROI(self):
        rows=[]
        for j in range(10):
            day=date(2025,1,4)+timedelta(days=7*j)
            rows.append({'race_key':[str(day.year),day.strftime('%m%d'),'05','01','01','01'],
                'ai':{'status':'settled','payout_yen':200} if j==0 else {'status':'no_bet'}})
        value=finance.weekly_financial(rows)
        self.assertGreater(value['zero_purchase_fraction'],0.);self.assertLess(value['zero_purchase_fraction'],1.)
        self.assertEqual(value['ROI_percent_ci'],[200.,200.]);self.assertFalse(value['DD_bootstrapped'])

    def test_probability_filter_precedes_EV_argmax_and_resettles(self):
        r=('2025','0301','05','01','01','01');roster=('01','02','03')
        schedule=core.Schedule(r,datetime(2025,3,1,15,tzinfo=core.JST),None,roster,True,'reviewed_historical_schedule')
        odds={'01':core.decimal(900),'02':core.decimal(20),'03':core.decimal(1)}
        snapshot=core.Snapshot(r,datetime(2025,3,1,14,49,tzinfo=core.JST),None,odds,'fake',True,publication_precision='minute')
        out=core.Outcome(r,'normal_final',{'02':2000},True,True)
        ledger=core.run_simulation([core.PredictionRace(r,{'01':.01,'02':.1,'03':.89})],[schedule],[snapshot],[out],
            policy=core.WinPolicy(probability_method='identity',timing_basis='jra_publication_time',evaluation_start='2025-03-01',evaluation_end='2025-03-01'))
        self.assertEqual(ledger['rows'][0]['ai_horse'],'01')
        packet={r:{'outcome':{'status':'normal_final','payouts_yen_per_100':{'02':2000},'verified':True,'complete_winner_set':True,'reason':None}}}
        filtered=finance.apply_probability_filter(ledger,packet,{r:'02'})
        self.assertEqual(filtered['rows'][0]['ai_horse'],'02');self.assertEqual(filtered['rows'][0]['ai']['payout_yen'],2000)
        self.assertEqual(ledger['rows'][0]['ai_horse'],'01')

    def test_candidate_rule_does_not_use_profit_or_CI_crossing(self):
        self.assertTrue(report.retain([-.01,-.02,.01],-.005))
        self.assertFalse(report.retain([-.1,.01,.01],-.005));self.assertFalse(report.retain([-.1,-.1,1.],.1))

    def test_final_plan_comparisons_follow_candidate_outcome(self):
        p=report.future_plan({'selected':'Stage2_ABCD'},{'selected_AI_candidate':None,'frozen_market_control':'MH'})
        self.assertEqual(p['major_comparisons'][0]['control'],'old_feature_control');self.assertEqual(p['major_comparisons'][1]['control'],'raw_market')
        self.assertEqual(p['confidence_each'],.975);self.assertFalse(p['training_and_evaluation_for_2026_executed'])
        p=report.future_plan({'selected':'ABCD_E'},{'selected_AI_candidate':'MHA','frozen_market_control':'MH'})
        self.assertEqual(p['major_comparisons'][0]['control'],'Stage2_ABCD');self.assertEqual(p['major_comparisons'][1]['control'],'MH')

    def test_experiment_caps_and_old_contract_are_fixed(self):
        p=c.plan();self.assertEqual(p['new_LightGBM_fits_max'],7);self.assertEqual(p['integration']['max_optimizations'],24)
        self.assertEqual(p['lambda']['max_evaluations'],40);self.assertEqual(oldc.spec(tuple('ABCD'))['formula_version'],'causal_groups_20261019_v2')

    def test_final_stage_not_exposed_in_development(self):
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):cli.main(['--stage','final'])


if __name__=='__main__':unittest.main()
