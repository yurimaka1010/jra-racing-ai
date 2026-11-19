"""Fictional Stage 4 contracts. Real DB driver and sockets blocked by run_tests.py."""
from datetime import date,datetime,timedelta
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json,math,unittest
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
from jra_racing_ai.ml import safety
from jra_racing_ai.stage4 import contracts as c,direct,quotes,data,models,history,store,oof,finance,accuracy
from jra_racing_ai.stage3 import scoring
from jra_racing_ai.stage3.softmax import normalized_scores
from jra_racing_ai.win_sim import core
from jra_racing_ai import stage1_analysis as select,stage1
from jra_racing_ai.db_connection import validate_audit_sql


def fictional(start=date(2022,1,1),days=4):
    rows=[]
    for j in range(days):
        d=start+timedelta(days=j)
        for i,(h,a) in enumerate(zip((.6,.3,.1),(.5,.2,.3)),1):
            r={f:0. for f in c.H_FEATURES+c.EXTRA}
            r.update(entry_key=f'{d.year}{d:%m%d}05010101{i:02d}',date=d.isoformat(),winner_label=int(i==1+j%2),
                history_probability=h,ability_probability=a,market_probability=h,source_H_model_id='FICTIONAL_H',ability_model_id='FICTIONAL_A')
            rows.append(r)
    return pd.DataFrame(rows)


RACE=('2025','0301','05','01','01','01')
ROSTER=('01','02')
START=datetime(2025,3,1,15,tzinfo=core.JST)


def row(clock='03011359',prices=('0020','0040'),category='1'):
    return dict(race_key=RACE,source_table='apd_sokuho_o1',record_type='O1',status=category,published=clock,sale='7',
        payload=''.join(f'{i:02d}'+p+f'{i:02d}' for i,p in enumerate(prices,1))+' '*8*(28-len(prices)),oversized=False)


def schedule():return core.Schedule(RACE,START,None,ROSTER,True,'reviewed_historical_schedule')


class DirectTests(unittest.TestCase):
    def test_zero_coefficients_reproduce_H(self):
        f=fictional()
        for name,n in (('Hcal',1),('Hstack',2)):
            doc=dict(version=c.VERSION,config_id=name,scales=direct.scales(f).tolist(),coefficients=[0.]*n)
            np.testing.assert_allclose(direct.predict(f,doc),f.history_probability,atol=1e-15)

    def test_gradient_matches_numeric(self):
        f=fictional()
        for config in ('Hcal','Hstack'):
            b,x,codes,_=direct.design(f,config,direct.scales(f));y=direct.targets(f,codes);theta=np.linspace(.1,.2,x.shape[1])
            for strength in c.STRENGTHS:
                _,g=direct.loss_gradient(theta,b,x,codes,y,strength)
                for i in range(len(theta)):
                    l=theta.copy();r=theta.copy();l[i]-=1e-6;r[i]+=1e-6
                    num=(direct.loss_gradient(r,b,x,codes,y,strength)[0]-direct.loss_gradient(l,b,x,codes,y,strength)[0])/2e-6
                    self.assertAlmostEqual(g[i],num,places=7)

    def test_scale_equal_race_not_equal_runner(self):
        f=fictional(days=2).drop(index=[5]);f.loc[3:4,'history_probability']=[.7,.3];f.loc[3:4,'ability_probability']=[.8,.2]
        values=[]
        for _,part in f.groupby(f.entry_key.str[:16]):
            p=np.log(part[['history_probability','ability_probability']].to_numpy());values.append(np.mean((p-p.mean(axis=0))**2,axis=0))
        np.testing.assert_allclose(direct.scales(f),np.sqrt(np.mean(values,axis=0)))

    def test_fixed_scales_and_prediction_ignore_labels(self):
        f=fictional();scale=direct.scales(f);doc=dict(version=c.VERSION,config_id='Hstack',scales=scale.tolist(),coefficients=[.1,-.03])
        before=direct.predict(f,doc);f.winner_label=np.nan
        np.testing.assert_array_equal(direct.predict(f,doc),before)
        self.assertEqual(doc['scales'],scale.tolist())

    def test_zero_scale_and_nonfinite_rejected(self):
        f=fictional();f.history_probability=1/3
        with self.assertRaises(safety.MLFailure):direct.scales(f)
        f=fictional();f.loc[0,'ability_probability']=np.nan
        with self.assertRaises(safety.MLFailure):direct.inputs(f)

    def test_duplicate_and_wrong_sums_rejected(self):
        f=fictional()
        with self.assertRaises(safety.MLFailure):direct.inputs(pd.concat([f,f.iloc[:1]]))
        f.loc[0,'history_probability']=.5
        with self.assertRaises(safety.MLFailure):direct.inputs(f)

    def test_softmax_finite_sum_and_constant_invariance(self):
        x=np.array([1e4,1e4+1,-1e4,-1e4+2]);codes=np.array([0,0,1,1])
        p,l=normalized_scores(x,codes);q,_=normalized_scores(x+np.array([200,200,-700,-700]),codes)
        self.assertTrue(np.isfinite(l).all());np.testing.assert_allclose(np.bincount(codes,weights=p),1.)
        np.testing.assert_allclose(p,q,atol=1e-13)

    def test_jsd_boundaries_are_label_free_linear_and_ties_unique(self):
        f=fictional();b=direct.boundaries(f);f.winner_label=np.nan
        self.assertEqual(b,direct.boundaries(f));self.assertEqual(b['method'],'linear')
        self.assertEqual(direct.stratum(b['q90'],b),'high')
        self.assertEqual(direct.stratum(b['q50']-1e-8,b),'low')
        self.assertEqual(direct.stratum(.2,dict(q50=.1,q90=.3)),'middle')

    def test_no_extra_terms_and_budget_fixed(self):
        plan=c.plan();self.assertEqual(plan['direct']['max_optimizations'],12);self.assertEqual(plan['LightGBM_new_fits_max'],5)
        for name,n in (('Hcal',1),('Hstack',2)):
            self.assertEqual(direct.design(fictional(),name,[1.,1.])[1].shape[1],n)

    def test_chronology_before_optimizer(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(direct,'minimize') as opt:
            with self.assertRaises(safety.MLFailure):
                direct.fit(Path(td),'Hstack',2023,fictional(date(2023,1,1)),fictional(date(2022,9,8)),.01,c.direct_split(2023))
            opt.assert_not_called()

    def test_prediction_doc_roundtrip(self):
        doc=dict(version=c.VERSION,config_id='Hstack',scales=[.5,.4],coefficients=[.1,.03])
        np.testing.assert_array_equal(direct.predict(fictional(),doc),direct.predict(fictional(),json.loads(json.dumps(doc))))


class HistoryTests(unittest.TestCase):
    def test_query_is_bounded_select_and_minute_end(self):
        sql,args=quotes.query([RACE],{RACE:START});validate_audit_sql(sql)
        self.assertIn('LIMIT 2',sql);self.assertIn('public.apd_sokuho_o1',sql);self.assertEqual(args[-1],'03011359')
        with self.assertRaises(safety.MLFailure):quotes.query([RACE]*33,{RACE:START})

    def test_boundary_future_stale_and_roster_change(self):
        pick=lambda rows:select.select_quote(RACE,rows,ROSTER,schedule(),60,300)
        q,_=pick([row(),row('03011400'),row('03011401')]);self.assertEqual(q.published_at.hour,13);self.assertEqual(q.published_at.minute,59)
        self.assertIsNone(pick([row('03011400')])[0]);self.assertEqual(pick([row('03011354')])[1],'odds_stale')
        self.assertIsNone(pick([row(prices=('0020','----'))])[0])

    def test_unknown_deleted_final_and_duplicate_fail_closed(self):
        pick=lambda rows:select.select_quote(RACE,rows,ROSTER,schedule(),60,300)
        self.assertIsNone(pick([row(category='5')])[0]);self.assertIsNone(pick([row(),row()])[0])
        self.assertIsNone(pick([row(),dict(row(),status='0')])[0])

    def test_delta_keeps_time_and_log_normalized_probabilities(self):
        q60,_=select.select_quote(RACE,[row()],ROSTER,schedule(),60,300)
        q30,_=select.select_quote(RACE,[row('03011429',prices=('0030','0030'))],ROSTER,schedule(),30,300)
        packet={'quotes':{'30':stage1._quote_document(q30)}}
        values,reason=quotes.delta_features(packet,stage1._quote_document(q60));self.assertIsNone(reason)
        self.assertAlmostEqual(values['01'][c.EXTRA[0]],math.log(.5)-math.log(2/3));self.assertEqual(values['01'][c.EXTRA[1]],30.)

    def test_history_missing_retains_primary_rows(self):
        f=fictional(date(2025,3,1),days=1);r=RACE
        packet={r:{'quotes':{'10':{'present':True},'30':None}}};extra={r:{'quote':None}}
        with patch.object(data,'market',return_value=packet),patch.object(quotes,'load',return_value=extra):
            output=quotes.augment(Path('unused'),f)
        self.assertEqual(output.entry_key.tolist(),f.entry_key.tolist());self.assertTrue(output[c.EXTRA[0]].isna().all())
        self.assertTrue((output[c.EXTRA[2]]==1).all());self.assertFalse(history.useful(output))

    def test_new_format_not_accepted_by_old_loader(self):
        from jra_racing_ai.research import model as old
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            p=Path(td);store.write(p,'model.json',dict(format_version=4,model_kind='stage4_H60_v1'))
            with self.assertRaises(safety.MLFailure):old.load(p,root=p)

    def test_all_dates_2026_and_after_sep2025_rejected(self):
        for d in (date(2025,10,1),date(2026,1,1),date(2026,9,30)):
            with self.assertRaises(safety.MLFailure):data.market_frame(Path('unused'),d,d)
            r=(str(d.year),d.strftime('%m%d'),'05','01','01','01')
            with self.assertRaises(safety.MLFailure):quotes.query([r],{r:datetime(d.year,d.month,d.day,15,tzinfo=core.JST)})
        with self.assertRaises(safety.MLFailure):c.h_split(2026)

    def test_packet_parser_stops_before_post_boundary_record(self):
        raw={'records':[{'race_key':['2025','0930','05','01','01','01'],'value':1},
                        {'race_key':['2025','1001','05','01','01','01'],'value':'DO_NOT_PARSE'}]}
        with TemporaryDirectory() as td:
            p=Path(td)/'packets.json';p.write_text(json.dumps(raw,indent=2),encoding='utf-8')
            with patch.object(data.json,'loads',wraps=json.loads) as loads:values=list(data.packet_rows(p))
            self.assertEqual(len(values),1);self.assertEqual(loads.call_count,1)

    def test_mismatched_rosters_rejected_without_partial_normalizing(self):
        a=[core.PredictionRace(RACE,{'01':.6,'02':.4})];b=[core.PredictionRace(RACE,{'01':.6,'03':.4})]
        with self.assertRaises(safety.MLFailure):scoring.align(a,b)

    def test_merged_probabilities_reject_extra_horse_in_same_race(self):
        f=fictional(days=1);p=f[['entry_key','ability_probability']].copy()
        extra=p.iloc[[0]].copy();extra.entry_key=extra.entry_key.str[:-2]+'04'
        with self.assertRaises(safety.MLFailure):data.merge_probability(f,pd.concat([p,extra]),'ability_probability')

    def test_zero_residual_reproduces_market_and_excludes_result_columns(self):
        f=fictional();booster=MagicMock();booster.predict.return_value=np.zeros(len(f))
        bundle={'metadata':{'model_kind':'stage4_H60_v1'},'boosters':{'win':booster}}
        p=models.predict(bundle,f);np.testing.assert_allclose(p,f.market_probability,atol=1e-15)
        f['final_odds']=999.;f['finish_position']=1;f['winner_label']=0
        np.testing.assert_array_equal(models.predict(bundle,f),p)
        self.assertNotIn('final_odds',booster.predict.call_args.args[0].columns)
        self.assertNotIn('winner_label',booster.predict.call_args.args[0].columns)
        self.assertTrue(booster.predict.call_args.kwargs['raw_score'])

    def test_H60_real_serialization_and_resumption_without_extra_fit(self):
        from jra_racing_ai.research import contracts as oldc
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(store,'ref_json',return_value=oldc.plan()):
            folder=Path(td);store.write(folder,'references.json',{})
            frames=[fictional(date(2022,1,1),days=30),fictional(date(2022,9,8)),fictional(date(2022,11,8))]
            for f in frames:f[c.EXTRA[0]]=np.arange(len(f))*.01
            bundle,meta=models.fit(folder,'H60',2023,c.h_split(2023),frames)
            loaded=models.load(folder/'models'/meta['model_id'])
            np.testing.assert_array_equal(models.predict(bundle,frames[1]),models.predict(loaded,frames[1]))
            with patch('lightgbm.train',side_effect=AssertionError('must_not_refit')):
                _,again=models.fit(folder,'H60',2023,c.h_split(2023),frames)
            self.assertEqual(meta['model_id'],again['model_id']);self.assertEqual(meta['reload_max_difference'],0.)

    def test_OOF_future_model_duplicate_and_cutoff_rejected(self):
        year=2023;f=fictional(date(year,1,1));sp=c.h_split(year)
        f['feature_history_cutoff']=[(date.fromisoformat(d)-timedelta(days=7)).isoformat() for d in f.date]
        for field in ('train_end','validation_end','calibration_end'):f[field]=sp[field]
        summary={'model_id':'FICTIONAL_H','split':sp};oof.validate_rows(f,summary,year)
        for bad in ('model_id','duplicate','cutoff','future_date'):
            wrong=f.copy()
            if bad=='model_id':wrong.source_H_model_id='FUTURE_H'
            if bad=='duplicate':wrong=pd.concat([wrong,wrong.iloc[[0]]])
            if bad=='cutoff':wrong.feature_history_cutoff=wrong.date
            if bad=='future_date':wrong.date='2024-01-01'
            with self.assertRaises(safety.MLFailure):oof.validate_rows(wrong,summary,year)


def ticket(day,paid=0,*,status='settled',number='01'):
    race=(str(day.year),day.strftime('%m%d'),'05','01','01','01')
    bet={'status':status,'reason':None if status!='pending' else 'result_unavailable','stake_yen':100,
        'payout_yen':paid if status=='settled' else None,'net_profit_yen':paid-100 if status=='settled' else None,
        'hit':paid>0 if status=='settled' else None}
    return {'race_key':race,'month':day.strftime('%Y-%m'),'scheduled_start':day.isoformat()+'T15:00:00+09:00',
        'ai_horse':number,'ai':bet,'reason':'ev_selected','selected':{'probability':.2,'decision_odds':10.,'expected_return':1.,'horse_number':number}}


class FinancialTests(unittest.TestCase):
    def test_official_payout_not_decision_price_and_pending(self):
        outcome=core.Outcome(RACE,'normal_final',{'01':230},True,True)
        paid=core.settle('01',outcome,ROSTER)
        self.assertEqual(paid['payout_yen'],230);self.assertEqual(paid['net_profit_yen'],130)
        pending=core.settle('01',None,ROSTER);self.assertIsNone(pending['payout_yen']);self.assertEqual(pending['status'],'pending')

    def test_fixed_EV_argmax_and_one_ticket(self):
        p=[core.PredictionRace(RACE,{'01':.6,'02':.4})]
        sc=core.Schedule(RACE,START,None,ROSTER,True,'reviewed_historical_schedule')
        snap=core.Snapshot(RACE,START-timedelta(minutes=12),None,{'01':core.decimal(2),'02':core.decimal(4)},'fictional',True,publication_precision='minute')
        policy=core.WinPolicy(probability_method='identity',timing_basis='jra_publication_time')
        ledger=core.run_simulation(p,[sc],[snap],[],policy=policy)
        r=ledger['rows'][0];self.assertEqual(r['ai_horse'],'02');self.assertEqual(r['ai']['stake_yen'],100)
        self.assertEqual(ledger['summary']['ai']['bet_count'],1)

    def test_27_months_include_no_purchase_months(self):
        rows=[ticket(date(2023,1,7),paid=300)];w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        f=finance.financial(rows,w)
        self.assertEqual(len(f['monthly']),27);self.assertEqual(f['no_purchase_months'],26)
        self.assertAlmostEqual(f['monthly_mean_profit_yen_including_27_months'],200/27)
        self.assertEqual(f['strictly_profitable_years'],1);self.assertFalse(finance.retain(f)['retain_financial_research_candidate'])

    def test_zero_bets_and_all_losses_not_falsely_certain(self):
        rows=[ticket(date(2023,1,7))];no=[finance.skip(rows[0],'no_buy')];w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        z=finance.financial(no,w);self.assertIsNone(z['financial']['return_rate_percent']);self.assertEqual(z['financial']['net_profit_yen'],0)
        stats=w.summarize(no);self.assertIsNone(stats['95_percent']['ROI_percent_ci']);self.assertEqual(stats['zero_purchase_replica_fraction'],1.)
        loss=w.summarize(rows);self.assertEqual(loss['95_percent']['ROI_percent_ci'],[0.,0.]);self.assertTrue(loss['all_loss_degenerate_CI_does_not_prove_true_ROI_zero'])

    def test_pending_is_never_zero_payout_or_profit(self):
        rows=[ticket(date(2023,1,7),status='pending')];w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        f=finance.financial(rows,w);self.assertIsNone(f['financial']['payout_yen']);self.assertIsNone(f['financial']['net_profit_yen'])
        self.assertIsNone(f['weekly_uncertainty']['95_percent']['profit_yen_ci']);self.assertEqual(f['financial']['unsettled_count'],1)

    def test_paired_week_draws_identical_and_venue_cluster_shared(self):
        r1=ticket(date(2023,1,7),300);r2=ticket(date(2023,1,8),0)
        r2['race_key']=(*r2['race_key'][:2],'06',*r2['race_key'][3:]);rows=[r1,r2]
        w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        self.assertEqual(len(w.weeks),1);pair=w.paired(rows,deepcopy(rows))
        self.assertEqual(pair['95_percent_ci'],[0.,0.]);self.assertEqual(pair['profit_difference_yen'],0.)
        self.assertEqual(w.summarize(rows)['shared_indices_sha256'],pair['shared_indices_sha256'])

    def test_high_D_rule_only_skips_outside_fixed_layer(self):
        a=ticket(date(2023,1,7),300);b=ticket(date(2023,1,8));before=deepcopy([a,b])
        result=finance.restrict_high(before,{tuple(a['race_key'])})
        self.assertEqual(result[0],a);self.assertEqual(result[1]['ai']['status'],'no_bet');self.assertEqual(before,[a,b])

    def test_no_purchase_year_does_not_count_as_profitable(self):
        rows=[ticket(date(2023,1,7),300),ticket(date(2024,1,7),300)];w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        f=finance.financial(rows,w);self.assertEqual(f['strictly_profitable_years'],2);self.assertEqual(f['no_purchase_years'],1)
        self.assertTrue(finance.retain(f)['retain_financial_research_candidate'])

    def test_risk_removes_payout_and_ticket_with_distinct_meanings(self):
        rows=[ticket(date(2023,1,7),500),ticket(date(2023,1,8),0)];w=finance.Weekly([r['race_key'] for r in rows],repetitions=100)
        f=finance.financial(rows,w);self.assertEqual(f['financial']['net_profit_yen'],300)
        self.assertEqual(f['risk']['profit_after_zeroing_largest_payout_keeping_its_stake_yen'],-200)
        self.assertEqual(f['risk']['profit_after_removing_largest_payout_ticket_yen'],-100)
        self.assertEqual(f['risk']['max_drawdown_yen'],100)

    def test_finance_requires_preexisting_accuracy_freeze(self):
        with TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)),patch.object(accuracy,'load') as loader:
            with self.assertRaises(safety.MLFailure):finance.run(Path(td))
            loader.assert_not_called()

    def test_2026_rejected_even_by_direct_predict_or_financial_week(self):
        f=fictional(date(2026,1,1));doc=dict(version=c.VERSION,config_id='Hcal',scales=[1.,1.],coefficients=[0.])
        with self.assertRaises(safety.MLFailure):direct.predict(f,doc)
        with self.assertRaises(safety.MLFailure):finance.Weekly([ticket(date(2026,1,1))['race_key']])


if __name__=='__main__':unittest.main()
