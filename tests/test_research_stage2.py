"""Fictional-only causal, compatibility and offset regression checks."""
from copy import deepcopy
from datetime import date,timedelta,datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

from jra_racing_ai.ml import features as old_features,safety
from jra_racing_ai.research import contracts as c,features as f,model as m,metrics,io,market_data
from jra_racing_ai.win_sim import core


def source(day,*,ids=None):
    out=[]
    for i in range(1,4):
        out.append(dict(kaisai_nen=str(day.year),kaisai_tsukihi=day.strftime('%m%d'),keibajo_code='05',
            kaisai_kai='01',kaisai_nichime='01',race_bango='01',horse_number=f'{i:02d}',
            horse_id=(ids[i-1] if ids else f'FAKE{i:06d}'),finish_position=f'{i:02d}',
            surface='11',distance='1600',declared_runners='04',actual_runners='03',frame=str(i),
            age='04',sex='1',weight_raw='550',weight_kg='55',start_time='1500',previous_start='0000',
            grade=' ',previous_grade_revision=' ',condition_minimum_age='005',
            condition_2='005',condition_3='005',condition_4='005',condition_5='005',
            race_type='13',weight_type='4',jockey='00001',previous_jockey_revision='00000',
            trainer='00011',anomaly='0',late3f_raw=str(340+10*i),corner4=f'{i:02d}',
            margin1='1  ' if i>1 else '   ',margin2='   ',margin3='   '))
    return out


class CausalGroupsTests(unittest.TestCase):
    def setUp(self):
        self.past=source(date(2024,11,1));self.target=source(date(2024,11,15))

    def test_legacy_constants_are_not_replaced(self):
        old=tuple(old_features.FEATURES)
        self.assertEqual(c.spec()['features'],list(old))
        self.assertNotIn('actual_runners',old_features.DATA_FIELDS)
        for group in c.GROUPS:
            self.assertEqual(c.spec((group,))['groups'],[group])
        self.assertEqual(old_features.FEATURES,old)

    def test_legacy_feature_values_match_original_generator(self):
        def legacy(rows):
            return [{k:r.get(k,'') for k in old_features.DATA_FIELDS} for r in rows]
        expected=old_features.build_features(legacy(self.target),history=legacy(self.past))
        actual=f.build(self.target,history=self.past)
        for a,b in zip(actual,expected):
            self.assertEqual({k:a[k] for k in old_features.FEATURES},b)

    def test_actual_starters_do_not_change_declared_registered_count(self):
        actual=f.build(self.target,history=self.past)
        self.assertEqual(actual[1]['a_previous_finish_percentile'],.5)
        self.assertEqual(actual[1]['declared_runners'],4)

    def test_current_results_never_enter_current_features(self):
        changed=deepcopy(self.target)
        for i,r in enumerate(changed):
            r.update(finish_position=str(3-i).zfill(2),actual_runners='28',late3f_raw='100',corner4='01',
                     margin1='Z  ',margin2='A  ')
        self.assertEqual(f.build(self.target,history=self.past),f.build(changed,history=self.past))

    def test_future_records_do_not_change_past_features(self):
        future=source(date(2025,4,1))
        self.assertEqual(f.build(self.target,history=self.past),f.build(self.target,history=self.past+future))

    def test_seven_day_delay_is_inclusive_and_six_is_excluded(self):
        day=date(2024,11,1)
        self.assertIsNone(f.build(source(day+timedelta(days=6)),history=self.past)[0]['a_previous_finish_percentile'])
        self.assertEqual(f.build(source(day+timedelta(days=7)),history=self.past)[0]['a_previous_finish_percentile'],1.)

    def test_last_three_weights_are_fixed(self):
        history=[]
        for day in (date(2024,10,1),date(2024,10,8),date(2024,10,15)):
            history.extend(source(day))
        history[-3]['finish_position']='03';history[-1]['finish_position']='01'
        x=f.build(self.target,history=history)[0]
        self.assertAlmostEqual(x['a_recent3_finish_percentile'],(.5+.25)/1.75)

    def test_B_codes_remain_categories_without_invented_order(self):
        changed=deepcopy(self.target)
        for r in changed:r['condition_4']='999'
        x=f.build(changed,history=self.past)[0]
        self.assertEqual(x['b_condition'],'999')
        self.assertEqual(x['b_condition_transition'],'005>999')
        changed[0]['condition_4']='888'
        self.assertIsNone(f.build(changed,history=self.past)[0]['b_condition'])

    def test_C_relative_3F_uses_all_starters(self):
        x=f.build(self.target,history=self.past)[0]
        self.assertEqual(x['c_previous_late3f_seconds'],35.)
        self.assertEqual(x['c_previous_late3f_minus_race_mean'],-1.)
        self.assertEqual(x['c_previous_late3f_rank_percentile'],1.)
        incomplete=f.build(self.target,history=self.past[:1])[0]
        self.assertIsNone(incomplete['c_previous_late3f_minus_race_mean'])

    def test_C_missing_other_runner_is_not_silently_ignored(self):
        changed=deepcopy(self.past);changed[-1]['late3f_raw']='999';changed[-1]['anomaly']='4'
        x=f.build(self.target,history=changed)[0]
        self.assertIsNone(x['c_previous_late3f_rank_percentile'])
        self.assertEqual(x['c_previous_late3f_seconds'],35.)

    def test_C_tied_times_have_average_ranks(self):
        changed=deepcopy(self.past);changed[1]['late3f_raw']='350'
        x=f.build(self.target,history=changed)[0]
        self.assertEqual(x['c_previous_late3f_rank_percentile'],.75)

    def test_C_obstacle_one_furlong_is_not_used_as_3F(self):
        changed=deepcopy(self.past)
        for r in changed:r['race_type']='18'
        self.assertIsNone(f.build(self.target,history=changed)[0]['c_previous_late3f_seconds'])

    def test_adjacent_and_additional_margins_are_not_winner_gap(self):
        changed=deepcopy(self.past);changed[1].update(margin1='112',margin2='2  ',margin3='3  ')
        x=f.build(self.target,history=changed)[1]
        self.assertEqual(x['c_previous_adjacent_margin_lengths'],1.5)
        self.assertEqual(x['c_previous_additional_margin_present'],1.)
        for token in ('H  ','K  ','T  ','Z  '):self.assertIsNone(f.MARGINS[token])

    def test_D_population_includes_other_horses(self):
        target=source(date(2024,11,15),ids=['OTHER00001','OTHER00002','OTHER00003'])
        x=f.build(target,history=self.past)[0]
        self.assertEqual(x['d_jockey_starts_365d'],3)
        self.assertAlmostEqual(x['d_jockey_smoothed_win_365d'],1/3)
        self.assertEqual(x['d_trainer_starts_365d'],3)

    def test_D_cancelled_runner_is_not_a_start_but_DNF_is(self):
        changed=deepcopy(self.past)
        changed[1].update(anomaly='1',finish_position='00')
        changed[2].update(anomaly='4',finish_position='00')
        x=f.build(self.target,history=changed)[0]
        self.assertEqual(x['d_jockey_starts_365d'],2)
        self.assertAlmostEqual(x['d_jockey_smoothed_win_365d'],.5)

    def test_D_actual_historical_jockey_is_counted_after_revision(self):
        changed=deepcopy(self.past)
        for r in changed:r['previous_jockey_revision']='00002'
        self.assertEqual(f.build(self.target,history=changed)[0]['d_jockey_starts_365d'],3)

    def test_D_window_expires(self):
        old=source(date(2023,11,1));x=f.build(self.target,history=old)[0]
        self.assertEqual(x['d_jockey_starts_365d'],0)

    def test_optional_missing_values_do_not_change_eligibility(self):
        first=f.Builder();second=f.Builder()
        changed=deepcopy(self.target)
        for r in changed:
            for k in ('grade','condition_4','late3f_raw','corner4','jockey','trainer','actual_runners'):r[k]=None
        _,a=first.transform_day(self.target);_,b=second.transform_day(changed)
        self.assertEqual([r['eligible'] for r in a],[r['eligible'] for r in b])


class OffsetAndContractsTests(unittest.TestCase):
    def frame(self,start,n):
        import pandas as pd
        rows=[]
        for j in range(n):
            day=start+timedelta(days=j)
            for i,q in enumerate((.5,.3,.2),1):
                rows.append(dict(entry_key=f'{day.year}{day:%m%d}05010101{i:02d}',date=day.isoformat(),
                    winner_label=int(i==1),market_probability=q,market_rank=float(i),market_log_odds=1.,market_inverse_sum=1.25))
        return pd.DataFrame(rows)

    def test_logit_and_sigmoid_roundtrip(self):
        import numpy as np
        q=np.array([.01,.2,.79]);self.assertTrue(np.allclose(m.sigmoid(m.offset(q)),q,atol=1e-15))

    def test_zero_residual_reproduces_market_and_probability_sum(self):
        import numpy as np
        x=self.frame(date(2024,1,1),1);spec=c.spec(anchor=True,ability=False)
        booster=MagicMock();booster.predict.return_value=np.zeros(3)
        bundle={'metadata':{'feature_spec':spec,'categories':{},'calibration':{'temperature':1.}},'boosters':{'win':booster}}
        p=m.predict(bundle,x)
        self.assertTrue(np.allclose(p,x.market_probability.to_numpy(),atol=1e-15))
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertTrue(booster.predict.call_args.kwargs['raw_score'])

    def test_real_small_anchor_save_reload_keeps_offset(self):
        import numpy as np
        train=self.frame(date(2024,6,1),20);valid=self.frame(date(2024,9,8),5);cal=self.frame(date(2024,11,8),5)
        split={'validation_end':'2024-10-31','evaluation':[('2025-01-01','2025-09-30')]}
        spec=c.spec(anchor=True,ability=False)
        b=m.fit(train,valid,cal,spec,split=split,parameters=c.plan()['anchor_parameters'])
        # No valid split with this fictional sample/minimum leaf size: residual must remain zero.
        self.assertTrue(np.allclose(m.predict(b,cal),cal.market_probability.to_numpy(),atol=1e-15))
        with tempfile.TemporaryDirectory() as td,patch.object(c,'ROOT',Path(td)):
            d,_=m.save(b,Path(td));loaded=m.load(d)
            self.assertTrue(np.allclose(m.predict(b,cal),m.predict(loaded,cal),atol=1e-15))
            self.assertEqual(loaded['metadata']['feature_spec']['offset']['scale'],'logit_normalized_inverse_odds')

    def test_old_loader_is_preserved(self):
        with patch.object(safety,'read_json',return_value={'format_version':1}),patch.object(m.legacy,'load_models',return_value={'old':True}) as loader:
            self.assertEqual(m.load(Path('fake'),root=Path('fake')),{'old':True})
            loader.assert_called_once()

    def test_changed_feature_order_is_rejected(self):
        value=c.spec(('A',));value['features']=list(reversed(value['features']))
        with self.assertRaises(safety.MLFailure):c.validate_spec(value)

    def test_market_anchor_never_contains_best242_predictions(self):
        spec=c.spec(('A','B'),anchor=True,history=True)
        self.assertFalse(any('best242' in name for name in spec['features']))
        self.assertEqual(spec['offset']['temperature'],1.)

    def test_sealed_period_is_rejected_before_SQL(self):
        with self.assertRaises(safety.MLFailure):
            io.source_query({},date(2026,1,1),date(2026,9,30))
        with self.assertRaises(safety.MLFailure):
            market_data.load(Path('fake'),date(2026,1,1),date(2026,9,30))

    def test_market_asof_bound_is_one_minute_before_decision(self):
        r=('2025','0301','05','01','01','01')
        sql,params=market_data.asof_query([r],{r:datetime(2025,3,1,15,tzinfo=core.JST)})
        self.assertIn('03011449',params)
        self.assertNotIn('03011450',params)
        self.assertNotIn('final_odds',sql)

    def test_yearly_splits_have_seven_day_separation(self):
        from jra_racing_ai.research.experiment import annual_split
        for year in (2023,2024,2025):
            s=annual_split(year)
            self.assertGreater((date.fromisoformat(s['validation_start'])-date.fromisoformat(s['train_end'])).days,7)
            self.assertGreater((date.fromisoformat(s['calibration_start'])-date.fromisoformat(s['validation_end'])).days,7)
        with self.assertRaises(safety.MLFailure):annual_split(2026)

    def test_weekly_bootstrap_sign_and_identical_zero(self):
        r=('2025','0301','05','01','01','01')
        p=[core.PredictionRace(r,{'01':.7,'02':.3})]
        ref=[core.PredictionRace(r,{'01':.5,'02':.5})]
        value=metrics.compare(p,ref,{r:'01'},confidence=.975)
        self.assertLess(value['race_week_cluster']['ci_high'],0)
        self.assertEqual(value['race_week_cluster']['confidence'],.975)
        self.assertEqual(metrics.compare(p,p,{r:'01'})['race_week_cluster']['ci_high'],0.)


class FreezeAndCohortTests(unittest.TestCase):
    def test_annual_candidate_retention_requires_two_years(self):
        from jra_racing_ai.research.report import retention
        comp={'difference_new_minus_baseline':-.01,'interpretation':'mean_improved_uncertainty_remains'}
        self.assertFalse(retention(comp,[-.1,.01,.01])['retain_development_candidate'])
        result=retention(comp,[-.1,-.01,.01])
        self.assertTrue(result['retain_development_candidate'])
        self.assertFalse(result['formal_improvement_certified'])

    def test_retention_does_not_use_hit_rate_or_profit(self):
        from jra_racing_ai.research.report import retention
        comp={'difference_new_minus_baseline':.001,'interpretation':'mean_worse_uncertainty_remains'}
        self.assertFalse(retention(comp,[-.01,-.01,1.])['retain_development_candidate'])

    def test_final_plan_is_locked_and_bonferroni_adjusted(self):
        from jra_racing_ai.research.report import future_protocol
        value=future_protocol({'ability_candidate':'ABCD','anchor_candidate':'market_ABCD_history','market_control':'market_history'})
        self.assertEqual(value['confidence_each'],.975)
        self.assertEqual(value['primary_comparisons'],2)
        self.assertFalse(value['entry_point_currently_enabled'])
        self.assertEqual(value['calibration'],['2025-11-08','2025-12-24'])
        self.assertEqual(value['formal_improvement_requires'],'CI_upper_strictly_below_zero')

    def test_final_entry_point_stops_without_accessing_data(self):
        import importlib.util,io as streams,json
        from contextlib import redirect_stdout
        path=Path(__file__).resolve().parents[1]/'scripts/stage2_final.py'
        spec=importlib.util.spec_from_file_location('reserved_final',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with redirect_stdout(streams.StringIO()) as output:
            self.assertEqual(module.main(['--execute']),2)
        value=json.loads(output.getvalue())
        self.assertEqual(value['reason'],'final_evaluation_not_authorized')
        self.assertFalse(value['performed_io'])

    def test_development_CLI_does_not_accept_final_stage(self):
        from jra_racing_ai.research.cli import main
        import io as streams
        from contextlib import redirect_stderr
        with redirect_stderr(streams.StringIO()),self.assertRaises(SystemExit):main(['--stage','final'])

    def test_early_quote_missing_does_not_change_primary_roster(self):
        import pandas as pd
        from jra_racing_ai import stage1
        from jra_racing_ai.ml.market import MarketRace
        r=('2025','0301','05','01','01','01')
        quote=MarketRace(r,{'01':core.decimal('2'),'02':core.decimal('3'),'03':core.decimal('6')},
            'apd_sokuho_o1','1','pre_race',datetime(2025,3,1,14,49,tzinfo=core.JST))
        packet={'quotes':{'10':stage1._quote_document(quote),'30':None}}
        frame=pd.DataFrame({'entry_key':[''.join(r)+h for h in ('01','02','03')]})
        actual=market_data.augment(frame,{r:packet})
        self.assertEqual(len(actual),3)
        self.assertTrue((actual.market_history_missing==1).all())
        self.assertTrue(actual.market_log_probability_change.isna().all())
        self.assertAlmostEqual(actual.market_probability.sum(),1.)

    def test_current_final_odds_and_payout_keys_are_ignored(self):
        past=source(date(2024,11,1));target=source(date(2024,11,15));changed=deepcopy(target)
        for r in changed:r.update(final_odds=9999,payout=999999,current_finish='01',current_late3f='001')
        self.assertEqual(f.build(target,history=past),f.build(changed,history=past))

    def test_D_future_mounts_do_not_change_past_rates(self):
        history=source(date(2024,11,1));target=source(date(2024,11,15))
        more=history+source(date(2025,1,1),ids=['OTHER00001','OTHER00002','OTHER00003'])
        a=f.build(target,history=history);b=f.build(target,history=more)
        self.assertEqual([[r[k] for k in c.GROUPS['D']] for r in a],[[r[k] for k in c.GROUPS['D']] for r in b])

    def test_categories_are_fixed_from_training_only(self):
        import pandas as pd
        spec=c.spec(('B',));row={k:None for k in spec['features']};row['b_condition']='005'
        train=pd.DataFrame([row]);categories=m.schema(train,spec)
        unseen=dict(row,b_condition='999');matrix=m.matrix(pd.DataFrame([unseen]),spec,categories)
        self.assertEqual(matrix.b_condition.iloc[0],'__UNKNOWN__')
        self.assertNotIn('v:999',categories['b_condition'])

    def test_week_cluster_keeps_same_week_venues_together(self):
        races=[('2025',md,venue,'01','01','01') for md,venue in [('0301','05'),('0302','06'),('0308','05')]]
        new=[core.PredictionRace(r,{'01':.6,'02':.4}) for r in races]
        old=[core.PredictionRace(r,{'01':.5,'02':.5}) for r in races]
        result=metrics.compare(new,old,{r:'01' for r in races})
        self.assertEqual(result['race_week_cluster']['weeks'],2)

    def test_all_group_order_is_versioned_and_subsets_do_not_replace_legacy(self):
        self.assertEqual(c.spec(tuple('DCBA'))['features'],c.spec(tuple('ABCD'))['features'])
        self.assertEqual(c.spec(('A',))['features'][:len(old_features.FEATURES)],list(old_features.FEATURES))


if __name__=='__main__':unittest.main()
