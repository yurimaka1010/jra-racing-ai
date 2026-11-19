"""Synthetic/mocked Stage 7a tests: no real database, labels or model fitting."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import date,datetime,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
from jra_racing_ai.ml import safety
from jra_racing_ai.research import contracts as rc,model as hm
from jra_racing_ai.win_sim import core
from jra_racing_ai.stage7 import contracts as c,store as s,inputs,sources,survey,planning,forward,evaluation,cli

RACE=('2025','0105','05','01','01','01')
FUTURE=('2026','1024','05','04','06','01')
T=core.instant('2025-01-05T10:00:00+09:00')

def row(stamp='01050959',**extra):
    return {'race_key':list(RACE),'record_type':'O1','status':'1','sale':'7','payload':'0100200102004002'+' '*8*26,
        'published':stamp,'source_table':'apd_sokuho_o1','oversized':False,'unsafe_version':False,**extra}

def quote(race=RACE,stamp='2025-01-05T09:59:00+09:00'):
    return {'race_key':list(race),'odds':{'01':2.,'02':4.},'source':'synthetic','category':'1','basis':'pre_race_publication_time',
        'published_at':stamp,'minutes_before_start':4.,'cancelled_outside_roster':0}

def forward_plan():
    return {'created_at':'2026-10-21T07:00:00+09:00','start_date':'2026-10-24','end_date':'2027-04-18',
        'references':{str(c.MODEL/'model.json'):'fake_meta_hash',str(c.MODEL/'win.lgb.txt'):'fake_model_hash'}}

def registered():
    return {'race_key':list(FUTURE),'roster':['01','02'],'scheduled_start':'2026-10-24T10:10:00+09:00',
        'reason':None,'registered_at':'2026-10-24T08:00:00+09:00'}

class TimingTests(unittest.TestCase):
    def test_T3_minute_end_and_exact_120_second_age(self):
        for stamp in ('01050959','01050958'):
            q,why=inputs.select(RACE,[row(stamp)],['01','02'],T,120);self.assertIsNone(why);self.assertIsNotNone(q)

    def test_T3_rejects_older_than120_without_relaxation(self):
        q,why=inputs.select(RACE,[row('01050957')],['01','02'],T,120)
        self.assertIsNone(q);self.assertEqual(why,'odds_stale')

    def test_T10_and_T30_allow_exact300_seconds(self):
        self.assertIsNotNone(inputs.select(RACE,[row('01050955')],['01','02'],T,300)[0])
        self.assertIsNone(inputs.select(RACE,[row('01050954')],['01','02'],T,300)[0])

    def test_minute_start_before_cutoff_is_insufficient(self):
        self.assertIsNone(inputs.select(RACE,[row()],['01','02'],T-timedelta(microseconds=1),120)[0])

    def test_observed_path_still_requires_minute_end(self):
        r=row(first_seen_at='2025-01-05T09:59:30+09:00')
        self.assertIsNone(inputs.select(RACE,[r],['01','02'],T-timedelta(seconds=1),300,require_seen=True)[0])

    def test_late_receipt_cannot_backfill_publication_before_cutoff(self):
        late=row(first_seen_at='2025-01-05T10:00:00.000001+09:00')
        self.assertEqual(inputs.select(RACE,[late],['01','02'],T,300,require_seen=True),(None,'no_on_time_observation'))
        late['first_seen_at']=T.isoformat()
        self.assertIsNotNone(inputs.select(RACE,[late],['01','02'],T,300,require_seen=True)[0])

    def test_bad_latest_and_duplicate_latest_never_use_older_quote(self):
        for rows in ([row(payload='BROKEN'),row('01050958')],[row(),row(),row('01050958')]):
            self.assertIsNone(inputs.select(RACE,rows,['01','02'],T,300)[0])

    def test_unknown_category_and_roster_mismatch_rejected(self):
        self.assertIsNone(inputs.select(RACE,[row(status='0')],['01','02'],T,120)[0])
        self.assertIsNone(inputs.select(RACE,[row()],['01','02','03'],T,120)[0])
        self.assertIsNone(inputs.select(RACE,[row(race_key=list(FUTURE))],['01','02'],T,120)[0])

    def test_clock_future_before_observation_rejected(self):
        self.assertEqual(inputs.select(RACE,[row(first_seen_at='2025-01-05T09:58:00+09:00')],['01','02'],T,300,require_seen=True)[1],
            'odds_observed_before_publication')

    def test_SE_HR_results_and_final_odds_not_in_query(self):
        sql,args=inputs.query([RACE],{RACE:'2025-01-05T10:03:00+09:00'})
        for text in ('jvd_se','jvd_hr','chakujun','SELECT s.*','UPDATE','DELETE'):self.assertNotIn(text,sql)
        self.assertIn("s.data_kubun::text='1'",sql);self.assertIn('01050958',args);self.assertIn('01050959',args)
        self.assertEqual(args[-2:],('01050958','01050959'))

    def test_both_query_branches_block_sealed_period(self):
        for historic in (True,False):
            for stamp in ('0101','0930'):
                race=('2026',stamp,'05','01','01','01')
                with self.assertRaises(safety.MLFailure):inputs.query([race],{race:'2026-01-01T10:00:00+09:00'},historical=historic)

    def test_card_query_projects_no_result_fields_and_requires_future(self):
        for table,fields in (('jvd_ra',forward.publication.RA_FIELDS),('jvd_se',forward.SE_FIELDS)):
            sql,_=forward.cards_query(date(2026,10,24),table,fields)
            self.assertNotIn('chakujun',sql);self.assertNotIn('payout',sql);self.assertNotIn('time',sql)
        with self.assertRaises(safety.MLFailure):forward.cards_query(date(2026,9,30),'jvd_ra',forward.publication.RA_FIELDS)

class ProjectionAndInputTests(unittest.TestCase):
    def test_projection_never_JSON_decodes_outcome_or_raw_O1(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'synthetic.json'
            record={'race_key':list(RACE),'scheduled_start':'2025-01-05T10:03:00+09:00','schedule_asof_verified':False,
                'quotes':{'10':quote(),'30':None},'failures':{'30':'missing'},'outcome':{'secret_result':'DO_NOT_PARSE'},'raw_selected_O1':{'x':'IGNORE'}}
            text=json.dumps({'records':[record]},indent=2).replace('"DO_NOT_PARSE"','not_valid_json').replace('"IGNORE"','not_valid_json')
            p.write_text(text,encoding='utf-8');values=list(sources.packets(p))
            self.assertEqual(len(values),1);self.assertEqual(set(values[0]),sources.ALLOWED)

    def test_projection_stops_before_post_boundary_values(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'synthetic.json'
            rec={'race_key':['2025','1001','05','01','01','01'],'scheduled_start':'PRIVATE_BAD_CLOCK','schedule_asof_verified':False,
                'quotes':{},'failures':{},'outcome':{'secret':'UNPARSED'}}
            p.write_text(json.dumps({'records':[rec]},indent=2).replace('"PRIVATE_BAD_CLOCK"','not_valid_json'),encoding='utf-8')
            self.assertEqual(list(sources.packets(p)),[])

    def test_H8_uses_same_formula_and_T30_missing_does_not_drop_horses(self):
        frame=inputs.features(RACE,quote(),None)
        self.assertEqual(len(frame),2);self.assertEqual(set(frame.columns),{'entry_key','date',*c.H_FIELDS})
        self.assertTrue(frame.market_history_missing.eq(1).all());self.assertTrue(frame.market_log_probability_change.isna().all())
        self.assertAlmostEqual(frame.market_probability.sum(),1.)

    def test_H8_uses_actual_publication_elapsed_not_nominal27(self):
        earlier=quote(stamp='2025-01-05T09:31:00+09:00');current=quote(stamp='2025-01-05T09:59:00+09:00')
        out=inputs.features(RACE,current,earlier);self.assertTrue(out.market_elapsed_minutes.eq(28.).all())
        self.assertTrue(out.market_log_probability_change.eq(0.).all())

    def test_zero_residual_recreates_market_exactly_without_extra_temperature(self):
        f=inputs.features(RACE,quote(),None);spec=rc.spec(anchor=True,ability=False,history=True)
        booster=MagicMock();booster.predict.return_value=np.zeros(len(f))
        bundle={'metadata':{'feature_spec':spec,'categories':{},'calibration':{'temperature':1.}},'boosters':{'win':booster}}
        np.testing.assert_allclose(hm.predict(bundle,f),f.market_probability.to_numpy(),atol=1e-15)
        self.assertTrue(booster.predict.call_args.kwargs['raw_score'])

    def test_current_results_cannot_change_generated_H8(self):
        a=inputs.features(RACE,quote(),None);changed=quote();changed.update(winner_label=1,final_odds=999,finish=1,payout=1000)
        pd.testing.assert_frame_equal(a,inputs.features(RACE,changed,None))

    def test_fixed_year_roles_keep_gaps_and_no_2026_inputs(self):
        self.assertIn('2023_train',sources.roles(date(2022,8,31)))
        self.assertNotIn('2023_train',sources.roles(date(2022,9,1)))
        self.assertNotIn('2023_early',sources.roles(date(2022,9,7)))
        self.assertIn('2023_calibration',sources.roles(date(2022,11,8)))
        self.assertEqual(sources.roles(date(2026,1,1)),[])

    def test_C_not_reduced_for_missing_early_quote(self):
        target={'race_key':list(RACE),'roster':['01','02'],'scheduled_start':'2025-01-05T10:03:00+09:00',
            'quotes':{'10':quote(stamp='2025-01-05T09:52:00+09:00'),'30':None},'roles':['2025_evaluation'],
            'in_B':True,'schedule_asof_verified':False}
        result,f=survey.assess(target,[row()]);self.assertTrue(result['in_C']);self.assertTrue(result['T30_missing']);self.assertEqual(len(f),2)

class RecorderTests(unittest.TestCase):
    def prepared(self,stamps,change=None):
        reg=registered();current=deepcopy(reg)
        if change:current.update(change)
        observations=[row('10240959',race_key=list(FUTURE),first_seen_at='2026-10-24T09:59:50+09:00')]
        writes={};events=[];clock=MagicMock(side_effect=[core.instant(x) for x in stamps])
        patches=[patch.object(forward,'observations',return_value=observations),patch.object(s,'sha',return_value='fictional_hash'),
            patch.object(s,'verify_refs'),patch.object(s,'write',side_effect=lambda p,v:writes.update({p.name:deepcopy(v)})),
            patch.object(s,'event',side_effect=lambda *a,**k:events.append((a,k))),
            patch.object(inputs,'predict',return_value=({'H':{'01':.6,'02':.4},'raw_market_T10':{'01':2/3,'02':1/3}},inputs.features(FUTURE,quote(FUTURE,'2026-10-24T09:59:00+09:00'),None)))]
        return reg,current,writes,events,clock,patches

    def invoke(self,stamps,change=None):
        from contextlib import ExitStack
        reg,current,writes,events,clock,patches=self.prepared(stamps,change)
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            value=forward.decide(Path('fictional_missing_folder'),reg,current,forward_plan(),{},clock=clock)
        return value,writes,events

    def test_input_fixed_after_cutoff_and_saved_before_grace_deadline(self):
        stamps=[f'2026-10-24T10:00:0{i}+09:00' for i in range(1,5)];value,writes,_=self.invoke(stamps)
        self.assertEqual(value['status'],'saved');self.assertFalse(writes['prediction.json']['result_accessed'])
        self.assertEqual(writes['prediction.json']['temperature'],1.)

    def test_before_cutoff_no_prediction_or_decision_is_saved(self):
        value,writes,_=self.invoke(['2026-10-24T09:59:59+09:00'])
        self.assertEqual(value['status'],'pending');self.assertEqual(writes,{})

    def test_after_save_deadline_is_missing_without_prediction(self):
        value,writes,_=self.invoke(['2026-10-24T10:01:01+09:00'])
        self.assertEqual(value['reason'],'save_deadline_missed');self.assertNotIn('prediction.json',writes)

    def test_slow_calculation_is_missing(self):
        value,writes,_=self.invoke(['2026-10-24T10:00:01+09:00','2026-10-24T10:01:01+09:00'])
        self.assertEqual(value['reason'],'calculation_deadline_missed');self.assertNotIn('prediction.json',writes)

    def test_slow_write_and_slow_receipt_not_success(self):
        value,_,events=self.invoke(['2026-10-24T10:00:01+09:00','2026-10-24T10:00:02+09:00','2026-10-24T10:01:01+09:00','2026-10-24T10:01:02+09:00'])
        self.assertEqual(value['status'],'missing');self.assertTrue(events)

    def test_schedule_and_roster_change_both_cause_missing(self):
        for change in ({'scheduled_start':'2026-10-24T10:11:00+09:00'},{'roster':['01']}):
            value,writes,_=self.invoke(['2026-10-24T10:00:01+09:00'],change)
            self.assertEqual(value['reason'],'schedule_or_roster_changed');self.assertNotIn('prediction.json',writes)

    def test_final_category_change_is_not_clock_or_roster_change(self):
        reg=registered();current=dict(reg,reason='not_pre_race_schedule',RA_status='7')
        self.assertTrue(forward.same_saved_clock_and_roster(reg,current));self.assertFalse(forward.same_card(reg,current))

    def test_first_seen_never_backdated_or_overwritten(self):
        with TemporaryDirectory() as td:
            folder=Path(td)
            def mkdir(p):p.mkdir(parents=True,exist_ok=True);return p
            def write(p,v):
                with p.open('x',encoding='utf-8') as f:json.dump(v,f)
            with patch.object(s,'mkdir',side_effect=mkdir),patch.object(s,'write',side_effect=write):
                forward.capture(folder,row(),T);forward.capture(folder,row(),T-timedelta(minutes=3))
            records=list((folder/'observations').glob('*.json'));self.assertEqual(len(records),1)
            self.assertEqual(json.loads(records[0].read_text())['first_seen_at'],T.isoformat())

    def test_duplicate_lock_cannot_overwrite_existing_recorder(self):
        with TemporaryDirectory() as td,patch.object(s.safety,'local_path',side_effect=lambda p,**_:p):
            folder=Path(td)
            with s.lock(folder):
                with self.assertRaises(safety.MLFailure):
                    with s.lock(folder):pass
            self.assertFalse((folder/'recorder.lock').exists())

    def test_date_guard_no_historical_replay_or_sealed_window(self):
        for race in (RACE,('2026','0930','05','01','01','01')):
            with self.assertRaises(safety.MLFailure):c.future(race,forward_plan(),core.instant('2026-10-24T08:00:00+09:00'))

    def test_0_fit_entry_has_no_training_command(self):
        with patch('sys.stderr'),self.assertRaises(SystemExit):cli.main(['fit','--run-id','a'*32])

    def test_full_tick_captures_first_seen_then_commits_and_ignores_final_status_change(self):
        from contextlib import nullcontext
        import hashlib
        now=[core.instant('2026-10-24T09:59:50+09:00')]
        reg=registered();reg.pop('registered_at');reg['RA_status']='2';card=[reg]
        spec=rc.spec(anchor=True,ability=False,history=True);booster=MagicMock();booster.predict.return_value=np.zeros(2)
        bundle={'metadata':{'feature_spec':spec,'categories':{},'calibration':{'temperature':1.}},'boosters':{'win':booster}}
        raw=[]
        for minutes,stamp in ((10,'10240959'),(30,'10240939')):
            v=row(stamp,race_key=list(FUTURE));v.update(dict(zip(forward.RACE_KEY,FUTURE)),minutes=minutes)
            raw.append(v)
        db=MagicMock();db.aggregate.return_value=raw
        def mkdir(p):p.mkdir(parents=True,exist_ok=True);return p
        def write(p,value):
            with p.open('x',encoding='utf-8') as f:json.dump(value,f,allow_nan=False)
        with TemporaryDirectory() as td:
            root=Path(td);day=mkdir(root/'forward/2026-10-24');mkdir(day/'races');(root/'forward_plan.json').write_text('{}')
            with patch.object(c,'clock',side_effect=lambda:now[0]),patch.object(s,'mkdir',side_effect=mkdir), \
                    patch.object(s,'read',side_effect=lambda p:json.loads(p.read_text(encoding='utf-8'))),patch.object(s,'write',side_effect=write), \
                    patch.object(s,'sha',side_effect=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()),patch.object(s,'verify_refs'), \
                    patch.object(forward,'live_cards',side_effect=lambda *a:{FUTURE:deepcopy(card[0])}), \
                    patch.object(publication_module:=forward.publication,'_columns',return_value={}),patch.object(s,'event'):
                first=forward.tick(day,forward_plan(),bundle,lambda:nullcontext(db))
                self.assertEqual(first['status_counts'],{'pending':1})
                now[0]=core.instant('2026-10-24T10:00:05+09:00')
                second=forward.tick(day,forward_plan(),bundle,lambda:nullcontext(db))
                self.assertEqual(second['status_counts'],{'saved':1})
                path=day/'races'/''.join(FUTURE);before=(path/'prediction.json').read_bytes()
                values=forward.frozen_records(root,forward_plan());self.assertEqual(len(values),1)
                self.assertAlmostEqual(sum(values[0]['probabilities']['H'].values()),1.)
                self.assertEqual(values[0]['quote_T10']['first_seen_at'],'2026-10-24T09:59:50+09:00')
                self.assertIsNone(values[0]['quote_T30'])
                now[0]=core.instant('2026-10-24T10:20:00+09:00');card[0]=dict(reg,reason='not_pre_race_schedule',RA_status='7')
                third=forward.tick(day,forward_plan(),bundle,lambda:nullcontext(db))
                self.assertEqual(third['status_counts'],{'saved':1});self.assertEqual(before,(path/'prediction.json').read_bytes())

class EvaluationAndPlanningTests(unittest.TestCase):
    def test_interim_labels_blocked_before_fixed_period_ends(self):
        with self.assertRaises(safety.MLFailure):evaluation.evaluate([],{},forward_plan(),core.instant('2027-04-18T23:00:00+09:00'))

    def test_future_adapter_ties_race_weights_and_global_auc(self):
        plan=dict(forward_plan(),end_date='2026-11-01');records=[];results={}
        for i,n in enumerate((2,3)):
            race=FUTURE[:-1]+(f'{i+1:02d}',);roster=[f'{j+1:02d}' for j in range(n)];p=dict.fromkeys(roster,1/n)
            records.append({'race_key':list(race),'roster':roster,'probabilities':{'H':p,'raw_market_T10':p},'save_completed_at':'2026-10-24T10:00:00+09:00'})
            results[race]={'status':'normal_final','horses':roster,'winners':['01']}
        out=evaluation.evaluate(records,results,plan,core.instant('2026-11-02T08:00:00+09:00'),repetitions=12)
        self.assertAlmostEqual(out['H']['within_race_auc'],.5);self.assertAlmostEqual(out['H']['top1_accuracy_percent'],100*(1/2+1/3)/2)
        self.assertEqual(out['difference_H_minus_market'],0.);self.assertTrue(out['bootstrap']['all_models_and_metrics_share_draws'])

    def test_power_duration_unstable_estimate_uses26_resource_cap(self):
        rows=[]
        for year in c.YEARS:
            for day in (f'{year}-01-07',f'{year}-01-14'):
                for horse,y in (('01',1),('02',0)):
                    rows.append({'entry_key':day.replace('-','')+'05010101'+horse,'date':day,'winner_label':y,'H':.5,'raw_market':.5})
        out=planning.duration(pd.DataFrame(rows),repetitions=30)
        self.assertEqual(out['chosen_weeks'],26);self.assertFalse(out['adequate_estimate_within_26']);self.assertFalse(out['future_power_guaranteed'])

    def test_future_metrics_do_not_relax_old_date_guard(self):
        from jra_racing_ai.stage5 import contracts as old
        with self.assertRaises(safety.MLFailure):old.guard(date(2026,10,24),date(2026,10,24))

    def test_plan_and_targets_verified_before_survey_DB_connect(self):
        session=MagicMock()
        with patch.object(planning,'verify',side_effect=safety.MLFailure('artifact_invalid')):
            with self.assertRaises(safety.MLFailure):survey.run(Path('fictional_missing_folder'),session)
        session.assert_not_called()

    def test_completed_prediction_bytes_without_receipt_are_not_evaluable(self):
        with TemporaryDirectory() as td:
            folder=Path(td);(folder/'prediction.json').write_text('{}',encoding='utf-8')
            self.assertFalse(forward.eligible_record(folder,forward_plan()))

    def test_post_save_change_veto_never_overwrites_original_prediction(self):
        with TemporaryDirectory() as td:
            folder=Path(td);(folder/'decision.json').touch();(folder/'events').mkdir()
            (folder/'events/change.json').write_text('{}',encoding='utf-8')
            with patch.object(s,'read',return_value={'status':'saved'}),patch.object(s,'write') as write:
                self.assertFalse(forward.eligible_record(folder,forward_plan()));write.assert_not_called()

    def test_same_counterfactual_draws_are_paired_not_independent(self):
        from jra_racing_ai.stage5.metrics import WeekBootstrap
        y=np.array([1,0,1,0]);p=np.array([.6,.4,.4,.6]);r=np.array([0,0,1,1]);w=np.array(['2026-10-19']*2+['2026-10-26']*2)
        boot=WeekBootstrap(w,repetitions=15);_,a=boot.measure(y,p,r);_,b=boot.measure(y,p,r)
        self.assertTrue(np.array_equal(a['within_race_auc']-b['within_race_auc'],np.zeros(15)))

if __name__=='__main__':unittest.main()
