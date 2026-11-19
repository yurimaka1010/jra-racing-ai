"""Zero-fit synthetic tests. No actual rows, DB driver, task registration or network."""
from datetime import date,datetime,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy
import unittest
import numpy as np
from jra_racing_ai.pedigree_forward.measurement import calendar_bootstrap,monday
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.pedigree_forward import contracts as c,protocol,recorder,evaluation
from jra_racing_ai.d39_enrichment.features import Builder as Ability
from jra_racing_ai.pedigree_research.features import Builder as Blood
from jra_racing_ai.pedigree_research import data as pdata
from jra_racing_ai.win_sim.core import JST
from jra_racing_ai.pedigree_forward.timing import quote_valid


def entries():
    return [dict(kaisai_nen='2026',kaisai_tsukihi='1031',keibajo_code='05',kaisai_kai='01',kaisai_nichime='01',
        race_bango='01',horse_number=f'{i:02d}',horse_id=f'{i:010d}',surface='11',distance='1600',
        declared_runners='03',frame=str(i),age='03',sex='1',weight_raw='560',grade=' ',condition_3='703',
        race_type='12',weight_type='1',jockey='00001',trainer='00002') for i in (1,2,3)]


def setup_models():
    from jra_racing_ai.d39_enrichment.contracts import spec as spec47
    class Booster:
        def predict(self,frame,**kwargs):return np.asarray([.1,.3,.6])
    bundles={};freeze={'models':{},'code_commit_id':'source-sha256:test'}
    for name in c.NAMES:
        spec=spec47() if name=='reference47' else pdata.spec(name=='pedigree45')
        cats={k:['__MISSING__','__UNKNOWN__'] for k in spec['categorical_features']}
        meta={'feature_spec':spec,'features':spec['features'],'categories':cats,'model_id':c.MODEL_IDS[name],
              'temperature':c.TEMPERATURE[name]}
        bundles[name]=(Booster(),meta)
        freeze['models'][name]={'features':spec['features'],
            'input_types':{f:'category' if f in cats else 'float64' for f in spec['features']}}
    return bundles,freeze


def context():return {'ability':Ability({},None),'blood':Blood({}),'info':{'lag_days':7,'cutoff':'2026-10-24'}}


def registration():
    return {'race_key':['2026','1031','05','01','01','01'],'scheduled_start':'2026-10-31T10:00:00+09:00',
            'input_cutoff':'2026-10-31T09:50:00+09:00','roster':['01','02','03'],'reason':None}


def quote(prices=None):
    return {'odds':prices or {'01':2.,'02':3.,'03':4.},'published_at':'2026-10-31T09:48:00+09:00',
            'first_seen_at':'2026-10-31T09:48:30+09:00'}


class DesignTests(unittest.TestCase):
    def test_calendar_week_origin(self):
        self.assertEqual(monday(date(2026,10,31)),date(2026,10,26))

    def test_empty_weeks_stay_in_calendar(self):
        calendar=[str(date(2024,1,1)+timedelta(weeks=i)) for i in range(21)]
        b=calendar_bootstrap([calendar[0],calendar[0],calendar[20],calendar[20]],calendar,repetitions=30)
        self.assertEqual(b.counts.shape,(30,21))
        np.testing.assert_array_equal(b.codes,[0,0,20,20])

    def test_identical_pair_has_zero_difference_SE(self):
        calendar=[str(date(2024,1,1)+timedelta(weeks=i)) for i in range(21)]
        weeks=[w for w in calendar for _ in range(2)]
        b=calendar_bootstrap(weeks,calendar,repetitions=50)
        y=np.tile([1,0],21);p=np.tile([.6,.4],21);codes=np.repeat(np.arange(21),2)
        _,a=b.measure(y,p,codes);_,z=b.measure(y,p,codes)
        for k in a:np.testing.assert_array_equal(a[k]-z[k],np.zeros(50))

    def test_unknown_week_rejected(self):
        with self.assertRaises(MLFailure):calendar_bootstrap(['2030-01-01'],['2024-01-01']*21)

    def test_formal_window_if_ready_before_first_day(self):
        a,b=protocol.formal_window(datetime(2026,10,26,10,tzinfo=JST))
        self.assertEqual((a,b),(date(2026,10,26),date(2027,3,21)))

    def test_late_registration_uses_next_full_week(self):
        a,b=protocol.formal_window(datetime(2026,10,31,11,tzinfo=JST),datetime(2026,10,31,9,35,tzinfo=JST))
        self.assertEqual((a,b),(date(2026,11,2),date(2027,3,28)))

    def test_current_outcomes_cannot_change_input_frames(self):
        a=entries();b=copy.deepcopy(a)
        for row in b:row.update(finish_position='01',anomaly='7',time_raw='1234',odds=9999,payout=100000)
        x=recorder.frames(context(),a);z=recorder.frames(context(),b)
        import pandas as pd
        for name in c.NAMES:pd.testing.assert_frame_equal(x[name],z[name])

    def test_blood_features_join_by_key_not_input_row_order(self):
        import pandas as pd
        a=recorder.frames(context(),entries());b=recorder.frames(context(),list(reversed(entries())))
        for name in c.NAMES:pd.testing.assert_frame_equal(a[name],b[name])

    def test_invalid_registration_number_refused(self):
        a=entries();a[0]['horse_id']='0000000000'
        with self.assertRaises(MLFailure):recorder.frames(context(),a)

    def test_no_odds_does_not_block_forecasts_but_blocks_R1(self):
        bundles,freeze=setup_models()
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t,patch.object(c,'ROOT',Path(t)):
            snapshot={'payload':entries(),'path':'synthetic','sha256':'test','first_seen_at':'2026-10-31T09:49:00+09:00'}
            record=recorder.calculate(Path(t),registration(),snapshot,None,bundles,context(),freeze,
                calculated_at=datetime(2026,10,31,9,50,tzinfo=JST))
            self.assertTrue(record['P1_prediction_side_complete']);self.assertFalse(record['P2_P3_odds_ready'])
            self.assertEqual(sum(r['R1_virtual_stake_yen'] for r in record['rows']),0)
            for model in record['models'].values():self.assertAlmostEqual(sum(model['probability'].values()),1)

    def test_one_model_missing_keeps_other_forecasts_and_no_R1(self):
        bundles,freeze=setup_models();bundles['pedigree45']=None
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t,patch.object(c,'ROOT',Path(t)):
            snapshot={'payload':entries(),'path':'synthetic','sha256':'test','first_seen_at':'2026-10-31T09:49:00+09:00'}
            r=recorder.calculate(Path(t),registration(),snapshot,quote(),bundles,context(),freeze,
                calculated_at=datetime(2026,10,31,9,50,tzinfo=JST))
            self.assertEqual(r['models']['control39']['status'],'saved')
            self.assertEqual(r['models']['pedigree45']['status'],'missing')
            self.assertFalse(r['P1_prediction_side_complete'])
            self.assertEqual(sum(x['R1_virtual_stake_yen'] for x in r['rows']),0)

    def test_R1_one_top100_each_without_EV_filter(self):
        bundles,freeze=setup_models()
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t,patch.object(c,'ROOT',Path(t)):
            snapshot={'payload':entries(),'path':'synthetic','sha256':'test','first_seen_at':'2026-10-31T09:49:00+09:00'}
            r=recorder.calculate(Path(t),registration(),snapshot,quote({'01':1.1,'02':1.1,'03':1.1}),bundles,context(),freeze,
                calculated_at=datetime(2026,10,31,9,50,tzinfo=JST))
            for name in c.NAMES:
                stakes=[x for x in r['rows'] if x['model']==name and x['R1_virtual_stake_yen']]
                self.assertEqual(len(stakes),1);self.assertEqual(stakes[0]['horse_number'],'03')

    def test_deadline_missed_never_calls_predict(self):
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t,patch.object(c,'ROOT',Path(t)),patch.object(recorder,'calculate',side_effect=AssertionError):
            r=recorder.decide(Path(t),registration(),registration(),{},None,{}, {},clock=lambda:datetime(2026,10,31,9,52,tzinfo=JST))
            self.assertEqual(r['reason'],'save_deadline_missed')

    def test_saved_receipt_prevents_recalculation(self):
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t,patch.object(c,'ROOT',Path(t)),patch.object(recorder,'calculate',side_effect=AssertionError):
            c.write(Path(t)/'receipt.json',{'status':'saved','sentinel':1})
            r=recorder.decide(Path(t),{},None,{},None,{}, {})
            self.assertEqual(r['sentinel'],1)

    def test_before_cutoff_is_pending(self):
        with TemporaryDirectory(dir=c.BASE/'test-temp') as t:
            r=recorder.decide(Path(t),registration(),registration(),{},None,{}, {},clock=lambda:datetime(2026,10,31,9,49,tzinfo=JST))
            self.assertEqual(r['status'],'pending')

    def test_final_evaluation_blocked_until_period_closed(self):
        with patch.object(protocol,'load',return_value=({}, {'window_end_date':'2027-03-21'})):
            with self.assertRaises(MLFailure):protocol.require_closed(Path('test'),datetime(2027,3,21,23,59,tzinfo=JST))
            self.assertEqual(protocol.require_closed(Path('test'),datetime(2027,3,22,tzinfo=JST))['window_end_date'],'2027-03-21')

    def test_before_first_day_only_preflight(self):
        with patch.object(protocol,'load',return_value=({},{})),patch.object(c,'clock',return_value=datetime(2026,10,30,10,tzinfo=JST)), \
             patch.object(recorder,'preflight',return_value={'status':'ready'}),patch.object(recorder.history,'build',side_effect=AssertionError):
            self.assertEqual(recorder.run(Path('test'))['reason'],'before_first_record_day')

    def test_future_or_stale_quotes_refused(self):
        q=quote();self.assertTrue(quote_valid(q,['01','02','03'],registration()['input_cutoff']))
        for field,value in (('first_seen_at','2026-10-31T09:50:01+09:00'),('published_at','2026-10-31T09:50:00+09:00'),
                            ('published_at','2026-10-31T09:44:00+09:00')):
            bad=dict(q,**{field:value})
            self.assertFalse(quote_valid(bad,['01','02','03'],registration()['input_cutoff']))


class PopulationTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(r,anomaly='0',actual_runners='03',finish_position=f'{i:02d}') for i,r in enumerate(entries(),1)]
        record={'roster':['01','02','03'],'models':{n:{'status':'saved','probability':{'01':.5,'02':.3,'03':.2}} for n in c.NAMES},
                'quote_T10':quote({'01':2.,'02':4.,'03':8.}),'input_cutoff':registration()['input_cutoff']}
        return record,rows

    def test_normal_all_populations(self):
        record,rows=self.fixture();p=evaluation.population(record,rows)
        self.assertTrue(all(p[k] for k in ('P1','P2','P3')))

    def test_no_odds_is_only_P1(self):
        record,rows=self.fixture();record['quote_T10']=None;p=evaluation.population(record,rows)
        self.assertTrue(p['P1']);self.assertFalse(p['P2']);self.assertFalse(p['P3'])

    def test_abnormal_whole_race_excluded_from_P1_but_not_P3(self):
        record,rows=self.fixture();rows[1]['anomaly']='4';p=evaluation.population(record,rows)
        self.assertFalse(p['P1']);self.assertFalse(p['P2']);self.assertTrue(p['P3'])

    def test_cancellation_filter_happens_after_whole_race_exclusion(self):
        record,rows=self.fixture();rows[1]['anomaly']='1';p=evaluation.population(record,rows)
        self.assertFalse(p['P1']);self.assertEqual(p['P1_P2_remaining_probabilities'],{})
        self.assertTrue(p['P3'])

    def test_refund_and_losses(self):
        for flag in ('1','2','3'):
            r=evaluation.r1_outcome('01',[{'horse_number':'01','anomaly':flag}],None)
            self.assertEqual(r['status'],'refunded');self.assertEqual(r['profit_yen'],0)
        for flag in ('4','5'):
            r=evaluation.r1_outcome('01',[{'horse_number':'01','anomaly':flag}],None,payout_state='normal_confirmed')
            self.assertEqual(r['payout_yen'],0)

    def test_payout_never_from_decision_odds_and_special_pending(self):
        rows=[{'horse_number':'01','anomaly':'0','finish_position':'01'}]
        self.assertEqual(evaluation.r1_outcome('01',rows,{'01':230},payout_state='normal_confirmed')['payout_yen'],230)
        self.assertEqual(evaluation.r1_outcome('01',rows,{'01':230})['status'],'unsettled')

    def test_inverse_market_uses_same_roster(self):
        record,_=self.fixture();q=evaluation.market_probability(record,['01','02','03'])
        self.assertAlmostEqual(sum(q.values()),1);self.assertAlmostEqual(q['01'],4/7)

    def test_final_gate_uses_within_AUC_and_point_guardrails(self):
        rules={'thresholds':{'minimum_within_race_AUC_gain':.004,'race_LL_increase_max':.005,'Brier_increase_max':.0001}}
        self.assertTrue(evaluation.acceptance(.004,[.001,.01],.005,.0001,rules))
        self.assertFalse(evaluation.acceptance(.004,[-.001,.01],0,0,rules))
        self.assertFalse(evaluation.acceptance(.01,[.001,.02],.006,0,rules))


if __name__=='__main__':unittest.main()
