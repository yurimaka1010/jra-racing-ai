"""Fictitious exact-key HR and saved predictions; no real fits, network or data."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from jra_racing_ai.multiwager import lab as m
from jra_racing_ai.multiwager.outcomes import rank_reference
from jra_racing_ai.multiwager.probabilities import plackett_luce
from jra_racing_ai.ml.safety import MLFailure


def record(paid=3):
    return {'race_id':'2025030105010101','date':'2025-03-01','paid_positions':paid,
            'finishes':{'01':1,'02':2,'03':3,'04':4}}


def hr(paid=3):
    r={'status':'2','record_type':'HR','actual_runners':'04','oversized':False,
       'race_key':('2025','0301','05','01','01','01')}
    refs,_=rank_reference(record(paid)['finishes'],paid)
    for w,(_,n,aw,cw) in m.LAYOUTS.items():
        for flag in ('unestablished','special','refund'):r[w+'_'+flag]='0'
        winners=[k.replace('-','') for k,v in refs[w].items() if v]
        for i in range(1,n+1):
            values=(winners[i-1],'000000120','01'.zfill(cw)) if i<=len(winners) else (' '*aw,' '*9,' '*cw)
            for p,v in zip('abc',values):r[f'{w}_{i}{p}']=v
    if paid==2:r.update(place_3a='00',place_3b=' '*9,place_3c=' '*2)
    return r


def normal(w='place',rows=None,rec=None):
    return m.normalize_hr([hr()] if rows is None else rows,rec or record(),w,'a'*64)


class HRTargetTests(unittest.TestCase):
    def test_three_wagers_have_independent_complete_labels_not_confirmed_money(self):
        for w,n in [('place',3),('wide',3),('umaren',1)]:
            out=normal(w);self.assertEqual(out['target_status'],'confirmed')
            self.assertEqual(len(out['official_targets']),n);self.assertIsNone(out['payouts_yen_per_100'])
            self.assertEqual(out['money_status'],'unconfirmed')
            self.assertTrue(all(v=='000000120' for v in out['raw_amount_fields'].values()))

    def test_exact_two_payee_sentinel_not_third_winner(self):
        out=normal(rows=[hr(2)],rec=record(2));self.assertEqual(out['official_targets'],['01','02'])

    def test_two_payee_sentinel_with_amount_is_not_silently_unused(self):
        r=hr(2);r['place_3b']='000000070';self.assertEqual(normal(rows=[r],rec=record(2))['target_status'],'pending')

    def test_three_payee_zero_horse_is_special(self):
        r=hr();r.update(place_3a='00',place_3b=' '*9,place_3c=' '*2)
        self.assertEqual(normal(rows=[r])['target_status'],'pending')

    def test_missing_duplicate_deleted_correction_held(self):
        self.assertEqual(normal(rows=[])['target_status'],'pending')
        self.assertEqual(normal(rows=[hr(),hr()])['target_status'],'pending')
        for status in ['0','9','1',None,'7']:
            r=hr();r['status']=status;self.assertEqual(normal(rows=[r])['target_status'],'pending')

    def test_wager_flags_do_not_stop_another_normal_wager(self):
        for flag in ['special','refund','unestablished']:
            r=hr();r['wide_'+flag]='1'
            self.assertEqual(normal('wide',[r])['target_status'],'pending')
            self.assertEqual(normal('place',[r])['target_status'],'confirmed')

    def test_invalid_ascii_width_null_or_zero_held(self):
        for field,value in [('place_1b',None),('place_1b','00000012'),('place_1a','０１'),
                            ('place_1b','000000000'),('place_1c','00'),('place_4a','01')]:
            r=hr();r[field]=value;self.assertEqual(normal(rows=[r])['target_status'],'pending')

    def test_pairs_canonicalize_order_and_reject_repeated_horse(self):
        r=hr();r['umaren_1a']='0201';self.assertEqual(normal('umaren',[r])['official_targets'],['01-02'])
        r['umaren_1a']='0101';self.assertEqual(normal('umaren',[r])['target_status'],'pending')

    def test_incomplete_target_set_and_different_roster_pending(self):
        r=hr();r.update(wide_3a=' '*4,wide_3b=' '*9,wide_3c=' '*3)
        self.assertEqual(normal('wide',[r])['target_status'],'pending')
        r=hr();r['actual_runners']='05';self.assertEqual(normal(rows=[r])['target_status'],'pending')

    def test_top3_ties_not_inferred_as_official_pairs(self):
        rec=record();rec['finishes']['03']=2
        self.assertEqual(normal(rec=rec)['target_status'],'pending')

    def test_full_six_part_key_must_match_not_just_date_and_race_number(self):
        r=hr();r['race_key']=('2025','0301','05','02','01','01')
        self.assertEqual(normal(rows=[r])['reason'],'hr_full_key_mismatch')

    def test_official_target_without_a_valid_source_hash_is_not_verified(self):
        with self.assertRaises(MLFailure):m.normalize_hr([hr()],record(),'place','not-a-source-hash')

    def test_sealed_or_live_dates_rejected_before_hr_fields(self):
        for date,key in [('2026-03-01','2026030105010101'),('2026-10-24','2026102405040501')]:
            rec=record();rec.update(date=date,race_id=key)
            with self.assertRaises(MLFailure):normal(rows=[{'unexpected':'secret'}],rec=rec)

    def test_event_grid_complete_marginal_masses_and_no_renormalization(self):
        values=plackett_luce({'01':.4,'02':.3,'03':.2,'04':.1},2)
        for w in ['win','place','wide','umaren']:
            rows=m.event_values(values[w],['01','02','03','04'],w,2)
            self.assertTrue(all(r['probability']==values[w][r['target']] for r in rows))
        values['wide'].pop('01-02')
        with self.assertRaises(MLFailure):m.event_values(values['wide'],['01','02','03','04'],'wide',2)

    def test_asof_payee_key_never_guessed_from_field_size(self):
        with self.assertRaises(MLFailure):m.event_values({'01':.5,'02':.5},['01','02'],'place',None)

    def test_direct_place_marginals_are_not_forced_to_sum_to_payee_count(self):
        values={'01':.7,'02':.4,'03':.2}
        with self.assertRaises(MLFailure):m.event_values(values,list(values),'place',2)
        rows=m.event_values(values,list(values),'place',2,require_mass=False)
        self.assertEqual([r['probability'] for r in rows],[.7,.4,.2])

    def test_sql_guard_precedes_schema_and_any_database_access(self):
        db=Mock();rec=record();rec.update(date='2026-01-01',race_id='2026010105010101')
        with self.assertRaises(MLFailure):m.fetch_hr(db,[rec])
        self.assertEqual(db.mock_calls,[])


class SavedDevelopmentAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.export=self.base/'export';self.export.mkdir();self.baseline=self.base/'baseline';self.baseline.mkdir();self.out=self.base/'out'
        self.patch=patch.object(m.c,'BASE',self.base);self.patch.start();self.addCleanup(self.patch.stop)
        rec=record();self.rec=rec;strength={'01':.4,'02':.3,'03':.2,'04':.1};p=plackett_luce(strength,3);labels,_=rank_reference(rec['finishes'],3)
        path=self.export/'keiba-lab.csv'
        columns=['race_id','date','model_id','horse_no','finish_position','win_prob','pred_rank',
                 'model_artifact_id','source_sha256','predicted_at','horse_name','distance','surface']
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for model in ('H','raw_market','H_style','H_pace'):
                for i,(h,v) in enumerate(strength.items(),1):writer.writerow(dict(race_id=rec['race_id'],date=rec['date'],model_id=model,
                    horse_no=int(h),finish_position=i,win_prob=v,pred_rank=i,model_artifact_id='saved-'+model,
                    source_sha256='a'*64,predicted_at='',horse_name='',distance='',surface=''))
        self.write(self.export/'manifest.json',{'evaluation_kind':'stage6_development','csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        path=self.baseline/'races.local.jsonl';path.write_text(json.dumps(rec)+'\n',encoding='utf-8')
        self.write(self.baseline/'data_manifest.json',{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        path=self.baseline/'predictions.local.jsonl'
        path.write_text('\n'.join(json.dumps({'race_id':rec['race_id'],'date':rec['date'],'model':name,
            'probabilities':{w:p[w] for w in m.c.WAGERS},'rank_reference_labels':labels,
            'evaluation_kind':'retrospective_saved_prediction_replay'}) for name in m.SERIES)+'\n',encoding='utf-8')
        self.write(self.baseline/'prediction_commit.json',{'models':{'H_2025':{'model_id':'annual-H'},'ABCD_2025':{'model_id':'annual-ABCD'}}})

    def write(self,path,value):path.write_text(json.dumps(value),encoding='utf-8')

    def test_offline_packet_reuses_all_saved_values_and_keeps_missing_results_pending(self):
        summary=m.prepare(export=self.export,baseline=self.baseline,output=self.out)
        packet=json.loads((self.out/summary['artifact_id']/'packet.local.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['new_fits'],0);self.assertEqual(summary['win_rows'],16)
        self.assertEqual(len(packet['races']),1);race=packet['races'][0]
        self.assertEqual(race['win_models']['H']['probabilities']['01'],.4)
        self.assertIsNone(race['win_models']['H']['predicted_at'])
        self.assertTrue(all(v['official_targets'] is None for v in race['official'].values()))
        self.assertIsNone(packet['ROI'])
        with patch.object(m,'ROOT',self.out):self.assertEqual(m.load_current(),packet)

    def test_verified_hr_labels_connect_without_inventing_money(self):
        row=hr();row['race_key']=('2025','0301','05','01','01','01')
        session=Mock();session.return_value.__enter__=Mock(return_value=Mock());session.return_value.__exit__=Mock(return_value=False)
        with patch.object(m,'fetch_hr',return_value=([row],{}, {'status':'bounded_read_complete','rows':1})):
            summary=m.prepare(session=session,export=self.export,baseline=self.baseline,output=self.out)
        packet=json.loads((self.out/summary['artifact_id']/'packet.local.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['official_status_counts']['wide/confirmed'],1)
        with (self.out/summary['artifact_id']/'multiwager.local.csv').open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        self.assertTrue(all(r['official_hit'] in ('0','1') for r in rows))
        self.assertTrue(all(r['official_payout_yen_per_100']=='' for r in rows))

    def test_tampered_new_packet_refused_and_old_export_remains_identical(self):
        original=(self.export/'keiba-lab.csv').read_bytes()
        summary=m.prepare(export=self.export,baseline=self.baseline,output=self.out)
        (self.out/summary['artifact_id']/'packet.local.json').write_text('{}',encoding='utf-8')
        with patch.object(m,'ROOT',self.out):
            with self.assertRaises(MLFailure):m.load_current()
        self.assertEqual((self.export/'keiba-lab.csv').read_bytes(),original)

    def test_incomplete_stored_reference_does_not_publish_a_partial_cohort(self):
        path=self.baseline/'predictions.local.jsonl';path.write_text('',encoding='utf-8')
        with self.assertRaises(MLFailure):m.prepare(export=self.export,baseline=self.baseline,output=self.out)
        self.assertFalse((self.out/'current.json').exists())

