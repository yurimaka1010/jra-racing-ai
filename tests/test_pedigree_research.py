import copy
from datetime import date
import unittest

from jra_racing_ai.pedigree_research.features import Builder, Pool, stable_id, cell
from jra_racing_ai.pedigree_research import contracts as c, sources
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.db_connection import validate_audit_sql


def horse(i): return f'{i:010d}'


def race(day='2024-01-01', *, identities=(1,2), ranks=(1,2), surface='11', number='01'):
    d = date.fromisoformat(day)
    return [dict(kaisai_nen=str(d.year),kaisai_tsukihi=d.strftime('%m%d'),keibajo_code='05',
        kaisai_kai='01',kaisai_nichime='01',race_bango=number,horse_number=f'{i+1:02d}',
        horse_id=horse(h),finish_position=str(ranks[i]),actual_runners=str(len(identities)),
        anomaly='0',surface=surface,distance='1600') for i,h in enumerate(identities)]


def masters(h=1,s=100,m=200,b=300,status='1'):
    return dict(ketto_toroku_bango=horse(h),ketto_joho_01a=horse(s),ketto_joho_02a=horse(m),
                ketto_joho_05a=horse(b),data_kubun=status)


class PedigreeFeaturesTests(unittest.TestCase):
    def test_exact_six_finite_features_unknown(self):
        x=Builder({}).transform_day(race(),prediction=True)
        self.assertEqual(len(x),2)
        self.assertEqual(set(x[0]),set(c.FEATURES))
        self.assertEqual(x[0]['p_sire_level'],.5)
        self.assertEqual(x[0]['p_sire_condition_deviation'],0)
        self.assertEqual(x[0]['p_sire_support'],0)

    def test_cell_boundaries(self):
        self.assertEqual(cell('11','1400'),'turf:le1400')
        self.assertEqual(cell('11','1401'),'turf:gt1400_le1800')
        self.assertEqual(cell('23','1800'),'dirt:gt1400_le1800')
        self.assertEqual(cell('23','1801'),'dirt:gt1800')

    def test_unsupported_never_assigned(self):
        for s in ('51','27','99',None): self.assertIsNone(cell(s,'1600'))
        self.assertIsNone(cell('11','bad'))

    def test_stable_id_keeps_zeroes(self):
        self.assertEqual(stable_id('0000000001'),'0000000001')
        for v in (1,'0000000000','１２３４５６７８９０',' 000000001','bad'): self.assertIsNone(stable_id(v))

    def test_seven_day_boundary(self):
        b=Builder({horse(1):{'sire':horse(100)},horse(3):{'sire':horse(100)}})
        b.transform_day(race())
        a=b.transform_day(race('2024-01-07',identities=(3,4)),prediction=True)[0]
        z=b.transform_day(race('2024-01-08',identities=(3,4)),prediction=True)[0]
        self.assertEqual(a['p_sire_support'],0)
        self.assertEqual(z['p_sire_support'],1)
        self.assertGreater(z['p_sire_level'],.5)

    def test_own_horse_excluded_from_parent_and_prior(self):
        b=Builder({horse(1):{'sire':horse(100)}}); b.transform_day(race())
        x=b.transform_day(race('2024-01-08'),prediction=True)[0]
        self.assertEqual(x['p_sire_support'],0)
        self.assertAlmostEqual(x['p_sire_level'],0)

    def test_offspring_balanced_not_start_weighted(self):
        p=Pool()
        for _ in range(8):p.add(horse(1),1.)
        p.add(horse(2),0.)
        n,s=p.without(horse(3)); self.assertEqual(n,2); self.assertEqual(s/n,.5)

    def test_hierarchical_shrinkage_formula(self):
        b=Builder({horse(1):{'sire':horse(100)},horse(3):{'sire':horse(100)}})
        b.transform_day(race());x=b.transform_day(race('2024-01-08',identities=(3,4)),prediction=True)[0]
        level=(1+10*.5)/11
        self.assertAlmostEqual(x['p_sire_level'],level)
        self.assertAlmostEqual(x['p_sire_condition_deviation'],(1+5*level)/6-level)

    def test_current_results_and_prices_ignored(self):
        b=Builder({});a=race();z=copy.deepcopy(a)
        for r in z:r.update(finish_position='99',anomaly='7',odds=999,payout=100000,soha_time='0000')
        self.assertEqual(b.transform_day(a,prediction=True),b.transform_day(z,prediction=True))

    def test_entire_irregular_race_excluded(self):
        b=Builder({horse(1):{'sire':horse(100)},horse(3):{'sire':horse(100)}})
        a=race();a[1]['anomaly']='4';b.transform_day(a)
        x=b.transform_day(race('2024-01-08',identities=(3,4)),prediction=True)[0]
        self.assertEqual(x['p_sire_support'],0);self.assertEqual(b.counts['excluded_history_races'],1)

    def test_actual_count_not_registered_count(self):
        b=Builder({});a=race();a[0]['actual_runners']='3';b.transform_day(a)
        b.transform_day(race('2024-01-08'),prediction=True)
        self.assertEqual(len(b.global_pool.horses),0)

    def test_future_appending_does_not_change_past(self):
        b=Builder({});b.transform_day(race());x=b.transform_day(race('2024-01-08'),prediction=True)
        frozen=copy.deepcopy(x);b.transform_day(race('2024-02-01',ranks=(2,1)))
        self.assertEqual(x,frozen)
        z=Builder({});z.transform_day(race());self.assertEqual(x,z.transform_day(race('2024-01-08'),prediction=True))

    def test_unknown_horse_not_dropped(self):
        a=race();a[0]['horse_id']=None
        self.assertEqual(len(Builder({}).transform_day(a,prediction=True)),2)

    def test_role_groups_independent(self):
        b=Builder({horse(1):{'sire':horse(100),'damsire':horse(200)},horse(3):{'sire':horse(101),'damsire':horse(200)}})
        b.transform_day(race());x=b.transform_day(race('2024-01-08',identities=(3,4)),prediction=True)[0]
        self.assertEqual(x['p_sire_support'],0);self.assertEqual(x['p_damsire_support'],1)

    def test_historical_day_order_rejected(self):
        b=Builder({});b.transform_day(race())
        with self.assertRaises(MLFailure):b.transform_day(race())

    def test_duplicate_entries_rejected(self):
        a=race();a.append(a[0])
        with self.assertRaises(MLFailure):Builder({}).transform_day(a)

    def test_query_selects_only_identifiers(self):
        for table in ('jvd_um','jvd_sk','jvd_hn'):
            sql,args=sources.query(table,[horse(1)]);validate_audit_sql(sql)
            self.assertNotIn('bamei',sql);self.assertNotIn('ruikei',sql);self.assertNotIn('SELECT *',sql)
        with self.assertRaises(MLFailure):sources.query('jvd_se',[horse(1)])

    def test_lineage_links_and_maternal_derivation(self):
        m=masters();hn=[dict(data_kubun='1',hanshoku_toroku_bango=horse(i),ketto_joho_01a=horse(300) if i==200 else horse(999)) for i in (100,200,300)]
        links,s=sources.link({horse(1)},[m],[m],hn)
        self.assertEqual(links[horse(1)],{'sire':horse(100),'damsire':horse(300)})
        self.assertEqual(s['counts']['maternal_path_agrees'],1)

    def test_conflicting_maternal_path_unknown(self):
        hn=[dict(data_kubun='1',hanshoku_toroku_bango=horse(i),ketto_joho_01a=horse(999)) for i in (100,200,300)]
        links,s=sources.link({horse(1)},[masters()],[],hn)
        self.assertIsNone(links[horse(1)]['damsire'])

    def test_deleted_and_conflicting_ids_not_recovered_by_fallback(self):
        m=masters();links,_=sources.link({horse(1)},[dict(m,data_kubun='0')],[m],[])
        self.assertEqual(links[horse(1)],{'sire':None,'damsire':None})
        idx,_=sources.index([m,dict(m,ketto_joho_01a=horse(888))]);self.assertNotIn(horse(1),idx)

    def test_contract_zero_fit_and_fixed_strengths(self):
        spec=c.spec();self.assertEqual(spec['real_fits'],0);self.assertFalse(spec['production_connected'])
        self.assertEqual(len(spec['features']),6)

    def test_support_counts_distinct_horses(self):
        b=Builder({horse(i):{'sire':horse(100)} for i in (1,3)})
        b.transform_day(race());b.transform_day(race('2024-01-08'))
        x=b.transform_day(race('2024-01-15',identities=(3,4)),prediction=True)[0]
        self.assertEqual(x['p_sire_support'],1)

    def test_supported_but_different_cell_has_zero_deviation(self):
        b=Builder({horse(i):{'sire':horse(100)} for i in (1,3)})
        b.transform_day(race());rows=race('2024-01-08',identities=(3,4),surface='23')
        x=b.transform_day(rows,prediction=True)[0]
        self.assertGreater(x['p_sire_level'],.5);self.assertEqual(x['p_sire_support'],0)
        self.assertEqual(x['p_sire_condition_deviation'],0)

    def test_projection_preserves_rows_labels_and_common_values(self):
        import pandas as pd
        from jra_racing_ai.pedigree_research import data
        frame=pd.DataFrame([dict(entry_key='1',date='2024-01-01',winner_label=1,
                               **{f:None for f in data.spec()['features']})])
        extra=pd.DataFrame([dict(entry_key='1',**{f:0 for f in c.FEATURES})])
        a=data.merge(frame,extra,pedigree=False);b=data.merge(frame,extra,pedigree=True)
        pd.testing.assert_frame_equal(a,b[a.columns])
        self.assertEqual(len(b.columns),48) # 45 features + key/date/label

    def test_projection_rejects_optional_missing_rows_instead_of_dropping(self):
        import pandas as pd
        from jra_racing_ai.pedigree_research import data
        frame=pd.DataFrame([dict(entry_key='1',date='2024-01-01',winner_label=1,
                               **{f:None for f in data.spec()['features']})])
        extra=pd.DataFrame([dict(entry_key='2',**{f:0 for f in c.FEATURES})])
        with self.assertRaises(MLFailure):data.merge(frame,extra,pedigree=True)

    def test_cli_failure_never_prints_exception_even_if_log_cannot_be_saved(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from contextlib import redirect_stdout
        from unittest.mock import patch
        import importlib.util, io, json, sys
        from jra_racing_ai.pedigree_research import preparation
        path=Path(__file__).resolve().parents[1]/'scripts/prepare_pedigree.py'
        spec=importlib.util.spec_from_file_location('pedigree_cli_test',path)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        out=io.StringIO()
        with TemporaryDirectory() as tmp, \
             patch.object(sys,'argv',['prepare_pedigree.py','verify','--run-id','a'*32]), \
             patch.object(cli.safety,'local_path',return_value=Path(tmp)), \
             patch.object(preparation,'verify',side_effect=ValueError('DO_NOT_PRINT_PRIVATE_CANARY')), \
             patch.object(cli.c,'write',side_effect=OSError('DO_NOT_PRINT_PRIVATE_CANARY')), redirect_stdout(out):
            self.assertEqual(cli.main(),1)
        self.assertEqual(json.loads(out.getvalue()),{'status':'stopped','reason':'internal_error','stage':'verify'})
        self.assertNotIn('CANARY',out.getvalue())


if __name__=='__main__':unittest.main()
