"""Delayed-history population regression, synthetic DB only, zero fits."""
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
import csv
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jra_racing_ai.core import RACE_KEY
from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.pedigree_forward import contracts as c, history


def race(day, number='01', status='7'):
    return dict(zip((*RACE_KEY, 'status'), (str(day.year), day.strftime('%m%d'), '05', '01', '01', number, status)))


class HistoryPopulationTests(unittest.TestCase):
    def test_cancelled_and_deleted_not_history_gaps(self):
        rows=[race(date(2026,10,24)), race(date(2026,10,24),'02','9'), race(date(2026,10,24),'03','0')]
        seen={tuple(rows[0][k] for k in RACE_KEY)}
        value=history.check_population(rows,seen)
        self.assertEqual(value['final_races'],1)
        self.assertEqual(value['cancelled_races'],1)
        self.assertEqual(value['deleted_races'],1)
        self.assertEqual(value['cancelled_or_deleted_in_history'],0)

    def test_cancelled_key_in_runner_source_stops(self):
        row=race(date(2026,10,24),status='9')
        with self.assertRaises(MLFailure):history.check_population([row], {tuple(row[k] for k in RACE_KEY)})

    def test_unconfirmed_categories_still_stop(self):
        for status in ('1','2','3','4','5','6','X',''):
            with self.subTest(status=status),self.assertRaises(MLFailure):
                history.check_population([race(date(2026,10,24),status=status)],set())

    def test_missing_normal_final_race_still_stops(self):
        with self.assertRaises(MLFailure):history.check_population([race(date(2026,10,24))],set())

    def test_duplicate_conflicting_versions_stop(self):
        row=race(date(2026,10,24))
        with self.assertRaises(MLFailure):history.check_population([row,dict(row,status='9')],set())

    def test_empty_and_cancelled_only_period_is_valid(self):
        self.assertEqual(history.check_population([],set())['final_races'],0)
        self.assertEqual(history.check_population([race(date(2026,10,24),status='0')],set())['deleted_races'],1)

    def test_twenty_day_download_month_split_seven_day_cutoff_and_cache(self):
        # History extends 20 calendar days beyond the base, through target minus 7.
        target=date(2026,11,14); base_end=date(2026,10,18); cutoff=target-timedelta(days=7)
        calls=[]
        class DB:
            def iter_batches(self,operation,sql,args,**limits):
                self_args=args; calls.append(('read',args))
                d=date(int(args[0]),int(args[1][:2]),int(args[1][2:]))
                row=race(d);row.pop('status');row.update(horse_number='01',time_raw='1234')
                yield [row]
            def aggregate(self,operation,sql,args,**limits):
                d=date(int(args[0]),int(args[1][:2]),int(args[1][2:]))
                return [race(d),race(d,'02','9'),race(d,'03','0')]
        @contextmanager
        def session():yield DB()
        def source_query(schema,first,last,plan):
            calls.append(('query',first,last))
            sql=('SELECT ra.kaisai_nen FROM ONLY public.jvd_ra ra WHERE ra.kaisai_nen::text=%s '
                 'AND ra.kaisai_tsukihi::text BETWEEN %s AND %s LIMIT %s')
            return sql,(str(first.year),first.strftime('%m%d'),last.strftime('%m%d'),20001),('horse_number',)
        with TemporaryDirectory(dir=c.BASE/'test-temp') as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            (raw/'source_manifest.json').write_text(json.dumps({'latest_final_date':str(base_end)}),encoding='utf-8')
            (raw/'plan.json').write_text('{}',encoding='utf-8')
            with patch.object(c,'ROOT',root),patch.object(c,'RAW_PARENT',raw),\
                 patch.object(history.old_data,'columns',return_value={'se':{'soha_time':{'type_is_builtin':True,'type_name':'text'}}}),\
                 patch.object(history.old_data,'source_query',side_effect=source_query):
                parts=history.extra_sources(root/'run',target,session)
                self.assertEqual(len(parts),2)
                self.assertEqual(calls[0],('query',date(2026,10,19),date(2026,10,31)))
                self.assertEqual([x for x in calls if x[0]=='query'][-1][2],cutoff)
                for path,item in parts:
                    self.assertEqual(item['population']['cancelled_races'],1)
                    self.assertEqual(item['population']['deleted_races'],1)
                    self.assertLessEqual(date.fromisoformat(item['end']),cutoff)
                    with path.open(encoding='utf-8') as stream:self.assertEqual(len(list(csv.DictReader(stream))),1)
                before=len(calls)
                again=history.extra_sources(root/'run',target,session)
                self.assertEqual(len(calls),before)
                self.assertEqual(parts,again)
                info=c.read(root/'run'/'history'/str(target)/'manifest.json')
                self.assertEqual(info['population_version'],history.HISTORY_POPULATION_VERSION)


if __name__=='__main__':unittest.main()
