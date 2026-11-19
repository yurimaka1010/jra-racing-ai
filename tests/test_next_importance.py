import unittest
from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
from jra_racing_ai.d39_next import contracts as c,importance as m


def sample():
    names=c.spec()['features'];booster=MagicMock();booster.feature_name.return_value=names;booster.current_iteration.return_value=219
    gain=np.arange(47,dtype=float);gain[1]=gain[0]
    booster.feature_importance.side_effect=lambda importance_type:gain.copy() if importance_type=='gain' else np.arange(47)
    meta={'features':names,'best_iteration':219,'model_id':'fake'}
    return booster,meta,{'rows':100,'missing':{name:100 if name=='going' else 0 for name in names}}


class ImportanceTests(unittest.TestCase):
    def test_all_features_and_translations_present_including_zero(self):
        b,meta,train=sample();rows=m.annotate(b,meta,train)
        self.assertEqual(len(rows),47);self.assertEqual({r['feature'] for r in rows},set(meta['features']))
        self.assertTrue(all(r['japanese'] and r['feature'] in r['display_label'] for r in rows))
        self.assertEqual(rows[-2]['feature_index'],0);self.assertEqual(rows[-1]['feature_index'],1)
        self.assertAlmostEqual(sum(r['gain_share_percent'] for r in rows),100.)

    def test_all_missing_and_zero_are_separate_flags(self):
        b,meta,train=sample();rows=m.annotate(b,meta,train)
        going=next(r for r in rows if r['feature']=='going');venue=next(r for r in rows if r['feature']=='venue')
        self.assertTrue(going['train_all_missing']);self.assertFalse(venue['train_all_missing']);self.assertEqual(venue['status'],'Gain/Split=0')

    def test_difference_and_rank_improvement_signs(self):
        b,meta,train=sample();old=m.annotate(b,meta,train);new=deepcopy(old)
        new[0].update(gain=new[0]['gain']+10,gain_rank=2)
        compared=m.compare(old,new)[0]
        self.assertEqual(compared['gain_difference'],10);self.assertEqual(compared['rank_improvement'],-1)

    def test_rest_aggregate_is_exactly_the_added_eight_columns(self):
        b,meta,train=sample();summary=m.rest_summary(m.annotate(b,meta,train))
        self.assertEqual(len(summary['rows']),8);self.assertEqual({r['feature'] for r in summary['rows']},set(c.EXTRA))
        self.assertEqual(summary['top10_count'],8)


if __name__=='__main__':unittest.main()
