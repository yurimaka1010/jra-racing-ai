"""Fabricated rosters/mock boosters only. No new fits, DB, or private input."""
import csv
from copy import deepcopy
from io import StringIO
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from jra_racing_ai import model_diagnostics as d
from jra_racing_ai.ml.safety import MLFailure


def source(day='2023-01-07'):
    rows = []
    for race in (1, 2):
        for horse in (1, 2, 3):
            row = {'entry_key': day.replace('-', '') + f'050101{race:02d}{horse:02d}', 'date': day,
                   'winner_label': str(float(horse == 2)), 'evaluation_year': day[:4],
                   'source_H_model_id': 'synthetic-H', 'H_style_model_id': 'synthetic-style',
                   'H_pace_model_id': 'synthetic-pace'}
            for model, values in {'H': (.6, .3, .1), 'raw_market': (.5, .3, .2),
                                  'H_style': (.4, .4, .2), 'H_pace': (.1, .7, .2)}.items():
                row[model] = str(values[horse - 1])
            rows.append(row)
    return rows


def frame(rows=None):
    rows = source() if rows is None else rows
    s = StringIO(newline=''); writer = csv.DictWriter(s, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows); s.seek(0)
    out = d.development_frame(s); out['ABCD'] = [.25, .5, .25] * 2
    return out


class AuxiliaryMetricsTests(unittest.TestCase):
    def test_confusion_counts_and_scores_match_sklearn(self):
        y = [0, 0, 1, 1, 1, 0]; pred = [0, 1, 1, 1, 0, 0]
        v = d.binary_metrics(y, pred)
        self.assertEqual([v[k] for k in ('true_negative', 'false_positive', 'false_negative', 'true_positive')],
                         confusion_matrix(y, pred).ravel().tolist())
        for name, function in [('accuracy', accuracy_score), ('precision', precision_score),
                               ('recall', recall_score), ('f1', f1_score)]:
            self.assertEqual(v[name], function(y, pred))

    def test_no_positive_prediction_is_undefined_precision_not_fake_zero(self):
        v = d.binary_metrics([0, 1, 0], [0, 0, 0])
        self.assertIsNone(v['precision']); self.assertEqual(v['recall'], 0); self.assertEqual(v['f1'], 0)
        self.assertEqual(v['predicted_positive'], 0)

    def test_no_actual_positive_or_no_denominator_is_null(self):
        v = d.binary_metrics([0, 0], [0, 0])
        self.assertIsNone(v['precision']); self.assertIsNone(v['recall']); self.assertIsNone(v['f1'])

    def test_invalid_class_decisions_and_shape_rejected(self):
        for y, p in [([], []), ([0, 1], [1]), ([0, 2], [0, 1]), ([0, 1], [.1, .9]), ([0, np.nan], [0, 1])]:
            with self.subTest(y=y, p=p), self.assertRaises(MLFailure): d.binary_metrics(y, p)

    def test_one_top_per_race_and_no_result_argument(self):
        f = frame(); v = d.top1_decisions(f.entry_key.tolist(), f.H)
        self.assertEqual(v.tolist(), [1, 0, 0, 1, 0, 0])
        self.assertEqual(sum(v), 2)

    def test_tie_break_is_numeric_horse_number_not_input_order(self):
        keys = ['202301070501010110', '202301070501010102', '202301070501010101']
        self.assertEqual(d.top1_decisions(keys, [.4, .4, .2]).tolist(), [0, 1, 0])

    def test_top1_rejects_duplicates_bad_sum_nan_or_single_horse(self):
        keys = ['202301070501010101', '202301070501010102']
        for a, b in [(keys, [.5, .4]), (keys, [np.nan, .5]), (keys[:1], [1]), ([keys[0], keys[0]], [.5, .5])]:
            with self.assertRaises(MLFailure): d.top1_decisions(a, b)

    def test_half_threshold_and_top1_do_not_change_saved_fractional_rule(self):
        results = d.measure(frame()); style = results['H_style']
        self.assertEqual(style['existing_metrics']['top1_accuracy_percent'], 50)
        self.assertEqual(style['race_top1_horse_number_tie_break']['recall'], 0)
        self.assertIsNone(style['probability_ge_0_5']['precision'])
        self.assertEqual(results['H_pace']['race_top1_horse_number_tie_break']['f1'], 1)

    def test_measure_keeps_inputs_and_H_is_fixed_comparator(self):
        f = frame(); before = f.copy(deep=True); out = d.measure(f)
        import pandas as pd
        pd.testing.assert_frame_equal(f, before)
        self.assertTrue(all(v == 0 for v in out['H']['difference_from_H'].values()))

    def test_existing_points_have_exact_identity_and_detect_metric_change(self):
        v = d.measure(frame()); expected = {n: x['existing_metrics'] for n, x in v.items()}
        self.assertEqual(d.assert_matches(v, expected), 0)
        changed = deepcopy(expected); changed['H']['brier_score'] += .001
        with self.assertRaises(MLFailure): d.assert_matches(v, changed)


class ImportanceTests(unittest.TestCase):
    def booster(self, names=('market_probability', 'market_rank', 'market_history_missing'),
                gain=(7., 3., 0.), splits=(4, 2, 0)):
        b = MagicMock(); b.feature_name.return_value = list(names)
        b.feature_importance.side_effect = lambda importance_type: gain if importance_type == 'gain' else splits
        return b, {'features': list(names)}

    def test_feature_order_complete_zero_features_and_gain_share(self):
        b, meta = self.booster(); out = d.importance(b, meta)
        self.assertEqual([r['feature'] for r in out], meta['features'])
        self.assertEqual([r['gain_share'] for r in out], [.7, .3, 0])
        self.assertEqual(out[-1]['split_count'], 0); b.fit.assert_not_called()

    def test_all_zero_gain_has_undefined_shares(self):
        b, meta = self.booster(gain=(0, 0, 0), splits=(0, 0, 0))
        self.assertTrue(all(v['gain_share'] is None for v in d.importance(b, meta)))

    def test_feature_name_order_mismatch_or_duplicate_cannot_be_silently_renamed(self):
        b, meta = self.booster(); meta['features'] = list(reversed(meta['features']))
        with self.assertRaises(MLFailure): d.importance(b, meta)
        b, meta = self.booster(names=('a', 'a', 'b'))
        with self.assertRaises(MLFailure): d.importance(b, meta)

    def test_gain_split_shapes_nan_negative_and_fractional_split_rejected(self):
        for options in ({'gain': (1, 2)}, {'gain': (1, np.nan, 2)}, {'gain': (-1, 2, 3)},
                        {'splits': (1, 2, 3.5)}, {'splits': (1, -1, 2)}):
            b, meta = self.booster(**options)
            with self.subTest(options=options), self.assertRaises(MLFailure): d.importance(b, meta)

    def test_unsafe_feature_labels_are_not_executed_or_rendered(self):
        b, meta = self.booster(names=('<script>', 'b', 'c'))
        with self.assertRaises(MLFailure): d.importance(b, meta)

    def test_tied_gain_keeps_saved_feature_order(self):
        b, meta = self.booster(gain=(2, 2, 1))
        self.assertEqual([r['gain_rank'] for r in d.importance(b, meta)], [1, 2, 3])

    def test_model_trained_on_sealed_period_stops_before_loading_trees(self):
        from pathlib import Path
        spec = {'directory': Path('synthetic'), 'metadata_sha256': None}
        with patch.object(d.safety, 'read_json', return_value={'split': {'train_end': '2026-01-01'}}), \
                patch('lightgbm.Booster') as loader:
            with self.assertRaisesRegex(MLFailure, '^invalid_split$'):
                d.model_importance(spec, lambda p, expected=None: p)
            loader.assert_not_called()

    def test_missing_split_does_not_implicitly_allow_an_unknown_model(self):
        from pathlib import Path
        spec = {'directory': Path('synthetic'), 'metadata_sha256': None}
        with patch.object(d.safety, 'read_json', return_value={'features': ['a']}), patch('lightgbm.Booster') as loader:
            with self.assertRaises(MLFailure): d.model_importance(spec, lambda p, expected=None: p)
            loader.assert_not_called()


class IsolationAndReportTests(unittest.TestCase):
    def test_sealed_and_Stage7_dates_rejected_before_interpreting_label(self):
        for day in ('2026-01-01', '2026-09-30', '2026-10-24', '2027-03-14', '2025-10-01'):
            rows = source(day); rows[0]['winner_label'] = 'SECRET_NOT_TO_PARSE'
            with self.subTest(day=day), self.assertRaisesRegex(MLFailure, '^invalid_split$'): frame(rows)

    def test_incomplete_roster_duplicate_or_two_winners_rejected(self):
        for mutation in ('duplicate', 'winner', 'probability'):
            rows = source()
            if mutation == 'duplicate': rows.append(dict(rows[0]))
            elif mutation == 'winner': rows[0]['winner_label'] = '1'
            else: rows[0]['H'] = '.7'
            with self.subTest(mutation=mutation), self.assertRaises(MLFailure): frame(rows)

    def test_invalid_feature_or_label_not_coerced(self):
        for field, value in [('H', 'inf'), ('winner_label', '.5'), ('evaluation_year', '2024')]:
            rows = source(); rows[0][field] = value
            with self.assertRaises(MLFailure): frame(rows)

    def test_html_is_offline_escaped_and_offset_limit_visible(self):
        b, meta = ImportanceTests().booster()
        obj = {'metrics': {'pooled': d.measure(frame())}, 'importance': [{
            'label': '<script>alert(1)</script>', 'model_id': 'synthetic-H', 'external_offset': True,
            'rows': d.importance(b, meta)}]}
        text = d.render(obj)
        self.assertIn('木の残差部分', text); self.assertIn('外部の市場offset', text)
        self.assertIn('&lt;script&gt;', text); self.assertNotIn('<script>', text)
        for token in ('https://', 'http://', 'fetch(', 'cdn.', '<iframe'): self.assertNotIn(token, text)

    def test_cli_never_outputs_raw_exception_records(self):
        with patch.object(d, 'run', side_effect=RuntimeError('secret password raw record')), patch('sys.stdout', new_callable=StringIO) as out:
            self.assertEqual(d.main(['--run-id', 'a' * 32]), 1)
            self.assertEqual(json.loads(out.getvalue()), {'status': 'stopped', 'reason': 'internal_error', 'stage': 'model_diagnostics'})

    def test_cli_has_no_training_or_arbitrary_input_arguments(self):
        with patch.object(d, 'run') as run, patch('sys.stdout', new_callable=StringIO):
            self.assertEqual(d.main(['--run-id', '../stage7']), 1)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
