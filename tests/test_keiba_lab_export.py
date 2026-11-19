"""Synthetic CSV fixtures only; no private saved data, DB, training, or network."""
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jra_racing_ai import keiba_lab_export as e
from jra_racing_ai.ml.safety import MLFailure


def fixture(day='2023-01-07', race_no=1):
    race = day.replace('-', '') + f'050101{race_no:02d}'
    probabilities = {'H': (.65, .25, .10), 'raw_market': (.50, .30, .20),
                     'H_style': (.45, .45, .10), 'H_pace': (.20, .60, .20)}
    rows = []
    for i in range(3):
        row = dict(entry_key=race + f'{i+1:02d}', date=day, winner_label=str(float(i == 1)),
                   evaluation_year=day[:4], raw_market_source='saved_T10_inverse_win_odds')
        for name, values in probabilities.items():
            row[name] = repr(values[i])
            if e.MODEL_COLUMNS[name]:
                row[e.MODEL_COLUMNS[name]] = 'synthetic-' + name
        rows.append(row)
    metadata = {r['entry_key']: dict(finish_position=('02', '01', '03')[i], distance='1600',
                surface='11', race_name='', going='', **{'class': ''}, horse_name='') for i, r in enumerate(rows)}
    return {race: rows}, {race: metadata}


def prediction_csv(selected):
    rows = [r for group in selected.values() for r in group]
    s = StringIO(newline=''); w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    s.seek(0)
    return s


def convert(selected, metadata):
    return e.convert(selected, metadata, 'synthetic-export', 'a' * 64)


def restored(rows):
    return list(csv.DictReader(StringIO(e.csv_text(rows).lstrip('\ufeff'), newline='')))


class KeibaLabExportTests(unittest.TestCase):
    def setUp(self):
        self.selected, self.metadata = fixture()

    def test_schema_exact_25_columns_and_utf8_bom_crlf(self):
        rows, _ = convert(self.selected, self.metadata)
        text = e.csv_text(rows)
        self.assertTrue(text.startswith('\ufeff')); self.assertIn('\r\n', text)
        self.assertEqual(text.splitlines()[0].lstrip('\ufeff').split(','), list(e.COLUMNS))
        self.assertEqual(len(e.COLUMNS), 25); self.assertEqual(len(restored(rows)), 12)

    def test_saved_probabilities_unchanged_and_ranks_follow_probabilities(self):
        rows, _ = convert(self.selected, self.metadata)
        for row in rows:
            old = self.selected[row['race_id']][row['horse_no'] - 1]
            self.assertEqual(float(row['win_prob']), float(old[row['model_id']]))
        ties = [r for r in rows if r['model_id'] == 'H_style']
        self.assertEqual([r['pred_rank'] for r in ties], [1, 2, 3])

    def test_real_finish_not_prediction_rank_or_binary_label_inference(self):
        rows, _ = convert(self.selected, self.metadata)
        for name in e.MODEL_COLUMNS:
            part = [r for r in rows if r['model_id'] == name]
            self.assertEqual([r['finish_position'] for r in part], [2, 1, 3])

    def test_model_artifact_ids_and_raw_no_fit_method_identity(self):
        rows, _ = convert(self.selected, self.metadata)
        for r in rows:
            self.assertEqual(r['model_artifact_id'], 'no_fit:raw_market:T10' if r['model_id'] == 'raw_market'
                             else 'synthetic-' + r['model_id'])
            self.assertEqual(r['source_run_id'], e.S6.name); self.assertEqual(r['source_sha256'], 'a' * 64)

    def test_unknown_metadata_is_blank_and_intake_blockers_are_explicit(self):
        rows, summary = convert(self.selected, self.metadata)
        for field in ('horse_name', 'race_name', 'going', 'surface', 'class'):
            self.assertTrue(all(r[field] == '' for r in rows))
            self.assertEqual(summary['missing_cells_by_column'][field], 12)
            self.assertIn('missing_display_' + field, summary['missing_display_warning_codes'])
        self.assertTrue(summary['keiba_lab_v1_1_import_ready'])
        self.assertEqual(summary['intake_blocker_codes'], [])
        self.assertFalse(summary['legacy_strict_display_import_ready'])
        self.assertEqual(summary['receiver_contract_revision'], e.RECEIVER_REVISION)

    def test_actual_known_display_fields_preserved_without_placeholders(self):
        for group in self.metadata.values():
            for i, row in enumerate(group.values()):
                row.update(surface='芝', going='良', race_name='架空レース', horse_name=f'架空{i}', **{'class': 'テスト'})
        rows, summary = convert(self.selected, self.metadata)
        self.assertTrue(summary['keiba_lab_v1_1_import_ready'])
        self.assertEqual(rows[0]['horse_name'], '架空0')

    def test_no_fabricated_prerace_times_or_payouts(self):
        rows, _ = convert(self.selected, self.metadata)
        for row in rows:
            self.assertEqual(row['evaluation_kind'], 'stage6_development')
            self.assertEqual(row['source_stage'], 'stage6')
            self.assertTrue(all(row[k] == '' for k in ('predicted_at', 'start_time', 'win_payout_100')))

    def test_all_four_existing_metric_sets_match_round_trip_including_ties(self):
        rows, _ = convert(self.selected, self.metadata)
        check = e.reconcile(self.selected, restored(rows), self.metadata)
        self.assertEqual(check['verified_models'], 4)
        self.assertEqual(check['maximum_metric_difference'], 0)
        self.assertEqual(check['metrics']['H_style']['csv_round_trip']['top1_accuracy_percent'], 50.)
        self.assertEqual(check['metrics']['H_pace']['csv_round_trip']['top1_accuracy_percent'], 100.)

    def test_csv_probability_change_is_detected(self):
        rows, _ = convert(self.selected, self.metadata); rows = restored(rows)
        rows[0]['win_prob'] = '.64'
        with self.assertRaises(MLFailure): e.reconcile(self.selected, rows, self.metadata)

    def test_csv_rank_or_model_provenance_change_is_detected(self):
        rows, _ = convert(self.selected, self.metadata)
        for field, value in [('pred_rank', '3'), ('model_artifact_id', 'other'), ('evaluation_kind', 'prerace_saved'),
                             ('source_stage', 'stage7'), ('predicted_at', '2023-01-07T10:00:00+09:00')]:
            with self.subTest(field=field):
                changed = restored(rows); changed[0][field] = value
                with self.assertRaises(MLFailure): e.reconcile(self.selected, changed, self.metadata)

    def test_swapping_two_nonwinner_results_is_detected(self):
        rows, _ = convert(self.selected, self.metadata); rows = restored(rows)
        rows[0]['finish_position'], rows[2]['finish_position'] = rows[2]['finish_position'], rows[0]['finish_position']
        with self.assertRaises(MLFailure): e.reconcile(self.selected, rows, self.metadata)

    def test_extra_missing_or_duplicate_csv_rows_cannot_change_cohort(self):
        rows, _ = convert(self.selected, self.metadata)
        for altered in (restored(rows)[:-1], restored(rows) + restored(rows)[:1]):
            with self.assertRaises(MLFailure): e.reconcile(self.selected, altered, self.metadata)

    def test_winner_label_conflict_with_saved_official_result_stops(self):
        rows = next(iter(self.selected.values())); rows[0]['winner_label'] = '1'; rows[1]['winner_label'] = '0'
        with self.assertRaises(MLFailure): convert(self.selected, self.metadata)

    def test_missing_official_results_never_filled_from_binary_labels(self):
        next(iter(self.metadata.values()))[next(iter(self.selected)) + '01']['finish_position'] = ''
        rows, summary = convert(self.selected, self.metadata)
        self.assertTrue(all(r['finish_position'] == '' for r in rows))
        check = e.reconcile(self.selected, restored(rows), self.metadata)
        self.assertEqual(check['verified_models'], 0); self.assertEqual(check['pending_result_races'], 1)
        self.assertEqual(summary['result_pending_races'], {'unsupported_or_incomplete_official_result': 1})

    def test_dead_heat_zero_or_anomaly_not_guessed_into_contiguous_results(self):
        for values in (('01', '01', '03'), ('00', '01', '02'), ('DNF', '01', '02')):
            source = deepcopy(self.metadata)
            for row, value in zip(next(iter(source.values())).values(), values): row['finish_position'] = value
            rows, _ = convert(self.selected, source)
            self.assertTrue(all(r['finish_position'] == '' for r in rows))

    def test_full_race_key_and_horse_set_required(self):
        incomplete = deepcopy(self.metadata)
        next(iter(incomplete.values())).popitem()
        with self.assertRaises(MLFailure): convert(self.selected, incomplete)
        with self.assertRaises(MLFailure): e.identity('202301070501010101', '2023-01-08')

    def test_exact_source_join_and_duplicate_rejection(self):
        race = next(iter(self.selected)); stream = StringIO(newline='')
        fields = [*e.RACE_KEY, 'horse_number', 'finish_position', 'distance', 'surface']
        source = []
        for i, row in enumerate(self.metadata[race].values(), 1):
            source.append(dict(zip(e.RACE_KEY, (race[:4], race[4:8], race[8:10], race[10:12], race[12:14], race[14:16])),
                               horse_number=f'{i:02d}', **{k: row[k] for k in fields if k not in e.RACE_KEY and k != 'horse_number'}))
        w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(source); stream.seek(0)
        self.assertEqual(e.source_rows(stream, self.selected), self.metadata)
        w.writerow(source[0]); stream.seek(0)
        with self.assertRaises(MLFailure): e.source_rows(stream, self.selected)

    def test_multiple_races_expand_without_cutting_a_race_or_model(self):
        another, extra = fixture(race_no=2); selected = self.selected | another; source = self.metadata | extra
        actual = e.select_rows(prediction_csv(selected), '2023-01-01', '2023-09-30', 2)
        rows, _ = convert(actual, source)
        self.assertEqual(len(rows), 24)
        self.assertEqual(e.reconcile(actual, restored(rows), source)['metrics']['H']['source']['races'], 2)
        first = e.select_rows(prediction_csv(selected), '2023-01-01', '2023-09-30', 1)
        self.assertEqual(list(first), list(self.selected))

    def test_date_filter_is_explicit_and_no_races_is_fixed_error(self):
        with self.assertRaisesRegex(MLFailure, '^insufficient_history$'):
            e.select_rows(prediction_csv(self.selected), '2024-01-01', '2024-09-30', 1)

    def test_sealed_stage7_and_non_stage6_months_rejected_before_labels(self):
        for day in ('2026-01-01', '2026-09-30', '2026-10-24', '2027-03-14', '2025-10-01', '2022-09-30'):
            selected, _ = fixture(day)
            next(iter(selected.values()))[0]['winner_label'] = 'DO_NOT_INTERPRET'
            with self.subTest(day=day), self.assertRaisesRegex(MLFailure, '^invalid_split$'):
                e.select_rows(prediction_csv(selected), '2023-01-01', '2025-09-30', 1)

    def test_nonfinite_probability_invalid_sum_or_unknown_model_id_stops(self):
        for field, value in [('H', 'nan'), ('H', '1.1'), ('H', '.7'), ('source_H_model_id', ''),
                             ('raw_market_source', 'guessed-final-odds')]:
            selected = deepcopy(self.selected); next(iter(selected.values()))[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(MLFailure):
                e.select_rows(prediction_csv(selected), '2023-01-01', '2025-09-30', 1)

    def test_limits_and_reversed_dates_stop(self):
        for limit in (0, 1001):
            with self.assertRaises(MLFailure): e.select_rows(prediction_csv(self.selected), '2023-01-01', '2025-09-30', limit)
        with self.assertRaises(MLFailure): e.select_rows(prediction_csv(self.selected), '2025-01-01', '2023-01-01', 1)
        rows, _ = convert(self.selected, self.metadata)
        with patch.object(e, 'MAX_BYTES', 20), self.assertRaisesRegex(MLFailure, '^input_limit$'): e.csv_text(rows)

    def test_formula_like_metadata_rejected_without_changing_source(self):
        next(iter(self.metadata.values()))[next(iter(self.selected)) + '01']['horse_name'] = '=SECRET()'
        with self.assertRaisesRegex(MLFailure, '^invalid_input$'): convert(self.selected, self.metadata)

    def test_cli_only_fixed_diagnostic_not_exception_or_records(self):
        with patch.object(e, 'export', side_effect=RuntimeError('SECRET horse odds password')), patch('sys.stdout', new_callable=StringIO) as output:
            self.assertEqual(e.main([]), 1)
            self.assertEqual(json.loads(output.getvalue()), {'status': 'stopped', 'reason': 'internal_error', 'stage': 'keiba_lab_export'})

    def test_synthetic_header_matches_receiver_without_reading_saved_data(self):
        path = Path(__file__).resolve().parents[2] / 'sites/jra-private-lab/docs/csv-template-v1.1.csv'
        if path.exists():
            self.assertEqual(path.read_text(encoding='utf-8-sig').strip().split(','), list(e.COLUMNS))


if __name__ == '__main__':
    unittest.main()
