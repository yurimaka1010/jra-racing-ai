"""Fictitious normalized records only; no real HR, credentials, DB, fits or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.multiwager import official_io as io
from jra_racing_ai.multiwager.outcomes import official_labels

RACE = '2025030105010101'
HASH = 'a' * 64


def rosters():
    return {RACE: ['01', '02', '03', '04']}


def record(wager='place'):
    return {'race_id': RACE, 'date': '2025-03-01', 'wager': wager, 'state': 'normal',
        'payouts': {'01': 120, '02': 150} if wager == 'place' else {'01-02': 300},
        'source_sha256': HASH, 'complete_winning_set': True, 'flags_verified': True,
        'units': 'JPY_PER_100', 'revision_policy_verified': True}


def document(*records):
    return {'schema_version': io.SCHEMA_VERSION, 'records': list(records) if records else [record()]}


def parse(d=None, *, reviewed=(HASH,), r=None):
    return io.parse_normalized_outcomes(document() if d is None else d, rosters() if r is None else r,
                                         locally_verified_source_sha256=reviewed)


class OfficialNormalizedParsingTests(unittest.TestCase):
    def test_ready_record_reuses_official_outcome_and_labels(self):
        ready, summary = parse()
        self.assertEqual(summary['ready_outcomes'], 1); self.assertEqual(summary['pending_records'], 0)
        event = ready[(RACE, 'place')]
        labels, why = official_labels(event, rosters()[RACE])
        self.assertIsNone(why); self.assertEqual(labels, {'01': 1, '02': 1, '03': 0, '04': 0})
        self.assertFalse(summary['production_ROI_enabled'])

    def test_file_claims_alone_never_make_ready(self):
        ready, summary = io.parse_normalized_outcomes(document(), rosters())
        self.assertEqual(ready, {})
        self.assertEqual(summary['pending_reasons'], {'separate_local_official_confirmation_required': 1})

    def test_confirmation_of_other_source_cannot_enable_record(self):
        ready, summary = parse(reviewed=('b' * 64,))
        self.assertFalse(ready); self.assertEqual(summary['pending_records'], 1)

    def test_all_three_wagers_use_canonical_targets_and_same_roster(self):
        ready, summary = parse(document(record('place'), record('wide'), record('umaren')))
        self.assertEqual(len(ready), 3); self.assertEqual(summary['absent_race_wager_records'], 0)
        for wager in ('wide', 'umaren'):
            labels, why = official_labels(ready[(RACE, wager)], rosters()[RACE])
            self.assertIsNone(why); self.assertEqual(len(labels), 6); self.assertEqual(sum(labels.values()), 1)

    def test_missing_context_stays_missing_and_pending(self):
        for name in io._CONTEXT:
            for null in (False, True):
                d = document()
                if null: d['records'][0][name] = None
                else: del d['records'][0][name]
                before = deepcopy(d); ready, summary = parse(d)
                self.assertFalse(ready); self.assertEqual(summary['pending_reasons'], {'official_context_missing': 1})
                self.assertEqual(d, before)

    def test_unknown_units_unverified_flags_or_incomplete_set_stay_pending(self):
        for name, value in [('units', 'UNKNOWN'), ('flags_verified', False), ('revision_policy_verified', False),
                             ('complete_winning_set', False)]:
            d = document(); d['records'][0][name] = value
            ready, summary = parse(d)
            self.assertFalse(ready)
            self.assertIn('official_mapping_unit_flags_or_revision_unconfirmed', summary['pending_reasons'])

    def test_special_refund_cancelled_and_unknown_states_are_pending_not_losses(self):
        for state in ('refund', 'special_payment', 'not_established', 'not_offered', 'cancelled', 'unconfirmed', 'unknown_state'):
            d = document(); d['records'][0]['state'] = state
            ready, summary = parse(d)
            self.assertFalse(ready); self.assertEqual(summary['pending_records'], 1)
            self.assertIn('special_refund_or_unconfirmed_state_pending', summary['pending_reasons'])

    def test_missing_winning_set_not_zero_yen_or_all_losers(self):
        d = document(); d['records'][0]['payouts'] = {}
        ready, summary = parse(d)
        self.assertFalse(ready); self.assertEqual(summary['pending_reasons'], {'official_winning_set_missing': 1})

    def test_date_guard_precedes_malformed_label_interpretation(self):
        for day in ('2026-01-01', '2026-09-30', '2026-10-24', '2025-10-01'):
            d = document(); d['records'][0].update(date=day, race_id=day.replace('-', '') + '05010101', payouts='RAW_SECRET')
            with self.subTest(day=day), self.assertRaises(MLFailure) as error: parse(d)
            self.assertEqual(error.exception.code, 'invalid_split')

    def test_rosters_cannot_smuggle_sealed_or_forward_dates(self):
        r = {'2026010105010101': ['01', '02', '03']}
        with self.assertRaises(MLFailure) as error: parse(r=r)
        self.assertEqual(error.exception.code, 'invalid_split')

    def test_duplicate_race_wager_record_rejected(self):
        with self.assertRaises(MLFailure) as error: parse(document(record(), record()))
        self.assertEqual(error.exception.code, 'duplicate_entry')

    def test_unrelated_roster_not_inferred_from_payout_horses(self):
        with self.assertRaises(MLFailure): parse(r={})
        for horses in (['01', '01', '02'], ['01', '02'], ['01', '02', 3]):
            with self.assertRaises(MLFailure): parse(r={RACE: horses})

    def test_bad_pair_and_outside_roster_rejected_even_when_state_pending(self):
        for target in ('02-01', '01-01', '01-05', '1-02', '01-02-03'):
            d = document(record('wide')); d['records'][0].update(state='unconfirmed', payouts={target: 100})
            with self.subTest(target=target), self.assertRaises(MLFailure): parse(d)

    def test_amount_must_be_exact_positive_integer_not_coerced(self):
        for amount in (0, -1, True, 100., '100', None, io.MAX_AMOUNT + 1):
            d = document(); d['records'][0]['payouts'] = {'01': amount}
            with self.subTest(amount=amount), self.assertRaises(MLFailure): parse(d)

    def test_boolean_confirmation_types_are_strict(self):
        for value in ('true', 1, 0, [], {}):
            d = document(); d['records'][0]['flags_verified'] = value
            with self.assertRaises(MLFailure): parse(d)

    def test_schema_and_extra_fields_are_strict(self):
        for change in ('version', 'extra_top', 'extra_record', 'missing_core', 'records_type'):
            d = document()
            if change == 'version': d['schema_version'] = 'unrecognized'
            elif change == 'extra_top': d['locally_verified_source_sha256'] = [HASH]
            elif change == 'extra_record': d['records'][0]['raw_SQL_row'] = 'SECRET'
            elif change == 'missing_core': del d['records'][0]['source_sha256']
            else: d['records'] = {}
            with self.subTest(change=change), self.assertRaises(MLFailure): parse(d)

    def test_bad_hash_and_wager_rejected(self):
        for name, value in [('source_sha256', 'x' * 64), ('source_sha256', 'a' * 63), ('wager', 'tansho'), ('date', 20250301)]:
            d = document(); d['records'][0][name] = value
            with self.assertRaises(MLFailure): parse(d)
        with self.assertRaises(MLFailure): parse(reviewed='a' * 64)

    def test_fixed_count_caps_checked_before_processing(self):
        with patch.object(io, 'MAX_RECORDS', 1), self.assertRaises(MLFailure): parse(document(record(), record('wide')))
        with patch.object(io, 'MAX_RACES', 0), self.assertRaises(MLFailure): parse()

    def test_summary_contains_aggregate_only(self):
        _, summary = parse(document(record(), record('wide')))
        text = json.dumps(summary)
        self.assertNotIn(RACE, text); self.assertNotIn('01-02', text); self.assertNotIn(HASH, text)
        self.assertEqual(summary['record_counts_by_wager'], {'place': 1, 'wide': 1})


class OfficialNormalizedFileTests(unittest.TestCase):
    def test_small_local_utf8_file_loads_with_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'normalized.json'
            path.write_text(json.dumps(document()), encoding='utf-8-sig')
            ready, summary = io.load_normalized_outcomes(path, rosters(), root=Path(tmp), locally_verified_source_sha256=(HASH,))
            self.assertEqual(len(ready), 1); self.assertEqual(len(summary['normalized_file_sha256']), 64)

    def test_duplicate_json_keys_at_top_or_inside_payouts_rejected(self):
        samples = [json.dumps(document()).replace('"records":', '"records":[],"records":', 1),
                   json.dumps(document()).replace('"01": 120', '"01":120,"01":120')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.json'
            for text in samples:
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(MLFailure): io.load_normalized_outcomes(path, rosters(), root=Path(tmp))

    def test_nonfinite_and_malformed_json_raise_fixed_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.json'
            for raw in (b'{', b'{"schema_version":NaN,"records":[]}', b'\xff\xfeSECRET_PRIVATE_RECORD'):
                path.write_bytes(raw)
                with self.assertRaises(MLFailure) as error: io.load_normalized_outcomes(path, rosters(), root=Path(tmp))
                self.assertNotIn('SECRET', str(error.exception)); self.assertNotIn('PRIVATE', repr(error.exception))

    def test_file_byte_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'too_large.json'; path.write_bytes(b' ' * 100)
            with patch.object(io, 'MAX_FILE_BYTES', 99), self.assertRaises(MLFailure) as error:
                io.load_normalized_outcomes(path, rosters(), root=Path(tmp))
            self.assertEqual(error.exception.code, 'input_limit')

    def test_unsafe_sync_path_rejected_before_read(self):
        with patch.object(io.safety, 'local_path', side_effect=MLFailure('unsafe_path')), \
                patch.object(Path, 'open') as read:
            with self.assertRaises(MLFailure): io.load_normalized_outcomes(Path(r'C:\OneDrive\secret.json'), rosters())
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
