"""Bounded local input of already normalized, separately reviewed official data.

This is not an HR decoder or a verification authority. File claims alone never
make an outcome ready: the caller must also supply separately reviewed source
hashes. Unknown/missing context and special states remain pending, not losses.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from ..ml import safety
from . import contracts as c
from .outcomes import OfficialOutcome, official_labels, target_key, validate_official

BASE = c.BASE
SCHEMA_VERSION = 'jra.multiwager.official.normalized.v1'
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_RACES = 10_000
MAX_RECORDS = 3 * MAX_RACES
MAX_AMOUNT = 2 ** 53 - 1  # JSON/JavaScript exact integer boundary, not an HR rule.
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_REQUIRED = frozenset(('race_id', 'date', 'wager', 'source_sha256'))
_CONTEXT = frozenset(('state', 'payouts', 'complete_winning_set', 'flags_verified', 'units',
                       'revision_policy_verified'))
_BOOLEANS = ('complete_winning_set', 'flags_verified', 'revision_policy_verified')


def _hash(value):
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise safety.MLFailure('invalid_input')


def _rosters(rosters):
    if type(rosters) is not dict:
        raise safety.MLFailure('invalid_input')
    if len(rosters) > MAX_RACES:
        raise safety.MLFailure('input_limit')
    for race, horses in rosters.items():
        if type(race) is not str or not re.fullmatch('[0-9]{16}', race):
            raise safety.MLFailure('invalid_identity')
        value = race[:4] + '-' + race[4:6] + '-' + race[6:8]
        c.identity(race, value)  # Reject sealed/forward dates before any label.
        if type(horses) not in (tuple, list, set) or not 3 <= len(horses) <= 28:
            raise safety.MLFailure('invalid_identity')
        if any(type(horse) is not str for horse in horses) or len(set(horses)) != len(horses):
            raise safety.MLFailure('invalid_identity')
        for horse in horses:
            target_key([horse])


def _provided_context(record, roster):
    """Validate present fields even in pending states; never coerce raw values."""
    for name in _BOOLEANS:
        value = record.get(name)
        if value is not None and type(value) is not bool:
            raise safety.MLFailure('invalid_input')
    for name, pattern in (('state', '[a-z_]{1,48}'), ('units', '[A-Z0-9_]{1,48}')):
        value = record.get(name)
        if value is not None and (type(value) is not str or not re.fullmatch(pattern, value)):
            raise safety.MLFailure('invalid_input')
    payouts = record.get('payouts')
    if payouts is None:
        return
    if type(payouts) is not dict or len(payouts) > (28 if record['wager'] == 'place' else 378):
        raise safety.MLFailure('invalid_input')
    for key, amount in payouts.items():
        if type(key) is not str:
            raise safety.MLFailure('invalid_identity')
        parts = key.split('-')
        if (len(parts) != (1 if record['wager'] == 'place' else 2) or target_key(parts) != key
                or not set(parts) <= set(roster)):
            raise safety.MLFailure('invalid_identity')
        if type(amount) is not int or not 1 <= amount <= MAX_AMOUNT:
            raise safety.MLFailure('invalid_input')


def parse_normalized_outcomes(document, rosters, *, locally_verified_source_sha256=()):
    """Return ready outcomes and anonymous counts; pending data creates no labels.

    ``locally_verified_source_sha256`` is an explicit caller-side assertion that
    a separate local reconciliation has confirmed the source, field mapping,
    monetary unit, normal-state flags and revision policy. This function cannot
    establish that factual verification. An empty assertion is the safe default.
    Missing context remains missing; no flag or amount is filled or repaired.
    """
    if (type(document) is not dict or set(document) != {'schema_version', 'records'}
            or document['schema_version'] != SCHEMA_VERSION or type(document['records']) is not list):
        raise safety.MLFailure('invalid_schema')
    if len(document['records']) > MAX_RECORDS:
        raise safety.MLFailure('input_limit')
    _rosters(rosters)
    if type(locally_verified_source_sha256) not in (tuple, list, set, frozenset):
        raise safety.MLFailure('invalid_input')
    if len(locally_verified_source_sha256) > MAX_RECORDS:
        raise safety.MLFailure('input_limit')
    for digest in locally_verified_source_sha256:
        _hash(digest)
    reviewed = set(locally_verified_source_sha256)
    seen = set(); ready = {}; reasons = Counter(); wager_counts = Counter()
    for record in document['records']:
        if (type(record) is not dict or not _REQUIRED <= set(record)
                or not set(record) <= _REQUIRED | _CONTEXT):
            raise safety.MLFailure('invalid_schema')
        # This guard precedes interpreting payouts, state, or confirmation claims.
        if type(record['date']) is not str:
            raise safety.MLFailure('invalid_date')
        c.identity(record['race_id'], record['date'])
        if type(record['wager']) is not str or record['wager'] not in c.WAGERS:
            raise safety.MLFailure('invalid_input')
        _hash(record['source_sha256'])
        key = (record['race_id'], record['wager'])
        if key in seen:
            raise safety.MLFailure('duplicate_entry')
        seen.add(key); wager_counts[record['wager']] += 1
        if record['race_id'] not in rosters:
            raise safety.MLFailure('invalid_identity')
        roster = rosters[record['race_id']]
        _provided_context(record, roster)
        if any(name not in record or record[name] is None for name in _CONTEXT):
            reasons['official_context_missing'] += 1
            continue
        outcome = OfficialOutcome(**record)
        reason = validate_official(outcome, roster)
        if reason is None and outcome.source_sha256 not in reviewed:
            reason = 'separate_local_official_confirmation_required'
        if reason:
            reasons[reason] += 1
            continue
        labels, reason = official_labels(outcome, roster)
        if reason is not None or labels is None:
            raise safety.MLFailure('artifact_invalid')
        ready[key] = outcome
    summary = {'schema_version': SCHEMA_VERSION, 'input_records': len(document['records']),
        'roster_races': len(rosters), 'ready_outcomes': len(ready), 'pending_records': sum(reasons.values()),
        'pending_reasons': dict(sorted(reasons.items())), 'record_counts_by_wager': dict(sorted(wager_counts.items())),
        'absent_race_wager_records': len(rosters) * len(c.WAGERS) - len(seen),
        'source_confirmation_is_external': True, 'automatic_official_verification': False,
        'raw_HR_fields_decoded': False, 'database_connections': 0, 'production_ROI_enabled': False}
    return ready, summary


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise safety.MLFailure('invalid_input')
        result[key] = value
    return result


def _nonfinite(_):
    raise safety.MLFailure('invalid_input')


def load_normalized_outcomes(path, rosters, *, root=BASE, locally_verified_source_sha256=()):
    """Read one bounded local JSON file, rejecting sync/reparse and duplicate keys."""
    try:
        path = safety.local_path(Path(path), root=root, exists=True)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise safety.MLFailure('input_limit')
        with path.open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise safety.MLFailure('input_limit')
        document = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique, parse_constant=_nonfinite)
        ready, summary = parse_normalized_outcomes(document, rosters,
            locally_verified_source_sha256=locally_verified_source_sha256)
        summary['normalized_file_sha256'] = hashlib.sha256(raw).hexdigest()
        return ready, summary
    except safety.MLFailure:
        raise
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        raise safety.MLFailure('invalid_input') from None
