"""Rank references and confirmed official outcomes are deliberately separate."""
from dataclasses import dataclass
from itertools import combinations
import math
import re

from ..ml.safety import MLFailure
from . import contracts as c


def target_key(horses):
    if not isinstance(horses, (tuple, list)) or len(horses) not in (1, 2):
        raise MLFailure('invalid_input')
    if (any(not isinstance(h, str) or not re.fullmatch('[0-9]{2}', h) or not 1 <= int(h) <= 28 for h in horses)
            or len(set(horses)) != len(horses)):
        raise MLFailure('invalid_identity')
    return '-'.join(sorted(horses))


def rank_reference(finishes, paid_positions):
    """No official settlement claim; top-three dead heats remain unresolved."""
    if type(paid_positions) is not int or paid_positions not in (2, 3) or not isinstance(finishes, dict):
        raise MLFailure('invalid_input')
    if not 3 <= len(finishes) <= 28:
        raise MLFailure('invalid_identity')
    for horse, rank in finishes.items():
        target_key([horse])
        if type(rank) is not int or not 1 <= rank <= len(finishes):
            raise MLFailure('invalid_input')
    top = {rank: [h for h, r in finishes.items() if r == rank] for rank in (1, 2, 3)}
    if any(len(horses) != 1 for horses in top.values()):
        return None, 'top3_dead_heat_or_rank_gap_needs_official_targets'
    first_three = {top[k][0] for k in (1, 2, 3)}; first_two = {top[k][0] for k in (1, 2)}
    place = {top[k][0] for k in range(1, paid_positions + 1)}
    pairs = list(combinations(sorted(finishes), 2))
    return {'place': {h: int(h in place) for h in sorted(finishes)},
            'umaren': {target_key(list(pair)): int(set(pair) == first_two) for pair in pairs},
            'wide': {target_key(list(pair)): int(set(pair) <= first_three) for pair in pairs}}, None


@dataclass(frozen=True, repr=False)
class OfficialOutcome:
    race_id: str
    date: str
    wager: str
    state: str
    payouts: dict
    source_sha256: str
    complete_winning_set: bool
    flags_verified: bool
    units: str
    revision_policy_verified: bool

    def __repr__(self):
        return 'OfficialOutcome(details=local_only)'


def validate_official(outcome, roster):
    if type(outcome) is not OfficialOutcome or outcome.wager not in c.WAGERS:
        raise MLFailure('invalid_input')
    c.identity(outcome.race_id, outcome.date)
    if not isinstance(roster, (tuple, list, set)) or not 3 <= len(roster) <= 28 or len(set(roster)) != len(roster):
        raise MLFailure('invalid_identity')
    for horse in roster: target_key([horse])
    if (not isinstance(outcome.source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', outcome.source_sha256)
            or not isinstance(outcome.payouts, dict)):
        raise MLFailure('invalid_input')
    if outcome.state != 'normal':
        return 'special_refund_or_unconfirmed_state_pending'
    if (outcome.complete_winning_set is not True or outcome.flags_verified is not True
            or outcome.revision_policy_verified is not True or outcome.units != 'JPY_PER_100'):
        return 'official_mapping_unit_flags_or_revision_unconfirmed'
    if not outcome.payouts:
        return 'official_winning_set_missing'
    for key, amount in outcome.payouts.items():
        if not isinstance(key, str):
            raise MLFailure('invalid_identity')
        parts = key.split('-')
        if len(parts) != (1 if outcome.wager == 'place' else 2) or target_key(parts) != key or not set(parts) <= set(roster):
            raise MLFailure('invalid_identity')
        if type(amount) is not int or amount <= 0:
            raise MLFailure('invalid_input')
    return None


def official_labels(outcome, roster):
    reason = validate_official(outcome, roster)
    if reason:
        return None, reason
    keys = sorted(roster) if outcome.wager == 'place' else [target_key(list(p)) for p in combinations(sorted(roster), 2)]
    return {key: int(key in outcome.payouts) for key in keys}, None


def settle_ticket(outcome, roster, target, stake_yen=100):
    """Settle an already fixed research ticket; never choose a ticket or infer price."""
    if type(stake_yen) is not int or stake_yen <= 0 or stake_yen % 100:
        raise MLFailure('invalid_input')
    target_key(target.split('-') if isinstance(target, str) else [])
    if not set(target.split('-')) <= set(roster):
        raise MLFailure('invalid_identity')
    reason = validate_official(outcome, roster)
    if reason:
        return {'status': 'pending', 'reason': reason, 'investment_yen': stake_yen,
                'payout_yen': None, 'net_profit_yen': None}
    if len(target.split('-')) != (1 if outcome.wager == 'place' else 2) or target_key(target.split('-')) != target:
        raise MLFailure('invalid_identity')
    paid = outcome.payouts.get(target, 0) * (stake_yen // 100)
    return {'status': 'settled', 'reason': None, 'investment_yen': stake_yen,
            'hit': target in outcome.payouts, 'payout_yen': paid, 'net_profit_yen': paid - stake_yen}


def summarize_settlements(records):
    """Aggregate already fixed/settled tickets by wager and month; no selection."""
    if not isinstance(records, list): raise MLFailure('invalid_input')
    groups = {}
    for record in records:
        day = c.day(record.get('date')); wager = record.get('wager')
        if wager not in c.WAGERS or record.get('status') not in ('settled', 'pending'):
            raise MLFailure('invalid_input')
        invested = record.get('investment_yen')
        if type(invested) is not int or invested <= 0 or invested % 100:
            raise MLFailure('invalid_input')
        for key in (wager, day.isoformat()[:7] + '/' + wager):
            group = groups.setdefault(key, {'tickets': 0, 'investment_yen': 0, 'unsettled': 0,
                'settled_tickets': 0, 'settled_investment_yen': 0, 'settled_payout_yen': 0, 'settled_hits': 0})
            group['tickets'] += 1; group['investment_yen'] += invested
            if record['status'] == 'pending':
                if record.get('payout_yen') is not None or record.get('net_profit_yen') is not None:
                    raise MLFailure('invalid_input')
                group['unsettled'] += 1
            else:
                payout = record.get('payout_yen'); hit = record.get('hit')
                if (type(payout) is not int or payout < 0 or type(hit) is not bool
                        or record.get('net_profit_yen') != payout - invested or hit != (payout > 0)):
                    raise MLFailure('invalid_input')
                group['settled_tickets'] += 1; group['settled_investment_yen'] += invested
                group['settled_payout_yen'] += payout; group['settled_hits'] += int(hit)
    for group in groups.values():
        complete = group['unsettled'] == 0
        group['payout_yen'] = group['settled_payout_yen'] if complete else None
        group['net_profit_yen'] = group['payout_yen'] - group['investment_yen'] if complete else None
        group['return_rate_percent'] = 100 * group['payout_yen'] / group['investment_yen'] if complete and group['investment_yen'] else None
        group['settled_subset_net_profit_yen'] = group['settled_payout_yen'] - group['settled_investment_yen']
    return {'groups': groups, 'ticket_count': len(records), 'return_rate_percent_when_no_tickets': None,
            'pending_is_not_a_loss': True, 'fixed_ticket_selection_not_optimized': True}
