"""Four-model entertainment projection. No fitting or formal study I/O."""
from copy import deepcopy
import re

from .win_sim import core

NAMES = ('old_D39', 'new_D39', 'control_D39', 'H')
RULE = 'watch_R1_saved_top1_win_100_yen_no_EV_filter'


def ticket(status='pending', reason='result_unavailable', *, stake=100, payout=None, hit=None):
    return {'status': status, 'reason': reason, 'stake_yen': stake, 'payout_yen': payout,
            'net_profit_yen': payout - stake if payout is not None else None, 'hit': hit}


def project(forecast, official_rows, outcome, *, checked_at):
    """Use saved rank 1 only. A veto never becomes a scored prediction."""
    state = {'result_status': 'pending', 'reason': 'result_unavailable', 'finishes': {},
             'podium': [], 'models': {}, 'checked_at': checked_at, 'rule': RULE,
             'formal_evaluation': False, 'payout_status': 'pending'}
    for name in NAMES:
        model = forecast.get('models', {}).get(name, {})
        top = [x for x in model.get('rows', []) if x.get('rank') == 1]
        saved = model.get('status') == 'saved' and len(top) == 1
        eligible = saved and forecast.get('valid_for_results') is True and not forecast.get('forecast_kind')
        state['models'][name] = {
            'prediction_saved': saved, 'eligible': eligible,
            'horse_no': top[0]['horse_no'] if saved else None,
            'probability': top[0]['probability'] if saved else None,
            'model_id': model.get('model_id'), 'predicted_at': model.get('predicted_at'),
            'top_finish': None, 'top_hit': None,
            'selection_state': 'pending' if eligible else 'held' if saved else 'missing',
            'ticket': ticket() if eligible else ticket('excluded', forecast.get('reason') or 'forecast_missing_or_vetoed', stake=0),
        }
    roster = set(forecast.get('roster', []))
    if not roster or len(official_rows) != len(roster) or {r.get('horse') for r in official_rows} != roster:
        state['reason'] = 'result_roster_missing_or_duplicate'; return state
    expected_ids = forecast.get('horse_ids', {})
    identities_ok = all(not expected_ids or row.get('horse_id') == expected_ids.get(row['horse']) for row in official_rows)
    normal = core.outcome_reason(outcome, tuple(roster)) is None
    # A confirmed normal HR supplies the complete winning set and exact JPY/100
    # even while SE is still the declared-entry version. Never invent other ranks.
    if identities_ok and normal and all(row.get('status') in ('1', '2') for row in official_rows):
        if outcome.race != tuple(forecast['race_key']):
            state['reason'] = 'hr_se_winner_mismatch'; return state
        winners = set(outcome.payouts_yen_per_100)
        state.update(result_status='winner_confirmed', reason='full_order_not_received',
                     result_basis='official_confirmed_HR_winning_set_only', payout_status='confirmed',
                     full_order_confirmed=False, finishes={h: 1 for h in winners},
                     podium=[{'horse_no':h, 'finish':1} for h in sorted(winners)])
        for model in state['models'].values():
            if not model['eligible']: continue
            won = model['horse_no'] in winners
            model.update(top_hit=won, top_finish=1 if won else None, selection_state='win' if won else 'loss',
                         ticket=core.settle(model['horse_no'], outcome, tuple(roster)))
        return state
    if any((expected_ids and row.get('horse_id') != expected_ids.get(row['horse']))
           or row.get('status') not in ('5', '6', '7')
           or row.get('anomaly') not in ('0', '1', '2', '3', '4', '5', '6', '7')
           or not re.fullmatch('[0-9]{2}', row.get('rank') or '') for row in official_rows):
        state['reason'] = 'result_status_identity_or_cancellation_unconfirmed'; return state
    ranks = {}
    for row in official_rows:
        rank, anomaly = int(row['rank']), row['anomaly']
        if anomaly in ('1', '2', '3'):
            ranks[row['horse']] = '返還対象'
        elif rank == 0 and anomaly in ('4', '5'):
            ranks[row['horse']] = '競走中止' if anomaly == '4' else '失格'
        elif 1 <= rank <= len(roster):
            ranks[row['horse']] = rank
        else:
            state['reason'] = 'result_rank_unconfirmed'; return state
    winners = {h for h, r in ranks.items() if r == 1}
    if normal and (outcome.race != tuple(forecast['race_key']) or set(outcome.payouts_yen_per_100) != winners):
        state['reason'] = 'hr_se_winner_mismatch'; return state
    if not winners:
        state['reason'] = 'unestablished_or_no_confirmed_winner'; return state
    if any(r['status'] != '7' for r in official_rows) and not normal:
        state.update(result_status='provisional', reason='full_order_flash_waiting_for_confirmed_hr',
                     finishes=ranks, podium=[{'horse_no':h, 'finish':rank} for h,rank in
                     sorted(ranks.items(),key=lambda v:(v[1] if type(v[1]) is int else 999,v[0]))
                     if type(rank) is int and rank <= 3], full_order_confirmed=False)
        return state
    if len(winners) > 1 and not normal:
        state['reason'] = 'dead_heat_or_unconfirmed_winner'; return state
    state.update(result_status='confirmed', reason=None, finishes=ranks,
                 full_order_confirmed=True,
                 result_basis='official_SE7' if all(r['status'] == '7' for r in official_rows) else 'official_full_order_SE5_6_with_confirmed_HR',
                 payout_status='confirmed' if normal else 'pending',
                 payout_reason=None if normal else outcome.reason if outcome else 'result_unavailable')
    state['podium'] = [{'horse_no': h, 'finish': rank} for h, rank in
                       sorted(ranks.items(), key=lambda v: (v[1] if type(v[1]) is int else 999, v[0]))
                       if type(rank) is int and rank <= 3]
    by_horse = {r['horse']: r for r in official_rows}
    for model in state['models'].values():
        if not model['eligible']: continue
        horse = model['horse_no']; finish = ranks[horse]
        model['top_finish'] = finish
        if finish == '返還対象':
            model.update(selection_state='refunded', top_hit=None,
                         ticket=ticket('refunded', 'official_SE_confirmed_refund', payout=100, hit=False))
            continue
        model['top_hit'] = finish == 1
        model['selection_state'] = 'win' if model['top_hit'] else 'loss'
        # Without an official amount, even a known loser is kept unsettled.
        model['ticket'] = core.settle(horse, outcome, tuple(roster))
        if model['ticket']['status'] != 'settled':
            model['ticket']['hit'] = model['top_hit']
    return state


def scoreboard(races):
    """Per-model denominators and settled-only cash; no formal evaluator reused."""
    output = {}
    for name in NAMES:
        items = [r['watch']['models'][name] for r in races]
        eligible = [m for m in items if m['eligible']]
        decided = [m for m in eligible if type(m['top_hit']) is bool]
        settled = [m['ticket'] for m in eligible if m['ticket']['status'] in ('settled', 'refunded')]
        pending = [m for m in eligible if m['ticket']['status'] == 'pending']
        stake = sum(t['stake_yen'] for t in settled); paid = sum(t['payout_yen'] for t in settled)
        hits = sum(m['top_hit'] for m in decided)
        output[name] = {
            'target_races': len(races), 'prediction_saved': sum(m['prediction_saved'] for m in items),
            'eligible_races': len(eligible), 'missing': sum(not m['prediction_saved'] for m in items),
            'held': sum(m['selection_state'] == 'held' for m in items),
            'result_confirmed_races': len(decided), 'hits': hits,
            'hit_rate': hits / len(decided) if decided else None,
            'refunded': sum(m['selection_state'] == 'refunded' for m in eligible),
            'bet_count': len(eligible), 'committed_investment_yen': 100 * len(eligible),
            'settled_count': len(settled), 'unsettled_count': len(pending),
            'investment_yen': stake, 'payout_yen': paid, 'net_profit_yen': paid - stake,
            'return_rate': paid / stake if stake else None,
            'whole_day_payout_yen': None if pending else paid,
            'whole_day_profit_yen': None if pending else paid - stake,
        }
    return output


def attach(forecasts, observations, *, checked_at):
    races = []
    for original in forecasts:
        race = deepcopy(original)
        rows, outcome = observations.get(race['race_id'], ([], None))
        race['watch'] = project(race, rows, outcome, checked_at=checked_at)
        races.append(race)
    return races, scoreboard(races)
