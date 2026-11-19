"""Pure finite Plackett--Luce event probabilities for one complete race.

The common order distribution has no dead heats. These are model-implied
probabilities, not official labels and not products of independent horse events.
Only the first three positions are enumerated, at most 28*27*26 prefixes.
There are no files, database calls, loggers, outcomes, prices, or fitting here.
"""
from __future__ import annotations

from itertools import combinations
import math
from numbers import Real
import re

from ..ml.safety import MLFailure

ASSUMPTION = 'plackett_luce_without_dead_heats'
_HORSE = re.compile(r'(?:0[1-9]|1[0-9]|2[0-8])\Z')


def _conditional(weights, excluded):
    """Scale each remaining field anew; never subtract a dominant horse's mass.

    Normalizing only once can underflow small strengths or suffer cancellation
    when the overwhelmingly strongest horse is removed. The remaining field
    needs its own scale, even when the first-place event rounded to zero.
    """
    remaining = tuple(i for i in range(len(weights)) if i not in excluded)
    scale = max(weights[i] for i in remaining)
    scaled = tuple(weights[i] / scale for i in remaining)
    total = math.fsum(scaled)  # The largest term is exactly one: total is positive.
    return tuple((i, value / total) for i, value in zip(remaining, scaled))


def _bounded(value):
    # Guard mathematical range, allowing only floating point roundoff at one.
    if not math.isfinite(value) or value < 0 or value > 1 + 1e-12:
        raise MLFailure('invalid_input')
    return min(1., value)


def plackett_luce(strengths: dict[str, float], place_paid_positions: int) -> dict:
    """Return coherent win/place/umaren/wide probabilities from positive strengths.

    ``place_paid_positions`` must be the caller's confirmed two/three-payee
    regime. It is never inferred from final or registered field size here.
    Unordered pair keys use ascending, zero-padded numbers, e.g. ``01-02``.

    Win and umaren masses sum to one. Rank-1/2/3 masses each sum to one.
    Place mass sums to the specified payee count; wide mass sums to three,
    because a three-horse podium contains three unordered pairs. These identities
    depend on this explicitly no-dead-heat order model; official outcome and
    refund/special-payment handling belongs to separate code.
    """
    if (not isinstance(strengths, dict) or not 3 <= len(strengths) <= 28
            or type(place_paid_positions) is not int or place_paid_positions not in (2, 3)):
        raise MLFailure('invalid_input')
    if any(type(key) is not str or _HORSE.fullmatch(key) is None for key in strengths):
        raise MLFailure('invalid_identity')
    horses = tuple(sorted(strengths)); weights = []
    for horse in horses:
        value = strengths[horse]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise MLFailure('invalid_input')
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            raise MLFailure('invalid_input') from None
        if not math.isfinite(value) or value <= 0:
            raise MLFailure('invalid_input')
        weights.append(value)
    weights = tuple(weights); n = len(weights)
    pairs = tuple(combinations(range(n), 2))
    rank1 = [0.] * n; rank2 = [[] for _ in range(n)]; rank3 = [[] for _ in range(n)]
    umaren = {pair: [] for pair in pairs}; wide_extra = {pair: [] for pair in pairs}

    for i, first in _conditional(weights, ()):
        rank1[i] = first
        for j, second_given_first in _conditional(weights, (i,)):
            prefix2 = first * second_given_first
            rank2[j].append(prefix2)
            umaren[tuple(sorted((i, j)))].append(prefix2)
            for k, third_given_pair in _conditional(weights, (i, j)):
                prefix3 = prefix2 * third_given_pair
                rank3[k].append(prefix3)
                # The first/second pair is already represented by umaren mass.
                # Each ordered triple contributes its two other wide pairs.
                wide_extra[tuple(sorted((i, k)))].append(prefix3)
                wide_extra[tuple(sorted((j, k)))].append(prefix3)

    rank2 = [math.fsum(v) for v in rank2]; rank3 = [math.fsum(v) for v in rank3]
    r1 = {horse: _bounded(rank1[i]) for i, horse in enumerate(horses)}
    r2 = {horse: _bounded(rank2[i]) for i, horse in enumerate(horses)}
    r3 = {horse: _bounded(rank3[i]) for i, horse in enumerate(horses)}
    place = {horse: _bounded(math.fsum((rank1[i], rank2[i], rank3[i] if place_paid_positions == 3 else 0.)))
             for i, horse in enumerate(horses)}
    pair_mass = {pair: math.fsum(values) for pair, values in umaren.items()}
    pair_names = {pair: horses[pair[0]] + '-' + horses[pair[1]] for pair in pairs}
    umaren_result = {pair_names[pair]: _bounded(pair_mass[pair]) for pair in pairs}
    wide_result = {pair_names[pair]: _bounded(math.fsum((pair_mass[pair], *wide_extra[pair]))) for pair in pairs}
    result = {'win': dict(r1), 'place': place, 'umaren': umaren_result, 'wide': wide_result,
              'rank_1': r1, 'rank_2': r2, 'rank_3': r3, 'assumption': ASSUMPTION}
    for name, expected in (('win', 1.), ('rank_1', 1.), ('rank_2', 1.), ('rank_3', 1.),
                           ('place', float(place_paid_positions)), ('umaren', 1.), ('wide', 3.)):
        if not math.isclose(math.fsum(result[name].values()), expected, rel_tol=0., abs_tol=1e-10):
            raise MLFailure('invalid_input')
    return result
