"""Fabricated fields and exhaustive order enumeration; no I/O, private data, fits."""
from itertools import combinations, permutations
import math
import unittest

from jra_racing_ai.ml.safety import MLFailure
from jra_racing_ai.multiwager.probabilities import ASSUMPTION, plackett_luce


def exhaustive(strengths, paid):
    """Small-field oracle enumerates *full* order, independently of prefix code."""
    horses = tuple(sorted(strengths))
    result = {name: {h: 0. for h in horses} for name in ('win', 'place', 'rank_1', 'rank_2', 'rank_3')}
    for name in ('umaren', 'wide'):
        result[name] = {'-'.join(pair): 0. for pair in combinations(horses, 2)}
    total = 0.
    for order in permutations(horses):
        mass = 1.
        for index, horse in enumerate(order):
            mass *= strengths[horse] / sum(strengths[h] for h in order[index:])
        total += mass
        result['win'][order[0]] += mass
        for index in range(3):
            result[f'rank_{index + 1}'][order[index]] += mass
        for horse in order[:paid]:
            result['place'][horse] += mass
        result['umaren']['-'.join(sorted(order[:2]))] += mass
        for pair in combinations(order[:3], 2):
            result['wide']['-'.join(sorted(pair))] += mass
    assert abs(total - 1) < 1e-12
    return result


class SharedOrderProbabilityTests(unittest.TestCase):
    def assert_matches_oracle(self, strengths, paid):
        got = plackett_luce(strengths, paid); expected = exhaustive(strengths, paid)
        for name, values in expected.items():
            self.assertEqual(set(values), set(got[name]))
            for key, value in values.items():
                self.assertAlmostEqual(got[name][key], value, places=13)

    def test_four_horse_full_order_oracle_three_payees(self):
        self.assert_matches_oracle({'01': 1., '02': 2., '03': 3., '04': 5.}, 3)

    def test_five_horse_full_order_oracle_two_payees(self):
        self.assert_matches_oracle({'01': .2, '02': .5, '03': 1., '04': 3., '05': 10.}, 2)

    def test_uniform_field_closed_forms(self):
        n = 6; field = {f'{h:02d}': 1. for h in range(1, n + 1)}
        for k in (2, 3):
            got = plackett_luce(field, k)
            for horse in field:
                self.assertAlmostEqual(got['win'][horse], 1 / n)
                self.assertAlmostEqual(got['place'][horse], k / n)
                for rank in (1, 2, 3): self.assertAlmostEqual(got[f'rank_{rank}'][horse], 1 / n)
            for key in got['umaren']:
                self.assertAlmostEqual(got['umaren'][key], 1 / math.comb(n, 2))
                self.assertAlmostEqual(got['wide'][key], 3 / math.comb(n, 2))

    def test_two_and_three_payees_are_distinct_without_changing_pair_events(self):
        field = {'01': 1., '02': 2., '03': 3., '04': 4.}
        two = plackett_luce(field, 2); three = plackett_luce(field, 3)
        self.assertEqual(two['win'], three['win']); self.assertEqual(two['umaren'], three['umaren'])
        self.assertEqual(two['wide'], three['wide'])
        for h in field:
            self.assertAlmostEqual(three['place'][h] - two['place'][h], three['rank_3'][h])

    def test_all_probability_sum_identities(self):
        got = plackett_luce({f'{h:02d}': float(h * h) for h in range(1, 19)}, 3)
        for name in ('win', 'rank_1', 'rank_2', 'rank_3', 'umaren'):
            self.assertAlmostEqual(math.fsum(got[name].values()), 1., places=13)
        self.assertAlmostEqual(math.fsum(got['place'].values()), 3., places=13)
        self.assertAlmostEqual(math.fsum(got['wide'].values()), 3., places=13)

    def test_three_horses_all_wide_pairs_are_certain(self):
        got = plackett_luce({'01': 1., '02': 2., '03': 30.}, 3)
        for p in got['wide'].values(): self.assertAlmostEqual(p, 1.)
        for p in got['place'].values(): self.assertAlmostEqual(p, 1.)

    def test_pair_probability_is_not_product_of_independent_marginals(self):
        got = plackett_luce({'01': 1., '02': 2., '03': 3., '04': 4.}, 3)
        self.assertNotAlmostEqual(got['umaren']['01-02'], got['win']['01'] * got['win']['02'])
        self.assertNotAlmostEqual(got['wide']['01-02'], got['place']['01'] * got['place']['02'])

    def test_relative_scale_invariance(self):
        field = {'01': .1, '02': .2, '03': .4, '04': .8}
        got = plackett_luce(field, 2)
        for scale in (1e-200, 1e200):
            other = plackett_luce({h: v * scale for h, v in field.items()}, 2)
            for name in ('win', 'place', 'umaren', 'wide', 'rank_1', 'rank_2', 'rank_3'):
                for key, value in got[name].items(): self.assertAlmostEqual(value, other[name][key], places=14)

    def test_insertion_order_does_not_change_values_or_pair_key_order(self):
        field = {'04': 4., '02': 2., '03': 3., '01': 1.}
        a = plackett_luce(field, 3); b = plackett_luce(dict(reversed(list(field.items()))), 3)
        self.assertEqual(a, b)
        self.assertEqual(list(a['umaren']), ['01-02', '01-03', '01-04', '02-03', '02-04', '03-04'])

    def test_dominant_horse_does_not_cancel_remaining_probability_mass(self):
        # A once-normalized vector followed by total-minus-first would lose the field.
        got = plackett_luce({'01': 1e308, '02': 1e-308, '03': 2e-308}, 2)
        self.assertEqual(got['win']['01'], 1.)
        self.assertAlmostEqual(got['rank_2']['02'], 1 / 3)
        self.assertAlmostEqual(got['rank_2']['03'], 2 / 3)
        self.assertAlmostEqual(got['rank_3']['02'], 2 / 3)
        self.assertAlmostEqual(got['rank_3']['03'], 1 / 3)
        self.assertAlmostEqual(got['umaren']['01-02'], 1 / 3)
        self.assertAlmostEqual(got['umaren']['01-03'], 2 / 3)

    def test_large_equal_strengths_do_not_overflow_total(self):
        got = plackett_luce({'01': 1e308, '02': 1e308, '03': 1e308}, 3)
        self.assertAlmostEqual(got['win']['01'], 1 / 3)
        self.assertAlmostEqual(got['wide']['01-02'], 1.)

    def test_smallest_positive_strengths_work(self):
        got = plackett_luce({'01': 5e-324, '02': 5e-324, '03': 5e-324}, 2)
        self.assertAlmostEqual(got['place']['01'], 2 / 3)

    def test_twenty_eight_runner_limit_and_all_pair_count(self):
        field = {f'{h:02d}': float(h) for h in range(1, 29)}
        got = plackett_luce(field, 3)
        self.assertEqual(len(got['wide']), math.comb(28, 2))
        self.assertAlmostEqual(math.fsum(got['wide'].values()), 3.)

    def test_assumption_explicit_and_alias_maps_independent(self):
        got = plackett_luce({'01': 1, '02': 2, '03': 3}, 3)
        self.assertEqual(got['assumption'], ASSUMPTION)
        value = got['rank_1']['01']; got['win']['01'] = 0.
        self.assertEqual(got['rank_1']['01'], value)


class InvalidProbabilityInputTests(unittest.TestCase):
    def test_unsupported_field_sizes_rejected(self):
        for n in (0, 1, 2, 29):
            with self.subTest(n=n), self.assertRaises(MLFailure):
                plackett_luce({f'{h:02d}': 1 for h in range(1, n + 1)}, 3)

    def test_payee_regime_must_be_confirmed_integer_two_or_three(self):
        field = {'01': 1., '02': 2., '03': 3.}
        for k in (None, '', '3', 1, 4, 2., True, False):
            with self.subTest(k=k), self.assertRaises(MLFailure): plackett_luce(field, k)

    def test_nonfinite_nonpositive_and_boolean_strengths_rejected(self):
        for value in (0., -1., float('nan'), float('inf'), -float('inf'), True, False, None, '1', 10**400):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(MLFailure):
                plackett_luce({'01': value, '02': 2., '03': 3.}, 3)

    def test_invalid_horse_keys_rejected_without_normalization(self):
        for key in ('1', '00', '29', ' 01', '01 ', '０１', 1, None, '01-02'):
            with self.subTest(key=key), self.assertRaises(MLFailure):
                plackett_luce({key: 1., '02': 2., '03': 3.}, 3)

    def test_colliding_zero_padded_alias_is_not_silently_combined(self):
        with self.assertRaises(MLFailure): plackett_luce({'01': 1., '1': 2., '02': 3., '03': 4.}, 3)

    def test_mapping_or_list_is_not_guessed_as_a_field(self):
        for value in (None, [], [('01', 1.), ('02', 2.), ('03', 3.)]):
            with self.assertRaises(MLFailure): plackett_luce(value, 3)

    def test_errors_are_fixed_codes_without_values(self):
        with self.assertRaises(MLFailure) as error:
            plackett_luce({'SECRET_HORSE_OR_PASSWORD': 987654321., '02': 2., '03': 3.}, 3)
        self.assertEqual(str(error.exception), 'invalid_identity')
        self.assertNotIn('SECRET', repr(error.exception))


if __name__ == '__main__':
    unittest.main()
