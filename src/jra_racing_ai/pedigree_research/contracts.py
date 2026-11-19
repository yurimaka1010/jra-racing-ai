"""Small, versioned research contract; this module grants no training permission."""
from ..d39_enrichment import contracts as enriched
from ..ml import safety

BASE = enriched.BASE
ROOT = BASE / 'pedigree-research'
INITIAL = enriched.PARENT
REFERENCE = enriched.folder('f581701e5453474988425affca1b8faa')
VERSION = 'pedigree_offspring_balanced_7day_v1'
FEATURES = tuple(f'p_{role}_{name}' for role in ('sire', 'damsire')
                 for name in ('level', 'condition_deviation', 'support'))
CELLS = tuple(f'{surface}:{band}' for surface in ('turf', 'dirt')
              for band in ('le1400', 'gt1400_le1800', 'gt1800'))
read = enriched.read
sha = enriched.sha
clock = enriched.clock


def write(path, value):
    safety.write_json(path, value, root=ROOT)


def spec():
    return {'formula_version': VERSION, 'features': list(FEATURES),
        'categorical_features': [], 'lag_days': 7,
        'score': '(actual_starters-finish)/(actual_starters-1);higher_is_better',
        'conditions': list(CELLS), 'unsupported_conditions': 'prior_level;zero_deviation;zero_support',
        'history': 'whole_clean_races;all_available_progeny;own_horse_excluded',
        'weighting': 'mean_within_each_offspring_then_equal_mean_across_offspring',
        'prior': 'expanding_mean_of_offspring_means_excluding_target;fallback_0.5',
        'overall_shrinkage_offspring': 10., 'cell_shrinkage_offspring': 5.,
        'strength_selection': 'fixed_a_priori;no_empirical_or_future_fitting',
        'support': 'distinct_other_offspring_in_current_cell',
        'identity': 'UM_ancestor_01a_and_05a;HN_link;no_name_matching',
        'existing_features_replaced': False, 'production_connected': False, 'real_fits': 0}
