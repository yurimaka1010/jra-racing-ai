"""Streaming event-level probability quality, separate from return calculations."""
from array import array
import math

from ..ml.safety import MLFailure

EDGES = (0., .01, .02, .05, .1, .2, .3, .5, 1.)


class Scores:
    def __init__(self):
        self.p = array('d'); self.y = bytearray(); self.loss = 0.; self.brier = 0.
        self.races = 0; self.top_hits = 0; self.race_loss = 0.
        self.bins = [[0, 0., 0] for _ in range(len(EDGES) - 1)]
        self.expected_count = 0.; self.observed_count = 0.

    def add(self, probabilities, labels):
        if not probabilities or set(probabilities) != set(labels):
            raise MLFailure('invalid_identity')
        positive = 0
        for key in sorted(probabilities):
            p = probabilities[key]; y = labels[key]
            if (isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                    or type(y) is not int or y not in (0, 1)):
                raise MLFailure('invalid_input')
            q = max(1e-15, min(1 - 1e-15, p)); loss = -y * math.log(q) - (1 - y) * math.log1p(-q)
            self.p.append(p); self.y.append(y); self.loss += loss; self.brier += (p - y) ** 2
            positive += y
            index = next(i for i in range(len(EDGES) - 1) if EDGES[i] <= p and (p < EDGES[i+1] or EDGES[i+1] == 1))
            b = self.bins[index]; b[0] += 1; b[1] += p; b[2] += y
        best = min(probabilities, key=lambda key: (-probabilities[key], key))
        self.top_hits += labels[best]; self.races += 1
        self.expected_count += sum(probabilities.values()); self.observed_count += positive

    def result(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        n = len(self.y); y = np.frombuffer(self.y, dtype=np.uint8)
        return {'races': self.races, 'targets': n, 'positive_targets': int(y.sum()),
            'binary_log_loss': self.loss / n if n else None, 'brier_score': self.brier / n if n else None,
            'roc_auc': float(roc_auc_score(y, np.asarray(self.p))) if n and set(y) == {0, 1} else None,
            'top_probability_target_hit_rate_percent': 100 * self.top_hits / self.races if self.races else None,
            'mean_probability_sum_per_race': self.expected_count / self.races if self.races else None,
            'mean_positive_targets_per_race': self.observed_count / self.races if self.races else None,
            'calibration': [{'lower': EDGES[i], 'upper': EDGES[i+1], 'targets': b[0],
                'mean_probability': b[1]/b[0] if b[0] else None, 'observed_rate': b[2]/b[0] if b[0] else None}
                for i, b in enumerate(self.bins)], 'label_basis': 'unique_top3_rank_reference_pending_official_HR',
            'ROI': None, 'ROI_reason': 'official_multiwager_payout_flags_units_versions_not_connected'}
