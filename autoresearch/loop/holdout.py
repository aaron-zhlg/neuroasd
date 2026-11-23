"""Locked test set: institutions the loop never trains, scores or selects on.

The loop picks the best of many variants on its own LOSO folds, so its scores
are optimistic. These institutions are removed from `autoresearch.trial.Source`
and are scored only by `autoresearch.loop.final_holdout`, a few times in total,
on a model already chosen without them. Only that number is compared with the
papers.

Chosen 2026-11-23 with a fixed seed, not by hand: the non-screen institutions
were shuffled with `numpy.random.default_rng(20261123)` and taken in order until
they held >= 20% of the 1,443 subjects. Result: 7 institutions, 11 sites,
323 subjects (156 ASD / 167 control), ABIDE I and II.

The coder may not edit this file or `autoresearch/trial.py`.
"""

from __future__ import annotations

import numpy as np

from neuroasd.linear_baseline import GROUPINGS

HOLDOUT_SEED = 20261123
HOLDOUT_INSTITUTIONS = ("PITT", "TRINITY", "ABIDEII-IP_1", "ABIDEII-EMC_1", "CALTECH", "LEUVEN", "UCLA")
HOLDOUT_SUBJECTS = 323


def holdout_mask(sites) -> np.ndarray:
    """True for subjects in the locked test set."""
    inst = np.array([GROUPINGS["institution"](s) for s in np.asarray(sites)])
    return np.isin(inst, HOLDOUT_INSTITUTIONS)
