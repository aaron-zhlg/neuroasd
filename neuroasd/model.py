"""The model the autoresearch loop improves.

`autoresearch/trial.py` calls `fit_predict(train, test)` once per held-out fold.
It returns a score per test subject (higher = more likely control) and a hard
label (0 = ASD, 1 = control). AUC is computed from the scores; accuracy,
balanced accuracy, and the two recalls are computed from the labels.

Inputs come only from the harness. `train` and `test` expose:

    fc(atlas)   FC matrices (n, p, p), float32, NaN-free; atlas in ATLASES
    age, sex    phenotype arrays (sex: 1 = male, 2 = female)
    groups      fold grouping labels (site or institution, same as the outer split)
    sites       SITE_ID per subject
    y           labels, train split only

The starting point is the formal baseline (experiments/linear_baseline_dcan_v1):
tangent-space features on the Gordon atlas plus kernel ridge classification.
"""

from __future__ import annotations

import numpy as np

from neuroasd.linear_baseline import KernelRidgeClassifier, tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    fcs = np.concatenate([train.fc("gordon"), test.fc("gordon")])
    n_train = len(train.y)
    feats = tangent_features(fcs, np.arange(n_train))
    clf = KernelRidgeClassifier().fit(feats[:n_train], train.y, train.groups)
    scores = clf.decision_function(feats[n_train:])
    return scores, (scores > 0).astype(int)
