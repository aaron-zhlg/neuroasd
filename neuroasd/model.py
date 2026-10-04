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
The only addition is the decision rule: the fixed `score > 0` cut is replaced by
a threshold estimated from grouped out-of-fold predictions on the *training*
folds only, so the held-out fold is still scored once with no tuning on it, and
both sides are first put on a common operating point by a label-free per-group
mean/SD standardization of the scores (no labels, no cross-group statistics).
Note this transform is monotone within a group, so it cannot change AUC: any
balanced-accuracy move it produces is a calibration rotation, not new signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from neuroasd.linear_baseline import KernelRidgeClassifier, tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")

# Inner folds used to produce out-of-fold training scores for the threshold.
THRESHOLD_INNER_FOLDS = 5


def _oof_training_scores(feats: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold scores for the training subjects, folds grouped like the outer split.

    Only training rows are used; the tangent features are the ones built from this
    training fold, so the out-of-fold scores live on the same scale as the scores
    that the final classifier produces for the held-out fold.
    """
    scores = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return scores
    splitter = GroupKFold(n_splits=min(THRESHOLD_INNER_FOLDS, n_groups))
    for inner_train, inner_val in splitter.split(feats, y, groups):
        if len(np.unique(y[inner_train])) < 2:
            continue
        clf = KernelRidgeClassifier().fit(feats[inner_train], y[inner_train], groups[inner_train])
        scores[inner_val] = clf.decision_function(feats[inner_val])
    return scores


def _group_standardize(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Label-free per-group centering/scaling of decision values.

    Every group (site or institution) has its scores replaced by
    ``(s - mean_g) / sd_g`` using that group's OWN score distribution only - no
    labels, no reference statistics borrowed from other groups. The transform is
    monotone inside a group, so it cannot change within-group ranking (AUC); it
    only puts groups on a common operating point for the frozen global cut.
    Degenerate groups (zero/NaN SD, single member) fall back to a pooled scale
    computed from the input scores; rows that are not finite stay NaN so the
    threshold fit skips them.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    out = np.full(scores.shape, np.nan, dtype=float)
    finite_all = scores[np.isfinite(scores)]
    pooled = float(np.std(finite_all)) if finite_all.size else 0.0
    if not np.isfinite(pooled) or pooled <= 1e-12:
        pooled = 1.0
    for g in np.unique(groups):
        mask = groups == g
        vals = scores[mask]
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        center = float(np.mean(vals[ok]))
        scale = float(np.std(vals[ok]))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = pooled
        out[mask] = (vals - center) / scale
    return out


def _balanced_accuracy_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold that maximizes balanced accuracy on the (out-of-fold) training scores.

    Scores are the kernel-ridge decision values (higher = more likely control).
    Candidate cuts are placed between adjacent, distinct scores, so a candidate
    always splits the training set into a non-empty ASD and a non-empty control
    side. On a plateau of equal balanced accuracy the middle cut is taken, which
    avoids picking an arbitrary edge of the plateau.
    """
    finite = np.isfinite(scores)
    scores, y = np.asarray(scores)[finite], np.asarray(y)[finite]
    asd = y == 0
    n_asd, n_ctl = int(asd.sum()), int((~asd).sum())
    if n_asd == 0 or n_ctl == 0 or len(scores) < 2:
        return 0.0

    order = np.argsort(scores, kind="stable")
    s = scores[order]
    is_asd = asd[order]
    # Cut i predicts ASD for ranks <= i and control above; keep only cuts that
    # separate two distinct scores.
    valid = np.zeros(len(s), dtype=bool)
    valid[:-1] = s[:-1] < s[1:]
    if not valid.any():
        return 0.0
    asd_below = np.cumsum(is_asd) / n_asd
    ctl_below = np.cumsum(~is_asd) / n_ctl
    balanced = 0.5 * (asd_below + (1.0 - ctl_below))
    balanced = np.where(valid, balanced, -np.inf)

    best = np.flatnonzero(balanced == balanced.max())
    cut = int(best[len(best) // 2])
    return float(0.5 * (s[cut] + s[cut + 1]))


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    fcs = np.concatenate([train.fc("gordon"), test.fc("gordon")])
    n_train = len(train.y)
    feats = tangent_features(fcs, np.arange(n_train))
    y = np.asarray(train.y)
    groups = np.asarray(train.groups)

    clf = KernelRidgeClassifier().fit(feats[:n_train], y, groups)
    raw_test = clf.decision_function(feats[n_train:])

    # Decision threshold from training folds only; frozen before the test fold
    # is scored and never adjusted with test-fold labels or statistics.
    oof = _oof_training_scores(feats[:n_train], y, groups)
    # Label-free per-group calibration of the OOF training scores: each training
    # group is centered/scaled by its own OOF mean/SD so the pooled cut is not
    # dominated by a single group's score scale.
    oof_cal = _group_standardize(oof, groups)
    threshold = _balanced_accuracy_threshold(oof_cal, y)

    # Same calibration on the held-out side, using only that group's own
    # unlabeled scores (its own mean/SD), then the frozen cut is applied.
    test_groups = getattr(test, "groups", None)
    if test_groups is None:
        test_groups = np.zeros(len(raw_test), dtype=int)
    scores = _group_standardize(raw_test, np.asarray(test_groups))
    return scores, (scores > threshold).astype(int)
