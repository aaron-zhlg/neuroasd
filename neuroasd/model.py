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

Base recipe: iter10 (`iter10-loso-institution-v1`, PR #18) scored per subject.
Gordon+HCP tangent concatenation, age/sex, L2 logistic at fixed C=1e-2, and the
raw decision value cut at 0. The per-group score z-scoring of iter10 is removed:
it used the unlabeled scores of the whole held-out site, which the cross-site
papers do not do.

This version keeps the feature block and the estimator exactly as they are and
changes only the operating point: the cut comes from grouped OUT-OF-FOLD
decision values of the TRAINING rows (`_oof_threshold`), maximizing ASD recall
subject to control recall >= `MIN_CONTROL_RECALL` on those training-fold OOF
scores. The scored scalar is then applied to each test subject independently, so
no test score, test label, or test-set statistic of any kind enters it.

Per-subject rule: a test subject's score and label may depend only on the
training split and on that subject's own data, never on other test subjects.
The linter checks this by scoring the test set whole and in pieces.

Feature concatenation is not stacking. For a true ensemble (one model per
atlas / feature type, optional GNN member, OOF logistic meta-learner or
score averaging) import `neuroasd.ensemble` and change one mechanism. The
scored path remains `fit_predict`.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from neuroasd.linear_baseline import tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")
# The feature block of the incumbent: Gordon tangent space with the HCP tangent
# space stacked beside it (one estimator on the concatenated features).
FEATURE_ATLASES = ("gordon", "hcp")

# PRE-REGISTERED, FIXED estimator strength for the logistic swap: no inner-CV, no
# model selection, no best epoch, no test-fold hyperparameter choice. The ledger
# showed the ridge solution is already regularization-saturated, so one strong L2
# value is fixed a priori (d ~ 1.2e5 standardized columns, n ~ 850 training rows).
LOGREG_C = 1e-2

# Operating-point selection: grouped inner folds over the TRAINING rows only, one
# scalar cut, control recall floor as in gates.json.
INNER_FOLDS = 5
MIN_CONTROL_RECALL = 0.73
ASD_LABEL = 0
FALLBACK_THRESHOLD = 0.0


class L2LogisticClassifier:
    """Strongly L2-regularized logistic regression with a ridge-compatible interface.

    Drop-in replacement for the former ``KernelRidgeClassifier`` estimator: same
    ``fit(x, y, groups)`` signature (``groups`` is accepted and ignored because the
    penalty is pre-registered, not selected) and the same ``decision_function(x)``
    score interface (higher = more likely control) that the harness consumes.
    Deterministic: L-BFGS with a
    fixed ``random_state``, no subsampling, no randomness in the fit.
    """

    def __init__(self, c: float = LOGREG_C, max_iter: int = 2000, tol: float = 1e-4) -> None:
        self.c = c
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> "L2LogisticClassifier":
        self.model_ = LogisticRegression(
            C=self.c,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=0,
        ).fit(x, y)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(x)


def _make_estimator() -> L2LogisticClassifier:
    """The one estimator fitted per fold (and per inner grouped fold)."""
    return L2LogisticClassifier(c=LOGREG_C)


def _covariate_columns(train, test) -> np.ndarray:
    """Age and sex as two extra columns for the design (train, then test rows).

    Age is centered/scaled with TRAINING-fold mean/SD only; non-finite ages fall
    back to the training mean. The binary sex column follows the contract fitted
    on the training rows (the higher/female code observed there maps to 1, any
    code unseen in training maps to 0), so no test-fold statistic or label enters.
    Convention documented by the harness: sex 1 = male, 2 = female.
    """
    n_train = len(train.y)
    age = np.concatenate([np.asarray(train.age).ravel(), np.asarray(test.age).ravel()]).astype(float)
    sex = np.concatenate([np.asarray(train.sex).ravel(), np.asarray(test.sex).ravel()])

    mean_age = float(np.nanmean(age[:n_train]))
    std_age = float(np.nanstd(age[:n_train]))
    if not np.isfinite(mean_age):
        mean_age = 0.0
    if not np.isfinite(std_age) or std_age <= 1e-8:
        std_age = 1.0
    age = np.where(np.isfinite(age), age, mean_age)

    if np.issubdtype(sex.dtype, np.number):
        sex_num = sex.astype(float)
        tr_vals = np.unique(sex_num[:n_train][np.isfinite(sex_num[:n_train])])
        female = float(tr_vals[-1]) if tr_vals.size else None
        sex_col = (sex_num == female).astype(float) if female is not None else np.zeros(len(sex))
    else:
        tr_vals = sorted(set(sex[:n_train].tolist()))
        female = tr_vals[-1] if tr_vals else None
        sex_col = (sex == female).astype(float) if female is not None else np.zeros(len(sex))

    return np.column_stack([(age - mean_age) / std_age, sex_col])


def _tangent_block(train, test, atlas: str) -> np.ndarray:
    """Tangent-space features for the concatenated train+test rows of one atlas.

    The tangent reference (log-Euclidean mean) is estimated from the TRAINING
    rows only, so the held-out fold contributes no statistics; both `train` and
    `test` expose the same subject order through `fc(atlas)`.
    """
    fcs = np.concatenate([train.fc(atlas), test.fc(atlas)])
    return tangent_features(fcs, np.arange(len(train.y)))


def _design(train, test) -> tuple[np.ndarray, int]:
    """Standardized Gordon+HCP tangent columns plus age/sex; ``(feats, n_train)``.

    Rows are [train rows, test rows]. Tangent references and the column scalers
    come from the training rows only, so the held-out fold contributes nothing
    to the features or the fit.
    """
    n_train = len(train.y)
    blocks = [_tangent_block(train, test, atlas) for atlas in FEATURE_ATLASES]
    cov = _covariate_columns(train, test).astype(blocks[0].dtype)
    feats = np.concatenate(blocks + [cov], axis=1)
    del blocks
    mu = feats[:n_train].mean(axis=0, dtype=np.float64).astype(feats.dtype)
    sd = feats[:n_train].std(axis=0, dtype=np.float64).astype(feats.dtype)
    sd[sd == 0] = 1.0
    feats -= mu
    feats /= sd
    return feats, n_train


def threshold_from_oof(y, scores, min_control_recall: float = MIN_CONTROL_RECALL) -> float:
    """Decision cut (higher score = control) from TRAINING-fold OOF scores.

    Maximize ASD recall subject to control recall >= ``min_control_recall``;
    among cuts reaching the same ASD recall take the one closest to 0 (the
    incumbent cut), which is also the most control-conservative end of that
    plateau. Only the vectors passed in are used: no test score, no test label.
    """
    y = np.asarray(y).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    ok = np.isfinite(scores)
    y, scores = y[ok], scores[ok]
    asd, ctl = y == ASD_LABEL, y != ASD_LABEL
    if asd.sum() == 0 or ctl.sum() == 0:
        return FALLBACK_THRESHOLD

    unique = np.unique(scores)
    candidates = np.concatenate([unique[:1] - 1.0, 0.5 * (unique[:-1] + unique[1:]), unique[-1:] + 1.0])
    best_key: tuple[float, float] | None = None
    best_cut = FALLBACK_THRESHOLD
    for cut in candidates:
        cut = float(cut)
        if float((scores[ctl] > cut).mean()) < min_control_recall:
            continue
        key = (float((scores[asd] <= cut).mean()), -abs(cut))
        if best_key is None or key > best_key:
            best_key, best_cut = key, cut
    return best_cut


def _oof_threshold(feats: np.ndarray, y, groups) -> float:
    """Operating point from out-of-fold decision values of the TRAINING rows only.

    Grouped inner folds mirror the outer split (same grouping label); each inner
    fold refits the fixed-C logistic on its own training part and scores the held
    out part, so the OOF scores never see their own labels. The features come
    from the training-fold tangent reference/scalers already (see `_design`), and
    no test row is in ``feats``. Falls back to the incumbent cut (0.0) when inner
    folds are impossible.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(np.unique(y)) < 2:
        return FALLBACK_THRESHOLD

    oof = np.full(n, np.nan, dtype=float)
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
    for inner_train, inner_val in splitter.split(np.zeros((n, 1)), y, groups):
        if len(np.unique(y[inner_train])) < 2:
            continue
        clf = _make_estimator().fit(feats[inner_train], y[inner_train], groups[inner_train])
        oof[inner_val] = clf.decision_function(feats[inner_val])
    return threshold_from_oof(y, oof)


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    # Features: Gordon+HCP tangent concat plus age/sex, standardized on the
    # training rows only (unchanged from the incumbent).
    feats, n_train = _design(train, test)
    y = np.asarray(train.y)
    groups = np.asarray(train.groups)

    # Operating point: chosen from training-fold out-of-fold decision values only.
    threshold = _oof_threshold(feats[:n_train], y, groups)

    # Estimator: a single strongly L2-regularized logistic regression at the
    # pre-registered fixed strength on the training rows (unchanged).
    clf = _make_estimator().fit(feats[:n_train], y, groups)
    raw_test = clf.decision_function(feats[n_train:])

    # Per-subject decision: each test subject is compared with the one scalar cut
    # learned from the training folds. Nothing is computed from the other test
    # subjects, so scoring the test set whole or in pieces gives the same labels.
    return raw_test, (raw_test > threshold).astype(int)
