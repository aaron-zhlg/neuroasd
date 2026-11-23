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

from neuroasd.linear_baseline import tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")

# PRE-REGISTERED, FIXED estimator strength for the logistic swap: no inner-CV, no
# model selection, no best epoch, no test-fold hyperparameter choice. The ledger
# showed the ridge solution is already regularization-saturated, so one strong L2
# value is fixed a priori (d ~ 1.2e5 standardized columns, n ~ 850 training rows).
LOGREG_C = 1e-2


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
    """The one estimator fitted per fold."""
    return L2LogisticClassifier(c=LOGREG_C)


def _covariate_columns(train, test) -> np.ndarray:
    """Age and sex as two extra columns for the ridge input (train, then test rows).

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


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    n_train = len(train.y)
    # Feature block = Gordon tangent space with the HCP tangent space stacked
    # beside it (a single estimator on the concatenated features, no meta-learner).
    gordon = _tangent_block(train, test, "gordon")
    hcp = _tangent_block(train, test, "hcp")
    feats = np.concatenate([gordon, hcp], axis=1)
    # Two extra covariate columns on the tangent features; standardized with
    # training-fold statistics only, so the held-out fold contributes nothing.
    cov = _covariate_columns(train, test).astype(feats.dtype)
    feats = np.concatenate([feats, cov], axis=1)
    # Standardize the whole concatenated block with TRAINING-fold statistics
    # only; the held-out rows are transformed with those same statistics, so the
    # estimator sees exactly the training-fold-scaled feature space.
    mu = feats[:n_train].mean(axis=0, dtype=np.float64).astype(feats.dtype)
    sd = feats[:n_train].std(axis=0, dtype=np.float64).astype(feats.dtype)
    sd[sd == 0] = 1.0
    feats -= mu
    feats /= sd
    y = np.asarray(train.y)
    groups = np.asarray(train.groups)

    # Estimator: a single strongly L2-regularized logistic regression at the
    # pre-registered fixed strength (no CV on any fold, one fit per fold) on the
    # same standardized Gordon+HCP+covariate block the ridge was fitted on.
    clf = _make_estimator().fit(feats[:n_train], y, groups)
    raw_test = clf.decision_function(feats[n_train:])

    # Per-subject decision: the raw decision value cut at 0. Nothing is
    # computed from the other test subjects.
    return raw_test, (raw_test > 0).astype(int)
