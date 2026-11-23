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

Features and the estimator strength are unchanged. Two training-only knobs now
sit on top of the incumbent:

1. An ASYMMETRIC ASD class weight drawn from the pre-registered grid
   `ASD_WEIGHT_GRID`. Each candidate weight is scored by grouped OUT-OF-FOLD
   decision values of the TRAINING rows only; the winner maximizes OOF ASD
   recall subject to OOF control recall >= `MIN_CONTROL_RECALL`, ties broken
   towards the smaller (more conservative) weight, with weight 1.0 as the
   fallback when no grid point clears the floor.
2. The operating point (one scalar cut) from those same training-fold OOF
   scores, maximizing ASD recall subject to the same control-recall floor.

Both are fit on training rows only and applied to each test subject
independently, so no test score, test label, or test-set statistic of any kind
enters the score or the label.

Per-subject rule: a test subject's score and label may depend only on the
training split and on that subject's own data, never on other test subjects.
The linter checks this by scoring the test set whole and in pieces.

3. STACKING (this iteration). The Gordon+HCP member is no longer the only model:
   a Power tangent member and a Markov tangent member (same recipe: tangent
   space + age/sex, training-only reference/scalers, L2 logistic at C=1e-2, the
   same ASD class weight) are fitted beside it. Their grouped out-of-fold
   decision values are combined by ONE logistic meta-learner at C=1e-2
   (`neuroasd.ensemble.stack_scores`) trained only on training-fold OOF rows;
   its decision values on those rows are the stacked OOF scores, and the SAME
   iter1_a rule is applied to them, so the scalar cut is chosen at 73%
   training-fold control recall in the stacked score's own units. No feature
   columns are concatenated across members: `gordon`+`hcp` remain ONE member
   (the incumbent estimator), Power and Markov are separate models whose
   *scores* are stacked.

Only training rows and training labels decide the members, the meta-learner,
the weight and the cut; test rows are scored subject by subject with those
training-derived scalars. Nothing is a function of the other test subjects.

4. FC ESTIMATE (this iteration). The single change on top of the iter4 stack is
   the connectome estimate that enters every member's tangent map: each
   subject's Pearson correlation matrix is replaced by a Ledoit-Wolf shrunk
   covariance converted back to a correlation, `neuroasd.linear_baseline.
   ledoit_wolf_correlation`, with the closed-form Ledoit-Wolf intensity of that
   subject's own matrix (the effective sample size behind the matrix, which the
   harness does not expose, is estimated per subject from the median of that
   subject's squared off-diagonal correlations). Because the intensity is a
   per-subject
   scalar computable row-by-row, the transform is identical whether a subject is
   scored alone or inside the batch, and the fixed 0.1 identity shrink of the
   tangent map is skipped (`shrinkage=None`). The same estimator is applied to
   every member (Gordon+HCP, Power, Markov) so members still differ only in
   atlas. No feature columns are added, no cut rule is touched, and no test-set
   statistic is used.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from neuroasd.ensemble import average_scores, stack_scores
from neuroasd.linear_baseline import ledoit_wolf_correlation, tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")
# The feature block of the incumbent: Gordon tangent space with the HCP tangent
# space stacked beside it (one estimator on the concatenated features).
FEATURE_ATLASES = ("gordon", "hcp")

# One member per atlas set. `gordon`+`hcp` stay ONE member (column concatenation
# inside a single estimator, exactly the incumbent); Power and Markov are
# separate estimators whose OOF decision values are STACKED with a logistic
# meta-learner. The sets and their order are fixed a priori.
MEMBER_ATLAS_SETS = (FEATURE_ATLASES, ("power",), ("markov",))
# Fixed strength of the meta-learner (same value as the members: no inner-CV,
# no best epoch, no choice against the evaluation folds).
META_C = 1e-2

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
CONTROL_LABEL = 1
FALLBACK_THRESHOLD = 0.0

# PRE-REGISTERED, FIXED grid for the asymmetric ASD class weight (the control
# weight stays 1.0). Declared before any measurement, never extended from the
# scored folds: the choice among these three values is made on grouped OOF
# scores of the TRAINING rows only, weight 1.0 first and 1.0 as fallback.
ASD_WEIGHT_GRID = (1.0, 2.0, 4.0)


class L2LogisticClassifier:
    """Strongly L2-regularized logistic regression with a ridge-compatible interface.

    Drop-in replacement for the former ``KernelRidgeClassifier`` estimator: same
    ``fit(x, y, groups)`` signature (``groups`` is accepted and ignored because the
    penalty is pre-registered, not selected) and the same ``decision_function(x)``
    score interface (higher = more likely control) that the harness consumes.
    Deterministic: L-BFGS with a
    fixed ``random_state``, no subsampling, no randomness in the fit.

    ``asd_weight`` scales only the ASD class (label 0) in the loss; the control
    class (label 1) keeps weight 1.0. ``asd_weight == 1.0`` is the incumbent.
    """

    def __init__(
        self,
        c: float = LOGREG_C,
        asd_weight: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-4,
    ) -> None:
        self.c = c
        self.asd_weight = float(asd_weight)
        self.max_iter = max_iter
        self.tol = tol

    def _class_weight(self):
        if abs(self.asd_weight - 1.0) < 1e-12:
            return None
        return {ASD_LABEL: float(self.asd_weight), CONTROL_LABEL: 1.0}

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> "L2LogisticClassifier":
        self.model_ = LogisticRegression(
            C=self.c,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self._class_weight(),
            random_state=0,
        ).fit(x, y)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(x)


def _make_estimator(asd_weight: float = 1.0) -> L2LogisticClassifier:
    """The one estimator fitted per fold (and per inner grouped fold)."""
    return L2LogisticClassifier(c=LOGREG_C, asd_weight=asd_weight)


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
    # NEW (iter9): the per-subject Pearson FC estimate is replaced by a
    # Ledoit-Wolf shrunk covariance converted back to a correlation before the
    # tangent map. The shrinkage intensity is estimated from each subject's own
    # matrix only (see `ledoit_wolf_correlation`), so this is a per-subject
    # re-parameterization of the same data, not a cross-subject statistic; the
    # fixed 0.1 identity shrink of `tangent_features` is skipped because the
    # matrix is already regularized.
    fcs = ledoit_wolf_correlation(fcs, inplace=True)
    return tangent_features(fcs, np.arange(len(train.y)), shrinkage=None)


def _member_design(train, test, atlases: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """Standardized tangent columns of ONE member plus age/sex; ``(feats, n_train)``.

    Rows are [train rows, test rows]. Tangent references and the column scalers
    come from the training rows only, so the held-out fold contributes nothing
    to the features or the fit. Called once per atlas set (member). Blocking the
    atlases of a *single* member is concatenation inside that member, which is
    what the incumbent does; members themselves are never column-concatenated.
    """
    n_train = len(train.y)
    blocks = [_tangent_block(train, test, atlas) for atlas in atlases]
    cov = _covariate_columns(train, test).astype(blocks[0].dtype)
    feats = np.concatenate(blocks + [cov], axis=1)
    del blocks
    mu = feats[:n_train].mean(axis=0, dtype=np.float64).astype(feats.dtype)
    sd = feats[:n_train].std(axis=0, dtype=np.float64).astype(feats.dtype)
    sd[sd == 0] = 1.0
    feats -= mu
    feats /= sd
    return feats, n_train


def _design(train, test) -> tuple[np.ndarray, int]:
    """The incumbent's Gordon+HCP member design (kept for callers/tests)."""
    return _member_design(train, test, FEATURE_ATLASES)


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


def _recalls(y, scores, cut: float) -> tuple[float, float]:
    """(ASD recall, control recall) on the given scores at one scalar cut."""
    y = np.asarray(y).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    ok = np.isfinite(scores)
    y, scores = y[ok], scores[ok]
    asd, ctl = y == ASD_LABEL, y != ASD_LABEL
    if asd.sum() == 0 or ctl.sum() == 0:
        return 0.0, 0.0
    return float((scores[asd] <= cut).mean()), float((scores[ctl] > cut).mean())


def _oof_scores(feats: np.ndarray, y, groups, asd_weight: float = 1.0) -> np.ndarray:
    """Grouped out-of-fold decision values of the TRAINING rows only.

    Grouped inner folds mirror the outer split (same grouping label); each inner
    fold refits the fixed-C logistic (at this ASD class weight) on its own
    training part and scores the held-out part, so the OOF scores never see
    their own labels. No test row is in ``feats``.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(np.unique(y)) < 2:
        return oof

    splitter = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
    for inner_train, inner_val in splitter.split(np.zeros((n, 1)), y, groups):
        if len(np.unique(y[inner_train])) < 2:
            continue
        clf = _make_estimator(asd_weight).fit(feats[inner_train], y[inner_train], groups[inner_train])
        oof[inner_val] = clf.decision_function(feats[inner_val])
    return oof


def _oof_threshold(feats: np.ndarray, y, groups, asd_weight: float = 1.0) -> float:
    """Operating point from out-of-fold decision values of the TRAINING rows only.

    Falls back to the incumbent cut (0.0) when inner folds are impossible.
    """
    return threshold_from_oof(y, _oof_scores(feats, y, groups, asd_weight))


def select_asd_weight(
    feats: np.ndarray,
    y,
    groups,
    grid: tuple[float, ...] = ASD_WEIGHT_GRID,
) -> tuple[float, float]:
    """Pick the ASD class weight and the cut on TRAINING-fold OOF scores only.

    For every pre-registered weight the grouped OOF decision values are produced
    from the training rows, the incumbent cut rule (`threshold_from_oof`) is
    applied to those same OOF scores, and the resulting OOF ASD / control recalls
    are compared. The winner maximizes ASD recall subject to control recall
    >= `MIN_CONTROL_RECALL`, ties broken towards the smaller weight (the more
    conservative one). If no grid point clears the floor the fallback is weight
    1.0 with its own OOF cut, i.e. exactly the incumbent operating point.

    Only training rows and training labels are used; test rows are not an
    argument, so this cannot leak.
    """
    y = np.asarray(y).ravel()
    baseline: tuple[float, float] | None = None
    best_key: tuple[float, float] | None = None
    best: tuple[float, float] | None = None

    for weight in grid:
        weight = float(weight)
        oof = _oof_scores(feats, y, groups, weight)
        if not np.isfinite(oof).any():
            continue
        cut = threshold_from_oof(y, oof)
        asd_recall, control_recall = _recalls(y, oof, cut)
        if baseline is None:
            baseline = (weight, cut)  # grid starts at 1.0 = incumbent
        if control_recall + 1e-12 < MIN_CONTROL_RECALL:
            continue
        key = (asd_recall, -weight)
        if best_key is None or key > best_key:
            best_key, best = key, (weight, cut)

    if best is not None:
        return best
    if baseline is not None:
        return baseline
    return 1.0, FALLBACK_THRESHOLD


def _member_oof(
    feats: np.ndarray,
    n_train: int,
    y,
    groups,
    asd_weight: float = 1.0,
) -> np.ndarray:
    """Grouped out-of-fold decision values of ONE member, TRAINING rows only.

    The split comes from the training labels/groups alone, so every member of the
    stack is scored on the SAME inner folds and the columns entering the
    meta-learner are paired row-by-row. No test row, label or statistic is used.
    """
    y = np.asarray(y).ravel()
    groups = np.asarray(groups).ravel()
    x_train = feats[:n_train]

    oof = np.full(n_train, np.nan, dtype=float)
    n_groups = len(np.unique(groups))
    if n_groups >= 2 and len(np.unique(y)) >= 2:
        splitter = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
        for inner_train, inner_val in splitter.split(np.zeros((n_train, 1)), y, groups):
            if len(np.unique(y[inner_train])) < 2:
                continue
            clf = _make_estimator(asd_weight).fit(
                x_train[inner_train], y[inner_train], groups[inner_train]
            )
            oof[inner_val] = clf.decision_function(x_train[inner_val])
    return oof


def _member_test_scores(
    feats: np.ndarray,
    n_train: int,
    y,
    groups,
    asd_weight: float = 1.0,
) -> np.ndarray:
    """One member fitted on ALL training rows, scored on the held-out rows."""
    clf = _make_estimator(asd_weight).fit(feats[:n_train], np.asarray(y).ravel(), np.asarray(groups).ravel())
    return clf.decision_function(feats[n_train:])


def _meta_oof(member_oof: np.ndarray, y) -> np.ndarray:
    """The stack's training-fold OOF scores, one scalar per training row.

    The meta-learner is fitted by `stack_scores` on the OOF score columns of the
    TRAINING rows and its decision values on those same rows are the stacked OOF
    scores; the iter1_a weight/cut rule is then applied to them exactly as before
    (73% training-fold control recall), so the operating point lives in the
    stacked score's own units. Members are the only thing stacked: this is not a
    feature-column concatenation. No test row, label or statistic is used.
    """
    y = np.asarray(y).ravel()
    member_oof = np.asarray(member_oof, dtype=float)
    if member_oof.ndim != 2 or member_oof.shape[0] != len(y):
        return np.full(len(y), np.nan, dtype=float)
    if not np.isfinite(member_oof).any():
        return np.full(len(y), np.nan, dtype=float)
    stacked, _ = stack_scores(member_oof, y, member_oof, c=META_C)
    return np.asarray(stacked, dtype=float)


def select_stacked_weight(
    oof_by_weight: dict[float, np.ndarray],
    y,
) -> tuple[float, float]:
    """Pick the ASD class weight and the cut on the STACKED training-fold OOF scores.

    Same rule as `select_asd_weight`, applied to the stacked OOF scores of the
    meta-learner: maximize OOF ASD recall subject to OOF control recall
    >= `MIN_CONTROL_RECALL`, ties broken towards the smaller (more conservative)
    weight, weight 1.0 with its own cut as the fallback. Only training rows and
    training labels are used; test rows are not an argument.
    """
    y = np.asarray(y).ravel()
    baseline: tuple[float, float] | None = None
    best_key: tuple[float, float] | None = None
    best: tuple[float, float] | None = None

    for weight in ASD_WEIGHT_GRID:
        weight = float(weight)
        member_oof = oof_by_weight.get(weight)
        if member_oof is None:
            continue
        stacked = _meta_oof(member_oof, y)
        cut = threshold_from_oof(y, stacked)
        asd_recall, control_recall = _recalls(y, stacked, cut)
        if baseline is None:
            baseline = (weight, cut)  # grid starts at 1.0 = incumbent
        if control_recall + 1e-12 < MIN_CONTROL_RECALL:
            continue
        key = (asd_recall, -weight)
        if best_key is None or key > best_key:
            best_key, best = key, (weight, cut)

    if best is not None:
        return best
    if baseline is not None:
        return baseline
    return 1.0, FALLBACK_THRESHOLD


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    # Members: one per pre-registered atlas set. `gordon`+`hcp` is the incumbent
    # estimator (one logistic on concatenated Gordon+HCP tangent features);
    # Power and Markov are separate estimators on their own tangent space. Each
    # member's features are standardized on training rows only.
    y = np.asarray(train.y).ravel()
    groups = np.asarray(train.groups).ravel()
    weights = [float(w) for w in ASD_WEIGHT_GRID]

    designs: list[tuple[np.ndarray, int]] = []
    oof_cols: dict[float, list[np.ndarray]] = {w: [] for w in weights}
    for atlas_set in MEMBER_ATLAS_SETS:
        feats, n_train = _member_design(train, test, atlas_set)
        for weight in weights:
            oof_cols[weight].append(_member_oof(feats, n_train, y, groups, weight))
        designs.append((feats, n_train))

    oof_matrix = {w: np.column_stack(cols) for w, cols in oof_cols.items()}
    del oof_cols

    # Training-only knobs, now in the STACKED score's own units: the ASD class
    # weight and the single scalar cut come from the meta-learner's OOF scores of
    # the TRAINING rows (test rows are never passed in). Weight 1.0 / cut 0.0
    # remains the fallback, i.e. the incumbent operating point.
    asd_weight, threshold = select_stacked_weight(oof_matrix, y)

    # Held-out scores: each member refit on ALL training rows at the selected
    # weight, then the one meta-learner (fit on the training OOF columns only)
    # applied per subject. The cut is the unchanged iter1_a training-fold OOF
    # rule, used unmapped.
    test_matrix = np.column_stack(
        [_member_test_scores(feats, n_train, y, groups, asd_weight) for feats, n_train in designs]
    )
    del designs
    _, test_scores = stack_scores(oof_matrix[asd_weight], y, test_matrix, c=META_C)
    if not np.isfinite(test_scores).all():
        test_scores = average_scores(test_matrix)

    # Per-subject decision: each test subject is compared with the one scalar cut
    # learned from the training folds. Nothing is computed from the other test
    # subjects, so scoring the test set whole or in pieces gives the same labels.
    return test_scores, (test_scores > threshold).astype(int)
