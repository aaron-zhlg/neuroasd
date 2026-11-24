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

5. UNIVARIATE EDGE SELECTION (this iteration). Inside every fit — each outer
   fold AND each inner-OOF fit — a per-edge two-sample |t| statistic between ASD
   and control rows is computed from THAT fit's TRAINING rows only, the edges are
   ranked by it, and the top-K per member are kept. K is PRE-REGISTERED per
   member in `FEATURE_SELECTION_K` (20,000 for Gordon+HCP, 8,000 for Power,
   12,000 for Markov) and was declared before measurement: it is never tuned on
   OOF, validation or test data, and no other hyperparameter changes. The
   statistic does not depend on the ASD class weight, so ONE selection per fit is
   reused across the `ASD_WEIGHT_GRID` points, but no selection is ever shared
   across folds: each inner fold selects from its own training part, the final
   refit selects from all training rows of the outer fold, and the same mask is
   applied to that fit's validation/test rows. Age and sex are always kept.
   Read-only counters of the LW path, the blocks and the selection are stored in
   the module-level `LAST_FEATURE_COUNTERS` and printed once per `fit_predict`
   call to stderr; they never touch a fitted array, a score, or the cut.

6. SECOND REGULARIZATION SCALE (this iteration, as a STACKED COLUMN). On the
   SAME Gordon+HCP member design (same tangent blocks, same training-only
   standardization, same training-only edge mask) a SECOND, independently
   regularized L2 logistic is fitted at the pre-registered, fixed strength
   `SECOND_MEMBER_C` (1e-1) beside the incumbent `LOGREG_C` (1e-2). Their
   decision values are NOT averaged: each scale is its own column handed to the
   unchanged `stack_scores` meta-learner (META_C=1e-2), so the mixing weight
   between the coarse and the fine regularization is FITTED on training-fold OOF
   rows instead of being hand-set. The meta-learner input is therefore four
   columns -- (Gordon+HCP @ C=1e-2, Gordon+HCP @ C=1e-1, Power, Markov) -- while
   Power and Markov keep exactly one column each. The column layout is identical
   in the inner-OOF path and in the final training refit (no rule asymmetry), the
   ASD class weight, the weight grid, the meta-learner strength and the operating
   point are untouched, and only TRAINING rows enter any fit: no held-out or test
   statistic (no z-scoring, ranking or thresholding across test subjects) enters.

7. NONLINEAR MEMBER, EQUAL-WEIGHT MIXED (this iteration). ONE
   `HistGradientBoostingClassifier` (fixed pre-registered configuration: 60
   iterations, lr 0.1, 8 leaves, min_samples_leaf 15, L2 1.0, 64 bins, no early
   stopping) is fitted on the SAME Gordon+HCP member design through the SAME
   pre-registered |t| ranking: it sees the top-`HGB_FEATURES` (2,000) edges of
   that ranking plus age/sex, with the same training-row standardization. Its
   P(control) is averaged EQUAL-WEIGHT (0.5/0.5, fixed, no fitted mixing weight,
   no calibration) with the incumbent L2 logistic C=1e-2 P(control) of that
   member's first stacked column; the mixed column is emitted in log-odds, a
   strictly monotone transform of the average, so the column keeps the
   incumbent's magnitude for the unchanged `stack_scores` meta-learner. One
   booster is fitted per inner split (its own training rows) and one on all
   training rows of the fold; the ASD class-weight grid keeps acting only on the
   linear columns (the booster is unweighted and therefore fitted once per split,
   not once per grid point). No column is shared across members, the cut rule is
   untouched, and no test row/label/statistic enters.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from neuroasd.ensemble import average_scores, stack_scores
from neuroasd.linear_baseline import (
    LW_LAMBDA_MAX,
    LW_LAMBDA_MIN,
    ledoit_wolf_correlation,
    tangent_features,
)

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

# PRE-REGISTERED, FIXED second regularization scale: a SECOND logistic member on
# the same Gordon+HCP design, fitted at `SECOND_MEMBER_C`. Declared a priori;
# never selected on OOF/validation/test data, never per fold, never tuned. Its
# decision values enter the meta-learner as their OWN column (no hand-set mixing
# weight): the training-fold OOF rows decide how much each scale contributes.
SECOND_MEMBER_C = 1e-1
# Which members get the second scale (only the incumbent Gordon+HCP design).
SECOND_SCALE_ATLAS_SETS = (FEATURE_ATLASES,)

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

# PRE-REGISTERED, FIXED nonlinear member (this iteration). ONE histogram gradient
# boosting classifier is fitted on the SAME Gordon+HCP member design (same
# tangent blocks, same training-only standardization, same training-only edge
# selection, age/sex always kept) and its P(control) is averaged EQUAL-WEIGHT
# (0.5/0.5) with the incumbent L2 logistic C=1e-2 P(control) inside that member's
# first stacked column. Everything below is declared a priori and fixed: no
# inner-CV, no early stopping on any held-out/OOF data, no per-fold choice, no
# tuning against the scored folds. `HGB_FEATURES` bounds the booster to the
# top-M edges of the SAME pre-registered |t| ranking the logistic member already
# uses (M << K), so the extra fits stay inside the stage time budget.
HGB_ATLAS_SETS = (FEATURE_ATLASES,)
HGB_FEATURES = 2000
HGB_MAX_ITER = 60
HGB_LEARNING_RATE = 0.1
HGB_MAX_LEAF_NODES = 8
HGB_MIN_SAMPLES_LEAF = 15
HGB_L2 = 1.0
HGB_MAX_BINS = 64
# The mixer is a FIXED equal-weight average of two probabilities (no fitted
# mixing weight, no calibration). Log-odds are only the units the mixed
# probability is expressed in, so the member column keeps the incumbent's scale
# for the unchanged logistic meta-learner: logit is strictly monotone in the
# average, i.e. the same ordering the average defines.
MIX_WEIGHT = 0.5
MIX_EPS = 1e-6

# PRE-REGISTERED, FIXED per-member edge budgets for the univariate selection.
# Declared a priori (never tuned on OOF/validation/test data), roughly a sixth of
# each atlas' edge count: Gordon+HCP ~119.9k edges -> 20,000; Power 34,716 ->
# 8,000; Markov -> 12,000. `DEFAULT_SELECTION_K` is only a fallback for an atlas
# set without a declared budget.
FEATURE_SELECTION_K = {
    ("gordon", "hcp"): 20000,
    ("power",): 8000,
    ("markov",): 12000,
}
DEFAULT_SELECTION_K = 8000
# Rows gathered per chunk when accumulating the selection statistic.
SELECTION_CHUNK = 64
# Read-only counters of the last fit_predict call (LW path + selection). Written
# once per call, never read back by the model.
LAST_FEATURE_COUNTERS: dict = {}


def _control_column(classes, proba) -> int:
    """Index of the control-probability column of a two-class ``predict_proba``."""
    classes = np.asarray(classes).ravel()
    hit = np.flatnonzero(classes == CONTROL_LABEL)
    return int(hit[0]) if hit.size else int(np.asarray(proba).shape[1] - 1)


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

    def predict_control_proba(self, x: np.ndarray) -> np.ndarray:
        """P(control) per row; used only by the fixed equal-weight mixer."""
        proba = self.model_.predict_proba(x)
        col = _control_column(self.model_.classes_, proba)
        return np.asarray(proba[:, col], dtype=float)


class HistGradientBoostingMember:
    """Fixed-configuration histogram gradient boosting emitting P(control).

    Same ``fit(x, y, groups)`` contract as the linear estimator (``groups`` is
    accepted and ignored: the configuration is pre-registered, not selected) and
    a probability interface. Hyperparameters are module constants — no inner-CV,
    no early stopping against held-out data, no per-fold choice, no random
    subsampling beyond the seeded internal binning.
    """

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> "HistGradientBoostingMember":
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=HGB_LEARNING_RATE,
            max_iter=HGB_MAX_ITER,
            max_leaf_nodes=HGB_MAX_LEAF_NODES,
            min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
            l2_regularization=HGB_L2,
            max_bins=HGB_MAX_BINS,
            early_stopping=False,
            random_state=0,
        ).fit(np.asarray(x), np.asarray(y).ravel())
        return self

    def predict_control_proba(self, x: np.ndarray) -> np.ndarray:
        proba = self.model_.predict_proba(np.asarray(x))
        col = _control_column(self.model_.classes_, proba)
        return np.asarray(proba[:, col], dtype=float)


def _make_hgb() -> HistGradientBoostingMember:
    """The one pre-registered nonlinear member (fixed configuration)."""
    return HistGradientBoostingMember()


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped so the value is always finite."""
    p = np.clip(np.asarray(p, dtype=float), MIX_EPS, 1.0 - MIX_EPS)
    return np.log(p / (1.0 - p))


def _mixed_control_score(p_logreg: np.ndarray, p_hgb: np.ndarray | None) -> np.ndarray:
    """FIXED equal-weight average of the two P(control) values, in log-odds units.

    ``0.5 * (p_logreg + p_hgb)`` is the mixed probability; the returned value is
    its log-odds, a strictly monotone transform of that average, so the member
    column keeps the incumbent's magnitude for the unchanged meta-learner. No
    fitted mixing weight, no calibration, no cross-subject statistic: both
    probabilities for a row come from models fitted on that fit's training rows
    and that row's own features.
    """
    p_logreg = np.asarray(p_logreg, dtype=float).ravel()
    if p_hgb is None:
        return _logit(p_logreg)
    p_hgb = np.asarray(p_hgb, dtype=float).ravel()
    if p_hgb.shape != p_logreg.shape:
        return _logit(p_logreg)
    mixed = np.where(np.isfinite(p_hgb), MIX_WEIGHT * p_logreg + (1.0 - MIX_WEIGHT) * p_hgb, p_logreg)
    return _logit(mixed)


def _make_estimator(asd_weight: float = 1.0, c: float = LOGREG_C) -> L2LogisticClassifier:
    """The one estimator fitted per fold (and per inner grouped fold).

    ``c`` is the L2 strength: the pre-registered incumbent ``LOGREG_C`` or the
    pre-registered second scale ``SECOND_MEMBER_C`` for the extra member on the
    Gordon+HCP design. It is never chosen from data.
    """
    return L2LogisticClassifier(c=float(c), asd_weight=asd_weight)


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


def _tangent_block(train, test, atlas: str, counters: dict | None = None) -> np.ndarray:
    """Tangent-space features for the concatenated train+test rows of one atlas.

    The tangent reference (log-Euclidean mean) is estimated from the TRAINING
    rows only, so the held-out fold contributes no statistics; both `train` and
    `test` expose the same subject order through `fc(atlas)`.

    ``counters`` is read-only bookkeeping for the stderr counter line (LW
    intensities, block shape, finiteness). It never changes a returned value:
    the matrices returned with ``return_lambda=True`` are bit-identical.
    """
    fcs = np.concatenate([train.fc(atlas), test.fc(atlas)])
    # NEW (iter9): the per-subject Pearson FC estimate is replaced by a
    # Ledoit-Wolf shrunk covariance converted back to a correlation before the
    # tangent map. The shrinkage intensity is estimated from each subject's own
    # matrix only (see `ledoit_wolf_correlation`), so this is a per-subject
    # re-parameterization of the same data, not a cross-subject statistic; the
    # fixed 0.1 identity shrink of `tangent_features` is skipped because the
    # matrix is already regularized.
    fcs, lambdas = ledoit_wolf_correlation(fcs, inplace=True, return_lambda=True)
    if counters is not None:
        lw = counters["lw"]
        lambdas = np.asarray(lambdas, dtype=np.float64).ravel()
        finite = lambdas[np.isfinite(lambdas)]
        lw["n_subjects"] += int(lambdas.size)
        lw["n_nonfinite"] += int(lambdas.size - finite.size)
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            lw["lambda_raw_min"] = lo if lw["lambda_raw_min"] is None else min(lw["lambda_raw_min"], lo)
            lw["lambda_raw_max"] = hi if lw["lambda_raw_max"] is None else max(lw["lambda_raw_max"], hi)
        lw["at_min_clip"] += int((lambdas <= LW_LAMBDA_MIN).sum())
        lw["at_max_clip"] += int((lambdas >= LW_LAMBDA_MAX).sum())
    block = tangent_features(fcs, np.arange(len(train.y)), shrinkage=None)
    if counters is not None:
        counters["blocks"][atlas] = {
            "shape": [int(s) for s in block.shape],
            "nonfinite": int(np.count_nonzero(~np.isfinite(block))),
        }
    return block


def _member_design(
    train, test, atlases: tuple[str, ...], counters: dict | None = None
) -> tuple[np.ndarray, int, int]:
    """Standardized tangent columns of ONE member plus age/sex; ``(feats, n_train, n_cov)``.

    Rows are [train rows, test rows]. Tangent references and the column scalers
    come from the training rows only, so the held-out fold contributes nothing
    to the features or the fit. Called once per atlas set (member). Blocking the
    atlases of a *single* member is concatenation inside that member, which is
    what the incumbent does; members themselves are never column-concatenated.
    The last ``n_cov`` columns are age/sex, which the edge selection never drops.
    """
    n_train = len(train.y)
    blocks = [_tangent_block(train, test, atlas, counters) for atlas in atlases]
    cov = _covariate_columns(train, test).astype(blocks[0].dtype)
    n_cov = int(cov.shape[1])
    feats = np.concatenate(blocks + [cov], axis=1)
    del blocks
    mu = feats[:n_train].mean(axis=0, dtype=np.float64).astype(feats.dtype)
    sd = feats[:n_train].std(axis=0, dtype=np.float64).astype(feats.dtype)
    sd[sd == 0] = 1.0
    feats -= mu
    feats /= sd
    return feats, n_train, n_cov


def _design(train, test) -> tuple[np.ndarray, int]:
    """The incumbent's Gordon+HCP member design (kept for callers/tests)."""
    feats, n_train, _ = _member_design(train, test, FEATURE_ATLASES)
    return feats, n_train


def _sum_and_sumsq(feats: np.ndarray, rows: np.ndarray, n_edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Column sums and sums of squares over ``rows`` only (chunked, float64)."""
    rows = np.asarray(rows)
    total = np.zeros(n_edge, dtype=np.float64)
    total_sq = np.zeros(n_edge, dtype=np.float64)
    for start in range(0, len(rows), SELECTION_CHUNK):
        chunk = feats[rows[start : start + SELECTION_CHUNK], :n_edge].astype(np.float64)
        total += chunk.sum(axis=0)
        total_sq += np.einsum("ij,ij->j", chunk, chunk)
    return total, total_sq


def _edge_effect_size(feats: np.ndarray, rows: np.ndarray, y, n_edge: int) -> np.ndarray:
    """|two-sample t| per edge column, computed from the given rows only.

    ``rows`` are ROW INDICES into ``feats``; ``y`` is the label vector of the
    fitting fold. Only the rows passed in are read, so for an inner-OOF fit this
    is that inner fit's own training part and the held-out part never enters.
    """
    rows = np.asarray(rows)
    labels = np.asarray(y).ravel()[rows]
    asd_rows = rows[labels == ASD_LABEL]
    ctl_rows = rows[labels != ASD_LABEL]
    n_asd, n_ctl = len(asd_rows), len(ctl_rows)
    stat = np.zeros(n_edge, dtype=np.float64)
    if n_asd < 2 or n_ctl < 2:
        return stat
    sum_a, sumsq_a = _sum_and_sumsq(feats, asd_rows, n_edge)
    sum_c, sumsq_c = _sum_and_sumsq(feats, ctl_rows, n_edge)
    mean_a, mean_c = sum_a / n_asd, sum_c / n_ctl
    var_a = np.maximum(sumsq_a - sum_a * sum_a / n_asd, 0.0) / (n_asd - 1)
    var_c = np.maximum(sumsq_c - sum_c * sum_c / n_ctl, 0.0) / (n_ctl - 1)
    se = np.sqrt(var_a / n_asd + var_c / n_ctl)
    good = se > 0
    stat[good] = np.abs(mean_a[good] - mean_c[good]) / se[good]
    stat[~np.isfinite(stat)] = 0.0
    return stat


def _select_edges_with_ranking(
    feats: np.ndarray, rows: np.ndarray, y, k: int, n_cov: int
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Top-k edge columns by |t| plus every covariate column, and the ranking.

    Returns ``(mask, kept, total_edges, order)`` where ``order`` is the full
    |t| ranking (descending) of the edge columns computed from ``rows`` only. The
    nonlinear member reuses the SAME ranking at its own pre-registered budget
    ``HGB_FEATURES``, so it never selects on a different statistic.
    """
    n_edge = int(feats.shape[1] - n_cov)
    if n_edge <= 0:
        return np.arange(feats.shape[1], dtype=np.intp), 0, 0, np.zeros(0, dtype=np.intp)
    keep_k = int(min(int(k), n_edge))
    stat = _edge_effect_size(feats, rows, y, n_edge)
    order = np.argsort(-stat, kind="stable")
    keep = np.sort(order[:keep_k]).astype(np.intp)
    cov = np.arange(n_edge, feats.shape[1], dtype=np.intp)
    return np.concatenate([keep, cov]), keep_k, n_edge, order


def _select_edges(
    feats: np.ndarray, rows: np.ndarray, y, k: int, n_cov: int
) -> tuple[np.ndarray, int, int]:
    """Top-k edge columns by |t| plus every covariate column: ``(mask, kept, total)``.

    The rank is computed from ``rows`` only (that fit's training rows), ties are
    broken by column index so the mask is deterministic, and K is the
    pre-registered budget passed in — never selected on any held-out or OOF
    score.
    """
    mask, kept, total, _ = _select_edges_with_ranking(feats, rows, y, k, n_cov)
    return mask, kept, total


def _hgb_columns(order: np.ndarray, n_columns: int, n_cov: int, n_hgb: int) -> np.ndarray:
    """Columns of the nonlinear member: top-M edges of the shared |t| ranking + age/sex.

    ``order`` is the edge ranking already computed for this fit from its own
    training rows; ``M = HGB_FEATURES`` is pre-registered and fixed (never tuned).
    The covariate columns are never dropped, exactly as for the linear member.
    """
    order = np.asarray(order)
    if order.size == 0:
        return np.arange(int(n_columns) - int(n_cov), int(n_columns), dtype=np.intp)
    take = int(min(max(int(n_hgb), 1), order.size))
    edges = np.sort(order[:take]).astype(np.intp)
    cov = np.arange(int(n_columns) - int(n_cov), int(n_columns), dtype=np.intp)
    return np.concatenate([edges, cov])


def _hgb_probability(store, idx: int, inner_val: np.ndarray) -> np.ndarray | None:
    """The pre-computed P(control) of one inner split, if it lines up row-for-row."""
    if store is None or idx >= len(store):
        return None
    val_rows, probs = store[idx]
    if len(val_rows) != len(inner_val) or not np.array_equal(np.asarray(val_rows), np.asarray(inner_val)):
        return None
    return probs


def _inner_folds(n_train: int, y, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """The grouped inner split of the TRAINING rows (mirrors the outer grouping)."""
    y = np.asarray(y).ravel()
    groups = np.asarray(groups).ravel()
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(np.unique(y)) < 2:
        return []
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
    return [(tr, va) for tr, va in splitter.split(np.zeros((n_train, 1)), y, groups)]


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
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    inner_masks: list[np.ndarray],
    asd_weight: float = 1.0,
    c_scales: tuple[float, ...] = (LOGREG_C,),
    hgb_mix_column: int | None = None,
    hgb_inner: list | None = None,
) -> np.ndarray:
    """Grouped out-of-fold decision values of ONE member, TRAINING rows only.

    Returns ``(n_train, len(c_scales))``: ONE COLUMN PER PRE-REGISTERED L2
    STRENGTH. The split comes from the training labels/groups alone, so every
    member of the stack is scored on the SAME inner folds and the columns
    entering the meta-learner are paired row-by-row. No test row, label or
    statistic is used.

    Each inner fit classifies only the edges selected from ITS OWN training part
    (``inner_masks``); the univariate selection does not depend on the ASD class
    weight, so the same masks are reused for every grid point of this fit.

    ``c_scales`` lists the pre-registered L2 strengths of this member: one entry
    is the incumbent single estimator; two entries fit one independently
    regularized logistic per strength on the SAME inner training rows with the
    SAME mask and SAME ASD class weight, and their inner-validation decision
    values are returned as TWO SEPARATE COLUMNS — they are NOT averaged here, so
    the logistic meta-learner (fitted on training-fold OOF rows only) sets how
    much each regularization scale contributes. No hand-set mixing weight.
    """
    y = np.asarray(y).ravel()
    oof = np.full((n_train, len(c_scales)), np.nan, dtype=float)
    for idx, ((inner_train, inner_val), mask) in enumerate(zip(inner_folds, inner_masks)):
        if len(np.unique(y[inner_train])) < 2:
            continue
        x_train = feats[np.ix_(inner_train, mask)]
        x_val = feats[np.ix_(inner_val, mask)]
        hgb_p = _hgb_probability(hgb_inner, idx, inner_val)
        for j, c in enumerate(c_scales):
            clf = _make_estimator(asd_weight, c=float(c)).fit(x_train, y[inner_train], None)
            values = clf.decision_function(x_val)
            if hgb_p is not None and hgb_mix_column is not None and j == int(hgb_mix_column):
                # FIXED equal-weight average of the logistic and the boosting
                # P(control); the booster was fitted on this inner split's own
                # training rows only (same mask ranking, its own column budget).
                values = _mixed_control_score(clf.predict_control_proba(x_val), hgb_p)
            oof[inner_val, j] = values
    return oof


def _member_test_scores(
    feats: np.ndarray,
    n_train: int,
    y,
    mask: np.ndarray,
    asd_weight: float = 1.0,
    c_scales: tuple[float, ...] = (LOGREG_C,),
    hgb_mix_column: int | None = None,
    hgb_test: np.ndarray | None = None,
) -> np.ndarray:
    """One member fitted on ALL training rows, scored on the held-out rows.

    Returns ``(n_test, len(c_scales))``: one column per pre-registered L2
    strength, mirroring exactly the column layout `_member_oof` produced for the
    training rows (no rule asymmetry between the OOF and the refit path).

    ``mask`` is the selection computed from all training rows of this fold, so
    both the fit and the scoring use the same columns. With two ``c_scales`` the
    two independently regularized logistic fits see the identical training rows
    and columns and their held-out decision values stay in SEPARATE COLUMNS (no
    averaging), so the meta-learner trained on the OOF columns applies the same
    fitted mixing weight here; each held-out subject is scored from its own row
    only.
    """
    x_train = feats[np.ix_(np.arange(n_train), mask)]
    x_test = feats[np.ix_(np.arange(n_train, feats.shape[0]), mask)]
    out = np.empty((x_test.shape[0], len(c_scales)), dtype=float)
    for j, c in enumerate(c_scales):
        clf = _make_estimator(asd_weight, c=float(c)).fit(x_train, np.asarray(y).ravel(), None)
        values = clf.decision_function(x_test)
        if hgb_test is not None and hgb_mix_column is not None and j == int(hgb_mix_column):
            # Same fixed equal-weight mixer as the OOF path (no rule asymmetry):
            # the booster was refit on ALL training rows, and each held-out
            # subject is scored from its own row only.
            values = _mixed_control_score(clf.predict_control_proba(x_test), hgb_test)
        out[:, j] = values
    return out


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


def _new_counters() -> dict:
    """Read-only diagnostic counters for ONE fit_predict call."""
    return {
        "lw": {
            "n_subjects": 0,
            "lambda_raw_min": None,
            "lambda_raw_max": None,
            "at_min_clip": 0,
            "at_max_clip": 0,
            "n_nonfinite": 0,
        },
        "clip_bounds": [float(LW_LAMBDA_MIN), float(LW_LAMBDA_MAX)],
        "blocks": {},
        "members": {},
        "scores": {},
    }


def _emit_counters(counters: dict) -> None:
    """Print the counters ONCE per fit_predict call (stderr). Never scored."""
    try:
        payload = json.dumps(counters, sort_keys=True, default=str, allow_nan=False)
    except (TypeError, ValueError):
        payload = json.dumps(counters, sort_keys=True, default=str)
    sys.stderr.write(f"[neuroasd.model] neuroasd.model.LAST_FEATURE_COUNTERS {payload}\n")
    sys.stderr.flush()


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    global LAST_FEATURE_COUNTERS
    counters = _new_counters()

    # Members: one per pre-registered atlas set. `gordon`+`hcp` is the incumbent
    # estimator (one logistic on concatenated Gordon+HCP tangent features);
    # Power and Markov are separate estimators on their own tangent space. Each
    # member's features are standardized on training rows only.
    y = np.asarray(train.y).ravel()
    groups = np.asarray(train.groups).ravel()
    weights = [float(w) for w in ASD_WEIGHT_GRID]
    n_rows_train = len(y)
    inner_folds = _inner_folds(n_rows_train, y, groups)

    designs: list[dict] = []
    oof_cols: dict[float, list[np.ndarray]] = {w: [] for w in weights}
    for atlas_set in MEMBER_ATLAS_SETS:
        feats, n_train, n_cov = _member_design(train, test, atlas_set, counters)
        name = "+".join(atlas_set)
        k = int(FEATURE_SELECTION_K.get(tuple(atlas_set), DEFAULT_SELECTION_K))
        # PRE-REGISTERED second regularization scale: only the Gordon+HCP member
        # carries a second, independently regularized logistic (C=1e-1) beside the
        # incumbent C=1e-2. Both scales use this fold's mask and the same ASD
        # class weight, so nothing is selected from data; their decision values
        # stay in TWO SEPARATE META-LEARNER COLUMNS (never pre-averaged), so the
        # mixing weight is FITTED by `stack_scores` on training-fold OOF rows.
        c_scales = (
            (LOGREG_C, SECOND_MEMBER_C) if tuple(atlas_set) in SECOND_SCALE_ATLAS_SETS else (LOGREG_C,)
        )
        # PRE-REGISTERED K, selected inside this fit only: one mask per inner-OOF
        # fit (from that inner fit's training part) and one for the final refit
        # (from all training rows of this outer fold). Weight-independent, so the
        # three grid points share them; folds never do.
        inner_selections = [
            _select_edges_with_ranking(feats, inner_train, y, k, n_cov) for inner_train, _ in inner_folds
        ]
        inner_masks = [sel[0] for sel in inner_selections]
        final_mask, kept, total, final_order = _select_edges_with_ranking(
            feats, np.arange(n_train), y, k, n_cov
        )
        # PRE-REGISTERED nonlinear member: ONE histogram gradient boosting fit per
        # inner split (on that split's own training rows and its own pre-registered
        # top-M subset of the SAME |t| ranking) plus one refit on all training rows
        # of this fold. Its P(control) is mixed equal-weight with the incumbent
        # L2 logistic P(control) inside this member's first stacked column. The
        # booster is deliberately NOT weighted by the ASD class-weight grid, so it
        # is fitted ONCE per split and the grid keeps acting on the linear columns
        # exactly as before; nothing is fitted on held-out or test rows.
        hgb_mix_column = None
        hgb_inner = None
        hgb_test = None
        n_columns = int(feats.shape[1])
        n_rows = int(feats.shape[0])
        if tuple(atlas_set) in HGB_ATLAS_SETS:
            hgb_masks = [_hgb_columns(sel[3], n_columns, n_cov, HGB_FEATURES) for sel in inner_selections]
            hgb_inner = []
            for (inner_train, inner_val), hgb_mask in zip(inner_folds, hgb_masks):
                probs = np.full(len(inner_val), np.nan, dtype=float)
                if len(np.unique(y[inner_train])) >= 2:
                    booster = _make_hgb().fit(feats[np.ix_(inner_train, hgb_mask)], y[inner_train])
                    probs = booster.predict_control_proba(feats[np.ix_(inner_val, hgb_mask)])
                hgb_inner.append((np.asarray(inner_val), probs))
            final_hgb_mask = _hgb_columns(final_order, n_columns, n_cov, HGB_FEATURES)
            if len(np.unique(y)) >= 2:
                booster = _make_hgb().fit(feats[np.ix_(np.arange(n_train), final_hgb_mask)], y)
                hgb_test = booster.predict_control_proba(
                    feats[np.ix_(np.arange(n_train, n_rows), final_hgb_mask)]
                )
            c_list = [float(c) for c in c_scales]
            hgb_mix_column = c_list.index(float(LOGREG_C)) if float(LOGREG_C) in c_list else 0
        del inner_selections
        counters["members"][name] = {
            "edges_total": int(total),
            "edge_budget_k": int(k),
            "edges_kept": int(kept),
            "inner_selections": int(len(inner_masks)),
            "selected_inner_columns_first": int(len(inner_masks[0])) if inner_masks else 0,
            "n_train_rows": int(n_train),
            "n_test_rows": int(feats.shape[0] - n_train),
            "columns_total": int(feats.shape[1]),
            "columns_kept": int(len(final_mask)),
            "nonfinite_features": int(np.count_nonzero(~np.isfinite(feats))),
            "c_scales": [float(c) for c in c_scales],
            "hgb_features": int(HGB_FEATURES) if hgb_mix_column is not None else 0,
            "hgb_mix_column": None if hgb_mix_column is None else int(hgb_mix_column),
            "hgb_test_rows_scored": 0 if hgb_test is None else int(np.size(hgb_test)),
        }
        for weight in weights:
            oof_cols[weight].append(
                _member_oof(
                    feats,
                    n_train,
                    y,
                    inner_folds,
                    inner_masks,
                    weight,
                    c_scales,
                    hgb_mix_column=hgb_mix_column,
                    hgb_inner=hgb_inner,
                )
            )
        designs.append(
            {
                "feats": feats,
                "n_train": n_train,
                "mask": final_mask,
                "c_scales": c_scales,
                "hgb_mix_column": hgb_mix_column,
                "hgb_test": hgb_test,
            }
        )

    # `cols` holds one (n_train, n_scales) block per member, so the meta-learner
    # input becomes ONE COLUMN PER (member, pre-registered L2 scale): C=1e-2 and
    # C=1e-1 for Gordon+HCP, then Power, then Markov -> FOUR columns. No hand-set
    # average of the two Gordon+HCP scales is ever formed; the column count is
    # derived from the blocks rather than hard-coded.
    oof_matrix = {w: np.column_stack(cols) for w, cols in oof_cols.items()}
    del oof_cols
    n_meta_columns = int(next(iter(oof_matrix.values())).shape[1])

    # Training-only knobs, now in the STACKED score's own units: the ASD class
    # weight and the single scalar cut come from the meta-learner's OOF scores of
    # the TRAINING rows (test rows are never passed in). Weight 1.0 / cut 0.0
    # remains the fallback, i.e. the incumbent operating point.
    asd_weight, threshold = select_stacked_weight(oof_matrix, y)

    # Held-out scores: each member refit on ALL training rows at the selected
    # weight (with its own training-row selection mask), then the one
    # meta-learner (fit on the training OOF columns only) applied per subject.
    # The cut is the unchanged iter1_a training-fold OOF rule, used unmapped.
    test_matrix = np.column_stack(
        [
            _member_test_scores(
                d["feats"],
                d["n_train"],
                y,
                d["mask"],
                asd_weight,
                d["c_scales"],
                hgb_mix_column=d["hgb_mix_column"],
                hgb_test=d["hgb_test"],
            )
            for d in designs
        ]
    )
    del designs
    _, test_scores = stack_scores(oof_matrix[asd_weight], y, test_matrix, c=META_C)
    used_fallback = False
    if not np.isfinite(test_scores).all():
        test_scores = average_scores(test_matrix)
        used_fallback = True

    counters["scores"] = {
        "nonfinite_test_scores": int(np.count_nonzero(~np.isfinite(test_scores))),
        "average_fallback_used": bool(used_fallback),
        "meta_columns": int(n_meta_columns),
        "test_columns": int(test_matrix.shape[1]),
        "asd_weight": float(asd_weight),
        "cut": float(threshold),
    }
    LAST_FEATURE_COUNTERS = counters
    _emit_counters(counters)

    # Per-subject decision: each test subject is compared with the one scalar cut
    # learned from the training folds. Nothing is computed from the other test
    # subjects, so scoring the test set whole or in pieces gives the same labels.
    return test_scores, (test_scores > threshold).astype(int)

