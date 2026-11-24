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

7. NONLINEAR MEMBER ON ITS OWN FEATURE SPACE (this iteration). ONE
   `HistGradientBoostingClassifier` (fixed pre-registered configuration: 60
   iterations, lr 0.1, 8 leaves, min_samples_leaf 15, L2 1.0, 64 bins, no early
   stopping) is fitted, but NOT on the Gordon+HCP member design: it consumes the
   Power and Markov tangent EDGE blocks of the same subject, concatenated into
   ONE design (each block standardized on training rows only, age/sex appended
   once), reduced by ITS OWN top-`HGB_FEATURES` (2,000) |t| ranking computed
   inside each fit from that fit's training rows only — a different input space
   from the edges the four linear columns already linearize. If that block ever
   has fewer edge columns than the pre-registered K, every edge is used and the
   truncation is recorded in the read-only counters.

   Its P(control), expressed in LOG-ODDS, is ONE NEW FINAL COLUMN of the meta
   design matrix -- it is NOT mixed into any other column. The four PR #25
   columns (Gordon+HCP @ C=1e-2, Gordon+HCP @ C=1e-1, Power, Markov) are emitted
   exactly as before, so column 0 is again the pure C=1e-2 logistic log-odds and
   its ordering is monotone in that logistic's own score. The booster column is
   appended in the SAME last position in the inner-OOF path and in the held-out
   path, and the unchanged linear logistic meta-learner (`META_C=1e-2`, fitted by
   `ensemble.stack_scores` on training-fold OOF rows only) then learns its OWN
   weight on the booster instead of a hand-set 50/50 probability average: the
   mixing weight is fitted on training rows, not pre-registered. One booster is
   fitted per inner split (its own training rows and its own top-K subset of its
   own |t| ranking) and one on all training rows of the fold; the ASD class
   weight grid keeps acting only on the linear columns (the booster is unweighted
   and therefore fitted once per split, not once per grid point). Rows whose inner
   split could not be fitted keep the neutral log-odds 0.0 (P=0.5) so no training
   row is dropped from the meta fit. Read-only stderr diagnostics record whether
   the booster fitted per split and the meta-learner's learned weight on its
   column. No column is shared across members, the cut rule is untouched, and no
   test row/label/statistic enters.

8. L1 (SPARSE) PENALTY ON THE INCUMBENT MEMBER ONLY (this iteration). The ONE
   change of this round is the REGULARIZER of the estimator that produces stacked
   column 0 -- the Gordon+HCP member at the incumbent strength `L1_C =
   LOGREG_C = 1e-2`. It is an L1-penalized logistic (`penalty="l1"`,
   `solver="liblinear"` in this sklearn build, where the L2 path uses
   `penalty="l2"`, `solver="lbfgs"`) at the SAME fixed strength, on the SAME
   design: same log-Euclidean tangent map, same training-row-only
   standardization, same training-fold-only |t| top-K mask, same age/sex
   columns, same `ASD_WEIGHT_GRID`. An L1 penalty drives the majority of the
   ~20k masked edge coefficients to exactly zero, so the most heavily weighted
   member becomes a SPARSE edge model; if dense edge coefficients are the source
   of site-level variance this raises cross-site ranking (AUC) rather than
   moving the cut. Nothing is selected from data: penalty, strength, solver and
   iteration caps are pre-registered constants.

   FROZEN and unchanged: column 1 (Gordon+HCP at `SECOND_MEMBER_C = 1e-1`, still
   L2/lbfgs), column 2 (Power), column 3 (Markov), column 4 (the fixed-config
   booster), the column count and order, the boosters' feature space and
   hyperparameters, `META_C`, `stack_scores`, `threshold_from_oof`,
   `select_stacked_weight`, the 73% training-fold OOF control-recall floor and
   hence the cut, the top-K masking rule, the tangent map, the ASD weight grid
   and per-subject held-out scoring. The L1 estimator is used identically in the
   OOF path and in the held-out path (same penalty, same strength, same
   standardization, same mask), and every fit sees only its own training rows.

   READ-ONLY DIAGNOSTICS (never a fitted array, score or cut): the number of
   non-zero L1 coefficients of the column-0 refit, per-slot fitted-vs-neutral
   status for every ASD weight point and for the held-out matrix, the linear
   members' and the booster's K_requested/K_used/K_truncated, the selected ASD
   weight, and the meta-learner's coefficient vector by slot plus intercept are
   emitted as ONE stderr line prefixed `DIAG_JSON`. The harness result record
   reads no notes field (`autoresearch/trial.py` is read-only and writes only
   fixed keys), so the stderr line is the channel that carries them.

9. NONLINEAR DEMOGRAPHIC BASIS (this iteration). The ONE change is a NEW
   INFORMATION term inside the always-kept covariate block, not a new meta
   column: a pre-registered, fixed basis `COVARIATE_BASIS = (age, sex, age_sq,
   age_x_sex)` (`COVARIATE_BASIS_VERSION = 2`). The incumbent age and sex columns
   keep their positions exactly as before and TWO columns are APPENDED after
   them -- age^2 and age x sex -- each computed from that subject's own age/sex
   values only and standardized with the mean/SD of that fit's OWN training rows
   (zero-variance guard returns zeros, never raises). The block is applied
   identically to all four linear member designs and to the booster's design (it
   reuses the member covariate block), in the inner-OOF fits and in each fold's
   final refit. The block stays ALWAYS KEPT: `n_cov` grows from 2 to 4 and the
   top-K |t| mask is computed on the edge columns only, so the edge budget, the
   column count (still 5 meta columns) and their order are unchanged. Rationale:
   every member is a logistic model that is linear in age, so a nonlinear age
   effect and a sex-specific age slope are inputs no existing column can express.
   FROZEN: `threshold_from_oof`, `select_stacked_weight`, the 73% floor and hence
   the cut, `META_C`/`stack_scores`, the column layout, the booster and its
   hyperparameters, column 0's L1 estimator, the L2 columns, the top-K rule, the
   tangent map and per-subject held-out scoring. The `DIAG_JSON` stderr line also
   reports the basis names and each split's covariate width.
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

# PRE-REGISTERED, FIXED L1 (sparse) penalty for the INCUMBENT regularization slot.
# THIS ITERATION'S ONE CHANGE: the estimator producing stacked column 0 --
# Gordon+HCP at the incumbent strength -- now uses an L1 penalty at the SAME
# fixed strength `L1_C = LOGREG_C` instead of L2. The design (tangent map,
# training-row standardization, training-fold-only |t| top-K mask, age/sex), the
# ASD class-weight grid, the column layout and everything downstream are
# untouched; only the penalty of that one member changes. Penalty, strength,
# solver and iteration caps are declared a priori: never selected on OOF,
# validation, test or held-out data, never per fold. `liblinear` is the solver
# used for the L1 penalty in this sklearn build (the L2 path keeps `lbfgs`).
L1_MEMBER_ATLAS_SETS = (FEATURE_ATLASES,)
L1_PENALTY = "l1"
L1_C = LOGREG_C
L1_SOLVER = "liblinear"
L1_MAX_ITER = 200
L1_TOL = 1e-4
# The L2 penalty string of the unchanged columns (columns 1-4).
L2_PENALTY = "l2"

# PRE-REGISTERED, FIXED nonlinear demographic basis (THIS ITERATION'S ONE
# MECHANISM). The always-kept covariate block of EVERY member design (and of the
# booster's design, which reuses the same block) gains exactly TWO new columns
# beside the incumbent age and sex: age^2 and age x sex. The four linear member
# columns are logistic models that are linear in age, so a nonlinear age effect
# and a sex-specific age slope are inputs no existing column can express; this is
# a small, fixed, site-independent information addition. Both new columns come
# from that subject's own age/sex values only and are standardized with the
# mean/SD of that fit's OWN training rows (zero-variance guard -> zeros, never a
# raise). Names, ordering and count are declared a priori and never tuned; the
# existing age and sex columns keep their positions 0 and 1, and the two new
# columns are appended AFTER them, always kept (never subject to the top-K |t|
# edge mask, which lives on the edge columns only).
COVARIATE_BASIS = ("age", "sex", "age_sq", "age_x_sex")
COVARIATE_BASIS_VERSION = 2

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

# PRE-REGISTERED, FIXED nonlinear member. ONE histogram gradient boosting
# classifier is fitted on ITS OWN training-only-standardized feature space and
# its P(control) LOG-ODDS becomes ONE extra, FINAL column of the meta design
# matrix, so the unchanged logistic meta-learner fits its own weight on the
# booster (no hand-set probability average, nothing mixed into another column).
# Everything below is declared a priori and fixed: no inner-CV, no early
# stopping on any held-out/OOF data, no per-fold choice, no tuning against the
# scored folds. `HGB_FEATURES` bounds the booster to the top-K edges of its own
# |t| ranking (K << the edge count), so the extra fits stay inside the stage time
# budget.
# THIS ITERATION'S ONE CHANGE: the booster's FEATURE SPACE. Every prior run fed
# it the top-|t| slice of the Gordon+HCP member -- the very edges the linear
# columns already linearize -- and its ranking gain was flat at screen, loso-site
# and loso-institution. The unclosed variant is a different input space: the
# booster now consumes the Power and Markov tangent EDGE blocks of the same
# subject, concatenated into ONE design with age/sex appended once, reduced by
# ITS OWN training-fold-only top-K |t| ranking. Named by member name; the only
# member whose edges are NOT used is the incumbent Gordon+HCP block.
HGB_FEATURE_MEMBERS = ("power", "markov")
HGB_FEATURES = 2000
HGB_MAX_ITER = 60
HGB_LEARNING_RATE = 0.1
HGB_MAX_LEAF_NODES = 8
HGB_MIN_SAMPLES_LEAF = 15
HGB_L2 = 1.0
HGB_MAX_BINS = 64
# The booster is NOT mixed into any other column: its P(control) is emitted in
# log-odds as ONE extra, LAST column of the meta design matrix and the unchanged
# linear logistic meta-learner fits its own weight on it. `LOGIT_EPS` only clips
# probabilities so the log-odds are finite; it changes no ordering.
LOGIT_EPS = 1e-6

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
# READ-ONLY: the sparse-column diagnostics of the last fit_predict call. Written
# once per call, never read back by the model, never used for a score or a cut.
LAST_DIAGNOSTICS: dict = {}


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
        """P(control) per row (probability interface, symmetric with the booster)."""
        proba = self.model_.predict_proba(x)
        col = _control_column(self.model_.classes_, proba)
        return np.asarray(proba[:, col], dtype=float)


class L1LogisticClassifier:
    """L1-penalized (SPARSE) logistic regression, same interface as the L2 member.

    THIS ITERATION'S ONE MECHANISM. It is fitted on exactly the same design, with
    exactly the same ASD class weight and the same pre-registered strength as the
    incumbent `L2LogisticClassifier`; only the penalty (and therefore the solver
    that supports it, `L1_SOLVER = "liblinear"`) differs. The L1 penalty sets
    most of the ~20k masked edge coefficients to exactly zero, so the member
    becomes a sparse edge model instead of a dense one -- a regularization
    (denoising) change, not a new feature family.

    Same ``fit(x, y, groups)`` signature (``groups`` accepted and ignored: the
    penalty is pre-registered, never selected) and the same
    ``decision_function(x)`` score interface (higher = more likely control) the
    harness consumes. Deterministic: fixed solver, fixed seed, no subsampling.
    """

    def __init__(
        self,
        c: float = L1_C,
        asd_weight: float = 1.0,
        max_iter: int = L1_MAX_ITER,
        tol: float = L1_TOL,
    ) -> None:
        self.c = float(c)
        self.asd_weight = float(asd_weight)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _class_weight(self):
        if abs(self.asd_weight - 1.0) < 1e-12:
            return None
        return {ASD_LABEL: float(self.asd_weight), CONTROL_LABEL: 1.0}

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> "L1LogisticClassifier":
        self.model_ = LogisticRegression(
            C=self.c,
            penalty=L1_PENALTY,
            solver=L1_SOLVER,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self._class_weight(),
            random_state=0,
        ).fit(x, y)
        # READ-ONLY bookkeeping: how many coefficients the sparse fit kept.
        self.n_nonzero_coef_ = int(np.count_nonzero(np.asarray(self.model_.coef_)))
        self.n_coef_total_ = int(np.asarray(self.model_.coef_).size)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(x)

    def predict_control_proba(self, x: np.ndarray) -> np.ndarray:
        """P(control) per row (probability interface, symmetric with the L2 member)."""
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
    """Log-odds of a probability, clipped so the value is always finite.

    Used ONLY to express the booster's P(control) in the units of the linear
    member columns: the logit is strictly monotone in P(control), so it adds no
    ordering of its own and mixes with nothing.
    """
    p = np.clip(np.asarray(p, dtype=float).ravel(), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _make_estimator(
    asd_weight: float = 1.0, c: float = LOGREG_C, penalty: str = L2_PENALTY
) -> L2LogisticClassifier | L1LogisticClassifier:
    """The one estimator fitted per fold (and per inner grouped fold).

    ``c`` is the regularization strength: the pre-registered incumbent
    ``LOGREG_C`` or the pre-registered second scale ``SECOND_MEMBER_C`` for the
    extra member on the Gordon+HCP design. It is never chosen from data.

    ``penalty`` is the pre-registered penalty of that meta column: ``"l1"`` only
    for the incumbent Gordon+HCP slot (meta column 0, strength ``L1_C ==
    LOGREG_C``), ``"l2"`` for every other column, which keeps the unchanged L2
    estimator bit-identical (same class, same solver, same caps). Never selected
    from data.
    """
    if str(penalty) == L1_PENALTY:
        return L1LogisticClassifier(c=float(c), asd_weight=asd_weight)
    return L2LogisticClassifier(c=float(c), asd_weight=asd_weight)


def _train_standardize(values: np.ndarray, n_train: int) -> np.ndarray:
    """Standardize a covariate column with the TRAINING rows' mean/SD only.

    Reuses the training-row convention of `_covariate_columns`: the statistic is
    computed on ``values[:n_train]`` (that fit's own training rows, held-out rows
    never contribute), non-finite entries are replaced by the training mean and a
    zero/degenerate variance returns an all-zero column instead of raising. No
    cross-subject test statistic is used: each row's value is a function of that
    row's own age/sex only.
    """
    values = np.asarray(values, dtype=float).ravel()
    n_train = int(n_train)
    ref = values[:n_train]
    ref_finite = ref[np.isfinite(ref)]
    mean = float(ref_finite.mean()) if ref_finite.size else 0.0
    std = float(ref_finite.std()) if ref_finite.size else 0.0
    if not np.isfinite(mean):
        mean = 0.0
    values = np.where(np.isfinite(values), values, mean)
    if not np.isfinite(std) or std <= 1e-8:  # zero-variance guard: zeros, never raise
        return np.zeros(values.shape[0], dtype=float)
    return (values - mean) / std


def _covariate_columns(train, test) -> np.ndarray:
    """Age, sex and the fixed nonlinear basis as extra columns (train, then test).

    The always-kept block is ``COVARIATE_BASIS = (age, sex, age_sq, age_x_sex)``
    (``COVARIATE_BASIS_VERSION = 2``): the incumbent age and sex columns keep
    their positions 0 and 1 exactly as before, and the TWO new columns age^2 and
    age x sex are appended after them. Every column comes from that subject's own
    age/sex values only and is standardized with the TRAINING-fold mean/SD (zero
    variance -> zeros). Non-finite ages fall back to the training mean. The binary
    sex column follows the contract fitted on the training rows (the
    higher/female code observed there maps to 1, any code unseen in training maps
    to 0), so no test-fold statistic or label enters.
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

    # THIS ITERATION'S ONE MECHANISM: two new ALWAYS-KEPT covariate columns built
    # from each subject's own age/sex only, standardized on this fit's training
    # rows only (zero-variance -> zeros). `age` here is the training-mean imputed
    # age in years, so age^2 is the genuine nonlinear demographic term and
    # age x sex gives the sex-specific age slope the linear columns cannot express.
    age_sq = _train_standardize(age * age, n_train)
    age_x_sex = _train_standardize(age * sex_col, n_train)
    return np.column_stack([(age - mean_age) / std_age, sex_col, age_sq, age_x_sex])


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
    The last ``n_cov`` columns are the always-kept covariate block
    ``COVARIATE_BASIS = (age, sex, age_sq, age_x_sex)`` (width 4), which the edge
    selection never drops: the top-K |t| mask lives on the edge columns only, so
    its budget is unchanged.
    """
    n_train = len(train.y)
    blocks = [_tangent_block(train, test, atlas, counters) for atlas in atlases]
    cov = _covariate_columns(train, test).astype(blocks[0].dtype)
    n_cov = int(cov.shape[1])
    if counters is not None:
        # READ-ONLY: the pre-registered covariate basis and this split's width.
        basis = counters.setdefault(
            "covariates",
            {
                "basis": list(COVARIATE_BASIS),
                "version": int(COVARIATE_BASIS_VERSION),
                "widths": {},
            },
        )
        basis["widths"]["+".join(atlases)] = int(n_cov)
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
    |t| ranking (descending) of the edge columns computed from ``rows`` only.
    The nonlinear member uses the SAME function on its OWN design at its own
    pre-registered budget ``HGB_FEATURES``, so it selects on the same statistic
    but never on the linear member's ranking and never on other rows.
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
    """Columns of the nonlinear member: top-M edges of ITS OWN |t| ranking + covariates.

    ``order`` is the edge ranking of the BOOSTER'S OWN feature space (the
    concatenated Power+Markov tangent edge blocks), computed for this fit from
    this fit's training rows only; ``M = HGB_FEATURES`` is pre-registered and
    fixed (never tuned). When that block has fewer edge columns than ``M`` every
    edge is used (``take`` is clamped) and the truncation is recorded in the
    read-only counters. The covariate columns are never dropped, exactly as for
    the linear member.
    """
    order = np.asarray(order)
    if order.size == 0:
        return np.arange(int(n_columns) - int(n_cov), int(n_columns), dtype=np.intp)
    take = int(min(max(int(n_hgb), 1), order.size))
    edges = np.sort(order[:take]).astype(np.intp)
    cov = np.arange(int(n_columns) - int(n_cov), int(n_columns), dtype=np.intp)
    return np.concatenate([edges, cov])


def _hgb_column_oof(
    feats: np.ndarray,
    n_train: int,
    y,
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    inner_selections: list[tuple],
    n_cov: int,
    diag: dict,
) -> np.ndarray:
    """Booster P(control) LOG-ODDS for the training rows, as ONE meta column.

    Each inner split gets its OWN booster, fitted on that split's training rows
    and on its own pre-registered top-`HGB_FEATURES` subset of the |t| ranking of
    those same rows; the split's held-out part is scored from that booster only,
    so the column is out-of-fold with respect to the training rows and no row is
    scored by a model that saw its label. The value is the logit of P(control) --
    a strictly monotone transform, so this column orders rows exactly as the
    booster does.

    Rows of a split that could not be fitted keep the neutral log-odds 0.0
    (P = 0.5): the meta fit is not restricted to the remaining rows for that
    reason, and the read-only ``diag`` records how many splits fired.
    """
    y = np.asarray(y).ravel()
    column = np.zeros(int(n_train), dtype=float)
    n_columns = int(feats.shape[1])
    diag["inner_splits_total"] = int(len(inner_folds))
    diag["inner_splits_fitted"] = 0
    diag["inner_splits_single_class"] = 0
    diag["inner_splits_failed"] = 0
    diag["inner_rows_fit"] = None
    diag["inner_rows_scored"] = None
    diag["n_features_used"] = None
    masks = [_hgb_columns(sel[3], n_columns, n_cov, HGB_FEATURES) for sel in inner_selections]
    for (inner_train, inner_val), mask in zip(inner_folds, masks):
        if len(np.unique(y[inner_train])) < 2:
            diag["inner_splits_single_class"] += 1
            continue
        try:
            booster = _make_hgb().fit(feats[np.ix_(inner_train, mask)], y[inner_train])
            probs = booster.predict_control_proba(feats[np.ix_(inner_val, mask)])
        except Exception:  # pragma: no cover - defensive; a fold must never abort
            diag["inner_splits_failed"] += 1
            continue
        column[inner_val] = _logit(probs)
        diag["inner_splits_fitted"] += 1
        diag["inner_rows_fit"] = int(len(inner_train))
        diag["inner_rows_scored"] = int(len(inner_val))
        diag["n_features_used"] = int(len(mask))
    return column


def _hgb_column_test(
    feats: np.ndarray,
    n_train: int,
    y,
    final_order: np.ndarray,
    n_cov: int,
    diag: dict,
) -> np.ndarray:
    """Booster P(control) LOG-ODDS for the held-out rows (SAME last column slot).

    One booster refit on ALL training rows of the fold, on the top-`HGB_FEATURES`
    subset of the |t| ranking of those same rows; every held-out subject is scored
    from its own row only, so nothing here depends on the other held-out subjects.
    """
    n_train = int(n_train)
    n_rows = int(feats.shape[0])
    n_test = n_rows - n_train
    diag["final_fit"] = False
    if n_test <= 0 or len(np.unique(np.asarray(y).ravel())) < 2:
        return np.zeros(max(n_test, 0), dtype=float)
    mask = _hgb_columns(final_order, int(feats.shape[1]), n_cov, HGB_FEATURES)
    try:
        booster = _make_hgb().fit(feats[np.ix_(np.arange(n_train), mask)], np.asarray(y).ravel())
        probs = booster.predict_control_proba(feats[np.ix_(np.arange(n_train, n_rows), mask)])
    except Exception:  # pragma: no cover - defensive; a fold must never abort
        diag["final_failed"] = True
        return np.zeros(max(n_test, 0), dtype=float)
    diag["final_fit"] = True
    diag["final_rows_fit"] = int(n_train)
    diag["final_rows_scored"] = int(n_test)
    diag["final_n_features_used"] = int(len(mask))
    return _logit(probs)


def _meta_column_weights(oof_matrix: np.ndarray, y) -> list[float] | None:
    """READ-ONLY coefficients of the meta-learner that `stack_scores` fits.

    Mirrors `stack_scores`' fit exactly (same finite rows, same ``META_C``, same
    solver/seed, training rows only) and returns the learned weight of every meta
    column so the diagnostic can report how much weight the booster column got.
    The values returned here are NEVER used for a score or a cut: the scored model
    keeps the coefficients `stack_scores` itself fitted.
    """
    oof_matrix = np.asarray(oof_matrix, dtype=float)
    y = np.asarray(y).ravel()
    if oof_matrix.ndim != 2 or oof_matrix.shape[0] != len(y):
        return None
    ok = np.isfinite(oof_matrix).all(axis=1) & np.isfinite(y)
    if int(ok.sum()) < 2 or len(np.unique(y[ok])) < 2:
        return None
    model = LogisticRegression(
        C=META_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(oof_matrix[ok], y[ok])
    return [float(v) for v in np.asarray(model.coef_).ravel()]


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
    penalties: tuple[str, ...] | None = None,
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

    ``penalties`` is the pre-registered penalty of each column, aligned with
    ``c_scales``: ``"l1"`` for the incumbent Gordon+HCP slot of this iteration
    (sparse estimator, same strength, same design, same mask, same weight) and
    ``"l2"`` for every unchanged column. The same penalty tuple is used in the
    held-out path (`_member_test_scores`), so the sparse estimator is not
    treated differently out-of-fold.
    """
    y = np.asarray(y).ravel()
    n_scales = len(c_scales)
    if penalties is None:
        penalties = (L2_PENALTY,) * n_scales
    penalties = tuple(str(p) for p in penalties)
    if len(penalties) != n_scales:  # pragma: no cover - defensive
        penalties = (L2_PENALTY,) * n_scales
    oof = np.full((n_train, n_scales), np.nan, dtype=float)
    for (inner_train, inner_val), mask in zip(inner_folds, inner_masks):
        if len(np.unique(y[inner_train])) < 2:
            continue
        x_train = feats[np.ix_(inner_train, mask)]
        x_val = feats[np.ix_(inner_val, mask)]
        # CUT FIX (a): re-standardize with THIS inner split's training rows
        # only. `feats` was scaled with all training rows of the fold, which
        # includes this split's validation rows; the final refit is scaled with
        # exactly its own training rows, so the OOF path now mirrors it.
        mu = x_train.mean(axis=0, dtype=np.float64).astype(x_train.dtype)
        sd = x_train.std(axis=0, dtype=np.float64).astype(x_train.dtype)
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        x_train = (x_train - mu) / sd
        x_val = (x_val - mu) / sd
        for j, c in enumerate(c_scales):
            clf = _make_estimator(asd_weight, c=float(c), penalty=penalties[j]).fit(
                x_train, y[inner_train], None
            )
            oof[inner_val, j] = clf.decision_function(x_val)
        del x_train, x_val
    return oof


def _member_test_scores(
    feats: np.ndarray,
    n_train: int,
    y,
    mask: np.ndarray,
    asd_weight: float = 1.0,
    c_scales: tuple[float, ...] = (LOGREG_C,),
    penalties: tuple[str, ...] | None = None,
    diag: dict | None = None,
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

    ``penalties`` is the pre-registered penalty per column (``"l1"`` for the
    incumbent Gordon+HCP slot of this iteration, ``"l2"`` elsewhere), used
    IDENTICALLY to the OOF path. ``diag`` only receives the read-only non-zero
    coefficient count of the sparse fit; it never changes a returned score.
    """
    n_scales = len(c_scales)
    if penalties is None:
        penalties = (L2_PENALTY,) * n_scales
    penalties = tuple(str(p) for p in penalties)
    if len(penalties) != n_scales:  # pragma: no cover - defensive
        penalties = (L2_PENALTY,) * n_scales
    x_train = feats[np.ix_(np.arange(n_train), mask)]
    x_test = feats[np.ix_(np.arange(n_train, feats.shape[0]), mask)]
    out = np.empty((x_test.shape[0], n_scales), dtype=float)
    for j, c in enumerate(c_scales):
        clf = _make_estimator(asd_weight, c=float(c), penalty=penalties[j]).fit(
            x_train, np.asarray(y).ravel(), None
        )
        out[:, j] = clf.decision_function(x_test)
        if diag is not None and penalties[j] == L1_PENALTY:
            diag["final_penalty"] = L1_PENALTY
            diag["final_C"] = float(c)
            diag["final_solver"] = L1_SOLVER
            diag["final_n_columns_fitted"] = int(x_train.shape[1])
            diag["final_n_rows_fitted"] = int(x_train.shape[0])
            diag["final_n_nonzero_coef"] = int(getattr(clf, "n_nonzero_coef_", -1))
            diag["final_n_coef_total"] = int(getattr(clf, "n_coef_total_", -1))
            diag["final_fit"] = True
    del x_train, x_test
    return out


def _meta_oof(member_oof: np.ndarray, y, inner_folds=None) -> np.ndarray:
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
    if not inner_folds:
        stacked, _ = stack_scores(member_oof, y, member_oof, c=META_C)
        return np.asarray(stacked, dtype=float)
    # CUT FIX (b): cross-fit the meta-learner on the SAME grouped inner folds,
    # so each training row's stacked score comes from a meta fit that did not
    # see that row. Fitting and scoring the same rows made the stacked OOF
    # scores, and hence the cut and the ASD-weight choice, optimistic.
    stacked = np.full(len(y), np.nan, dtype=float)
    for inner_train, inner_val in inner_folds:
        if len(np.unique(y[inner_train])) < 2:
            continue
        _, val_scores = stack_scores(member_oof[inner_train], y[inner_train], member_oof[inner_val], c=META_C)
        stacked[inner_val] = val_scores
    if not np.isfinite(stacked).any():
        stacked, _ = stack_scores(member_oof, y, member_oof, c=META_C)
    return np.asarray(stacked, dtype=float)


def select_stacked_weight(
    oof_by_weight: dict[float, np.ndarray],
    y,
    inner_folds=None,
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
        stacked = _meta_oof(member_oof, y, inner_folds)
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


def _meta_coefficients(oof_matrix: np.ndarray, y) -> dict | None:
    """READ-ONLY coefficients + intercept of the meta-learner `stack_scores` fits.

    Mirrors `stack_scores`' fit exactly (same finite rows, same ``META_C``, same
    solver/seed, training rows only) and returns the learned weight of EVERY meta
    column plus the intercept, so the diagnostic can report how much weight the
    sparse column 0 and each other slot received. These values are NEVER used for
    a score or a cut: the scored model keeps the coefficients `stack_scores`
    itself fitted.
    """
    oof_matrix = np.asarray(oof_matrix, dtype=float)
    y = np.asarray(y).ravel()
    if oof_matrix.ndim != 2 or oof_matrix.shape[0] != len(y):
        return None
    ok = np.isfinite(oof_matrix).all(axis=1) & np.isfinite(y)
    if int(ok.sum()) < 2 or len(np.unique(y[ok])) < 2:
        return None
    try:
        model = LogisticRegression(
            C=META_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        ).fit(oof_matrix[ok], y[ok])
    except Exception:  # pragma: no cover - diagnostics must never raise
        return None
    coefs = [float(v) for v in np.asarray(model.coef_).ravel()]
    intercept = [float(v) for v in np.asarray(model.intercept_).ravel()]
    return {"coefs": coefs, "intercept": intercept}


def _slot_status(matrix: np.ndarray, column_specs: list[dict]) -> list[dict]:
    """READ-ONLY per-column status: fitted vs neutral-0.0 fallback, by slot.

    Reports, for EVERY slot of a meta design matrix, how many rows are finite and
    whether the column is identically zero -- the signature of the neutral 0.0
    fallback used when a column's fit was impossible. It reads the matrix that was
    already produced and changes nothing.
    """
    matrix = np.asarray(matrix, dtype=float)
    out: list[dict] = []
    if matrix.ndim != 2:
        return out
    for slot in range(matrix.shape[1]):
        col = matrix[:, slot]
        finite = np.isfinite(col)
        spec = column_specs[slot] if slot < len(column_specs) else {}
        out.append(
            {
                "slot": int(slot),
                "member": spec.get("member"),
                "penalty": spec.get("penalty"),
                "C": spec.get("c"),
                "rows": int(col.size),
                "finite_rows": int(finite.sum()),
                "neutral_zero": bool(np.all(np.abs(np.nan_to_num(col, nan=0.0)) < 1e-12)),
            }
        )
    return out


def _emit_diag_json(
    counters: dict,
    l1_diag: dict,
    column_specs: list[dict],
    oof_matrix: dict[float, np.ndarray],
    test_matrix: np.ndarray,
    y,
    asd_weight: float,
    threshold: float,
    hgb_test_col: np.ndarray | None,
) -> None:
    """ONE stderr line prefixed ``DIAG_JSON`` with the read-only diagnostics.

    Everything here is computed from arrays that were ALREADY produced: no fitted
    array, column value, score or cut is created or modified, and the whole body
    is wrapped by the caller so it can never raise into a fold.
    """
    l1_slot = None
    for slot, spec in enumerate(column_specs):
        if str(spec.get("penalty")) == L1_PENALTY:
            l1_slot = int(slot)
            break
    if "l1" in counters:
        counters["l1"]["column_slot"] = l1_slot
    meta = _meta_coefficients(oof_matrix.get(float(asd_weight)), y)
    payload = {
        "tag": "iter39-covariate-basis",
        "asd_weight": float(asd_weight),
        "cut": float(threshold),
        "covariate_basis": counters.get("covariates"),
        "meta_columns": int(len(column_specs)),
        "columns": column_specs,
        "columns_oof_selected_weight": _slot_status(oof_matrix.get(float(asd_weight)), column_specs),
        "columns_oof_by_weight": {
            f"{float(w):g}": _slot_status(m, column_specs) for w, m in oof_matrix.items()
        },
        "columns_heldout": _slot_status(test_matrix, column_specs),
        "l1": counters.get("l1", l1_diag),
        "hgb": counters.get("hgb"),
        "members": {
            name: {
                "c_scales": info.get("c_scales"),
                "penalties": info.get("penalties"),
                "K_requested": info.get("k_requested"),
                "K_used": info.get("k_used"),
                "K_truncated": info.get("k_truncated"),
            }
            for name, info in (counters.get("members") or {}).items()
        },
        "meta_coefs_by_slot": None if meta is None else [round(v, 6) for v in meta["coefs"]],
        "meta_intercept": None if meta is None else [round(v, 6) for v in meta["intercept"]],
        "hgb_column_appended": bool(hgb_test_col is not None),
    }
    try:
        line = json.dumps(payload, sort_keys=True, default=str, allow_nan=False)
    except (TypeError, ValueError):
        line = json.dumps(payload, sort_keys=True, default=str)
    sys.stderr.write(f"DIAG_JSON {line}\n")
    sys.stderr.flush()


def fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    global LAST_FEATURE_COUNTERS, LAST_DIAGNOSTICS
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
    # The booster's log-odds column is produced ONCE per fold (it is unweighted,
    # so it is the same for every ASD-weight grid point) and appended LAST.
    hgb_oof_col: np.ndarray | None = None
    hgb_test_col: np.ndarray | None = None
    # READ-ONLY diagnostics of the sparse (L1) column. Never a fitted array, a
    # score or a cut.
    l1_diag: dict = counters.setdefault("l1", {})
    column_specs: list[dict] = []
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
        # THIS ITERATION'S ONE MECHANISM: the penalty of each column of this
        # member. ONLY the incumbent Gordon+HCP slot at the incumbent strength
        # (`L1_C == LOGREG_C`) becomes an L1 (sparse) logistic; the second
        # Gordon+HCP scale (column 1), Power (column 2) and Markov (column 3)
        # keep the unchanged L2 estimator. Pre-registered, never chosen from
        # data, and used identically in the OOF path and the held-out path.
        penalties = tuple(
            L1_PENALTY
            if (tuple(atlas_set) in L1_MEMBER_ATLAS_SETS and abs(float(c) - float(L1_C)) < 1e-12)
            else L2_PENALTY
            for c in c_scales
        )
        l1_column = int(penalties.index(L1_PENALTY)) if L1_PENALTY in penalties else None
        if l1_column is not None:
            l1_diag["column_slot"] = None  # filled once the column layout is known
            l1_diag["member"] = "+".join(atlas_set)
            l1_diag["local_column"] = int(l1_column)
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
        # NOTE (this iteration): the nonlinear member is no longer fitted from this
        # member's design. It is fitted ONCE after this loop on its OWN feature
        # space -- the Power + Markov tangent EDGE blocks concatenated, see
        # `HGB_FEATURE_MEMBERS` -- so no Gordon+HCP edge reaches the booster. The
        # linear columns of this member, their masks, their two L2 scales and
        # their fits are unchanged.
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
            "penalties": [str(p) for p in penalties],
            "k_requested": int(k),
            "k_used": int(kept),
            "k_truncated": bool(int(kept) < int(k)),
        }
        for j, c in enumerate(c_scales):
            column_specs.append(
                {"member": name, "c": float(c), "penalty": str(penalties[j])}
            )
        for weight in weights:
            block = _member_oof(
                feats,
                n_train,
                y,
                inner_folds,
                inner_masks,
                weight,
                c_scales,
                penalties,
            )
            oof_cols[weight].append(block)
            if l1_column is not None:
                col = np.asarray(block[:, l1_column], dtype=float)
                l1_diag.setdefault("oof_by_weight", {})[f"{weight:g}"] = {
                    "fitted_rows": int(np.isfinite(col).sum()),
                    "rows_total": int(col.size),
                    "all_neutral_zero": bool(np.all(np.abs(np.nan_to_num(col)) < 1e-12)),
                }
            del block
        designs.append(
            {
                "name": name,
                "feats": feats,
                "n_train": n_train,
                "n_cov": n_cov,
                "mask": final_mask,
                "c_scales": c_scales,
                "penalties": penalties,
            }
        )

    # ------------------------------------------------------------------ #
    # NONLINEAR MEMBER ON ITS OWN FEATURE SPACE (this iteration's ONE change)
    # ------------------------------------------------------------------ #
    # Prior runs fed the booster the same Gordon+HCP top-|t| edges the linear
    # columns already linearize; its ranking gain was flat at all three stages.
    # Here the booster consumes the Power and Markov tangent EDGE blocks of the
    # same subject, concatenated into ONE design with the always-kept covariate
    # basis (age, sex, age^2, age x sex) appended once.
    # (`_member_design` standardizes each block on training rows only, and
    # per-column standardization makes the concatenation of the two already
    # standardized member blocks identical to standardizing the joint block, so
    # this is the concatenated design at no extra tangent cost.) Its reduction to
    # K = `HGB_FEATURES` edges uses its OWN |t| ranking computed INSIDE each fit
    # from that fit's training rows only -- one ranking per inner split plus one
    # from all training rows of the fold -- never from held-out or test rows.
    by_name = {d["name"]: d for d in designs}
    hgb_members = [by_name[n] for n in HGB_FEATURE_MEMBERS if n in by_name]
    if hgb_members:
        n_cov_hgb = int(hgb_members[0]["n_cov"])
        edge_blocks = [d["feats"][:, : d["feats"].shape[1] - d["n_cov"]] for d in hgb_members]
        hgb_feats = np.column_stack(edge_blocks + [hgb_members[0]["feats"][:, -n_cov_hgb:]])
        del edge_blocks
        hgb_diag = counters.setdefault("hgb", {})
        hgb_diag["feature_space"] = "+".join(HGB_FEATURE_MEMBERS) + "_tangent_edges"
        hgb_diag["column"] = "last"
        hgb_diag["K_requested"] = int(HGB_FEATURES)
        hgb_inner_selections = [
            _select_edges_with_ranking(hgb_feats, inner_train, y, HGB_FEATURES, n_cov_hgb)
            for inner_train, _ in inner_folds
        ]
        _, hgb_kept, hgb_total, hgb_final_order = _select_edges_with_ranking(
            hgb_feats, np.arange(n_rows_train), y, HGB_FEATURES, n_cov_hgb
        )
        hgb_diag["edges_total"] = int(hgb_total)
        hgb_diag["K_used"] = int(hgb_kept)
        hgb_diag["K_truncated"] = bool(hgb_kept < int(HGB_FEATURES))
        # ONE booster per inner split (that split's own training rows and its own
        # top-K subset) plus one refit on all training rows of the fold. Its
        # P(control) LOG-ODDS is ONE NEW FINAL meta column, in the SAME last slot
        # in the OOF and the held-out path; the ASD class-weight grid keeps acting
        # only on the four linear columns (the booster is unweighted, so it is
        # fitted once per split, not once per grid point). Rows a split cannot
        # score keep the neutral 0.0 (P=0.5).
        hgb_oof_col = _hgb_column_oof(
            hgb_feats, n_rows_train, y, inner_folds, hgb_inner_selections, n_cov_hgb, hgb_diag
        )
        hgb_test_col = _hgb_column_test(
            hgb_feats, n_rows_train, y, hgb_final_order, n_cov_hgb, hgb_diag
        )
        del hgb_inner_selections, hgb_feats
    del by_name, hgb_members

    # `cols` holds one (n_train, n_scales) block per member, so the meta-learner
    # input becomes ONE COLUMN PER (member, pre-registered L2 scale): C=1e-2 and
    # C=1e-1 for Gordon+HCP, then Power, then Markov -> FOUR columns. No hand-set
    # average of the two Gordon+HCP scales is ever formed; the column count is
    # derived from the blocks rather than hard-coded.
    oof_matrix = {w: np.column_stack(cols) for w, cols in oof_cols.items()}
    del oof_cols
    # NEW (this iteration): the booster's log-odds P(control) enters as ONE extra,
    # FINAL meta column -- identical column position for every ASD-weight grid
    # point, and (below) the same final column of the held-out matrix. The four
    # PR #25 columns keep their own order and content, so column 0 is again the
    # pure C=1e-2 logistic log-odds. `stack_scores` (META_C=1e-2) then fits the
    # booster's weight on training-fold OOF rows instead of a hand-set 50/50
    # probability average.
    if hgb_oof_col is not None:
        for weight in oof_matrix:
            oof_matrix[weight] = np.column_stack([oof_matrix[weight], hgb_oof_col])
    n_meta_columns = int(next(iter(oof_matrix.values())).shape[1])

    # Training-only knobs, now in the STACKED score's own units: the ASD class
    # weight and the single scalar cut come from the meta-learner's OOF scores of
    # the TRAINING rows (test rows are never passed in). Weight 1.0 / cut 0.0
    # remains the fallback, i.e. the incumbent operating point.
    asd_weight, threshold = select_stacked_weight(oof_matrix, y, inner_folds)

    # Held-out scores: each member refit on ALL training rows at the selected
    # weight (with its own training-row selection mask), then the one
    # meta-learner (fit on the training OOF columns only) applied per subject.
    # The cut is the unchanged iter1_a training-fold OOF rule, used unmapped.
    test_blocks = []
    for d in designs:
        block_diag = l1_diag if L1_PENALTY in d["penalties"] else None
        test_blocks.append(
            _member_test_scores(
                d["feats"],
                d["n_train"],
                y,
                d["mask"],
                asd_weight,
                d["c_scales"],
                d["penalties"],
                block_diag,
            )
        )
    test_matrix = np.column_stack(test_blocks)
    del test_blocks
    del designs
    # Same LAST column, same log-odds units: the booster refit on this fold's full
    # training rows. No rule asymmetry between the OOF and the held-out path.
    if hgb_test_col is not None:
        test_matrix = np.column_stack([test_matrix, hgb_test_col])
        column_specs.append(
            {"member": "+".join(HGB_FEATURE_MEMBERS), "c": None, "penalty": "hgb", "k": int(HGB_FEATURES)}
        )
    _, test_scores = stack_scores(oof_matrix[asd_weight], y, test_matrix, c=META_C)
    used_fallback = False
    if not np.isfinite(test_scores).all():
        test_scores = average_scores(test_matrix)
        used_fallback = True

    # READ-ONLY diagnostics (never scored, never used for a cut): did the booster
    # actually reach the meta-learner, and how much weight did the meta-learner
    # learn on its column? Reported in the single stderr counter line.
    if isinstance(counters.get("hgb"), dict):
        weights_seen = _meta_column_weights(oof_matrix[asd_weight], y)
        hgb_col = int(n_meta_columns) - 1 if hgb_test_col is not None else None
        counters["hgb"]["meta_column_index"] = hgb_col
        counters["hgb"]["meta_weights"] = None if weights_seen is None else [round(w, 6) for w in weights_seen]
        counters["hgb"]["meta_weight_hgb"] = (
            None
            if (weights_seen is None or hgb_col is None or hgb_col >= len(weights_seen))
            else round(float(weights_seen[hgb_col]), 6)
        )
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

    # READ-ONLY diagnostics of the sparse column and of every meta slot, emitted as
    # ONE stderr line prefixed DIAG_JSON. Wrapped so it can never raise into a fold
    # and never touches a fitted array, a column value, a score or the cut.
    try:
        _emit_diag_json(
            counters,
            l1_diag,
            column_specs,
            oof_matrix,
            test_matrix,
            y,
            asd_weight,
            threshold,
            hgb_test_col,
        )
        LAST_DIAGNOSTICS = dict(counters.get("l1") or {})
    except Exception:  # pragma: no cover - diagnostics must never break a fold
        pass

    # Per-subject decision: each test subject is compared with the one scalar cut
    # learned from the training folds. Nothing is computed from the other test
    # subjects, so scoring the test set whole or in pieces gives the same labels.
    return test_scores, (test_scores > threshold).astype(int)

