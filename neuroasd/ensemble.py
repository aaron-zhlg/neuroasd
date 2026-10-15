"""Leak-safe helpers for multi-atlas / multi-feature ensembles.

The incumbent in ``neuroasd.model`` concatenates Gordon and HCP tangent columns
into **one** classifier. That is concatenation, not stacking.

Use these helpers to change **one** combination rule:

* ``concat_blocks`` — one estimator on concatenated features (already in the tag).
* ``average_scores`` — mean of member decision values; no labels.
* ``stack_scores`` — logistic meta-learner on grouped out-of-fold member scores.

``GCNMember`` emits the same kind of decision values so it can be one extra
column. Do not replace the linear members with a GCN; that model overfits here.

Training-fold statistics only: tangent reference, scalers, member fits, the
meta-learner, and GCN weights. Test labels never enter.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from neuroasd.linear_baseline import KernelRidgeClassifier, fisherz_features, tangent_features

ATLASES = ("gordon", "hcp", "power", "markov")
FEATURE_KINDS = ("tangent", "fisherz")
OOF_FOLDS = 5
LOGREG_C = 1e-2

GCN_EPOCHS = 12
GCN_HIDDEN = 16
GCN_BATCH = 32
GCN_LR = 1e-2
GCN_WEIGHT_DECAY = 1e-2
GCN_DROPOUT = 0.5


class L2LogisticClassifier:
    """Fixed-C L2 logistic, same contract as the iter10 incumbent."""

    def __init__(self, c: float = LOGREG_C, max_iter: int = 2000, tol: float = 1e-4) -> None:
        self.c = c
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray, groups=None) -> "L2LogisticClassifier":
        self.model_ = LogisticRegression(
            C=self.c,
            solver="lbfgs",
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=0,
        ).fit(x, y)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(x)


def make_estimator(kind: str = "logreg"):
    if kind == "logreg":
        return L2LogisticClassifier()
    if kind == "ridge":
        return KernelRidgeClassifier()
    raise ValueError(f"unknown estimator {kind!r}; use 'logreg' or 'ridge'")


def feature_block(train, test, atlas: str, kind: str = "tangent") -> np.ndarray:
    """Train rows, then test rows. Tangent reference from training rows only."""
    if atlas not in ATLASES:
        raise KeyError(f"unknown atlas {atlas!r}; choose from {ATLASES}")
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown features {kind!r}; choose from {FEATURE_KINDS}")
    fcs = np.concatenate([train.fc(atlas), test.fc(atlas)])
    n_train = len(train.y)
    if kind == "tangent":
        return tangent_features(fcs, np.arange(n_train))
    return fisherz_features(fcs)


def covariate_block(train, test) -> np.ndarray:
    """Age z-score and binary sex. Scaler and sex coding from training rows only."""
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


def concat_blocks(*blocks: np.ndarray) -> np.ndarray:
    """Column-wise concatenation. This is not stacking."""
    if not blocks:
        raise ValueError("concat_blocks needs at least one block")
    return np.concatenate(blocks, axis=1)


def standardize(feats: np.ndarray, n_train: int) -> np.ndarray:
    """Column mean/SD from the training rows; applied to train and test rows."""
    out = np.array(feats, copy=True)
    mu = out[:n_train].mean(axis=0, dtype=np.float64).astype(out.dtype)
    sd = out[:n_train].std(axis=0, dtype=np.float64).astype(out.dtype)
    sd[sd == 0] = 1.0
    out -= mu
    out /= sd
    return out


def grouped_oof_scores(
    factory: Callable,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = OOF_FOLDS,
) -> np.ndarray:
    """Out-of-fold decision values, folds grouped like the outer split."""
    scores = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return scores
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    for inner_train, inner_val in splitter.split(x, y, groups):
        if len(np.unique(y[inner_train])) < 2:
            continue
        clf = factory().fit(x[inner_train], y[inner_train], groups[inner_train])
        scores[inner_val] = clf.decision_function(x[inner_val])
    return scores


def member_scores(
    train,
    test,
    atlas: str,
    *,
    kind: str = "tangent",
    estimator: str = "logreg",
) -> tuple[np.ndarray, np.ndarray]:
    """One linear member: ``(train_oof_scores, test_scores)``.

    Features are training-fold standardized. OOF uses grouped inner folds; test
    scores come from one fit on all training rows.
    """
    n_train = len(train.y)
    feats = standardize(feature_block(train, test, atlas, kind), n_train)
    y = np.asarray(train.y)
    groups = np.asarray(train.groups)
    oof = grouped_oof_scores(lambda: make_estimator(estimator), feats[:n_train], y, groups)
    clf = make_estimator(estimator).fit(feats[:n_train], y, groups)
    return oof, clf.decision_function(feats[n_train:])


def average_scores(member_matrix: np.ndarray) -> np.ndarray:
    """Mean over members (axis 1). Label-free; NaN members are skipped."""
    return np.nanmean(np.asarray(member_matrix, dtype=float), axis=1)


def stack_scores(
    oof_members: np.ndarray,
    y: np.ndarray,
    test_members: np.ndarray,
    c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic meta-learner on OOF member scores. Returns ``(train_meta, test_meta)``.

    The meta-learner is fit only on finite training rows of ``oof_members``.
    ``test_members`` is transformed with that fit; it does not affect the fit.
    Test labels are not an argument. Concatenating atlas *features* is not this.
    """
    oof_members = np.asarray(oof_members, dtype=float)
    test_members = np.asarray(test_members, dtype=float)
    y = np.asarray(y)
    if oof_members.ndim != 2 or test_members.ndim != 2:
        raise ValueError("member arrays must be (n, n_members)")
    if oof_members.shape[1] != test_members.shape[1]:
        raise ValueError("train and test member counts differ")
    if oof_members.shape[0] != len(y):
        raise ValueError("OOF rows must match training labels")

    ok = np.isfinite(oof_members).all(axis=1) & np.isfinite(y)
    if int(ok.sum()) < 2 or len(np.unique(y[ok])) < 2:
        return average_scores(oof_members), average_scores(test_members)

    meta = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    ).fit(oof_members[ok], y[ok])
    train_meta = np.full(len(y), np.nan)
    train_meta[ok] = meta.decision_function(oof_members[ok])
    test_ok = np.isfinite(test_members).all(axis=1)
    test_meta = np.full(len(test_members), np.nan)
    if test_ok.any():
        test_meta[test_ok] = meta.decision_function(test_members[test_ok])
    return train_meta, test_meta


class GCNMember:
    """Tiny two-layer GCN that emits decision values (higher = control).

    Node features are FC rows; adjacency is |FC|. Hyperparameters are fixed
    (few epochs, strong L2). Prefer Power (94 parcels). Gordon/HCP are slow.

    ``groups`` is accepted and ignored so the ``fit(x, y, groups)`` contract
    matches the linear members. Fit on training FC only.
    """

    def __init__(
        self,
        epochs: int = GCN_EPOCHS,
        hidden: int = GCN_HIDDEN,
        batch: int = GCN_BATCH,
        lr: float = GCN_LR,
        weight_decay: float = GCN_WEIGHT_DECAY,
        dropout: float = GCN_DROPOUT,
        seed: int = 0,
    ) -> None:
        self.epochs = epochs
        self.hidden = hidden
        self.batch = batch
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

    def fit(self, fcs: np.ndarray, y: np.ndarray, groups=None) -> "GCNMember":
        import torch
        from torch.nn.functional import cross_entropy

        from neuroasd.gcn import SimpleGCN

        fcs = np.asarray(fcs, dtype=np.float32)
        y = np.asarray(y)
        n, p, _ = fcs.shape
        cpu_state = torch.get_rng_state()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        try:
            self.model_ = SimpleGCN(
                in_features=p,
                hidden_dim=self.hidden,
                num_classes=2,
                dropout=self.dropout,
            )
            self.model_.apply(lambda m: _init_module(m, generator))
            opt = torch.optim.Adam(
                self.model_.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
            counts[counts == 0] = 1.0
            weight = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)

            x_all = torch.from_numpy(np.ascontiguousarray(fcs))
            adj_all = x_all.abs()
            y_all = torch.from_numpy(y.astype(np.int64))
            self.model_.train()
            for _ in range(self.epochs):
                order = torch.randperm(n, generator=generator).tolist()
                for start in range(0, n, self.batch):
                    idx = order[start : start + self.batch]
                    logits = self.model_(x_all[idx], adj_all[idx])
                    loss = cross_entropy(logits, y_all[idx], weight=weight)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.model_.eval()
        finally:
            torch.set_rng_state(cpu_state)
        return self

    def decision_function(self, fcs: np.ndarray) -> np.ndarray:
        import torch

        fcs = np.asarray(fcs, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(fcs))
        with torch.no_grad():
            logits = self.model_(x, x.abs())
        return logits[:, 1].cpu().numpy()


def _init_module(module, generator) -> None:
    import torch

    for name, param in module.named_parameters(recurse=False):
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param, generator=generator)
        elif "bias" in name:
            torch.nn.init.zeros_(param)


def gnn_member_scores(train, test, atlas: str = "power", **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """GCN member on one atlas: ``(train_oof_scores, test_scores)``.

    Default atlas is Power. Keep the linear members; add this as one extra
    column in ``average_scores`` or ``stack_scores``. OOF is several GCN fits
    per outer fold — too slow for Gordon/HCP inside a screen timeout.
    """
    fcs_tr = train.fc(atlas)
    fcs_te = test.fc(atlas)
    y = np.asarray(train.y)
    groups = np.asarray(train.groups)
    oof = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        splitter = GroupKFold(n_splits=min(OOF_FOLDS, n_groups))
        for inner_train, inner_val in splitter.split(fcs_tr, y, groups):
            if len(np.unique(y[inner_train])) < 2:
                continue
            clf = GCNMember(**kwargs).fit(fcs_tr[inner_train], y[inner_train], groups[inner_train])
            oof[inner_val] = clf.decision_function(fcs_tr[inner_val])
    clf = GCNMember(**kwargs).fit(fcs_tr, y, groups)
    return oof, clf.decision_function(fcs_te)
