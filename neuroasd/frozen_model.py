"""The iter10 model (`neuroasd.model.fit_predict`) fitted once and saved to a single `.npz`.

`fit_predict` refits everything per fold. `FrozenModel.fit` runs the same steps on
one training set and keeps what the held-out side needs: the two tangent whiteners,
the age/sex covariate contract, the feature mean/SD, the logistic regression
weights, and the training-only decision threshold. `FrozenModel.predict` then
reproduces `fit_predict`'s test-side output for new subjects.

Two scoring modes:

    per-group    (iter10, default) scores are standardized per group (site) using that
                 group's own unlabeled scores, then cut at the training-only OOF
                 threshold, exactly as in `fit_predict`. Score whole sites at a time: a
                 group of a single subject always gets score 0.
    per-subject  each subject is scored on its own: the raw logistic decision value,
                 labelled control when it is > 0 (probability > 0.5), as a plain
                 classifier's predict() does in random-split papers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from neuroasd.linear_baseline import tangent_project, tangent_whitener
from neuroasd.model import (
    LOGREG_C,
    _balanced_accuracy_threshold,
    _group_standardize,
    _make_estimator,
    _oof_training_scores,
)

ATLASES = ("gordon", "hcp")


@dataclass
class FrozenModel:
    whiten_gordon: np.ndarray
    whiten_hcp: np.ndarray
    age_mean: float
    age_std: float
    female_code: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    logreg_c: float = LOGREG_C
    per_subject: bool = False

    @classmethod
    def fit(cls, fc: dict[str, np.ndarray], age, sex, y, groups, per_subject: bool = False) -> "FrozenModel":
        """Same steps as `fit_predict` on the training side. `groups` drive the OOF threshold.

        With `per_subject` the OOF threshold is skipped; `predict` cuts raw scores at 0.
        """
        rows = np.arange(len(y))
        whiten = {atlas: tangent_whitener(fc[atlas], rows) for atlas in ATLASES}
        age = np.asarray(age, dtype=float)
        sex = np.asarray(sex, dtype=float)
        age_mean, age_std = float(np.nanmean(age)), float(np.nanstd(age))
        if not np.isfinite(age_mean):
            age_mean = 0.0
        if not np.isfinite(age_std) or age_std <= 1e-8:
            age_std = 1.0
        codes = np.unique(sex[np.isfinite(sex)])
        partial = cls(whiten["gordon"], whiten["hcp"], age_mean, age_std,
                      float(codes[-1]) if codes.size else np.nan, None, None, None, 0.0, 0.0)

        feats = partial._raw_features(fc, age, sex)
        partial.feature_mean = feats.mean(axis=0, dtype=np.float64).astype(feats.dtype)
        partial.feature_std = feats.std(axis=0, dtype=np.float64).astype(feats.dtype)
        partial.feature_std[partial.feature_std == 0] = 1.0
        feats -= partial.feature_mean
        feats /= partial.feature_std

        y, groups = np.asarray(y), np.asarray(groups)
        clf = _make_estimator().fit(feats, y, groups).model_
        partial.coef = clf.coef_.ravel().copy()
        partial.intercept = float(clf.intercept_[0])
        partial.per_subject = per_subject
        if per_subject:
            return partial
        oof = _oof_training_scores(feats, y, groups)
        partial.threshold = _balanced_accuracy_threshold(_group_standardize(oof, groups), y)
        return partial

    def _raw_features(self, fc: dict[str, np.ndarray], age, sex) -> np.ndarray:
        feats = np.concatenate([tangent_project(fc["gordon"], self.whiten_gordon),
                                tangent_project(fc["hcp"], self.whiten_hcp)], axis=1)
        age = np.asarray(age, dtype=float)
        age = np.where(np.isfinite(age), age, self.age_mean)
        sex_col = (np.asarray(sex, dtype=float) == self.female_code).astype(float)
        cov = np.column_stack([(age - self.age_mean) / self.age_std, sex_col]).astype(feats.dtype)
        return np.concatenate([feats, cov], axis=1)

    def decision_function(self, fc: dict[str, np.ndarray], age, sex) -> np.ndarray:
        """Raw logistic decision values (higher = more likely control), before group standardization."""
        feats = self._raw_features(fc, age, sex)
        feats -= self.feature_mean
        feats /= self.feature_std
        return feats @ self.coef + self.intercept

    def predict(self, fc: dict[str, np.ndarray], age, sex, groups=None) -> tuple[np.ndarray, np.ndarray]:
        """Scores and labels (0 = ASD, 1 = control). Per-group mode matches `fit_predict`."""
        raw = self.decision_function(fc, age, sex)
        if self.per_subject:
            return raw, (raw > 0).astype(int)
        scores = _group_standardize(raw, np.asarray(groups))
        return scores, (scores > self.threshold).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})

    @classmethod
    def load(cls, path: Path) -> "FrozenModel":
        with np.load(path) as data:
            values = {f.name: data[f.name] for f in fields(cls)}
        for name, value in values.items():
            if value.ndim == 0:
                values[name] = float(value)
        values["per_subject"] = bool(values["per_subject"])
        return cls(**values)
