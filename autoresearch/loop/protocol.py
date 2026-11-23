"""Hard checks against the historical best-epoch evaluation leak.

An earlier revision reported LOSO AUC 0.707 by picking, per fold, the epoch
with the highest score on the held-out site. That uses the evaluation set to
select the model. Final-epoch scoring of the same config is 0.623; the leak
was worth about 0.07 AUC. Gates and reports must use final-epoch metrics only.
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_GATE_METRICS = frozenset({"best_auc", "best_auc_mean"})
REQUIRED_MODEL_SELECTION = ("final epoch", "final fit")

# Edits that typically reintroduce the leak.
_LEAK_PATTERNS = (
    re.compile(r"""gate\[[^\]]*(best_auc)""", re.I),
    re.compile(r"""["']metric["']\s*:\s*["']best_auc""", re.I),
    re.compile(r"""observed\s*=\s*summary\[[\"']best_auc""", re.I),
    re.compile(r"""verdict.*best_auc_mean""", re.I),
    re.compile(r"""reported_metrics\s*=\s*.*best_""", re.I),
    re.compile(r"""model_selection\s*=\s*[\"']best""", re.I),
)


# A model must get its data only through fit_predict(train, test). Reading files
# directly would reach the locked test institutions (autoresearch.loop.holdout)
# and the test labels.
_DATA_ACCESS_PATTERNS = (
    re.compile(r"manifest\.csv|labels\.npy|file_ids\.txt|fc_path|roi_path"),
    re.compile(r"""["'][^"']*data/abide"""),
    re.compile(r"\bDATA_DIRS\b|\bSource\s*\(|autoresearch\.trial|autoresearch\.loop"),
    re.compile(r"\bHOLDOUT_|holdout_mask"),
)


def data_access_reasons(content: str) -> list[str]:
    """Reasons this model source would read data outside fit_predict's arguments."""
    return [
        f"matched {pattern.pattern!r}: models may only use the train/test splits passed to "
        "fit_predict; reading data files or the harness reaches the locked test set"
        for pattern in _DATA_ACCESS_PATTERNS
        if pattern.search(content)
    ]


def leak_reasons_in_source(content: str) -> list[str]:
    """Return reasons this source would reintroduce evaluation-set selection."""
    reasons: list[str] = []
    for pattern in _LEAK_PATTERNS:
        if pattern.search(content):
            reasons.append(
                f"matched {pattern.pattern!r}: this selects/gates on a best-epoch "
                "metric and leaks the evaluation set"
            )
    return reasons


def protocol_notes(result: dict[str, Any]) -> list[str]:
    """Notes the experimenter must surface; first items are hard failures."""
    notes: list[str] = []
    selection = str(result.get("model_selection", "")).lower()
    if not any(token in selection for token in REQUIRED_MODEL_SELECTION):
        notes.append(
            "PROTOCOL FAIL: result is not marked as a final fit / final epoch. "
            "Do not treat this number as an improvement."
        )

    verdict = result.get("verdict") or {}
    metric = str(verdict.get("metric", ""))
    if metric in FORBIDDEN_GATE_METRICS or metric.startswith("best_"):
        notes.append(
            f"PROTOCOL FAIL: gated on {metric!r}. Best-epoch gating leaks the "
            "evaluation set (historical LOSO bias ~0.07 AUC)."
        )

    summary = result.get("summary") or {}
    final_auc = summary.get("auc_mean")
    best_auc = summary.get("best_auc_mean")
    if isinstance(final_auc, (int, float)) and isinstance(best_auc, (int, float)):
        gap = float(best_auc) - float(final_auc)
        if gap > 0.03:
            notes.append(
                f"diagnostic: best-epoch AUC is {gap:.3f} above final-epoch "
                f"({best_auc:.3f} vs {final_auc:.3f}). Report only the final-epoch "
                "number; a similar gap (~0.07) was the old LOSO leak."
            )
    return notes


def protocol_ok(result: dict[str, Any]) -> bool:
    return not any(note.startswith("PROTOCOL FAIL") for note in protocol_notes(result))


# --- Per-subject scoring --------------------------------------------------------
#
# The cross-site papers score each test subject on its own. A model that z-scores,
# ranks, harmonizes or thresholds the held-out scores using the other test subjects
# borrows information they never had (iter10's per-group calibration lifted ASD
# recall from 55% to 75%). The check below scores the same test set whole, in two
# halves and one subject at a time; every subject must get the same score and label.

BATCH_ATOL = 1e-5


class _SyntheticSplit:
    def __init__(self, fc: dict, idx, groups, y, age, sex, with_labels: bool) -> None:
        import numpy as np

        self._fc, self._idx = fc, np.asarray(idx)
        self.groups = groups[self._idx]
        self.sites = groups[self._idx]
        self.age = age[self._idx]
        self.sex = sex[self._idx]
        self.y = y[self._idx] if with_labels else None

    def __len__(self) -> int:
        return len(self._idx)

    def fc(self, atlas: str):
        return self._fc[atlas][self._idx]


def _synthetic(n_sites: int = 4, per_site: int = 24, seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = n_sites * per_site
    y = np.tile(np.arange(2), n // 2)
    groups = np.repeat(np.array([f"S{k}" for k in range(n_sites)]), per_site)
    age = rng.uniform(7, 30, n)
    sex = rng.integers(1, 3, n)
    shift = rng.normal(0, 0.8, n_sites)
    fc = {}
    for atlas, p in (("gordon", 14), ("hcp", 15), ("power", 8), ("markov", 10)):
        mats = []
        for i, label in enumerate(y):
            ts = rng.standard_normal((60, p))
            ts[:, 1] += (0.6 * label + shift[i // per_site]) * ts[:, 0]
            mats.append(np.corrcoef(ts, rowvar=False))
        fc[atlas] = np.stack(mats).astype(np.float32)
    return fc, groups, y, age, sex


def batch_independence_reasons(fit_predict) -> list[str]:
    """Empty if every test subject is scored the same whole, in halves and alone."""
    import numpy as np

    fc, groups, y, age, sex = _synthetic()
    test_idx = np.where(groups == "S0")[0]
    train_idx = np.where(groups != "S0")[0]
    train = _SyntheticSplit(fc, train_idx, groups, y, age, sex, with_labels=True)

    def score(idx):
        s, p = fit_predict(train, _SyntheticSplit(fc, idx, groups, y, age, sex, with_labels=False))
        return np.asarray(s, dtype=float).reshape(-1), np.asarray(p).reshape(-1).astype(int)

    full_s, full_p = score(test_idx)
    if len(np.unique(full_p)) < 2 and np.ptp(full_s) == 0:
        return ["fit_predict gives every synthetic test subject the same score; cannot check per-subject scoring"]
    order = np.random.default_rng(0).permutation(len(test_idx))
    pieces = [order[: len(order) // 2], order[len(order) // 2 :], order[:1], order[-1:]]
    reasons: list[str] = []
    for piece in pieces:
        s, p = score(test_idx[piece])
        if not np.allclose(s, full_s[piece], atol=BATCH_ATOL, rtol=0) or not np.array_equal(p, full_p[piece]):
            reasons.append(
                f"per-subject rule broken: scoring {len(piece)} of {len(test_idx)} test subjects changes their "
                "scores or labels, so the model uses the other test subjects (e.g. per-site z-scoring, ranking, "
                "test-set harmonization or a threshold fitted on test scores). Each test subject must be scored "
                "from the training split and its own data only."
            )
            break
    return reasons
