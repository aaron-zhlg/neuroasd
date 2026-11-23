"""Score the current model once on the locked test institutions.

Fits `neuroasd.model.fit_predict` on all 1,120 development subjects and scores
the 323 locked subjects (`autoresearch.loop.holdout`). Reports the mean over the
7 locked institutions and the pooled numbers. Only this score is compared with
the papers.

The loop never calls this. Run it rarely, on a model the loop has already
chosen; every run is appended to outputs/autoresearch/holdout_log.jsonl so the
number of looks is on record.

    uv run python -m autoresearch.loop.final_holdout --name iter1 --note "first accepted base"
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from autoresearch.loop.holdout import HOLDOUT_INSTITUTIONS
from autoresearch.loop.paths import REPO_ROOT
from autoresearch.trial import ASD_LABEL, Source, Split, check_prediction, fold_metrics, git_revision
from neuroasd.linear_baseline import GROUPINGS

LOG = REPO_ROOT / "outputs" / "autoresearch" / "holdout_log.jsonl"


class _Combined:
    """Development rows first, then the locked rows, behind one Source interface."""

    def __init__(self, dev: Source, locked: Source) -> None:
        self._parts = (dev, locked)
        self.y = np.concatenate([dev.y, locked.y])
        self.sites = np.concatenate([dev.sites, locked.sites])
        self.age = np.concatenate([dev.age, locked.age])
        self.sex = np.concatenate([dev.sex, locked.sex])

    def fc(self, atlas: str) -> np.ndarray:
        return np.concatenate([part.fc(atlas) for part in self._parts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Which model this is (e.g. the accepted trial name)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    from neuroasd.model import fit_predict

    started = time.time()
    dev, locked = Source(), Source(split="holdout")
    data = _Combined(dev, locked)
    n_dev = len(dev.y)
    # Inner folds of the model group by institution, like the final LOSO stage.
    groups = np.array([GROUPINGS["institution"](s) for s in data.sites])
    train_idx, test_idx = np.arange(n_dev), np.arange(n_dev, len(data.y))
    score, pred = fit_predict(Split(data, train_idx, groups, with_labels=True),
                              Split(data, test_idx, groups, with_labels=False))
    score, pred = check_prediction(score, pred, len(test_idx))
    y, inst = data.y[test_idx], groups[test_idx]

    per = []
    for name in HOLDOUT_INSTITUTIONS:
        m = inst == name
        per.append({"institution": name, "n": int(m.sum()), **fold_metrics(y[m], score[m], pred[m])})
    keys = ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "auc")
    mean = {f"{k}_mean": round(float(np.mean([p[k] for p in per])), 4) for k in keys}
    asd = y == ASD_LABEL
    pooled = {
        "tp": int(((pred == ASD_LABEL) & asd).sum()), "fn": int(((pred != ASD_LABEL) & asd).sum()),
        "fp": int(((pred == ASD_LABEL) & ~asd).sum()), "tn": int(((pred != ASD_LABEL) & ~asd).sum()),
    }
    pooled["asd_recall"] = round(pooled["tp"] / max(pooled["tp"] + pooled["fn"], 1), 4)
    pooled["control_recall"] = round(pooled["tn"] / max(pooled["tn"] + pooled["fp"], 1), 4)

    entry = {
        "name": args.name, "note": args.note, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(), "train_subjects": n_dev, "test_subjects": int(len(test_idx)),
        "institution_mean": mean, "pooled": pooled, "per_institution": per,
        "runtime_seconds": round(time.time() - started, 1),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    looks = sum(1 for _ in LOG.open(encoding="utf-8")) if LOG.is_file() else 0
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(json.dumps({k: entry[k] for k in ("name", "institution_mean", "pooled")}, indent=1))
    print(f"look #{looks + 1} at the locked test set (log: {LOG})")


if __name__ == "__main__":
    main()
