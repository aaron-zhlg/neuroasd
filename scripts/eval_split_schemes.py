"""Score the iter10 model (`neuroasd.model.fit_predict`) under several split schemes.

    loso-site          leave-one-site-out, 33 two-class sites (same as the trial harness)
    loso-institution   leave-one-institution-out, 23 institutions (same as the trial harness)
    random-kfold       10-fold CV stratified by label only; sites overlap train/test
    site-kfold         10-fold CV stratified by site x label, i.e. every site is split
                       proportionally between train and test (BrainNetTF-style)

--scoring picks how test subjects are scored:

    per-group    iter10 as released (`neuroasd.model.fit_predict`): test scores are
                 standardized per site (or institution) from that group's own unlabeled
                 scores, then cut at a training-only OOF threshold
    per-subject  each test subject is scored on its own with the same features and
                 logistic regression, labelled control when the decision value is > 0;
                 this is how random-split papers score a classifier

For the k-fold schemes `groups` is the site. Every number comes from one fit per fold;
nothing is selected on a test fold.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from autoresearch.trial import (
    DATA_DIRS,
    STAGE_GROUPING,
    Source,
    Split,
    check_prediction,
    fold_metrics,
    summarize,
    test_groups,
)
from neuroasd.frozen_model import ATLASES, FrozenModel
from neuroasd.linear_baseline import GROUPINGS

SCHEMES = ("loso-site", "loso-institution", "random-kfold", "site-kfold")
SCORINGS = ("per-group", "per-subject")
REPO_ROOT = Path(__file__).resolve().parents[1]


def per_subject_fit_predict(train, test) -> tuple[np.ndarray, np.ndarray]:
    model = FrozenModel.fit({a: train.fc(a) for a in ATLASES}, train.age, train.sex, train.y, train.groups,
                            per_subject=True)
    return model.predict({a: test.fc(a) for a in ATLASES}, test.age, test.sex)


def folds_for(scheme: str, source, n_splits: int, repeat: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Return the per-subject groups the model sees and a list of (fold name, test indices)."""
    if scheme in ("loso-site", "loso-institution"):
        groups = np.array([GROUPINGS[STAGE_GROUPING[scheme]](s) for s in source.sites])
        return groups, [(g, np.where(groups == g)[0]) for g in test_groups(groups, source.y, None)]
    strata = source.y.astype(str) if scheme == "random-kfold" else np.char.add(source.sites.astype(str), source.y.astype(str))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
    folds = [(f"r{repeat}f{k}", test) for k, (_, test) in enumerate(splitter.split(np.zeros(len(strata)), strata))]
    return source.sites.copy(), folds


def run(scheme: str, source, n_splits: int, repeats: int, workers: int, scoring: str = "per-group") -> dict:
    from neuroasd.model import fit_predict as per_group_fit_predict

    fit_predict = per_subject_fit_predict if scoring == "per-subject" else per_group_fit_predict
    jobs = []
    for repeat in range(repeats if scheme.endswith("kfold") else 1):
        groups, folds = folds_for(scheme, source, n_splits, repeat)
        jobs += [(repeat, name, test_idx, groups) for name, test_idx in folds]

    def one(position: int, job) -> dict:
        repeat, name, test_idx, groups = job
        train_idx = np.setdiff1d(np.arange(len(source.y)), test_idx)
        random.seed(position)
        np.random.seed(position)
        started = time.time()
        score, pred = fit_predict(
            Split(source, train_idx, groups, with_labels=True),
            Split(source, test_idx, groups, with_labels=False),
        )
        score, pred = check_prediction(score, pred, len(test_idx))
        y = source.y[test_idx]
        record = {
            "repeat": repeat,
            "test_group": name,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "seconds": round(time.time() - started, 1),
            **fold_metrics(y, score, pred),
            "_y": y.tolist(),
            "_pred": pred.tolist(),
            "_sites": source.sites[test_idx].tolist(),
        }
        print(f"  [{scheme}] {name:<16} n={len(test_idx):<4} bacc={record['balanced_accuracy']:.3f} "
              f"acc={record['accuracy']:.3f} auc={record['auc']:.3f} ({record['seconds']:.0f}s)", flush=True)
        return record

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(lambda args: one(*args), enumerate(jobs)))

    y = np.concatenate([r.pop("_y") for r in records])
    pred = np.concatenate([r.pop("_pred") for r in records])
    for r in records:
        r.pop("_sites")
    summary = summarize(records)
    asd, ctl = y == 0, y != 0
    summary["balanced_accuracy_pooled"] = float(((pred[asd] == 0).mean() + (pred[ctl] != 0).mean()) / 2)
    per_repeat = {}
    for repeat in sorted({r["repeat"] for r in records}):
        sub = [r for r in records if r["repeat"] == repeat]
        per_repeat[repeat] = {k: float(np.mean([r[k] for r in sub])) for k in ("balanced_accuracy", "accuracy", "auc")}
    return {
        "scheme": scheme,
        "scoring": scoring,
        "n_splits": n_splits if scheme.endswith("kfold") else None,
        "repeats": len(per_repeat),
        "runtime_seconds": round(time.time() - started, 1),
        "summary": {k: round(v, 4) for k, v in summary.items()},
        "per_repeat": per_repeat,
        "folds": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Directory holding abide_dcan, abide_dcan_hcp, ...")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--scoring", choices=SCORINGS, default="per-group")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "split_schemes")
    args = parser.parse_args()

    source = Source({atlas: args.data_root / path.name for atlas, path in DATA_DIRS.items()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scheme in args.schemes:
        print(f"== {scheme} ({args.scoring})", flush=True)
        result = run(scheme, source, args.n_splits, args.repeats, args.workers, args.scoring)
        out = args.out_dir / f"{scheme}__{args.scoring}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        s = result["summary"]
        print(f"== {scheme} ({args.scoring}): bacc {s['balanced_accuracy_mean']:.4f} ± {s['balanced_accuracy_std']:.4f}  "
              f"acc {s['accuracy_mean']:.4f}  auc {s['auc_mean']:.4f}  "
              f"({s['folds']} folds, {result['runtime_seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
