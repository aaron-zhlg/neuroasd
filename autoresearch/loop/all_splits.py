"""Score one model under every split scheme and record recall for each.

Schemes (all on the 1,120 development subjects; the locked set is never used):

    random-kfold      10-fold, stratified by label only (how most papers split)
    site-kfold        10-fold, stratified by site + label (sites seen in training)
    loso-site         leave one site out (sites never seen in training)
    loso-institution  leave one institution out

Each test subject is still scored on its own (per-subject rule). Results go to
`<out-dir>/all_splits.json` and a Markdown table to `<out-dir>/all_splits.md`.
Run it from a checkout of the model to score (e.g. a trial PR branch):

    uv run python -m autoresearch.loop.all_splits --name iter4 --out-dir experiments/iter4_v1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from autoresearch.trial import Source, Split, check_prediction, fold_metrics, git_revision, test_groups
from neuroasd.linear_baseline import GROUPINGS

SCHEMES = ("random-kfold", "site-kfold", "loso-site", "loso-institution")
LABELS = {
    "random-kfold": "随机 10 折",
    "site-kfold": "按站点分层 10 折",
    "loso-site": "留一站点",
    "loso-institution": "留一机构",
}
KEYS = ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "auc")


def folds_for(scheme: str, source, seed: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """(groups the model sees, [(fold name, test indices)])."""
    if scheme in ("loso-site", "loso-institution"):
        grouping = "site" if scheme == "loso-site" else "institution"
        groups = np.array([GROUPINGS[grouping](s) for s in source.sites])
        return groups, [(g, np.where(groups == g)[0]) for g in test_groups(groups, source.y, None)]
    strata = source.y.astype(str) if scheme == "random-kfold" else np.char.add(source.sites.astype(str), source.y.astype(str))
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [(f"f{k}", test) for k, (_, test) in enumerate(splitter.split(np.zeros(len(strata)), strata))]
    return source.sites.copy(), folds


def run_scheme(scheme: str, source, workers: int, seed: int) -> dict:
    from neuroasd.model import fit_predict

    groups, folds = folds_for(scheme, source, seed)
    pooled_y, pooled_pred = [], []

    def one(position: int, fold) -> dict:
        name, test_idx = fold
        train_idx = np.setdiff1d(np.arange(len(source.y)), test_idx)
        random.seed(position)
        np.random.seed(position)
        score, pred = fit_predict(Split(source, train_idx, groups, with_labels=True),
                                  Split(source, test_idx, groups, with_labels=False))
        score, pred = check_prediction(score, pred, len(test_idx))
        y = source.y[test_idx]
        return {"fold": name, "n": int(len(test_idx)), "y": y, "pred": pred, **fold_metrics(y, score, pred)}

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(lambda a: one(*a), enumerate(folds)))
    for r in records:
        pooled_y.append(r.pop("y"))
        pooled_pred.append(r.pop("pred"))
    y, pred = np.concatenate(pooled_y), np.concatenate(pooled_pred)
    asd = y == 0
    return {
        "scheme": scheme,
        "folds": len(records),
        "fold_mean": {f"{k}_mean": round(float(np.mean([r[k] for r in records])), 4) for k in KEYS},
        "fold_std": {f"{k}_std": round(float(np.std([r[k] for r in records])), 4) for k in KEYS},
        "pooled": {
            "asd_recall": round(float((pred[asd] == 0).mean()), 4),
            "control_recall": round(float((pred[~asd] != 0).mean()), 4),
        },
        "runtime_seconds": round(time.time() - started, 1),
        "per_fold": records,
    }


def markdown(name: str, results: list[dict]) -> str:
    lines = [f"### {name}：各种测法的 recall（1,120 人，每人单独判断）", "",
             "| 测法 | ASD recall | 对照 recall | 平衡准确率 | 准确率 | AUC |", "|---|---|---|---|---|---|"]
    for r in results:
        m = r["fold_mean"]
        lines.append(f"| {LABELS[r['scheme']]} | {m['sensitivity_mean']:.1%} | {m['specificity_mean']:.1%} | "
                     f"{m['balanced_accuracy_mean']:.1%} | {m['accuracy_mean']:.1%} | {m['auc_mean']:.3f} |")
    lines += ["", "各折平均。随机 10 折和按站点分层 10 折里，测试的人来自训练见过的站点（和多数论文一样）；"
              "留一站点和留一机构测试的是没见过的站点。锁住的 323 人没有用到。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--schemes", nargs="*", default=list(SCHEMES), choices=SCHEMES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Source()
    results = []
    for scheme in args.schemes:
        print(f"[all_splits] {args.name}: {scheme} ...", flush=True)
        results.append(run_scheme(scheme, source, args.workers, args.seed))
        m = results[-1]["fold_mean"]
        print(f"  ASD recall {m['sensitivity_mean']:.3f}  control recall {m['specificity_mean']:.3f}  "
              f"AUC {m['auc_mean']:.3f}  ({results[-1]['runtime_seconds'] / 60:.1f} min)", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "all_splits.json").write_text(json.dumps({
        "name": args.name, "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(), "subjects": int(len(source.y)), "seed": args.seed, "results": results,
    }, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "all_splits.md").write_text(markdown(args.name, results), encoding="utf-8")
    print(markdown(args.name, results))


if __name__ == "__main__":
    main()
