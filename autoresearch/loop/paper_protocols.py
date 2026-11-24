"""Score the current model the way each paper scored theirs (ABIDE I only).

Papers mostly use ABIDE I alone, so this run keeps only the ABIDE I subjects of
DCAN (all of them, including the ABIDE I part of the locked test set: no
choice is made from these numbers, they are only for the comparison).

    random-10fold   10-fold, stratified by label           (ASD-DiagNet 2019, Heinsfeld 2018)
    brainnettf      70/10/20 split stratified by site, 5 repeats; train on the 70% only
                                                            (Kan et al. 2022, BrainNetTF)
    brainmass       70/15/15 split stratified by site, 10 repeats; train on the 70% only
                                                            (Yang et al. 2024, BrainMass)
    loso-site       leave one ABIDE I site out              (Abraham 2017, inter-site)

Every test subject is scored on its own, as in `all_splits`. The validation
part of the 70/x/y splits is not used (the model picks its settings inside
the training part), so the model trains on less data than the papers' models.

    uv run python -m autoresearch.loop.paper_protocols --out-dir experiments/paper_protocols
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from autoresearch.trial import DATA_DIRS, Source, Split, check_prediction, fold_metrics, git_revision, test_groups

PROTOCOLS = ("random-10fold", "brainnettf", "brainmass", "loso-site")
KEYS = ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "auc")


class AbideISource(Source):
    """All ABIDE I subjects of DCAN (development + locked)."""

    def __init__(self, data_dirs=DATA_DIRS) -> None:
        self.data_dirs = data_dirs
        with (data_dirs["gordon"] / "processed" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        keep = np.array([not row["SITE_ID"].startswith("ABIDEII") for row in all_rows])
        self._keep = np.flatnonzero(keep)
        self.rows = [all_rows[i] for i in self._keep]
        self.file_ids = [row["FILE_ID"] for row in self.rows]
        self.y = np.array([int(row["label"]) for row in self.rows])
        self.sites = np.array([row["SITE_ID"] for row in self.rows])
        self.age = np.array([float(row["AGE_AT_SCAN"]) for row in self.rows])
        self.sex = np.array([int(float(row["SEX"])) for row in self.rows])
        self._fc = {}
        self._lock = threading.Lock()


def _site_strata(source) -> np.ndarray:
    strata = np.char.add(source.sites.astype(str), source.y.astype(str))
    values, counts = np.unique(strata, return_counts=True)
    rare = np.isin(strata, values[counts < 5])
    strata[rare] = source.y[rare].astype(str)
    return strata


def folds_for(protocol: str, source) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(fold name, train indices, test indices)]."""
    n = len(source.y)
    if protocol == "random-10fold":
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        return [(f"f{k}", tr, te) for k, (tr, te) in enumerate(splitter.split(np.zeros(n), source.y))]
    if protocol in ("brainnettf", "brainmass"):
        repeats, test_share = (5, 0.20) if protocol == "brainnettf" else (10, 0.15)
        strata = _site_strata(source)
        out = []
        for r in range(repeats):
            rest, test = next(StratifiedShuffleSplit(1, test_size=test_share, random_state=r).split(np.zeros(n), strata))
            train_share = 0.70 / (1.0 - test_share)
            tr_pos, _ = next(StratifiedShuffleSplit(1, train_size=train_share, random_state=100 + r)
                             .split(np.zeros(len(rest)), strata[rest]))
            out.append((f"r{r}", rest[tr_pos], test))
        return out
    groups = source.sites
    return [(g, np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in test_groups(groups, source.y, None)]


def run_protocol(protocol: str, source, workers: int) -> dict:
    from neuroasd.model import fit_predict

    folds = folds_for(protocol, source)
    groups = source.sites.copy()

    def one(position: int, fold) -> dict:
        name, train_idx, test_idx = fold
        random.seed(position)
        np.random.seed(position)
        score, pred = fit_predict(Split(source, train_idx, groups, with_labels=True),
                                  Split(source, test_idx, groups, with_labels=False))
        score, pred = check_prediction(score, pred, len(test_idx))
        return {"fold": name, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                **fold_metrics(source.y[test_idx], score, pred),
                "y": source.y[test_idx].tolist(), "score": np.asarray(score, dtype=float).round(6).tolist()}

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(lambda a: one(*a), enumerate(folds)))
    return {
        "protocol": protocol,
        "folds": len(records),
        "mean": {k: round(float(np.mean([r[k] for r in records])), 4) for k in KEYS},
        "std": {k: round(float(np.std([r[k] for r in records])), 4) for k in KEYS},
        "runtime_seconds": round(time.time() - started, 1),
        "per_fold": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocols", nargs="*", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--file-name", default="paper_protocols.json")
    parser.add_argument("--data", choices=("abide1", "dev"), default="abide1",
                        help="abide1: all ABIDE I subjects; dev: the 1,120 ABIDE I+II development subjects")
    args = parser.parse_args()

    source = AbideISource() if args.data == "abide1" else Source()
    asd = int((source.y == 0).sum())
    print(f"[paper_protocols] {args.data}: {len(source.y)} subjects ({asd} ASD), {len(set(source.sites))} sites", flush=True)
    results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for protocol in args.protocols:
        print(f"[paper_protocols] {protocol} ...", flush=True)
        results.append(run_protocol(protocol, source, args.workers))
        m, s = results[-1]["mean"], results[-1]["std"]
        print(f"  ASD recall {m['sensitivity']:.3f}±{s['sensitivity']:.3f}  control recall {m['specificity']:.3f}"
              f"±{s['specificity']:.3f}  acc {m['accuracy']:.3f}  AUC {m['auc']:.3f}±{s['auc']:.3f}"
              f"  ({results[-1]['runtime_seconds'] / 60:.1f} min)", flush=True)
        (args.out_dir / args.file_name).write_text(json.dumps({
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_revision(),
            "subjects": int(len(source.y)), "asd": asd, "results": results,
        }, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
