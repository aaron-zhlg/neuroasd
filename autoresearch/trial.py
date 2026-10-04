"""Run one autoresearch trial of `neuroasd.model` and score it against the gates.

The harness owns data loading, fold splits, and metrics. The model only sees
what `Split` exposes (FC per atlas, age, sex, fold groups, site, and training
labels), so it cannot read FIQ, head motion, or test labels.

Stages, cheap to expensive (see `autoresearch/program.md`):

    screen            leave-one-site-out on 6 mixed ABIDE I / II sites
    loso-site         leave-one-site-out on all 33 sites with both classes
    loso-institution  leave-one-institution-out on 23 institutions (final)

Primary metric: fold-mean balanced accuracy. Every metric comes from one fit
per fold; nothing is selected on a test fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from autoresearch.loop.paths import trials_dir
from neuroasd.linear_baseline import GROUPINGS, clean_fc

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = {
    "gordon": REPO_ROOT / "data" / "abide_dcan",
    "hcp": REPO_ROOT / "data" / "abide_dcan_hcp",
    "power": REPO_ROOT / "data" / "abide_dcan_power",
    "markov": REPO_ROOT / "data" / "abide_dcan_markov",
}
LEDGER_PATH = REPO_ROOT / "outputs" / "autoresearch" / "ledger.jsonl"
GATES_PATH = Path(__file__).resolve().parent / "gates.json"

# Mixed ABIDE I / II sites with both classes and 47-161 subjects each.
SCREEN_SITES = ("NYU", "UM_1", "USM", "ABIDEII-OHSU_1", "ABIDEII-NYU_1", "ABIDEII-SDSU_1")

STAGES = ("screen", "loso-site", "loso-institution")
STAGE_GROUPING = {"screen": "site", "loso-site": "site", "loso-institution": "institution"}
FINAL_STAGE = "loso-institution"
PRIMARY_METRIC = "balanced_accuracy_mean"
MODEL_SELECTION = "final fit (no selection on the evaluation set)"
ASD_LABEL = 0

PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 3


class Source:
    """All subjects of the DCAN ABIDE I + II dataset; FC loaded lazily per atlas."""

    def __init__(self, data_dirs: dict[str, Path] = DATA_DIRS) -> None:
        self.data_dirs = data_dirs
        with (data_dirs["gordon"] / "processed" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.file_ids = [row["FILE_ID"] for row in self.rows]
        self.y = np.array([int(row["label"]) for row in self.rows])
        self.sites = np.array([row["SITE_ID"] for row in self.rows])
        self.age = np.array([float(row["AGE_AT_SCAN"]) for row in self.rows])
        self.sex = np.array([int(float(row["SEX"])) for row in self.rows])
        self._fc: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def fc(self, atlas: str) -> np.ndarray:
        with self._lock:
            if atlas not in self._fc:
                if atlas not in self.data_dirs:
                    raise KeyError(f"unknown atlas {atlas!r}; choose from {sorted(self.data_dirs)}")
                root = self.data_dirs[atlas]
                with (root / "processed" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if [row["FILE_ID"] for row in rows] != self.file_ids:
                    raise RuntimeError(f"{root} does not hold the same subjects as data/abide_dcan")
                stack = [np.load(root / row["fc_path"]) for row in rows]
                self._fc[atlas] = clean_fc(np.stack(stack).astype(np.float32))
            return self._fc[atlas]


class Split:
    """What a model may see for one side of a fold. `y` is None on the test side."""

    def __init__(self, source, idx: np.ndarray, groups: np.ndarray, with_labels: bool) -> None:
        self._source = source
        self._idx = idx
        self.groups = groups[idx]
        self.sites = source.sites[idx]
        self.age = source.age[idx]
        self.sex = source.sex[idx]
        self.y = source.y[idx] if with_labels else None

    def __len__(self) -> int:
        return len(self._idx)

    def fc(self, atlas: str) -> np.ndarray:
        return self._source.fc(atlas)[self._idx]


def fold_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    asd, ctl = y == ASD_LABEL, y != ASD_LABEL
    sensitivity = float((pred[asd] == ASD_LABEL).mean())
    specificity = float((pred[ctl] != ASD_LABEL).mean())
    tp = float(((pred == 1) & (y == 1)).sum())
    precision = tp / max(float((pred == 1).sum()), 1.0)
    recall = tp / max(float((y == 1).sum()), 1.0)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auc": float(roc_auc_score(y, score)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
    }


def check_prediction(score, pred, n: int) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=float).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if score.shape != (n,) or pred.shape != (n,):
        raise ValueError(f"fit_predict must return {n} scores and {n} labels, got {score.shape} / {pred.shape}")
    if not np.isfinite(score).all():
        raise ValueError("fit_predict returned non-finite scores")
    if not set(np.unique(pred)) <= {0, 1}:
        raise ValueError("fit_predict labels must be 0 (ASD) or 1 (control)")
    return score, pred.astype(int)


def test_groups(groups: np.ndarray, y: np.ndarray, only: tuple[str, ...] | None) -> list[str]:
    eligible = sorted(g for g in set(groups) if len(np.unique(y[groups == g])) == 2)
    if only is None:
        return eligible
    missing = [g for g in only if g not in eligible]
    if missing:
        raise SystemExit(f"not a two-class test fold: {', '.join(missing)}")
    return list(only)


def run_folds(source, stage: str, workers: int, log=print) -> list[dict]:
    from neuroasd.model import fit_predict

    grouping = STAGE_GROUPING[stage]
    groups = np.array([GROUPINGS[grouping](s) for s in source.sites])
    held_out = test_groups(groups, source.y, SCREEN_SITES if stage == "screen" else None)

    def one(position: int, group: str) -> dict:
        train_idx = np.where(groups != group)[0]
        test_idx = np.where(groups == group)[0]
        random.seed(position)
        np.random.seed(position)
        started = time.time()
        score, pred = fit_predict(
            Split(source, train_idx, groups, with_labels=True),
            Split(source, test_idx, groups, with_labels=False),
        )
        score, pred = check_prediction(score, pred, len(test_idx))
        record = {
            "test_group": group,
            "test_sites": sorted(set(source.sites[test_idx])),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "seconds": round(time.time() - started, 1),
            **fold_metrics(source.y[test_idx], score, pred),
        }
        log(f"  {group:<16} n={len(test_idx):<4} bacc={record['balanced_accuracy']:.3f} "
            f"acc={record['accuracy']:.3f} auc={record['auc']:.3f}")
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        folds = list(pool.map(lambda args: one(*args), enumerate(held_out)))
    return folds


def summarize(folds: list[dict]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ("balanced_accuracy", "accuracy", "auc", "sensitivity", "specificity", "f1"):
        values = np.array([fold[key] for fold in folds])
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    n = np.array([fold["test_size"] for fold in folds], dtype=float)
    acc = np.array([fold["accuracy"] for fold in folds])
    summary["accuracy_pooled"] = float((acc * n).sum() / n.sum())
    summary["folds"] = len(folds)
    return summary


def load_gates() -> dict:
    return json.loads(GATES_PATH.read_text(encoding="utf-8"))


def either_split_improves(site: float, inst: float, ref_site: float, ref_inst: float, rule: dict) -> dict:
    """PASS if one split gains >= improve_margin and the other drops <= regress_tolerance."""
    margin, tolerance = float(rule["improve_margin"]), float(rule["regress_tolerance"])
    d_site, d_inst = site - ref_site, inst - ref_inst
    site_wins = d_site >= margin and d_inst >= -tolerance
    inst_wins = d_inst >= margin and d_site >= -tolerance
    return {
        "passed": bool(site_wins or inst_wins),
        "delta_site": round(d_site, 4),
        "delta_institution": round(d_inst, 4),
        "improved_on": [name for name, won in (("site", site_wins), ("institution", inst_wins)) if won],
        "margin": round(max(min(d_site - margin, d_inst + tolerance), min(d_inst - margin, d_site + tolerance)), 4),
    }


def site_stage_summary(name: str) -> dict[str, float]:
    path = trials_dir() / f"loso-site__{name}" / "result.json"
    if not path.is_file():
        raise SystemExit(f"loso-institution needs the loso-site result of the same trial: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def apply_gate(stage: str, name: str, summary: dict[str, float]) -> dict:
    gates = load_gates()
    gate = gates["gates"][stage]
    observed = summary[PRIMARY_METRIC]
    if stage != FINAL_STAGE:
        threshold = float(gate["threshold"])
        return {
            "stage": stage,
            "metric": PRIMARY_METRIC,
            "threshold": threshold,
            "observed": round(observed, 4),
            "margin": round(observed - threshold, 4),
            "passed": observed >= threshold,
            "rationale": gate["rationale"],
        }
    reference = gates["reference"]
    site = summary["site_" + PRIMARY_METRIC]
    rule = either_split_improves(
        site, observed,
        reference["loso-site"][PRIMARY_METRIC], reference["loso-institution"][PRIMARY_METRIC],
        gates["rule"],
    )
    return {
        "stage": stage,
        "metric": PRIMARY_METRIC,
        "rule": "either split improves; the other does not regress",
        "observed": round(observed, 4),
        "observed_site": round(site, 4),
        **rule,
        "rationale": gate["rationale"],
    }


def git_revision() -> dict[str, str]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return "unknown"

    return {"branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"), "commit": run("git", "rev-parse", "--short", "HEAD")}


def append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


class SyntheticSource:
    """Small fake dataset with the same interface as `Source`, for lint smoke tests."""

    def __init__(self, n_sites: int = 4, per_site: int = 24, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        n = n_sites * per_site
        self.y = np.tile(np.arange(2), n // 2)
        self.sites = np.repeat([f"S{k}" for k in range(n_sites)], per_site)
        self.age = rng.uniform(7, 30, n)
        self.sex = rng.integers(1, 3, n)
        self._fc = {}
        for atlas, p in (("gordon", 14), ("hcp", 15), ("power", 8), ("markov", 10)):
            mats = []
            for label in self.y:
                ts = rng.standard_normal((60, p))
                ts[:, 1] += 0.6 * label * ts[:, 0]
                mats.append(np.corrcoef(ts, rowvar=False))
            self._fc[atlas] = np.stack(mats).astype(np.float32)

    def fc(self, atlas: str) -> np.ndarray:
        return self._fc[atlas]


def smoke_check() -> dict:
    """Run neuroasd.model.fit_predict on synthetic data; raises if the interface breaks."""
    from neuroasd.model import fit_predict

    source = SyntheticSource()
    groups = source.sites
    test_idx = np.where(groups == "S0")[0]
    train_idx = np.where(groups != "S0")[0]
    score, pred = fit_predict(
        Split(source, train_idx, groups, with_labels=True),
        Split(source, test_idx, groups, with_labels=False),
    )
    score, pred = check_prediction(score, pred, len(test_idx))
    return fold_metrics(source.y[test_idx], score, pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Short trial identifier")
    parser.add_argument("--stage", choices=STAGES, default="screen")
    parser.add_argument("--note", default="", help="One-line hypothesis being tested")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("AUTORESEARCH_TRIAL_WORKERS", "2")))
    parser.add_argument("--quiet", action="store_true", help="Suppress per-fold logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = git_revision()
    print(f"Trial   : {args.name}")
    print(f"Stage   : {args.stage} ({STAGE_GROUPING[args.stage]} folds)")
    print(f"Branch  : {revision['branch']} @ {revision['commit']}")
    if args.note:
        print(f"Note    : {args.note}")
    print("")

    started = time.time()
    source = Source()
    folds = run_folds(source, args.stage, args.workers, log=(lambda _m: None) if args.quiet else print)
    summary = summarize(folds)
    if args.stage == FINAL_STAGE:
        site = site_stage_summary(args.name)
        summary.update({f"site_{key}": value for key, value in site.items() if key != "folds"})
    elapsed = time.time() - started
    verdict = apply_gate(args.stage, args.name, summary)

    result = {
        "name": args.name,
        "stage": args.stage,
        "note": args.note,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "git": revision,
        "data": {"subjects": len(source.y), "atlases": sorted(DATA_DIRS)},
        "model_selection": MODEL_SELECTION,
        "summary": {key: round(value, 4) for key, value in summary.items()},
        "folds": folds,
        "verdict": verdict,
    }
    trial_dir = trials_dir() / f"{args.stage}__{args.name}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    append_ledger({key: result[key] for key in ("name", "stage", "note", "ran_at", "git", "model_selection", "summary", "verdict")})

    print("")
    for key, label in (("balanced_accuracy", "bal. acc"), ("accuracy", "accuracy"), ("auc", "auc"),
                       ("sensitivity", "ASD rec."), ("specificity", "ctl rec.")):
        print(f"{label:9s}: {summary[key + '_mean']:.3f} ± {summary[key + '_std']:.3f}")
    print(f"runtime  : {elapsed / 60:.1f} min")
    if args.stage == FINAL_STAGE:
        print(f"gate     : site {verdict['delta_site']:+.4f}, institution {verdict['delta_institution']:+.4f} "
              f"vs reference (improved on: {', '.join(verdict['improved_on']) or 'none'})")
    else:
        print(f"gate     : {PRIMARY_METRIC} {verdict['observed']:.4f} vs threshold {verdict['threshold']:.4f} "
              f"(margin {verdict['margin']:+.4f})")
    print(f"result   : {trial_dir / 'result.json'}")
    print(f"VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}")
    raise SystemExit(PASS_EXIT_CODE if verdict["passed"] else FAIL_EXIT_CODE)


if __name__ == "__main__":
    main()
