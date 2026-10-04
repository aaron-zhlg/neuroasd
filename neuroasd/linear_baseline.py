"""Linear connectome baseline evaluated with leave-one-site-out CV.

Features (per subject, from its FC matrix):
  fisherz  Fisher-z transformed upper triangle.
  tangent  Shrunk correlation projected to the tangent space at the
           log-Euclidean mean of the *training* subjects (IMPAC recipe).

Classifier: ridge classification (targets +/-1) solved in kernel form, so all
regularization strengths are evaluated from one eigendecomposition. The
strength is chosen by grouped inner CV on the training folds only.

Two outer splits are reported: leave-one-site-out (each SITE_ID is a fold) and
the stricter leave-one-institution-out (cohorts from the same institution,
e.g. ABIDE I KKI and ABIDE II KKI_1, are held out together).

Every fitted quantity (tangent reference, feature scaling, ridge strength) is
computed from the training fold. Folds with a single class are used for
training but never as a test fold, since AUC is undefined there.

Robustness: the fit is deterministic, so `--repeats N` refits each fold on N
stratified 90% subsamples of the training set and reports the spread.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "abide_dcan"
DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "linear_baseline_dcan_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "linear_baseline_dcan_v1"

# Several institutions contributed more than one cohort (ABIDE I and II, or
# numbered ABIDE I sub-sites). Holding out a whole institution removes any
# scanner/protocol overlap between the training and test folds.
INSTITUTIONS = {
    "ABIDEII-KKI_1": "KKI",
    "ABIDEII-KUL_3": "LEUVEN",
    "LEUVEN_1": "LEUVEN",
    "LEUVEN_2": "LEUVEN",
    "ABIDEII-NYU_1": "NYU",
    "ABIDEII-NYU_2": "NYU",
    "ABIDEII-OHSU_1": "OHSU",
    "ABIDEII-ONRC_2": "OLIN",
    "ABIDEII-SDSU_1": "SDSU",
    "ABIDEII-TCD_1": "TRINITY",
    "ABIDEII-UCLA_1": "UCLA",
    "UCLA_1": "UCLA",
    "UCLA_2": "UCLA",
    "UM_1": "UM",
    "UM_2": "UM",
    "ABIDEII-USM_1": "USM",
}
GROUPINGS = {
    "site": lambda site: site,
    "institution": lambda site: INSTITUTIONS.get(site, site),
}

SHRINKAGE = 0.1
ALPHAS = np.logspace(-1, 5, 13)
INNER_FOLDS = 5
SUBSAMPLE_FRACTION = 0.9
ASD_LABEL = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features", nargs="+", choices=["fisherz", "tangent"], default=["tangent", "fisherz"])
    parser.add_argument("--repeats", type=int, default=5, help="Subsampled refits per fold (0 = full fit only)")
    parser.add_argument("--group-by", nargs="+", choices=sorted(GROUPINGS), default=["site", "institution"],
                        help="Outer CV grouping(s): leave-one-site-out and/or leave-one-institution-out")
    parser.add_argument("--sites", nargs="*", default=None,
                        help="Restrict test folds to these sites (or institutions with --group-by institution)")
    parser.add_argument("--name", default="linear_baseline_dcan_v1")
    parser.add_argument("--workers", type=int, default=4, help="Parallel fold fits (each holds ~1 GB)")
    return parser.parse_args()


def load_dataset(data_dir: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with (data_dir / "processed" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fcs = np.stack([np.load(data_dir / row["fc_path"]) for row in rows]).astype(np.float32)
    return rows, clean_fc(fcs)


def clean_fc(fcs: np.ndarray) -> np.ndarray:
    """Zero-variance parcels give NaN rows; treat them as unconnected (r = 0, diagonal 1)."""
    fcs = np.nan_to_num(fcs, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.arange(fcs.shape[-1])
    fcs[:, diag, diag] = 1.0
    return fcs


def _apply_eig(mats: np.ndarray, fn) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mats)
    return (vecs * fn(vals)[..., None, :]) @ np.swapaxes(vecs, -1, -2)


def _upper(mats: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(mats.shape[-1], k=k)


def fisherz_features(fcs: np.ndarray) -> np.ndarray:
    iu = _upper(fcs, 1)
    return np.arctanh(np.clip(fcs[:, iu[0], iu[1]], -0.999, 0.999)).astype(np.float32)


def shrink(fcs: np.ndarray, shrinkage: float = SHRINKAGE) -> np.ndarray:
    out = (1.0 - shrinkage) * fcs.astype(np.float64) + shrinkage * np.eye(fcs.shape[-1])
    return (out + np.swapaxes(out, -1, -2)) / 2.0


def tangent_features(fcs: np.ndarray, train_idx: np.ndarray, block: int = 64) -> np.ndarray:
    """Tangent-space embedding at the log-Euclidean mean of the training subjects."""
    n, p = fcs.shape[0], fcs.shape[1]
    log_sum = np.zeros((p, p))
    for start in range(0, len(train_idx), block):
        log_sum += _apply_eig(shrink(fcs[train_idx[start : start + block]]), np.log).sum(axis=0)
    reference = _apply_eig((log_sum / len(train_idx))[None], np.exp)[0]
    whiten = _apply_eig(reference[None], lambda v: 1.0 / np.sqrt(v))[0]

    iu = _upper(fcs, 0)
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((n, len(iu[0])), dtype=np.float32)
    for start in range(0, n, block):
        whitened = whiten @ shrink(fcs[start : start + block]) @ whiten
        feats[start : start + block] = _apply_eig(whitened, np.log)[:, iu[0], iu[1]] * weight
    return feats


class KernelRidgeClassifier:
    """Ridge classification with +/-1 targets, solved in kernel form."""

    def __init__(self, alphas: np.ndarray = ALPHAS, inner_folds: int = INNER_FOLDS) -> None:
        self.alphas = alphas
        self.inner_folds = inner_folds

    @staticmethod
    def _dual(kernel: np.ndarray, targets: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, float]:
        offset = float(targets.mean())
        vals, vecs = np.linalg.eigh(kernel)
        proj = vecs.T @ (targets - offset)
        duals = vecs @ (proj[:, None] / (vals[:, None] + alphas[None, :]))
        return duals, offset

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "KernelRidgeClassifier":
        xs = np.array(x, copy=True)
        self.mean_ = xs.mean(axis=0, dtype=np.float64).astype(xs.dtype)
        self.scale_ = xs.std(axis=0, dtype=np.float64).astype(xs.dtype)
        self.scale_[self.scale_ == 0] = 1.0
        xs -= self.mean_
        xs /= self.scale_
        kernel = (xs @ xs.T).astype(np.float64)
        targets = np.where(y == 1, 1.0, -1.0)

        n_groups = len(np.unique(groups))
        scores = np.zeros(len(self.alphas))
        if n_groups >= 2:
            splitter = GroupKFold(n_splits=min(self.inner_folds, n_groups))
            for tr, va in splitter.split(xs, y, groups):
                if len(np.unique(y[va])) < 2:
                    continue
                duals, offset = self._dual(kernel[np.ix_(tr, tr)], targets[tr], self.alphas)
                decision = kernel[np.ix_(va, tr)] @ duals + offset
                scores += [roc_auc_score(y[va], decision[:, j]) for j in range(len(self.alphas))]
        self.alpha_ = float(self.alphas[int(np.argmax(scores))])

        duals, self.offset_ = self._dual(kernel, targets, np.array([self.alpha_]))
        self.coef_ = xs.T @ duals[:, 0].astype(xs.dtype)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.scale_) @ self.coef_ + self.offset_


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score > 0).astype(int)
    asd, ctl = y == ASD_LABEL, y != ASD_LABEL
    tp = float(((pred == 1) & (y == 1)).sum())
    precision = tp / max(float((pred == 1).sum()), 1.0)
    recall = tp / max(float((y == 1).sum()), 1.0)
    return {
        "accuracy": float((pred == y).mean()),
        "auc": float(roc_auc_score(y, score)),
        # Same convention as neuroasd.train.evaluate: F1 of label 1 (control).
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "sensitivity": float((pred[asd] == ASD_LABEL).mean()),
        "specificity": float((pred[ctl] != ASD_LABEL).mean()),
    }


def stratified_subsample(y: np.ndarray, idx: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y[idx]):
        members = idx[y[idx] == label]
        keep.append(rng.choice(members, size=int(round(SUBSAMPLE_FRACTION * len(members))), replace=False))
    return np.sort(np.concatenate(keep))


def test_groups(groups: np.ndarray, y: np.ndarray, only: list[str] | None) -> list[str]:
    eligible = sorted(g for g in set(groups) if len(np.unique(y[groups == g])) == 2)
    return [g for g in eligible if not only or g in only]


def aggregate(folds: list[dict], key_prefix: str = "") -> dict[str, float]:
    n = np.array([f["test_size"] for f in folds], dtype=float)
    out: dict[str, float] = {"folds": len(folds), "subjects": int(n.sum())}
    for metric in ("accuracy", "auc", "f1", "sensitivity", "specificity"):
        values = np.array([f[metric] for f in folds])
        out[f"{key_prefix}{metric}_mean"] = round(float(values.mean()), 4)
        out[f"{key_prefix}{metric}_std"] = round(float(values.std()), 4)
    acc = np.array([f["accuracy"] for f in folds])
    out[f"{key_prefix}accuracy_pooled"] = round(float((acc * n).sum() / n.sum()), 4)
    return out


def evaluate(
    rows: list[dict[str, str]],
    fcs: np.ndarray,
    features: list[str],
    repeats: int,
    only_sites: list[str] | None = None,
    log=print,
    workers: int = 1,
    grouping: str = "site",
) -> dict[str, dict]:
    """Leave-one-group-out, where a group is a site or a whole institution."""
    y = np.array([int(r["label"]) for r in rows])
    sites = np.array([r["SITE_ID"] for r in rows])
    groups = np.array([GROUPINGS[grouping](s) for s in sites])
    datasets = np.array([r.get("DATASET", "abide1") for r in rows])
    fisherz = fisherz_features(fcs) if "fisherz" in features else None

    def run_task(group: str, run_name: str, train: np.ndarray, feature: str) -> dict:
        test = np.where(groups == group)[0]
        x = fisherz if feature == "fisherz" else tangent_features(fcs, train)
        clf = KernelRidgeClassifier().fit(x[train], y[train], groups[train])
        return {
            "test_group": group,
            "test_sites": sorted(set(sites[test])),
            "dataset": "+".join(sorted(set(datasets[test]))),
            "run": run_name,
            "feature": feature,
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "alpha": clf.alpha_,
            **metrics_from_scores(y[test], clf.decision_function(x[test])),
        }

    tasks = []
    for group in test_groups(groups, y, only_sites):
        train_full = np.where(groups != group)[0]
        runs = [("full", train_full)] + [
            (f"r{seed}", stratified_subsample(y, train_full, seed)) for seed in range(repeats)
        ]
        tasks += [(group, run_name, train, feature) for run_name, train in runs for feature in features]

    records: dict[tuple[str, str, str], dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, *task): task for task in tasks}
        for future in as_completed(futures):
            rec = future.result()
            records[(rec["test_group"], rec["run"], rec["feature"])] = rec
            if rec["run"] == "full":
                log(f"[{grouping}] {rec['test_group']:16s} n={rec['test_size']:3d} {rec['feature']:8s} "
                    f"acc={rec['accuracy']:.3f} auc={rec['auc']:.3f} "
                    f"[{len(records)}/{len(tasks)} tasks, {time.time() - t0:.0f}s]")

    results: dict[str, dict] = {f: {"folds": [], "repeat_folds": []} for f in features}
    for group, run_name, _train, feature in tasks:
        rec = records[(group, run_name, feature)]
        results[feature]["folds" if run_name == "full" else "repeat_folds"].append(rec)
    return results


def summarize(results: dict[str, dict], repeats: int) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for feature, res in results.items():
        folds = res["folds"]
        entry = {"all_folds": aggregate(folds)}
        for dataset in sorted({f["dataset"] for f in folds}):
            entry[f"{dataset}_folds"] = aggregate([f for f in folds if f["dataset"] == dataset])
        if repeats:
            per_repeat = []
            for seed in range(repeats):
                sel = [f for f in res["repeat_folds"] if f["run"] == f"r{seed}"]
                per_repeat.append(aggregate(sel))
            entry["subsample_repeats"] = {
                metric: {
                    "mean": round(float(np.mean([r[metric] for r in per_repeat])), 4),
                    "std": round(float(np.std([r[metric] for r in per_repeat])), 4),
                }
                for metric in ("accuracy_mean", "accuracy_pooled", "auc_mean")
            }
        summary[feature] = entry
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "linear_baseline.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    rows, fcs = load_dataset(args.data_dir)
    n_sites = len({r["SITE_ID"] for r in rows})
    log(f"{len(rows)} subjects, {fcs.shape[1]} ROIs, {n_sites} sites; features={args.features} "
        f"repeats={args.repeats} group_by={args.group_by}")
    results, summary = {}, {}
    for grouping in args.group_by:
        results[grouping] = evaluate(
            rows, fcs, args.features, args.repeats, args.sites, log, args.workers, grouping
        )
        summary[grouping] = summarize(results[grouping], args.repeats)
        log(json.dumps({grouping: summary[grouping]}, indent=2))

    (args.output_dir / "folds.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    try:
        data_ref = args.data_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        data_ref = args.data_dir.name
    run_config = {
        "name": args.name,
        "validation": {
            "site": "leave-one-site-out over every site with both classes; single-class sites are training-only",
            "institution": "leave-one-institution-out: all cohorts of an institution (ABIDE I and II, numbered "
                           "sub-sites) are held out together; see INSTITUTIONS in neuroasd/linear_baseline.py",
            "run": args.group_by,
        },
        "model_selection": "ridge strength by inner CV grouped like the outer split (AUC); no test-fold selection",
        "data": {
            "data_dir": data_ref,
            "num_subjects": len(rows),
            "num_rois": int(fcs.shape[1]),
            "num_sites": n_sites,
        },
        "features": args.features,
        "hyperparameters": {
            "tangent_shrinkage": SHRINKAGE,
            "tangent_reference": "log-Euclidean mean of training subjects",
            "ridge_alphas": [float(a) for a in ALPHAS],
            "inner_cv": f"GroupKFold({INNER_FOLDS}) by the outer grouping",
            "repeats": args.repeats,
            "subsample_fraction": SUBSAMPLE_FRACTION,
        },
        "reproduce": f"uv run python -m neuroasd.linear_baseline --data-dir {data_ref} --repeats {args.repeats}",
    }
    results_json = {
        "name": args.name,
        "evaluated_at": datetime.now(timezone.utc).date().isoformat(),
        "label_convention": "label 0 = ASD, 1 = control; sensitivity = ASD recall; f1 = control F1 (as neuroasd.train)",
        "summary": summary,
        "folds": {
            grouping: {feature: res["folds"] for feature, res in by_feature.items()}
            for grouping, by_feature in results.items()
        },
    }
    (args.experiment_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    (args.experiment_dir / "results.json").write_text(json.dumps(results_json, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {args.experiment_dir / 'results.json'}")
    log_handle.close()


if __name__ == "__main__":
    main()
