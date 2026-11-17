"""Recall (sensitivity) of the iter10 model under the release's split schemes.

ASD is the positive class (label 0). For every scheme and scoring this reruns the same
one-fit-per-fold evaluation as `scripts/eval_split_schemes.py`, keeps every test
subject's prediction, and reports:

    ASD recall      (sensitivity, TP / (TP + FN))
    control recall  (specificity, TN / (TN + FP))
    ASD precision   (PPV, TP / (TP + FP))
    NPV, F1 (ASD positive), balanced accuracy

each as a fold mean +/- SD (as in the report) and pooled over all test predictions,
together with the pooled confusion matrix. For k-fold schemes, pooled counts sum the
three reshuffles (every subject appears three times).

    uv run python -m scripts.eval_recall --data-root data
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
from sklearn.metrics import roc_auc_score

from autoresearch.trial import ASD_LABEL, DATA_DIRS, Source, Split, check_prediction
from scripts.eval_split_schemes import folds_for, per_subject_fit_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "experiments" / "iter10_estimator-swap-replace-the-singl_v1" / "recall.json"
RUNS = (
    ("loso-institution", "per-group"),
    ("loso-institution", "per-subject"),
    ("loso-site", "per-group"),
    ("loso-site", "per-subject"),
    ("site-kfold", "per-subject"),
    ("random-kfold", "per-subject"),
)


def counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    asd_true, asd_pred = y == ASD_LABEL, pred == ASD_LABEL
    return {
        "tp": int((asd_true & asd_pred).sum()),
        "fn": int((asd_true & ~asd_pred).sum()),
        "fp": int((~asd_true & asd_pred).sum()),
        "tn": int((~asd_true & ~asd_pred).sum()),
    }


def rates(c: dict[str, int]) -> dict[str, float]:
    def ratio(a: int, b: int) -> float:
        return float(a / b) if b else float("nan")

    recall, specificity = ratio(c["tp"], c["tp"] + c["fn"]), ratio(c["tn"], c["tn"] + c["fp"])
    precision = ratio(c["tp"], c["tp"] + c["fp"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "asd_recall": recall,
        "control_recall": specificity,
        "asd_precision": precision,
        "npv": ratio(c["tn"], c["tn"] + c["fn"]),
        "asd_f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": ratio(c["tp"] + c["tn"], sum(c.values())),
    }


def run(scheme: str, scoring: str, source, n_splits: int, repeats: int, workers: int) -> dict:
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
        score, pred = fit_predict(
            Split(source, train_idx, groups, with_labels=True),
            Split(source, test_idx, groups, with_labels=False),
        )
        score, pred = check_prediction(score, pred, len(test_idx))
        y = source.y[test_idx]
        c = counts(y, pred)
        print(f"  [{scheme}/{scoring}] {name:<16} n={len(test_idx):<4} "
              f"ASD recall={rates(c)['asd_recall']:.3f} ctl recall={rates(c)['control_recall']:.3f}", flush=True)
        return {"repeat": repeat, "test_group": name, "test_size": int(len(test_idx)), **c,
                **rates(c), "auc": float(roc_auc_score(y, score)),
                "_idx": test_idx.tolist(), "_pred": pred.tolist(), "_score": score.tolist()}

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        folds = list(pool.map(lambda args: one(*args), enumerate(jobs)))

    pooled = {k: sum(f[k] for f in folds) for k in ("tp", "fn", "fp", "tn")}
    keys = ("asd_recall", "control_recall", "asd_precision", "npv", "asd_f1", "balanced_accuracy", "accuracy", "auc")
    fold_mean = {}
    for k in keys:
        # A fold with no predicted ASD has undefined precision; nanmean skips it.
        values = np.array([f[k] for f in folds], dtype=float)
        fold_mean[f"{k}_mean"] = round(float(np.nanmean(values)), 4)
        fold_mean[f"{k}_std"] = round(float(np.nanstd(values)), 4)
    predictions = [
        {"repeat": f["repeat"], "fold": f["test_group"], "subject": source.file_ids[i], "site": str(source.sites[i]),
         "label": int(source.y[i]), "pred": p, "score": round(s, 5)}
        for f in folds for i, p, s in zip(f.pop("_idx"), f.pop("_pred"), f.pop("_score"))
    ]
    return {
        "scheme": scheme,
        "scoring": scoring,
        "folds_total": len(folds),
        "runtime_seconds": round(time.time() - started, 1),
        "pooled_confusion": pooled,
        "pooled": {k: round(v, 4) for k, v in rates(pooled).items()},
        "fold_mean": fold_mean,
        "folds": folds,
        "_predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--predictions-out", type=Path, default=REPO_ROOT / "outputs" / "recall" / "predictions.jsonl")
    args = parser.parse_args()

    source = Source({atlas: args.data_root / path.name for atlas, path in DATA_DIRS.items()})
    results = {}
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8") as handle:
        for scheme, scoring in RUNS:
            print(f"== {scheme} ({scoring})", flush=True)
            result = run(scheme, scoring, source, args.n_splits, args.repeats, args.workers)
            for row in result.pop("_predictions"):
                handle.write(json.dumps({"scheme": scheme, "scoring": scoring, **row}) + "\n")
            results[f"{scheme}__{scoring}"] = result
            p = result["pooled"]
            print(f"== {scheme} ({scoring}): ASD recall {p['asd_recall']:.4f}  control recall "
                  f"{p['control_recall']:.4f}  ASD precision {p['asd_precision']:.4f}  "
                  f"({result['runtime_seconds'] / 60:.1f} min)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": "iter10 (Gordon+HCP tangent + age/sex, L2 logistic C=1e-2)",
        "positive_class": "ASD (label 0)",
        "subjects": int(len(source.y)),
        "asd": int((source.y == ASD_LABEL).sum()),
        "control": int((source.y != ASD_LABEL).sum()),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
