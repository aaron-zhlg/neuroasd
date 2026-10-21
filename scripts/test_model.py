"""Score subjects with a model saved by `scripts.train_model` and print the metrics.

    # score the model's own held-out test subjects (from model.json)
    uv run python -m scripts.test_model --data-root /path/to/data --model outputs/models/institution_NYU/model.npz

    # score other subjects instead: chosen sites, or everyone
    uv run python -m scripts.test_model --data-root /path/to/data --model ... --sites UM_1 USM --out predictions.csv
    uv run python -m scripts.test_model --data-root /path/to/data --model ... --all

    # check that the saved-model path reproduces fit_predict on a held-out institution
    uv run python -m scripts.test_model --data-root /path/to/data --check-holdout NYU

The data root holds `abide_dcan` and `abide_dcan_hcp` in the harness format
(`processed/manifest.csv` + FC `.npy` files). Per-group models standardize scores per
group (`--group-by`, default: the model's training grouping) from that group's own unlabeled scores, so score
whole sites; per-subject models score each subject on its own.

Subjects the model was trained on are flagged: their metrics are training-set fit,
not generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from autoresearch.trial import DATA_DIRS, Source, Split, fold_metrics
from neuroasd.frozen_model import ATLASES, FrozenModel
from neuroasd.linear_baseline import GROUPINGS

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("balanced_accuracy", "accuracy", "auc", "sensitivity", "specificity")


def report(y: np.ndarray, score: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    pooled = fold_metrics(y, score, pred)
    print(f"{'group':<16} {'n':>5}  " + "  ".join(f"{m[:8]:>8}" for m in METRICS))
    per_group = {}
    for g in sorted(set(groups)):
        mask = groups == g
        if len(np.unique(y[mask])) < 2:
            print(f"{g:<16} {mask.sum():>5}  (one class only, skipped)")
            continue
        per_group[g] = fold_metrics(y[mask], score[mask], pred[mask])
        print(f"{g:<16} {mask.sum():>5}  " + "  ".join(f"{per_group[g][m]:>8.3f}" for m in METRICS))
    group_mean = {m: float(np.mean([v[m] for v in per_group.values()])) for m in METRICS} if per_group else {}
    print("-" * 70)
    print(f"{'pooled':<16} {len(y):>5}  " + "  ".join(f"{pooled[m]:>8.3f}" for m in METRICS))
    if group_mean:
        print(f"{'group mean':<16} {len(per_group):>5}  " + "  ".join(f"{group_mean[m]:>8.3f}" for m in METRICS))
    return {"pooled": pooled, "group_mean": group_mean, "per_group": per_group}


def check_holdout(source, holdout: str) -> None:
    """Fit on every other institution, then compare the saved-model path with fit_predict."""
    from neuroasd.model import fit_predict

    groups = np.array([GROUPINGS["institution"](s) for s in source.sites])
    train_idx, test_idx = np.where(groups != holdout)[0], np.where(groups == holdout)[0]
    if not len(test_idx):
        raise SystemExit(f"unknown institution {holdout!r}; choose from {sorted(set(groups))}")

    expected, expected_pred = fit_predict(Split(source, train_idx, groups, True), Split(source, test_idx, groups, False))
    model = FrozenModel.fit({a: source.fc(a)[train_idx] for a in ATLASES}, source.age[train_idx],
                            source.sex[train_idx], source.y[train_idx], groups[train_idx])
    score, pred = model.predict({a: source.fc(a)[test_idx] for a in ATLASES}, source.age[test_idx],
                                source.sex[test_idx], groups[test_idx])

    diff = float(np.max(np.abs(score - expected)))
    same = int((pred == expected_pred).sum())
    print(f"holdout {holdout}: max |score diff| = {diff:.2e}, identical labels {same}/{len(pred)}")
    report(source.y[test_idx], score, pred, groups[test_idx])
    if diff > 1e-4 or same != len(pred):
        raise SystemExit("FAIL: the saved-model path does not reproduce fit_predict")
    print("PASS: the saved-model path reproduces fit_predict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None, help="model.npz written by scripts.train_model")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--sites", nargs="*", default=None, help="Score these SITE_IDs instead of the model's test set")
    parser.add_argument("--all", action="store_true", help="Score every subject instead of the model's test set")
    parser.add_argument("--group-by", choices=sorted(GROUPINGS), default=None,
                        help="Groups for the per-group score standardization and the per-group report "
                             "(default: the grouping the model was trained with, else site)")
    parser.add_argument("--out", type=Path, default=None, help="Write per-subject predictions to this CSV")
    parser.add_argument("--json", type=Path, default=None, help="Write the metrics to this JSON")
    parser.add_argument("--check-holdout", metavar="INSTITUTION", default=None)
    args = parser.parse_args()

    source = Source({atlas: args.data_root / path.name for atlas, path in DATA_DIRS.items()})
    if args.check_holdout:
        check_holdout(source, args.check_holdout)
        return

    if args.model is None:
        raise SystemExit("--model is required (a model.npz written by scripts.train_model)")
    meta_path = args.model.with_name("model.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    if args.all:
        idx = np.arange(len(source.y))
    elif args.sites:
        idx = np.where(np.isin(source.sites, args.sites))[0]
    elif "test_file_ids" in meta:
        wanted = set(meta["test_file_ids"])
        idx = np.array([i for i, fid in enumerate(source.file_ids) if fid in wanted], dtype=int)
        print(f"scoring the model's {len(idx)} held-out test subjects ({meta['split']['kind']} split)")
    else:
        raise SystemExit(f"{meta_path} lists no test subjects; pass --sites or --all")
    if not len(idx):
        raise SystemExit("no subjects to score")
    group_by = args.group_by or meta.get("split", {}).get("groups", "site")
    groups = np.array([GROUPINGS[group_by](s) for s in source.sites[idx]])
    model = FrozenModel.load(args.model)
    score, pred = model.predict({a: source.fc(a)[idx] for a in ATLASES}, source.age[idx], source.sex[idx], groups)

    seen = set(meta.get("training_file_ids", []))
    overlap = sum(source.file_ids[i] in seen for i in idx)
    if overlap:
        print(f"WARNING: {overlap}/{len(idx)} subjects were in the model's training set; "
              "these metrics are training-set fit, not generalization.\n")

    metrics = report(source.y[idx], score, pred, groups)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["FILE_ID", "SITE_ID", "label", "score", "pred"])
            for k, i in enumerate(idx):
                writer.writerow([source.file_ids[i], source.sites[i], int(source.y[i]), f"{score[k]:.6f}", int(pred[k])])
        print(f"\npredictions: {args.out}")
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
