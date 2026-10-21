"""Train the iter10 model and score it, either on one split (saves a model) or by cross-validation.

Pick how the test subjects are chosen with --split:

    site         hold out whole sites; the model never sees them
    institution  hold out whole institutions; stricter than site, because some
                 institutions contributed several sites
    random       random subjects stratified by diagnosis; every site appears in both
                 train and test, so scores are higher than for unseen sites
    stratified   random subjects drawn from every site in proportion, stratified by
                 site x diagnosis (BrainNetTF-style split)

One split, saves `<out-dir>/model.npz` and `<out-dir>/model.json` (split, training and
test FILE_IDs, test scores):

    uv run python -m scripts.train_model --data-root /path/to/data --split institution --holdout NYU
    uv run python -m scripts.train_model --data-root /path/to/data --split random --test-size 0.2 --seed 0

Cross-validation with --cv (one fit per fold, no model saved): leave-one-site-out or
leave-one-institution-out over every two-class group, or k-fold for random / stratified:

    uv run python -m scripts.train_model --data-root /path/to/data --split institution --cv
    uv run python -m scripts.train_model --data-root /path/to/data --split random --cv --folds 10 --repeats 3

--scoring picks how test subjects are scored (default depends on --split):

    per-group    iter10 as released: test scores are standardized per site / institution
                 from that group's own unlabeled scores, then cut at a training-only
                 threshold. Default for site / institution, where a test group is a
                 whole site.
    per-subject  each subject scored on its own (logistic decision > 0 = control), the
                 way random-split papers score a classifier. Default for random /
                 stratified, where a site has only a few test subjects.

`scripts.test_model --model <out-dir>/model.npz` scores a saved model's test subjects again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from autoresearch.trial import DATA_DIRS, Source
from neuroasd.frozen_model import ATLASES, FrozenModel
from neuroasd.linear_baseline import GROUPINGS
from scripts.eval_split_schemes import run
from scripts.test_model import report

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("site", "institution", "random", "stratified")
CV_SCHEMES = {"site": "loso-site", "institution": "loso-institution", "random": "random-kfold", "stratified": "site-kfold"}
DEFAULT_SCORING = {"site": "per-group", "institution": "per-group", "random": "per-subject", "stratified": "per-subject"}


def choose_test(source, split: str, holdout: list[str] | None, test_size: float, seed: int) -> np.ndarray:
    """Indices of the test subjects for the chosen split."""
    if split in ("site", "institution"):
        groups = np.array([GROUPINGS[split](s) for s in source.sites])
        if not holdout:
            raise SystemExit(f"--split {split} needs --holdout; choose from {sorted(set(groups))}")
        unknown = sorted(set(holdout) - set(groups))
        if unknown:
            raise SystemExit(f"unknown {split}(s) {unknown}; choose from {sorted(set(groups))}")
        return np.where(np.isin(groups, holdout))[0]
    strata = source.y if split == "random" else np.char.add(source.sites.astype(str), source.y.astype(str))
    _, test = train_test_split(np.arange(len(source.y)), test_size=test_size, random_state=seed, stratify=strata)
    return np.sort(test)


def cross_validate(source, args: argparse.Namespace) -> None:
    scheme = CV_SCHEMES[args.split]
    result = run(scheme, source, args.folds, args.repeats, args.workers, args.scoring)
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scheme}__{args.scoring}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    s = result["summary"]
    print(f"\n{scheme} ({args.scoring} scoring): {s['folds']} folds, {result['repeats']} repeat(s), fold mean ± SD")
    for key, label in (("balanced_accuracy", "bal. acc"), ("accuracy", "accuracy"), ("auc", "auc"),
                       ("sensitivity", "ASD rec."), ("specificity", "ctl rec.")):
        print(f"  {label:9s}: {s[key + '_mean']:.4f} ± {s[key + '_std']:.4f}")
    print(f"result: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Directory holding abide_dcan and abide_dcan_hcp")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--holdout", nargs="+", default=None, help="Sites or institutions to hold out")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for random / stratified")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for random / stratified")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: outputs/models/<split>_<holdout or seed>, or outputs/cv/ with --cv")
    parser.add_argument("--cv", action="store_true", help="Cross-validate instead of one split")
    parser.add_argument("--folds", type=int, default=10, help="k for random / stratified --cv")
    parser.add_argument("--repeats", type=int, default=1, help="Reshuffled k-fold repeats for --cv")
    parser.add_argument("--workers", type=int, default=2, help="Parallel folds for --cv (~1.5 GB each)")
    parser.add_argument("--scoring", choices=("per-group", "per-subject"), default=None,
                        help="Default: per-group for site / institution, per-subject for random / stratified")
    args = parser.parse_args()
    args.scoring = args.scoring or DEFAULT_SCORING[args.split]

    source = Source({atlas: args.data_root / path.name for atlas, path in DATA_DIRS.items()})
    if args.cv:
        cross_validate(source, args)
        return
    test_idx = choose_test(source, args.split, args.holdout, args.test_size, args.seed)
    train_idx = np.setdiff1d(np.arange(len(source.y)), test_idx)
    tag = "-".join(args.holdout) if args.holdout else f"seed{args.seed}"
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "models" / f"{args.split}_{tag}"

    # Site (or institution) groups drive the inner OOF threshold folds and the
    # per-group score standardization, as in the matching LOSO stage.
    group_key = "institution" if args.split == "institution" else "site"
    groups = np.array([GROUPINGS[group_key](s) for s in source.sites])
    print(f"split {args.split}: train {len(train_idx)}, test {len(test_idx)} subjects, {args.scoring} scoring")

    model = FrozenModel.fit({a: source.fc(a)[train_idx] for a in ATLASES}, source.age[train_idx],
                            source.sex[train_idx], source.y[train_idx], groups[train_idx],
                            per_subject=args.scoring == "per-subject")
    model.save(out_dir / "model.npz")
    score, pred = model.predict({a: source.fc(a)[test_idx] for a in ATLASES}, source.age[test_idx],
                                source.sex[test_idx], groups[test_idx])
    print()
    metrics = report(source.y[test_idx], score, pred, groups[test_idx])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "name": "iter10",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "model": "Gordon+HCP tangent (shrinkage 0.1) + age/sex, training-set standardization, "
                 f"L2 logistic regression C={model.logreg_c}",
        "scoring": args.scoring,
        "label_convention": "0 = ASD, 1 = control; score: higher = more likely control",
        "split": {"kind": args.split, "holdout": args.holdout, "test_size": args.test_size, "seed": args.seed,
                  "groups": group_key},
        "threshold": model.threshold,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_metrics": metrics,
        "training_file_ids": [source.file_ids[i] for i in train_idx],
        "test_file_ids": [source.file_ids[i] for i in test_idx],
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nmodel: {out_dir / 'model.npz'}")


if __name__ == "__main__":
    main()
