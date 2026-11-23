"""Score the iter42 GNN (PR #13) on ABIDE I + II (DCAN preprocessing, 1,443 subjects).

The GNN trials (PRs #2-13) only ever ran on ABIDE I (PCP C-PAC, Harvard-Oxford 111
ROIs, 20 sites, 884 subjects). PCP never released ABIDE II, so the same input does not
exist for it. This script runs iter42 unchanged on the DCAN ABIDE I + II build that
iter10 uses, with the Power 2011 atlas (94 parcels, the closest in size to HO 111),
and on iter10's held-out groups:

    loso-site          33 two-class sites (same list as iter10's harness)
    loso-institution   23 two-class institutions (same mapping as iter10)

The training and scoring code is iter42's `run_fold`, unchanged: Fisher-z node
features, train-fold per-site edge quantile maps, cosine LR, per-epoch EMA read once at
the final epoch, site-adversarial head, CORAL, site group-DRO, site-stratified mixup,
DropEdge p=0.1, and 8-draw DropEdge averaging at the final read. Defaults: 100 epochs,
batch 32, lr 1e-3, hidden 64, dropout 0.5, seed 42 (same as the loso-full trials).
The only addition is a pass-through wrapper on the `roc_auc_score` that iter42's final
read calls, which keeps the per-subject probabilities, so ASD recall, control recall
and balanced accuracy are computed from exactly the scores behind the reported AUC.

    uv run python -m scripts.gnn_dcan_eval --data-dir data/abide_dcan_power --grouping site
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import autoresearch.trial as trial
from neuroasd.fc_dataset import AbideFCDataset

ASD_LABEL = 0
# Same mapping as iter10's neuroasd.linear_baseline.INSTITUTIONS.
INSTITUTIONS = {
    "ABIDEII-KKI_1": "KKI", "ABIDEII-KUL_3": "LEUVEN", "LEUVEN_1": "LEUVEN", "LEUVEN_2": "LEUVEN",
    "ABIDEII-NYU_1": "NYU", "ABIDEII-NYU_2": "NYU", "ABIDEII-OHSU_1": "OHSU", "ABIDEII-ONRC_2": "OLIN",
    "ABIDEII-SDSU_1": "SDSU", "ABIDEII-TCD_1": "TRINITY", "ABIDEII-UCLA_1": "UCLA", "UCLA_1": "UCLA",
    "UCLA_2": "UCLA", "UM_1": "UM", "UM_2": "UM", "ABIDEII-USM_1": "USM",
}

_last_eval: dict = {}
_roc_auc_score = trial.roc_auc_score


def _recording_roc_auc_score(labels, probs, *args, **kwargs):
    """Pass-through: in trial.py only iter42's final TTA read calls this."""
    _last_eval.update(labels=list(labels), probs=list(probs))
    return _roc_auc_score(labels, probs, *args, **kwargs)


trial.roc_auc_score = _recording_roc_auc_score


def fold_metrics(y: np.ndarray, p_control: np.ndarray) -> dict[str, float]:
    pred = (p_control > 0.5).astype(int)  # argmax of 2 logits
    asd, ctl = y == ASD_LABEL, y != ASD_LABEL
    sens = float((pred[asd] == ASD_LABEL).mean())
    spec = float((pred[ctl] != ASD_LABEL).mean())
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": (sens + spec) / 2,
            "auc": float(roc_auc_score(y, p_control)), "asd_recall": sens, "control_recall": spec}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="data/abide_dcan (Gordon) or data/abide_dcan_hcp")
    parser.add_argument("--grouping", choices=["site", "institution"], required=True)
    parser.add_argument("--only", nargs="*", default=None, help="Score only these held-out groups (smoke test)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.backends.mps.is_available() else "mps")
    dataset = AbideFCDataset(args.data_dir)
    y_all = np.array([int(r["label"]) for r in dataset.records])
    sites = np.array([r["SITE_ID"] for r in dataset.records])
    groups = np.array([INSTITUTIONS.get(s, s) if args.grouping == "institution" else s for s in sites])
    eligible = sorted(g for g in set(groups) if len(np.unique(y_all[groups == g])) == 2)
    if args.only:
        eligible = [g for g in eligible if g in args.only]
    config = trial.TrialConfig(epochs=args.epochs, batch_size=32, lr=1e-3, hidden_dim=64, dropout=0.5,
                               weight_decay=1e-4)
    print(f"{len(dataset)} subjects, {int(dataset[0]['node_features'].shape[0])} nodes, "
          f"{len(eligible)} held-out {args.grouping}s, device {device}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    folds, predictions = [], []
    started = time.time()
    for position, group in enumerate(eligible, 1):
        test_idx = np.where(groups == group)[0].tolist()
        train_idx = np.where(groups != group)[0].tolist()
        t0 = time.time()
        _last_eval.clear()
        raw = trial.run_fold(dataset, train_idx, test_idx, config, args.seed, device)
        y, p = np.array(_last_eval["labels"]), np.array(_last_eval["probs"])
        m = fold_metrics(y, p)
        assert abs(m["auc"] - raw["auc"]) < 1e-9, "recorded probabilities do not match iter42's metric"
        assert abs(m["accuracy"] - raw["accuracy"]) < 1e-9, "recorded probabilities do not match iter42's accuracy"
        folds.append({"test_group": group, "test_size": len(test_idx), "train_size": len(train_idx),
                      **m, "best_auc_diagnostic": raw["best_auc"], "seconds": round(time.time() - t0, 1)})
        predictions += [{"fold": group, "subject": i, "label": int(l), "p_control": round(float(q), 5)}
                        for i, l, q in zip([dataset.records[k]["FILE_ID"] for k in test_idx], y, p)]
        print(f"  [{position}/{len(eligible)}] {group:<16} n={len(test_idx):<4} bacc={m['balanced_accuracy']:.3f} "
              f"acc={m['accuracy']:.3f} auc={m['auc']:.3f} asd_rec={m['asd_recall']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        summary = {f"{k}_{s}": round(float(fn([f[k] for f in folds])), 4)
                   for k in ("balanced_accuracy", "accuracy", "auc", "asd_recall", "control_recall")
                   for s, fn in (("mean", np.mean), ("std", np.std))}
        args.out.write_text(json.dumps({
            "model": "iter42 GNN (PR #13, release gnn-iter42-v1), training and scoring code unchanged",
            "data": str(args.data_dir), "grouping": args.grouping, "seed": args.seed, "config": vars(config),
            "model_selection": "final-epoch EMA weights, no selection on the held-out group",
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "folds_done": len(folds), "folds_total": len(eligible),
            "runtime_seconds": round(time.time() - started, 1),
            "summary": summary, "folds": folds, "predictions": predictions,
        }, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
