"""Compare the PR #30 paper-protocol run with the papers' own numbers.

Reads `paper_protocols.json` (written by `autoresearch.loop.paper_protocols`) and
prints two tables:

1. The model as it is: its own cut (set on training data only), fold means.
2. ROC read-out: on the same test scores, the ASD recall at the control recall
   the paper reports (per fold, then averaged). This is not a deployable rule
   (the cut is read off the test ROC); it only answers "with the same control
   recall as the paper, how many ASD would we catch?", which is fair to compare
   because AUC does not depend on any cut.

    uv run python -m scripts.paper_compare experiments/pr30_release/paper_protocols/paper_protocols.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_curve

ASD = 0
# protocol -> [(paper, ASD recall, control recall, accuracy, AUC or None)], all ABIDE I, values in %.
PAPERS = {
    "random-10fold": [("ASD-DiagNet 2019 (arXiv 1904.07577, Table II)", 67.8, 72.8, 70.1, None),
                      ("Heinsfeld 2018 (as re-run by ASD-DiagNet, Table II)", 69.3, 61.0, 65.4, None)],
    "brainnettf": [("BrainNetTF 2022 (NeurIPS, Table 1)", 72.5, 69.3, 71.0, 80.2)],
    "brainmass": [("BrainMass 2024 (arXiv 2403.01433, Table II)", 74.26, 71.89, 72.75, None)],
    "loso-site": [("Abraham 2017 (arXiv 1611.06066, Table 2, inter-site, all 871)", 61.0, 72.3, 66.8, None)],
}
NAMES = {"random-10fold": "Random 10-fold", "brainnettf": "70/10/20 x5, site-stratified",
         "brainmass": "70/15/15 x10, site-stratified", "loso-site": "Leave one site out"}


def asd_recall_at(y: np.ndarray, score: np.ndarray, control_recall: float) -> float:
    # score is high for controls; flip so ASD is the positive class.
    fpr, tpr, _ = roc_curve((y == ASD).astype(int), -score)
    return float(np.interp(1.0 - control_recall, fpr, tpr))


def main() -> None:
    data = json.loads(open(sys.argv[1], encoding="utf-8").read())
    print(f"{data['subjects']} subjects ({data['asd']} ASD), git {data['git']}\n")
    print("| Protocol | Who | ASD recall | Control recall | Accuracy | AUC |")
    print("|---|---|---|---|---|---|")
    for r in data["results"]:
        m, s = r["mean"], r["std"]
        print(f"| {NAMES[r['protocol']]} | **PR #30** | {100 * m['sensitivity']:.1f} ± {100 * s['sensitivity']:.1f} | "
              f"{100 * m['specificity']:.1f} ± {100 * s['specificity']:.1f} | {100 * m['accuracy']:.1f} | "
              f"{m['auc']:.3f} ± {s['auc']:.3f} |")
        for name, sen, spe, acc, auc in PAPERS[r["protocol"]]:
            print(f"| | {name} | {sen} | {spe} | {acc} | {'' if auc is None else f'{auc / 100:.3f}'} |")
    print("\nROC read-out: PR #30 ASD recall at the paper's control recall (fold mean)\n")
    print("| Protocol | Paper | Paper ASD recall | PR #30 ASD recall at the same control recall |")
    print("|---|---|---|---|")
    for r in data["results"]:
        for name, sen, spe, _, _ in PAPERS[r["protocol"]]:
            vals = [asd_recall_at(np.array(f["y"]), np.array(f["score"]), spe / 100) for f in r["per_fold"]
                    if len(set(f["y"])) == 2]
            print(f"| {NAMES[r['protocol']]} | {name} (control {spe}) | {sen} | {100 * np.mean(vals):.1f} |")


if __name__ == "__main__":
    main()
