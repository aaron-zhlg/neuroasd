"""Chord diagrams of the connections the PR #30 model relies on.

PR #30 is a stack of several models (Gordon+HCP, Power, Markov, a tree model),
so there is no single weight per connection. Each Gordon connection is scored by
its activation pattern (Haufe et al. 2014): the correlation, across subjects,
between the connection's Fisher-z FC and the model's score. The score is the
out-of-fold score from random 10-fold on the 1,120 development subjects (each
subject scored by a model that never saw them). Both FC and score are centred
within each site first, so site differences do not show up as connections.

Sign: the score is high for controls, so the pattern is flipped: red = the
connection is stronger in subjects the model scores as ASD, blue = weaker.

    uv run python -m scripts.plot_chord_pr30 --out-dir experiments/pr30_release/chord
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

import scripts.chord_common as cc
from autoresearch.trial import DATA_DIRS, Source, Split, check_prediction


def oof_scores(source, workers: int) -> np.ndarray:
    from neuroasd.model import fit_predict

    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(np.zeros(len(source.y)), source.y))
    scores = np.full(len(source.y), np.nan)

    def one(k):
        train, test = folds[k]
        random.seed(k)
        np.random.seed(k)
        s, p = fit_predict(Split(source, train, source.sites, True), Split(source, test, source.sites, False))
        s, _ = check_prediction(s, p, len(test))
        return test, s

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for test, s in pool.map(one, range(10)):
            scores[test] = s
    return scores


def site_center(x: np.ndarray, sites: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).copy()
    for s in np.unique(sites):
        m = sites == s
        x[m] -= x[m].mean(axis=0)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-edges", type=int, default=100)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cc.MODEL_NAME = "PR #30"

    source = Source()
    table = cc.parcel_table(DATA_DIRS["gordon"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_file = args.out_dir / "oof_scores.json"
    if score_file.is_file():
        scores = np.array(json.loads(score_file.read_text())["score"])
    else:
        print("random 10-fold out-of-fold scores ...", flush=True)
        scores = oof_scores(source, args.workers)
        score_file.write_text(json.dumps({"file_id": source.file_ids, "y": source.y.tolist(),
                                          "score": scores.tolist()}) + "\n", encoding="utf-8")

    fc = source.fc("gordon")
    p = fc.shape[1]
    iu = np.triu_indices(p, 1)
    z = np.arctanh(np.clip(fc[:, iu[0], iu[1]], -0.999, 0.999))
    z = site_center(z, source.sites)
    s = site_center(scores[:, None], source.sites)[:, 0]
    r = (z.T @ s) / (np.sqrt((z ** 2).sum(axis=0)) * np.sqrt((s ** 2).sum()) + 1e-12)
    mat = np.zeros((p, p))
    mat[iu] = -r  # ASD direction
    mat = mat + mat.T

    subtitle = (f"activation pattern: corr. of FC with out-of-fold score, centred within site, "
                f"{len(source.y):,} subjects")
    edges = cc.plot_roi_chord(mat, table, args.top_edges, args.out_dir / "chord_rois.png", subtitle)
    with (args.out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "parcel_a", "network_a", "parcel_b", "network_b", "pattern_asd_direction"])
        for rank, (i, j, w) in enumerate(edges, 1):
            writer.writerow([rank, table[i]["name"], table[i]["network"], table[j]["name"], table[j]["network"],
                             f"{w:.4f}"])

    nc = cc.network_counts(mat, table, args.top_percent)
    cc.plot_network_chord(nc, args.out_dir / "chord_networks.png", subtitle)
    nets, rate = nc["networks"], nc["top_edges"] / nc["all_edges"]
    pairs = []
    for a in range(len(nets)):
        for b in range(a, len(nets)):
            if nc["possible"][a, b] == 0:
                continue
            n = nc["asd_up"][a, b] + nc["asd_down"][a, b]
            pairs.append({"a": nets[a], "b": nets[b], "top": int(n), "asd_up": int(nc["asd_up"][a, b]),
                          "asd_down": int(nc["asd_down"][a, b]), "possible": int(nc["possible"][a, b]),
                          "enrichment": round(float(n / (nc["possible"][a, b] * rate)), 2)})
    pairs.sort(key=lambda row: -row["top"])
    (args.out_dir / "networks.json").write_text(json.dumps({
        "subjects": int(len(source.y)), "top_percent": args.top_percent, "top_edges": nc["top_edges"],
        "pattern_abs_max": round(float(np.abs(mat).max()), 4), "network_pairs": pairs}, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(pairs[:8], indent=1))


if __name__ == "__main__":
    main()
