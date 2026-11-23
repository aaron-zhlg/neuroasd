"""Chord diagrams of the connections the iter42 GNN relies on.

Same measure as the iter10 figures (`scripts/plot_chord_iter10.py` on the iter10
branch), so the two models can be read side by side: each edge is scored by its
*activation pattern* (Haufe et al. 2014), the Pearson correlation across subjects
between the edge's Fisher-z FC value and the model's output. For a nonlinear model
this says which connections the output tracks, not a causal weight.

Model output, two options:

    --scores FILE   (default for the release figures) held-out predictions from
                    `scripts/gnn_dcan_eval.py`: each subject is scored by the model that
                    did not see its site. Score and edges are centered within each
                    held-out site before correlating, so site offsets do not enter.
    no --scores     iter42's `run_fold`, unchanged, trained once on every subject and
                    scoring those same subjects. The GNN fits its training set almost
                    perfectly (training AUC ~0.97), so this mostly tracks the labels.

The output is logit(P(control)), flipped so positive = ASD direction: red = connection
is stronger in subjects the model scores as ASD, blue = weaker.

    --atlas ho      ABIDE I, PCP C-PAC, Harvard-Oxford 111 ROIs (release gnn-iter42-v1)
    --atlas power   ABIDE I + II, DCAN, Power 2011 94 parcels

Figures: `chord_rois.png` (top `--top-edges` edges) and `chord_networks.png`
(network pairs among the top `--top-percent` % of edges). Also writes `edges.csv`
and `networks.json`.

    uv run python -m scripts.gnn_chord --atlas power --data-dir data/abide_dcan_power --out-dir experiments/x/chord
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, PathPatch, Wedge
from matplotlib.path import Path as MplPath

import autoresearch.trial as trial
from neuroasd.fc_dataset import AbideFCDataset

ASD_UP, ASD_DOWN = "#C44E52", "#4C78A8"

# Harvard-Oxford networks as assigned in neuroasd.roi_atlas.
HO_NETWORKS = {
    "Default": ("DMN", "#2A9D8F"), "Frontoparietal": ("FPN", "#7A5EA8"), "Salience": ("SN", "#C75B7A"),
    "DorsalAttention": ("DAN", "#6B8E23"), "Language": ("LN", "#E09F3E"), "Auditory": ("AN", "#C9A227"),
    "Somatomotor": ("SMN", "#5B8FA8"), "Visual": ("VN", "#E07A3D"), "Limbic": ("BLN", "#8FBC8F"),
    "Subcortical": ("SCN", "#3D5A80"), "Unknown": ("?", "#B0B0B0"),
}
# Power 2011 systems (Power et al., Neuron 72:665), from the parcel names in parcels.txt.
POWER_NETWORKS = {
    "Default": ("DMN", "#2A9D8F"), "FrontoParietal": ("FPN", "#7A5EA8"), "CinguloOperc": ("CON", "#C75B7A"),
    "Salience": ("SAL", "#E07A9F"), "VentralAttn": ("VAN", "#E09F3E"), "DorsalAttn": ("DAN", "#6B8E23"),
    "Memory": ("MEM", "#8FBC8F"), "Auditory": ("AUD", "#C9A227"), "SMmouth": ("SMm", "#8CB4CC"),
    "SMhand": ("SMh", "#5B8FA8"), "Visual": ("VIS", "#E07A3D"), "Uncertain": ("UNC", "#B0B0B0"),
    "Subcortical": ("SUB", "#3D5A80"),
}
POWER_PREFIX = {
    "a3": "Default", "a4": "SMhand", "a5": "Visual", "a6": "FrontoParietal", "a7": "VentralAttn",
    "a14": "CinguloOperc", "a15": "DorsalAttn", "a16": "SMmouth", "a20": "Salience", "a21": "Memory",
    "a22": "Memory", "a23": "Memory", "a24": "Auditory",
}


def ho_table(data_dir: Path) -> list[dict]:
    from neuroasd.roi_atlas import roi_table

    rows = []
    for row in roi_table(data_dir):
        hemi = row["hemi"] if row["hemi"] in ("L", "R") else "M"
        rows.append({"index": int(row["index"]), "name": str(row["name"]), "hemi": hemi,
                     "network": str(row["network"]), "label": str(row["abbr"]).removeprefix("L.").removeprefix("R.")})
    return rows


def power_table(data_dir: Path) -> list[dict]:
    # One name per line; the first Power parcel has an empty name in the DCAN label table.
    names = [line.strip() for line in (data_dir / "processed" / "parcels.txt").read_text(encoding="utf-8").splitlines()]
    rows, counter = [], {}
    for index, name in enumerate(names):
        prefix = name.split("_")[0]
        if not name:
            hemi, network = "C", "Uncertain"
        elif name.isupper():  # FreeSurfer subcortical, e.g. THALAMUS_LEFT, BRAIN_STEM
            parts = name.split("_")
            hemi = "L" if parts[-1] == "LEFT" else "R" if parts[-1] == "RIGHT" else "M"
            network, label = "Subcortical", (parts[0][:4].title() if name != "UNKNOWN" else "?")
            if name == "UNKNOWN":
                network, hemi = "Uncertain", "C"
        else:
            hemi = "C"  # Power cortical parcels carry no hemisphere in the DCAN names
            network = POWER_PREFIX.get(prefix, "Uncertain")
        if network != "Subcortical":
            counter[network] = counter.get(network, 0) + 1
            label = f"{POWER_NETWORKS[network][0]}{counter[network]}"
        rows.append({"index": index, "name": name, "hemi": hemi, "network": network, "label": label})
    return rows


def ring_order(table: list[dict], networks: dict) -> np.ndarray:
    """Left hemisphere, then midline/unlateralized parcels, then right hemisphere mirrored."""
    rank = {name: i for i, name in enumerate(networks)}

    def key(row):
        r = rank[row["network"]]
        side = {"L": 0, "C": 1, "M": 1, "R": 2}[row["hemi"]]
        return (side, -r if side == 2 else r, row["index"])

    return np.array(sorted(range(len(table)), key=lambda i: key(table[i])))


def _radial_text(ax, angle, r, text, size, color, weight="normal"):
    rotation, ha = np.degrees(angle), "left"
    if np.cos(angle) < 0:
        rotation, ha = rotation + 180, "right"
    ax.text(r * np.cos(angle), r * np.sin(angle), text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", rotation=rotation, rotation_mode="anchor")


def plot_roi_chord(mat, table, networks, top_edges, out, title):
    p = len(table)
    order = ring_order(table, networks)
    pos = np.empty(p, dtype=int)
    pos[order] = np.arange(p)
    step = 2 * np.pi / p
    angles = np.pi / 2 - (np.arange(p) + 0.5) * step
    iu = np.triu_indices(p, 1)
    vals = mat[iu]
    top = np.argsort(-np.abs(vals))[:top_edges]
    edges = sorted(((int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in top), key=lambda e: abs(e[2]))
    vmax, vmin = max(abs(e[2]) for e in edges), min(abs(e[2]) for e in edges)

    fig, ax = plt.subplots(figsize=(12, 12.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.62, 1.62)
    for i, j, w in edges:
        a, b = angles[pos[i]], angles[pos[j]]
        strength = (abs(w) - vmin) / (vmax - vmin + 1e-12)
        ts = np.linspace(0, 1, 50)
        p0, p1 = np.array([np.cos(a), np.sin(a)]) * 0.97, np.array([np.cos(b), np.sin(b)]) * 0.97
        ctrl = (p0 + p1) * 0.12
        curve = (1 - ts)[:, None] ** 2 * p0 + 2 * ((1 - ts) * ts)[:, None] * ctrl + ts[:, None] ** 2 * p1
        ax.plot(curve[:, 0], curve[:, 1], color=ASD_UP if w > 0 else ASD_DOWN,
                alpha=0.25 + 0.6 * strength, linewidth=0.6 + 2.6 * strength, solid_capstyle="round")
    deg = np.degrees(step)
    for ring, roi in enumerate(order):
        start = np.degrees(angles[ring]) - deg / 2
        ax.add_patch(Wedge((0, 0), 1.06, start, start + deg, width=0.07,
                           facecolor=networks[table[roi]["network"]][1], edgecolor="white", linewidth=0.2))
    for ring, roi in enumerate(order):
        _radial_text(ax, angles[ring], 1.085, table[roi]["label"], 5.5, "#222")
    runs, start = [], 0
    for k in range(1, p + 1):
        if k == p or (table[order[k]]["network"], table[order[k]]["hemi"]) != (
                table[order[start]]["network"], table[order[start]]["hemi"]):
            runs.append((start, k - 1, table[order[start]]["network"]))
            start = k
    for first, last, network in runs:
        if last - first < 2:
            continue
        abbr, color = networks[network]
        _radial_text(ax, 0.5 * (angles[first] + angles[last]), 1.27, abbr, 9, color, "bold")
    ax.set_title(title, fontsize=12, pad=4)
    used = {r["network"] for r in table}
    handles = [Patch(color=c, label=f"{a}  {n}") for n, (a, c) in networks.items() if n in used]
    handles += [Patch(color=ASD_UP, label="stronger in ASD-scored"), Patch(color=ASD_DOWN, label="weaker in ASD-scored")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7.5, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return edges[::-1]


def _ribbon_path(a0, a1, b0, b1, r):
    def arc(t0, t1):
        ts = np.linspace(t0, t1, max(2, int(abs(t1 - t0) / 0.02)))
        return [(r * np.cos(t), r * np.sin(t)) for t in ts]

    pa, pb = arc(a0, a1), arc(b0, b1)[1:]
    verts = pa + [(0, 0), (r * np.cos(b0), r * np.sin(b0))] + pb + [(0, 0), (r * np.cos(a0), r * np.sin(a0))]
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(pa) - 1) + [MplPath.CURVE3, MplPath.CURVE3]
             + [MplPath.LINETO] * len(pb) + [MplPath.CURVE3, MplPath.CURVE3])
    return MplPath(verts, codes)


def network_counts(mat, table, networks, top_percent):
    nets = list(networks)
    net_of = np.array([nets.index(r["network"]) for r in table])
    iu = np.triu_indices(len(table), 1)
    vals = mat[iu]
    k = int(round(len(vals) * top_percent / 100))
    top = set(np.argsort(-np.abs(vals))[:k].tolist())
    m = len(nets)
    up, down, possible = np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, m))
    for e in range(len(vals)):
        a, b = sorted((net_of[iu[0][e]], net_of[iu[1][e]]))
        possible[a, b] += 1
        if e in top:
            (up if vals[e] > 0 else down)[a, b] += 1
    return {"networks": nets, "top_edges": k, "all_edges": len(vals), "asd_up": up, "asd_down": down,
            "possible": possible}


def plot_network_chord(nc, networks, out, title):
    nets, up, down = nc["networks"], nc["asd_up"], nc["asd_down"]
    counts = up + down
    degree = (counts + np.triu(counts, 1).T).sum(axis=1) + np.diag(counts)
    keep = [i for i in range(len(nets)) if degree[i] > 0]
    gap = 0.025
    span = 2 * np.pi - gap * len(keep)
    unit = span / degree[keep].sum()
    fig, ax = plt.subplots(figsize=(10.5, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.75, 1.45)
    arcs, cursor = {}, np.pi / 2
    for i in keep:
        width = degree[i] * unit
        arcs[i] = [cursor - width, cursor]
        cursor -= width + gap
    fill = {i: arcs[i][1] for i in keep}
    pairs = sorted(((i, j) for i in keep for j in keep if j >= i and counts[i, j] > 0), key=lambda ij: -counts[ij])
    for i, j in pairs:
        w = counts[i, j] * unit
        a1 = fill[i]
        a0 = a1 - w
        fill[i] = a0
        b1 = fill[j]
        b0 = b1 - w
        fill[j] = b0
        frac_up = up[i, j] / counts[i, j]
        ax.add_patch(PathPatch(_ribbon_path(a1, a0, b1, b0, 0.95), facecolor=ASD_UP if frac_up >= 0.5 else ASD_DOWN,
                               edgecolor="white", linewidth=0.3, alpha=0.25 + 0.5 * abs(frac_up - 0.5) * 2))
    for i in keep:
        abbr, color = networks[nets[i]]
        t0, t1 = arcs[i]
        ax.add_patch(Wedge((0, 0), 1.03, np.degrees(t0), np.degrees(t1), width=0.08, facecolor=color, edgecolor="white"))
        _radial_text(ax, 0.5 * (t0 + t1), 1.07, f"{abbr} ({int(degree[i])})", 10, color, "bold")
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[Patch(color=ASD_UP, label="mostly stronger in ASD-scored"),
                       Patch(color=ASD_DOWN, label="mostly weaker in ASD-scored")],
              loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
    ax.text(0, -1.62, "Ribbon width = number of top edges joining the two networks (self-loops = within-network);\n"
            "arc length = edges touching the network; shading = how one-sided the sign is.",
            ha="center", fontsize=8, color="#555")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", choices=["ho", "power"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Figure title prefix, e.g. 'GNN iter42, ABIDE I (HO 111)'")
    parser.add_argument("--top-edges", type=int, default=100)
    parser.add_argument("--top-percent", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores", type=Path, default=None, help="gnn_dcan_eval.py output with held-out predictions")
    args = parser.parse_args()

    networks = HO_NETWORKS if args.atlas == "ho" else POWER_NETWORKS
    table = ho_table(args.data_dir) if args.atlas == "ho" else power_table(args.data_dir)
    dataset = AbideFCDataset(args.data_dir)
    assert len(table) == int(dataset[0]["node_features"].shape[0]), "parcel table does not match the FC size"

    if args.scores is not None:
        held_out = json.loads(args.scores.read_text(encoding="utf-8"))
        by_id = {r["FILE_ID"]: i for i, r in enumerate(dataset.records)}
        rows = held_out["predictions"]
        everyone = [by_id[r["subject"]] for r in rows]
        folds = np.array([r["fold"] for r in rows])
        p = np.clip(np.array([r["p_control"] for r in rows], dtype=float), 1e-6, 1 - 1e-6)
        score_asd = -np.log(p / (1 - p))
        fit = {"auc": float("nan")}
        source = f"held-out scores ({held_out['folds_done']} {held_out['grouping']} folds), centered per fold"
    else:
        folds, source = None, "fit on all subjects and scored on them"
        everyone, score_asd, fit = _full_fit(dataset, args)
    _draw(args, dataset, table, networks, everyone, score_asd, folds, fit, source)


def _full_fit(dataset, args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    recorded: dict = {}
    real_auc = trial.roc_auc_score

    def recording_auc(labels, probs, *a, **k):  # only iter42's final TTA read calls this in trial.py
        recorded.update(labels=list(labels), probs=list(probs))
        return real_auc(labels, probs, *a, **k)

    trial.roc_auc_score = recording_auc
    config = trial.TrialConfig(epochs=args.epochs, batch_size=32, lr=1e-3, hidden_dim=64, dropout=0.5, weight_decay=1e-4)
    everyone = list(range(len(dataset)))
    print(f"fitting iter42 on all {len(everyone)} subjects ...", flush=True)
    fit = trial.run_fold(dataset, everyone, everyone, config, args.seed, device)
    p = np.clip(np.array(recorded["probs"], dtype=float), 1e-6, 1 - 1e-6)
    return everyone, -np.log(p / (1 - p)), fit  # ASD direction


def _draw(args, dataset, table, networks, everyone, score_asd, folds, fit, source):
    n = len(table)
    iu = np.triu_indices(n, 1)
    edges = np.stack([np.arctanh(np.clip(dataset[i]["node_features"].numpy(), -0.999, 0.999))[iu]
                      for i in everyone]).astype(np.float64)
    centered = np.asarray(score_asd, dtype=float).copy()
    groups = np.zeros(len(everyone), dtype=int) if folds is None else np.unique(folds, return_inverse=True)[1]
    for g in np.unique(groups):  # within-fold centering (one group when there are no folds)
        m = groups == g
        edges[m] -= edges[m].mean(axis=0)
        centered[m] -= centered[m].mean()
    r = edges.T @ centered / (np.sqrt((edges ** 2).sum(axis=0)) * np.sqrt((centered ** 2).sum()) + 1e-12)
    mat = np.zeros((n, n))
    mat[iu] = r
    mat = mat + mat.T

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"activation pattern (corr. with model output), {len(everyone):,} subjects, {source}"
    top = plot_roi_chord(mat, table, networks, args.top_edges, args.out_dir / "chord_rois.png",
                         f"{args.label}: top {args.top_edges} connections by model pattern\n{subtitle}")
    with (args.out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "parcel_a", "network_a", "parcel_b", "network_b", "pattern_asd_direction"])
        for rank, (i, j, w) in enumerate(top, 1):
            writer.writerow([rank, table[i]["name"], table[i]["network"], table[j]["name"], table[j]["network"], f"{w:.4f}"])
    nc = network_counts(mat, table, networks, args.top_percent)
    plot_network_chord(nc, networks, args.out_dir / "chord_networks.png",
                       f"{args.label}: network pairs among the top {nc['top_edges']:,} edges "
                       f"({args.top_percent:g}%)\n{subtitle}")
    rate = nc["top_edges"] / nc["all_edges"]
    pairs = []
    for a in range(len(nc["networks"])):
        for b in range(a, len(nc["networks"])):
            if nc["possible"][a, b] == 0:
                continue
            cnt = nc["asd_up"][a, b] + nc["asd_down"][a, b]
            pairs.append({"a": nc["networks"][a], "b": nc["networks"][b], "top": int(cnt),
                          "asd_up": int(nc["asd_up"][a, b]), "asd_down": int(nc["asd_down"][a, b]),
                          "possible": int(nc["possible"][a, b]),
                          "enrichment": round(float(cnt / (nc["possible"][a, b] * rate)), 2)})
    pairs.sort(key=lambda x: -x["top"])
    (args.out_dir / "networks.json").write_text(json.dumps({
        "method": "Pearson r between each edge's Fisher-z FC and logit(P(control)) of iter42, "
                  "sign flipped so positive = ASD direction; scores: " + source,
        "atlas": args.atlas, "subjects": len(everyone), "nodes": n,
        "full_fit_training_auc": None if np.isnan(fit["auc"]) else round(float(fit["auc"]), 4),
        "pattern_abs_max": round(float(np.abs(r).max()), 4),
        "top_percent": args.top_percent, "top_edges": nc["top_edges"], "all_edges": nc["all_edges"],
        "network_pairs": pairs,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_auc": fit["auc"], "abs_max": float(np.abs(r).max()), "top_pairs": pairs[:8]}, indent=1))


if __name__ == "__main__":
    main()
