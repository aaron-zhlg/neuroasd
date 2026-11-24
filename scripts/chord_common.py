"""Chord drawing helpers shared by the chord scripts (taken from scripts/plot_chord_iter10.py)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch, Wedge
from matplotlib.path import Path as MplPath

from autoresearch.trial import DATA_DIRS, Source

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "model"

# Gordon 2016 community names as they appear in parcels.txt, in ring order (front to back
# roughly: control/attention systems, default, sensory, then unassigned and subcortical).
NETWORKS = {
    "Default": ("DMN", "#2A9D8F"),
    "FrontoParietal": ("FPN", "#7A5EA8"),
    "CinguloOperc": ("CON", "#C75B7A"),
    "Salience": ("SAL", "#E07A9F"),
    "VentralAttn": ("VAN", "#E09F3E"),
    "DorsalAttn": ("DAN", "#6B8E23"),
    "CinguloParietal": ("CPN", "#A0522D"),
    "RetrosplenialTemporal": ("RTN", "#8FBC8F"),
    "Auditory": ("AUD", "#C9A227"),
    "SMmouth": ("SMm", "#8CB4CC"),
    "SMhand": ("SMh", "#5B8FA8"),
    "Visual": ("VIS", "#E07A3D"),
    "None": ("None", "#B0B0B0"),
    "Subcortical": ("SUB", "#3D5A80"),
}
ASD_UP, ASD_DOWN = "#C44E52", "#4C78A8"


def parcel_table(data_dir: Path) -> list[dict]:
    names = (data_dir / "processed" / "parcels.txt").read_text(encoding="utf-8").split()
    rows = []
    for index, name in enumerate(names):
        parts = name.split("_")
        if parts[0].isdigit():  # "12_L_Default"
            hemi, network, label = parts[1], "_".join(parts[2:]), f"{parts[1]}{parts[0]}"
        else:  # "THALAMUS_LEFT", "BRAIN_STEM"
            hemi = "L" if parts[-1] == "LEFT" else "R" if parts[-1] == "RIGHT" else "M"
            network = "Subcortical"
            label = f"{hemi}.{parts[0][:4].title()}" if hemi != "M" else "Stem"
        if network not in NETWORKS:
            raise ValueError(f"unknown Gordon network {network!r} in {name!r}")
        rows.append({"index": index, "name": name, "hemi": hemi, "network": network, "label": label})
    return rows


def ring_order(table: list[dict]) -> np.ndarray:
    """Left hemisphere clockwise from 12 o'clock, right hemisphere mirrored back, midline last.

    The same network meets itself across the midline at the bottom of the ring.
    """
    rank = {name: i for i, name in enumerate(NETWORKS)}

    def key(row):
        r = rank[row["network"]]
        if row["hemi"] == "L":
            return (0, r, row["index"])
        if row["hemi"] == "R":
            return (2, -r, row["index"])
        return (1, r, row["index"])  # brain stem sits between the two subcortical runs

    return np.array(sorted(range(len(table)), key=lambda i: key(table[i])))


def to_matrix(vector: np.ndarray, p: int) -> np.ndarray:
    iu = np.triu_indices(p)  # tangent_project keeps the diagonal, row-major upper triangle
    mat = np.zeros((p, p))
    mat[iu] = vector
    mat = mat + np.triu(mat, 1).T
    np.fill_diagonal(mat, 0.0)  # diagonal = node "variance" terms, not connections
    return mat


def _ribbon_path(a0, a1, b0, b1, r):
    """Closed path: arc a0->a1 on the ring, bezier to b0, arc b0->b1, bezier back to a0."""
    def arc(t0, t1):
        ts = np.linspace(t0, t1, max(2, int(abs(t1 - t0) / 0.02)))
        return [(r * np.cos(t), r * np.sin(t)) for t in ts]

    verts, codes = [], []
    pa = arc(a0, a1)
    verts += pa
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(pa) - 1)
    verts += [(0, 0), (r * np.cos(b0), r * np.sin(b0))]
    codes += [MplPath.CURVE3, MplPath.CURVE3]
    pb = arc(b0, b1)[1:]
    verts += pb
    codes += [MplPath.LINETO] * len(pb)
    verts += [(0, 0), (r * np.cos(a0), r * np.sin(a0))]
    codes += [MplPath.CURVE3, MplPath.CURVE3]
    return MplPath(verts, codes)


def _radial_text(ax, angle, r, text, size, color, weight="normal"):
    rotation = np.degrees(angle)
    ha = "left"
    if np.cos(angle) < 0:
        rotation += 180
        ha = "right"
    ax.text(r * np.cos(angle), r * np.sin(angle), text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", rotation=rotation, rotation_mode="anchor")


def plot_roi_chord(mat: np.ndarray, table: list[dict], top_edges: int, out: Path, subtitle: str) -> list[tuple]:
    p = len(table)
    order = ring_order(table)
    pos = np.empty(p, dtype=int)
    pos[order] = np.arange(p)
    step = 2 * np.pi / p
    angles = np.pi / 2 - (np.arange(p) + 0.5) * step  # clockwise from 12 o'clock

    iu = np.triu_indices(p, 1)
    vals = mat[iu]
    top = np.argsort(-np.abs(vals))[:top_edges]
    edges = sorted(((int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in top), key=lambda e: abs(e[2]))
    vmax = max(abs(e[2]) for e in edges)
    vmin = min(abs(e[2]) for e in edges)

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
        color = NETWORKS[table[roi]["network"]][1]
        start = np.degrees(angles[ring]) - deg / 2
        ax.add_patch(Wedge((0, 0), 1.06, start, start + deg, width=0.07, facecolor=color, edgecolor="none"))

    involved = {i for i, j, _ in edges} | {j for i, j, _ in edges}
    for roi in involved:
        _radial_text(ax, angles[pos[roi]], 1.085, table[roi]["label"], 5.5, "#222")

    # Network labels at the middle of each same-network run.
    runs, start = [], 0
    for k in range(1, p + 1):
        if k == p or (table[order[k]]["network"], table[order[k]]["hemi"]) != (
                table[order[start]]["network"], table[order[start]]["hemi"]):
            runs.append((start, k - 1, table[order[start]]["network"]))
            start = k
    for first, last, network in runs:
        if last - first < 3:
            continue
        abbr, color = NETWORKS[network]
        _radial_text(ax, 0.5 * (angles[first] + angles[last]), 1.27, abbr, 9, color, "bold")

    ax.set_title(f"{MODEL_NAME}: top {top_edges} Gordon connections by model pattern\n{subtitle}", fontsize=12, pad=4)
    handles = [Patch(color=c, label=f"{a}  {n}") for n, (a, c) in NETWORKS.items()]
    handles += [Patch(color=ASD_UP, label="stronger in ASD-scored"), Patch(color=ASD_DOWN, label="weaker in ASD-scored")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=6, fontsize=7.5, frameon=False)
    ax.text(0, -1.5, "Left hemisphere on the right half (clockwise from top), right hemisphere mirrored;\n"
            "the same network meets across the midline at the bottom.", ha="center", fontsize=7.5, color="#555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return edges[::-1]


def network_counts(mat: np.ndarray, table: list[dict], top_percent: float) -> dict:
    p = len(table)
    nets = list(NETWORKS)
    net_of = np.array([nets.index(r["network"]) for r in table])
    iu = np.triu_indices(p, 1)
    vals = mat[iu]
    k = int(round(len(vals) * top_percent / 100))
    top = np.argsort(-np.abs(vals))[:k]
    m = len(nets)
    pos_counts, neg_counts, total = np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, m))
    for e in range(len(vals)):
        a, b = sorted((net_of[iu[0][e]], net_of[iu[1][e]]))
        total[a, b] += 1
    for e in top:
        a, b = sorted((net_of[iu[0][e]], net_of[iu[1][e]]))
        (pos_counts if vals[e] > 0 else neg_counts)[a, b] += 1
    return {"networks": nets, "top_edges": k, "all_edges": len(vals),
            "asd_up": pos_counts, "asd_down": neg_counts, "possible": total}


def plot_network_chord(nc: dict, out: Path, subtitle: str) -> None:
    nets, up, down = nc["networks"], nc["asd_up"], nc["asd_down"]
    counts = up + down
    full = counts + np.triu(counts, 1).T
    degree = full.sum(axis=1) + np.diag(counts)  # within-network edges use both ends of the same arc
    keep = [i for i in range(len(nets)) if degree[i] > 0]
    gap = 0.025
    span = 2 * np.pi - gap * len(keep)
    total = degree[keep].sum()

    fig, ax = plt.subplots(figsize=(10.5, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.75, 1.45)

    arcs, cursor = {}, np.pi / 2
    for i in keep:
        width = span * degree[i] / total
        arcs[i] = [cursor - width, cursor]
        cursor -= width + gap
    fill = {i: arcs[i][1] for i in keep}  # consume each arc clockwise from its start

    pairs = sorted(((i, j) for i in keep for j in keep if j >= i and counts[i, j] > 0), key=lambda ij: -counts[ij])
    unit = span / total
    for i, j in pairs:
        w = counts[i, j] * unit
        a1 = fill[i]
        a0 = a1 - w
        fill[i] = a0
        b1 = fill[j]
        b0 = b1 - w
        fill[j] = b0
        frac_up = up[i, j] / counts[i, j]
        color = ASD_UP if frac_up >= 0.5 else ASD_DOWN
        alpha = 0.25 + 0.5 * abs(frac_up - 0.5) * 2
        ax.add_patch(PathPatch(_ribbon_path(a1, a0, b1, b0, 0.95), facecolor=color, edgecolor="white",
                               linewidth=0.3, alpha=alpha))

    for i in keep:
        abbr, color = NETWORKS[nets[i]]
        t0, t1 = arcs[i]
        ax.add_patch(Wedge((0, 0), 1.03, np.degrees(t0), np.degrees(t1), width=0.08, facecolor=color, edgecolor="white"))
        mid = 0.5 * (t0 + t1)
        _radial_text(ax, mid, 1.07, f"{abbr} ({int(degree[i])})", 10, color, "bold")

    ax.set_title(f"{MODEL_NAME}: network pairs among the top {nc['top_edges']:,} Gordon edges "
                 f"({nc['top_edges'] / nc['all_edges']:.0%})\n{subtitle}", fontsize=12)
    handles = [Patch(color=ASD_UP, label="mostly stronger in ASD-scored"),
               Patch(color=ASD_DOWN, label="mostly weaker in ASD-scored")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
    ax.text(0, -1.62, "Ribbon width = number of top edges joining the two networks (self-loops = within-network);\n"
            "arc length = edges touching the network; shading = how one-sided the sign is.",
            ha="center", fontsize=8, color="#555")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


