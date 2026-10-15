"""Figures for GNN attribution: ROI ranking bars and a connectivity chord plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Wedge

from neuroasd.dcan_atlas import NETWORK_ABBR, NETWORK_COLORS, sort_order


def plot_top_rois(
    node_scores: np.ndarray,
    table: list[dict],
    output_path: Path,
    top_k: int = 20,
    title: str = "Top ROIs by |attribution|",
    xlabel: str = "Attribution (ASD direction)",
) -> None:
    order = np.argsort(-np.abs(node_scores))[:top_k]
    names = [f"{table[i]['abbr']}  {table[i]['short']}" for i in order][::-1]
    values = node_scores[order][::-1]
    colors = [NETWORK_COLORS.get(str(table[i]["network"]), "#888") for i in order][::-1]

    height = max(4.0, 0.32 * top_k + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, height))
    ax.barh(range(top_k), values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(top_k), labels=names, fontsize=8)
    ax.axvline(0.0, color="#333", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _bezier_chord(ax, p0, p1, width: float, color: str, alpha: float) -> None:
    mid = ((p0[0] + p1[0]) * 0.18, (p0[1] + p1[1]) * 0.18)
    ts = np.linspace(0.0, 1.0, 40)
    xs = (1 - ts) ** 2 * p0[0] + 2 * (1 - ts) * ts * mid[0] + ts**2 * p1[0]
    ys = (1 - ts) ** 2 * p0[1] + 2 * (1 - ts) * ts * mid[1] + ts**2 * p1[1]
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=width, solid_capstyle="round")


def _network_spans(order: np.ndarray, table: list[dict]) -> list[tuple[int, int, str]]:
    """Consecutive runs of the same network *and* hemisphere.

    Left FPN and right FPN stay two labels that meet at 6 o'clock, instead of
    collapsing into one band.
    """
    spans: list[tuple[int, int, str]] = []
    start = 0
    current = (str(table[int(order[0])]["hemi"]), str(table[int(order[0])]["network"]))
    for i in range(1, len(order)):
        key = (str(table[int(order[i])]["hemi"]), str(table[int(order[i])]["network"]))
        if key != current:
            spans.append((start, i - 1, current[1]))
            start = i
            current = key
    spans.append((start, len(order) - 1, current[1]))
    return spans


def plot_chord(
    edge_scores: np.ndarray,
    table: list[dict],
    output_path: Path,
    top_k: int = 80,
    title: str = "Attributed connections",
    label_rois: bool | None = None,
) -> None:
    """Draw attributed edges with a two-level ring: ROI ticks + network band.

    Per-ROI names are omitted when the atlas is large (Gordon / HCP); the
    network band stays. Pass ``label_rois`` to override.
    """
    n = edge_scores.shape[0]
    if label_rois is None:
        label_rois = n <= 120
    order = sort_order(table)
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)

    abs_edges = np.abs(np.triu(edge_scores, k=1))
    flat = abs_edges.ravel()
    keep = min(top_k, int((flat > 0).sum()) or top_k)
    top_idx = np.argpartition(flat, -keep)[-keep:]
    pairs = [(int(i // n), int(i % n), float(edge_scores[i // n, i % n])) for i in top_idx]
    pairs.sort(key=lambda item: abs(item[2]))

    max_abs = max((abs(w) for _, _, w in pairs), default=1.0) or 1.0
    # Start at 12 o'clock; each ROI occupies an equal slice.
    step = 2 * np.pi / n
    centers = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + np.pi / 2 + step / 2
    radius = 1.0
    xs = radius * np.cos(centers)
    ys = radius * np.sin(centers)

    fig, ax = plt.subplots(figsize=(11.5, 11.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.set_title(title, pad=8)

    for i, j, weight in pairs:
        a, b = inv[i], inv[j]
        width = 0.35 + 2.4 * (abs(weight) / max_abs)
        alpha = 0.16 + 0.5 * (abs(weight) / max_abs)
        color = "#C44E52" if weight >= 0 else "#4C78A8"
        _bezier_chord(ax, (xs[a], ys[a]), (xs[b], ys[b]), width, color, alpha)

    roi_width = 0.07
    theta = 360.0 / n
    roi_outer = radius + roi_width
    wedge_lw = 0.08 if n > 120 else 0.18

    for ring_pos, roi_index in enumerate(order):
        color = NETWORK_COLORS.get(str(table[roi_index]["network"]), "#888")
        start = np.degrees(centers[ring_pos]) - theta / 2
        ax.add_patch(
            Wedge(
                (0, 0),
                roi_outer,
                start,
                start + theta,
                width=roi_width,
                facecolor=color,
                edgecolor="white",
                linewidth=wedge_lw,
            )
        )

    def _radial_text(angle: float, r: float, text: str, fontsize: float, color: str, weight: str = "normal") -> None:
        rotation = np.degrees(angle)
        if np.cos(angle) < 0:
            rotation += 180
            ha = "right"
        else:
            ha = "left"
        ax.text(
            r * np.cos(angle),
            r * np.sin(angle),
            text,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )

    # Inside → outside: ROI ring, optional ROI names, network names.
    if label_rois:
        roi_label_r = roi_outer + 0.035
        for ring_pos, roi_index in enumerate(order):
            label = str(table[roi_index]["abbr"]).removeprefix("L.").removeprefix("R.")
            _radial_text(centers[ring_pos], roi_label_r, label, 4.3, "#222")

    net_label_r = roi_outer + (0.12 if not label_rois else 0.28)
    for first, last, network in _network_spans(order, table):
        mid = 0.5 * (centers[first] + centers[last])
        _radial_text(
            mid,
            net_label_r,
            NETWORK_ABBR.get(network, network),
            8.5,
            NETWORK_COLORS.get(network, "#333"),
            weight="bold",
        )

    present = [name for name in NETWORK_COLORS if name != "Unknown" and any(row["network"] == name for row in table)]
    handles = [
        FancyBboxPatch((0, 0), 0.01, 0.01, boxstyle="square,pad=0.4", facecolor=NETWORK_COLORS[name])
        for name in present
    ]
    ax.legend(
        handles,
        [f"{NETWORK_ABBR.get(name, '?')}  {name}" for name in present],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=min(5, max(1, len(present))),
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
