"""Attribute the current model and write chord / ROI figures.

Auto-detects the model kind:

* a ``.pt`` checkpoint or ``--loso-folds-dir`` → GNN integrated gradients
* otherwise → linear coefficients from ``neuroasd.model.fit_predict``

GNN::

    uv run python -m neuroasd.explain \\
        --checkpoint outputs/gcn_baseline/final_model.pt \\
        --output-dir experiments/baseline_gcn_v1/figures

Linear (Gordon / HCP chords under ``outputs/explain_linear/``)::

    uv run python -m neuroasd.explain --kind linear
    # Same outer split as the institution-out scores (not a single all-subject fit):
    uv run python -m neuroasd.explain --kind linear --group-by institution \\
        --output-dir outputs/explain_linear_institution
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from neuroasd.eval import load_model
from neuroasd.fc_dataset import AbideFCDataset, collate_graphs
from neuroasd.gcn import SimpleGCN
from neuroasd.plots import plot_chord, plot_top_rois
from neuroasd.roi_atlas import roi_table
from neuroasd.train import pick_device

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"
ASD_CLASS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("auto", "gnn", "linear"),
        default="auto",
        help="Model family. auto: GNN if a checkpoint / LOSO fold dir is given, else linear.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to final_model.pt (or a directory containing it)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write figures and tables (default: <checkpoint-dir>/explain)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="Subjects to attribute. Falls back to all if split.json is missing.",
    )
    parser.add_argument(
        "--site",
        type=str,
        default="",
        help="If set, only attribute subjects from this SITE_ID (for LOSO folds).",
    )
    parser.add_argument(
        "--loso-folds-dir",
        type=Path,
        default=None,
        help="If set, explain every fold's held-out site and write the mean "
        "attribution to --output-dir (LOSO experiment figure).",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="split.json from training (default: same directory as the checkpoint)",
    )
    parser.add_argument(
        "--target",
        choices=("asd", "pred"),
        default="asd",
        help="Logit to attribute: always ASD, or the model's predicted class",
    )
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-rois", type=int, default=20)
    parser.add_argument("--top-edges", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--group-by",
        choices=("all", "site", "institution"),
        default="all",
        help="Linear only. all = one fit on every subject (draft figure). "
        "institution = mean coefficients over the 23 institution-out training "
        "folds (same split as those scores). site = the 33 site-out folds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Linear only: parallel folds when --group-by is site or institution.",
    )
    return parser.parse_args()


def is_gnn_path(path: Path | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    if candidate.suffix == ".pt":
        return True
    if candidate.is_dir():
        return (candidate / "final_model.pt").exists() or (candidate / "best_model.pt").exists()
    return False


def resolve_kind(kind: str, checkpoint: Path | None, loso_folds_dir: Path | None) -> str:
    if kind != "auto":
        return kind
    if loso_folds_dir is not None or is_gnn_path(checkpoint):
        return "gnn"
    return "linear"


def resolve_checkpoint(path: Path | None) -> Path:
    if path is None:
        candidate = Path(__file__).resolve().parents[1] / "outputs" / "gcn_baseline"
    else:
        candidate = path
    if candidate.is_dir():
        for name in ("final_model.pt", "best_model.pt"):
            found = candidate / name
            if found.exists():
                return found
        raise FileNotFoundError(f"No final_model.pt or best_model.pt in {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def select_indices(
    dataset: AbideFCDataset,
    split_name: str,
    split_json: Path | None,
    site: str = "",
) -> list[int]:
    if site:
        return [
            i
            for i, record in enumerate(dataset.records)
            if str(record["SITE_ID"]) == site
        ]
    if split_name == "all" or split_json is None or not split_json.exists():
        return list(range(len(dataset)))
    split = json.loads(split_json.read_text(encoding="utf-8"))
    key = "val_idx" if split_name == "val" else "train_idx"
    return [int(i) for i in split[key]]


def _target_indices(logits: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "asd":
        return torch.full((logits.size(0),), ASD_CLASS, device=logits.device, dtype=torch.long)
    return logits.argmax(dim=1)


def integrated_gradients(
    model: SimpleGCN,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    target: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IG on node features and adjacency, with a zero baseline."""
    baseline_x = torch.zeros_like(node_features)
    baseline_a = torch.zeros_like(adjacency)
    acc_x = torch.zeros_like(node_features)
    acc_a = torch.zeros_like(adjacency)

    for step in range(1, steps + 1):
        alpha = step / steps
        x = (baseline_x + alpha * (node_features - baseline_x)).detach().requires_grad_(True)
        adj = (baseline_a + alpha * (adjacency - baseline_a)).detach().requires_grad_(True)
        logits = model(x, adj)
        chosen = logits.gather(1, target.view(-1, 1)).sum()
        grad_x, grad_a = torch.autograd.grad(chosen, (x, adj), retain_graph=False)
        acc_x = acc_x + grad_x
        acc_a = acc_a + grad_a

    ig_x = (node_features - baseline_x) * (acc_x / steps)
    ig_a = (adjacency - baseline_a) * (acc_a / steps)
    return ig_x.detach(), ig_a.detach()


def attribute_loader(
    model: SimpleGCN,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
    ig_steps: int,
) -> dict[str, np.ndarray | list]:
    model.eval()
    node_sum: np.ndarray | None = None
    edge_sum: np.ndarray | None = None
    n_graphs = 0
    file_ids: list[str] = []
    preds: list[int] = []
    labels: list[int] = []

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            logits = model(node_features, adjacency)
            pred = logits.argmax(dim=1).cpu().numpy()
        target = _target_indices(logits, y, target_mode)

        ig_x, ig_a = integrated_gradients(
            model, node_features, adjacency, target, steps=ig_steps
        )
        # Node score: signed contribution of that ROI's connectivity profile.
        node_batch = ig_x.sum(dim=-1).cpu().numpy()
        edge_batch = ig_a.cpu().numpy()
        # Symmetrize: the GCN uses an undirected |FC| adjacency.
        edge_batch = 0.5 * (edge_batch + np.transpose(edge_batch, (0, 2, 1)))
        for i in range(edge_batch.shape[0]):
            np.fill_diagonal(edge_batch[i], 0.0)

        if node_sum is None:
            node_sum = node_batch.sum(axis=0)
            edge_sum = edge_batch.sum(axis=0)
        else:
            node_sum += node_batch.sum(axis=0)
            edge_sum += edge_batch.sum(axis=0)
        n_graphs += node_batch.shape[0]
        file_ids.extend(batch["file_id"])
        preds.extend(int(v) for v in pred)
        labels.extend(y.cpu().tolist())

    assert node_sum is not None and edge_sum is not None
    return {
        "node_mean": node_sum / n_graphs,
        "edge_mean": edge_sum / n_graphs,
        "n_graphs": n_graphs,
        "file_ids": file_ids,
        "preds": np.asarray(preds),
        "labels": np.asarray(labels),
    }


def write_node_table(
    path: Path, table: list[dict], node_scores: np.ndarray
) -> None:
    rows = []
    for row, score in zip(table, node_scores, strict=True):
        rows.append({**row, "attribution": float(score), "abs_attribution": float(abs(score))})
    rows.sort(key=lambda item: -item["abs_attribution"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "roi_id",
                "abbr",
                "name",
                "short",
                "hemi",
                "network",
                "network_abbr",
                "attribution",
                "abs_attribution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_top_edges(
    path: Path, table: list[dict], edge_scores: np.ndarray, top_k: int
) -> None:
    n = edge_scores.shape[0]
    abs_u = np.abs(np.triu(edge_scores, k=1))
    n_off = int((abs_u > 0).sum()) or n * (n - 1) // 2
    keep = min(top_k, n_off)
    idx = np.argpartition(abs_u.ravel(), -keep)[-keep:]
    pairs = []
    for flat in idx:
        i, j = divmod(int(flat), n)
        pairs.append((i, j, float(edge_scores[i, j])))
    pairs.sort(key=lambda item: -abs(item[2]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["src_index", "src", "dst_index", "dst", "attribution", "abs_attribution"]
        )
        for i, j, weight in pairs:
            writer.writerow(
                [
                    i,
                    table[i]["short"],
                    j,
                    table[j]["short"],
                    f"{weight:.6f}",
                    f"{abs(weight):.6f}",
                ]
            )


def write_attribution_outputs(
    out_dir: Path,
    table: list[dict],
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    title: str,
    top_rois: int,
    top_edges: int,
    extra_summary: dict,
    xlabel: str = "Attribution (ASD direction)",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "node_attribution.npy", node_scores)
    np.save(out_dir / "edge_attribution.npy", edge_scores)
    write_node_table(out_dir / "node_attribution.csv", table, node_scores)
    write_top_edges(out_dir / "top_edges.csv", table, edge_scores, top_edges)
    plot_top_rois(
        node_scores, table, out_dir / "top_rois.png", top_k=top_rois, title=title, xlabel=xlabel
    )
    plot_chord(edge_scores, table, out_dir / "chord.png", top_k=top_edges, title=title)

    ranked = np.argsort(-np.abs(node_scores))
    summary = {
        "output_dir": str(out_dir),
        "model": "SimpleGCN",
        "method": "integrated gradients on node features and |FC| adjacency",
        "explained_at": datetime.now(timezone.utc).isoformat(),
        "top_rois": [
            {
                "name": table[int(i)]["name"],
                "network": table[int(i)]["network"],
                "attribution": round(float(node_scores[int(i)]), 6),
            }
            for i in ranked[:top_rois]
        ],
        "figures": {"top_rois": "top_rois.png", "chord": "chord.png"},
        **extra_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def explain_checkpoint(
    checkpoint: Path | str,
    output_dir: Path | str | None = None,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    split: str = "val",
    site: str = "",
    target: str = "asd",
    ig_steps: int = 24,
    batch_size: int = 8,
    top_rois: int = 20,
    top_edges: int = 80,
    cpu: bool = False,
) -> dict:
    """Attribute one checkpoint and write tables + figures into `output_dir`.

    Each experiment branch should call this with *its* `final_model.pt` and *its*
    figure directory, e.g. `experiments/<name>/figures`.
    """
    checkpoint_path = resolve_checkpoint(Path(checkpoint))
    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(force_cpu=cpu)
    try:
        model = load_model(checkpoint_path, device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match SimpleGCN on this branch. "
            "Run explain from the same branch that trained the model."
        ) from exc

    dataset = AbideFCDataset(data_dir)
    split_json = checkpoint_path.parent / "split.json"
    indices = select_indices(
        dataset,
        split,
        split_json if split_json.exists() else None,
        site=site,
    )
    if not indices:
        raise SystemExit(f"No subjects matched site={site!r} split={split}")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    result = attribute_loader(model, loader, device, target, ig_steps)
    table = roi_table(data_dir)
    node_scores = np.asarray(result["node_mean"], dtype=np.float64)
    edge_scores = np.asarray(result["edge_mean"], dtype=np.float64)
    return write_attribution_outputs(
        out_dir,
        table,
        node_scores,
        edge_scores,
        title=f"GNN-attributed connections ({checkpoint_path.parent.name})",
        top_rois=top_rois,
        top_edges=top_edges,
        xlabel="Integrated-gradients attribution (ASD logit)",
        extra_summary={
            "checkpoint": str(checkpoint_path),
            "target": target,
            "site": site or None,
            "split": site or (split if split_json.exists() else "all"),
            "n_subjects": int(result["n_graphs"]),
            "ig_steps": ig_steps,
        },
    )


def explain_loso_folds(
    folds_dir: Path | str,
    output_dir: Path | str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    target: str = "asd",
    ig_steps: int = 24,
    batch_size: int = 8,
    top_rois: int = 20,
    top_edges: int = 80,
    cpu: bool = False,
) -> dict:
    """Attribute each LOSO fold on its held-out site; write the subject-weighted mean."""
    folds_path = Path(folds_dir)
    out_dir = Path(output_dir)
    dataset = AbideFCDataset(data_dir)
    table = roi_table(data_dir)
    device = pick_device(force_cpu=cpu)

    node_sum = None
    edge_sum = None
    n_total = 0
    fold_summaries = []

    sites = sorted(p.name for p in folds_path.iterdir() if p.is_dir())
    for site in sites:
        ckpt = folds_path / site / "final_model.pt"
        if not ckpt.exists():
            print(f"skip {site}: no final_model.pt", flush=True)
            continue
        indices = select_indices(dataset, "all", None, site=site)
        if not indices:
            print(f"skip {site}: no subjects", flush=True)
            continue
        model = load_model(ckpt, device)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
        result = attribute_loader(model, loader, device, target, ig_steps)
        node = np.asarray(result["node_mean"], dtype=np.float64)
        edge = np.asarray(result["edge_mean"], dtype=np.float64)
        n = int(result["n_graphs"])
        node_sum = node * n if node_sum is None else node_sum + node * n
        edge_sum = edge * n if edge_sum is None else edge_sum + edge * n
        n_total += n
        fold_summaries.append({"site_id": site, "n_subjects": n})
        print(f"  {site:<10} n={n}  top={table[int(np.argmax(np.abs(node)))]['abbr']}", flush=True)

    if n_total == 0 or node_sum is None or edge_sum is None:
        raise SystemExit(f"No LOSO folds explained under {folds_path}")

    return write_attribution_outputs(
        out_dir,
        table,
        node_sum / n_total,
        edge_sum / n_total,
        title="GNN-attributed connections (LOSO-CV, site-weighted mean)",
        top_rois=top_rois,
        top_edges=top_edges,
        xlabel="Integrated-gradients attribution (ASD logit)",
        extra_summary={
            "target": target,
            "split": "held-out site per fold, then subject-weighted mean",
            "n_subjects": n_total,
            "n_folds": len(fold_summaries),
            "ig_steps": ig_steps,
            "folds": fold_summaries,
        },
    )


def main() -> None:
    args = parse_args()
    kind = resolve_kind(args.kind, args.checkpoint, args.loso_folds_dir)
    if kind == "linear":
        from neuroasd.linear_explain import explain_linear

        summary = explain_linear(
            output_dir=args.output_dir,
            group_by=args.group_by,
            top_rois=args.top_rois,
            top_edges=args.top_edges,
            workers=args.workers,
        )
        print(f"Wrote figures to {summary['output_dir']} ({kind})")
        for atlas, paths in summary.get("figures", {}).items():
            print(f"  {atlas}: {summary['output_dir']}/{paths['chord']}")
        return
    if args.loso_folds_dir is not None:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --loso-folds-dir")
        summary = explain_loso_folds(
            folds_dir=args.loso_folds_dir,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            target=args.target,
            ig_steps=args.ig_steps,
            batch_size=args.batch_size,
            top_rois=args.top_rois,
            top_edges=args.top_edges,
            cpu=args.cpu,
        )
    else:
        summary = explain_checkpoint(
            checkpoint=args.checkpoint or Path("outputs/gcn_baseline"),
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            split=args.split,
            site=args.site,
            target=args.target,
            ig_steps=args.ig_steps,
            batch_size=args.batch_size,
            top_rois=args.top_rois,
            top_edges=args.top_edges,
            cpu=args.cpu,
        )
    print(f"Wrote figures to {summary['output_dir']}")
    print(f"Chord : {summary['output_dir']}/chord.png")
    print(f"ROIs  : {summary['output_dir']}/top_rois.png")
    print("Top 8 ROIs:")
    for row in summary["top_rois"][:8]:
        print(f"  {row['attribution']:+.4f}  {row['name']}  ({row['network']})")


if __name__ == "__main__":
    main()
