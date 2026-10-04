"""Train a simple GCN baseline on ABIDE FC graphs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from neuroasd.fc_dataset import AbideFCDataset, collate_graphs
from neuroasd.gcn import (
    SITE_ALIGN_LAMBDA,
    SITE_ALIGN_MIN_SUBJECTS,
    SimpleGCN,
    site_coral_loss,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "abide"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "gcn_baseline"


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/GPU exists")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N epochs (all epochs still saved to train.log)",
    )
    return parser.parse_args()


class TrainLogger:
    """Write training messages to stdout and train.log."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = log_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@torch.no_grad()
def evaluate(model: SimpleGCN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["label"].to(device)

        logits = model(node_features, adjacency)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds)),
        "auc": float(roc_auc_score(all_labels, all_probs)),
    }


class _GradReverse(torch.autograd.Function):
    """Gradient-reversal layer: identity forward, -lam-scaled gradient backward."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = float(lam)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(inputs: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(inputs, lam)


def site_adversarial_loss(
    adv_head: nn.Module | None,
    embedding: torch.Tensor,
    site_ids: list[str],
    site_to_index: dict[str, int] | None,
    lam: float,
    stats: dict[str, int] | None = None,
) -> torch.Tensor:
    """Training-only site-adversarial (DANN-style) penalty on the graph embedding.

    `adv_head` classifies the gradient-reversed mean-pooled embedding into the
    TRAINING fold's SITE_IDs only, so minimising the diagnostic loss pushes the
    embedding towards a site-invariant representation while the reversed
    gradient (~ -lam) is what reaches the encoder. Subjects whose site is not a
    training-fold class are dropped, and when fewer than two distinct
    training-fold sites are present in the batch the penalty is exactly zero (a
    one-class CrossEntropy carries no signal).

    `stats` is log-only bookkeeping; every key it touches is initialised here and
    the site-count loop over an empty/absent mapping is a no-op, so the
    instrumentation path cannot raise.
    """
    if stats is not None:
        stats.setdefault("adv_batches_seen", 0)
        stats.setdefault("adv_batches", 0)
        stats.setdefault("adv_sites_sum", 0)
        stats.setdefault("adv_subjects_sum", 0)
        stats.setdefault("sub_quota_sites", 0)

    if adv_head is None or not site_to_index or lam <= 0.0:
        return embedding.new_zeros(())

    counts: dict[str, int] = {}
    positions: list[int] = []
    targets: list[int] = []
    for position, site in enumerate(site_ids):
        class_index = site_to_index.get(site)
        if class_index is None:
            continue
        counts[site] = counts.get(site, 0) + 1
        positions.append(position)
        targets.append(class_index)

    if stats is not None:
        stats["adv_batches_seen"] += 1
        stats["adv_sites_sum"] += len(counts)
        stats["adv_subjects_sum"] += len(positions)
        stats["sub_quota_sites"] += sum(
            1 for count in counts.values() if count < SITE_ALIGN_MIN_SUBJECTS
        )

    if len(counts) < 2:
        return embedding.new_zeros(())

    index = torch.tensor(positions, dtype=torch.long, device=embedding.device)
    target = torch.tensor(targets, dtype=torch.long, device=embedding.device)
    logits = adv_head(grad_reverse(embedding, lam))[index]
    if stats is not None:
        stats["adv_batches"] += 1
    return F.cross_entropy(logits, target)


# Worst-site (group-DRO) reweighting of the CLASSIFICATION loss. Sites with fewer
# than SITE_DRO_MIN_SUBJECTS subjects in a batch contribute no risk estimate, the
# per-site risk is a detached EMA that is created inside train_one_epoch (so it
# resets every epoch and can never cross folds), and the softmax temperature is
# one fixed, non-tuned constant (uniform weights whenever no risk is known yet).
SITE_DRO_DECAY = 0.9
SITE_DRO_TEMPERATURE = 1.0
SITE_DRO_MIN_SUBJECTS = 2

# Train-fold-only site-stratified MIXUP on the node-feature batch. ONE fixed,
# deliberately untuned alpha; opt-in via train_one_epoch's mixup_alpha kwarg, so
# the default (0.0) path keeps the current op order bit-identical.
MIXUP_ALPHA = 0.2
MIXUP_SAME_SITE_ONLY = True

# Train-fold-only DropEdge-style adjacency perturbation: ONE fixed, deliberately
# untuned drop probability, opt-in through train_one_epoch's edge_drop_p kwarg so
# the default (0.0) path never touches the adjacency and stays bit-identical.
EDGE_DROP_P = 0.1


def edge_drop_mask(adjacency: torch.Tensor, p: float) -> torch.Tensor:
    """Bernoulli(1 - p) off-diagonal edge mask, symmetrized, self-loops preserved.

    One independent draw per (subject, edge) entry; the mask is symmetrized
    (mask = mask * mask.T, the pre-registered construction) so the perturbed graph
    stays undirected, and its diagonal is forced to 1 so self-loops are never
    dropped (the model's normalize_adjacency then renormalizes the surviving edge
    weights exactly as it does for an unperturbed graph). NB the product form makes
    an off-diagonal edge survive only if BOTH (i, j) and (j, i) draws survive, so
    the realised off-diagonal drop rate is ~2p - p^2 (~0.19 at p = 0.1); that
    realised fraction is what the log-only edge_drop_frac_mean counter reports.
    """
    size = adjacency.shape[-1]
    mask = (torch.rand_like(adjacency) >= p).to(adjacency.dtype)
    mask = mask * mask.transpose(-1, -2)
    eye = torch.eye(size, dtype=adjacency.dtype, device=adjacency.device)
    return mask * (1.0 - eye) + eye


def site_stratified_permutation(
    site_ids: list[str], size: int, device: torch.device
) -> torch.Tensor:
    """Permutation whose mixing partner always comes from the SAME batch site.

    Partners are drawn within each SITE_ID group, so mixup never interpolates
    across sites. A site with fewer than 2 subjects in the batch keeps the
    identity mapping for those rows (no cross-site mixing and no NaN).
    """
    perm = torch.arange(size, dtype=torch.long, device=device)
    groups: dict[str, list[int]] = {}
    for position in range(min(size, len(site_ids))):
        groups.setdefault(str(site_ids[position]), []).append(position)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        order = torch.randperm(len(positions)).tolist()
        index = torch.tensor(positions, dtype=torch.long, device=device)
        perm[index] = torch.tensor(
            [positions[i] for i in order], dtype=torch.long, device=device
        )
    return perm


def site_dro_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    site_ids: list[str] | None,
    risk: dict[str, float],
    criterion: nn.Module,
    stats: dict[str, float] | None = None,
    mixup_lam: float = 0.0,
    mixup_perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train-fold-only worst-site (group-DRO) reweighting of the classification CE.

    Each SITE_ID present in the batch with at least SITE_DRO_MIN_SUBJECTS subjects
    gets its own per-site mean CE; the sites are then weighted by
    w_s = softmax(r_s / T), with ONE fixed non-tuned temperature T = 1.0 and r_s a
    DETACHED per-site risk EMA passed in by the caller. The weights are rescaled to
    mean 1 and applied per sample, so when no risk is known yet (the first batch of
    the epoch) every weight is exactly 1.0 and the loss is the plain sample mean CE
    - the first optimizer step stays baseline-identical. Subjects of a site skipped
    for having too few members keep weight 1.0, so no sample leaves the loss. The
    risks themselves are plain floats and contribute no gradient of their own.
    """
    if stats is not None:
        for key in ("dro_batches_seen", "dro_batches", "dro_sites_sum", "dro_w_spread_sum"):
            stats.setdefault(key, 0.0)
        stats.setdefault("dro_w_min", 1.0)
        stats.setdefault("dro_w_max", 1.0)
        stats["dro_batches_seen"] += 1

    per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
    if mixup_lam > 0.0 and mixup_perm is not None:
        # Soft-target CE of the mixed batch: both mixture components stay
        # attached to the ANCHOR (unpermuted) row, so the per-site risk r_s and
        # the site-availability accounting below are exactly the unmixed ones.
        per_sample_ce = mixup_lam * per_sample_ce + (1.0 - mixup_lam) * F.cross_entropy(
            logits, labels[mixup_perm], reduction="none"
        )
    counts: dict[str, int] = {}
    for site in site_ids or ():
        counts[site] = counts.get(site, 0) + 1
    admitted = [site for site in counts if counts[site] >= SITE_DRO_MIN_SUBJECTS]
    if len(admitted) < 2:
        # A single-site batch carries no worst-site contrast: keep the plain mean
        # and leave the risk table untouched (with mixup, the plain mean of the
        # same soft per-sample terms).
        return criterion(logits, labels) if mixup_lam <= 0.0 else per_sample_ce.mean()

    scores = torch.tensor(
        [risk.get(site, 0.0) for site in admitted],
        dtype=per_sample_ce.dtype,
        device=per_sample_ce.device,
    )
    weights = torch.softmax(scores / SITE_DRO_TEMPERATURE, dim=0) * len(admitted)

    sample_weight = torch.ones_like(per_sample_ce)
    for position, site in enumerate(admitted):
        mask = torch.tensor(
            [value == site for value in site_ids], device=per_sample_ce.device
        )
        sample_weight[mask] = weights[position]
        observed = float(per_sample_ce[mask].mean().detach())
        previous = risk.get(site)
        risk[site] = (
            observed
            if previous is None
            else SITE_DRO_DECAY * previous + (1.0 - SITE_DRO_DECAY) * observed
        )

    if stats is not None:
        stats["dro_batches"] += 1
        stats["dro_sites_sum"] += len(admitted)
        spread = float(weights.max() - weights.min())
        stats["dro_w_spread_sum"] += spread
        stats["dro_w_min"] = min(stats["dro_w_min"], float(weights.min()))
        stats["dro_w_max"] = max(stats["dro_w_max"], float(weights.max()))

    return (sample_weight * per_sample_ce).sum() / sample_weight.sum()


def train_one_epoch(
    model: SimpleGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    site_align_lambda: float = SITE_ALIGN_LAMBDA,
    adv_head: nn.Module | None = None,
    adv_lambda: float = 0.0,
    adv_site_to_index: dict[str, int] | None = None,
    adv_stats: dict[str, int] | None = None,
    dro_stats: dict[str, float] | None = None,
    mixup_alpha: float = 0.0,
    mixup_stats: dict[str, float] | None = None,
    edge_drop_p: float = 0.0,
    edge_drop_stats: dict[str, float] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    # Per-site risk table for the worst-site reweighting. Created HERE, so it is
    # rebuilt at the start of every epoch and can never carry state across folds
    # or contribute a gradient.
    site_risk: dict[str, float] = {}
    if adv_stats is not None:
        for key in (
            "adv_batches_seen",
            "adv_batches",
            "adv_sites_sum",
            "adv_subjects_sum",
            "sub_quota_sites",
        ):
            adv_stats.setdefault(key, 0)

    for batch in loader:
        node_features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["label"].to(device)

        # Train-fold-only site-stratified MIXUP, applied to the node-FEATURE
        # batch only (never the adjacency/edge weights, the pooled embedding or
        # the EMA shadow copy). One lam ~ Beta(alpha, alpha) per batch; the
        # partner comes from the same batch site, and rows of a site with fewer
        # than 2 subjects in the batch keep the identity. Only training-fold
        # subjects ever pass through here.
        mixup_lam = 0.0
        mixup_perm: torch.Tensor | None = None
        if mixup_alpha > 0.0 and node_features.size(0) > 1:
            mixup_lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            mixup_perm = site_stratified_permutation(
                batch["site_id"], node_features.size(0), node_features.device
            )
            node_features = mixup_lam * node_features + (
                1.0 - mixup_lam
            ) * node_features[mixup_perm]
            if mixup_stats is not None:
                mixup_stats.setdefault("mixup_batches", 0.0)
                mixup_stats.setdefault("mixup_lam_sum", 0.0)
                mixup_stats.setdefault("mixup_paired_frac_sum", 0.0)
                mixup_stats["mixup_batches"] += 1.0
                mixup_stats["mixup_lam_sum"] += mixup_lam
                paired = int(
                    (
                        mixup_perm
                        != torch.arange(mixup_perm.numel(), device=mixup_perm.device)
                    ).sum()
                )
                mixup_stats["mixup_paired_frac_sum"] += paired / mixup_perm.numel()

        optimizer.zero_grad()
        # Train-fold-only DropEdge-style adjacency perturbation, applied to this
        # training batch's edge-weight tensor immediately before the forward (the
        # model normalizes the adjacency inside embed()). Only TRAINING batches
        # ever reach this branch: the held-out loader runs with the default 0.0,
        # and evaluate() plus the single final-epoch EMA read are untouched. The
        # EMA shadow copy holds weights, not edges, so it cannot be perturbed.
        if edge_drop_p > 0.0:
            mask = edge_drop_mask(adjacency, edge_drop_p)
            adjacency = adjacency * mask
            if edge_drop_stats is not None:
                edge_drop_stats.setdefault("edge_drop_batches", 0.0)
                edge_drop_stats.setdefault("edge_drop_dropped_sum", 0.0)
                edge_drop_stats["edge_drop_batches"] += 1.0
                size = mask.shape[-1]
                off_total = float(mask.shape[0] * size * (size - 1))
                off_kept = float(mask.sum()) - float(mask.shape[0] * size)
                edge_drop_stats["edge_drop_dropped_sum"] += (
                    1.0 - off_kept / off_total if off_total > 0.0 else 0.0
                )
        embedding = model.embed(node_features, adjacency)
        logits = model.classifier(embedding)
        # Classification loss reweighted towards the batch's worst training-fold
        # sites (uniform weights until a site risk is known, so epoch 0's first
        # step is the plain sample mean CE). Held-out subjects never appear in the
        # training loader, so the held-out site enters neither the risks nor the
        # weights.
        loss = site_dro_classification_loss(
            logits,
            labels,
            batch["site_id"],
            site_risk,
            criterion,
            dro_stats,
            mixup_lam=mixup_lam,
            mixup_perm=mixup_perm,
        )
        if site_align_lambda > 0.0:
            # Second-order alignment of the batch's per-site embeddings. The
            # loader only ever contains training-fold subjects, so the held-out
            # site contributes no statistics here.
            loss = loss + site_align_lambda * site_coral_loss(embedding, batch["site_id"])
        if adv_head is not None and adv_site_to_index and adv_lambda > 0.0:
            # Training-only site-adversarial term on the gradient-reversed
            # embedding. The adversarial head is a separate branch, so the
            # reversed gradient reaches the encoder but never the classifier's
            # own weights; at adv_lambda == 0 this branch is skipped entirely and
            # the step is baseline-identical.
            loss = loss + site_adversarial_loss(
                adv_head,
                embedding,
                batch["site_id"],
                adv_site_to_index,
                adv_lambda,
                adv_stats,
            )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(force_cpu=args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.output_dir / "train.log")

    logger.log(f"Started at {datetime.now(timezone.utc).isoformat()}")
    logger.log(f"Output dir: {args.output_dir}")
    logger.log(f"Args: {vars(args)}")

    dataset = AbideFCDataset(args.data_dir)
    labels = np.array([int(dataset[i]["label"]) for i in range(len(dataset))])

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    train_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    num_nodes = int(dataset[0]["node_features"].shape[0])
    model = SimpleGCN(
        in_features=num_nodes,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.log(f"Device: {device}")
    logger.log(f"Subjects: {len(dataset)} (train={len(train_idx)}, val={len(val_idx)})")
    logger.log(f"Nodes per graph: {num_nodes}")
    logger.log("")

    best_auc = -1.0
    history: list[dict[str, float | int]] = []
    final_metrics: dict[str, float] = {}

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)
            final_metrics = val_metrics

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                }
            )

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "num_nodes": num_nodes,
                        "best_auc": best_auc,
                    },
                    args.output_dir / "best_model.pt",
                )

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                logger.log(
                    f"Epoch {epoch:03d} | "
                    f"loss={train_loss:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.3f} | "
                    f"val_auc={val_metrics['auc']:.3f} | "
                    f"val_f1={val_metrics['f1']:.3f}"
                )
    finally:
        # The reported checkpoint is the last epoch, not the best one: picking the best
        # epoch by val AUC would select a model using the very set it is scored on.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "num_nodes": num_nodes,
                "final_metrics": final_metrics,
            },
            args.output_dir / "final_model.pt",
        )

        summary = {
            "device": str(device),
            "num_subjects": len(dataset),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "epochs_run": len(history),
            "model_selection": "final epoch (no selection on the evaluation set)",
            "reported_metrics": final_metrics,
            "diagnostic_best_epoch_auc": best_auc,
            "checkpoint": str(args.output_dir / "final_model.pt"),
            "diagnostic_checkpoint": str(args.output_dir / "best_model.pt"),
            "log_file": str(args.output_dir / "train.log"),
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "split.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        logger.log("")
        if final_metrics:
            logger.log(
                f"Final-epoch val AUC: {final_metrics['auc']:.3f} "
                f"(reported) | best-epoch AUC: {best_auc:.3f} (diagnostic only)"
            )
        logger.log(f"Saved checkpoint: {args.output_dir / 'final_model.pt'}")
        logger.log(f"Saved metrics:  {args.output_dir / 'metrics.json'}")
        logger.log(f"Saved log:      {args.output_dir / 'train.log'}")
        logger.close()


if __name__ == "__main__":
    main()
