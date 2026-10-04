"""Run one autoresearch trial and score it against the promotion gates.

A trial trains the current branch's model under a given hyperparameter config and
reports metrics for one of three evaluation stages (see `autoresearch/program.md`):

    screen       fast multi-seed random split, used to reject bad ideas
    loso-subset  leave-one-site-out on the largest sites, mid-cost confirmation
    loso-full    leave-one-site-out on all 20 sites, publication-grade

Gating uses **final-epoch** metrics, never best-epoch, so that the number of epochs
stays an honest hyperparameter and no model selection happens on the test fold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from neuroasd.fc_dataset import AbideFCDataset, collate_graphs
from neuroasd.gcn import FISHER_CLIP, SimpleGCN, quantile_normalize_edges
from neuroasd.train import evaluate, pick_device, train_one_epoch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "abide"
TRIALS_DIR = REPO_ROOT / "outputs" / "autoresearch" / "trials"
LEDGER_PATH = REPO_ROOT / "outputs" / "autoresearch" / "ledger.jsonl"
GATES_PATH = Path(__file__).resolve().parent / "gates.json"

# Sites with the largest held-out sets; the cheap LOSO stage uses only these.
SUBSET_SITES = ("NYU", "UM_1", "USM", "UCLA_1", "YALE")

STAGES = ("screen", "loso-subset", "loso-full")

# Grid size of the per-site edge-weight quantile map (train-fold statistics only).
EDGE_QUANTILE_GRID = 129

# Decay of the long-window weight EMA, applied ONCE PER EPOCH. Per-epoch 0.99 gives
# an effective window of ~1/(1-0.99) = 100 epochs, i.e. the whole cosine anneal
# (a per-step 0.99 would only span ~4 epochs and just repeat iter18's null 10-epoch
# SWA). The EMA never influences the live model, so step 0 is bit-identical to the
# iter25 winner and the averaged weights are read exactly once, at the final epoch.
EMA_DECAY = 0.99

# Max weight of the training-only site-adversarial (gradient-reversal) penalty.
# Lambda ramps linearly 0 -> ADV_LAMBDA_MAX across epochs, so epoch 0 is
# baseline-identical and the last epoch carries the full adversarial pressure.
ADV_LAMBDA_MAX = 0.1

PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 3


@dataclass(frozen=True)
class TrialConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    dropout: float
    weight_decay: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Short trial identifier")
    parser.add_argument("--stage", choices=STAGES, default="screen")
    parser.add_argument("--note", default="", help="One-line hypothesis being tested")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 2026],
        help="Seeds averaged in the screen stage (LOSO stages use the first seed)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-fold logs")
    return parser.parse_args()


def git_revision() -> dict[str, str]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return "unknown"

    return {
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("git", "rev-parse", "--short", "HEAD"),
    }


def build_loader(
    dataset: AbideFCDataset,
    indices,
    batch_size: int,
    shuffle: bool,
    collate_fn=collate_graphs,
):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def subject_edge_values(dataset: AbideFCDataset) -> tuple[np.ndarray, list[str]]:
    """Fisher-z |FC| upper-triangle edge weights per subject, plus its SITE_ID.

    Only raw per-subject edge values are cached here (no fold statistics), so the
    cache is always safe to reuse across folds.
    """
    cached = getattr(dataset, "_edge_value_cache", None)
    if cached is not None:
        return cached

    upper = np.triu_indices(int(dataset[0]["adjacency"].shape[0]), k=1)
    values = []
    for index in range(len(dataset.records)):
        magnitude = np.clip(
            np.abs(dataset[index]["adjacency"].numpy()).astype(np.float32), 0.0, FISHER_CLIP
        )
        values.append(np.arctanh(magnitude)[upper].astype(np.float32))
    cached = (np.stack(values), [str(record["SITE_ID"]) for record in dataset.records])
    dataset._edge_value_cache = cached
    return cached


def fit_edge_quantile_tables(
    dataset: AbideFCDataset, train_indices
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Per-site edge-weight quantile maps fit on TRAINING-fold edges only.

    For each site present in the training folds, the site's |FC| edge weights are
    rank/quantile mapped onto the pooled training-fold Fisher-z quantiles, which
    removes site-specific edge scale/outliers while preserving within-site edge
    ordering (the map is monotone). No held-out subject, site statistic or label
    is touched. Sites unseen in training fall back to the pooled mapping.
    """
    edges, sites = subject_edge_values(dataset)
    train_indices = list(train_indices)
    grid = np.linspace(0.0, 1.0, EDGE_QUANTILE_GRID)
    pooled_q = torch.from_numpy(
        np.quantile(
            np.concatenate([edges[index] for index in train_indices]), grid
        ).astype(np.float32)
    )

    indices_by_site: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        indices_by_site[sites[index]].append(index)

    tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for site, site_indices in indices_by_site.items():
        site_q = np.maximum.accumulate(
            np.quantile(np.concatenate([edges[index] for index in site_indices]), grid)
        )
        tables[site] = (torch.from_numpy(site_q.astype(np.float32)), pooled_q)
    return tables, pooled_q


def build_collate(
    tables: dict[str, tuple[torch.Tensor, torch.Tensor]], pooled_table
):
    """Collate that rank/quantile normalizes each batch's |FC| edge weights per site."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        collated = collate_graphs(batch)
        collated["adjacency"] = quantile_normalize_edges(
            collated["adjacency"], collated["site_id"], tables, pooled_table
        )
        return collated

    return collate


def run_fold(
    dataset: AbideFCDataset,
    train_indices,
    test_indices,
    config: TrialConfig,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Train once and return final-epoch metrics (plus best-epoch for reference)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Per-site |FC| edge-weight quantile maps, fit on this fold's training edges
    # only. Sites unseen in training (e.g. the held-out site in a LOSO fold) fall
    # back to the pooled training-fold mapping, so no held-out statistics are used.
    tables, pooled_q = fit_edge_quantile_tables(dataset, train_indices)
    collate = build_collate(tables, (pooled_q, pooled_q))

    train_loader = build_loader(
        dataset, train_indices, config.batch_size, True, collate
    )
    test_loader = build_loader(
        dataset, test_indices, config.batch_size, False, collate
    )

    num_nodes = int(dataset[0]["node_features"].shape[0])
    model = SimpleGCN(
        in_features=num_nodes,
        hidden_dim=config.hidden_dim,
        num_classes=2,
        dropout=config.dropout,
    ).to(device)

    # Training-only site-adversarial head over the mean-pooled embedding. Its
    # classes are THIS fold's training sites (sorted), so the held-out site is
    # never a class. The head is initialised under a restored RNG state so the
    # baseline's batch-shuffle stream is untouched, and its parameters join the
    # SAME Adam optimizer (Adam is per-parameter, so the extra params change
    # nothing for the encoder at lambda = 0).
    adv_site_to_index = {site: index for index, site in enumerate(sorted(tables))}
    rng_state = torch.get_rng_state()
    adv_head = (
        nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(adv_site_to_index)),
        ).to(device)
        if adv_site_to_index
        else None
    )
    torch.set_rng_state(rng_state)
    adv_stats: dict[str, int] = {
        "adv_batches_seen": 0,
        "adv_batches": 0,
        "adv_sites_sum": 0,
        "adv_subjects_sum": 0,
        "sub_quota_sites": 0,
    }
    # Log-only bookkeeping for the worst-site (group-DRO) classification
    # reweighting; never read for gating or selection.
    dro_stats: dict[str, float] = {}
    parameters = list(model.parameters()) + (
        list(adv_head.parameters()) if adv_head is not None else []
    )
    optimizer = torch.optim.Adam(
        parameters, lr=config.lr, weight_decay=config.weight_decay
    )
    # Cosine LR annealing (LR schedule only; peak LR stays config.lr). The gated
    # metric is the FINAL-EPOCH auc_mean, and the ledger shows scores peak
    # mid-training then decay under a constant LR for all 100 epochs (~0.075
    # final-vs-best gap); annealing toward zero targets that largest measured
    # untapped quantity without epoch selection, early stopping, or held-out data.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # Shadow EMA of the weights (long window, decay EMA_DECAY per epoch). It is a
    # detached copy only; the live model trains exactly as the iter25 winner does.
    ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    best_auc = -1.0
    metrics: dict[str, float] = {}
    for step in range(config.epochs):
        # Linear 0 -> ADV_LAMBDA_MAX ramp over epochs: exactly 0 at epoch 0 (so
        # the first epoch and its first optimizer step are baseline-identical)
        # and ADV_LAMBDA_MAX at the final epoch.
        adv_lambda = ADV_LAMBDA_MAX * step / max(1, config.epochs - 1)
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            adv_head=adv_head,
            adv_lambda=adv_lambda,
            adv_site_to_index=adv_site_to_index,
            adv_stats=adv_stats,
            dro_stats=dro_stats,
        )
        scheduler.step()  # advance the LR once per epoch, after its optimizer steps
        with torch.no_grad():
            live_state = model.state_dict()
            for key, value in ema_state.items():
                value.mul_(EMA_DECAY).add_(live_state[key], alpha=1.0 - EMA_DECAY)
        metrics = evaluate(model, test_loader, device)
        best_auc = max(best_auc, metrics["auc"])

    # Single final-epoch read of the EMA weights, reported as this stage's number:
    # no epoch selection, no early stopping, no best-epoch read.
    final_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(ema_state)
    metrics = evaluate(model, test_loader, device)
    model.load_state_dict(final_state)

    # Log-only instrumentation (fold totals; never touches the reported
    # final-epoch metric): whether the adversarial branch ever fired, the
    # average number of sites a training batch carried, and how often a
    # participating site contributed fewer than SITE_ALIGN_MIN_SUBJECTS subjects.
    batches_seen = int(adv_stats["adv_batches_seen"])
    applied = int(adv_stats["adv_batches"])
    sites_total = int(adv_stats["adv_sites_sum"])
    sub_quota_total = int(adv_stats["sub_quota_sites"])
    adv_sites_mean = sites_total / batches_seen if batches_seen else 0.0
    adv_batch_fraction = applied / batches_seen if batches_seen else 0.0
    print(
        f"  [adv] batches={batches_seen} applied={applied} "
        f"frac={adv_batch_fraction:.3f} mean_sites={adv_sites_mean:.2f} "
        f"sub_quota_sites={sub_quota_total}",
        flush=True,
    )

    # Log-only worst-site reweighting diagnostics: how many sites entered the
    # reweighted CE term per batch, and the spread of w_s = softmax(r_s / 1.0).
    dro_batches = int(dro_stats.get("dro_batches", 0))
    dro_seen = int(dro_stats.get("dro_batches_seen", 0))
    dro_applied = dro_batches / dro_seen if dro_seen else 0.0
    dro_sites_mean = (
        float(dro_stats.get("dro_sites_sum", 0.0)) / dro_batches if dro_batches else 0.0
    )
    dro_w_spread_mean = (
        float(dro_stats.get("dro_w_spread_sum", 0.0)) / dro_batches
        if dro_batches
        else 0.0
    )
    print(
        f"  [dro] batches={dro_seen} "
        f"applied_frac={dro_applied:.3f} sites_entering={dro_sites_mean:.2f} "
        f"w_spread_mean={dro_w_spread_mean:.4f} "
        f"w_min={float(dro_stats.get('dro_w_min', 1.0)):.4f} "
        f"w_max={float(dro_stats.get('dro_w_max', 1.0)):.4f}",
        flush=True,
    )

    return {
        **metrics,
        "best_auc": best_auc,
        "adv_batches_seen": float(batches_seen),
        "adv_batches_applied": float(applied),
        "adv_sites_mean": float(adv_sites_mean),
        "sub_quota_sites": float(sub_quota_total),
        "dro_batches_applied": float(dro_batches),
        "dro_batch_fraction": float(dro_applied),
        "dro_sites_mean": float(dro_sites_mean),
        "dro_w_spread_mean": float(dro_w_spread_mean),
    }


def summarize(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ("accuracy", "auc", "f1", "best_auc"):
        values = [float(fold[key]) for fold in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def group_indices_by_site(dataset: AbideFCDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record_index, record in enumerate(dataset.records):
        groups[str(record["SITE_ID"])].append(record_index)
    return dict(sorted(groups.items()))


def run_screen_stage(
    dataset: AbideFCDataset,
    config: TrialConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict[str, float]]:
    labels = np.array([int(record["label"]) for record in dataset.records])
    runs: list[dict] = []

    for seed in args.seeds:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_ratio, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
        metrics = run_fold(dataset, train_idx, val_idx, config, seed, device)
        runs.append({"seed": seed, "test_size": len(val_idx), **metrics})
        if not args.quiet:
            print(
                f"  seed {seed} | acc={metrics['accuracy']:.3f} "
                f"auc={metrics['auc']:.3f} f1={metrics['f1']:.3f}",
                flush=True,
            )

    return runs, summarize(runs)


def run_loso_stage(
    dataset: AbideFCDataset,
    config: TrialConfig,
    args: argparse.Namespace,
    device: torch.device,
    sites: tuple[str, ...] | None,
) -> tuple[list[dict], dict[str, float]]:
    all_groups = group_indices_by_site(dataset)
    groups = all_groups
    if sites is not None:
        missing = [site for site in sites if site not in all_groups]
        if missing:
            raise SystemExit(f"Unknown SITE_ID(s) in subset: {', '.join(missing)}")
        groups = {site: all_groups[site] for site in sites}

    seed = args.seeds[0]
    runs: list[dict] = []

    for position, (site_id, test_indices) in enumerate(groups.items(), start=1):
        # Held-out site is always excluded from training, even when scoring a subset.
        train_indices = [
            index
            for other_site, indices in all_groups.items()
            if other_site != site_id
            for index in indices
        ]
        metrics = run_fold(dataset, train_indices, test_indices, config, seed, device)
        runs.append(
            {
                "site_id": site_id,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                **metrics,
            }
        )
        if not args.quiet:
            print(
                f"  [{position}/{len(groups)}] {site_id:<9} n={len(test_indices):<4} "
                f"acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}",
                flush=True,
            )

    return runs, summarize(runs)


def load_gates() -> dict:
    return json.loads(GATES_PATH.read_text(encoding="utf-8"))


# Reported / gated metrics must be final-epoch. Best-epoch selection on the
# evaluation fold leaked ~0.07 LOSO AUC in an earlier revision (0.707 vs 0.623).
FORBIDDEN_GATE_METRICS = frozenset({"best_auc", "best_auc_mean"})
MODEL_SELECTION = "final epoch (no selection on the evaluation set)"


def apply_gate(stage: str, summary: dict[str, float]) -> dict:
    gate = load_gates()["gates"][stage]
    metric = gate["metric"]
    if metric in FORBIDDEN_GATE_METRICS or metric.startswith("best_"):
        raise SystemExit(
            f"refusing to gate on {metric!r}: best-epoch metrics leak the "
            "evaluation set (historical LOSO bias ~0.07 AUC)"
        )
    observed = summary[metric]
    passed = observed >= gate["threshold"]
    return {
        "stage": stage,
        "metric": metric,
        "threshold": gate["threshold"],
        "observed": round(observed, 4),
        "margin": round(observed - gate["threshold"], 4),
        "passed": passed,
        "rationale": gate["rationale"],
    }


def append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def main() -> None:
    args = parse_args()
    config = TrialConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )

    device = pick_device(force_cpu=args.cpu)
    dataset = AbideFCDataset(args.data_dir)
    revision = git_revision()

    print(f"Trial   : {args.name}")
    print(f"Stage   : {args.stage}")
    print(f"Branch  : {revision['branch']} @ {revision['commit']}")
    print(f"Device  : {device}")
    print(f"Config  : {asdict(config)}")
    if args.note:
        print(f"Note    : {args.note}")
    print("")

    started = time.time()
    if args.stage == "screen":
        runs, summary = run_screen_stage(dataset, config, args, device)
    elif args.stage == "loso-subset":
        runs, summary = run_loso_stage(dataset, config, args, device, SUBSET_SITES)
    else:
        runs, summary = run_loso_stage(dataset, config, args, device, None)
    elapsed = time.time() - started

    verdict = apply_gate(args.stage, summary)

    result = {
        "name": args.name,
        "stage": args.stage,
        "note": args.note,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "device": str(device),
        "git": revision,
        "config": asdict(config),
        "seeds": args.seeds,
        "model_selection": MODEL_SELECTION,
        "summary": {key: round(value, 4) for key, value in summary.items()},
        "runs": runs,
        "verdict": verdict,
    }

    trial_dir = TRIALS_DIR / f"{args.stage}__{args.name}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    append_ledger(
        {
            key: result[key]
            for key in (
                "name",
                "stage",
                "ran_at",
                "git",
                "config",
                "model_selection",
                "summary",
                "verdict",
            )
        }
    )

    print("")
    print(f"accuracy : {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
    print(f"auc      : {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    print(f"f1       : {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
    print(f"runtime  : {elapsed / 60:.1f} min")
    print("")
    print(
        f"gate     : {verdict['metric']} {verdict['observed']:.4f} "
        f"vs threshold {verdict['threshold']:.4f} "
        f"(margin {verdict['margin']:+.4f})"
    )
    print(f"result   : {trial_dir / 'result.json'}")
    print(f"VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}")

    raise SystemExit(PASS_EXIT_CODE if verdict["passed"] else FAIL_EXIT_CODE)


if __name__ == "__main__":
    main()
