"""Turn a linear `fit_predict` model into per-atlas edge weights for chord plots.

The scored model in `neuroasd.model` concatenates tangent-space blocks (upper
triangle, including the diagonal) and optional age/sex columns. This module
refits that estimator, reads its coefficients, and reshapes each atlas block
into a symmetric matrix. Positive values are flipped to the ASD direction so
the colors match the GNN integrated-gradients figures (red = ASD).

``group_by`` decides which fit the figure comes from:

* ``all`` (default, ~1 min): one fit on every subject. Draft figure only.
  This is not the model used for the leave-one-institution-out scores.
* ``institution``: mean coefficients over the 23 institution-out training
  folds — the same outer split as the reported institution-out metrics.
  About 20 minutes.
* ``site``: same idea for the 33 site-out folds.

    uv run python -m neuroasd.explain --kind linear --group-by institution \\
        --output-dir outputs/explain_linear_institution
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuroasd.dcan_atlas import ATLAS_DIRS, atlas_table, n_tangent_features
from neuroasd.linear_baseline import GROUPINGS

REPO_ROOT = Path(__file__).resolve().parents[1]
N_COVARIATES = 2


def estimator_coef(clf) -> np.ndarray:
    """Coefficient vector of a ridge or logistic estimator (higher = control)."""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    model = getattr(clf, "model_", None)
    if model is not None and hasattr(model, "coef_"):
        return np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    raise TypeError(f"estimator {type(clf).__name__} has no coefficient vector")


def atlases_used_by_model() -> list[str]:
    """Which atlas strings `fit_predict` actually loads."""
    from neuroasd import model as scored

    source = inspect.getsource(scored.fit_predict)
    used = [name for name in scored.ATLASES if f'"{name}"' in source or f"'{name}'" in source]
    return used or ["gordon"]


def split_coef_blocks(
    coef: np.ndarray,
    atlases: list[str],
    n_parcels: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Split a concatenated coefficient vector into per-atlas tangent blocks."""
    sizes = [n_tangent_features(n_parcels[name]) for name in atlases]
    n_atlas = int(sum(sizes))
    if len(coef) == n_atlas:
        cov = np.zeros(0)
        cursor = 0
    elif len(coef) == n_atlas + N_COVARIATES:
        cov = coef[n_atlas:]
        cursor = 0
    else:
        raise ValueError(
            f"coefficient length {len(coef)} does not match atlases {atlases} "
            f"(expected {n_atlas} or {n_atlas + N_COVARIATES})"
        )
    blocks = {}
    for name, size in zip(atlases, sizes, strict=True):
        blocks[name] = coef[cursor : cursor + size]
        cursor += size
    return blocks, cov


def coef_to_edge_matrix(block: np.ndarray, n_parcels: int) -> np.ndarray:
    """Undo the tangent upper-triangle (k=0) packing, including the diagonal."""
    iu = np.triu_indices(n_parcels, k=0)
    if block.size != len(iu[0]):
        raise ValueError(f"block has {block.size} weights, atlas has {n_parcels} parcels")
    mat = np.zeros((n_parcels, n_parcels), dtype=np.float64)
    mat[iu] = block
    return mat + mat.T - np.diag(np.diag(mat))


def asd_direction(coef: np.ndarray) -> np.ndarray:
    """Flip so positive weights increase the ASD score, matching the GNN plots."""
    return -np.asarray(coef, dtype=np.float64)


def _empty_test(source, groups: np.ndarray):
    from autoresearch.trial import Split

    return Split(source, np.array([], dtype=int), groups, with_labels=False)


def _build_features(train, test, atlases: list[str]) -> np.ndarray:
    from neuroasd import model as scored

    blocks = [scored._tangent_block(train, test, atlas) for atlas in atlases]
    feats = np.concatenate(blocks, axis=1)
    if hasattr(scored, "_covariate_columns"):
        cov = scored._covariate_columns(train, test).astype(feats.dtype)
        feats = np.concatenate([feats, cov], axis=1)
    n_train = len(train.y)
    mu = feats[:n_train].mean(axis=0, dtype=np.float64).astype(feats.dtype)
    sd = feats[:n_train].std(axis=0, dtype=np.float64).astype(feats.dtype)
    sd[sd == 0] = 1.0
    feats -= mu
    feats /= sd
    return feats


def fit_fold_coef(source, train_idx: np.ndarray, groups: np.ndarray, atlases: list[str]) -> np.ndarray:
    from autoresearch.trial import Split
    from neuroasd import model as scored

    train = Split(source, train_idx, groups, with_labels=True)
    test = _empty_test(source, groups)
    feats = _build_features(train, test, atlases)
    clf = scored._make_estimator().fit(feats[: len(train.y)], np.asarray(train.y), np.asarray(train.groups))
    return estimator_coef(clf)


def average_coefficients(
    source,
    atlases: list[str],
    group_by: str,
    workers: int,
) -> tuple[np.ndarray, list[str]]:
    """Mean coefficient vector over training folds, or one fit on all subjects.

    ``all`` is a visualization shortcut: one fit on every subject.
    ``institution`` / ``site`` average the training-fold coefficients of the
    same outer split used when scoring those held-out groups.
    """
    from concurrent.futures import ThreadPoolExecutor

    from autoresearch.trial import test_groups

    groups = np.array([GROUPINGS[group_by](site) if group_by != "all" else "all" for site in source.sites])
    if group_by == "all":
        coef = fit_fold_coef(source, np.arange(len(source.y)), groups, atlases)
        return coef, ["all"]

    held_out = test_groups(groups, source.y, None)
    train_sets = [np.where(groups != group)[0] for group in held_out]

    def one(idx: np.ndarray) -> np.ndarray:
        return fit_fold_coef(source, idx, groups, atlases)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        coefs = list(pool.map(one, train_sets))
    return np.mean(np.stack(coefs), axis=0), held_out


def explain_linear(
    output_dir: Path | str | None = None,
    group_by: str = "all",
    top_rois: int = 20,
    top_edges: int = 80,
    workers: int = 2,
    atlases: list[str] | None = None,
) -> dict:
    """Fit the current scored linear model and write a chord plot per atlas.

    Default ``group_by="all"`` is a cheap draft. Pass ``group_by="institution"``
    if the figure should use the same training folds as leave-one-institution-out.
    """
    from autoresearch.trial import Source

    if group_by not in {"all", "site", "institution"}:
        raise ValueError("group_by must be all, site, or institution")

    from neuroasd import model as scored
    from neuroasd.explain import write_attribution_outputs

    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "outputs" / "explain_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    used = list(atlases) if atlases else atlases_used_by_model()
    source = Source()
    n_parcels = {name: source.fc(name).shape[-1] for name in used}
    coef, folds = average_coefficients(source, used, group_by, workers)
    asd_coef = asd_direction(coef)
    blocks, cov = split_coef_blocks(asd_coef, used, n_parcels)
    model_name = type(scored._make_estimator()).__name__

    summaries = []
    figures = {}
    for name, block in blocks.items():
        table = atlas_table(name, ATLAS_DIRS[name])
        edge = coef_to_edge_matrix(block, n_parcels[name])
        node = edge.sum(axis=1)
        atlas_dir = out_dir / name
        xlabel = "Linear coefficient (ASD direction)"
        title = f"Linear coefficients, {name} tangent ({group_by})"
        write_attribution_outputs(
            atlas_dir,
            table,
            node,
            edge,
            title=title,
            top_rois=top_rois,
            top_edges=top_edges,
            xlabel=xlabel,
            extra_summary={
                "model": model_name,
                "method": "linear coefficients reshaped from tangent upper triangle",
                "atlas": name,
                "group_by": group_by,
            },
        )
        summaries.append(
            {
                "atlas": name,
                "n_parcels": n_parcels[name],
                "top_roi": table[int(np.argmax(np.abs(node)))]["name"],
            }
        )
        figures[name] = {"chord": f"{name}/chord.png", "top_rois": f"{name}/top_rois.png"}

    summary = {
        "output_dir": str(out_dir),
        "kind": "linear",
        "model": model_name,
        "method": "mean linear coefficients over training folds" if group_by != "all" else "one fit on all subjects",
        "explained_at": datetime.now(timezone.utc).isoformat(),
        "group_by": group_by,
        "folds": folds,
        "atlases": used,
        "covariates_age_sex": [float(v) for v in cov] if cov.size else [],
        "atlases_summary": summaries,
        "figures": figures,
        "note": "Positive weights increase the ASD score. group_by=all is for visualization only.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
