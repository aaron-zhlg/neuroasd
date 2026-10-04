"""Simple graph convolution network for FC-based classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FISHER_CLIP = 0.999


def fisher_z_transform(fc_values: torch.Tensor) -> torch.Tensor:
    """Fisher z (arctanh) of FC correlation values, clipped so the result is finite.

    Raw correlations are compressed near 0 and stretched near +/-1; arctanh makes
    the sampling variance of r roughly constant, so the near-+/-1 tail where
    scanner/site effects concentrate no longer dominates the raw feature scale.
    Correlations are clipped to +/-FISHER_CLIP first so atanh stays finite
    (< atanh(0.999) ~ 3.8) and no NaN/inf can enter the forward pass.
    """
    return torch.atanh(fc_values.clamp(-FISHER_CLIP, FISHER_CLIP))


def quantile_normalize_edges(
    adjacency: torch.Tensor,
    site_ids: list[str],
    tables: dict[str, tuple[torch.Tensor, torch.Tensor]],
    pooled_table: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Per-site rank/quantile normalization of the Fisher-z |FC| edge weights.

    `tables[site] = (source, target)` are quantile grids fit on training-fold
    edges of that site only. Each edge weight is mapped through its site's grid
    onto the common target grid, so site-specific edge scale/outliers are
    removed while the within-site ordering of edges (the signal) is preserved
    because the map is monotone. Sites unseen in training use `pooled_table`.
    """
    weights = fisher_z_transform(adjacency)
    mapped = weights.clone()
    for site in dict.fromkeys(site_ids):
        source, target = tables.get(site, pooled_table)
        mask = torch.tensor([value == site for value in site_ids], device=weights.device)
        values = weights[mask].contiguous()
        index = torch.searchsorted(source, values).clamp(1, source.numel() - 1)
        low, high = source[index - 1], source[index]
        frac = ((values - low) / (high - low).clamp(min=1e-6)).clamp(0.0, 1.0)
        mapped[mask] = target[index - 1] + frac * (target[index] - target[index - 1])
    return mapped


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loops. Input shape: (B, N, N)."""
    adj = adjacency.clone()
    batch_size, num_nodes, _ = adj.shape
    eye = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
    adj = adj + eye
    degree = adj.sum(dim=-1).clamp(min=1e-6)
    inv_sqrt = degree.pow(-0.5)
    adj = inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)
    return adj


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Skip path only needs a parameterised 1x1 projection when dims differ.
        self.skip = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else None
        )
        if self.skip is not None:
            # Starting the dim-changing skip at an exact zero map removes the
            # free random linear pathway, so any learned skip is a deviation
            # from "no skip" rather than plus-capacity at init.
            nn.init.zeros_(self.skip.weight)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # node_features: (B, N, F), adjacency: (B, N, N)
        support = self.linear(node_features)
        residual = node_features if self.skip is None else self.skip(node_features)
        return residual + torch.bmm(adjacency, support)


class SimpleGCN(nn.Module):
    """Two-layer GCN with global mean pooling for graph classification."""

    def __init__(
        self,
        in_features: int = 111,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GraphConvLayer(in_features, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Fisher z on the raw FC correlation values before they are treated as
        # node features (single feature-distribution change; nothing rescaled or
        # harmonized per site).
        node_features = fisher_z_transform(node_features)

        adj_norm = normalize_adjacency(adjacency)

        x = self.conv1(node_features, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = x.mean(dim=1)
        return self.classifier(graph_embedding)
