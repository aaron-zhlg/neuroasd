# iter14_per-site-rank-quantile-normaliza v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-19
> **Hypothesis:** Per-site rank/quantile normalization of the Fisher-z |FC| edge weights (quantile maps fit on training-fold edges of that site only, pooled training-fold fallback for unseen sites, monotone so intra-site edge ordering is preserved) removes site-specific edge scale/outliers and improves cross-site LOSO final-epoch AUC over the 0.6895 Fisher-z mean-only incumbent.
> **Files changed:** neuroasd/gcn.py, autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.634 ± 0.119 |
| **AUC** | **0.692 ± 0.116** |
| F1 | 0.644 ± 0.127 |

Diagnostic best-epoch AUC (not reported): 0.7652.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter14_per-site-rank-quantile-normaliza --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
