# iter15_a-train-fold-only-coral-style-se v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-19
> **Hypothesis:** A train-fold-only CORAL-style second-order site-alignment penalty on the mean-pooled graph embedding (per-batch site covariances matched to the batch-pooled covariance, dimension-normalized, lambda=0.1 added to cross-entropy) reduces the per-site LOSO AUC spread the incumbent leaves untargeted.
> **Files changed:** neuroasd/gcn.py, neuroasd/train.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.640 ± 0.121 |
| **AUC** | **0.692 ± 0.115** |
| F1 | 0.642 ± 0.137 |

Diagnostic best-epoch AUC (not reported): 0.7672.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter15_a-train-fold-only-coral-style-se --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
