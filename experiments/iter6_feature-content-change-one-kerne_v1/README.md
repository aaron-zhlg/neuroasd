# iter6_feature-content-change-one-kerne v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-10-04
> **Hypothesis:** Feature-content change: one kernel-ridge fit on the training-fold-standardized concatenation of Gordon + HCP tangent-space features gives within-group ranking signal that calibration/stacking could not.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **0.708 ± 0.085** |
| Accuracy | 0.701 ± 0.089 |
| AUC | 0.776 ± 0.091 |
| ASD recall | 0.715 ± 0.122 |
| Control recall | 0.702 ± 0.096 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter6_feature-content-change-one-kerne --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
