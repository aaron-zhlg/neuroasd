# iter30_adding-one-fixed-configuration-h v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Adding one fixed-configuration HistGradientBoostingClassifier member on the existing age/sex + Gordon+HCP tangent design and equal-weight averaging its P(control) with the incumbent L2 logistic C=1e-2 P(control) inside that member's first stacked column injects threshold/interaction structure the linear logistic cannot express, raising stacked ranking (AUC) while ASD recall rises with it.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.712 ± 0.172** |
| Control recall | 0.751 ± 0.165 |
| Balanced accuracy | 0.732 ± 0.070 |
| Accuracy | 0.724 ± 0.078 |
| AUC | 0.788 ± 0.094 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter30_adding-one-fixed-configuration-h --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
