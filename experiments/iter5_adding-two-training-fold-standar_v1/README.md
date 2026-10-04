# iter5_adding-two-training-fold-standar v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-10-04
> **Hypothesis:** Adding two training-fold-standardized covariate columns (age z-score, binary sex) to the Gordon tangent ridge input gives the ridge linear kernel within-institution ranking signal beyond the connectome features, which calibration provably cannot provide.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **0.703 ± 0.071** |
| Accuracy | 0.694 ± 0.070 |
| AUC | 0.770 ± 0.079 |
| ASD recall | 0.712 ± 0.114 |
| Control recall | 0.693 ± 0.098 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter5_adding-two-training-fold-standar --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
