# iter38_replacing-only-stacked-meta-colu v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Replacing only stacked meta column 0's estimator (Gordon+HCP at the same fixed C=1e-2) with an L1-penalized sparse logistic — identical tangent map, training-only standardization, |t| top-K mask, age/sex and ASD class-weight grid, all other columns frozen — denoises the most heavily weighted member by zeroing most of the ~20k edge coefficients, so cross-site AUC rises instead of the cut merely moving (diagnostics emitted once as DIAG_JSON on stderr).
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.741 ± 0.171** |
| Control recall | 0.732 ± 0.160 |
| Balanced accuracy | 0.736 ± 0.068 |
| Accuracy | 0.727 ± 0.073 |
| AUC | 0.791 ± 0.100 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter38_replacing-only-stacked-meta-colu --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
