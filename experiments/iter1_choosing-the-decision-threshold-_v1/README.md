# iter1_choosing-the-decision-threshold- v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-22
> **Hypothesis:** Choosing the decision threshold from training-fold out-of-fold predictions (maximizing ASD recall s.t. control recall >= 73%) will recover ASD sensitivity that the fixed 0.5 cut loses, without touching the test-set distribution.
> **Files changed:** neuroasd/linear_baseline.py, neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.598 ± 0.190** |
| Control recall | 0.751 ± 0.167 |
| Balanced accuracy | 0.675 ± 0.083 |
| Accuracy | 0.670 ± 0.090 |
| AUC | 0.779 ± 0.087 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter1_choosing-the-decision-threshold- --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
