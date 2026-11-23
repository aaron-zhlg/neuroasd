# iter1_a-training-fold-only-asymmetric- v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-22
> **Hypothesis:** A training-fold-only asymmetric class-weight adjustment for the logistic head can raise ASD recall while preserving control recall >= 73%.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.601 ± 0.181** |
| Control recall | 0.780 ± 0.151 |
| Balanced accuracy | 0.690 ± 0.080 |
| Accuracy | 0.687 ± 0.084 |
| AUC | 0.789 ± 0.087 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter1_a-training-fold-only-asymmetric- --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
