# iter13_training-fit-only-univariate-edg v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Training-fit-only univariate edge selection (top-K by |two-sample t| between ASD/control rows, K pre-registered as 20k Gordon+HCP / 8k Power / 12k Markov, recomputed inside every outer and inner-OOF fit) removes noisy edges so each member ranks ASD higher; discriminator: loso-site AUC must beat base 0.8039 by more than the ~0.10 fold SD.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.661 ± 0.157** |
| Control recall | 0.773 ± 0.140 |
| Balanced accuracy | 0.717 ± 0.076 |
| Accuracy | 0.714 ± 0.081 |
| AUC | 0.790 ± 0.095 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter13_training-fit-only-univariate-edg --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
