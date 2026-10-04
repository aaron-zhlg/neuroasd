# iter25_cosine-lr-annealing-on-the-exist v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** Cosine LR annealing on the existing Adam optimizer (T_max=100, eta_min=1e-5, peak LR unchanged at 1e-3) recovers the ~0.075 final-vs-best-epoch decay under a constant LR, lifting final-epoch auc_mean over the 0.6917 iter15 parent without any epoch selection.
> **Files changed:** autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.632 ± 0.102 |
| **AUC** | **0.703 ± 0.116** |
| F1 | 0.649 ± 0.120 |

Diagnostic best-epoch AUC (not reported): 0.759.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter25_cosine-lr-annealing-on-the-exist --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
