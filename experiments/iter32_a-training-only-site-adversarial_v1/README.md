# iter32_a-training-only-site-adversarial v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** A training-only site-adversarial head on a gradient-reversal layer over the mean-pooled graph embedding (lambda ramped 0 -> 0.1, training-fold SITE_IDs only, skipped when fewer than two sites are in the batch) squeezes site-specific structure out of the representation the diagnostic classifier sees and lifts cross-site LOSO final-epoch auc_mean above the iter29 EMA baseline 0.7130 — re-implemented with SITE_ALIGN_MIN_SUBJECTS bound and the log-only bookkeeping made total.
> **Files changed:** neuroasd/train.py, autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.639 ± 0.101 |
| **AUC** | **0.713 ± 0.115** |
| F1 | 0.685 ± 0.100 |

Diagnostic best-epoch AUC (not reported): 0.7592.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter32_a-training-only-site-adversarial --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
