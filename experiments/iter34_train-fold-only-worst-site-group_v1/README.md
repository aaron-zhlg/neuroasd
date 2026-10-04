# iter34_train-fold-only-worst-site-group v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** Train-fold-only worst-site (group-DRO) reweighting of the classification loss: per-site mean CE over the batch's training-fold SITE_IDs (skip <2-subject sites), detached per-epoch-reset per-site risk EMA (decay 0.9), softmax(r_s/T=1.0)-weighted sample mean (uniform weights at epoch 0 -> step 0 baseline-identical), attacking the weak-site tail.
> **Files changed:** neuroasd/train.py, autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.619 ± 0.122 |
| **AUC** | **0.715 ± 0.114** |
| F1 | 0.670 ± 0.106 |

Diagnostic best-epoch AUC (not reported): 0.7619.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter34_train-fold-only-worst-site-group --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
