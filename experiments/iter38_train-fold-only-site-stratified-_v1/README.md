# iter38_train-fold-only-site-stratified- v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** Train-fold-only site-stratified mixup (one fixed MIXUP_ALPHA=0.2, opt-in kwarg default 0.0) on the [B,111] node-feature batch with soft-target CE, both components anchored to the anchor row's site, lifts final-epoch LOSO auc_mean above the iter34 bar 0.7145.
> **Files changed:** neuroasd/train.py, autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.638 ± 0.093 |
| **AUC** | **0.717 ± 0.118** |
| F1 | 0.684 ± 0.100 |

Diagnostic best-epoch AUC (not reported): 0.7594.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter38_train-fold-only-site-stratified- --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
