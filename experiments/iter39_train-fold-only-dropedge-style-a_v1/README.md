# iter39_train-fold-only-dropedge-style-a v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** Train-fold-only DropEdge-style adjacency perturbation (fixed EDGE_DROP_P = 0.1, opt-in kwarg default 0.0, applied to the training batch's edge-weight tensor only, before the model's normalize_adjacency) regularizes message passing along a structural axis and lifts final-epoch LOSO auc_mean above the iter38 bar 0.7165.
> **Files changed:** neuroasd/train.py, autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.646 ± 0.101 |
| **AUC** | **0.717 ± 0.115** |
| F1 | 0.693 ± 0.100 |

Diagnostic best-epoch AUC (not reported): 0.7567.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter39_train-fold-only-dropedge-style-a --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
