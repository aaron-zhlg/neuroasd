# iter1_adding-a-residual-skip-h-h-f-h-w v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-19
> **Hypothesis:** Adding a residual skip (h = h + F(h)) with a 1x1 projection only when dims differ will improve cross-site ASD-vs-control generalization by reducing over-smoothing/feature washout in the 2-layer GCN.
> **Files changed:** neuroasd/gcn.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.609 ± 0.102 |
| **AUC** | **0.661 ± 0.139** |
| F1 | 0.629 ± 0.104 |

Diagnostic best-epoch AUC (not reported): 0.7613.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter1_adding-a-residual-skip-h-h-f-h-w --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
