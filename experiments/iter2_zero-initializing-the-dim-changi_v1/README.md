# iter2_zero-initializing-the-dim-changi v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-19
> **Hypothesis:** Zero-initializing the dim-changing (111->64) skip projection makes the residual skip an exact zero map at start, isolating anti-over-smoothing from extra first-layer linear capacity.
> **Files changed:** neuroasd/gcn.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.617 ± 0.098 |
| **AUC** | **0.661 ± 0.131** |
| F1 | 0.629 ± 0.109 |

Diagnostic best-epoch AUC (not reported): 0.7637.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter2_zero-initializing-the-dim-changi --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
