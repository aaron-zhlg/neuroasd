# iter6_fisher-z-transform-arctanh-with- v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-19
> **Hypothesis:** Fisher z-transform (arctanh, with FC correlations clipped to +/-0.999 before the transform) of the FC correlation values used as node features improves cross-site LOSO AUC.
> **Files changed:** neuroasd/gcn.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.644 ± 0.124 |
| **AUC** | **0.690 ± 0.120** |
| F1 | 0.649 ± 0.135 |

Diagnostic best-epoch AUC (not reported): 0.7616.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter6_fisher-z-transform-arctanh-with- --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
