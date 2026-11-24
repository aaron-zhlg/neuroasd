# iter39_appending-a-fixed-nonlinear-demo v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Appending a fixed nonlinear demographic basis (age^2 and age x sex, standardized on each fit's own training rows only) to the always-kept covariate block of every linear member design and the HGB column gives the linear stack a nonlinear age effect and a sex-specific age slope that no existing column can express, raising cross-site AUC without adding a meta column or spending control-recall margin.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.747 ± 0.172** |
| Control recall | 0.734 ± 0.154 |
| Balanced accuracy | 0.740 ± 0.069 |
| Accuracy | 0.730 ± 0.073 |
| AUC | 0.790 ± 0.100 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter39_appending-a-fixed-nonlinear-demo --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
