# iter9_replacing-each-subject-s-pearson v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Replacing each subject's Pearson FC estimate with a per-subject Ledoit-Wolf shrunk correlation (adaptive intensity, closed-form Schaefer-Strimmer/Ledoit-Wolf formula with a moment-estimated effective sample size) before the tangent map denoises small-sample FC and should raise ranking AUC of the existing 3-member stack above the iter4 base (site AUC 0.798) while leaving the cut rule untouched.
> **Files changed:** neuroasd/linear_baseline.py, neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.636 ± 0.178** |
| Control recall | 0.784 ± 0.136 |
| Balanced accuracy | 0.710 ± 0.091 |
| Accuracy | 0.708 ± 0.091 |
| AUC | 0.785 ± 0.099 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter9_replacing-each-subject-s-pearson --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
