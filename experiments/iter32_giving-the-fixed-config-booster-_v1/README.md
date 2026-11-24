# iter32_giving-the-fixed-config-booster- v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Giving the fixed-config booster a genuinely different feature space — the Power+Markov tangent edge blocks concatenated with age/sex, reduced by its own training-fold-only top-K=2000 |t| ranking instead of the Gordon+HCP 2000-edge slice the linear columns already linearize — can carry conditional-dependence information the four logistic columns lack, raising cross-site/institution AUC beyond the flat-deltoa booster-on-Gordon+HCP results.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.734 ± 0.174** |
| Control recall | 0.734 ± 0.166 |
| Balanced accuracy | 0.734 ± 0.065 |
| Accuracy | 0.724 ± 0.072 |
| AUC | 0.792 ± 0.094 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter32_giving-the-fixed-config-booster- --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
