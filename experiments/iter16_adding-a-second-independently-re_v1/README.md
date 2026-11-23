# iter16_adding-a-second-independently-re v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Adding a second, independently regularized L2 logistic member (different C, e.g. C=1e-1) on the same age/sex + Gordon + HCP concatenated tangent features and combining the two members with a fixed equal-weight average of their decision scores via neuroasd.ensemble raises cross-subject ranking (AUC) over the single-C=1e-2 member, because two regularization scales capture complementary large-scale vs fine-grained FC signal.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.701 ± 0.167** |
| Control recall | 0.742 ± 0.158 |
| Balanced accuracy | 0.721 ± 0.071 |
| Accuracy | 0.714 ± 0.078 |
| AUC | 0.789 ± 0.094 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter16_adding-a-second-independently-re --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
