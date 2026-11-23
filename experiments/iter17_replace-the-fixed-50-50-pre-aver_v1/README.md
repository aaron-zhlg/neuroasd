# iter17_replace-the-fixed-50-50-pre-aver v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Replace the fixed 50/50 pre-average of the Gordon+HCP member's two L2 scales (C=1e-2, C=1e-1) with two separate training-fold OOF score columns fed to the existing logistic meta-learner (stack_scores, META_C=1e-2, 4 columns instead of 3), so the mixing weight between coarse and fine regularization is FITTED on training-fold OOF rows instead of hand-set, which should raise cross-site ranking (AUC) at the unchanged cut.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.713 ± 0.171** |
| Control recall | 0.735 ± 0.167 |
| Balanced accuracy | 0.724 ± 0.067 |
| Accuracy | 0.715 ± 0.075 |
| AUC | 0.791 ± 0.093 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter17_replace-the-fixed-50-50-pre-aver --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
