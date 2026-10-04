# iter4_label-free-per-group-score-calib v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-10-04
> **Hypothesis:** Label-free per-group score calibration: z-score each institution's own decision values (training OOF by its own group mean/SD, held-out test fold by its own unlabeled mean/SD) before the frozen training-only balanced-accuracy cut, so per-institution operating-point mismatch stops mis-firing a single global threshold.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **0.693 ± 0.076** |
| Accuracy | 0.683 ± 0.073 |
| AUC | 0.770 ± 0.078 |
| ASD recall | 0.718 ± 0.122 |
| Control recall | 0.668 ± 0.115 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter4_label-free-per-group-score-calib --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
