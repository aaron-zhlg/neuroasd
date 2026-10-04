# iter2_a-decision-threshold-chosen-on-t v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-10-04
> **Hypothesis:** A decision threshold chosen on training folds only (max balanced accuracy over grouped out-of-fold training scores, frozen and applied to the held-out fold) lifts ASD recall without tuning on the evaluation fold.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **0.689 ± 0.069** |
| Accuracy | 0.666 ± 0.082 |
| AUC | 0.770 ± 0.078 |
| ASD recall | 0.775 ± 0.148 |
| Control recall | 0.604 ± 0.180 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter2_a-decision-threshold-chosen-on-t --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
