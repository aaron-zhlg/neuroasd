# iter42_eval-time-augmentation-averaging v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** Eval-time augmentation averaging: scoring the single final-epoch EMA weights over K=8 deterministic draws of the existing iter39 edge-drop mask (p=0.1) and averaging the per-subject probabilities reduces inference variance and lifts final-epoch LOSO auc_mean above 0.7169 (inference-only, training path untouched).
> **Files changed:** autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.642 ± 0.099 |
| **AUC** | **0.717 ± 0.115** |
| F1 | 0.693 ± 0.098 |

Diagnostic best-epoch AUC (not reported): 0.7567.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter42_eval-time-augmentation-averaging --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
