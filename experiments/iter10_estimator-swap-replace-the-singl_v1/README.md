# iter10_estimator-swap-replace-the-singl v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-10-04
> **Hypothesis:** Estimator swap: replace the single linear-kernel KernelRidgeClassifier with a strongly L2-regularized logistic regression at one fixed pre-registered strength (C=1e-2, no CV / no test-fold selection) fitted on the byte-identical training-fold-standardized Gordon+HCP tangent block, which may move within-group ranking that ridge has plateaued on.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **0.708 ± 0.081** |
| Accuracy | 0.701 ± 0.085 |
| AUC | 0.779 ± 0.087 |
| ASD recall | 0.708 ± 0.118 |
| Control recall | 0.709 ± 0.093 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter10_estimator-swap-replace-the-singl --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
