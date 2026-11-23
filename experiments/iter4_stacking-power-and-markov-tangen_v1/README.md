# iter4_stacking-power-and-markov-tangen v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-22
> **Hypothesis:** Stacking Power and Markov tangent members with the Gordon+HCP member via a training-fold-OOF logistic meta-learner (neuroasd.ensemble.stack_scores) adds ranking information (AUC above 0.8291) while the unchanged iter1_a training-only OOF cut still sets the operating point.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.621 ± 0.176** |
| Control recall | 0.781 ± 0.168 |
| Balanced accuracy | 0.701 ± 0.093 |
| Accuracy | 0.698 ± 0.098 |
| AUC | 0.780 ± 0.104 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter4_stacking-power-and-markov-tangen --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
