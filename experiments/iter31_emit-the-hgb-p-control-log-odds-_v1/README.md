# iter31_emit-the-hgb-p-control-log-odds- v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** 2026-11-23
> **Hypothesis:** Emit the HGB P(control) log-odds as one NEW final (fifth) meta-learner column on top of the byte-identical PR #25 four-column stack, so the meta-learner fits its own weight on the booster instead of pre-averaging it 50/50 into column 0 — this restores column-0 monotonicity/control recall and can only raise AUC if the booster carries ranking information the logistic lacks.
> **Files changed:** neuroasd/model.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **ASD recall** | **0.717 ± 0.174** |
| Control recall | 0.734 ± 0.171 |
| Balanced accuracy | 0.726 ± 0.067 |
| Accuracy | 0.716 ± 0.075 |
| AUC | 0.791 ± 0.093 |

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter31_emit-the-hgb-p-control-log-odds- --stage loso-institution
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
