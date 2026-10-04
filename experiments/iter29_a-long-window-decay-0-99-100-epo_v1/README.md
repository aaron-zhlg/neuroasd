# iter29_a-long-window-decay-0-99-100-epo v1 — Autoresearch promotion

Frozen after a `loso-full` PASS. Metrics are **final-epoch** only.

> **Evaluated:** 2026-09-20
> **Hypothesis:** A long-window (decay 0.99, ~100-epoch window spanning the whole cosine anneal) exponential moving average of the model weights, read exactly once at the final epoch with no epoch selection, recovers the +0.07 final-vs-best tail decay and lifts loso AUC above the 0.7027 iter25 winner.
> **Files changed:** autoresearch/trial.py

## Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.641 ± 0.101 |
| **AUC** | **0.713 ± 0.114** |
| F1 | 0.685 ± 0.101 |

Diagnostic best-epoch AUC (not reported): 0.759.

## Reproduce

```bash
uv run python -m autoresearch.trial --name iter29_a-long-window-decay-0-99-100-epo --stage loso-full
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
