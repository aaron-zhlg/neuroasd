# iter10_estimator-swap-replace-the-singl v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

Paper-oriented comparison (baseline, literature, what moved AUC vs the
operating point, how to reproduce this tag):
[`doc/paper_notes_iter10.md`](../../doc/paper_notes_iter10.md).
Git tag: `iter10-loso-institution-v1`.

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

## Scores under other split schemes

Same model, 1443 subjects, one fit per fold. Fold mean ± SD; k-fold rows are 10 folds × 3
reshuffles. Full numbers: [`split_schemes.json`](split_schemes.json).

| Split | Scoring | Bal. acc | Accuracy | AUC | ASD recall | Ctl recall |
|-------|---------|----------|----------|-----|------------|------------|
| Leave-one-institution-out (23) | per-group | 0.708 ± 0.081 | 0.701 ± 0.085 | 0.779 ± 0.087 | 0.708 | 0.709 |
| Leave-one-institution-out (23) | per-subject | 0.677 ± 0.094 | 0.674 ± 0.107 | 0.779 ± 0.087 | 0.550 | 0.803 |
| Leave-one-site-out (33) | per-group | 0.728 ± 0.094 | 0.719 ± 0.096 | 0.797 ± 0.098 | 0.745 | 0.712 |
| Leave-one-site-out (33) | per-subject | 0.683 ± 0.101 | 0.694 ± 0.096 | 0.797 ± 0.098 | 0.547 | 0.819 |
| Site-stratified 10-fold | per-group | 0.713 ± 0.039 | 0.710 ± 0.038 | 0.784 ± 0.041 | 0.739 | 0.687 |
| Site-stratified 10-fold | per-subject | 0.703 ± 0.035 | 0.722 ± 0.034 | 0.809 ± 0.035 | 0.546 | 0.859 |
| Random 10-fold | per-group | 0.671 ± 0.040 | 0.670 ± 0.039 | 0.736 ± 0.040 | 0.674 | 0.667 |
| Random 10-fold | per-subject | 0.694 ± 0.037 | 0.713 ± 0.036 | 0.801 ± 0.032 | 0.536 | 0.851 |

- **per-group** is iter10 as released: test scores are standardized per site / institution
  from that group's own unlabeled scores, then cut at a training-only threshold.
- **per-subject** scores every subject on its own (logistic decision > 0 = control), the way
  most papers score a classifier. Use these rows to compare with the literature.

Per-group scoring needs many test subjects per site. Under random k-fold a site often has
1–3 test subjects, so the standardization is noisy and lowers AUC (0.736 vs 0.801). Under
leave-one-group-out the test fold is one group, so AUC is identical for both scorings and
per-group scoring only moves the operating point (balanced ASD / control recall).

Reproduce (`--split site | institution | random | stratified`, `--scoring per-group | per-subject`):

```bash
uv run python -m scripts.train_model --data-root /path/to/data --split institution --cv
uv run python -m scripts.train_model --data-root /path/to/data --split random --cv --folds 10 --repeats 3
```
