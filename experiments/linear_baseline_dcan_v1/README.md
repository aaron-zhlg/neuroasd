# Linear baseline v1: ABIDE I + II (DCAN), leave-one-site-out and leave-one-institution-out

Reference baseline for the ABIDE I + II dataset in `data/abide_dcan`. Future models on this dataset should beat it under the same protocol.

> **Evaluated:** 2026-10-03
> **Data:** 1,443 subjects (631 ASD / 812 control), 35 sites, Gordon atlas (352 parcels)
> **Code:** `neuroasd/linear_baseline.py`; checks in `neuroasd/check_linear_baseline.py`
> **Model selection:** ridge strength by inner CV on training folds only; nothing is selected on a test fold

---

## Summary

Tangent-space connectivity with ridge classification gives 68.6% accuracy and AUC 0.775 under leave-one-site-out, and 67.0% / 0.770 under the stricter leave-one-institution-out.

| Split | Features | Folds | Accuracy (fold mean ± SD) | Accuracy (pooled) | AUC | ASD recall | Control recall |
|---|---|---|---|---|---|---|---|
| Leave-one-site-out | **tangent** | 33 | **68.6% ± 7.7** | 69.0% | **0.775 ± 0.101** | 54.3% | 80.4% |
| Leave-one-site-out | Fisher-z | 33 | 67.4% ± 8.8 | 69.2% | 0.747 ± 0.091 | 55.5% | 77.7% |
| Leave-one-institution-out | **tangent** | 23 | **67.0% ± 8.6** | 66.0% | **0.770 ± 0.078** | 55.1% | 79.8% |
| Leave-one-institution-out | Fisher-z | 23 | 65.3% ± 10.1 | 67.0% | 0.733 ± 0.087 | 56.1% | 75.2% |

Fold mean = unweighted mean over test folds (the primary number). Pooled = all test subjects together. F1 (control class, as in `neuroasd.train`) is in `results.json`.

**Robustness.** Each fold was refit on 5 stratified 90% subsamples of its training set:

| Split | Features | Accuracy (mean ± SD over 5 refits) | AUC (mean ± SD over 5 refits) |
|---|---|---|---|
| Leave-one-site-out | tangent | 68.6% ± 1.1 | 0.773 ± 0.003 |
| Leave-one-site-out | Fisher-z | 67.5% ± 0.2 | 0.746 ± 0.007 |
| Leave-one-institution-out | tangent | 65.9% ± 0.4 | 0.763 ± 0.003 |
| Leave-one-institution-out | Fisher-z | 65.5% ± 0.6 | 0.732 ± 0.005 |

The spread over refits is far smaller than the spread over sites, so differences between models should be judged with paired per-fold tests, not against the refit SD.

---

## Breakdown

Leave-one-site-out, by the release a test site belongs to:

| Features | ABIDE I sites (20, n = 789) | ABIDE II sites (13, n = 612) |
|---|---|---|
| tangent | 70.9% / AUC 0.803 | 64.9% / AUC 0.731 |
| Fisher-z | 67.8% / AUC 0.762 | 66.7% / AUC 0.724 |

ABIDE II sites are harder. Several of them are small (23-47 subjects) and some are strongly imbalanced (KKI_1: 24 ASD / 100 control).

### Site vs institution

Holding out whole institutions removes the case where the model trains on another cohort from the same scanner. For the 10 institutions with more than one cohort (tangent features):

| Institution | Cohorts | Accuracy, site split | Accuracy, institution split | AUC, institution split |
|---|---|---|---|---|
| KKI | KKI, KKI_1 | 70.3% | **53.8%** | 0.724 |
| Leuven | LEUVEN_1, LEUVEN_2, KUL_3 | 73.5% | **46.6%** | 0.797 |
| NYU | NYU, NYU_1, NYU_2 | 68.3% | 70.3% | 0.789 |
| OHSU | OHSU, OHSU_1 | 73.1% | 73.9% | 0.808 |
| Olin | OLIN, ONRC_2 | 69.0% | 65.5% | 0.708 |
| SDSU | SDSU, SDSU_1 | 66.2% | 67.5% | 0.765 |
| Trinity | TRINITY, TCD_1 | 67.2% | 65.6% | 0.704 |
| UCLA | UCLA_1, UCLA_2 | 66.7% | 66.7% | 0.759 |
| UM | UM_1, UM_2 | 67.3% | 67.3% | 0.798 |
| USM | USM, USM_1 | 70.5% | 67.6% | 0.874 |

Notes on the table:

- Site-split accuracy is pooled over the institution's cohorts that are test sites.
- For Leuven and NYU, the institution fold also tests the ASD-only cohorts (KUL_3, NYU_2), which are training-only under the site split.

Most institutions barely change. KKI and Leuven lose 17-27 points of accuracy, while their AUC stays at 0.72-0.80. The ranking of subjects survives; the decision threshold does not. A whole unseen institution shifts every score, and with a skewed class mix (KKI is 79% control; Leuven includes the all-ASD KUL_3 cohort) a fixed threshold at 0 then misclassifies one class wholesale. Site-level score calibration is a natural thing for later models to address.

---

## Methods

### Data

`data/abide_dcan`; full details are in `data/abide_dcan/README.md`.

- DCAN (ABCD-HCP pipeline) parcellated time series from FCP-INDI, Gordon 2014 atlas plus FreeSurfer subcortical (352 parcels).
- Frames with FD > 0.2 mm censored; subjects with less than 180 s retained are dropped.
- Excluded:
  - 8 ABIDE II re-releases of ABIDE I scans (`duplicate_scans.csv`);
  - 16 subjects with unscanned parcels (`incomplete_coverage.csv`);
  - ETHZ_1 (labels not on S3);
  - the UCLA_Long and UPSM_Long follow-up cohorts.
- Sites with only one class (`ABIDEII-KUL_3`, `ABIDEII-NYU_2`) are used for training only under the site split.

### Features

- **Fisher-z:** arctanh of the upper triangle of the Pearson FC matrix (61,776 features).
- **Tangent:**
  - each FC matrix is shrunk toward identity (0.1);
  - each is whitened by the log-Euclidean mean of the **training** subjects;
  - each is mapped with the matrix logarithm;
  - the upper triangle including the diagonal is kept, with off-diagonal entries weighted by √2 (62,128 features).

### Classifier

- Ridge classification with targets ±1, solved in kernel (dual) form, after standardizing features on the training fold.
- Ridge strength is chosen from 13 values in 10^-1 … 10^5 by 5-fold inner cross-validation on the training folds. The inner folds are grouped like the outer split (by site or by institution), and the score is AUC.
- Predicted label = control if the decision value > 0.

### Validation

- **Leave-one-site-out:** each `SITE_ID` with both classes is a test fold (33 folds).
- **Leave-one-institution-out:** cohorts from the same institution are merged into one fold (23 folds). The mapping is in `INSTITUTIONS` in `neuroasd/linear_baseline.py`.
- In every fold, the tangent reference, feature scaling, and ridge strength are fit on training subjects only.

### Leakage checks

| Check | Result |
|---|---|
| Duplicate scans between ABIDE I and II (FC similarity, motion trace, phenotype) | 11 pairs found and removed; none left (`data/abide_dcan/README.md`) |
| Within-site label permutation, full leave-one-site-out pipeline, Fisher-z, 3 seeds | accuracy 49.1-50.2%, AUC 0.47-0.50 (chance) |
| Synthetic checks (`python -m neuroasd.check_linear_baseline`) | kernel ridge matches scikit-learn; tangent reference ignores test subjects; single-class sites never tested; institution folds hold out every cohort; missing parcels stay finite |

The permutation check ran on the 1,460-subject build, before the last duplicate pair and the 16 coverage exclusions were removed. These changes only remove subjects, so they cannot introduce leakage.

---

## Comparison

| Model | Data | Validation | Accuracy | AUC |
|---|---|---|---|---|
| GCN, loso_cv_gcn_v1 | C-PAC ABIDE I, 884 subjects | Leave-one-site-out, 20 sites | 59.3% | 0.623 |
| Best autoresearch GCN | C-PAC ABIDE I, 884 subjects | Leave-one-site-out, 20 sites | 64.3% | 0.717 |
| **This baseline (tangent)** | DCAN ABIDE I + II, 1,443 subjects | Leave-one-site-out, 33 sites | **68.6%** | **0.775** |
| **This baseline (tangent)** | DCAN ABIDE I + II, 1,443 subjects | Leave-one-institution-out, 23 institutions | **67.0%** | **0.770** |

Strict leave-one-site-out results in the literature are about 65-67% accuracy and AUC 0.72-0.81 (`doc/literature_review_asd_fmri.md`).

---

## Caveats

- **ASD recall is low (54-56%).**
  - The training data are 56% control and the threshold is fixed at 0, so the model favours the majority class.
  - Accuracy alone hides this; report ASD recall alongside it.
- **The chosen ridge strength often lands at the edge of the grid.** In many folds it is 3.2 × 10^4 (one step below the top value), and in a few it is 0.1 (the bottom value). A wider grid might help slightly. It was kept fixed so that this run stays the reference.
- **Fold sizes vary from 7 to 239 subjects.** Fold-mean metrics weight small sites (OLIN 7, SBL 9, CMU 9) as much as NYU, which is why both fold-mean and pooled accuracy are reported.
- Same-institution cohorts make leave-one-site-out optimistic for KKI and Leuven. **Use leave-one-institution-out when the claim is generalization to a new institution.**

---

## Reproduce

```bash
uv run data/abide_dcan/scripts/build_dcan_dataset.py           # about 6 min, about 1 GB
uv run python -m neuroasd.check_linear_baseline
uv run python -m neuroasd.linear_baseline --repeats 5 --workers 4   # about 75 min on 4 cores
```

- `results.json`: summaries for both splits and per-fold metrics.
- `run_config.json`: settings.
- `outputs/linear_baseline_dcan_v1/` (local only): the log and per-refit folds.
