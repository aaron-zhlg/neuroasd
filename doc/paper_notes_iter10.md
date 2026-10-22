# Paper notes: iter10 snapshot

Pinned at git tag `iter10-loso-institution-v1` on branch
`experiment/trial-iter10_estimator-swap-replace-the-singl` (PR
[#18](https://github.com/aaron-zhlg/neuroasd/pull/18)). These notes are for
later writing, not a result by themselves. Frozen scores live in
`experiments/iter10_estimator-swap-replace-the-singl_v1/`.

Do not treat a LinkedIn round-number as the archival figure. Use the JSON
below.

---

## 1. Headline numbers

All metrics are **fold-mean, one fit per held-out fold**. Label 0 = ASD, 1 =
control. Sensitivity = ASD recall. Balanced accuracy = (ASD recall + control
recall) / 2.

| Split | Folds | Accuracy | Balanced accuracy | AUC | ASD recall | Control recall |
|---|---|---|---|---|---|---|
| Leave-one-site-out | 33 | 71.9% ± 9.6 | **72.8% ± 9.4** | **0.797 ± 0.098** | 74.5% | 71.2% |
| Leave-one-institution-out | 23 | **70.1% ± 8.5** | **70.8% ± 8.1** | **0.779 ± 0.087** | 70.8% | 70.9% |

Source: `experiments/iter10_estimator-swap-replace-the-singl_v1/results.json`
(`summary` and `site_*` fields from the institution-stage run). Evaluated
2026-10-04. Runtime ~17 min for the 23 institution folds (2 workers).

The institution split is the number to lead with. Several ABIDE I and II
cohorts share a scanner; leave-one-site-out still trains on that scanner.

---

## 2. What this snapshot is

A linear classifier on resting-state functional connectivity (FC), trained
and tested on DCAN-preprocessed ABIDE I + II.

**Recipe in `neuroasd/model.py` (this tag):**

1. Tangent-space FC at a log-Euclidean reference estimated on the **training
   fold only** (shrinkage 0.1).
2. Gordon (352 parcels) and HCP2016 (379 parcels) tangent blocks concatenated
   column-wise. **Not** a meta-learner / stacking ensemble. Power and Markov
   are built and aligned but unused.
3. Two covariate columns, training-fold standardized: age z-score and binary
   sex. FIQ and head motion are **not** used as features.
4. One L2 logistic regression per fold, **fixed** `C = 1e-2` (no inner CV, no
   test-fold hyperparameter choice).
5. Decision threshold: maximize balanced accuracy on grouped out-of-fold
   **training** scores; freeze it before the test fold is scored.
6. Label-free per-group z-scoring of decision values (each site/institution
   uses only its own unlabeled mean and SD). This transform is monotone
   inside a group, so it **cannot change AUC**.

Protocol rules that must stay in a methods section:

- Every fitted quantity (tangent reference, feature scaling, threshold) is
  computed inside the training fold.
- One fit per fold. No best-epoch, no test-site selection, no feature
  selection on the full dataset.
- Single-class sites after filtering (`ABIDEII-KUL_3`, `ABIDEII-NYU_2`) are
  training-only under leave-one-site-out; they are included in the matching
  institution fold.

---

## 3. Dataset (methods paragraph)

- **Source:** DCAN (ABCD-HCP) derivatives of ABIDE I and II, Gordon atlas
  unless noted. Same 1,443 subjects on HCP / Power / Markov.
- **N = 1,443** (631 ASD / 812 control), 35 sites, 23 institutions.
- Frame censoring at FD > 0.2 mm; keep subjects with ≥ 180 s retained data.
- **Duplicates removed:** 11 ABIDE II re-releases of ABIDE I scans
  (`data/abide_dcan/duplicate_scans.csv`).
- **Incomplete coverage removed:** 16 subjects with an unscanned parcel in
  some atlas (`data/abide_dcan/incomplete_coverage.csv`), so missingness
  cannot become a label shortcut.
- ETHZ_1 (37 subjects) is excluded; labels are only on NITRC.
- Full accounting: `data/abide_dcan/README.md`.

This is larger than Abraham 2017 (871, ABIDE I only) and smaller than IMPAC
(2,117, ABIDE I+II plus an unpublished hospital site).

---

## 4. Internal comparison (the claim against our own baseline)

Formal baseline, same subjects, same splits, tag `linear-baseline-dcan-v1`
(`experiments/linear_baseline_dcan_v1/`): Gordon tangent + kernel ridge,
fixed `score > 0` cut, no covariates, no second atlas.

| Split | Model | Accuracy | Balanced accuracy | AUC | ASD recall |
|---|---|---|---|---|---|
| Site | Gordon tangent + ridge (baseline) | 68.6% | 67.4% | 0.775 | 54.3% |
| Site | **This snapshot** | **71.9%** | **72.8%** | **0.797** | **74.5%** |
| Institution | Gordon tangent + ridge (baseline) | 67.0% | 67.5% | 0.770 | 55.1% |
| Institution | **This snapshot** | **70.1%** | **70.8%** | **0.779** | **70.8%** |

Institution-out: **+3.1 pp accuracy**, **+3.4 pp balanced accuracy**,
**+0.009 AUC**, **+15.7 pp ASD recall**.

Paired per-fold tests against the baseline were **not** run for this
snapshot. Do not write “significant” until that is done. The baseline README
already warns that site-to-site SD is much larger than subsample-refit SD.

### What actually moved the number

The loop added one mechanism at a time. Institution-out lineage:

| Step | Mechanism | Acc | Bal. acc | AUC | ASD rec. | Control rec. |
|---|---|---|---|---|---|---|
| Baseline | Gordon tangent + ridge, cut at 0 | 67.0% | 67.5% | 0.770 | 55.1% | 79.8% |
| iter2 | Training-only balanced-accuracy threshold | 66.6% | 68.9% | 0.770 | 77.5% | 60.4% |
| iter4 | Label-free per-group score z-scoring | 68.3% | 69.3% | 0.770 | 71.8% | 66.8% |
| iter5 | Age and sex columns | 69.4% | 70.3% | 0.770 | 71.2% | 69.3% |
| iter6 | Concatenate HCP tangent with Gordon | 70.1% | 70.8% | 0.776 | 71.5% | 70.2% |
| **iter10 (this tag)** | Swap ridge for L2 logistic, C=1e-2 | **70.1%** | **70.8%** | **0.779** | **70.8%** | **70.9%** |

Read this table before writing a “we invented X” sentence:

- **Threshold + calibration** (iter2, iter4) rebalance the operating point.
  AUC is unchanged at 0.770. Most of the ASD-recall gain is here. A paper
  that reports only accuracy/balanced accuracy without AUC will overstate
  new connectome signal.
- **Age/sex** (iter5) still does not move institution AUC off 0.770. Treat
  as operating-point / covariate adjustment unless a later analysis shows
  otherwise.
- **Gordon + HCP concatenation** (iter6) is the first step that moves AUC
  (0.770 → 0.776). That is the main *ranking* content change in this stack.
- **Ridge → logistic** (iter10) is a small AUC bump (0.776 → 0.779) with
  flat balanced accuracy versus iter6. The trial note says a harness PASS
  can come from the carried-over site split alone. Do **not** sell the
  estimator swap as the result. The snapshot is the *stacked recipe*, not
  proof that logistic beat ridge.

A defensible one-sentence model claim:

> Under leave-one-institution-out on 1,443 DCAN ABIDE I+II subjects, a
> training-fold-only tangent embedding of two atlases (Gordon + HCP) with
> L2 logistic regression, a training-only decision threshold, and
> label-free per-site score calibration reaches 70.1% accuracy and AUC
> 0.78, versus 67.0% / 0.77 for Gordon tangent + ridge on the same splits.

---

## 5. External comparison (what to put next to other papers)

Split the table. Mixing random-split 70% with LOSO 67% is how this
literature inflates.

### 5.1 Strict cross-site (the fair column)

| Paper | Data | N | Validation | Method | Accuracy | AUC |
|---|---|---|---|---|---|---|
| Nielsen 2013 | ABIDE I | 964 | Leave-one-subject-out | Weighted connections | 60.0% | – |
| Abraham 2017 | ABIDE I | 871 | LOSO | MSDL + tangent + L2 linear | 66.8% | – |
| Heinsfeld 2018 | ABIDE I | 1,035 | LOSO | Autoencoder + MLP | 65% | – |
| Harmonization 2023 | ABIDE I+II, 9 large sites | – | LOSO | ComBat/CovBat + DL | 60.7–62.7% | – |
| Traut 2022 (IMPAC) | ABIDE I+II+RDB | 2,117 | One unseen site (RDB) / EU-AIMS | Tangent + linear ensemble | – | **0.81 / 0.72** |
| **This snapshot** | ABIDE I+II, DCAN | 1,443 | **Leave-one-institution-out (23)** | Tangent, 2-atlas concat, L2 logistic | **70.1%** | **0.779** |
| **This snapshot** | same | 1,443 | **Leave-one-site-out (33)** | same | **71.9%** | **0.797** |

How to talk about this in a paper:

- Versus Abraham 2017: higher accuracy (70.1% vs 66.8%) on a larger, mixed
  I+II set, with a **stricter** institution hold-out. Not the same
  preprocessing or atlas. Do not call it a reproduction.
- Versus Heinsfeld LOSO 65%: same caveat, plus Heinsfeld’s per-site band is
  unusually tight; cite carefully.
- Versus Harmonization 2023 (60.7–62.7% LOSO on large ABIDE I+II sites): we
  are higher, with a linear model, which matches that paper’s finding that
  harmonization barely helps once the baseline already ignores site.
- Versus IMPAC: **do not claim we beat IMPAC.** IMPAC’s 0.80 overall AUC is
  mostly same-site subjects in a hidden test set. Their honest unseen-site
  number is AUC 0.81 on one hospital and **0.72 on EU-AIMS**. We have no
  EU-AIMS evaluation. Our 0.779 is 23 held-out institutions, which is a
  different and in some ways harder exam than one unseen site, but it is
  still ABIDE, not an external cohort.
- There is **no published 80% accuracy** under honest leave-one-site-out.
  Do not use 80% as a literature SOTA.

### 5.2 Random or site-stratified splits (do not compete on this row)

| Paper | Validation | Accuracy | AUC |
|---|---|---|---|
| Heinsfeld 2018 | 10-fold | 70% | – |
| Parisot 2018 | Stratified 10-fold, transductive population graph | 70.4% | 0.75 |
| Kan 2022 (BrainNetTF) | Site-stratified 70/10/20 | 71.0% | 0.80 |
| Dong 2025 | Leakage-free 5-fold | 58–72% (vote 72.2%) | 0.64–0.78 |
| BrainMass 2024 | Site-stratified 70/15/15 | 72.8% | – |
| Brain Harmony 2025 (independent ABIDE I) | Site-stratified 6:2:2 | 63.1% | – |

Talking point: we **match the 70–73% headline numbers** that GNN /
Transformer / foundation-model papers report, but under a protocol those
papers mostly avoid. Independently reproduced foundation models (Brain
Harmony) sit at 60–67% even on random splits. Dong 2025 already showed that
once leakage is removed, architectures collapse to ~70% on ABIDE I.

---

## 6. What is actually new (candidate contributions)

Write these as contributions only if the paper still uses this protocol.

1. **Institution-out ABIDE I+II, not just site-out.** Same-scanner cohorts
   in both releases make leave-one-site-out leak the test scanner. KKI and
   Leuven already showed this on the baseline: accuracy collapsed 17–27 pp
   while AUC stayed 0.72–0.80. The snapshot reports both splits.
2. **Dataset hygiene as a methods result.** Duplicate I/II scans and
   unscanned parcels were removed so they cannot act as labels. Subject
   identity is locked across four atlases.
3. **Honest linear recipe on DCAN I+II.** Tangent + strong L2 linear is the
   IMPAC winning class. We add a training-only threshold (ASD recall was
   55% with a cut at 0) and label-free per-group calibration for the
   KKI/Leuven threshold shift, without using test labels.
4. **Two-atlas concatenation as the ranking gain**, not deep models. The
   GCN on this dataset overfits. The stacked linear model is the result.
5. **A gated search loop that cannot pick the test fold.** Useful as a
   methods footnote if the paper is about automated search; drop it if the
   paper is only about the classifier.

### What this is *not*

- Not multi-atlas **stacking** (one model per atlas + logistic
  meta-learner). Concatenation is a different mechanism.
- Not an ensemble of feature types (Fisher-z vs tangent) and not a GNN
  member.
- Not structural MRI fusion.
- Not a claim of 80% accuracy.
- Not a claim that logistic regression beat ridge. iter10 ≈ iter6.

---

## 7. Caveats to keep in the paper

- Fold-mean accuracy on small sites is noisy (institution SD ~8–9 pp).
- ABIDE II sites are harder than ABIDE I on the baseline (64.9% vs 70.9%
  site-out). Report both or the I+II mix will look like a free lunch.
- One phenotype error is documented (sex disagrees for 0050810 / 29355).
- Per-group z-scoring uses the test group’s **unlabeled** score mean/SD.
  That is not a label leak, but it *is* a test-group statistic. A reviewer
  may call it transductive calibration. If that objection matters, run an
  ablation that applies only the training-derived threshold with no
  test-group scaling, and report AUC (which this step cannot change)
  alongside accuracy. Done for release `iter10-model-v1` (per-subject scoring,
  cut at 0): see [`report_iter10.md`](report_iter10.md).
- No external cohort (EU-AIMS LEAP, etc.).
- No paired Wilcoxon versus the linear baseline in this snapshot.
- `main` does not contain this model. Reproduce from the tag, not from
  `main`.

---

## 8. Related work to cite (minimum set)

Abraham et al., *NeuroImage* 2017 (tangent + LOSO). Traut et al.,
*NeuroImage* 2022 (IMPAC). Heinsfeld et al., *NeuroImage: Clinical* 2018
(10-fold vs LOSO gap). Dong et al., *Hum Brain Mapp* 2025 (leakage-free
~70%). Holiga et al., *Sci Transl Med* 2019 (replicable FC group effects,
large overlap with controls — why ~70% may be a biological ceiling).
Rosenblatt et al., *Nat Commun* 2024 (feature leakage). Di Martino et al.,
*Sci Data* 2017 (ABIDE II). Literature file: `doc/literature_review_asd_fmri.md`.

---

## 9. Reproduce this tag

```bash
git checkout iter10-loso-institution-v1

# Dataset must already exist (gitignored FC matrices).
uv run python -m autoresearch.trial \
  --name iter10_estimator-swap-replace-the-singl \
  --stage loso-institution
```

Scored function: `neuroasd.model.fit_predict`. Frozen scores and the exact
run note: `experiments/iter10_estimator-swap-replace-the-singl_v1/`.

Checkout `main` again when you are done; later ensemble work should not
edit this tag.
