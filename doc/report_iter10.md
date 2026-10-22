# iter10 report: ASD classification from resting-state fMRI across unseen institutions

Release [`iter10-model-v1`](https://github.com/aaron-zhlg/neuroasd/releases/tag/iter10-model-v1),
branch `experiment/trial-iter10_estimator-swap-replace-the-singl`.
Model details and the per-step lineage: [`paper_notes_iter10.md`](paper_notes_iter10.md).

## Summary

On 1,443 DCAN-preprocessed ABIDE I + II subjects, a linear classifier on two-atlas tangent-space
connectivity reaches **70.1% accuracy and AUC 0.78 under leave-one-institution-out** (23
institutions, with label-free per-site calibration). This protocol is stricter than both the
random or site-stratified splits used by most studies and the leave-one-site-out used by a few:
no scanner of the test institution is seen in training.

- **Same score, stricter test.** GNN, Transformer, and foundation-model studies report 70–73%
  accuracy (AUC up to 0.80) on random or site-stratified splits, where every test site is also in
  training. We reach a comparable 70.1% / AUC 0.78 on unseen institutions. Scoring each subject
  independently, the model reaches 69.4% / AUC 0.80 under leave-one-site-out.
- **On the same random splits**, scored per subject as those studies do, the model reaches
  **71–72% accuracy and AUC 0.80–0.81**.
- **Against cross-site studies**, under a stricter hold-out, accuracy is above Abraham 2017
  (66.8% leave-one-site-out) and AUC is within the range IMPAC reports on unseen cohorts
  (0.72–0.81).

## Model

- Tangent-space FC (log-Euclidean reference from training subjects only, shrinkage 0.1) on the
  Gordon (352 parcels) and HCP2016 (379 parcels) atlases, concatenated.
- Age and sex columns; all columns standardized with training statistics.
- L2 logistic regression, fixed `C = 1e-2` (no inner search, nothing chosen on a test fold).
- Two ways to turn scores into labels:
  - **per-subject**: each subject on its own, control if the decision value is > 0. This is how
    most papers score a classifier.
  - **per-group** (iter10 as released): each test site / institution is z-scored with its own
    *unlabeled* score mean and SD, then cut at a threshold learned on training folds. This needs
    a batch of subjects from the same site.

## Results

One fit per fold. Fold mean ± SD; k-fold rows are 10 folds × 3 reshuffles.

| Split | Scoring | Accuracy | Balanced acc. | AUC | ASD recall | Control recall |
|---|---|---|---|---|---|---|
| Leave-one-institution-out (23) | per-group | **70.1% ± 8.5** | **70.8% ± 8.1** | **0.779 ± 0.087** | 70.8% | 70.9% |
| Leave-one-institution-out (23) | per-subject | 67.4% ± 10.7 | 67.7% ± 9.4 | 0.779 ± 0.087 | 55.0% | 80.3% |
| Leave-one-site-out (33) | per-group | 71.9% ± 9.6 | 72.8% ± 9.4 | 0.797 ± 0.098 | 74.5% | 71.2% |
| Leave-one-site-out (33) | per-subject | 69.4% ± 9.6 | 68.3% ± 10.1 | 0.797 ± 0.098 | 54.7% | 81.9% |
| Site-stratified 10-fold | per-subject | 72.2% ± 3.4 | 70.3% ± 3.5 | 0.809 ± 0.035 | 54.6% | 85.9% |
| Random 10-fold | per-subject | 71.3% ± 3.6 | 69.4% ± 3.7 | 0.801 ± 0.032 | 53.6% | 85.1% |

Per-group scoring under random / stratified k-fold (a site has only 1–3 test subjects per fold,
so the per-site z-score is noise) is in
[`split_schemes.json`](../experiments/iter10_estimator-swap-replace-the-singl_v1/split_schemes.json);
it lowers AUC to 0.74–0.78 and should not be used for those splits.

## Comparison with published results

Split by protocol: random-split and cross-site numbers are not comparable.

**Cross-site (test site never seen in training)**

| Study | Data | N | Validation | Accuracy | AUC |
|---|---|---|---|---|---|
| Abraham 2017 | ABIDE I | 871 | LOSO | 66.8% | – |
| Heinsfeld 2018 | ABIDE I | 1,035 | LOSO | 65% | – |
| Harmonization 2023 | ABIDE I+II (9 sites) | – | LOSO | 60.7–62.7% | – |
| Traut 2022 (IMPAC) | ABIDE I+II + private | 2,117 | One unseen site / EU-AIMS | – | 0.81 / 0.72 |
| **iter10, per-subject** | ABIDE I+II (DCAN) | 1,443 | LOSO (33 sites) | **69.4%** | **0.797** |
| **iter10, per-group** | ABIDE I+II (DCAN) | 1,443 | Leave-one-institution-out (23) | **70.1%** | **0.779** |

**Random or site-stratified splits (sites shared between train and test)**

| Study | Validation | Accuracy | AUC |
|---|---|---|---|
| Heinsfeld 2018 | 10-fold | 70% | – |
| Parisot 2018 (GCN) | 10-fold, transductive | 70.4% | 0.75 |
| Kan 2022 (BrainNetTF) | Site-stratified split | 71.0% | 0.80 |
| BrainMass 2024 | Site-stratified split | 72.8% | – |
| **iter10, per-subject** | Random 10-fold | **71.3%** | **0.801** |
| **iter10, per-subject** | Site-stratified 10-fold | **72.2%** | **0.809** |

A linear model matches the deep-model headline numbers on random splits, and keeps ~67–70% when
whole institutions are held out.

## Same-data baseline

The fairest comparison is Abraham's recipe (tangent space + linear classifier, per-subject cut at
0) on our own data and splits, tag `linear-baseline-dcan-v1`:

| Split | Gordon tangent + ridge | iter10, per-subject | iter10, per-group |
|---|---|---|---|
| Leave-one-site-out | 68.6% / AUC 0.775 | 69.4% / 0.797 | 71.9% / 0.797 |
| Leave-one-institution-out | 67.0% / AUC 0.770 | 67.4% / 0.779 | 70.1% / 0.779 |

- Abraham's recipe already reaches 68.6% LOSO here, above the 66.8% in his paper; most of the
  gap to the literature comes from the data (DCAN preprocessing, ABIDE I+II, more subjects).
- The model change (second atlas, covariates, logistic regression) adds +0.4–0.9 pp accuracy and
  +0.01–0.02 AUC. No paired test has been run; fold SD is 8–10 pp.
- Per-group calibration adds a further +2.4–2.7 pp accuracy (+3.1–4.6 pp balanced accuracy).

## Finding: label-free per-site calibration

Under leave-one-group-out, z-scoring each test site's scores with its own unlabeled mean and SD
leaves AUC unchanged (the transform is monotone within a site) but moves the operating point:

| Split | Scoring | Balanced acc. | ASD recall | Control recall |
|---|---|---|---|---|
| Leave-one-site-out | per-subject | 68.3% | 54.7% | 81.9% |
| Leave-one-site-out | per-group | **72.8%** | **74.5%** | 71.2% |
| Leave-one-institution-out | per-subject | 67.7% | 55.0% | 80.3% |
| Leave-one-institution-out | per-group | **70.8%** | **70.8%** | 70.9% |

A cut at 0 misses about 45% of ASD subjects at unseen sites, because each site shifts the score
distribution. Calibrating per site, without labels, recovers balanced recall. The cost: the model
must score a batch of subjects from one site together; a single new subject cannot be calibrated.

## Limitations

- **Model selection on the evaluation splits.** iter10 was chosen by an automated loop that
  compared about ten variants on these same leave-one-site-out and leave-one-institution-out
  splits. Its scores on those splits are optimistic; the baseline was not selected this way.
- **No untouched test set.** No institution or external cohort (e.g. EU-AIMS LEAP) was held back
  from model selection.
- **Different data from the papers.** DCAN ABIDE I+II vs mostly C-PAC ABIDE I; the tables compare
  protocols, not identical experiments.
- **No significance tests** versus the baseline or the literature.
- **Per-group scoring is transductive.** It uses unlabeled statistics of the test site.

To support a stronger claim: freeze iter10, evaluate once on held-back institutions or an
external cohort, and run paired tests against the same-data baseline.

## Reproduce

Data: DCAN ABIDE I+II FC matrices (`data/abide_dcan`, `data/abide_dcan_hcp`; build with
`data/abide_dcan/scripts/build_dcan_dataset.py`).

```bash
# cross-validation, any split and scoring
uv run python -m scripts.train_model --data-root data --split institution --cv
uv run python -m scripts.train_model --data-root data --split site --cv --scoring per-subject
uv run python -m scripts.train_model --data-root data --split random --cv --folds 10 --repeats 3
uv run python -m scripts.train_model --data-root data --split stratified --cv --folds 10 --repeats 3

# train once on a split, save the model, then score it
uv run python -m scripts.train_model --data-root data --split institution --holdout NYU
uv run python -m scripts.test_model --data-root data --model outputs/models/institution_NYU/model.npz
```
