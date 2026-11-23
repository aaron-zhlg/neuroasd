# Autoresearch Program

Instructions for an AI agent running experiments on this repository. Read this file
completely before making any change.

---

## 1. Objective

**Raise ASD recall (sensitivity) on unseen sites**, on the DCAN ABIDE I + II
dataset (`data/abide_dcan`, 1,443 subjects, Gordon + HCP / Power / Markov
atlases), scored **per subject** the way the cross-site papers score it.

**Starting point:** the iter10 recipe in `neuroasd.model.fit_predict`
(`iter10-loso-institution-v1`, PR #18) scored per subject: Gordon+HCP tangent
concatenation, age/sex, L2 logistic at C=1e-2, raw decision value cut at 0.
iter10's per-group calibration is removed (see rule 9).

| Split | Model | ASD recall | Control recall | Balanced accuracy | AUC |
|---|---|---|---|---|---|
| Screen (6 sites) | base (iter10 per subject) | 57.1% | 85.0% | 71.1% | 0.830 |
| Leave-one-site-out (33) | base (iter10 per subject) | 54.7% | 81.9% | 68.3% | 0.797 |
| Leave-one-institution-out (23) | base (iter10 per subject) | 55.0% | 80.4% | 67.7% | 0.779 |
| Cross-site literature, ABIDE I | Heinsfeld 2018 / Abraham 2017 | 69% / 61% | 62% / 72% | – | – |

The base calls too few subjects ASD. Progress is by small steps on top of the
current **base**. A trial is **accepted** if one split gains ≥ 0.3 pp ASD
recall over the base, the other drops by at most 0.5 pp, **and control recall
is ≥ 60% on both splits**; its code becomes the new base. A review PR opens
only when an accepted trial is also ≥ +1 pp ASD recall over the last PR'd model.

**Primary metric:** fold-mean ASD recall (`sensitivity_mean`).
**Floor:** fold-mean control recall (`specificity_mean`) ≥ 0.60 at every stage,
so calling everyone ASD cannot win. Balanced accuracy, accuracy and AUC are
always reported. Moving the cut alone trades control recall for ASD recall and
stops at the floor; lasting gains need a better ranking (higher AUC).

No ceiling: every accepted step becomes the new base, so ASD recall only goes
up. The loop runs until the operator stops it (`--target-recall X` sets an
optional stop).

---

## 2. Non-negotiable rules

1. **Never commit to `main`.** Every trial runs on its own branch.
2. **Never modify `experiments/`.** Those directories are frozen published results.
   New results go in a new directory only after a `loso-institution` trial passes.
3. **Never modify `data/`.** The subject list is fixed (1,443 people).
4. **Change one thing at a time.** A trial that varies four ideas at once
   teaches nothing about which one mattered.
5. **Gate on one fit per fold.** `autoresearch/trial.py` reports these by design;
   do not add best-epoch or test-fold selection.
6. **Do not stack a new idea on a failed diff.** After a FAIL the loop archives
   the diff and restores coder-writable files to the current base, the last
   accepted loso-institution trial (or HEAD if there is none). The rejected mechanism stays in `workspace.ruled_out`.
7. **Do not tune against `loso-institution`.** It is the confirmation stage, not a
   search signal.
8. **Do not use FIQ or head motion as predictive features.** They differ by
   diagnosis and would inflate results. Age and sex are allowed.
9. **Score each test subject on its own.** A test subject's score and label may
   depend only on the training split and that subject's own data. No z-scoring,
   ranking, harmonization (ComBat etc.), clustering or thresholding across the
   test subjects. iter10's per-group calibration did this and lifted ASD recall
   from 55% to 75% with information the papers do not use. The linter scores the
   test set whole, in halves and one subject at a time, and FAILs any change.

---

## 3. Evaluation stages

Trials are cheap-to-expensive. Do not skip ahead.

| Stage | What it runs | Gate |
|-------|--------------|------|
| `screen` | Leave-one-site-out on 6 mixed sites (NYU, UM_1, USM, ABIDEII-OHSU_1, ABIDEII-NYU_1, ABIDEII-SDSU_1) | ASD recall ≥ 0.54, control recall ≥ 0.60 |
| `loso-site` | Leave-one-site-out on all 33 sites with both classes | ASD recall ≥ base site − 0.5 pp (below that the final stage cannot pass), control recall ≥ 0.60 |
| `loso-institution` | Leave-one-institution-out on 23 institutions (final) | accept: ASD recall on either split ≥ +0.3 pp vs the **base**, other drops ≤ 0.5 pp, control recall ≥ 0.60 on both. PR: also ≥ +1 pp vs the last PR (base: site 0.5466 / institution 0.5504) |

Thresholds live in `autoresearch/gates.json`. `trial.py` exits `0` on PASS and `3` on
FAIL.

A `loso-institution` PASS opens a review PR. The PR body lists ASD recall,
control recall, balanced accuracy, accuracy and AUC for both splits.

---

## 4. Branch policy

```
main                            stable code + published results + this framework
experiment/trial-<slug>         one trial, opened for review after a win
```

The loop never merges to `main` and never edits `gates.json`. After a full PASS
it keeps the winning code and searches for the next mechanism until institution
the operator stops it.

Failed trials leave no remote branch. Every run appends to
`outputs/autoresearch/ledger.jsonl`. **Read the ledger before proposing a new
config** so the same dead end is not explored twice. The old ABIDE I GCN ledger
is archived under `outputs/autoresearch/archive/`.

---

## 5. Search space

The harness owns data and metrics. You only edit `fit_predict` (and helpers).
`train` / `test` expose: `fc(atlas)` for `gordon`, `hcp`, `power`, `markov`;
`age`; `sex`; `groups`; `sites`; and `train.y`. Test labels are hidden.

The base has: age/sex columns, Gordon+HCP **concatenation**, L2 logistic at
C=1e-2, and the raw score cut at 0. It has **no** training-only threshold and
**no** per-group calibration (forbidden by rule 9).

**Concatenation is not stacking.** Concatenation = one classifier on columns
glued together (iter6). Stacking = one classifier per atlas (or feature type),
then a logistic meta-learner on grouped **out-of-fold** member scores inside
the training fold (`neuroasd.ensemble.stack_scores`). Score averaging
(`average_scores`) is a third, label-free rule.

Helpers: `neuroasd.ensemble` (`member_scores`, `stack_scores`, `average_scores`,
`GCNMember` / `gnn_member_scores`).

Tried on iter10 under the OLD goal (2026-10-15, balanced accuracy with
per-group calibration; all below iter10 there): stacking Power+Markov members,
Gordon/HCP score averaging, a four-atlas member bank, a Fisher-z member, a Power
GCN member, per-site feature standardization, Power graph-spectral features,
swapping HCP for Power or Markov, and L1 logistic (too slow: screen timed out).
That verdict does not carry over to per-subject ASD recall, so they may be
retried. Only this session's `workspace.ruled_out` is binding.

Any model family is allowed (linear, kernel, trees, MLP, GNN through
`neuroasd.ensemble`), as long as the folds finish inside the stage time limits
(screen 40 min, loso-site and loso-institution 120 min each).

Untested directions (all use only FC + age/sex, fit inside the training fold):

0. **A cut chosen on the training fold**: out-of-fold training scores (grouped
   like the outer split), cut where ASD recall is highest with control recall
   still ≥ 60%, or at balanced accuracy; or class weights / prior correction.
   Cheap, but only moves along the ROC curve.
1. **Regularization chosen by nested CV.** Pick C by inner grouped CV on the
   training fold instead of the fixed C=1e-2.
2. **Dimensionality reduction** before the classifier: PCA or univariate
   feature selection fit on training rows only.
3. **Better FC estimates**: shrinkage (e.g. Ledoit-Wolf-style) on the FC
   before the tangent map, or a different tangent reference mean.
4. **Partial correlation** derived from FC as a new feature type.
5. **Sample weighting**: e.g. site-balanced weights so large sites do not
   dominate the fit.
6. **A nonlinear classifier** on a reduced tangent space (kernel SVM, small
   MLP).

Accepted small steps can be refined: after an accept, tuning or extending the
same mechanism is a valid next trial.

Do not start from the old GCN trial PRs (#2–#13). Those were fit on C-PAC
ABIDE I (884 subjects, 111 ROIs) and are below this incumbent.

---

## 6. Trial checklist

Before running:

- [ ] Read `outputs/autoresearch/ledger.jsonl`; confirm this config is new.
- [ ] State a one-line hypothesis and pass it via `--note`.
- [ ] Confirm exactly one thing differs from the last win (or from
      `neuroasd/model.py` if there is no win yet).

After running:

- [ ] Record the outcome, including failures, with the observed margin.
- [ ] On FAIL, say what the result rules out — that is the useful output.
- [ ] On PASS at `screen`, promote to `loso-site`; on PASS there, promote to
      `loso-institution`.
- [ ] On PASS at `loso-institution`, create `experiments/<name>_v1/` and open
      the branch for review. Do not merge.

---

## 7. Commands

```bash
# Single trial, screening stage
./autoresearch/run_trial.sh --name class-weight --stage screen -- --note "class-weighted ridge"

# Promote a promising config
./autoresearch/run_trial.sh --name class-weight --stage loso-site -- --note "class-weighted ridge"

# Run the trial directly, without branch management
uv run python -m autoresearch.trial --name class-weight --stage screen --note "class-weighted ridge"

# Multi-agent write → lint → trial loop (needs an LLM key)
uv run python -m autoresearch.loop.check_loop
uv run python -m autoresearch.loop
```

---

## 8. Reporting

When reporting to the user, lead with the outcome: what was tried, ASD recall,
control recall, balanced accuracy, AUC, and whether it beat the base.
Include the margin, not just PASS/FAIL. Never claim an improvement from a
`screen` result alone — it is a filter, not evidence.
