# Autoresearch Program

Instructions for an AI agent running experiments on this repository. Read this file
completely before making any change.

---

## 1. Objective

Improve ASD vs control classification on the DCAN ABIDE I + II dataset
(`data/abide_dcan`, 1,443 subjects, Gordon + HCP / Power / Markov atlases).

**Starting point:** the tagged iter10 recipe in `neuroasd.model.fit_predict`
(`iter10-loso-institution-v1`, PR #18), not the old GCN PRs and not the
original Gordon-only ridge baseline.

| Split | Model | Accuracy | Balanced accuracy | AUC | ASD recall |
|---|---|---|---|---|---|
| Leave-one-site-out (33) | iter10 (incumbent) | 71.9% | 72.82% | 0.797 | 74.5% |
| Leave-one-institution-out (23) | iter10 (incumbent) | 70.1% | 70.84% | 0.779 | 70.8% |
| Site / institution | Gordon tangent + ridge (historical) | 68.6% / 67.0% | 67.35% / 67.47% | 0.775 / 0.770 | 54.3% / 55.1% |

Progress is by small steps on top of the current **base** (iter10 at first).
A trial is **accepted** if one split gains ≥ 0.3 pp balanced accuracy over the
base and the other drops by at most 0.5 pp; its code becomes the new base. A
review PR opens only when an accepted trial is also ≥ +1 pp over the last PR'd
model (iter10 until the first PR). Beating the 2026-10-03 ridge baseline does
not count. The original baseline stays at tag `linear-baseline-dcan-v1`.

**Primary metric:** fold-mean balanced accuracy
`(ASD recall + control recall) / 2`. Accuracy and AUC are always reported so
results can be compared with the literature (those papers mostly report
accuracy, then AUC; almost none report balanced accuracy).

Loop stop: leave-one-institution-out accuracy_mean ≥ 80%.

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

---

## 3. Evaluation stages

Trials are cheap-to-expensive. Do not skip ahead.

| Stage | What it runs | Gate |
|-------|--------------|------|
| `screen` | Leave-one-site-out on 6 mixed sites (NYU, UM_1, USM, ABIDEII-OHSU_1, ABIDEII-NYU_1, ABIDEII-SDSU_1) | balanced accuracy ≥ 0.70 |
| `loso-site` | Leave-one-site-out on all 33 sites with both classes | balanced accuracy ≥ 0.71 and ≥ base site − 0.5 pp (below that the final stage cannot pass) |
| `loso-institution` | Leave-one-institution-out on 23 institutions (final) | accept: either split ≥ +0.3 pp vs the **base**, other drops ≤ 0.5 pp. PR: also ≥ +1 pp vs the last PR (iter10: site 0.7282 / institution 0.7084) |

Thresholds live in `autoresearch/gates.json`. `trial.py` exits `0` on PASS and `3` on
FAIL.

A `loso-institution` PASS opens a review PR. The PR body lists balanced accuracy,
accuracy, AUC, ASD recall, and control recall for both splits.

---

## 4. Branch policy

```
main                            stable code + published results + this framework
experiment/trial-<slug>         one trial, opened for review after a win
```

The loop never merges to `main` and never edits `gates.json`. After a full PASS
it keeps the winning code and searches for the next mechanism until institution
accuracy reaches 80%.

Failed trials leave no remote branch. Every run appends to
`outputs/autoresearch/ledger.jsonl`. **Read the ledger before proposing a new
config** so the same dead end is not explored twice. The old ABIDE I GCN ledger
is archived under `outputs/autoresearch/archive/`.

---

## 5. Search space

The harness owns data and metrics. You only edit `fit_predict` (and helpers).
`train` / `test` expose: `fc(atlas)` for `gordon`, `hcp`, `power`, `markov`;
`age`; `sex`; `groups`; `sites`; and `train.y`. Test labels are hidden.

The incumbent already has: a training-only balanced-accuracy threshold,
label-free per-group score z-scoring, age/sex columns, Gordon+HCP
**concatenation**, and L2 logistic at C=1e-2. Do not retry those.

**Concatenation is not stacking.** Concatenation = one classifier on columns
glued together (iter6). Stacking = one classifier per atlas (or feature type),
then a logistic meta-learner on grouped **out-of-fold** member scores inside
the training fold (`neuroasd.ensemble.stack_scores`). Score averaging
(`average_scores`) is a third, label-free rule.

Helpers: `neuroasd.ensemble` (`member_scores`, `stack_scores`, `average_scores`,
`GCNMember` / `gnn_member_scores`). Keep the incumbent threshold and
calibration unless that is the one mechanism you are replacing.

Ideas worth testing, in order:

1. **True stacking.** One linear member per atlas (start with Gordon+HCP, then
   add Power or Markov as its own member). Combine with `stack_scores`. Do not
   concatenate the new atlas onto the existing feature block.
2. **Score averaging** of the same members. Different mechanism from stacking.
3. **A Fisher-z member** alongside tangent members, combined by stacking or
   averaging — not by concatenating Fisher-z columns onto tangent columns in
   the same trial.
4. **Optional GNN member** (`GCNMember`, Power atlas) as one extra score
   column. Do not drop the linear members. Do not train a GCN as the only model.

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

When reporting to the user, lead with the outcome: what was tried, balanced
accuracy, accuracy, AUC, ASD recall, and whether it beat the linear baseline.
Include the margin, not just PASS/FAIL. Never claim an improvement from a
`screen` result alone — it is a filter, not evidence.
