# ABIDE I + II from the DCAN derivatives

Functional connectivity for ABIDE I and ABIDE II, built from the parcellated time series that the DCAN (ABCD-HCP) pipeline publishes on the public FCP-INDI S3 bucket. Both releases went through the same pipeline, so they can be pooled without mixing preprocessing streams. No local fMRI preprocessing is needed.

```text
s3://fcp-indi/data/Projects/ABIDE/Derivatives/DCAN/sub-00<SUB_ID>/ses-<SITE>/func/
s3://fcp-indi/data/Projects/ABIDE2/Derivatives/DCAN/sub-<SUB_ID>/ses-<SITE>baseline/func/
```

## Build

```bash
uv run data/abide_dcan/scripts/build_dcan_dataset.py                    # all subjects, Gordon atlas
uv run data/abide_dcan/scripts/build_dcan_dataset.py --sites CALTECH    # smoke test
uv run data/abide_dcan/scripts/build_dcan_dataset.py --manifest-only --min-seconds 240
uv run data/abide_dcan/scripts/build_dcan_dataset.py --atlas HCP2016FreeSurferSubcortical --out-dir data/abide_dcan_hcp
uv run data/abide_dcan/scripts/build_dcan_dataset.py --atlas Power2011FreeSurferSubcortical --out-dir data/abide_dcan_power
uv run data/abide_dcan/scripts/build_dcan_dataset.py --atlas Markov2012FreeSurferSubcortical --out-dir data/abide_dcan_markov
```

The script declares its own dependencies (numpy, scipy, nibabel) inline, so `uv run` needs no project changes. Downloads are resumable. Generated files (`raw/`, `processed/`) are git-ignored; a full build takes about 6 minutes and about 1 GB per atlas. `--manifest-only` re-applies the filters to the cached FC without downloading.

## Processing

For each subject:

1. Pick one session, preferring the baseline session.
2. Download one atlas `ptseries` and the DCAN motion mask.
3. Drop frames flagged at FD > 0.2 mm, using the mask's own frame-removal vector.
4. Compute Pearson FC on the retained frames.
5. Save `processed/fc/<FILE_ID>_fc.npy` and delete the time series.

Then the subject filters below are applied when the manifest is written.

Outputs (same layout as `data/abide/processed`, so the training code works with `--data-dir data/abide_dcan`):

- `processed/manifest.csv` with `FILE_ID`, `SITE_ID`, `DATASET`, `label` (0 = ASD, 1 = control), age, sex, FIQ, retained mean FD, and retained seconds;
- `processed/summary.json` with per-site counts and the exclusion breakdown;
- `processed/duplicate_candidates.csv` with unreviewed possible duplicate scans (should be empty);
- `processed/parcels.txt` with parcel names.

## Defaults and exclusions

| Item | Value |
|---|---|
| Atlas | `Gordon2014FreeSurferSubcortical`: 333 cortical + 19 subcortical = 352 parcels |
| Frame censoring | FD > 0.2 mm |
| Minimum retained data | 180 s (change with `--min-seconds`, no re-download) |
| Labels, ABIDE I | `data/abide/phenotypic/Phenotypic_V1_0b_preprocessed1.csv` |
| Labels, ABIDE II | per-site `participants.tsv` from `ABIDE2/RawData` |
| Excluded: `ABIDEII-ETHZ_1` | no `participants.tsv` on S3; labels are only on NITRC |
| Excluded: `ABIDEII-UCLA_Long`, `ABIDEII-UPSM_Long` | follow-up rescans of ABIDE I participants (same subject IDs) |
| Excluded: fewer than 10 frames after censoring | `too_much_motion` |
| Excluded: re-released scans | `duplicate_scan`; ABIDE II copies listed in `duplicate_scans.csv` |
| Excluded: unscanned parcels | `incomplete_coverage`; subjects listed in `incomplete_coverage.csv` |

### Duplicate scans across releases

Some ABIDE II cohorts re-release ABIDE I scans under new subject IDs. Left in, they leak across folds: when ABIDE I KKI is the test site, copies of its subjects sit in ABIDE II KKI_1 in the training set.

Three independent signals were checked on every ABIDE I / ABIDE II pair from the same institution:

- **FC similarity:** Fisher-z FC correlation on the Gordon atlas. Different people correlate at 0.22 (median), and 99.9% of same-institution pairs fall below 0.45.
- **Motion trace:** identical DCAN motion summaries (total frames, retained frames, and retained mean FD).
- **Phenotype:** same age (within 0.02 years), sex, and diagnosis. This is weak on its own, because many sites record age as whole years.

Eleven pairs meet at least two signals. They are listed with their evidence in `duplicate_scans.csv`: 10 in KKI / KKI_1 and 1 in USM / USM_1. After them there is a clear gap: no other phenotype-matched pair exceeds FC r = 0.43 or shares a motion trace. Eight of the 11 pairs would otherwise pass the other filters; the remaining 3 fail the 180 s filter anyway, but stay listed in case that threshold is lowered.

The build keeps the ABIDE I copy and drops the ABIDE II one. Neither signal alone catches all pairs:

- 0050781 / 29354 has an identical motion trace and identical age, sex, and FIQ, but FC r is only 0.54.
- 0050790 / 29364 and 0050509 / 29506 have FC r of 0.82 and 0.94, but different motion summaries.

The sex field disagrees for 0050810 / 29355, so at least one release has phenotype errors.

On every build the script still flags new cross-site pairs that are not on the list, if they have FC r > `--duplicate-r` (0.7) or an identical motion trace. It writes them to `processed/duplicate_candidates.csv` for review but never drops them automatically: on coarse atlases (Power, Markov), unrelated people from the same site can exceed 0.7. Pairs within one site are ignored because they always share a fold. Three such within-site pairs in ABIDE I share a motion trace without similar FC: KKI 0050805 / 0050808, UM_1 0050279 / 0050286, and YALE 0050578 / 0050602.

### Unscanned parcels

Sixteen subjects have at least one parcel whose time series is constant in at least one atlas, which means the region was outside the field of view or lost to signal dropout. These are mostly unassigned ("None") ventral temporal and orbitofrontal parcels. Their FC rows are undefined, and any fill-in value makes them outliers in tangent space.

The subjects are concentrated in CMU (9 of 18) and ABIDE II NYU_2 (4), and 11 of the 16 are ASD, so the missing pattern could act as a site and label shortcut. They are therefore excluded from every atlas, which keeps the subject set identical across atlases. They are listed with per-atlas counts in `incomplete_coverage.csv`. The build also drops any subject with a zero-variance parcel that is not on the list, and warns that the list needs updating.

### Same-institution cohorts

Several institutions contributed separate cohorts to both releases: KKI, NYU, OHSU, OLIN/ONRC, SDSU, Trinity/TCD, UCLA, USM, and Leuven/KUL. UM also has two ABIDE I sub-sites. Even without duplicate people, leave-one-site-out then trains on the test site's scanner. `neuroasd.linear_baseline` therefore also reports leave-one-institution-out.

## Counts

Default build (2026-10-03): 2,119 candidates and 1,443 kept (631 ASD / 812 control; 789 ABIDE I, 654 ABIDE II; 35 sites). Dropped:

| Reason | Subjects |
|---|---|
| Below 180 s after censoring | 469 |
| No DCAN data | 100 |
| Too much motion | 83 |
| Incomplete coverage | 16 |
| Duplicate scan | 8 |

The HCP, Power, and Markov builds keep exactly the same 1,443 subjects.

Two sites have only ASD participants after filtering: `ABIDEII-KUL_3` and `ABIDEII-NYU_2`. Use them for training only, because AUC is undefined on them. After the coverage exclusion, CMU has 9 subjects (4 ASD / 5 control).

| Atlas | Parcels | Directory |
|---|---|---|
| `Gordon2014FreeSurferSubcortical` | 352 | `data/abide_dcan` |
| `HCP2016FreeSurferSubcortical` | 379 | `data/abide_dcan_hcp` |
| `Markov2012FreeSurferSubcortical` | 205 | `data/abide_dcan_markov` |
| `Power2011FreeSurferSubcortical` | 94 | `data/abide_dcan_power` |
