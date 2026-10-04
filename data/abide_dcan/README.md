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
```

The script declares its own dependencies (numpy, scipy, nibabel) inline, so `uv run` needs no project changes. Downloads are resumable. Generated files (`raw/`, `processed/`) are git-ignored; the full build takes about 6 minutes and about 1 GB.

## Processing

For each subject (followed by duplicate removal across the whole set, see below):

1. Pick one session, preferring the baseline session.
2. Download one atlas `ptseries` and the DCAN motion mask.
3. Drop frames flagged at FD > 0.2 mm, using the mask's own frame-removal vector.
4. Compute Pearson FC on the retained frames.
5. Save `processed/fc/<FILE_ID>_fc.npy` and delete the time series.

The output matches `data/abide/processed`, so the training code works unchanged with `--data-dir data/abide_dcan`:

- `processed/manifest.csv` with `FILE_ID`, `SITE_ID`, `DATASET`, `label` (0 = ASD, 1 = control), age, sex, FIQ, retained mean FD, and retained seconds;
- `processed/summary.json` with per-site counts and the exclusion breakdown;
- `processed/duplicates.csv` with the dropped duplicate scans;
- `processed/parcels.txt` with parcel names.

## Defaults and exclusions

| Item | Value |
|---|---|
| Atlas | `Gordon2014FreeSurferSubcortical`: 333 cortical + 19 subcortical = 352 parcels |
| Frame censoring | FD > 0.2 mm |
| Minimum retained data | 180 s (manifest-time filter; change with `--min-seconds`, no re-download) |
| Labels, ABIDE I | `data/abide/phenotypic/Phenotypic_V1_0b_preprocessed1.csv` |
| Labels, ABIDE II | per-site `participants.tsv` from `ABIDE2/RawData` |
| Excluded: `ABIDEII-ETHZ_1` | no `participants.tsv` on S3; labels are only on NITRC |
| Excluded: `ABIDEII-UCLA_Long`, `ABIDEII-UPSM_Long` | follow-up rescans of ABIDE I participants (same subject IDs) |
| Excluded: fewer than 10 frames after censoring | recorded as `too_much_motion` |
| Excluded: re-released scans | later copy of any pair with Fisher-z FC correlation > 0.7 (`--duplicate-r`); listed in `processed/duplicates.csv` |

### Duplicate scans across releases

Some ABIDE II cohorts re-release ABIDE I scans under new subject IDs. Comparing every pair of subjects' Fisher-z FC (Gordon atlas), different people correlate at 0.22 (median), and 99.99% of pairs fall below 0.49. Seven pairs reach 0.82-0.98, and every one of them is an ABIDE I subject matched to an ABIDE II subject with the same age and diagnosis:

| ABIDE I | ABIDE II | r |
|---|---|---|
| KKI 0050810 | KKI_1 29355 | 0.983 |
| KKI 0050811 | KKI_1 29357 | 0.981 |
| KKI 0050783 | KKI_1 29358 | 0.977 |
| KKI 0050780 | KKI_1 29353 | 0.975 |
| KKI 0050779 | KKI_1 29352 | 0.944 |
| USM 0050509 | USM_1 29506 | 0.939 |
| KKI 0050790 | KKI_1 29364 | 0.816 |

Left in, these leak across folds: when ABIDE I KKI is the test site, its copies in ABIDE II KKI_1 are in the training set. The build keeps the ABIDE I copy and drops the ABIDE II one. The sex field disagrees for 0050810/29355, so at least one release has a phenotype error.

Several institutions also contributed separate cohorts to both releases (KKI, NYU, OHSU, OLIN/ONRC, SDSU, Trinity/TCD, UCLA, USM, Leuven/KUL). Even without duplicate people, leave-one-site-out then trains on the test site's scanner. `neuroasd.linear_baseline` therefore also reports leave-one-institution-out.

Default build (2026-10-03): 2,119 candidates; 1,460 kept (642 ASD / 818 control; 800 ABIDE I, 660 ABIDE II; 36 sites). Dropped: 469 below 180 s after censoring, 100 without DCAN data, 83 with too much motion, 7 duplicate scans.

Three sites have only ASD participants after filtering (`ABIDEII-KUL_3`, `ABIDEII-NYU_2`, `ABIDEII-UCLA_1`). Use them for training only; AUC is undefined on them.

Other atlases available from the same derivatives: `HCP2016FreeSurferSubcortical` (379), `Power2011FreeSurferSubcortical`, `Markov2012FreeSurferSubcortical`.
