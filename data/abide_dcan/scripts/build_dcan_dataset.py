# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "nibabel"]
# ///
"""Build an ABIDE I + ABIDE II FC dataset from the public DCAN derivatives.

Both ABIDE releases were processed by the same DCAN (ABCD-HCP) pipeline and
published on the FCP-INDI S3 bucket with parcellated time series, so no local
fMRI preprocessing is needed:

    s3://fcp-indi/data/Projects/ABIDE/Derivatives/DCAN/sub-00<SUB_ID>/...
    s3://fcp-indi/data/Projects/ABIDE2/Derivatives/DCAN/sub-<SUB_ID>/...

For each subject this script downloads one atlas ptseries plus the DCAN motion
mask, drops frames above the FD threshold, computes the Pearson FC matrix and
deletes the time series. Output mirrors data/abide/processed so the existing
training code works with --data-dir data/abide_dcan.

Usage:
    uv run data/abide_dcan/scripts/build_dcan_dataset.py --sites ABIDEII-NYU_1 ABIDEII-BNI_1
    uv run data/abide_dcan/scripts/build_dcan_dataset.py            # everything
    uv run data/abide_dcan/scripts/build_dcan_dataset.py --manifest-only --min-seconds 240
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

S3_HTTP = "https://s3.amazonaws.com/fcp-indi"
DCAN_PREFIX = {
    "abide1": "data/Projects/ABIDE/Derivatives/DCAN",
    "abide2": "data/Projects/ABIDE2/Derivatives/DCAN",
}
ABIDE2_RAW_PREFIX = "data/Projects/ABIDE2/RawData"
# Follow-up rescans of ABIDE I participants (they reuse ABIDE I subject IDs).
ABIDE2_LONGITUDINAL_SITES = {"ABIDEII-UCLA_Long", "ABIDEII-UPSM_Long"}

DEFAULT_OUT = Path(__file__).resolve().parents[1]
# ABIDE I scans re-released in ABIDE II under new IDs; the ABIDE II copy is dropped.
KNOWN_DUPLICATES = DEFAULT_OUT / "duplicate_scans.csv"
# Subjects with a zero-variance (unscanned) parcel in any DCAN atlas; dropped from
# every atlas so all atlas datasets keep the same subjects.
KNOWN_INCOMPLETE = DEFAULT_OUT / "incomplete_coverage.csv"
ABIDE1_PHENO = (
    Path(__file__).resolve().parents[2]
    / "abide"
    / "phenotypic"
    / "Phenotypic_V1_0b_preprocessed1.csv"
)

MANIFEST_FIELDS = [
    "FILE_ID",
    "SUB_ID",
    "SITE_ID",
    "DATASET",
    "DX_GROUP",
    "label",
    "AGE_AT_SCAN",
    "SEX",
    "FIQ",
    "func_mean_fd",
    "remaining_seconds",
    "roi_path",
    "fc_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DCAN_PREFIX), default=sorted(DCAN_PREFIX))
    parser.add_argument("--atlas", default="Gordon2014FreeSurferSubcortical",
                        help="Gordon2014FreeSurferSubcortical (352), HCP2016FreeSurferSubcortical (379), "
                             "Power2011FreeSurferSubcortical, Markov2012FreeSurferSubcortical")
    parser.add_argument("--fd", type=float, default=0.2, help="FD threshold (mm) for frame censoring")
    parser.add_argument("--min-seconds", type=float, default=180.0,
                        help="Drop subjects with less retained data than this after censoring")
    parser.add_argument("--duplicate-r", type=float, default=0.7,
                        help="Report cross-site pairs whose Fisher-z FC correlates above this as possible duplicates")
    parser.add_argument("--sites", nargs="*", default=None, help="Only process these SITE_IDs")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N subjects (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true", help="Skip downloads; rebuild manifest")
    return parser.parse_args()


def http_get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def s3_list(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    prefixes: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        body = http_get(f"{S3_HTTP}?{urllib.parse.urlencode(params)}").decode()
        keys += re.findall(r"<Key>(.*?)</Key>", body)
        prefixes += re.findall(r"<Prefix>(.*?)</Prefix>", body)
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if not match:
            return keys, [p for p in prefixes if p != prefix]
        token = match.group(1)


def load_abide1_pheno() -> list[dict[str, str]]:
    with ABIDE1_PHENO.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "dataset": "abide1",
            "sub_id": row["SUB_ID"],
            "dcan_sub": f"sub-{int(row['SUB_ID']):07d}",
            "SITE_ID": row["SITE_ID"],
            "DX_GROUP": row["DX_GROUP"],
            "AGE_AT_SCAN": row.get("AGE_AT_SCAN", ""),
            "SEX": row.get("SEX", ""),
            "FIQ": row.get("FIQ", ""),
        }
        for row in rows
    ]


def load_abide2_pheno(cache_dir: Path, abide1_ids: set[int]) -> list[dict[str, str]]:
    cache = cache_dir / "abide2_participants.tsv"
    if not cache.exists():
        _, site_dirs = s3_list(f"{ABIDE2_RAW_PREFIX}/", delimiter="/")
        sites = [d.rstrip("/").split("/")[-1] for d in site_dirs]
        sites = [s for s in sites if s.startswith("ABIDEII-") and s not in ABIDE2_LONGITUDINAL_SITES]
        merged: list[dict[str, str]] = []
        header: list[str] = []
        for site in sites:
            try:
                raw = http_get(f"{S3_HTTP}/{ABIDE2_RAW_PREFIX}/{site}/participants.tsv")
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                # ETHZ_1 ships no participants.tsv on S3; its labels are only on NITRC.
                print(f"WARNING: no participants.tsv for {site}; skipping its subjects")
                continue
            text = raw.decode("utf-8", "replace")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            for row in reader:
                clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                merged.append(clean)
                header += [k for k in clean if k not in header]
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
            writer.writeheader()
            writer.writerows(merged)
    with cache.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out = []
    for row in rows:
        sub_id = int(row["participant_id"])
        if sub_id in abide1_ids:
            continue
        out.append(
            {
                "dataset": "abide2",
                "sub_id": str(sub_id),
                "dcan_sub": f"sub-{sub_id}",
                "SITE_ID": row["site_id"],
                "DX_GROUP": row["dx_group"],
                "AGE_AT_SCAN": row.get("age_at_scan", ""),
                "SEX": row.get("sex", ""),
                "FIQ": row.get("fiq", ""),
            }
        )
    return out


def file_id(subject: dict[str, str]) -> str:
    return f"{subject['dataset']}_{subject['dcan_sub'][4:]}"


def pick_session(keys: list[str], atlas: str) -> tuple[str, str] | None:
    """Return (ptseries_key, motion_mask_key) for one session, preferring baseline."""
    sessions: dict[str, dict[str, str]] = {}
    for key in keys:
        if "/func/" not in key:
            continue
        match = re.search(r"/(ses-[^/]+)/func/", key)
        if not match:
            continue
        ses = sessions.setdefault(match.group(1), {})
        if f"atlas-{atlas}_desc-filtered_timeseries.ptseries.nii" in key:
            ses["ts"] = key
        elif key.endswith("desc-filtered_motion_mask.mat"):
            ses["mask"] = key
    complete = sorted(s for s, files in sessions.items() if {"ts", "mask"} <= files.keys())
    if not complete:
        return None
    complete.sort(key=lambda s: (0 if "baseline" in s.lower() else 1, s))
    chosen = sessions[complete[0]]
    return chosen["ts"], chosen["mask"]


class TooMuchMotion(ValueError):
    pass


def censored_fc(ts_bytes: bytes, mask_bytes: bytes, fd: float) -> tuple[np.ndarray, dict]:
    with tempfile.NamedTemporaryFile(suffix=".ptseries.nii") as tmp:
        tmp.write(ts_bytes)
        tmp.flush()
        img = nib.load(tmp.name)
        ts = np.asarray(img.get_fdata(), dtype=np.float64)
        parcels = [str(name) for name in img.header.get_axis(1).name]

    motion = sio.loadmat(io.BytesIO(mask_bytes), squeeze_me=True, struct_as_record=False)["motion_data"]
    entry = min(motion, key=lambda e: abs(float(e.FD_threshold) - fd))
    if abs(float(entry.FD_threshold) - fd) > 1e-6:
        raise ValueError(f"FD threshold {fd} not in motion mask")
    removal = np.asarray(entry.frame_removal).astype(bool)
    if removal.shape[0] != ts.shape[0]:
        raise ValueError(f"mask length {removal.shape[0]} != {ts.shape[0]} frames")
    kept = ts[~removal]
    if kept.shape[0] < 10:
        raise TooMuchMotion(f"only {kept.shape[0]} frames survive FD<{fd}")
    with np.errstate(invalid="ignore", divide="ignore"):
        fc = np.corrcoef(kept, rowvar=False)
    meta = {
        "tr": float(entry.epi_TR),
        "total_frames": int(ts.shape[0]),
        "remaining_frames": int(kept.shape[0]),
        "remaining_seconds": float(kept.shape[0] * float(entry.epi_TR)),
        "remaining_mean_fd": float(entry.remaining_frame_mean_FD),
        "zero_variance_parcels": int((kept.std(axis=0) == 0).sum()),
        "n_parcels": int(ts.shape[1]),
    }
    return fc.astype(np.float32), meta | {"parcels": parcels}


def process_subject(subject: dict[str, str], args: argparse.Namespace, fc_dir: Path, meta_dir: Path) -> str:
    fid = file_id(subject)
    meta_path = meta_dir / f"{fid}.json"
    if meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached.get("atlas") == args.atlas and cached.get("fd_threshold") == args.fd:
            return "cached"
    prefix = f"{DCAN_PREFIX[subject['dataset']]}/{subject['dcan_sub']}/"
    keys, _ = s3_list(prefix)
    picked = pick_session(keys, args.atlas)
    record: dict = {"FILE_ID": fid, "atlas": args.atlas, "fd_threshold": args.fd}
    if picked is None:
        record["status"] = "no_dcan_data"
    else:
        ts_key, mask_key = picked
        try:
            fc, meta = censored_fc(http_get(f"{S3_HTTP}/{ts_key}"), http_get(f"{S3_HTTP}/{mask_key}"), args.fd)
        except TooMuchMotion as exc:
            record |= {"status": "too_much_motion", "error": str(exc), "ts_key": ts_key}
        except Exception as exc:  # noqa: BLE001 - per-subject failures are recorded, not fatal
            record |= {"status": "error", "error": repr(exc), "ts_key": ts_key}
        else:
            np.save(fc_dir / f"{fid}_fc.npy", fc)
            parcels = meta.pop("parcels")
            parcels_path = args.out_dir / "processed" / "parcels.txt"
            if not parcels_path.exists():
                parcels_path.write_text("\n".join(parcels) + "\n", encoding="utf-8")
            record |= {"status": "ok", "ts_key": ts_key} | meta
    meta_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record["status"]


def dx_group_to_label(dx_group: str) -> int:
    """ASD=0, control=1, matching data/abide/processed."""
    return {1: 0, 2: 1}[int(float(dx_group))]


def load_known_duplicates() -> dict[str, str]:
    """Map each dropped FILE_ID in duplicate_scans.csv to the FILE_ID that is kept."""
    with KNOWN_DUPLICATES.open(newline="", encoding="utf-8") as handle:
        return {row["dropped"]: row["kept"] for row in csv.DictReader(handle)}


def load_incomplete_coverage() -> set[str]:
    with KNOWN_INCOMPLETE.open(newline="", encoding="utf-8") as handle:
        return {row["FILE_ID"] for row in csv.DictReader(handle)}


def motion_fingerprint(meta: dict) -> tuple:
    return meta["total_frames"], meta["remaining_frames"], round(meta["remaining_mean_fd"], 6)


def find_duplicate_candidates(
    rows: list[dict[str, str]], metas: dict[str, dict], out_dir: Path, threshold: float
) -> list[dict[str, str]]:
    """Flag cross-site pairs that look like the same scan; report only, never drop.

    Two independent signals: Fisher-z FC correlation above `threshold` (different
    people stay below ~0.5 on Gordon; coarse atlases run higher), or an identical
    DCAN motion trace (frame counts and retained mean FD). Pairs within one site
    are ignored because they always share a fold.
    """
    if len(rows) < 2:
        return []
    first = np.load(out_dir / rows[0]["fc_path"])
    iu = np.triu_indices(first.shape[0], 1)
    feats = np.empty((len(rows), iu[0].size), dtype=np.float32)
    for k, row in enumerate(rows):
        fc = np.nan_to_num(np.load(out_dir / row["fc_path"]))
        feats[k] = np.arctanh(np.clip(fc[iu], -0.999, 0.999))
    feats -= feats.mean(axis=1, keepdims=True)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    sim = feats @ feats.T
    sites = np.array([row["SITE_ID"] for row in rows])
    prints = [motion_fingerprint(metas[row["FILE_ID"]]) for row in rows]
    candidates = []
    for i, j in zip(*np.triu_indices(len(rows), 1)):
        if sites[i] == sites[j]:
            continue
        same_motion = prints[i] == prints[j]
        if sim[i, j] > threshold or same_motion:
            candidates.append({
                "a": rows[i]["FILE_ID"],
                "b": rows[j]["FILE_ID"],
                "fc_r": f"{sim[i, j]:.3f}",
                "identical_motion": str(same_motion),
            })
    return candidates


def write_manifest(subjects: list[dict[str, str]], args: argparse.Namespace, meta_dir: Path) -> None:
    out_dir = args.out_dir / "processed"
    rows: list[dict[str, str]] = []
    metas: dict[str, dict] = {}
    dropped: dict[str, int] = {}
    known_duplicates = load_known_duplicates()
    incomplete = load_incomplete_coverage()
    unlisted_incomplete: list[str] = []
    for subject in subjects:
        fid = file_id(subject)
        meta_path = meta_dir / f"{fid}.json"
        if not meta_path.exists():
            dropped["not_processed"] = dropped.get("not_processed", 0) + 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reason = None
        if meta["status"] != "ok":
            reason = meta["status"]
        elif meta["atlas"] != args.atlas or abs(meta["fd_threshold"] - args.fd) > 1e-9:
            reason = "stale_atlas_or_fd"
        elif meta["remaining_seconds"] < args.min_seconds:
            reason = "short_after_censoring"
        elif subject["DX_GROUP"] not in {"1", "2", "1.0", "2.0"}:
            reason = "missing_dx"
        elif fid in known_duplicates:
            reason = "duplicate_scan"
        elif fid in incomplete or meta["zero_variance_parcels"] > 0:
            reason = "incomplete_coverage"
            if fid not in incomplete:
                unlisted_incomplete.append(fid)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        metas[fid] = meta
        rows.append(
            {
                "FILE_ID": fid,
                "SUB_ID": subject["sub_id"],
                "SITE_ID": subject["SITE_ID"],
                "DATASET": subject["dataset"],
                "DX_GROUP": str(int(float(subject["DX_GROUP"]))),
                "label": str(dx_group_to_label(subject["DX_GROUP"])),
                "AGE_AT_SCAN": subject["AGE_AT_SCAN"],
                "SEX": subject["SEX"],
                "FIQ": subject["FIQ"],
                "func_mean_fd": f"{meta['remaining_mean_fd']:.4f}",
                "remaining_seconds": f"{meta['remaining_seconds']:.1f}",
                "roi_path": meta["ts_key"],
                "fc_path": f"processed/fc/{fid}_fc.npy",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if unlisted_incomplete:
        print(f"WARNING: {len(unlisted_incomplete)} subject(s) with zero-variance parcels are not in "
              f"{KNOWN_INCOMPLETE.name} (dropped here only; add them so other atlases match): "
              f"{' '.join(unlisted_incomplete)}")
    candidates = find_duplicate_candidates(rows, metas, args.out_dir, args.duplicate_r)
    with (out_dir / "duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "b", "fc_r", "identical_motion"])
        writer.writeheader()
        writer.writerows(candidates)
    if candidates:
        print(f"WARNING: {len(candidates)} possible duplicate scan pair(s) not in {KNOWN_DUPLICATES.name}; "
              f"review {out_dir / 'duplicate_candidates.csv'}")
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    np.save(out_dir / "labels.npy", labels)
    (out_dir / "file_ids.txt").write_text("".join(r["FILE_ID"] + "\n" for r in rows), encoding="utf-8")

    per_site: dict[str, dict[str, int]] = {}
    for row in rows:
        site = per_site.setdefault(row["SITE_ID"], {"asd": 0, "control": 0})
        site["asd" if row["label"] == "0" else "control"] += 1
    summary = {
        "source": "FCP-INDI DCAN derivatives (ABCD-HCP pipeline)",
        "atlas": args.atlas,
        "fd_threshold": args.fd,
        "min_seconds": args.min_seconds,
        "n_subjects": len(rows),
        "n_asd": int((labels == 0).sum()),
        "n_control": int((labels == 1).sum()),
        "n_by_dataset": {d: sum(r["DATASET"] == d for r in rows) for d in sorted(DCAN_PREFIX)},
        "n_sites": len(per_site),
        "per_site": dict(sorted(per_site.items())),
        "dropped": dropped,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_site"}, indent=2))


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    raw_dir = args.out_dir / "raw"
    meta_dir = raw_dir / "meta"
    fc_dir = args.out_dir / "processed" / "fc"
    for path in (meta_dir, fc_dir):
        path.mkdir(parents=True, exist_ok=True)

    abide1 = load_abide1_pheno()
    abide1_ids = {int(s["sub_id"]) for s in abide1}
    subjects: list[dict[str, str]] = []
    if "abide1" in args.datasets:
        subjects += abide1
    if "abide2" in args.datasets:
        subjects += load_abide2_pheno(raw_dir / "phenotypic", abide1_ids)
    if args.sites:
        wanted = set(args.sites)
        subjects = [s for s in subjects if s["SITE_ID"] in wanted]
    if args.limit:
        subjects = subjects[: args.limit]
    print(f"Subjects selected: {len(subjects)}")

    if not args.manifest_only:
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_subject, s, args, fc_dir, meta_dir): s for s in subjects}
            for done, future in enumerate(as_completed(futures), start=1):
                status = future.result()
                counts[status] = counts.get(status, 0) + 1
                if done % 50 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)} {counts}", flush=True)

    write_manifest(subjects, args, meta_dir)


if __name__ == "__main__":
    main()
