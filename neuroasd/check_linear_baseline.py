"""No-data checks for neuroasd.linear_baseline (synthetic FC matrices).

    uv run python -m neuroasd.check_linear_baseline
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier

from neuroasd import linear_baseline as lb


def synthetic(n_per_site: int = 30, n_sites: int = 4, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, fcs = [], []
    for s in range(n_sites):
        for i in range(n_per_site):
            label = int(rng.integers(0, 2)) if s < n_sites - 1 else 0
            ts = rng.standard_normal((80, p))
            ts[:, 1] += (0.8 if label else 0.0) * ts[:, 0] + 0.3 * s * ts[:, 2]
            fcs.append(np.corrcoef(ts, rowvar=False))
            rows.append({"SITE_ID": f"S{s}", "label": str(label), "DATASET": "abide1"})
    return rows, np.stack(fcs).astype(np.float32)


def check_kernel_ridge_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 200))
    y = (x[:, 0] + 0.5 * rng.standard_normal(60) > 0).astype(int)
    groups = np.repeat(np.arange(6), 10)
    clf = lb.KernelRidgeClassifier(alphas=np.array([10.0])).fit(x, y, groups)
    xs = (x - x.mean(0)) / x.std(0)
    ref = RidgeClassifier(alpha=10.0).fit(xs, y)
    assert np.allclose(clf.decision_function(x), ref.decision_function(xs), atol=1e-6)


def check_tangent_uses_training_reference_only() -> None:
    _, fcs = synthetic()
    train = np.arange(60)
    base = lb.tangent_features(fcs, train)
    perturbed = fcs.copy()
    perturbed[60:] = np.eye(fcs.shape[1])
    again = lb.tangent_features(perturbed, train)
    assert np.allclose(base[:60], again[:60]), "test subjects changed the tangent reference"
    assert base.shape[1] == fcs.shape[1] * (fcs.shape[1] + 1) // 2


def check_missing_parcels_stay_finite() -> None:
    _, fcs = synthetic()
    fcs[0, 3, :] = np.nan
    fcs[0, :, 3] = np.nan
    clean = lb.clean_fc(fcs.copy())
    assert np.isfinite(clean).all() and clean[0, 3, 3] == 1.0 and clean[0, 3, 4] == 0.0
    feats = lb.tangent_features(clean, np.arange(60))
    others = np.abs(feats[1:]).max()
    assert np.abs(feats[0]).max() < 3 * others, "a missing parcel produced an outlier tangent feature"


def check_single_class_sites_are_training_only() -> None:
    rows, fcs = synthetic()
    y = np.array([int(r["label"]) for r in rows])
    sites = np.array([r["SITE_ID"] for r in rows])
    assert "S3" not in lb.test_groups(sites, y, None)
    results = lb.evaluate(rows, fcs, ["fisherz", "tangent"], repeats=2, log=lambda _m: None)
    for feature, res in results.items():
        assert {f["test_group"] for f in res["folds"]} == {"S0", "S1", "S2"}
        assert all(f["train_size"] == len(rows) - f["test_size"] for f in res["folds"])
        assert len(res["repeat_folds"]) == 2 * 3
    summary = lb.summarize(results, repeats=2)
    assert summary["tangent"]["all_folds"]["folds"] == 3
    assert summary["fisherz"]["all_folds"]["auc_mean"] > 0.6, summary


def check_institution_holds_out_all_cohorts() -> None:
    rows, fcs = synthetic(n_sites=4)
    for row, site in zip(rows, np.repeat(["KKI", "ABIDEII-KKI_1", "NYU", "ABIDEII-NYU_1"], 30)):
        row["SITE_ID"] = str(site)
    # ABIDEII-NYU_1 is single-class here; merged with NYU it still forms a test fold.
    results = lb.evaluate(rows, fcs, ["fisherz"], repeats=0, log=lambda _m: None, grouping="institution")
    folds = results["fisherz"]["folds"]
    assert {f["test_group"] for f in folds} == {"KKI", "NYU"}
    for fold in folds:
        assert fold["test_size"] == 60 and fold["train_size"] == 60
        assert len(fold["test_sites"]) == 2


def main() -> None:
    checks = [
        check_kernel_ridge_matches_sklearn,
        check_tangent_uses_training_reference_only,
        check_missing_parcels_stay_finite,
        check_single_class_sites_are_training_only,
        check_institution_holds_out_all_cohorts,
    ]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"{len(checks)} passed")


if __name__ == "__main__":
    main()
