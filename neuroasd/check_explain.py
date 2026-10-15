"""No-data checks for chord-plot helpers and model-kind detection.

    uv run python -m neuroasd.check_explain
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroasd.dcan_atlas import describe_parcel, n_tangent_features
from neuroasd.explain import is_gnn_path, resolve_kind
from neuroasd.linear_explain import asd_direction, coef_to_edge_matrix, split_coef_blocks
from neuroasd.plots import plot_chord, plot_top_rois


def check_gordon_parcel_names() -> None:
    row = describe_parcel("gordon", 0, "1_L_Default")
    assert row["hemi"] == "L" and row["network"] == "Default"
    row = describe_parcel("gordon", 1, "2_R_CinguloOperc")
    assert row["hemi"] == "R" and row["network"] == "CinguloOpercular"
    row = describe_parcel("gordon", 2, "CEREBELLUM_LEFT")
    assert row["hemi"] == "L" and row["network"] == "Subcortical"


def check_hcp_and_power_names() -> None:
    row = describe_parcel("hcp", 0, "R_V1_ROI")
    assert row["hemi"] == "R" and row["network"] == "Visual"
    row = describe_parcel("hcp", 1, "L_A1_ROI")
    assert row["hemi"] == "L" and row["network"] == "Auditory"
    row = describe_parcel("power", 0, "a6_Fronto-parietal_task_control9")
    assert row["network"] == "Frontoparietal"


def check_coef_reshape_roundtrip() -> None:
    p = 6
    rng = np.random.default_rng(0)
    block = rng.standard_normal(n_tangent_features(p))
    mat = coef_to_edge_matrix(block, p)
    assert mat.shape == (p, p)
    assert np.allclose(mat, mat.T)
    iu = np.triu_indices(p, k=0)
    assert np.allclose(mat[iu], block)


def check_split_gordon_hcp_and_covariates() -> None:
    sizes = {"gordon": 4, "hcp": 5}
    n_g = n_tangent_features(4)
    n_h = n_tangent_features(5)
    coef = np.arange(n_g + n_h + 2, dtype=float)
    blocks, cov = split_coef_blocks(coef, ["gordon", "hcp"], sizes)
    assert blocks["gordon"].tolist() == list(range(n_g))
    assert blocks["hcp"].tolist() == list(range(n_g, n_g + n_h))
    assert cov.tolist() == [n_g + n_h, n_g + n_h + 1]


def check_asd_direction_flips_control_positive_weights() -> None:
    assert asd_direction(np.array([1.0, -2.0])).tolist() == [-1.0, 2.0]


def check_kind_detection() -> None:
    assert resolve_kind("auto", None, None) == "linear"
    assert resolve_kind("auto", Path("outputs/gcn_baseline/final_model.pt"), None) == "gnn"
    assert is_gnn_path(Path("model.pt"))
    assert not is_gnn_path(Path("experiments/iter10_estimator-swap-replace-the-singl_v1"))
    assert resolve_kind("linear", Path("model.pt"), None) == "linear"
    assert resolve_kind("gnn", None, Path("folds")) == "gnn"


def check_chord_writes_png(tmp_path: Path) -> None:
    table = [
        {
            "index": i,
            "roi_id": i,
            "abbr": f"L.R{i}",
            "name": f"ROI {i}",
            "short": f"ROI {i}",
            "hemi": "L" if i < 4 else "R",
            "network": "Default" if i % 2 == 0 else "Visual",
            "network_abbr": "DMN" if i % 2 == 0 else "VN",
        }
        for i in range(8)
    ]
    rng = np.random.default_rng(1)
    edge = rng.standard_normal((8, 8))
    edge = 0.5 * (edge + edge.T)
    np.fill_diagonal(edge, 0.0)
    chord = tmp_path / "chord.png"
    bars = tmp_path / "top_rois.png"
    plot_chord(edge, table, chord, top_k=6, title="test")
    plot_top_rois(edge.sum(axis=1), table, bars, top_k=4, title="test")
    assert chord.is_file() and chord.stat().st_size > 0
    assert bars.is_file() and bars.stat().st_size > 0


def main() -> None:
    checks = [
        check_gordon_parcel_names,
        check_hcp_and_power_names,
        check_coef_reshape_roundtrip,
        check_split_gordon_hcp_and_covariates,
        check_asd_direction_flips_control_positive_weights,
        check_kind_detection,
    ]
    for fn in checks:
        fn()
        print(f"ok  {fn.__name__}")
    tmp = Path("/tmp/neuroasd_check_explain")
    tmp.mkdir(exist_ok=True)
    check_chord_writes_png(tmp)
    print("ok  check_chord_writes_png")
    print("all explain checks passed")


if __name__ == "__main__":
    main()
