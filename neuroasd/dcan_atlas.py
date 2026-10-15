"""ROI tables for DCAN atlases, in the same schema `plot_chord` already uses.

Harvard-Oxford (the old GCN) stays in `roi_atlas.py`. This module reads
`processed/parcels.txt` from a DCAN dataset directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroasd.roi_atlas import NETWORK_ABBR, NETWORK_COLORS, sort_order

REPO_ROOT = Path(__file__).resolve().parents[1]

ATLAS_DIRS = {
    "gordon": REPO_ROOT / "data" / "abide_dcan",
    "hcp": REPO_ROOT / "data" / "abide_dcan_hcp",
    "power": REPO_ROOT / "data" / "abide_dcan_power",
    "markov": REPO_ROOT / "data" / "abide_dcan_markov",
}

# Extra networks that appear in Gordon / Power but not Harvard-Oxford.
NETWORK_COLORS.update(
    {
        "CinguloOpercular": "#B56576",
        "VentralAttention": "#E56B6F",
    }
)
NETWORK_ABBR.update(
    {
        "CinguloOpercular": "CON",
        "VentralAttention": "VAN",
    }
)

_SUBCORTICAL_WORDS = (
    "CEREBELLUM",
    "THALAMUS",
    "CAUDATE",
    "PUTAMEN",
    "PALLIDUM",
    "HIPPOCAMPUS",
    "AMYGDALA",
    "ACCUMBENS",
    "DIENCEPHALON",
    "BRAIN_STEM",
    "BRAIN",
)

_GORDON_NETWORK = {
    "Default": "Default",
    "SMhand": "Somatomotor",
    "SMmouth": "Somatomotor",
    "Visual": "Visual",
    "FrontoParietal": "Frontoparietal",
    "Auditory": "Auditory",
    "None": "Unknown",
    "CinguloOperc": "CinguloOpercular",
    "DorsalAttn": "DorsalAttention",
    "VentralAttn": "VentralAttention",
    "RetrosplenialTemporal": "Default",
    "CinguloParietal": "Default",
    "Salience": "Salience",
}

_POWER_NETWORK = (
    ("default", "Default"),
    ("fronto-parietal", "Frontoparietal"),
    ("frontoparietal", "Frontoparietal"),
    ("cingulo-opercular", "CinguloOpercular"),
    ("dorsal", "DorsalAttention"),
    ("ventral", "VentralAttention"),
    ("visual", "Visual"),
    ("auditory", "Auditory"),
    ("mouth", "Somatomotor"),
    ("hand", "Somatomotor"),
    ("somatosensory", "Somatomotor"),
    ("salience", "Salience"),
    ("hippocampus", "Limbic"),
    ("memory", "Limbic"),
    ("temporal", "Language"),
    ("unknown", "Unknown"),
)

# Longest-first Glasser tokens so "V3A" wins over "V3".
_HCP_NETWORK = (
    (("V1", "V2", "V3A", "V3B", "V3CD", "V3", "V4t", "V4", "V6A", "V6", "V7", "V8",
      "VMV1", "VMV2", "VMV3", "VMV", "VVC", "LO1", "LO2", "LO3", "PIT", "FFC",
      "MST", "MT", "FST", "DVT", "ProS", "PH"), "Visual"),
    (("3a", "3b", "6ma", "6mp", "6d", "6v", "6r", "6a", "5L", "5m", "5mv",
      "24dd", "24dv", "SCEF", "PEF", "4", "1", "2"), "Somatomotor"),
    (("A1", "LBelt", "MBelt", "PBelt", "RI", "A4", "A5", "STGa", "TA2", "AII"), "Auditory"),
    (("55b", "SFL", "PSL", "STV", "44", "45", "IFJa", "IFJp", "IFSp", "IFSa"), "Language"),
    (("FEF", "IPS1", "7Pm", "7PL", "7Am", "7AL", "LIPv", "LIPd", "VIP", "MIP",
      "AIP", "IP0", "IP1", "IP2", "PFt"), "DorsalAttention"),
    (("8C", "8Av", "8Ad", "i6-8", "s6-8", "46", "9-46d", "a9-46v", "p9-46v",
      "p47r", "a47r", "47l", "p10p", "a10p", "10d", "PF", "PFm", "PGi", "PGs"),
     "Frontoparietal"),
    (("RSC", "POS1", "POS2", "PCV", "7m", "31pd", "31pv", "31a", "d23ab", "v23ab",
      "23c", "23d", "d32", "p32", "a24", "p24", "10r", "10v", "9m", "8BM",
      "STSda", "STSva", "STSdp", "STSvp", "TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
      "TF", "PeEc", "PHA1", "PHA2", "PHA3", "TGd", "TGv"), "Default"),
    (("33pr", "a24pr", "p24pr", "p32pr", "a32pr", "25", "s32", "11l", "13l",
      "OFC", "47s", "47m", "52", "PI", "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
      "MI", "AVI", "AAIC", "Pir"), "Salience"),
)


def read_parcel_names(data_dir: Path | str) -> list[str]:
    path = Path(data_dir) / "processed" / "parcels.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no parcels.txt under {data_dir}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def n_tangent_features(n_parcels: int) -> int:
    return n_parcels * (n_parcels + 1) // 2


def _hemi_from_name(name: str) -> str:
    upper = name.upper()
    if upper.endswith("_LEFT") or upper.startswith("L_") or "_L_" in f"_{upper}_":
        return "L"
    if upper.endswith("_RIGHT") or upper.startswith("R_") or "_R_" in f"_{upper}_":
        return "R"
    tokens = name.split("_")
    if len(tokens) >= 2 and tokens[1] in {"L", "R"}:
        return tokens[1]
    if tokens and tokens[0] in {"L", "R"}:
        return tokens[0]
    return "U"


def _is_subcortical(name: str) -> bool:
    upper = name.upper()
    return any(word in upper for word in _SUBCORTICAL_WORDS)


def _gordon_network(name: str) -> str:
    if _is_subcortical(name):
        return "Subcortical"
    parts = name.split("_")
    raw = "_".join(parts[2:]) if len(parts) >= 3 and parts[1] in {"L", "R"} else name
    return _GORDON_NETWORK.get(raw, "Unknown")


def _power_network(name: str) -> str:
    if _is_subcortical(name):
        return "Subcortical"
    lower = name.lower()
    for token, network in _POWER_NETWORK:
        if token in lower:
            return network
    return "Unknown"


def _hcp_network(name: str) -> str:
    if _is_subcortical(name):
        return "Subcortical"
    token = name
    if token.endswith("_ROI"):
        token = token[: -len("_ROI")]
    if token[:2] in {"L_", "R_"}:
        token = token[2:]
    for keys, network in _HCP_NETWORK:
        if token in keys:
            return network
    return "Unknown"


def _markov_network(name: str) -> str:
    return "Subcortical" if _is_subcortical(name) else "Unknown"


def _abbr(name: str, hemi: str) -> str:
    token = name
    if token.endswith("_ROI"):
        token = token[: -len("_ROI")]
    parts = token.split("_")
    if parts and parts[0] in {"L", "R"}:
        token = "_".join(parts[1:]) or token
    elif len(parts) >= 2 and parts[1] in {"L", "R"}:
        token = "_".join(parts[2:] or parts[:1])
    token = token.replace("CEREBELLUM", "CB").replace("THALAMUS", "THA")
    token = token.replace("HIPPOCAMPUS", "HIP").replace("AMYGDALA", "AMYG")
    short = token[:10]
    return f"{hemi}.{short}" if hemi in {"L", "R"} else short


_NETWORK_FN = {
    "gordon": _gordon_network,
    "hcp": _hcp_network,
    "power": _power_network,
    "markov": _markov_network,
}


def describe_parcel(atlas: str, index: int, name: str) -> dict[str, str | int]:
    hemi = _hemi_from_name(name)
    network = _NETWORK_FN.get(atlas, lambda _n: "Unknown")(name)
    abbr = _abbr(name, hemi)
    return {
        "index": index,
        "roi_id": name,
        "name": name,
        "short": name,
        "hemi": hemi,
        "network": network,
        "network_abbr": NETWORK_ABBR.get(network, "?"),
        "abbr": abbr,
    }


def atlas_table(atlas: str, data_dir: Path | str | None = None) -> list[dict[str, str | int]]:
    if atlas not in ATLAS_DIRS:
        raise KeyError(f"unknown atlas {atlas!r}; choose from {sorted(ATLAS_DIRS)}")
    root = Path(data_dir) if data_dir is not None else ATLAS_DIRS[atlas]
    return [describe_parcel(atlas, i, name) for i, name in enumerate(read_parcel_names(root))]


def atlas_size(atlas: str, data_dir: Path | str | None = None) -> int:
    return len(atlas_table(atlas, data_dir))


__all__ = [
    "ATLAS_DIRS",
    "NETWORK_ABBR",
    "NETWORK_COLORS",
    "atlas_size",
    "atlas_table",
    "n_tangent_features",
    "read_parcel_names",
    "sort_order",
]
