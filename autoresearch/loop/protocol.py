"""Hard checks against the historical best-epoch evaluation leak.

An earlier revision reported LOSO AUC 0.707 by picking, per fold, the epoch
with the highest score on the held-out site. That uses the evaluation set to
select the model. Final-epoch scoring of the same config is 0.623; the leak
was worth about 0.07 AUC. Gates and reports must use final-epoch metrics only.
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_GATE_METRICS = frozenset({"best_auc", "best_auc_mean"})
REQUIRED_MODEL_SELECTION = ("final epoch", "final fit")

# Edits that typically reintroduce the leak.
_LEAK_PATTERNS = (
    re.compile(r"""gate\[[^\]]*(best_auc)""", re.I),
    re.compile(r"""["']metric["']\s*:\s*["']best_auc""", re.I),
    re.compile(r"""observed\s*=\s*summary\[[\"']best_auc""", re.I),
    re.compile(r"""verdict.*best_auc_mean""", re.I),
    re.compile(r"""reported_metrics\s*=\s*.*best_""", re.I),
    re.compile(r"""model_selection\s*=\s*[\"']best""", re.I),
)


def leak_reasons_in_source(content: str) -> list[str]:
    """Return reasons this source would reintroduce evaluation-set selection."""
    reasons: list[str] = []
    for pattern in _LEAK_PATTERNS:
        if pattern.search(content):
            reasons.append(
                f"matched {pattern.pattern!r}: this selects/gates on a best-epoch "
                "metric and leaks the evaluation set"
            )
    return reasons


def protocol_notes(result: dict[str, Any]) -> list[str]:
    """Notes the experimenter must surface; first items are hard failures."""
    notes: list[str] = []
    selection = str(result.get("model_selection", "")).lower()
    if not any(token in selection for token in REQUIRED_MODEL_SELECTION):
        notes.append(
            "PROTOCOL FAIL: result is not marked as a final fit / final epoch. "
            "Do not treat this number as an improvement."
        )

    verdict = result.get("verdict") or {}
    metric = str(verdict.get("metric", ""))
    if metric in FORBIDDEN_GATE_METRICS or metric.startswith("best_"):
        notes.append(
            f"PROTOCOL FAIL: gated on {metric!r}. Best-epoch gating leaks the "
            "evaluation set (historical LOSO bias ~0.07 AUC)."
        )

    summary = result.get("summary") or {}
    final_auc = summary.get("auc_mean")
    best_auc = summary.get("best_auc_mean")
    if isinstance(final_auc, (int, float)) and isinstance(best_auc, (int, float)):
        gap = float(best_auc) - float(final_auc)
        if gap > 0.03:
            notes.append(
                f"diagnostic: best-epoch AUC is {gap:.3f} above final-epoch "
                f"({best_auc:.3f} vs {final_auc:.3f}). Report only the final-epoch "
                "number; a similar gap (~0.07) was the old LOSO leak."
            )
    return notes


def protocol_ok(result: dict[str, Any]) -> bool:
    return not any(note.startswith("PROTOCOL FAIL") for note in protocol_notes(result))
