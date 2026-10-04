"""Coder sub-agent: edits the scored model, one change at a time."""

from __future__ import annotations

from typing import Any

from orchestra import SubAgent

from autoresearch.loop.paths import NEUROASD_DIR, TRIAL_PY, is_writable, rel, resolve_repo_path
from autoresearch.loop.protocol import leak_reasons_in_source
from autoresearch.loop.workspace import load_workspace, note_code_change, required_stage

INSTRUCTIONS = """\
You write code that might improve ASD vs control classification on ABIDE I + II. \
You do not run training and you do not judge scores.

You are a FRESH instance each turn. You cannot see the experimenter conversation. \
If a previous trial exists, you MUST call read_last_insight (and read_workspace) \
before any edit, then implement ONE new change that follows that insight.

Starting point: `neuroasd/model.py` is the formal linear baseline \
(Gordon tangent + kernel ridge, experiments/linear_baseline_dcan_v1). \
Do not start from the old GCN PRs. The scored function is `fit_predict(train, test)`.

Rules:
- Interpret the insight and the workspace ``ruled_out`` list: what was tried, \
what the next_code_change says, what not to repeat. Do not retry a ruled-out \
mechanism. The working tree is the last loso-institution winner or HEAD — failed \
diffs are archived under ``ruled_out[].patch``, not left on disk.
- Change ONE thing per turn (one mechanism). Keep the first edit tiny.
- Failed edits revert to the last loso-institution winner (or HEAD if none). \
After a loso-institution PASS, add the next mechanism on top of the winning code.
- The experimenter scores `autoresearch/trial.py`, which calls \
`neuroasd.model.fit_predict`. If a change is not visible there, the trial will \
not measure it. Prefer editing `neuroasd/model.py`. You may also edit other \
`neuroasd/*.py` files or `autoresearch/trial.py` if the mechanism needs it.
- Inputs you may use: `train.fc(atlas)` for atlas in gordon/hcp/power/markov, \
plus age, sex, groups, sites, and train.y. Do NOT use FIQ or head motion as \
predictive features (they differ by diagnosis and inflate results).
- Preferred directions, in order:
  1. Class weights or a decision threshold chosen on training folds only \
     (ASD recall of the baseline is ~54%).
  2. Site / institution score calibration (KKI and Leuven lose accuracy under \
     leave-one-institution-out while AUC stays high).
  3. Multi-atlas stacking (HCP, Power, Markov are already aligned).
  4. Age and sex as extra features.
  5. Ensembles of linear models; a strongly regularized GNN only as one member.
- Primary metric is fold-mean balanced accuracy. Always think about ASD recall \
and control recall, not accuracy alone.
- You MAY edit files under neuroasd/ and autoresearch/trial.py. You may NOT edit \
gates.json, data/, experiments/, autoresearch/loop/, or medresearch/.
- NEVER select a model using the evaluation / held-out set. Do not gate, report, \
or save a "best epoch" on val/test/LOSO as the official score.
- After editing, state the single hypothesis you just implemented.

If the workspace status is needs_loso_site or needs_loso_institution, do not \
edit: that code is frozen until the current change is fully measured.
"""

FROZEN_STAGES = {"loso-site", "loso-institution"}


class CodeTools:
    """File tools restricted to the scored model path."""

    def __init__(self) -> None:
        self.touched: list[str] = []
        self._read_insight = False

    def as_tools(self) -> list:
        return [
            self.read_workspace,
            self.read_last_insight,
            self.list_trainable_files,
            self.read_file,
            self.replace_in_file,
            self.write_file,
            self.record_hypothesis,
        ]

    def sources(self) -> list[str]:
        return list(self.touched)

    def _track(self, path) -> str:
        name = rel(path)
        if name not in self.touched:
            self.touched.append(name)
        return name

    def read_workspace(self) -> dict[str, Any]:
        """Return session state: status, current hypothesis, files, last insight."""
        workspace = load_workspace()
        if workspace.get("last_insight"):
            self._read_insight = True
        return workspace

    def read_last_insight(self) -> dict[str, Any]:
        """Return the experimenter's latest insight so you can act on it.

        Call this before editing whenever a previous trial exists. This is the
        only memory of the measurement round; you do not inherit that conversation.
        """
        self._read_insight = True
        data = load_workspace()
        insight = data.get("last_insight")
        payload = {
            "ok": True,
            "last_insight": insight,
            "ruled_out": data.get("ruled_out") or [],
            "working_tree": (
                "last loso-institution winner, or HEAD if none. Failed diffs are "
                "archived under ruled_out[].patch, not on disk."
            ),
        }
        if not insight:
            payload["note"] = "no prior trial in this session"
        return payload

    def list_trainable_files(self) -> list[str]:
        """List Python files the coder is allowed to edit.

        Returns:
            Paths relative to the repo root.
        """
        files = sorted(rel(path) for path in NEUROASD_DIR.glob("*.py"))
        files.append(rel(TRIAL_PY))
        return files

    def read_file(self, path: str) -> str:
        """Read a text file in this repository.

        Args:
            path: Repo-relative path, e.g. 'neuroasd/model.py'.
        """
        target = resolve_repo_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        text = target.read_text(encoding="utf-8")
        self._track(target)
        return text

    def record_hypothesis(self, hypothesis: str) -> dict[str, Any]:
        """Set the one-line hypothesis for the change you are about to make.

        Call this before the first edit of a new iteration.

        Args:
            hypothesis: One sentence, one mechanism, e.g. 'class-weighted ridge'.
        """
        workspace = load_workspace()
        if required_stage(workspace) in FROZEN_STAGES:
            return {
                "ok": False,
                "error": "code is frozen until the current change finishes LOSO",
                "workspace": workspace,
            }
        workspace["hypothesis"] = hypothesis.strip()
        from autoresearch.loop.workspace import save_workspace

        save_workspace(workspace)
        return {"ok": True, "hypothesis": hypothesis.strip()}

    def _guard_write(self, target, content: str) -> None:
        if not is_writable(target):
            raise PermissionError(
                f"not writable: {rel(target)}. Only neuroasd/*.py and "
                "autoresearch/trial.py may be edited."
            )
        reasons = leak_reasons_in_source(content)
        if reasons:
            raise ValueError(
                "refusing edit that reintroduces best-epoch evaluation leak: "
                + "; ".join(reasons)
            )
        workspace = load_workspace()
        if required_stage(workspace) in FROZEN_STAGES:
            raise PermissionError(
                "code is frozen (status "
                f"{workspace.get('status')}); wait for the experimenter to finish"
            )
        if workspace.get("last_insight") and not self._read_insight:
            raise PermissionError(
                "call read_last_insight before editing so the next change "
                "follows the experimenter's write-up"
            )

    def replace_in_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        """Replace one unique substring in a writable training file.

        Args:
            path: Repo-relative path to edit.
            old: Exact text to find (must occur once).
            new: Replacement text.
        """
        target = resolve_repo_path(path)
        current = target.read_text(encoding="utf-8")
        count = current.count(old)
        if count != 1:
            raise ValueError(f"old text occurs {count} times; need exactly one match")
        updated = current.replace(old, new, 1)
        self._guard_write(target, updated)
        target.write_text(updated, encoding="utf-8")
        workspace = note_code_change(rel(target))
        return {
            "ok": True,
            "path": self._track(target),
            "iteration": workspace["iteration"],
            "name": workspace["current_name"],
            "status": workspace["status"],
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Overwrite a writable training file with new contents.

        Prefer replace_in_file for small edits.

        Args:
            path: Repo-relative path to write.
            content: Full new file text.
        """
        target = resolve_repo_path(path)
        self._guard_write(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        workspace = note_code_change(rel(target))
        return {
            "ok": True,
            "path": self._track(target),
            "iteration": workspace["iteration"],
            "name": workspace["current_name"],
            "status": workspace["status"],
        }


class CoderAgent(SubAgent):
    """Writes one model / training change per assignment."""

    name = "coder"
    description = (
        "Edits the scored model (neuroasd/model.py and related neuroasd/ files). "
        "Use this to implement exactly one hypothesized improvement. Not for "
        "running trials or scoring metrics."
    )
    instructions = INSTRUCTIONS

    def __init__(self, *, promote: bool = True, push: bool = True, **kwargs: Any):
        kwargs.setdefault("max_tool_rounds", 32)
        super().__init__(**kwargs)

    def create_tools(self) -> CodeTools:
        return CodeTools()
