"""Experimenter sub-agent: run a gated trial and write an honest insight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from orchestra import SubAgent

from autoresearch.loop.paths import GATES_PATH, LEDGER_PATH, PROGRAM_PATH, REPO_ROOT, trials_dir
from autoresearch.loop.promote import promote
from autoresearch.loop.protocol import protocol_notes, protocol_ok
from autoresearch.loop.workspace import (
    apply_trial_outcome,
    coder_finished_cleanly,
    lint_passed,
    load_workspace,
    record_unmeasured_trial,
    required_stage,
    save_insight,
)

INSTRUCTIONS = """\
You measure a code change and explain what the number means. You do not edit \
model code.

Rules:
- Read the workspace first. Run the stage it requires; do not skip ahead.
- Call run_trial exactly once, then store_insight and stop. If screen PASSed, \
the lead will dispatch a fresh experimenter for loso-site. Do not chain stages.
- Official metric is fold-mean balanced accuracy from one fit per fold. \
Accuracy, AUC, ASD recall, and control recall are always reported alongside it.
- A 2026-08-27 revision showed that picking the best epoch on the evaluation \
set inflated LOSO from 0.623 to 0.707 (~0.07 AUC). If a result is selected on \
the test fold, it is not an improvement.
- screen PASS is a filter, not evidence of a better model. Only \
loso-institution PASS justifies promotion.
- On FAIL, say what the result rules out. On PASS, say the margin versus the gate.
- Never claim a win from a protocol failure.
- After run_trial, call store_insight with a coder-actionable writeup: what was \
tried, balanced accuracy vs gate, what this rules out, and the ONE next code change \
worth trying (or "promote to next stage" if the same code should continue).

Return a short structured insight the coder can act on.
"""

def measurement_coverage(files_changed: list[str] | None) -> list[str]:
    """Warn when the scored path (`model.py` / `trial.py`) likely missed the coder's edit."""
    files = files_changed or []
    notes: list[str] = []
    scored = any(path.endswith("trial.py") or path.endswith("model.py") for path in files)
    training_only = any(
        path.endswith(("train.py", "loso_cv.py", "gcn.py")) for path in files
    )
    if training_only and not scored:
        notes.append(
            "MEASUREMENT GAP: coder edited train.py/loso_cv.py/gcn.py but not "
            "neuroasd/model.py or autoresearch/trial.py. Trials call "
            "neuroasd.model.fit_predict, so this trial may not measure the change."
        )
    return notes


STAGE_TIMEOUT = {
    "screen": 20 * 60,
    "loso-site": 90 * 60,
    "loso-institution": 90 * 60,
}


class ExperimenterTools:
    """Run autoresearch.trial and interpret the gate, not the prose."""

    def __init__(self, *, promote_on_pass: bool = True, push: bool = True):
        self.promote_on_pass = promote_on_pass
        self.push = push
        self.touched: list[str] = []
        self._ran_trial = False

    def as_tools(self) -> list:
        return [
            self.read_workspace,
            self.read_gates,
            self.read_ledger,
            self.read_program,
            self.run_trial,
            self.read_result,
            self.store_insight,
        ]

    def sources(self) -> list[str]:
        return list(self.touched)

    def read_workspace(self) -> dict[str, Any]:
        """Return the current write→run session state (status, hypothesis, files)."""
        return load_workspace()

    def read_gates(self) -> dict[str, Any]:
        """Return the pass thresholds and measured baselines."""
        return json.loads(GATES_PATH.read_text(encoding="utf-8"))

    def read_program(self) -> str:
        """Return the autoresearch agent rules (search space, no-leak protocol)."""
        return PROGRAM_PATH.read_text(encoding="utf-8")

    def read_ledger(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent trial records so dead ends are not repeated.

        Args:
            limit: Maximum number of trailing ledger rows (1-100).
        """
        limit = max(1, min(int(limit), 100))
        if not LEDGER_PATH.exists():
            return []
        rows = [
            json.loads(line)
            for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows[-limit:]

    def read_result(self, name: str, stage: str) -> dict[str, Any]:
        """Load a written trial result.json.

        Args:
            name: Trial slug, e.g. 'iter1_class-weight'.
            stage: 'screen', 'loso-site', or 'loso-institution'.
        """
        path = trials_dir() / f"{stage}__{name}" / "result.json"
        if not path.is_file():
            raise FileNotFoundError(str(path.relative_to(REPO_ROOT)))
        self.touched.append(str(path.relative_to(REPO_ROOT)))
        return json.loads(path.read_text(encoding="utf-8"))

    def run_trial(self, note: str = "") -> dict[str, Any]:
        """Train and score the current code at the workspace-required stage.

        The stage is taken from the workspace (screen → loso-site → loso-institution).
        You cannot pick a more expensive stage yourself.

        Args:
            note: Optional extra note stored on the result (defaults to hypothesis).
        """
        if self._ran_trial:
            return {
                "ok": False,
                "error": (
                    "this experimenter already ran one stage. "
                    "A fresh instance will run the next stage if the gate passed."
                ),
            }
        workspace = load_workspace()
        if not coder_finished_cleanly(workspace):
            return {
                "ok": False,
                "error": (
                    "coder did not finish cleanly; refusing to train a partial edit. "
                    f"coder_ok={workspace.get('coder_ok')!r} "
                    f"error={workspace.get('coder_error') or 'unfinished tool loop'}"
                ),
                "workspace": workspace,
            }
        if not lint_passed(workspace):
            return {
                "ok": False,
                "error": (
                    "lint has not passed; code is not qualified to train. "
                    f"lint_ok={workspace.get('lint_ok')!r}"
                ),
                "workspace": workspace,
            }
        stage = required_stage(workspace)
        if stage is None:
            return {
                "ok": False,
                "error": (
                    f"no trial to run (status={workspace.get('status')}). "
                    "The coder must land a new one-thing change first."
                ),
                "workspace": workspace,
            }
        name = str(workspace.get("current_name") or "").strip()
        if not name:
            return {"ok": False, "error": "workspace has no current_name"}

        self._ran_trial = True
        hypothesis = note.strip() or str(workspace.get("hypothesis") or "")
        command = [
            sys.executable,
            "-m",
            "autoresearch.trial",
            "--name",
            name,
            "--stage",
            stage,
            "--note",
            hypothesis,
        ]
        env = os.environ.copy()
        env.setdefault("AUTORESEARCH_TRIAL_WORKERS", "2")
        # One BLAS thread per fold worker so 2 folds do not oversubscribe an Air.
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env.setdefault(key, "1")
        ran = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=STAGE_TIMEOUT[stage],
            env=env,
            check=False,
        )
        result_path = trials_dir() / f"{stage}__{name}" / "result.json"
        if not result_path.is_file():
            stdout_tail = ran.stdout[-2000:]
            stderr_tail = ran.stderr[-2000:]
            workspace = record_unmeasured_trial(
                "trial produced no result.json",
                exit_code=ran.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
            self.touched.append(f"crash:{stage}")
            return {
                "ok": False,
                "error": "trial produced no result.json",
                "exit_code": ran.returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "workspace_status": workspace.get("status"),
            }

        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.touched.append(str(result_path.relative_to(REPO_ROOT)))
        notes = protocol_notes(result)
        failed_protocol = not protocol_ok(result)
        if failed_protocol:
            result["verdict"] = {
                **result.get("verdict", {}),
                "passed": False,
                "protocol_failed": True,
                "protocol_notes": notes,
            }

        workspace = apply_trial_outcome(
            stage,
            name,
            result["verdict"],
            result.get("summary") or {},
            protocol_failed=failed_protocol,
        )

        promotion = None
        last = (workspace.get("last_results") or {}).get(stage) or {}
        if (
            self.promote_on_pass
            and stage == "loso-institution"
            and last.get("passed")
            and not failed_protocol
        ):
            promotion = promote(result, push=self.push)

        summary = result.get("summary") or {}
        verdict = result.get("verdict") or {}
        hint = _insight_hint(stage, verdict, failed_protocol, workspace)
        payload = {
            "ok": True,
            "name": name,
            "stage": stage,
            "hypothesis": hypothesis,
            "files_changed": workspace.get("files_changed"),
            "balanced_accuracy_mean": summary.get("balanced_accuracy_mean"),
            "accuracy_mean": summary.get("accuracy_mean"),
            "auc_mean": summary.get("auc_mean"),
            "sensitivity_mean": summary.get("sensitivity_mean"),
            "specificity_mean": summary.get("specificity_mean"),
            "model_selection": result.get("model_selection"),
            "gate": {
                "metric": verdict.get("metric"),
                "threshold": verdict.get("threshold"),
                "observed": verdict.get("observed"),
                "margin": verdict.get("margin"),
                "passed": verdict.get("passed"),
            },
            "protocol_notes": notes,
            "workspace_status": workspace.get("status"),
            "promotion": promotion,
            "insight_hint": hint,
            "measurement_notes": measurement_coverage(workspace.get("files_changed")),
            "exit_code": ran.returncode,
        }
        save_insight(
            {
                "name": name,
                "stage": stage,
                "hypothesis": hypothesis,
                "balanced_accuracy_mean": summary.get("balanced_accuracy_mean"),
                "auc_mean": summary.get("auc_mean"),
                "margin": verdict.get("margin"),
                "passed": bool(verdict.get("passed")) and not failed_protocol,
                "ruled_out": None if payload["gate"]["passed"] else hypothesis,
                "hint": hint,
                "narrative": "",
                "next_code_change": "",
            }
        )
        return payload

    def store_insight(
        self,
        narrative: str,
        next_code_change: str = "",
        ruled_out: str = "",
    ) -> dict[str, Any]:
        """Save a coder-actionable insight into the workspace.

        The next coder instance is fresh and cannot see this conversation.
        This write-up is how it knows what to try next.

        Args:
            narrative: What was measured, balanced accuracy vs the gate, and why.
            next_code_change: One concrete code change to try next, or empty if
                the same code should continue to the next evaluation stage.
            ruled_out: What this result eliminates, if anything.
        """
        workspace = load_workspace()
        insight = dict(workspace.get("last_insight") or {})
        insight["narrative"] = narrative.strip()
        insight["next_code_change"] = next_code_change.strip()
        insight["ruled_out"] = ruled_out.strip() or insight.get("ruled_out")
        save_insight(insight)
        return {"ok": True, "last_insight": insight}


def _insight_hint(
    stage: str,
    verdict: dict[str, Any],
    failed_protocol: bool,
    workspace: dict[str, Any],
) -> str:
    if failed_protocol:
        return "Protocol failed. Discard the number. Do not iterate on it."
    if not verdict.get("passed"):
        return (
            f"{stage} FAIL (margin {verdict.get('margin')}). "
            "This change is ruled out at this stage. Next: coder, one new mechanism."
        )
    if stage == "screen":
        return (
            "screen PASS is only a filter. Do not claim the model is better. "
            "Next: experimenter runs loso-site on the same code."
        )
    if stage == "loso-site":
        return (
            "loso-site PASS. Next: experimenter runs loso-institution on the same code. "
            "Still not a published improvement."
        )
    last = (workspace.get("last_results") or {}).get(stage) or {}
    if last.get("did_not_beat_last_win"):
        return (
            "loso-institution met the gate but did not beat the last win's "
            "balanced accuracy. Reverted to the winning snapshot. Next: coder, "
            "one new mechanism."
        )
    last_win = workspace.get("last_win") or {}
    acc = last_win.get("accuracy_mean")
    if workspace.get("status") == "target_reached":
        return (
            "loso-institution PASS and accuracy target reached. A review PR may be open. "
            "Loop stops. Do not raise gates.json."
        )
    return (
        "loso-institution PASS. Keep this code as the new baseline and add ONE new "
        f"mechanism. Last win balanced acc {last_win.get('balanced_accuracy_mean')} "
        f"accuracy {acc}. A review PR may be open; do not wait for merge. "
        "Do not raise gates.json."
    )


class ExperimenterAgent(SubAgent):
    """Runs one gated trial and returns an insight."""

    name = "experimenter"
    description = (
        "Scores the current model with autoresearch.trial (balanced accuracy "
        "against the linear baseline), and returns what the result rules in or out. "
        "Use after the coder has landed a change, or to promote a passing change "
        "to the next stage. Not for writing model code."
    )
    instructions = INSTRUCTIONS

    def __init__(self, *, promote: bool = True, push: bool = True, **kwargs: Any):
        kwargs.setdefault("max_tool_rounds", 24)
        super().__init__(**kwargs)
        self._promote = promote
        self._push = push

    def create_tools(self) -> ExperimenterTools:
        return ExperimenterTools(promote_on_pass=self._promote, push=self._push)
