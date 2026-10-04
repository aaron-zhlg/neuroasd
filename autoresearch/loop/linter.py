"""Lint agent: mechanical checks that the pending edit is qualified to train.

Pass/fail is decided by tools, not by the model. The experimenter must not run
until ``lint_ok`` is true.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

from orchestra import SubAgent

from autoresearch.loop.experimenter import measurement_coverage
from autoresearch.loop.paths import REPO_ROOT, resolve_repo_path
from autoresearch.loop.protocol import leak_reasons_in_source
from autoresearch.loop.workspace import load_workspace, record_lint

INSTRUCTIONS = """\
You lint the coder's pending edit. You do not write model code and you do not train.

Call lint_changed_files once. That tool decides PASS/FAIL. Return its report \
verbatim. Do not override a FAIL.
"""

_SCORED_IMPORT = (
    "from neuroasd.model import fit_predict\n"
    "from autoresearch.trial import smoke_check\n"
    "smoke_check()\n"
)


def _resolve_changed(path: str) -> Path | None:
    try:
        target = resolve_repo_path(path)
    except ValueError:
        raw = Path(path)
        target = raw if raw.is_absolute() else (REPO_ROOT / raw)
    if target.is_file():
        return target
    return None


def lint_paths(files_changed: list[str]) -> dict[str, Any]:
    """Return a structured lint report. ``passed`` is the only verdict that matters."""
    errors: list[str] = []
    notes: list[str] = []
    checked: list[str] = []

    if not files_changed:
        errors.append("no files_changed; coder produced nothing to lint")

    for rel_path in files_changed:
        target = _resolve_changed(rel_path)
        if target is None:
            errors.append(f"missing file: {rel_path}")
            continue
        checked.append(rel_path)
        source = target.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(target))
        except SyntaxError as exc:
            errors.append(f"syntax error in {rel_path}: {exc}")
        errors.extend(f"{rel_path}: {reason}" for reason in leak_reasons_in_source(source))

    errors.extend(measurement_coverage(files_changed))

    touches_scored_tree = any(
        path.endswith(("model.py", "trial.py", "linear_baseline.py"))
        for path in files_changed
    )
    if touches_scored_tree:
        probe = subprocess.run(
            [sys.executable, "-c", _SCORED_IMPORT],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            errors.append(
                "import check failed (neuroasd.model.fit_predict / trial.smoke_check):\n"
                + (probe.stderr or probe.stdout)[-1500:]
            )
        else:
            notes.append("scored-path smoke_check ok")

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "notes": notes,
        "files_checked": checked,
    }


class LintTools:
    """Hard lint gate for the pending coder edit."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def as_tools(self) -> list:
        return [self.lint_changed_files]

    def sources(self) -> list[str]:
        return list(self.touched)

    def lint_changed_files(self) -> dict[str, Any]:
        """Parse, leak-check, and import-check the coder's pending files.

        PASS/FAIL is decided here. A FAIL blocks training.

        Returns:
            Report with passed, errors, notes, and files_checked.
        """
        workspace = load_workspace()
        report = lint_paths(list(workspace.get("files_changed") or []))
        record_lint(report)
        self.touched = list(report.get("files_checked") or [])
        return report


class LintAgent(SubAgent):
    """Runs mechanical lint. The tool, not the model, decides if code is qualified."""

    name = "linter"
    description = (
        "Lints the coder's pending edit: syntax, evaluation-leak patterns, "
        "whether trial.py will actually score the change, and imports of "
        "neuroasd.model.fit_predict / autoresearch.trial. Use after a coder "
        "finishes and before any training. Not for writing code or running trials."
    )
    instructions = INSTRUCTIONS

    def __init__(self, *, promote: bool = True, push: bool = True, **kwargs: Any):
        kwargs.setdefault("max_tool_rounds", 8)
        super().__init__(**kwargs)

    def create_tools(self) -> LintTools:
        return LintTools()
