"""Promote a loso-institution PASS: freeze results and open a review PR.

Commits only this trial's training/evaluation files plus its frozen experiment
folder, on a branch cut from main. Never merges to main and never edits
gates.json. The PR body leads with scores.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from autoresearch.loop.paths import EXPERIMENTS_DIR, REPO_ROOT, rel, results_dir
from autoresearch.loop.workspace import load_workspace


def _git(*args: str, cwd: Path | None = None) -> str:
    ran = subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ran.returncode != 0:
        raise RuntimeError(ran.stderr.strip() or ran.stdout.strip() or "git failed")
    return ran.stdout.strip()


def _promotion_base() -> str:
    for ref in ("origin/main", "main"):
        ran = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if ran.returncode == 0:
            return ref
    raise RuntimeError("no main ref to branch the review PR from")


def _is_training_source(path: Path) -> bool:
    neuroasd = (REPO_ROOT / "neuroasd").resolve()
    trial = (REPO_ROOT / "autoresearch" / "trial.py").resolve()
    resolved = path.resolve()
    return resolved == trial or resolved.is_relative_to(neuroasd)


def _changed_promotable_files() -> list[Path]:
    raw = _git("status", "--porcelain")
    paths: list[Path] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        rel_path = line[3:].strip().strip('"')
        if " -> " in rel_path:
            rel_path = rel_path.split(" -> ", 1)[1]
        path = (REPO_ROOT / rel_path).resolve()
        if path.is_file() and _is_training_source(path):
            paths.append(path)
    return paths


def source_files_to_promote() -> list[Path]:
    """Model/training files that produced the score. Porcelain alone is not enough.

    Also include neuroasd/*.py and trial.py that differ from main, so a stacked
    winner still ships the full model even if this iteration only touched one file.
    """
    found: dict[str, Path] = {}
    for path in _changed_promotable_files():
        found[rel(path)] = path
    for rel_path in load_workspace().get("files_changed") or []:
        path = (REPO_ROOT / str(rel_path)).resolve()
        if path.is_file() and _is_training_source(path):
            found[rel(path)] = path
    try:
        base = _promotion_base()
        named = _git("diff", "--name-only", base, "--", "neuroasd", "autoresearch/trial.py")
    except RuntimeError:
        named = ""
    for rel_path in named.splitlines():
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        path = (REPO_ROOT / rel_path).resolve()
        if path.is_file() and _is_training_source(path):
            found[rel(path)] = path
    return list(found.values())


def freeze_relpaths(name: str, stage: str) -> list[str]:
    """This trial's score files only — not earlier experiments/ or results/."""
    folder = EXPERIMENTS_DIR / f"{name}_v1"
    return [
        rel(results_dir() / f"{stage}__{name}" / "result.json"),
        rel(folder / "results.json"),
        rel(folder / "run_config.json"),
        rel(folder / "README.md"),
    ]


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pm(mean: Any, std: Any, digits: int = 3) -> str:
    if mean is None:
        return "n/a"
    text = _fmt(mean, digits)
    if std is None:
        return text
    return f"{text} ± {_fmt(std, digits)}"


def pr_title(result: dict[str, Any]) -> str:
    """One-line PR title that a reviewer can scan in the list."""
    summary = result.get("summary") or {}
    verdict = result.get("verdict") or {}
    name = result.get("name") or "trial"
    stage = result.get("stage") or "unknown"
    passed = "PASS" if verdict.get("passed") else "FAIL"
    margin = verdict.get("margin")
    margin_s = f"{float(margin):+.3f}" if isinstance(margin, (int, float)) else "n/a"
    return (
        f"Trial {name}: {stage} bal.acc {_fmt(summary.get('balanced_accuracy_mean'))} "
        f"({passed} {margin_s})"
    )


def pr_body(result: dict[str, Any], *, prior: dict[str, Any] | None = None) -> str:
    """PR description. Scores come first so review does not start at the diff."""
    summary = result.get("summary") or {}
    verdict = result.get("verdict") or {}
    workspace = load_workspace()
    if prior is None:
        prior = workspace.get("last_results") or {}
    hypothesis = workspace.get("hypothesis") or result.get("note") or ""
    files = ", ".join(workspace.get("files_changed") or []) or "(see git diff)"
    passed = "PASS" if verdict.get("passed") else "FAIL"
    margin = verdict.get("margin")
    margin_s = f"{float(margin):+.4f}" if isinstance(margin, (int, float)) else "n/a"
    name = result.get("name") or "trial"
    stage = result.get("stage") or "unknown"
    reproduce = (
        f"uv run python -m autoresearch.trial --name {name} --stage {stage}"
    )

    lines = [
        "## Scores",
        "",
        "Official metric is **fold-mean balanced accuracy** (one fit per fold). "
        "Accuracy and AUC are reported so the number can be compared with the literature.",
        "",
        f"**This trial (`{stage}`)**",
        "",
        "| Metric | Mean ± Std |",
        "|--------|------------|",
        f"| **Balanced accuracy** | **{_fmt_pm(summary.get('balanced_accuracy_mean'), summary.get('balanced_accuracy_std'))}** |",
        f"| Accuracy | {_fmt_pm(summary.get('accuracy_mean'), summary.get('accuracy_std'))} |",
        f"| AUC | {_fmt_pm(summary.get('auc_mean'), summary.get('auc_std'))} |",
        f"| ASD recall | {_fmt_pm(summary.get('sensitivity_mean'), summary.get('sensitivity_std'))} |",
        f"| Control recall | {_fmt_pm(summary.get('specificity_mean'), summary.get('specificity_std'))} |",
        "",
        (
            f"**Gate:** `{verdict.get('metric', 'balanced_accuracy_mean')}` "
            f"{_fmt(verdict.get('observed'), 4)} "
            + (
                f"site {verdict.get('delta_site', 'n/a')} / "
                f"institution {verdict.get('delta_institution', 'n/a')} "
                if verdict.get("delta_site") is not None
                else f"vs {_fmt(verdict.get('threshold'), 4)} (margin **{margin_s}**) "
            )
            + f"→ **{passed}**"
        ),
        "",
    ]

    ladder_rows = []
    for prior_stage in ("screen", "loso-site", "loso-institution"):
        row = prior.get(prior_stage)
        if not isinstance(row, dict):
            continue
        if prior_stage == stage:
            continue
        ladder_rows.append(
            f"| `{prior_stage}` | {_fmt(row.get('balanced_accuracy_mean') or row.get('auc_mean'))} | "
            f"{row.get('margin', 'n/a')} | "
            f"{'PASS' if row.get('passed') else 'FAIL'} |"
        )
    if ladder_rows:
        lines.extend(
            [
                "Earlier stages of the same change:",
                "",
                "| Stage | Bal. acc | Margin | Verdict |",
                "|-------|-----|--------|---------|",
                *ladder_rows,
                "",
            ]
        )

    lines.extend(
        [
            "## Hypothesis",
            "",
            hypothesis or "(none recorded)",
            "",
            f"**Files:** {files}",
            "",
            "## Reproduce",
            "",
            "```bash",
            reproduce,
            "```",
            "",
            "Do not merge to `main` until a human has reviewed the protocol and the diff. "
            "Do not raise `gates.json` from this PR.",
            "",
        ]
    )
    return "\n".join(lines)


def _gh(*args: str) -> str:
    ran = subprocess.run(
        ["gh", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ran.returncode != 0:
        raise RuntimeError(ran.stderr.strip() or ran.stdout.strip() or "gh failed")
    return ran.stdout.strip()


def open_review_pr(
    branch: str,
    result: dict[str, Any],
    *,
    base: str = "main",
) -> dict[str, Any]:
    """Open (or reuse) a review PR whose body leads with this trial's scores."""
    title = pr_title(result)
    existing = subprocess.run(
        ["gh", "pr", "view", branch, "--json", "url", "-q", ".url"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0 and existing.stdout.strip():
        return {
            "url": existing.stdout.strip(),
            "title": title,
            "created": False,
        }

    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(pr_body(result))
        body_file = handle.name
    try:
        url = _gh(
            "pr",
            "create",
            "--base",
            base,
            "--head",
            branch,
            "--title",
            title,
            "--body-file",
            body_file,
        )
    finally:
        Path(body_file).unlink(missing_ok=True)
    return {"url": url, "title": title, "created": True}


def write_experiment_freeze(result: dict[str, Any]) -> Path:
    """Write experiments/<name>_v1/ so the number can be reproduced later."""
    name = str(result["name"])
    folder = EXPERIMENTS_DIR / f"{name}_v1"
    folder.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    config = result.get("config") or {}
    (folder / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    run_config = {
        "name": f"{name}_v1",
        "validation": result["stage"],
        "model_selection": result.get(
            "model_selection", "final epoch (no selection on the evaluation set)"
        ),
        "data": {
            "dataset": "DCAN ABIDE I + II (Gordon / optional HCP, Power, Markov)",
            "num_subjects": 1443,
            "source": "data/abide_dcan",
        },
        "hyperparameters": config,
        "reproduce": (
            f"uv run python -m autoresearch.trial --name {name} "
            f"--stage {result['stage']}"
        ),
    }
    (folder / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    workspace = load_workspace()
    hypothesis = workspace.get("hypothesis") or result.get("note") or ""
    files = ", ".join(workspace.get("files_changed") or []) or "(see git diff)"
    readme = f"""# {name} v1 — Autoresearch promotion

Frozen after a `loso-institution` PASS. Metrics are one fit per fold.

> **Evaluated:** {date.today().isoformat()}
> **Hypothesis:** {hypothesis}
> **Files changed:** {files}

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Balanced accuracy** | **{summary.get('balanced_accuracy_mean', 0):.3f} ± {summary.get('balanced_accuracy_std', 0):.3f}** |
| Accuracy | {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f} |
| AUC | {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f} |
| ASD recall | {summary.get('sensitivity_mean', 0):.3f} ± {summary.get('sensitivity_std', 0):.3f} |
| Control recall | {summary.get('specificity_mean', 0):.3f} ± {summary.get('specificity_std', 0):.3f} |

## Reproduce

```bash
{run_config['reproduce']}
```

Do not merge to `main` until a human has reviewed the protocol and the diff.
"""
    (folder / "README.md").write_text(readme, encoding="utf-8")
    return folder


def promote(result: dict[str, Any], *, push: bool) -> dict[str, Any]:
    """Copy the score, freeze experiments/, commit a narrow trial branch."""
    name = str(result["name"])
    stage = str(result["stage"])
    if stage != "loso-institution":
        raise ValueError("promotion is only allowed after a loso-institution PASS")

    staged_result = results_dir() / f"{stage}__{name}" / "result.json"
    staged_result.parent.mkdir(parents=True, exist_ok=True)
    staged_result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    experiment_dir = write_experiment_freeze(result)

    source_files = source_files_to_promote()
    if not source_files:
        raise RuntimeError(
            "refusing to promote scores without the training/model source "
            "(neuroasd/*.py or autoresearch/trial.py) that produced them"
        )
    to_add = {rel(path) for path in source_files}
    to_add.update(freeze_relpaths(name, stage))
    for rel_path in to_add:
        if not (REPO_ROOT / rel_path).is_file():
            raise RuntimeError(f"promotion file missing: {rel_path}")

    branch = f"experiment/trial-{name}"
    existing = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=REPO_ROOT,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"branch {branch} already exists; review it by hand")

    base = _promotion_base()
    note = result.get("note") or load_workspace().get("hypothesis") or "no note"
    worktree = Path(tempfile.mkdtemp(prefix="neuroasd-promote-"))
    shutil.rmtree(worktree)
    pushed = False
    push_error = ""
    pr = None
    pr_error = ""
    try:
        _git("worktree", "add", "-b", branch, str(worktree), base)
        for rel_path in sorted(to_add):
            dest = worktree / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel_path, dest)
        _git("add", "--", *sorted(to_add), cwd=worktree)
        _git("commit", "-m", f"Trial {name} ({stage}): {note}", cwd=worktree)
        remotes = _git("remote")
        if push and "origin" in remotes.split():
            try:
                _git("push", "-u", "origin", branch, cwd=worktree)
                pushed = True
            except RuntimeError as exc:
                push_error = str(exc)
        if pushed:
            try:
                pr = open_review_pr(branch, result)
            except (RuntimeError, FileNotFoundError, OSError) as exc:
                pr_error = str(exc)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
        )

    return {
        "branch": branch,
        "files": sorted(to_add),
        "experiment_dir": rel(experiment_dir),
        "pushed": pushed,
        "push_error": push_error,
        "pr": pr,
        "pr_error": pr_error,
        "merged_to_main": False,
        "gates_updated": False,
        "note": "Human review required before merging to main or raising gates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a review PR whose description leads with trial scores."
    )
    parser.add_argument("--pr-from-result", required=True, help="Path to result.json")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base", default="main")
    args = parser.parse_args()
    result = json.loads(Path(args.pr_from_result).read_text(encoding="utf-8"))
    info = open_review_pr(args.branch, result, base=args.base)
    print(info.get("url") or json.dumps(info))


if __name__ == "__main__":
    main()
