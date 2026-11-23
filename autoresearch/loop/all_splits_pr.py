"""After a review PR opens: score its branch under the k-fold schemes and comment.

Runs in the background (started by the experimenter, never blocks the loop).
Checks the PR branch out into a temporary worktree, runs `all_splits` for
random-kfold and site-kfold, takes leave-one-site-out / leave-one-institution-out
from the branch's own result.json, writes all four to
`outputs/autoresearch/all_splits/<name>/` and posts the table on the PR.

    python -m autoresearch.loop.all_splits_pr <branch> <trial name> [<pr url>]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from autoresearch.loop.paths import REPO_ROOT

DATA_LINKS = {
    "data/abide_dcan/processed": REPO_ROOT / "data" / "abide_dcan" / "processed",
    "data/abide_dcan_hcp": REPO_ROOT / "data" / "abide_dcan_hcp",
    "data/abide_dcan_markov": REPO_ROOT / "data" / "abide_dcan_markov",
    "data/abide_dcan_power": REPO_ROOT / "data" / "abide_dcan_power",
}


def _run(cmd: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def main() -> None:
    branch, name = sys.argv[1], sys.argv[2]
    pr_url = sys.argv[3] if len(sys.argv) > 3 else ""
    out = REPO_ROOT / "outputs" / "autoresearch" / "all_splits" / name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("a", encoding="utf-8")
    work = REPO_ROOT.parent / f"split-{name[:40]}"
    try:
        _run(["git", "fetch", "-q", "origin", branch], REPO_ROOT)
        _run(["git", "worktree", "remove", "--force", str(work)], REPO_ROOT)
        made = _run(["git", "worktree", "add", "-q", "--detach", str(work), f"origin/{branch}"], REPO_ROOT)
        if made.returncode != 0:
            log.write(made.stderr)
            return
        for rel_link, target in DATA_LINKS.items():
            link = work / rel_link
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(os.path.realpath(target))
        (work / ".venv").symlink_to(os.path.realpath(REPO_ROOT / ".venv"))
        ran = _run([str(work / ".venv" / "bin" / "python"), "-m", "autoresearch.loop.all_splits", "--name", name,
                    "--out-dir", str(out), "--schemes", "random-kfold", "site-kfold"], work,
                   env={**os.environ, "PYTHONWARNINGS": "ignore"})
        log.write(ran.stdout + ran.stderr)
        if ran.returncode != 0:
            return
        kfold = json.loads((out / "all_splits.json").read_text(encoding="utf-8"))["results"]
        result_path = work / "autoresearch" / "results" / f"loso-institution__{name}" / "result.json"
        summary = json.loads(result_path.read_text(encoding="utf-8"))["summary"] if result_path.is_file() else {}
        rows = [(r["scheme"], r["fold_mean"]) for r in kfold]
        if summary:
            for scheme, prefix in (("loso-site", "site_"), ("loso-institution", "")):
                rows.append((scheme, {f"{k}_mean": summary.get(f"{prefix}{k}_mean") for k in
                                      ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "auc")}))
        from autoresearch.loop.all_splits import LABELS

        lines = [f"### 各种测法的 recall：{name}", "", "1,120 名开发集受试者，每人单独判断，锁定的 323 人未使用。", "",
                 "| 测法 | ASD recall | 对照 recall | 平衡准确率 | AUC |", "|---|---|---|---|---|"]
        for scheme, m in rows:
            if m.get("sensitivity_mean") is None:
                continue
            lines.append(f"| {LABELS[scheme]} | {m['sensitivity_mean']:.1%} | {m['specificity_mean']:.1%} | "
                         f"{m['balanced_accuracy_mean']:.1%} | {m['auc_mean']:.3f} |")
        lines += ["", "随机 10 折、按站点分层 10 折：测试的人来自训练见过的站点（多数论文的测法）。"
                  "留一站点、留一机构：测试站点训练时没见过。"]
        table = "\n".join(lines) + "\n"
        (out / "all_splits.md").write_text(table, encoding="utf-8")
        if pr_url and shutil.which("gh"):
            posted = _run(["gh", "pr", "comment", pr_url, "--body", table], REPO_ROOT)
            log.write(posted.stdout + posted.stderr)
    finally:
        _run(["git", "worktree", "remove", "--force", str(work)], REPO_ROOT)
        log.close()


if __name__ == "__main__":
    main()
