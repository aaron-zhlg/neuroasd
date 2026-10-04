"""Exercise the write → measure → insight → rewrite state machine.

This does not call an LLM and does not train. It checks whether the
loop's bookkeeping has bugs: role order, insight handoff, frozen code, and
the historical best-epoch leak guard.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

# Isolated session files, set before importing workspace-backed modules.
_TMP = Path(tempfile.mkdtemp(prefix="autoresearch-loop-"))
os.environ["AUTORESEARCH_LOOP_WORKSPACE"] = str(_TMP / "workspace.json")
os.environ["AUTORESEARCH_LOOP_TRIALS_DIR"] = str(_TMP / "trials")
os.environ["AUTORESEARCH_LOOP_RESULTS_DIR"] = str(_TMP / "results")
os.environ["AUTORESEARCH_LOOP_WRITE_DIR"] = str(_TMP)

from autoresearch.loop.coder import CodeTools  # noqa: E402
from autoresearch.loop.experimenter import ExperimenterTools, measurement_coverage  # noqa: E402
from autoresearch.loop.linter import lint_paths  # noqa: E402
from autoresearch.loop.orchestrator import _forced_assignment  # noqa: E402
from autoresearch.loop.promote import pr_body, pr_title  # noqa: E402
from autoresearch.loop.protocol import leak_reasons_in_source, protocol_ok  # noqa: E402
from autoresearch.loop.workspace import (  # noqa: E402
    apply_trial_outcome,
    ensure_stage_recorded,
    load_workspace,
    mark_coder_outcome,
    mark_pending_stage,
    next_role,
    record_lint,
    record_unmeasured_trial,
    save_insight,
)

PASSED = 0
FAILED = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok   {name}")
        return
    FAILED += 1
    print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def _fake_verdict(stage: str, passed: bool, auc: float = 0.62) -> dict:
    return {
        "stage": stage,
        "metric": "auc_mean",
        "threshold": 0.63,
        "observed": auc,
        "margin": round(auc - 0.63, 4),
        "passed": passed,
    }


def _write_scratch(tools: CodeTools, name: str, body: str = "x = 1\n") -> dict:
    path = _TMP / name
    return tools.write_file(str(path), body)


def test_role_order() -> None:
    print("role order")
    _check("idle starts with coder", next_role() == "coder")
    forced = _forced_assignment(None, "improve the gnn")
    _check("lead forces coder first", forced is not None and forced.subagent == "coder")


def test_write_then_measure() -> None:
    print("write → measure")
    tools = CodeTools()
    tools.read_last_insight()
    result = _write_scratch(tools, "change.py", "value = 1\n")
    workspace = load_workspace()
    _check("write opens needs_screen", workspace["status"] == "needs_screen", workspace["status"])
    _check("write records file", bool(workspace["files_changed"]), str(workspace["files_changed"]))
    _check("unfinished coder is still next", next_role() == "coder")
    _check("write reports iteration 1", result["iteration"] == 1)
    mark_coder_outcome(True)
    _check("after clean coder, linter is next", next_role() == "linter")
    record_lint({"passed": True, "errors": [], "files_checked": workspace["files_changed"]})
    _check("after lint PASS, experimenter is next", next_role() == "experimenter")
    _check("lint+coder allow a trial attempt", required_stage_ok())


def required_stage_ok() -> bool:
    from autoresearch.loop.workspace import coder_finished_cleanly, lint_passed, required_stage

    workspace = load_workspace()
    return (
        coder_finished_cleanly(workspace)
        and lint_passed(workspace)
        and required_stage(workspace) == "screen"
    )


def test_idle_trial_refused() -> None:
    print("trial refused when idle")
    # Reset by writing a fresh workspace through a new fail path: set status idle.
    from autoresearch.loop.workspace import default_workspace, save_workspace

    save_workspace(default_workspace())
    tools = ExperimenterTools(promote_on_pass=False, push=False)
    payload = tools.run_trial()
    _check("run_trial idle is not ok", payload.get("ok") is False)
    save_workspace(default_workspace())


def test_fail_insight_then_coder() -> None:
    print("fail → insight → fresh coder")
    from autoresearch.loop.workspace import save_workspace, default_workspace

    save_workspace(default_workspace())
    writer = CodeTools()
    writer.read_last_insight()
    writer.record_hypothesis("tiny dropout tweak")
    _write_scratch(writer, "model.py", "dropout = 0.4\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})

    apply_trial_outcome(
        "screen",
        load_workspace()["current_name"],
        _fake_verdict("screen", False, 0.61),
        {"auc_mean": 0.61},
        protocol_failed=False,
    )
    save_insight(
        {
            "stage": "screen",
            "passed": False,
            "auc_mean": 0.61,
            "ruled_out": "tiny dropout tweak",
            "next_code_change": "try class weights, not another dropout",
            "narrative": "screen FAIL. dropout 0.4 did not beat 0.63.",
        }
    )
    workspace = load_workspace()
    _check("FAIL goes to awaiting_new_code", workspace["status"] == "awaiting_new_code")
    _check("FAIL next role is coder", next_role() == "coder")
    _check("FAIL resets coder_ok", workspace.get("coder_ok") is None)
    _check("FAIL reverts the scratch edit", not (_TMP / "model.py").exists())
    _check("insight persisted", bool(workspace.get("last_insight")))
    ruled = workspace.get("ruled_out") or []
    _check("FAIL records ruled_out", bool(ruled), str(ruled))
    patch = Path((ruled[0] or {}).get("patch") or "")
    _check(
        "FAIL archives the failed diff",
        patch.is_dir() and (patch / "files").exists(),
        str(patch),
    )
    _check(
        "ruled_out summary from insight",
        any(item.get("summary") == "tiny dropout tweak" for item in ruled),
        str(ruled),
    )

    blind = CodeTools()
    try:
        _write_scratch(blind, "model.py", "dropout = 0.3\n")
        _check("blind coder blocked without insight", False, "write succeeded")
    except PermissionError as exc:
        _check("blind coder blocked without insight", "read_last_insight" in str(exc), str(exc))

    informed = CodeTools()
    insight = informed.read_last_insight()
    _check(
        "fresh coder sees next_code_change",
        insight["last_insight"]["next_code_change"].startswith("try class weights"),
    )
    _check(
        "fresh coder sees the ruled_out list",
        bool(insight.get("ruled_out"))
        and insight["ruled_out"][0].get("hypothesis") == "tiny dropout tweak",
        str(insight.get("ruled_out")),
    )
    informed.record_hypothesis("class-weighted loss")
    second = _write_scratch(informed, "model.py", "weight = True\n")
    _check("insight-driven write starts iteration 2", second["iteration"] == 2)
    _check("new write returns to needs_screen", load_workspace()["status"] == "needs_screen")


def test_promotion_freezes_code() -> None:
    print("promotion freezes coder")
    from autoresearch.loop.workspace import default_workspace, save_workspace

    save_workspace(default_workspace())
    tools = CodeTools()
    tools.read_last_insight()
    _write_scratch(tools, "gcn_edit.py", "hidden = 64\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})
    name = load_workspace()["current_name"]
    apply_trial_outcome("screen", name, _fake_verdict("screen", True, 0.64), {"auc_mean": 0.64}, False)
    _check("screen PASS → loso-site", load_workspace()["status"] == "needs_loso_site")
    _check("subset still experimenter", next_role() == "experimenter")

    frozen = CodeTools()
    frozen.read_last_insight()
    try:
        _write_scratch(frozen, "gcn_edit.py", "hidden = 128\n")
        _check("coder frozen on loso-site", False, "write succeeded")
    except PermissionError as exc:
        _check("coder frozen on loso-site", "frozen" in str(exc), str(exc))

    apply_trial_outcome(
        "loso-site", name, _fake_verdict("loso-site", True, 0.68), {"auc_mean": 0.68}, False
    )
    apply_trial_outcome(
        "loso-institution",
        name,
        _fake_verdict("loso-institution", True, 0.67),
        {"auc_mean": 0.67, "accuracy_mean": 0.60},
        False,
    )
    after_win = load_workspace()
    _check(
        "loso-institution PASS below target continues",
        after_win["status"] == "awaiting_new_code",
        after_win["status"],
    )
    _check("next role is coder after a win", next_role() == "coder")
    _check(
        "win keeps the scratch file",
        (_TMP / "gcn_edit.py").read_text() == "hidden = 64\n",
    )
    stacked = CodeTools()
    stacked.read_last_insight()
    stacked.record_hypothesis("stack a worse change")
    _write_scratch(stacked, "gcn_edit.py", "hidden = 256\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})
    apply_trial_outcome(
        "screen",
        load_workspace()["current_name"],
        _fake_verdict("screen", False, 0.61),
        {"auc_mean": 0.61},
        False,
    )
    _check(
        "FAIL after a win restores the snapshot",
        (_TMP / "gcn_edit.py").read_text() == "hidden = 64\n",
        (_TMP / "gcn_edit.py").read_text() if (_TMP / "gcn_edit.py").exists() else "missing",
    )
    apply_trial_outcome(
        "loso-institution",
        name,
        _fake_verdict("loso-institution", True, 0.70),
        {"auc_mean": 0.70, "accuracy_mean": 0.81},
        False,
    )
    _check("accuracy target stops the loop", load_workspace()["status"] == "target_reached")
    _check("target reached has no next role", next_role() is None)


def test_protocol_and_coverage() -> None:
    print("protocol guards")
    _check(
        "best-epoch gate source rejected",
        bool(leak_reasons_in_source('observed = summary["best_auc_mean"]')),
    )
    leak = {
        "model_selection": "best epoch",
        "verdict": {"metric": "best_auc_mean"},
        "summary": {"auc_mean": 0.62, "best_auc_mean": 0.70},
    }
    _check("best-epoch result is protocol fail", not protocol_ok(leak))
    honest = {
        "model_selection": "final fit (no selection on the evaluation set)",
        "verdict": {"metric": "balanced_accuracy_mean"},
        "summary": {"auc_mean": 0.62},
    }
    _check("final-fit result is protocol ok", protocol_ok(honest))
    notes = measurement_coverage(["neuroasd/train.py"])
    _check("train.py-only change warns measurement gap", any("MEASUREMENT GAP" in n for n in notes))
    _check(
        "gcn.py-only change warns measurement gap",
        any("MEASUREMENT GAP" in n for n in measurement_coverage(["neuroasd/gcn.py"])),
    )
    _check(
        "model.py-only change has no gap",
        measurement_coverage(["neuroasd/model.py"]) == [],
    )


def test_coder_fail_and_lint_block_training() -> None:
    print("unqualified code cannot train")
    from autoresearch.loop.workspace import default_workspace, save_workspace

    save_workspace(default_workspace())
    tools = CodeTools()
    tools.read_last_insight()
    _write_scratch(tools, "broken.py", "def (\n")
    mark_coder_outcome(False, "tool loop did not settle within 16 rounds")
    _check("coder FAIL next is coder", next_role() == "coder")
    refused = ExperimenterTools(promote_on_pass=False, push=False).run_trial()
    _check("coder FAIL blocks trial", refused.get("ok") is False and "coder" in refused["error"])

    mark_coder_outcome(True)
    _check("clean coder without lint → linter", next_role() == "linter")
    report = lint_paths(load_workspace()["files_changed"])
    _check("syntax error fails lint", report["passed"] is False, str(report["errors"]))
    record_lint(report)
    _check("lint FAIL next is coder", next_role() == "coder")
    refused_lint = ExperimenterTools(promote_on_pass=False, push=False).run_trial()
    _check("lint FAIL blocks trial", refused_lint.get("ok") is False and "lint" in refused_lint["error"])
    _check("lint insight is waiting for coder", "Lint FAIL" in str(load_workspace()["last_insight"]))


def test_pr_body_leads_with_scores() -> None:
    print("review PR leads with scores")
    result = {
        "name": "dropout03",
        "stage": "loso-institution",
        "note": "less regularisation",
        "summary": {
            "balanced_accuracy_mean": 0.671,
            "balanced_accuracy_std": 0.08,
            "auc_mean": 0.75,
            "auc_std": 0.09,
            "accuracy_mean": 0.62,
            "accuracy_std": 0.1,
            "sensitivity_mean": 0.58,
            "sensitivity_std": 0.1,
            "specificity_mean": 0.76,
            "specificity_std": 0.1,
            "f1_mean": 0.61,
            "f1_std": 0.11,
        },
        "verdict": {
            "metric": "balanced_accuracy_mean",
            "observed": 0.671,
            "threshold": 0.66,
            "margin": 0.011,
            "passed": True,
            "delta_site": 0.012,
            "delta_institution": 0.011,
        },
    }
    title = pr_title(result)
    body = pr_body(
        result,
        prior={"screen": {"balanced_accuracy_mean": 0.64, "margin": 0.01, "passed": True}},
    )
    _check("PR title has balanced acc and PASS", "0.671" in title and "PASS" in title, title)
    _check("PR body starts with Scores", body.lstrip().startswith("## Scores"), body[:80])
    _check(
        "balanced accuracy appears before hypothesis",
        body.index("**Balanced accuracy**") < body.index("## Hypothesis"),
    )


def test_promote_requires_training_source() -> None:
    print("promote commits training source, not just results")
    from autoresearch.loop.promote import source_files_to_promote
    from autoresearch.loop.workspace import default_workspace, save_workspace

    data = default_workspace()
    data["files_changed"] = ["neuroasd/model.py"]
    save_workspace(data)
    sources = source_files_to_promote()
    _check(
        "files_changed model.py is promoted even if git status misses it",
        any(path.name == "model.py" and path.is_file() for path in sources),
        str(sources),
    )
    from autoresearch.loop.promote import freeze_relpaths

    frozen = freeze_relpaths("iter25_cosine", "loso-institution")
    _check(
        "freeze paths are this trial only",
        all("iter25_cosine" in path for path in frozen)
        and not any("iter6" in path or "iter1_" in path for path in frozen),
        str(frozen),
    )


def test_crash_or_skip_does_not_respin_experimenter() -> None:
    print("unmeasured trial cannot respin experimenter")
    from autoresearch.loop.workspace import default_workspace, save_workspace

    save_workspace(default_workspace())
    writer = CodeTools()
    writer.read_last_insight()
    writer.record_hypothesis("broken inner split")
    _write_scratch(writer, "broken.py", "value = 1\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})
    _check("qualified code still sends experimenter", next_role() == "experimenter")
    mark_pending_stage()

    record_unmeasured_trial("trial produced no result.json", exit_code=1)
    after_crash = load_workspace()
    _check(
        "crash goes to awaiting_new_code",
        after_crash["status"] == "awaiting_new_code",
        after_crash["status"],
    )
    _check("crash next role is coder", next_role() == "coder")
    _check("crash reverts the edit", not (_TMP / "broken.py").exists())
    _check(
        "crash records the stage",
        bool((after_crash.get("last_results") or {}).get("screen")),
        str(after_crash.get("last_results")),
    )
    _check("crash records ruled_out", bool(after_crash.get("ruled_out")))

    save_workspace(default_workspace())
    writer = CodeTools()
    writer.read_last_insight()
    writer.record_hypothesis("skipped measurement")
    _write_scratch(writer, "skipped.py", "value = 2\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})
    mark_pending_stage()
    ensure_stage_recorded()
    after_skip = load_workspace()
    _check(
        "empty experimenter is fail-closed",
        after_skip["status"] == "awaiting_new_code" and next_role() == "coder",
        f"status={after_skip['status']} role={next_role()}",
    )

    save_workspace(default_workspace())
    writer = CodeTools()
    writer.read_last_insight()
    _write_scratch(writer, "ok.py", "value = 3\n")
    mark_coder_outcome(True)
    record_lint({"passed": True, "errors": []})
    mark_pending_stage()
    name = load_workspace()["current_name"]
    apply_trial_outcome(
        "screen", name, _fake_verdict("screen", True, 0.64), {"auc_mean": 0.64}, False
    )
    ensure_stage_recorded()
    after_pass = load_workspace()
    _check(
        "screen PASS is not treated as a skip",
        after_pass["status"] == "needs_loso_site" and next_role() == "experimenter",
        f"status={after_pass['status']} role={next_role()}",
    )


def test_experimenter_runs_one_stage() -> None:
    print("one stage per experimenter")
    tools = ExperimenterTools(promote_on_pass=False, push=False)
    tools._ran_trial = True
    refused = tools.run_trial()
    _check("second run_trial refused", refused.get("ok") is False, str(refused))
    _check(
        "reason is already ran one stage",
        "already ran" in str(refused.get("error")),
        str(refused.get("error")),
    )


def test_either_split_gate() -> None:
    print("either-split gate")
    from autoresearch.trial import either_split_improves

    rule = {"improve_margin": 0.01, "regress_tolerance": 0.005}
    win = either_split_improves(0.685, 0.672, 0.6735, 0.6747, rule)
    _check("site +1.15pp / institution -0.27pp PASSes", win["passed"] and "site" in win["improved_on"])
    lose = either_split_improves(0.690, 0.660, 0.6735, 0.6747, rule)
    _check("site win with institution -1.5pp FAILs", not lose["passed"])
    both = either_split_improves(0.690, 0.690, 0.6735, 0.6747, rule)
    _check("both splits +1.5pp PASSes", both["passed"])


def test_coder_cannot_write_gates() -> None:
    print("path guards")
    tools = CodeTools()
    tools.read_last_insight()
    try:
        tools.write_file("autoresearch/gates.json", "{}")
        _check("gates.json not writable", False)
    except PermissionError:
        _check("gates.json not writable", True)


def main() -> None:
    print(f"loop check  tmp={_TMP}\n")
    tests = [
        test_role_order,
        test_idle_trial_refused,
        test_write_then_measure,
        test_fail_insight_then_coder,
        test_promotion_freezes_code,
        test_protocol_and_coverage,
        test_coder_fail_and_lint_block_training,
        test_pr_body_leads_with_scores,
        test_promote_requires_training_source,
        test_crash_or_skip_does_not_respin_experimenter,
        test_experimenter_runs_one_stage,
        test_either_split_gate,
        test_coder_cannot_write_gates,
    ]
    for test in tests:
        try:
            test()
        except Exception:
            global FAILED
            FAILED += 1
            print(f"  FAIL {test.__name__} crashed")
            traceback.print_exc()
        print()
    print(f"{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
