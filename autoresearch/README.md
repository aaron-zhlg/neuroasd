# Autoresearch

The lab and the PI for ASD vs control classification on DCAN ABIDE I + II.

`trial.py` + `gates.json` run one honest trial. `loop/` is a multi-agent
driver that keeps proposing one code change, linting it, measuring it, and
iterating. Built on [orchestra](https://github.com/aaron-zhlg/orchestra).

`program.md` is written for the agent. This file is written for you.

The loop starts from the formal linear baseline
(`experiments/linear_baseline_dcan_v1/`, `neuroasd/model.py`), not from the
old GCN PRs on C-PAC ABIDE I.

---

## Idea

Manual hyperparameter tuning is slow and easy to fool yourself with. This
framework makes four things mechanical:

1. **Cheap before expensive.** A full leave-one-institution-out sweep takes
   about an hour, so every idea is first screened on 6 sites.
2. **Honest scoring.** Gating uses one fit per fold, so no model is selected
   by peeking at the test fold.
3. **Nothing is lost, nothing is cluttered.** Failed trials are deleted as
   branches but recorded in a local ledger, so the same dead end is not
   explored twice.
4. **Write → lint → measure → insight.** Agents do not share a chat. Each
   worker is a new instance; the next coder only sees the last insight.

---

## Layout

```text
autoresearch/
├── program.md      instructions the agent reads before each trial
├── gates.json      pass thresholds; reference = linear_baseline_dcan_v1
├── trial.py        harness: data, folds, metrics; calls neuroasd.model.fit_predict
├── run_trial.sh    branch lifecycle: create, run, push or delete
├── results/        result.json of trials that passed (committed on trial branches)
└── loop/
    ├── orchestrator.py    lead + CLI
    ├── coder.py
    ├── linter.py
    ├── experimenter.py
    ├── workspace.py       session state and next_role()
    ├── protocol.py        final-fit / leak guards
    ├── promote.py         review branch + score-first PR
    └── check_loop.py      no-LLM checks of the state machine

neuroasd/model.py              the scored model (starts as the linear baseline)

outputs/autoresearch/          local only, gitignored
├── ledger.jsonl               append-only record of every trial in this reset
├── archive/                   old ABIDE I GCN loop memory (do not delete)
├── trials/<stage>__<name>/    per-trial result.json
└── loop/
    ├── workspace.json         multi-agent session (survives Ctrl-C)
    └── logs/
```

---

## Stages

| Stage | Evaluation | Gate |
|-------|------------|------|
| `screen` | LOSO on 6 mixed ABIDE I / II sites | balanced accuracy ≥ 0.64 |
| `loso-site` | LOSO on all 33 two-class sites | balanced accuracy ≥ 0.65 |
| `loso-institution` | Leave-one-institution-out, 23 institutions | either split +1 pp vs the linear baseline, other ≤ 0.5 pp drop |

Primary metric: fold-mean balanced accuracy. Every result also reports
accuracy, AUC, ASD recall, and control recall.

Only a `loso-institution` pass justifies a new directory under `experiments/`
and a review PR. A `screen` PASS is only a filter.

---

## Multi-agent loop

Each worker is a **new instance** with an empty context window. The lead
dispatches **one** of them per round. They do not reuse chat history.

Cross-round memory is a file, not a conversation:
`outputs/autoresearch/loop/workspace.json`. Restarting the process continues
from that file. `--fresh` wipes it.

| Agent | File | Job |
|-------|------|-----|
| Lead | `loop/orchestrator.py` | Plan → dispatch one worker → evaluate → synthesize. Does not edit or train. |
| Coder | `loop/coder.py` | One mechanism per turn, under `neuroasd/` or `trial.py`. Must read the last insight before editing. |
| Linter | `loop/linter.py` | Mechanical PASS/FAIL, including `fit_predict` smoke check. |
| Experimenter | `loop/experimenter.py` | Run the workspace-required stage via `trial.py`, then write an insight. |

```text
idle
  → coder writes one mechanism (usually neuroasd/model.py)
  → linter (syntax, no leak, smoke_check)
  → experimenter runs one required stage
  → screen FAIL   → revert the diff; coder (new idea)
  → screen PASS   → same code, loso-site
  → site PASS     → same code, loso-institution
  → institution PASS → review PR; keep the code; coder stacks the next mechanism
  → institution accuracy ≥ 0.80 → stop
```

Writable: `neuroasd/*.py` and `autoresearch/trial.py`. Not writable:
`gates.json`, `data/`, `experiments/`, `autoresearch/loop/`, `medresearch/`,
`.env`.

A protocol-clean `loso-institution` PASS opens a review PR (scores first),
snapshots the winning files as the new baseline, and **keeps searching**.
It does not wait for merge and does not raise `gates.json`. The loop stops
when `loso-institution` accuracy_mean reaches `--target-acc` (default 0.80)
or on Ctrl-C.

---

## Usage

```bash
./autoresearch/run_trial.sh --name class-weight --note "class-weighted ridge"

uv run python -m autoresearch.trial --name class-weight --stage screen --note "class-weighted ridge"

set -a; source .env; set +a
uv run python -m autoresearch.loop.check_loop
uv run python -m autoresearch.loop                # until 80% accuracy or Ctrl-C
uv run python -m autoresearch.loop --fresh        # wipe workspace and restart
```

The old ABIDE I GCN session is in `outputs/autoresearch/archive/`. Do not
point the new loop at that workspace.

---

## Branch policy

```text
main                       stable code and published results
experiment/trial-<slug>    one trial, opened for review after a win
```

Neither `run_trial.sh` nor the agent loop merges to `main` on its own.
Old review PRs #2–#13 stay open; they are not the starting point.

---

## Adjusting the gates

Edit `autoresearch/gates.json`. The reference is `linear_baseline_dcan_v1`.
Raise a threshold only after a better configuration has been merged to main.
