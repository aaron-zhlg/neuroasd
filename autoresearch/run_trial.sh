#!/usr/bin/env bash
# Run one autoresearch trial on a throwaway branch and keep it only if it wins.
#
# Usage:
#   ./autoresearch/run_trial.sh --name <slug> [--stage <stage>] [--note "..."] -- [trial args]
#
# Examples:
#   ./autoresearch/run_trial.sh --name dropout03 -- --dropout 0.3
#   ./autoresearch/run_trial.sh --name class-weight --stage loso-site -- --note "class-weighted ridge"
#   ./autoresearch/run_trial.sh --name baseline --note "calibrate gates" --
#
# Trials branch from the current branch unless --base says otherwise.
#
# Policy (see autoresearch/program.md):
#   PASS -> commit the change plus result.json, push experiment/trial-<slug>
#   FAIL -> return to the base branch and delete the trial branch; the ledger keeps
#           the record at outputs/autoresearch/ledger.jsonl

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME=""
STAGE="screen"
NOTE=""
# Branch from wherever we are, so trials work before this framework reaches main.
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
KEEP_ON_FAIL=0
TRIAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --note) NOTE="$2"; shift 2 ;;
    --base) BASE_BRANCH="$2"; shift 2 ;;
    --keep-on-fail) KEEP_ON_FAIL=1; shift ;;
    --) shift; TRIAL_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$NAME" ]]; then
  echo "error: --name is required" >&2
  exit 64
fi

TRIAL_BRANCH="experiment/trial-${NAME}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty; commit or stash before running a trial" >&2
  git status --short >&2
  exit 65
fi

if git show-ref --verify --quiet "refs/heads/${TRIAL_BRANCH}"; then
  echo "error: branch ${TRIAL_BRANCH} already exists; pick another --name" >&2
  exit 65
fi

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
cleanup_to_start() {
  git checkout --quiet "$START_BRANCH"
}

echo "==> Creating ${TRIAL_BRANCH} from ${BASE_BRANCH}"
git checkout --quiet "$BASE_BRANCH"
git checkout --quiet -b "$TRIAL_BRANCH"

echo "==> Running trial (stage=${STAGE})"
set +e
uv run python -m autoresearch.trial \
  --name "$NAME" \
  --stage "$STAGE" \
  --note "$NOTE" \
  ${TRIAL_ARGS[@]+"${TRIAL_ARGS[@]}"}
TRIAL_EXIT=$?
set -e

RESULT_FILE="outputs/autoresearch/trials/${STAGE}__${NAME}/result.json"

if [[ $TRIAL_EXIT -eq 0 ]]; then
  echo "==> PASS: keeping ${TRIAL_BRANCH}"
  mkdir -p "autoresearch/results/${STAGE}__${NAME}"
  cp "$RESULT_FILE" "autoresearch/results/${STAGE}__${NAME}/result.json"
  git add -A
  git commit --quiet -m "Trial ${NAME} (${STAGE}): ${NOTE:-no note}"
  git push --quiet -u origin "$TRIAL_BRANCH"
  echo "==> Pushed ${TRIAL_BRANCH}; opening a review PR (scores first)"
  set +e
  uv run python -m autoresearch.loop.promote \
    --pr-from-result "$RESULT_FILE" \
    --branch "$TRIAL_BRANCH"
  set -e
  echo "==> Review before merging to main"
  exit 0
fi

if [[ $TRIAL_EXIT -ne 3 ]]; then
  echo "==> ERROR: trial crashed (exit ${TRIAL_EXIT}); leaving ${TRIAL_BRANCH} in place" >&2
  exit $TRIAL_EXIT
fi

echo "==> FAIL: gate not cleared"
if [[ $KEEP_ON_FAIL -eq 1 ]]; then
  echo "==> Keeping ${TRIAL_BRANCH} locally (--keep-on-fail)"
  exit 3
fi

# Drop the trial's edits: reset tracked files, remove new ones. `git clean -fd` without
# -x leaves outputs/ alone, so the ledger survives.
git reset --quiet --hard HEAD
git clean --quiet -fd
cleanup_to_start
git branch --quiet -D "$TRIAL_BRANCH"
echo "==> Discarded ${TRIAL_BRANCH}; record kept in outputs/autoresearch/ledger.jsonl"
exit 3
