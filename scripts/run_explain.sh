#!/usr/bin/env bash
# Attribute the current model and write chord / ROI figures.
#
# Auto-detects GNN vs linear:
#   a .pt checkpoint or --loso-folds-dir → GNN integrated gradients
#   otherwise → linear coefficients from neuroasd.model
#
# Usage:
#   ./scripts/run_explain.sh
#   ./scripts/run_explain.sh --kind linear --output-dir outputs/explain_linear
#   ./scripts/run_explain.sh --checkpoint outputs/gcn_baseline/final_model.pt \
#       --output-dir experiments/baseline_gcn_v1/figures
#   ./scripts/run_explain.sh --loso-folds-dir outputs/loso_cv_gcn_v1/folds \
#       --output-dir experiments/loso_cv_gcn_v1/figures
#
# Any extra flags are forwarded to `python -m neuroasd.explain`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  PASSTHROUGH+=("$1")
  shift
done

echo "==> Explaining model (auto-detect GNN vs linear)"
if ((${#PASSTHROUGH[@]} > 0)); then
  uv run python -m neuroasd.explain "${PASSTHROUGH[@]}"
elif [[ -f outputs/gcn_baseline/final_model.pt ]]; then
  uv run python -m neuroasd.explain \
    --checkpoint outputs/gcn_baseline/final_model.pt \
    --output-dir experiments/baseline_gcn_v1/figures
else
  uv run python -m neuroasd.explain --kind linear
fi

echo "==> Done"
