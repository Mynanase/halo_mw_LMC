#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi
MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--preflight-only" ]]; then
  echo "unknown option: $MODE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPOSITORY"

CONFIGS=(
  configs/runs/density_solved_r8_40_solver_lsq_linear_benchmark.toml
  configs/runs/density_solved_r8_40_solver_lsq_linear_repeat2.toml
  configs/runs/density_solved_r8_40_solver_lsq_linear_repeat3.toml
  configs/runs/density_solved_r8_40_solver_dense_nnls_benchmark.toml
  configs/runs/density_solved_r8_40_solver_dense_nnls_repeat2.toml
  configs/runs/density_solved_r8_40_solver_dense_nnls_repeat3.toml
  configs/runs/density_solved_r8_40_solver_dual_ridge_benchmark.toml
  configs/runs/density_solved_r8_40_solver_dual_ridge_repeat2.toml
  configs/runs/density_solved_r8_40_solver_dual_ridge_repeat3.toml
)
RUNS=(
  runs/density-solved-r8-40-solver-lsq-linear-benchmark
  runs/density-solved-r8-40-solver-lsq-linear-repeat2
  runs/density-solved-r8-40-solver-lsq-linear-repeat3
  runs/density-solved-r8-40-solver-dense-nnls-benchmark
  runs/density-solved-r8-40-solver-dense-nnls-repeat2
  runs/density-solved-r8-40-solver-dense-nnls-repeat3
  runs/density-solved-r8-40-solver-dual-ridge-benchmark
  runs/density-solved-r8-40-solver-dual-ridge-repeat2
  runs/density-solved-r8-40-solver-dual-ridge-repeat3
)

for config in "${CONFIGS[@]}"; do
  if [[ "$MODE" == "--preflight-only" ]]; then
    scripts/run_density_solved_r8_40_case.sh "$config" --preflight-only
  else
    scripts/run_density_solved_r8_40_case.sh "$config"
  fi
done

if [[ "$MODE" == "--preflight-only" ]]; then
  exit 0
fi

conda run -n halo_lmc python \
  scripts/compare_density_solved_r8_40_weight_solvers.py \
  "${RUNS[@]}" \
  --output .agent-local/benchmarks/r8_40_weight_solver_comparison.json
