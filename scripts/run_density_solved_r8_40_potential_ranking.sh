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
CASE_LAUNCHER="$SCRIPT_DIR/run_density_solved_r8_40_case.sh"
CONFIGS=(
  "configs/runs/density_solved_r8_40_potential_ranking_tol1e7.toml"
  "configs/runs/density_solved_r8_40_potential_ranking_tol1e8.toml"
)
LAUNCHER_OPTIONS=()
if [[ -n "$MODE" ]]; then
  LAUNCHER_OPTIONS+=("$MODE")
fi

cd "$REPOSITORY"
for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  echo "[$((index + 1))/${#CONFIGS[@]}] starting $config"
  "$CASE_LAUNCHER" "$config" "${LAUNCHER_OPTIONS[@]}"
  echo "[$((index + 1))/${#CONFIGS[@]}] completed $config"
done

if [[ "$MODE" == "--preflight-only" ]]; then
  echo "both fixed-point potential-ranking preflights passed"
  exit 0
fi

COMPARISON_OUTPUT=".agent-local/benchmarks/r8_40_potential_ranking_comparison.json"
conda run -n halo_lmc python \
  scripts/compare_density_solved_r8_40_potential_ranking.py \
  runs --output "$COMPARISON_OUTPUT"
echo "paired potential-ranking comparison: $COMPARISON_OUTPUT"
