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
  "configs/runs/density_solved_r8_40_tol1e7_benchmark.toml"
  "configs/runs/density_solved_r8_40_tol1e8_benchmark.toml"
  "configs/runs/density_solved_r8_40_reg1e5_benchmark.toml"
  "configs/runs/density_solved_r8_40_reg1e4_benchmark.toml"
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
  echo "all four remaining 8--40 kpc case preflights passed"
else
  echo "all four remaining 8--40 kpc cases completed"
fi
