#!/usr/bin/env bash
# Stage-1 fixed-point screening driver: runs the 12 shard configs in parallel
# with a hard resource cap so the shared server keeps >50% of its cores.
#
#   - 12 shards, each 4 fixed points, sequential inside a shard
#   - per shard: OPENBLAS_NUM_THREADS=1 (dense weight solve, single core)
#                OMP_NUM_THREADS=4 (AGAMA batched orbit integration)
#   - worst-case simultaneous load: 12 x 4 = 48 of 112 cores (~43%)
#   - all shard processes run under `nice -n 10`
#
# usage: scripts/run_density_solved_r8_40_stage1_screen.sh [--preflight-only]
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--preflight-only" ]]; then
  echo "unknown option: $MODE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPOSITORY"

SHARDS=12
THREADS_PER_SHARD=4
MAX_LOAD=$((SHARDS * THREADS_PER_SHARD))
CORES="$(nproc)"
if (( MAX_LOAD * 2 > CORES )); then
  echo "resource cap violated: $MAX_LOAD threads on $CORES cores exceeds 50%" >&2
  exit 1
fi

CONFIGS=()
for number in 01 02 03 04 05 06 07 08 09 10 11 12; do
  CONFIGS+=("configs/runs/density_solved_r8_40_stage1_screen_shard${number}.toml")
done

STAGING="$REPOSITORY/.agent-local/benchmarks/stage1-screen"
mkdir -p "$STAGING"

export PYTHONPATH="$REPOSITORY/Agama-master${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1

PIDS=()
for config in "${CONFIGS[@]}"; do
  shard_id="$(basename "$config" .toml | sed 's/.*shard//')"
  shard_staging="$STAGING/shard${shard_id}"
  rm -rf "$shard_staging"
  mkdir -p "$shard_staging"

  if [[ -n "$MODE" ]]; then
    conda run -n halo_lmc python -m halo_mw_lmc preflight "$config" --stage run \
      > "$shard_staging/preflight.log" 2>&1
    echo "preflight passed: $config"
    continue
  fi

  {
    echo "preflight $config"
    conda run -n halo_lmc python -m halo_mw_lmc preflight "$config" --stage run \
      > "$shard_staging/preflight.log" 2>&1

    git rev-parse HEAD > "$shard_staging/git-head.txt"
    sha256sum \
      data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt \
      data_for_model/synthetic/desi_year1_kgiants_25x25x4.npz \
      > "$shard_staging/input-sha256.txt" 2>/dev/null || true
    {
      conda run -n halo_lmc python -c \
        "import sys; print(sys.version); import numpy, scipy; print('numpy', numpy.__version__); print('scipy', scipy.__version__)"
      # printenv exits 1 for unset names, which would trip `set -e`.
      printenv OPENBLAS_NUM_THREADS OMP_NUM_THREADS || true
      echo "OMP_NUM_THREADS(effective)=${THREADS_PER_SHARD}"
    } > "$shard_staging/environment.txt"

    echo "run $config"
    nice -n 10 env OMP_NUM_THREADS="$THREADS_PER_SHARD" \
      /usr/bin/time -v -o "$shard_staging/time-v.txt" \
      conda run -n halo_lmc python -m halo_mw_lmc run "$config" \
      > "$shard_staging/stdout.log" 2> "$shard_staging/stderr.log"
    echo "done $config"
  } > "$shard_staging/progress.log" 2>&1 &
  PIDS+=($!)
done

if [[ -n "$MODE" ]]; then
  exit 0
fi

FAILED=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    shard_id="$(basename "${CONFIGS[$index]}" .toml | sed 's/.*shard//')"
    echo "shard $shard_id FAILED; see $STAGING/shard${shard_id}/" >&2
    FAILED=1
  fi
done
exit "$FAILED"
