#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_CONFIG [--preflight-only]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_CONFIG="$1"
MODE="${2:-}"
if [[ -n "$MODE" && "$MODE" != "--preflight-only" ]]; then
  echo "unknown option: $MODE" >&2
  exit 2
fi

cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY/Agama-master${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t PREFLIGHT < <(
  conda run -n halo_lmc python -m halo_mw_lmc.benchmark \
    "$RUN_CONFIG"
)
if [[ ${#PREFLIGHT[@]} -ne 4 ]]; then
  echo "benchmark preflight did not return the expected run information" >&2
  exit 1
fi
RUN_ID="${PREFLIGHT[0]}"
RUN_DIRECTORY="${PREFLIGHT[1]}"
CATALOGUE="${PREFLIGHT[2]}"
TARGET_DENSITY="${PREFLIGHT[3]}"

conda run -n halo_lmc python -c \
  "import agama, astropy, matplotlib, numpy, scipy, skopt"
conda run -n halo_lmc python -m halo_mw_lmc -v "$RUN_CONFIG"

if [[ "$MODE" == "--preflight-only" ]]; then
  echo "preflight passed: $RUN_ID"
  exit 0
fi

STAGING="$REPOSITORY/.agent-local/benchmarks/r8_40/$RUN_ID"
if [[ -e "$STAGING" ]]; then
  echo "benchmark metadata staging directory already exists: $STAGING" >&2
  exit 1
fi
mkdir -p "$STAGING"
METADATA="$RUN_DIRECTORY/benchmark_metadata"
copy_metadata() {
  if [[ -d "$STAGING" && -d "$RUN_DIRECTORY" ]]; then
    mkdir -p "$METADATA"
    cp "$STAGING"/* "$METADATA/"
  fi
}
trap copy_metadata EXIT

git rev-parse HEAD > "$STAGING/git-head.txt"
git status --porcelain --untracked-files=all > "$STAGING/git-status.txt"
sha256sum "$CATALOGUE" "$TARGET_DENSITY" > "$STAGING/input-sha256.txt"
conda run -n halo_lmc python -c \
  "import agama, astropy, matplotlib, numpy, scipy, skopt, sys; print(sys.version); print('agama', getattr(agama, '__version__', 'unknown')); print('astropy', astropy.__version__); print('matplotlib', matplotlib.__version__); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('skopt', skopt.__version__)" \
  > "$STAGING/environment.txt"
printf '%q ' /usr/bin/time -v conda run -n halo_lmc python -m halo_mw_lmc "$RUN_CONFIG" \
  > "$STAGING/command.txt"
printf '\n' >> "$STAGING/command.txt"

/usr/bin/time -v -o "$STAGING/time-v.txt" \
  conda run -n halo_lmc python -m halo_mw_lmc "$RUN_CONFIG" \
  > "$STAGING/stdout.log" 2> "$STAGING/stderr.log"

copy_metadata
trap - EXIT
echo "benchmark complete: $RUN_DIRECTORY"
