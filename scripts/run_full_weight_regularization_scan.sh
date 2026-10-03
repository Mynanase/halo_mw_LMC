#!/usr/bin/env bash
set -euo pipefail

# Use the locked environment when uv is available; Tycho exposes the same
# dependencies through the persistent halo_lmc environment.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED=0

# The immutable source snapshot does not carry ignored, data-only inputs.
DATA_ROOT="${HALO_MW_DATA_ROOT:-/home/tqiu/halo_mw_LMC}"
if [[ -d "$DATA_ROOT/data_for_model" && ! -d data_for_model ]]; then
    ln -s "$DATA_ROOT/data_for_model" data_for_model
fi
if [[ -d "$DATA_ROOT/.agent-local" && ! -d .agent-local ]]; then
    ln -s "$DATA_ROOT/.agent-local" .agent-local
fi

PYTHON="${HALO_MW_PYTHON:-/home/tqiu/miniforge3/envs/halo_lmc/bin/python}"
if command -v uv >/dev/null 2>&1; then
    TEST=(uv run --locked --extra inference --extra astronomy python -m unittest tests.test_full_weight_regularization)
    SCAN=(uv run --locked --extra inference --extra astronomy python scripts/full_weight_regularization_scan.py)
else
    TEST=("$PYTHON" -m unittest tests.test_full_weight_regularization)
    SCAN=("$PYTHON" scripts/full_weight_regularization_scan.py)
fi

"${TEST[@]}"
"${SCAN[@]}"
