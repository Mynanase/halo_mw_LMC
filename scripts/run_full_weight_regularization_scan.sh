#!/usr/bin/env bash
set -euo pipefail

# Use the locked environment when uv is available (absolute path: the remote
# host's non-login shell does not put ~/.local/bin on PATH); fall back to the
# persistent halo_lmc interpreter otherwise.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED=0
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/halo-mw-matplotlib}"

# The immutable source snapshot does not carry ignored, data-only inputs.
DATA_ROOT="${HALO_MW_DATA_ROOT:-/home/tqiu/halo_mw_LMC}"
if [[ -d "$DATA_ROOT/data_for_model" && ! -d data_for_model ]]; then
    ln -s "$DATA_ROOT/data_for_model" data_for_model
fi
if [[ -d "$DATA_ROOT/.agent-local" && ! -d .agent-local ]]; then
    ln -s "$DATA_ROOT/.agent-local" .agent-local
fi

PYTHON="${HALO_MW_PYTHON:-/home/tqiu/.local/bin/python3}"
UV="${HALO_MW_UV:-/home/tqiu/.local/bin/uv}"
if [[ -x "$UV" ]]; then
    TEST=("$UV" run --locked --extra inference --extra astronomy python -m unittest tests.test_full_weight_regularization tests.test_entropy_newton)
    SCAN=("$UV" run --locked --extra inference --extra astronomy python scripts/full_weight_regularization_scan.py)
else
    TEST=("$PYTHON" -m unittest tests.test_full_weight_regularization tests.test_entropy_newton)
    SCAN=("$PYTHON" scripts/full_weight_regularization_scan.py)
fi

"${TEST[@]}"
"${SCAN[@]}"
