#!/usr/bin/env bash
set -euo pipefail

# Use the locked environment when uv is available; Tycho exposes the same
# dependencies through the persistent halo_lmc environment.
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
if [[ "${HALO_MW_LOCAL_SMOKE:-0}" == "1" && -L .agent-local ]]; then
    LOCAL_RESULTS="${TMPDIR:-/tmp}/halo-mw-local-smoke-results"
    SOURCE_ROOT="$(readlink .agent-local)"
    LOCAL_AGENT_LOCAL="${SOURCE_ROOT}/benchmarks/full_weight_regularization/local_smoke"
    mkdir -p "$LOCAL_RESULTS"
    if [[ ! -e "$LOCAL_AGENT_LOCAL" && ! -L "$LOCAL_AGENT_LOCAL" ]]; then
        ln -s "$LOCAL_RESULTS" "$LOCAL_AGENT_LOCAL"
    fi
fi

PYTHON="${HALO_MW_PYTHON:-/home/tqiu/.local/bin/python3}"
UV="${HALO_MW_UV:-/home/tqiu/.local/bin/uv}"
if [[ -z "${UV}" || ! -x "${UV}" ]]; then
    if [[ -x /Users/qttao/Documents/Research/halo_mw_LMC/.venv/bin/python ]]; then
        PYTHON=/Users/qttao/Documents/Research/halo_mw_LMC/.venv/bin/python
    fi
fi
if [[ "${HALO_MW_LOCAL_SMOKE:-0}" == "1" ]]; then
    SCAN_MODE=(--local-smoke)
else
    SCAN_MODE=()
fi
if [[ -x "$UV" ]]; then
    TEST=("$UV" run --locked --extra inference --extra astronomy python -m unittest tests.test_full_weight_regularization)
    SCAN=("$UV" run --locked --extra inference --extra astronomy python scripts/full_weight_regularization_scan.py)
    SCAN+=("${SCAN_MODE[@]}")
else
    TEST=("$PYTHON" -m unittest tests.test_full_weight_regularization)
    SCAN=("$PYTHON" scripts/full_weight_regularization_scan.py)
    SCAN+=("${SCAN_MODE[@]}")
fi

"${TEST[@]}"
"${SCAN[@]}"
