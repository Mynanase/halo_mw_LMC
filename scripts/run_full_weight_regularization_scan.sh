#!/usr/bin/env bash
set -euo pipefail

# Use the locked environment on hosts where uv is available.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED=0

uv run --locked python -m unittest tests.test_full_weight_regularization
uv run --locked python scripts/full_weight_regularization_scan.py
