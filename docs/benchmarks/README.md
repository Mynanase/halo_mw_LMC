# Committed benchmark evidence

Small, curated evidence files backing `docs/solve_performance_diagnostics.md`
and the r8-40 experiment documents. Raw run directories, coverage outputs, and
reproduction scripts stay in local `.agent-local/` (machine-specific, not
portable); only compact machine-generated summaries and verbatim log excerpts
are committed here.

Everything below was produced on the production server (112-core, conda env
`halo_lmc`, Agama-master vendored) between 2026-08-31 and 2026-09-05. Each file
records its own provenance header; the full reproduction commands are in
`docs/solve_performance_diagnostics.md` (Reproduction section).

## Files

| file | produced by | backs |
|---|---|---|
| `r8_40_weight_solver_comparison.json` | `scripts/compare_density_solved_r8_40_weight_solvers.py` over the 3x3 solver-backend benchmark | three-backend comparison conclusion (dense_nnls is the only KKT-qualified backend but ~2x slower than `lsq_linear`; `dual_ridge` disqualified; active recipe stays `lsq_linear`) |
| `agama_omp_scan_results.json` | `.agent-local/benchmarks/agama_omp_scan.py` (round-robin, 2 repeats, full 11250-orbit problem, 10% subsample for the 1-thread baseline) | Finding 4b: AGAMA integration is 11-15 s of a ~500 s trial at any reasonable thread count |
| `solve_timing_logs.txt` | verbatim `step_timing_profile` / `solve_breakdown` / `solve_scale_test` / `solve_timing_*` logs | Findings 2, 3, 4, 5, 7 (per-step budget, BLAS spin-wait, solve internals, knob variants) |
| `full_run_wall_budget.txt` | `/usr/bin/time -v` excerpts + metadata + a standalone report re-render | Finding 8: pinned full run 209 s vs 541 s unpinned; solve wall is a >3x trajectory lottery at one problem fingerprint |

## Interpretation caveats

- `r8_40_weight_solver_comparison.json` `wall_seconds` is the **solve-only**
  wall copied from run metadata, not the whole-case wall (Finding 8).
- The committed comparison JSON reflects the run before the OpenBLAS pin
  entered the case launcher; the pinned solve trajectory differs (Finding 8),
  so objectives across thread settings must not be compared directly.
- `production_ready` remains false: the comparison is an engineering selection,
  and the active recipe stays `lsq_linear`.
