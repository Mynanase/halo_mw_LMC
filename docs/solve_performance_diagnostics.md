# Weight-solve performance diagnostics

Recorded 2026-08-31 after a measurement-only diagnostic campaign. No
pipeline code was changed; this file preserves the findings and the
recommended fixes so they survive outside `.agent-local/`. The measurement
scripts and raw logs live in `.agent-local/benchmarks/` (untracked by
design): `step_timing_profile.py`, `solve_timing.py`, `solve_breakdown.py`,
`solve_scale_test.py`.

## Measurement setup

- Machine: 112 logical CPUs (2x Xeon Platinum 8280L, 2 NUMA nodes), idle
  during measurement.
- Environment: `halo_lmc` conda env (Python 3.12, NumPy/SciPy 1.18),
  vendored AGAMA on `PYTHONPATH`.
- Case: one paper-best trial from `configs/runs/density_solved_benchmark.toml`
  (11,250 seeded orbits x 1,000 samples = 11.25 M samples, `density_solved`
  weights, `velocity_only` objective, `lsmr_tol = 1e-6`).

## Finding 1 - parallelism status

Only the AGAMA orbit integration is multicore (libgomp OpenMP, cpu/wall
about 110, 176 threads). Every other pipeline step is single-threaded
Python/NumPy/SciPy; the pipeline code itself contains no multiprocessing.
The weight solve *burns* about 64 cores of CPU but performs
single-threaded work (Finding 3).

## Finding 2 - per-step wall time (245.7 s total, peak RSS ~3.4 GiB)

| step | wall (s) | share | cpu/wall |
|---|---:|---:|---:|
| prepare_model_data | 0.46 | 0.2% | 1.0 |
| build_potential | 0.05 | 0.0% | 1.0 |
| potential.Tcirc | 0.07 | 0.0% | 106 (OpenMP) |
| agama.orbit integration | 9.65 | 3.9% | 110 (OpenMP) |
| orbit postprocess | 1.30 | 0.5% | 1.0 |
| build_orbit_density_response | 1.85 | 0.8% | 1.0 |
| **solve_density_weights** | **222.6** | **90.6%** | 63.7 (spin, Finding 3) |
| cartesian -> spherical | 2.40 | 1.0% | 1.0 |
| orbit support audit | 0.42 | 0.2% | 1.0 |
| velocity histograms (x3) | ~2.2 each | 2.7% | 1.0 |
| velocity likelihood (x3) | ~0.10 each | 0.1% | 1.0 |

The solve dominates the wall clock; speeding up integration cannot help a
single trial.

## Finding 3 - OpenBLAS spin-wait waste (three-condition experiment)

`solve_density_weights` ran in isolation under three thread-pool
conditions on the identical design matrix:

| condition | wall (s) | CPU (s) | cpu/wall | peak threads | outer iters |
|---|---:|---:|---:|---:|---:|
| default | 224.18 | 14,278 | 63.69 | 239 | 2309 |
| `OPENBLAS_NUM_THREADS=1` | 220.11 | 219 | 1.00 | 113 | 3939 |
| `agama.setNumThreads(1)` | 244.35 | 15,570 | 63.72 | 239 | 2309 |

- 100% of the ~64-core CPU burn is the numpy/OpenBLAS pool spinning
  between the small BLAS calls inside the solve; the AGAMA libgomp pool is
  innocent (parking it changes nothing, and the solve result is
  bit-identical).
- Wall time is essentially unchanged (224 -> 220 s): the spin wastes CPU
  quota, not wall time. About 14,200 core-seconds (98% of the process CPU)
  are wasted per trial.
- Multithreaded BLAS changes the reduction order and therefore the solver
  trajectory (2309 vs 3939 outer iterations; converged objective equal to
  ~1e-3 relative). Runs are only comparable at a fixed thread count.

## Finding 4 - AGAMA OpenMP efficiency (282-orbit subset)

| threads | wall (s) | speedup | efficiency |
|---:|---:|---:|---:|
| 1 | 4.76 | 1.00x | 100% |
| 14 | 0.47 | 10.1x | 72% |
| 28 | 0.34 | 13.8x | 49% |
| 56 | 0.32 | 14.9x | 27% |
| 112 | 0.26 | 18.0x | 16% |

28 threads captures 13.8x; the remaining 4x speedup costs 4x the CPU.

## Finding 5 - inside the solve (instrumented `lsq_linear`)

The design matrix handed to `scipy.optimize.lsq_linear` is
**7007 x 5967** = (1040 fitted density cells + 5967 regularization rows) x
5967 active orbits (the active-column filter drops 11,250 successful
orbits to 5,967), nnz = 389,250, 4.5 MiB, 65 nnz per column.

| component | time | share | detail |
|---|---:|---:|---|
| matvec | 107.8 s | 48.6% | 207,811 calls x 519 us (9.7 GiB/s, single-core bound) |
| rmatvec | 81.0 s | 36.5% | 198,446 calls x 408 us |
| other | 32.9 s | 14.8% | TRF bookkeeping + LSMR Python loop vector ops |

Inner LSMR totals ~203k iterations (51.6 per outer TRF iteration; 3,939
outer iterations). 85% of the wall is 406k *serial* sparse matvecs
streaming the same 4.5 MiB about 1.9 TB through one core (~1.4 GFLOP/s
effective). SciPy's TRF restarts LSMR from scratch every outer iteration
(`scipy/optimize/_lsq/trf.py`); `lsq_linear` (SciPy 1.18) exposes no
warm-start, preconditioner, or `x_scale` hook.

Consequence: the sparse Krylov loop cannot be threaded - iterations are
strictly sequential, sparse matvec is bandwidth-bound (2-4x at best on two
sockets), and the vector lengths (~6000-7000) are below the threshold
where BLAS threading pays for itself.

## Finding 6 - the solve is underdetermined (root cause of slow AND unstable)

1040 equations constrain 5967 unknowns; the `regularization_strength =
1e-6` L2 penalty contributes ~1e-5 against a cost of ~236, i.e. it is
numerically irrelevant. The solution wanders in a ~4900-dimensional flat
valley:

- the default solve exits on *cost-stall* (status 2) with optimality
  1.6e-2 - not a true optimum;
- a column-rescaled re-solve of the *identical* problem reaches a lower
  cost (236.683 vs 236.746) with materially different weights
  (sum(w) = 8.16 vs 6.33; max |dw| = 1.36 while the largest weight is
  0.27).

This single fact explains both the ~4000 outer iterations (the trust-region
path crawls through the valley) and the documented instability of solved
weights under 1e-7/1e-8 tolerance changes
(`docs/density_solved_r8_40_experiment.md`): the variation is trajectory
noise on a flat manifold, not physical signal.

## Finding 7 - solver knobs (measured, both rejected)

| variant | wall (s) | outer iters | matvecs | cost | outcome |
|---|---:|---:|---:|---:|---|
| default | 222.8 | 3939 | 406,257 | 236.746 | cost-stall exit, optimality 1.6e-2 |
| column scaling (x_scale analogue) | 343.7 | 4221 | 127,997 | 236.683 | 68% fewer matvecs and better cost, but slower wall; unexplained Python-side cost, re-verify in isolation before any use |
| `lsmr_maxiter = 15` | 496.7 | 20,000 (cap) | 720,361 | 236.879 | `success=False`; the pipeline would record `INVALID_TRIAL_PENALTY` |

Inner-iteration caps are catastrophic here, consistent with the earlier
project experience that loose inner tolerances stall the outer loop.

## Related known issue - objective bimodality (wide-scan review)

From the `density-solved-r8-40-wide-scan` sample review: objectives are
either normal velocity log-likelihoods (~1.33-1.36e5) or
`INVALID_TRIAL_PENALTY = 1e30`. The penalty fires when (a) the solver
returns `result.success = False` at `max_iter = 20000` - 11 of 50
wide-scan trials, several with excellent density fits and passing gates -
or (b) the `velocity_only` density gate fails. The 25-order-of-magnitude
separation starves the GP surrogate of gradient information, and the
non-converged trials are also the slowest ones (~20,000 outer iterations,
roughly 5x the paper-best trial).

## Recommendations (ranked; none implemented yet)

1. **Export `OPENBLAS_NUM_THREADS=1` for production runs.** Removes 98% of
   the process CPU at unchanged wall time. Fix one thread setting and keep
   it for cross-run comparability (Finding 3 trajectory dependence).
2. **On shared machines, call `agama.setNumThreads(28)` before
   integration** (optional): +4 s wall per trial, -65% integration CPU
   (Finding 4).
3. **Parallelize across trials, not within a solve.** After (1) each
   solve is single-core, so fixed-point batches (the five-point ranking
   test, benchmark cases) can run as parallel processes. The sequential GP
   optimizer limits adaptive scans; batch/async acquisition would be an
   optimizer design change and stays deferred alongside warm-start per
   `AGENTS.md`.
4. **Dense rewrite of the solve core (prototype first).** The problem is
   only 7007 x 5967 dense (335 MiB): a dense QR or normal-equation
   Cholesky + NNLS does real flops in BLAS (estimated seconds vs 222 s,
   i.e. 15-45x) and removes the trajectory noise of Finding 6. Risks:
   normal equations square the condition number under `lambda = 1e-6`;
   every trial objective shifts slightly, so all comparisons need
   re-baselining. Prototype under `.agent-local/benchmarks/` and validate
   against the current solver on several potentials before any production
   switch (benchmark changes stay separate from production changes).
5. **Science decision required: address the underdetermination.** Raising
   `regularization_strength` by orders of magnitude, or restricting the
   active-orbit set, fixes both the speed and the weight instability -
   but changes the weight semantics and requires an explicit decision and
   re-baselining.
6. **Reconsider the `converged` criterion / penalty coupling.** Trials
   with excellent density fits but `result.success = False` are mis-killed
   as 1e30 and are also the slowest trials; a sanity criterion based on
   the density fit and optimality, rather than `result.success` alone,
   would fix both the statistics and the worst-case wall time.
7. **Do not pursue:** threading the sparse Krylov loop (structurally
   serial, Finding 5) or `lsmr_maxiter` caps (Finding 7).

## Reproduction

```bash
cd /home/tqiu/halo_mw_LMC
# per-step audit + AGAMA OpenMP scaling
conda run -n halo_lmc python -u .agent-local/benchmarks/step_timing_profile.py
# thread-pool attribution (run each line separately)
conda run -n halo_lmc python -u .agent-local/benchmarks/solve_timing.py default
OPENBLAS_NUM_THREADS=1 conda run -n halo_lmc python -u .agent-local/benchmarks/solve_timing.py default
conda run -n halo_lmc python -u .agent-local/benchmarks/solve_timing.py agama1
# in-solve breakdown
OPENBLAS_NUM_THREADS=1 conda run -n halo_lmc python -u .agent-local/benchmarks/solve_breakdown.py
# solver-knob variants
OPENBLAS_NUM_THREADS=1 conda run -n halo_lmc python -u .agent-local/benchmarks/solve_scale_test.py
```

Each script writes a matching `.log` file beside it. `conda run` buffers
the child output until exit, so monitor progress with
`ps -eo pid,pcpu,nlwp,etime,args`.
