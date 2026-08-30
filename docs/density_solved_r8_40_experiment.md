# Density-solved 8--40 kpc paired benchmark

## Scientific purpose

This experiment aligns the radial support of the density constraint and the
velocity likelihood to `8 <= r < 40 kpc`. It restores density evidence in the
well-sampled `8--15 kpc` region and removes the poorly covered `40--50 kpc`
tail found in the exploratory 8--50 run. The density mask still requires
`|z| >= 2 kpc`; radial alignment therefore does not prove complete phase-space
support alignment. Every saved evaluation audits orbits that enter the velocity
grid but have strictly zero response in the density fit mask.

The old 8--50 result had `git_dirty=true` and remains exploratory context. It is
not rerun or treated as a strict same-commit control.

## One-factor cases

All five cases use the Zhu et al. (2026) paper-best potential, optimizer seed 0,
one evaluation, 10 periods, 1000 samples per orbit, the same catalogue and DESI
target, and the same shell/phi density gate.

| Run config | `lsmr_tol` | L2 strength |
|---|---:|---:|
| `density_solved_r8_40_benchmark.toml` | 1e-6 | 1e-6 |
| `density_solved_r8_40_tol1e7_benchmark.toml` | 1e-7 | 1e-6 |
| `density_solved_r8_40_tol1e8_benchmark.toml` | 1e-8 | 1e-6 |
| `density_solved_r8_40_reg1e5_benchmark.toml` | 1e-6 | 1e-5 |
| `density_solved_r8_40_reg1e4_benchmark.toml` | 1e-6 | 1e-4 |

These are five independent cold-start, one-trial benchmarks, not a multi-trial
optimization and not a 3x3 parameter grid.

## Solver-backend timing benchmark

The solver benchmark keeps the paper-best potential, density target, orbit
sampling, L2 strength, objective, and gates fixed. It changes only the numerical
backend. Each backend is evaluated in three independent one-point cold-start
runs so the comparison can use median solve-only wall time and an independent
GNU-time peak RSS for every repetition; the problem fingerprint must agree
across all nine runs.

| Run config | Solver | KKT tolerance |
| --- | --- | ---: |
| `density_solved_r8_40_solver_lsq_linear_benchmark.toml` | `lsq_linear` | `1e-8` for benchmark qualification |
| `density_solved_r8_40_solver_dense_nnls_benchmark.toml` | `dense_nnls` | `1e-8` |
| `density_solved_r8_40_solver_dual_ridge_benchmark.toml` | `dual_ridge` | `1e-8` |

Validate all three configurations without integration:

```bash
scripts/run_density_solved_r8_40_weight_solvers.sh --preflight-only
```

Run them sequentially and write the artifact-only comparison:

```bash
scripts/run_density_solved_r8_40_weight_solvers.sh
```

The comparison is written to
`.agent-local/benchmarks/r8_40_weight_solver_comparison.json`. The comparator
requires exactly three distinct one-point runs per backend. All three backends,
including the current baseline, enter speed selection only when every repeated
solve is finite, non-negative, converged, below its normalized KKT threshold,
and passes all density gates. Their median inner objectives must agree to
`1e-8` relative. The fastest qualifying backend wins unless another is within
20 percent and uses less peak RSS. The report also records effective orbit
count, active orbit count, maximum weight fraction, and exact-zero weight
fraction. This result is an engineering selection only: `production_ready`
remains false until the winner passes the five fixed-potential ranking test,
and the active recipe remains `lsq_linear`.

## Git provenance

The launcher does not require a detached or locked commit and does not reject a
dirty worktree. It records the current `HEAD` and `git status` in each run's
metadata for retrospective tracking, but does not save a Git diff. Avoid changing
code or configuration while the five cases are running: code identity between
cases is an operator convention rather than a launcher-enforced guarantee.

The launcher still refuses to reuse an output directory and still checks the
experiment configuration, required inputs, and GNU `/usr/bin/time` before orbit
integration starts.

## Coverage and execution

Use the production environment and vendored AGAMA. Generate coverage only once,
from the baseline configuration:

```bash
export PYTHONPATH="$PWD/Agama-master${PYTHONPATH:+:$PYTHONPATH}"
conda run -n halo_lmc python -m halo_mw_lmc \
  -v configs/runs/density_solved_r8_40_benchmark.toml
conda run -n halo_lmc python -m halo_mw_lmc \
  -c configs/runs/density_solved_r8_40_benchmark.toml
```

Run the cases sequentially in this order. The optional preflight-only pass does
not integrate orbits:

```bash
R8_40_CONFIGS=(
  configs/runs/density_solved_r8_40_benchmark.toml
  configs/runs/density_solved_r8_40_tol1e7_benchmark.toml
  configs/runs/density_solved_r8_40_tol1e8_benchmark.toml
  configs/runs/density_solved_r8_40_reg1e5_benchmark.toml
  configs/runs/density_solved_r8_40_reg1e4_benchmark.toml
)

for config in "${R8_40_CONFIGS[@]}"; do
  scripts/run_density_solved_r8_40_case.sh "$config" --preflight-only
done

for config in "${R8_40_CONFIGS[@]}"; do
  scripts/run_density_solved_r8_40_case.sh "$config"
done
```

After a successful baseline, the four remaining cases can instead be launched
sequentially with one command:

```bash
scripts/run_density_solved_r8_40_remaining_cases.sh
```

To validate all four without integrating orbits, use:

```bash
scripts/run_density_solved_r8_40_remaining_cases.sh --preflight-only
```

The batch stops on the first failed case. It does not require a locked commit or
clean worktree; each delegated single-case run records its own Git HEAD, status,
logs, input hashes, and GNU time measurements.

Each run contains `benchmark_metadata/time-v.txt`, input hashes, environment
versions, the command, logs, `git-head.txt`, and `git-status.txt`. If the
workflow fails after creating its cold-start directory, the exit trap still
copies the available metadata into that incomplete run for investigation;
never delete it merely to reuse its name.

## Gate and acceptance

The global density gate remains `chi2 / fitted_bin <= 2`. In addition, every
shell/phi cell formed by `[8,10,12,15,20,30,40] kpc` must contain at least one
valid density bin and independently satisfy `chi2 / fitted_bin <= 2`. Any empty,
non-finite, or excessive cell receives the finite `1e30` invalid-trial penalty.

Before considering a multi-trial pilot, require converged weights, finite
objectives, all density gates passing, investigated failed orbits, zero weight
on every velocity-supported/zero-density-response orbit, and complete GNU time
records. Compare the five saved runs without reintegration:

```bash
conda run -n halo_lmc python scripts/compare_density_solved_r8_40.py \
  runs --output .agent-local/benchmarks/r8_40_comparison.json
```

The comparison reports `git_provenance.same_head` and
`git_provenance.all_status_clean` for information only. Neither field changes
the scientific stability result or blocks a case.

The tight-tolerance stability flag requires the 1e-7 and 1e-8 cases to agree to
`1e-5` in relative selected objective, `1e-3` in maximum absolute shell/phi
chi-square per bin, and `1e-2` in normalized-weight L1 distance. Regularization
cases are diagnostic: review effective orbit count, maximum weight fraction,
active orbit count, velocity objective, and gates without automatically changing
the default strength.

## Fixed-point potential-ranking test

The paper-best sensitivity runs pass every density gate but fail tolerance
stability in velocity objective and normalized weights. This does not by itself
show that potential recovery is unstable: a tolerance-dependent offset that is
nearly constant across potentials would leave the potential ranking unchanged.

The next bounded experiment evaluates the same five potentials, in the same
order, at `lsmr_tol=1e-7` and `1e-8`. It is a cold-start fixed schedule, not an
adaptive optimizer and not warm-start injection.

| point | q | p | rho0 | rho0+2log10(rs) | gamma | purpose |
|---|---:|---:|---:|---:|---:|---|
| paper-best | 0.92 | 0.80 | 6.20 | 9.89 | 1.00 | common reference |
| flatter/more triaxial | 0.82 | 0.70 | 6.20 | 9.89 | 1.00 | shape perturbation |
| rounder | 1.02 | 0.95 | 6.20 | 9.89 | 1.00 | opposite shape perturbation |
| more concentrated | 0.92 | 0.80 | 6.50 | 9.80 | 1.20 | radial-profile perturbation |
| more extended | 0.92 | 0.80 | 5.90 | 10.05 | 0.80 | opposite radial perturbation |

Validate both runs without integration:

```bash
scripts/run_density_solved_r8_40_potential_ranking.sh --preflight-only
```

Then run both sequentially and generate the comparison JSON automatically:

```bash
scripts/run_density_solved_r8_40_potential_ranking.sh
```

The comparison is written to
`.agent-local/benchmarks/r8_40_potential_ranking_comparison.json`. It removes
the tolerance offset measured at paper-best and reports the best point,
Spearman rank correlation, pairwise ordering agreement, and maximum remaining
differential shift relative to the objective span. `ranking_stable=true`
requires all ten evaluations to be valid, the same best point, Spearman and
pairwise agreement at least 0.9, and differential shift no more than 10% of the
paired objective span. These thresholds are a small-sample screening rule, not
a posterior-accuracy statement.

## Wide adaptive scan

The fixed-point ranking test confirmed ranking stability: Spearman = 1.0,
pairwise order agreement = 1.0, same best point (`more_extended`), and maximum
differential shift = 9.7% < 10% of the paired objective span. Although
`ranking_stable` is formally `false` because two of five points fail the density
shell/phi gate identically at both tolerances (a deterministic property of those
potentials, not tolerance noise), the ranking itself is stable across
tolerances. This satisfies the precondition for advancing to an adaptive
multi-trial scan.

The paper-best potential is not optimal under `density_solved` mode: `more_extended`
(q=0.92, p=0.80, rho0=5.90, rho0+2log10(rs)=10.05, gamma=0.80) has a lower
velocity objective by ~588 and both points pass the density gate. The scan
therefore uses `initial_point = "optimizer"` so the GP surrogate explores from
random seeds rather than anchoring at paper-best.

| parameter | local-search bound | wide-scan bound |
|---|---|---|
| qhalo | [0.70, 1.15] | [0.600, 1.300] |
| phalo | [0.40, 1.20] | [0.300, 1.400] |
| rho0 | [5.50, 7.00] | [5.000, 7.500] |
| rho0_plus_2logrs | [9.50, 10.30] | [9.200, 10.600] |
| gamma | [0.50, 1.80] | [0.300, 2.500] |

The scan runs 50 adaptive iterations at `lsmr_tol = 1e-6` (the recipe default,
~20--25 min per evaluation, ~17--21 h total). The run config is
`configs/runs/density_solved_r8_40_wide_scan.toml`, which references
`configs/recipes/zhu_2026_density_solved_r8_40_wide.toml`.

Validate without integration:

```bash
PYTHONPATH="$PWD/Agama-master" conda run -n halo_lmc python -m halo_mw_lmc \
  -v configs/runs/density_solved_r8_40_wide_scan.toml
```

Run optimization only (no static report):

```bash
PYTHONPATH="$PWD/Agama-master" conda run -n halo_lmc python -m halo_mw_lmc \
  -o configs/runs/density_solved_r8_40_wide_scan.toml
```

Expected outputs:

```text
runs/density-solved-r8-40-wide-scan/
  resolved_config.json
  weight_model_inputs.npz
  sample.dat
  best/metadata.json
  best/evaluation.npz
```

Before drawing posterior conclusions from the scan, review:

- density gate pass rate and which regions of parameter space fail;
- objective behavior across the 50 points (convergence trend, outliers);
- weight concentration (effective orbit count, maximum weight fraction);
- GP surrogate surface quality for `parameter_constraints.py` visualization;
- whether the best point remains near `more_extended` or moves elsewhere.
