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
`.agent-local/benchmarks/r8_40_weight_solver_comparison.json` (curated
committed copy: `docs/benchmarks/r8_40_weight_solver_comparison.json`). The comparator
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
conda run -n halo_lmc python -m halo_mw_lmc validate configs/runs/density_solved_r8_40_benchmark.toml
conda run -n halo_lmc python -m halo_mw_lmc coverage configs/runs/density_solved_r8_40_benchmark.toml
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

## Joint density--velocity outer objective (2026-09-07)

The next experiment adds density fit quality directly to potential selection:

```text
w_hat(theta) = density-only non-negative least-squares solution
J(theta) = 0.5 * chi2_density(theta, w_hat) - log L_velocity(theta, w_hat)
```

This is the configured `density_velocity` mode. The question is whether ranking
potentials by both residual density and velocity fit improves the inference when
the inner weights are only approximate. Weights need not be individually
recovered to high precision; their resulting predictions and potential rankings
must still be checked for numerical sensitivity.

The joint mode replaces the earlier global and shell/phi density hard gates with
the continuous density loss. Inner solver failures still receive `1e30`; a
successful cost-stall exit is evaluated with both terms. The solver, L2 strength,
target/error normalization, orbit sampling, radial and vertical masks, velocity
bins, and fixed seed remain as in the corresponding velocity-only baseline.
There is no extra outer regularization term or arbitrary density multiplier.

| Run | Comparison baseline | Purpose |
| --- | --- | --- |
| `configs/runs/density_solved_r8_40_joint_benchmark.toml` | `density_solved_r8_40_benchmark.toml` | One paper-best trial, unchanged local bounds |
| `configs/runs/density_solved_r8_40_joint_wide_scan.toml` | `density_solved_r8_40_wide_scan.toml` | 50 iterations, unchanged wide bounds and optimizer-generated start |

Both runs have separate identities and fresh output directories containing
`joint`. Earlier configurations and their velocity-only results are retained.
Use the ordinary lifecycle CLI; the historical named-benchmark launcher does
not accept these new configurations.

Validate and run the single-potential case first on the production server:

```bash
conda run -n halo_lmc python -m halo_mw_lmc validate configs/runs/density_solved_r8_40_joint_benchmark.toml
OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/Agama-master${PYTHONPATH:+:$PYTHONPATH}" \
  conda run -n halo_lmc python -m halo_mw_lmc run configs/runs/density_solved_r8_40_joint_benchmark.toml
```

Check `objective = objective_density_velocity = 0.5 * chi2 +
objective_velocity` in the saved samples, along with the solver diagnostics and
density/velocity predictions. Before running the joint wide scan, complete the
full-catalogue benchmark and compare identical fixed potentials across solver
accuracy settings using the joint objective. Historical velocity-only ranking
results do not establish joint-objective stability. Production execution,
potential-ranking stability, and scientific recovery require separate evidence
for this new objective selection.

### Rescore the existing paired fixed-point artifacts

The historical five-point runs already persist `objective_density_velocity`,
`objective_velocity`, and `chi2`. Reuse these columns to compare the joint
objective at the same saved weights without reintegration or solver replay:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" conda run -n dp-jax python \
  scripts/compare_density_solved_r8_40_potential_ranking.py \
  PATH_TO_SAVED_RUNS --objective density_velocity --output joint_ranking.json
```

The default `--objective velocity_only` preserves the historical comparison.
Joint rescoring checks the saved loss decomposition, ignores the historical
density gate, and retains the existing finite-score, solver-convergence and
failed-orbit checks. The original `objective` can therefore be `1e30` for a point
whose joint score is usable; the reason for the old rejection must be the density
gate rather than solver failure. Input hashes, model settings and source
provenance must be reviewed before interpreting the comparison.

The screening thresholds remain the same: all ten points valid, same best point,
Spearman and pairwise agreement at least 0.9, and maximum offset-corrected shift
at most 10% of the paired objective span. Passing permits one discordant pair
among five points; it does not mean identical ranking or calibrated inference.
A smaller shift/span ratio may also reflect a larger objective span rather than
smaller numerical shifts, so inspect the absolute shifts and each loss component.

### Saved five-point joint rescoring result (2026-09-07)

The two historical runs were retrieved and rescored without reintegration.
Both record clean commit `aa9cd5d707c96b019d7df4e4ad0664979893cf70`, identical
input hashes, model settings apart from `lsmr_tol`, fixed-point schedules and
dependency versions. Thread-pool settings were not recorded, so this is not
validation of the newer pinned-BLAS execution baseline.

| Point | Joint objective, `1e-7` | Joint objective, `1e-8` | Rank, `1e-7` / `1e-8` |
| --- | ---: | ---: | ---: |
| paper-best | 133893.006 | 134094.153 | 2 / 3 |
| flatter/more triaxial | 134025.377 | 134045.821 | 3 / 2 |
| rounder | 134481.241 | 134561.068 | 4 / 4 |
| more concentrated | 135367.328 | 135539.789 | 5 / 5 |
| more extended | 133171.250 | 133184.022 | 1 / 1 |

All ten points are valid under joint rescoring: the solvers report success and
no orbits failed; the two formerly density-gated potentials now receive their
finite joint scores. `more_extended` remains best. Spearman and pairwise
agreement are both 0.9, with one discordant pair (paper-best versus flatter).
The maximum shift after subtracting the paper-best tolerance offset is 188.375,
or 7.9963% of the larger joint span (2355.767). The existing screening rule
therefore passes, but the complete ranking is not identical.

The velocity-only comparison has maximum differential shift 188.385 and span
1943.608 (9.6925%). The joint ratio is smaller mostly because its span is larger;
the absolute numerical sensitivity has barely changed. All weight solves have
status 2 (cost-change termination), not a strict KKT certificate. These five
points support a coarse ranking screen, not calibrated uncertainty or a claim
that individual orbit weights have been recovered.

Source files, SHA-256 manifests, configuration audit, comparison JSONs, a plot
and the reproducible report are kept locally under
`.agent-local/benchmarks/joint-ranking-20260907/`. The raw records remain outside
Git. The next production benchmark must use the chosen thread setting and the
joint recipe; historical rescoring is not a new adaptive optimization run.

## Historical velocity-only wide adaptive scan

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
PYTHONPATH="$PWD/Agama-master" conda run -n halo_lmc python -m halo_mw_lmc validate configs/runs/density_solved_r8_40_wide_scan.toml
```

Run optimization only (no static report):

```bash
PYTHONPATH="$PWD/Agama-master" conda run -n halo_lmc python -m halo_mw_lmc optimize configs/runs/density_solved_r8_40_wide_scan.toml
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

## Stage-1 joint-objective fixed-point screen

This stage moves from a single adaptive scan to a two-stage search: first a
parallel fixed-point screen to localize the active region, then a bounded GP
refinement inside that region. It runs the `density_velocity` objective so the
density term enters the objective (`J = chi2 / 2 + objective_velocity`) instead
of gating validity, which eliminates the `1e30` starvation that invalidated 22
of the 50 wide-scan points.

| parameter | screen box |
|---|---:|
| qhalo | [0.80, 1.28] |
| phalo | [0.70, 0.96] |
| rho0 | [5.55, 6.55] |
| rho0_plus_2logrs | [9.25, 10.20] |
| gamma | [0.70, 1.45] |

The design is 45 scrambled Sobol points (`scipy.stats.qmc.Sobol`, seed 0,
sliced from 64 for balance) plus three anchors (`paper_best`, the tol1e7
ranking best `more_extended`, and the wide-scan best). The box is the
convex hull of those three anchors with margin. Recipe:
`configs/recipes/zhu_2026_density_solved_r8_40_joint_screen.toml`; twelve shard run
configs `configs/runs/density_solved_r8_40_stage1_screen_shard01..12.toml`;
generator `scripts/generate_density_solved_r8_40_stage1_design.py`; driver
`scripts/run_density_solved_r8_40_stage1_screen.sh`, which caps the shared
server at 12 shards x `OMP_NUM_THREADS=4` = 48/112 cores (~43%) under `nice -n
10` and `OPENBLAS_NUM_THREADS=1` (the solve path is single-threaded; the pin
makes it deterministic per problem, Finding 9).

The screening recipe was named `zhu_2026_density_solved_r8_40_joint.toml` in
the original remote stage-1 commits. It is now named `joint_screen` to preserve
both that screening box and the local joint benchmark's narrower bounds.
Only its filename and recipe name changed; the twelve shard coordinates and
all numerical settings are unchanged. Existing resolved artifacts retain their
original recipe name. The stage-2 9.353 anchor also uses `joint_screen`.

Provenance note: the committed `wide_scan_best` anchor uses
rho0_plus_2logrs = 9.354, but the authoritative wide-scan record is 9.353
(rho0 5.616 + 2*log_rs 1.8685). A corrected 9.353 re-evaluation is scheduled
in stage 2; the 0.001 offset is negligible at the reported objective precision
but is recorded here for provenance.

Results (48/48 points finite under the joint objective; measured from
`.agent-local/tmp/analyze_stage1_v2.py`):

- convergence: 39 points reach cost-stall (status 2); 9 points stop at
  `weight_model.max_iter = 20000` (status 0, truncated solves);
- 7 points have density chi2 < 1e-6 (an exact density fit is achievable for
  those potentials under the underdetermined weight problem);
- 15 points carry 1--53 failed orbits; the old velocity-only validity gate
  would have excluded them;
- Spearman(joint, velocity) = 0.990, so the two objectives nearly agree on
  this set and the density term adds little discrimination except at the
  density-poor end;
- best converged, zero-failed-orbit point: J = 132834.861
  (qhalo 0.985, phalo 0.714, rho0 6.099, rho0_plus_2logrs 9.829, gamma 0.751),
  which improves on the wide-scan best objective 132872.5796;
- anchors: `wide_scan_best` ranks 5/48, `more_extended` ranks 9/48,
  `paper_best` ranks 26/48 (the fiducial potential is outside the active
  region);
- low-rho0_plus_2logrs edge signal: points with rho0_plus_2logrs in
  [9.28, 9.41] occupy ranks 3, 5, 8, 10, 12, 15, 16; points above 10.0
  settle in ranks 17, 27, 36, 41, 44--48 (except `more_extended` at rank 9);
- solve iterations: median ≈3009.5 over all 48 points (9 truncated at 20000),
  2185 over the 39 converged points.

Interpretation caveats: truncated solves make their J a lower bound (rank 1 is
one such point); exact-density-fit points make the density term
non-discriminating; failed-orbit points would have failed the old gate. The
screen identifies a region, not a converged posterior: rank 1 is a truncated
solve, so the best converged point (0.985, 0.714, 6.099, 9.829, 0.751; overall rank 2, behind only the truncated solve)
is the most reliable current optimum.

Reproduction: `scripts/generate_density_solved_r8_40_stage1_design.py`;
`scripts/run_density_solved_r8_40_stage1_screen.sh`; analysis
`.agent-local/tmp/analyze_stage1_v2.py`.

## Stage-2 protocol (approved, not yet executed at time of writing)

Stage 2 refines the active region with a bounded adaptive GP. It is a cold
start: the GP surrogate starts from an `adaptive` schedule with
`random_seed = 0` and `initial_point = "optimizer"`, and no stage-1 evaluation
is injected as a prior. Restricting the search box to the stage-1 active region
is experiment design (bound narrowing), not warm-start or historical-point
injection, so it is consistent with the cold-start contract. Unconverged
(truncated) stage-1 points are not reused as GP observations.

| parameter | stage-2 box |
|---|---:|
| qhalo | [0.93, 1.27] |
| phalo | [0.70, 0.96] |
| rho0 | [5.55, 6.42] |
| rho0_plus_2logrs | [9.20, 9.95] |
| gamma | [0.70, 1.40] |

60 iterations. Two side re-evaluations run independently: the exact anchor
rho0_plus_2logrs = 9.353 (correcting the 9.354 provenance offset), and the
stage-1 rank-1 point re-evaluated at `weight_model.max_iter = 60000` to check
whether its truncated objective improves materially.

Resource plan (respecting the shared-server cap): GP single process with
`OMP_NUM_THREADS = 24`; the two side runs at `OMP_NUM_THREADS = 16` each;
worst-case simultaneous 56/112 cores (50%). All runs keep
`OPENBLAS_NUM_THREADS = 1`.

Reproduction (configs authored by the stage-2 infrastructure work, to be
committed alongside this protocol): recipe
`configs/recipes/zhu_2026_density_solved_r8_40_joint_stage2.toml`; run config
`configs/runs/density_solved_r8_40_stage2_gp.toml`; side-run configs
`density_solved_r8_40_stage2_anchor_9353.toml` and
`density_solved_r8_40_stage2_rank1_converged.toml` (the latter uses recipe
`configs/recipes/zhu_2026_density_solved_r8_40_joint_maxiter60k.toml`).

## Corner constraint figure interpretation

The five-parameter corner figure
(`halo_mw_lmc/visualization/parameter_constraints.py`,
`build_parameter_constraints_corner_figure`) shows only `rho0`, `rs`, `gamma`,
`qhalo`, and `phalo`. Each off-diagonal panel draws the profiled
`delta_chi2 = 2.30` contours on GP-surrogate chi-square surfaces for the total
and density objectives, clipped to trial support and GP-uncertainty masks, with
the actual adaptive trials overlaid as points. Interpretation boundary: the
contours are a projection of the set of five-dimensional models compatible with
the data, not a posterior or calibrated uncertainty. `rho0` and `rs` are
coupled through the persisted `rho0_plus_2logrs` coordinate, so an elongated
band across `rho0`-`rs` is a degeneracy direction (compensating `rho0` and
`rs`), not independent freedom; the density term (`chi2`) is what pins the
three-dimensional DM distribution within `r <~ 50 kpc`.
