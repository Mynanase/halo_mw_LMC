# Density-solved 8--50 kpc support experiment

## Purpose

This experiment tests one isolated modelling change: extend the density fit
mask from `15 <= r < 40 kpc` to `8 <= r < 50 kpc`, matching the radial support
of the velocity likelihood. It keeps the Zhu et al. (2026) static,
constant-shape fiducial potential, orbit sampling, optimizer coordinates,
weight solver, objective, and DESI synthetic target unchanged.

The experiment does not replace the conservative checked-in recipe. Its recipe,
run ID, optimization output, and coverage output are all separate:

```text
configs/recipes/zhu_2026_density_solved_r8_50.toml
configs/runs/density_solved_r8_50_benchmark.toml
runs/density-solved-r8-50-paper-best-benchmark/
data_coverage-density-solved-r8-50-paper-best-benchmark/
```

The existing synthetic NPZ is reusable because both recipes use the same
`25x25x4` `(R,z,phi)` grid. The density fit mask is calculated at run time and
is not baked into the target arrays.

## Remaining support limitation

Only the radial boundaries are aligned. The density mask still requires
`|z| >= 2 kpc`, while the velocity likelihood does not currently expose an
equivalent explicit mask. Before interpreting this experiment scientifically,
audit whether any orbit contributes samples to valid velocity-likelihood cells
but has zero response inside the density fit mask. Do not assume radial
alignment alone proves full phase-space support alignment.

## Server procedure

Use the production environment and the vendored AGAMA checkout:

```bash
export PYTHONPATH="$PWD/Agama-master"
conda run -n halo_lmc python -c "import agama, skopt, scipy, matplotlib"
conda run -n halo_lmc python -m halo_mw_lmc \
  -v configs/runs/density_solved_r8_50_benchmark.toml
conda run -n halo_lmc python -m halo_mw_lmc \
  -c configs/runs/density_solved_r8_50_benchmark.toml
mkdir -p .agent-local/benchmarks
/usr/bin/time -v \
  -o .agent-local/benchmarks/density_solved_r8_50_paper_best.time.txt \
  conda run -n halo_lmc python -m halo_mw_lmc \
  configs/runs/density_solved_r8_50_benchmark.toml
```

Coverage and optimization are cold-start operations. Use new output paths if
either configured directory already exists; do not delete a previous run to
reuse its name.

## Paired comparison

Run the conservative benchmark separately with
`configs/runs/density_solved_benchmark.toml`. Compare the two paper-best trials
before any multi-trial scan:

- fitted density-bin count and `density_chi2_per_bin`;
- `weight_sum`, effective orbit count, active orbit count, and maximum weight
  fraction;
- solver iterations, optimality, cost, and convergence status;
- velocity log likelihood by component and phi sector;
- raw catalogue coverage in `8--15`, `15--40`, and `40--50 kpc`;
- failed orbit count, wall time, peak memory, and run-directory size.

The expanded experiment is not accepted if the selected objective receives the
`1e30` invalid-trial penalty, the density gate fails, the inner solve does not
converge, or the added radial shells cause pathological weight concentration.
The configured 10% synthetic density error must still be treated as a modelling
choice rather than a measured posterior uncertainty.
