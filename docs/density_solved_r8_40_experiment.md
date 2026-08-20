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
