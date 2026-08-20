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

## Preserve the server state and lock the code

Before replacing or committing a dirty server checkout, preserve its tracked
diff and status in the ignored local research area:

```bash
mkdir -p .agent-local/benchmarks/r8_40_preflight
git rev-parse HEAD > .agent-local/benchmarks/r8_40_preflight/original-head.txt
git status --porcelain --untracked-files=all \
  > .agent-local/benchmarks/r8_40_preflight/original-status.txt
git diff --binary HEAD \
  > .agent-local/benchmarks/r8_40_preflight/original-worktree.patch
```

Commit the implementation and configs once, deploy that exact commit to the
server, and confirm that `git status --porcelain --untracked-files=all` is empty.
Do not commit or upload the `.agent-local` patch. Record the full commit for all
five invocations:

```bash
R8_40_LOCKED_COMMIT="$(git rev-parse HEAD)"
```

The launcher refuses to run if this SHA changes, if the worktree becomes dirty,
if an output directory already exists, or if `/usr/bin/time` is not GNU time.

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

Run each case separately and in this order. The optional preflight-only command
does not integrate orbits:

```bash
scripts/run_density_solved_r8_40_case.sh \
  configs/runs/density_solved_r8_40_benchmark.toml \
  "$R8_40_LOCKED_COMMIT" --preflight-only

scripts/run_density_solved_r8_40_case.sh configs/runs/density_solved_r8_40_benchmark.toml "$R8_40_LOCKED_COMMIT"
scripts/run_density_solved_r8_40_case.sh configs/runs/density_solved_r8_40_tol1e7_benchmark.toml "$R8_40_LOCKED_COMMIT"
scripts/run_density_solved_r8_40_case.sh configs/runs/density_solved_r8_40_tol1e8_benchmark.toml "$R8_40_LOCKED_COMMIT"
scripts/run_density_solved_r8_40_case.sh configs/runs/density_solved_r8_40_reg1e5_benchmark.toml "$R8_40_LOCKED_COMMIT"
scripts/run_density_solved_r8_40_case.sh configs/runs/density_solved_r8_40_reg1e4_benchmark.toml "$R8_40_LOCKED_COMMIT"
```

Each run contains `benchmark_metadata/time-v.txt`, input hashes, environment
versions, the command, logs, and Git evidence. If the workflow fails after
creating its cold-start directory, the exit trap still copies the available
metadata into that incomplete run for investigation; never delete it merely to
reuse its name.

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

The tight-tolerance stability flag requires the 1e-7 and 1e-8 cases to agree to
`1e-5` in relative selected objective, `1e-3` in maximum absolute shell/phi
chi-square per bin, and `1e-2` in normalized-weight L1 distance. Regularization
cases are diagnostic: review effective orbit count, maximum weight fraction,
active orbit count, velocity objective, and gates without automatically changing
the default strength.
