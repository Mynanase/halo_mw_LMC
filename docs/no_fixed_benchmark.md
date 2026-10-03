# No-Fixed full-catalogue benchmark

Run this benchmark on the production server before starting the 10-trial pilot
or the 1000-trial optimization. It performs one complete evaluation at the Zhu
et al. paper-best potential: orbit integration, sparse density response, profile
weight solve, velocity likelihood, artifact persistence, and static reporting.

This is an engineering and numerical benchmark. The current density target is a
DESI K-giant tracer model while the orbit seeds and velocity constraints are
from LAMOST. Do not interpret the result as a production Milky Way potential
inference until the tracer populations, coordinate conventions, and selection
functions have been shown to be compatible.

## Server data layout

The ignored research inputs must be staged under the same repository-relative
paths used by the checked-in run configuration:

```text
data_for_model/
  lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt
  synthetic/desi_year1_kgiants_25x25x4.npz
```

The LAMOST table must provide the named columns
`x_gc,y_gc,z_gc,vx_gc,vy_gc,vz_gc,vr_err,vphi_err,vthe_err`. A catalogue `w`
column is not required in `density_solved` mode.

Verify the generated density artifact after transfer:

```bash
sha256sum data_for_model/synthetic/desi_year1_kgiants_25x25x4.npz
```

The expected SHA-256 is:

```text
23297298db6751446100778288b0656987a3c68bb54bf1ef6c9be5d858581e5c
```

## Preflight and execution

Activate the production Python environment, then run:

```bash
python -c "import agama, skopt, scipy, matplotlib"
halo-mw-lmc validate configs/runs/density_solved_benchmark.toml
halo-mw-lmc preflight configs/runs/density_solved_benchmark.toml --stage run
halo-mw-lmc coverage configs/runs/density_solved_benchmark.toml
mkdir -p .agent-local/benchmarks
/usr/bin/time -v \
  -o .agent-local/benchmarks/density_solved_paper_best.time.txt \
  halo-mw-lmc run configs/runs/density_solved_benchmark.toml
```

Coverage and optimization are cold-start operations. Their configured output
directories must not exist before execution; never delete or overwrite a
previous run merely to reuse its name.

The default command performs the single optimization evaluation and then builds
the static report exclusively from the saved best snapshot. It does not
reintegrate the orbits for plotting.

## Acceptance checks

The trial log and `sample.dat` must report:

- a converged inner weight solve;
- finite selected, velocity-only, and density+velocity objectives;
- finite density chi-square and `density_chi2_per_bin <= 2.0`;
- successful and failed orbit counts;
- a positive finite total weight (no `sum(w) = 1` constraint is imposed; the
  unit-mass target sets the scale);
- solver diagnostics: `max_iter` used, `weight_solver_iterations` (`nit`),
  `weight_solver_optimality`, and `weight_solver_cost` finite;
- effective orbit count, active orbit count, and maximum weight fraction.

The run is not accepted if the selected objective uses the `1e30` invalid-trial
penalty, any orbit fails without investigation, or required artifacts are
missing. Inspect the complete best weights in `best/evaluation.npz`; the artifact
loader rejects non-finite or negative saved weights.

Expected outputs are:

```text
runs/density-solved-paper-best-benchmark/
  resolved_config.json
  weight_model_inputs.npz
  sample.dat
  best/metadata.json
  best/evaluation.npz
  inspection.json
  report/manifest.json
  report/
```

Record the wall time and peak resident memory from
`.agent-local/benchmarks/density_solved_paper_best.time.txt`, plus the run
directory size. Use those measurements to budget the 10-trial pilot. Do not
start the 1000-trial run until the pilot has been reviewed for density-gate pass
rate, objective behavior, failed orbits, and weight concentration.
