# Project instructions for agents

## Scope

This file applies to the entire repository. It contains shared, non-secret
project context and should remain suitable for committing to Git.

Machine-specific notes, private papers, unpublished material, and large local
artifacts belong under `.agent-local/`, which is intentionally ignored by Git.
Agents may read that directory when it exists, but must not assume that it is
available on another machine or upload its contents to an external service
without explicit user permission.

## Project overview

This repository implements a Zhu-style empirical orbit-superposition model for
the Milky Way stellar halo, with explicit `(R, z, phi)` density comparison and
optional velocity likelihood terms. The supported entry point is
`python -m halo_mw_lmc configs/runs/fix_weight.toml`; use `-v`, `-c`, or `-o`
for validation-only, coverage-only, or optimization-only execution. Numerical code
lives under `halo_mw_lmc/core/`, file adapters under `halo_mw_lmc/data/`, and
expensive execution under `halo_mw_lmc/workflows/`.

Observational inputs, generated model directories, PDFs, and most research data
are intentionally excluded from Git. Do not assume that ignored data are absent
merely because `git status` does not show them.

Use `python -m halo_mw_lmc -c configs/runs/fix_weight.toml` for data-only
coverage diagnostics before changing
the statistical treatment of sparse or empty 6D regions. Its sampling-density
plots are raw catalogue number densities, not selection-function-corrected
physical stellar densities.

## Local Python environment

- Two Conda environments are used, depending on where the run happens:
  - **Server / production runs**: `halo_lmc`. Always invoke it with

    ```bash
    conda run -n halo_lmc python <command>
    ```

  - **Local debugging**: `dp_jax` (the name has an underscore, not a dash).
    Use the same `conda run -n dp_jax python <command>` form. `dp_jax` is
    only present on the developer's local machine; do not assume it exists
    on the server.
- Do not silently switch between `halo_lmc` and `dp_jax`. If a command is
  requested in one environment and the dependency probe shows it is
  missing there, report the missing dependency instead of falling back to
  the other environment without asking.
- Last verified environment facts (`halo_lmc` on the server, 2026-08-04):
  - Python 3.12.13
  - NumPy, SciPy, Matplotlib, scikit-optimize (0.10.2), and astropy (8.0.1)
    are available.
  - AGAMA is vendored in the repository at `Agama-master/` (exact casing)
    and is **not** managed by pip/conda. It is a compiled AGAMA checkout;
    `agama.so` and the Python interface live in that directory. From
    `halo_lmc`, import it by putting the directory on `PYTHONPATH`:

    ```bash
    PYTHONPATH=/path/to/halo_mw_LMC/Agama-master \
      conda run -n halo_lmc python -c "import agama"
    ```

    Do not install or upgrade AGAMA through conda/pip. If the compiled
    extension no longer matches the current Python/NumPy ABI, rebuild it
    with the project's own Makefile. Last verified import: AGAMA 1.0,
    compiled 2026-07-24, on the server.
- The repository-local `.venv` is not the preferred scientific runtime
  unless the user explicitly changes this convention.
- Do not install or upgrade packages in `halo_lmc` or `dp_jax` without
  user approval. For disposable build experiments, use a temporary
  directory or temporary environment.

## Standard checks

Use the preferred Conda environment when its dependencies are sufficient:

```bash
conda run -n halo_lmc python -m unittest discover -s tests -v
conda run -n halo_lmc python -m compileall -q halo_mw_lmc apps/results.py
```

If a check needs Astropy, scikit-optimize, or AGAMA, first probe the selected
environment and report the missing dependency instead of silently switching
environments or installing packages.

## Current modelling decisions

- The active pipeline is cold-start only. Do not add warm-start, replay, resume,
  or historical-point injection to the main optimizer.
- Warm-start design is deferred until after October 2026 and should be developed
  in a separate experimental branch or entry point before consideration for the
  main pipeline.
- Cold-start runs use a fixed default random seed and must write into a new model
  directory rather than append to an existing `sample.dat`.
- Keep the coordinates used for model evaluation, `optimizer.tell()`, and
  persisted samples identical.
- Keep the 201-bin velocity grid used by the likelihood separate from plot
  smoothing. Diagnostic velocity plots aggregate three adjacent fitting bins
  by default (about 24 km/s per plotted bin); changing the plotting factor must
  not change the fitted likelihood.
- The default fixed-weight recipe uses the per-star `w` column from
  `halo_clean_N.txt` for every orbit sample in both density and velocity
  histograms. The explicit experimental `density_solved` recipe instead
  profiles non-negative orbit weights from the three-dimensional target for
  every trial potential; do not mix these two weight semantics within a run.
- In `density_solved` mode, use the sparse equal-time `(R,z,phi)` response,
  normalize each orbit by its actual finite sample count, force density scale
  to one, and use the same solved trial weights for the velocity likelihood.
  The target is normalized to unit mass inside the fit mask; the inverse-error
  least-squares solve sets the total weight scale directly, so do not impose
  `sum(w) = 1` or renormalize weights afterwards. Drop orbits with strictly
  zero fit-region response from the solve and restore zero weight in the full
  array. Use a tight fixed inner `lsmr_tol` (default `1e-6`) for the TRF outer
  solve: SciPy's `"auto"` rule keeps the inner LSMR solve coarse while the
  optimality residual is large and stalled outer convergence for tens of
  thousands of iterations, whereas the fixed value converges in a few hundred.
  Persist both velocity-only and density+velocity objectives; velocity-only
  runs must enforce the configured density chi2-per-bin gate.
- The density fit mask covers `15 <= r < 40 kpc` while the velocity likelihood
  still starts at `r = 8 kpc`; orbits outside the density mask have no density
  evidence and get weight only through the L2 penalty and solver scale. Aligning
  these boundaries (velocity floor to 15 kpc, or density coverage down to 8 kpc)
  is an open decision; do not assume it is resolved without asking.
- The separate `zhu_2026_density_solved_r8_40` experiment tests radial alignment
  at `8 <= r < 40 kpc` with velocity-matched shell/phi gates. It does not replace
  the conservative recipe or resolve the remaining `|z| >= 2 kpc` density-only
  mask. Run its five paper-best one-trial cases from one clean, locked commit as
  documented in `docs/density_solved_r8_40_experiment.md`; do not advance to a
  multi-trial scan until the tolerance and regularization diagnostics are reviewed.
- Exclude `r < 8 kpc` from the velocity likelihood because the empirical orbit
  library is incomplete there. The inner velocity histograms may remain
  available for diagnostics, but must not affect the optimizer objective.
- Keep initial-cell target-derived representative weights experimental and out
  of the optimizer. They are distinct from the trial-specific orbit-response
  solve documented in `docs/density_solved_weights.md`.
- The experimental No-Fixed target is generated from the local DESI year-1
  K-giant analytic model through `configs/synthetic_density/`. Treat it as a
  relative tracer-density shape with a configured synthetic error, not an
  absolute density measurement or posterior uncertainty. Preserve cell-volume
  averaging and the provenance described in `docs/desi_density_model.md`.
- The active potential is the static, constant-shape fiducial model from Zhu
  et al. (2026), equations 6--8: Ferrers barred bulge, thin and thick AGAMA
  `Disk` components, and a generalized-NFW triaxial `Spheroid` halo. Its
  source-of-truth implementation is `halo_mw_lmc/core/potentials.py`; see
  `docs/zhu_2026_potential.md`.
- Do not restore the abandoned time-dependent LMC backward/forward integration
  to this model. The paper's fiducial fit instead removes the mean LMC-induced
  velocity signal from the observational data.
- A synthetic AGAMA timing benchmark found the paper potential's orbit
  integration to be about 2.6 times the simplified potential; see
  `docs/zhu_2026_potential.md`. Re-run a full-catalogue benchmark in the
  production environment before a large optimization scan or after changing
  the component representation.
- Use `configs/runs/density_solved_benchmark.toml` for the one-trial,
  paper-best No-Fixed production benchmark. Treat it as engineering validation,
  not a production inference, and review `docs/no_fixed_benchmark.md` before
  advancing to a multi-trial scan.

## Local research material

Use these ignored locations when present:

- `.agent-local/papers/`: local papers, supplements, and citation exports.
- `.agent-local/notes/`: private research notes and paper/model mappings.
- `.agent-local/benchmarks/`: temporary benchmark inputs and raw timing output.

When a paper is important to implementation, record a short local index entry
with its title, authors, year, arXiv/DOI, local filename, and the specific model
section or equation being used. Do not commit copyrighted PDFs or unpublished
notes unless the user explicitly requests and authorizes it.

## Repository hygiene

- Preserve unrelated user changes and untracked files.
- Keep `archive/` out of the active import graph. It preserves historical source
  layout and known failures; do not repair or use it as a compatibility layer.
- Keep optimization free of plotting imports. Reports and Marimo must read
  persisted artifacts and must not re-run missing orbit integrations.
- Do not commit generated data, model outputs, local environments, or
  `.agent-local/`.
- Add reusable scientific decisions to this file or tracked documentation;
  keep machine paths, secrets, and private materials out of tracked files.
- Before changing the physical model, state the assumed paper/model and keep
  benchmark changes separate from production changes.
