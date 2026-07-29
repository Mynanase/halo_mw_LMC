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
optional velocity likelihood terms. The primary entry point is
`run_skopt_lamost_4phi.py`; the main model evaluation is in
`skopt_oint_lamost_4phi.py`, and reusable code lives under `halo_mw_lmc/`.

Observational inputs, generated model directories, PDFs, and most research data
are intentionally excluded from Git. Do not assume that ignored data are absent
merely because `git status` does not show them.

## Local Python environment

- Preferred Conda environment: `dp-jax` (the name is lowercase).
- Preferred invocation form:

  ```bash
  conda run -n dp-jax python <command>
  ```

- Last verified environment facts:
  - Python 3.11.15
  - NumPy, SciPy, Matplotlib, and pytest are available.
  - Astropy, scikit-optimize, and AGAMA were not importable when this file was
    created. Re-check before assuming they are still missing.
- The repository-local `.venv` is not the preferred scientific runtime unless
  the user explicitly changes this convention.
- Do not install or upgrade packages in `dp-jax` without user approval. For
  disposable build experiments, use a temporary directory or temporary
  environment.

## Standard checks

Use the preferred Conda environment when its dependencies are sufficient:

```bash
conda run -n dp-jax python -m unittest discover -s tests -v
conda run -n dp-jax python -m py_compile \
  run_skopt_lamost_4phi.py plot_best_fit.py skopt_oint_lamost_4phi.py
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
- The active potential is the static, constant-shape fiducial model from Zhu
  et al. (2026), equations 6--8: Ferrers barred bulge, thin and thick AGAMA
  `Disk` components, and a generalized-NFW triaxial `Spheroid` halo. Its
  source-of-truth implementation is `halo_mw_lmc/potentials.py`; see
  `docs/zhu_2026_potential.md`.
- Do not restore the abandoned time-dependent LMC backward/forward integration
  to this model. The paper's fiducial fit instead removes the mean LMC-induced
  velocity signal from the observational data.
- A synthetic AGAMA timing benchmark found the paper potential's orbit
  integration to be about 2.6 times the simplified potential; see
  `docs/zhu_2026_potential.md`. Re-run a full-catalogue benchmark in the
  production environment before a large optimization scan or after changing
  the component representation.

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
- Do not commit generated data, model outputs, local environments, or
  `.agent-local/`.
- Add reusable scientific decisions to this file or tracked documentation;
  keep machine paths, secrets, and private materials out of tracked files.
- Before changing the physical model, state the assumed paper/model and keep
  benchmark changes separate from production changes.
