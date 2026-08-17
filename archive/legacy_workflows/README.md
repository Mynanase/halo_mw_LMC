# Legacy workflow snapshot

## Provenance and scope

These files were moved from their original repository locations while
refactoring from source commit `a926d89` (`Restore catalogue orbit weights and
velocity mask`).  The move preserves file contents and the relative `back/`
and `funcs/` layouts.  Git records the files as renames, so earlier history can
be inspected with `git log --follow`.

This is a source snapshot, not a runnable release.  "Complete" here means that
the related source files present at `a926d89` were retained together.  It does
not mean that missing historical modules, ignored observational data, old
environments, or generated output were reconstructed.

Files now directly under this directory originally lived at the repository
root.  Files under `back/` and `funcs/` retain those original relative paths.
No `__init__.py` is provided: the archive is deliberately outside the active
Python package.

## Original workflow groups

- `run_skopt_lamost`, `run_skopt_lamost_4phi.py`, and
  `skopt_oint_lamost_4phi.py` are the optimizer shell, command-line driver, and
  evaluator from immediately before the package-level workflow refactor.
  `plot_best_fit.py` and `plot_data_coverage.py` are the corresponding
  pre-refactor diagnostic entry points.  They are retained as a production
  snapshot, but they are no longer formal entry points; current runs use the
  installed `halo-mw-lmc` package CLI (or `python -m halo_mw_lmc`).
- `run_skopt_mw.py`, `back/run_GPry_mw.py`, and
  `back/run_GPry_mw_mpi.py` are old axisymmetric optimizer drivers.
- `back/Bayes_oint_rot_mw_disk2.py` and
  `back/Bayes_oint_mw_disk2_LMC.py` contain older axisymmetric model
  evaluators.
- `run_skopt_LMC_back.py` and `Bayes_oint_LMC_back.py` implement the abandoned
  time-dependent LMC backward/forward experiment.  This is not the fiducial
  Zhu et al. (2026) production model.
- `Calculate_obs.py`, `Read_obs.py`, `velocs.py`, and `velocs_nerr.py` are
  shared histogram and velocity helpers for those older evaluators.
- `Calculate_obs_4phi.py`, `Read_obs_4phi.py`, and `_04_Agama_.py` are the
  compatibility facades that remained during the first `(R,z,phi)` refactor.
  Their supported functionality has moved into the active package.
- `_00_Data_.py`, `_02_SF_.py`, and `coords.py` are stand-alone plotting,
  selection-function, and coordinate-preprocessing utilities.
- `test_gpry.py`, `back/test_gpry.py`, `testtestrun_gpry_mw.py`, and
  `try_GP.py` are exploratory optimizer or surrogate-model prototypes, not
  tests in the current test suite.
- `funcs/` is an unused collection of older mathematical helpers.  Its modules
  were written as loose scripts rather than as a Python package.

## Import and path assumptions

The old code assumes that the former repository root is on `PYTHONPATH`.
Modules in `back/` use unqualified imports such as `_04_Agama_`, `velocs`,
`Calculate_obs`, and `Read_obs`; several modules in `funcs/` likewise import
siblings by bare module name.  Moving the files into an archive intentionally
does not rewrite those imports or imply that they are supported.

The scripts also contain machine-specific paths, principally
`/home/lzhu/halo_mw_LMC/`, `/home/lzhu/MW_Bayes/`, and
`/Users/yang/Desktop/`.  They assume output subdirectories such as `Orbits/`,
`SB_Rz/`, `params/`, `vvhist/`, `llp/`, and `figure/` and often perform work or
write files immediately when imported.

The archived `_04_Agama_.py` is the compatibility adapter that existed at
`a926d89`; it imports the then-current `halo_mw_lmc.orbits` module.  It is not
the self-contained AGAMA implementation from the earliest repository history.
That module path and other package paths imported by the pre-refactor entry
points changed during the new core/workflow split, and the active package does
not promise compatibility with them.

## Data that is not included

The workflows expect ignored observational and generated files, including at
least:

- `data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/halo_clean_N.txt`;
- `data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/halo_clean_S.txt`;
- `data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/SB_Rz_25_cleanrG12-30.txt`;
- the flattened `(R,z,phi)` density and velocity products consumed by the
  `Read_obs_4phi.py` compatibility readers;
- optimizer samples, orbit libraries, PDFs, checkpoints, and other generated
  model directories.

These data and outputs remain outside Git.  Private material under
`.agent-local/` is also intentionally excluded.

## Historical dependencies

Across the snapshot, the imported dependency set includes NumPy, SciPy,
Matplotlib, Astropy, AGAMA, galpy, scikit-optimize, emcee, mpi4py, GPRy,
pandas, scikit-learn, tqdm, and joblib.  No compatible environment is defined
for this archive.  In particular, some APIs used here belong to old package
versions, such as `emcee.mpi_pool.MPIPool` and scikit-learn's removed
`load_boston` dataset.

The loose `funcs/` collection also contains Python 2-era code.  Do not install
or downgrade packages in the production `dp-jax` environment to run these
files.

## Known incomplete or broken code

The following defects were present before archival and are preserved rather
than silently repaired:

1. `Bayes_oint_mw_disk2.py` does not exist in any Git object or branch in this
   repository, but it is imported by `run_skopt_mw.py`,
   `back/run_GPry_mw.py`, `back/run_GPry_mw_mpi.py`, and
   `testtestrun_gpry_mw.py`.  `back/Bayes_oint_rot_mw_disk2.py` may be related,
   but there is not enough evidence to rename it or treat it as a drop-in
   replacement.
2. `run_skopt_LMC_back.py` is not valid Python: adjacent `Real(...)` entries in
   its search-space list are missing commas.
3. `testtestrun_gpry_mw.py` has an unexpected indentation at its `return
   ll_tot` statement.  It also references an undefined `ll_tot` and later
   constructs an ensemble sampler with an undefined `ll_submit_one` callback.
4. `funcs/hermite.py` uses a Python 2 `print` statement and therefore fails to
   parse under Python 3.  Several `funcs/` imports also require manually adding
   that directory to `PYTHONPATH`.
5. `back/run_GPry_mw.py` and `back/run_GPry_mw_mpi.py` are byte-for-byte
   identical despite their different names.  Both are retained to preserve
   provenance.
6. The archived drivers execute expensive work at module import time, append
   to output files, and use hard-coded iteration counts and random behaviour.
   They are unsafe to use as libraries.
7. The required data products, directory trees, package versions, and original
   machine configuration were never committed, so the old results cannot be
   reproduced from this snapshot alone.
8. The time-dependent LMC backward/forward calculation is an abandoned model
   path and must not be reintroduced into the active fiducial pipeline.
9. The archived pre-refactor production entry points import former paths such
   as `halo_mw_lmc.config`, `halo_mw_lmc.plotting`, and
   `halo_mw_lmc.orbits`.  Those paths are not a compatibility contract of the
   reorganized package, so the archived commands may not import successfully
   against the current checkout.

Any attempt to revive one of these workflows should begin in a separate
experimental branch, explicitly reconstruct its data and environment contract,
and add tests before changing the archived snapshot.
