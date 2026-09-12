# Repository architecture

> Branch contract (`codex/flatten-research-native`, approved 2026-09-13).
> This describes the flat research-native layout. `main` keeps the earlier
> layered layout (`core/`, `data/`, `workflows/`, `visualization/`) until the
> flatten is validated and merged. The numerical contracts below are
> unchanged by the flatten.

## Stable numerical contract

The numerical modules accept only explicit values and NumPy-compatible
arrays:

| Quantity | Contract |
| --- | --- |
| Seed phase space | `float64 (N,6)`, columns `(x,y,z,vx,vy,vz)`, kpc and km/s |
| Seed weights | `float64 (N,)`, finite and non-negative; fixed from catalogue or profiled per trial |
| Target density/error | `float64 (n_R,n_z,n_phi)`, strict `(R,z,phi)` order |
| Analytic density grid | Cell-volume averages using `u=R^2/2`, `z`, `phi` quadrature |
| Orbit samples | `float64 (M,6)` plus integer parent `seed_index (M,)` |
| Orbit density response | CSR `(n_cells,n_successful_orbits)`, C-order flattened `(R,z,phi)` rows |
| Velocity grid | `(r,theta,phi,v)`, radians and km/s, 201 fitting bins by default |
| Trial result | `ModelEvaluation` containing arrays and scalars, never paths or figures |

The metadata-free legacy density file is flattened in `(z,R,phi)` order and is
accepted only for the historical 25x25x4 grid. Its transpose into the model
order happens exactly once in `catalogue.py`. Custom grids use an NPZ
target carrying explicit `r`, `z`, and `phi` edges, which are checked before
model evaluation.

## Flat module map

One module per pipeline stage, directly under `halo_mw_lmc/`:

| Module | Stage |
| --- | --- |
| `potential.py` | Static Zhu et al. (2026) potential |
| `orbits.py` | AGAMA orbit integration and spherical phase-space transforms |
| `grids.py` | Cylindrical and spherical velocity grid arrays |
| `density.py` | Density comparison, tracer density, sparse orbit response |
| `weights.py` | Catalogue weights and the profiled non-negative density solve |
| `velocity.py` | Velocity histograms, uncertainty, log-likelihood |
| `catalogue.py` | The single data boundary: catalogue, weights, density targets; documents array keys |
| `config.py` | Recipe/run TOML -> plain nested dicts; one visible defaults table |
| `prepare.py` | Preflight checks and one-time data preparation |
| `coverage.py` | Raw catalogue number-density coverage diagnostics |
| `evaluate.py` | One potential -> density comparison, weight solve, velocity score |
| `optimize.py` | Trial loop, sample file, fixed-point and adaptive ask/tell |
| `run.py` | The `run` lifecycle orchestration |
| `report.py`, `inspection.py`, `artifacts.py` | Versioned persistence, rebuild, reporting |
| `benchmark.py`, `solver_budget.py`, `weight_solver_benchmark.py` | Production gate and experiment runners |
| `synthetic_density.py` | Simulation-derived target generation |
| `plot_model.py`, `plot_coverage.py`, `plot_constraints.py`, `plot_weights.py`, `plot_convergence.py` | Figures from saved artifacts |
| `cli.py` | Argparse subcommands; flags are the docs |

## Dependency rules

Enforced by tests:

- `optimize.py` must not import `plot_*`, `report.py`, or Marimo;
- `plot_*`, `report.py`, and `apps/` read persisted artifacts only — they
  never reopen the source catalogue or rerun AGAMA integration;
- numerical modules (`potential`, `orbits`, `grids`, `density`, `weights`,
  `velocity`) never import configuration, catalogue loading, artifacts,
  plotting, Astropy, Matplotlib, scikit-optimize, or Marimo.

AGAMA is imported lazily inside `orbits.py` because it is required only when
a trial is actually evaluated.

## Configuration ownership

The reusable recipe owns scientific choices: potential, grid edges, fit masks,
velocity likelihood, weight model, outer objective, orbit sampling, search
coordinates, bounds, and rounding.
The run file owns data paths, run identity, output path, iterations, random
seed, coverage display settings, and report-only velocity coarsening.

Both files are parsed with plain `tomllib` into nested dicts. There are no
schema classes and no exact-field validation; `config.py` applies one visible
defaults table, resolves derived values (degree-to-radian, edge arrays), and
documents every key once in its module docstring. Scientific functions take
plain values or dicts, with defaults in their signatures. Core functions
never know which file supplied a value.
The resolved JSON written into every run is the provenance record used by
analysis; the checked-in TOML remains the human-editable source.

## Three thin defenses

1. Data boundary: `catalogue.py` (catalogue, weights, density targets) and the
   `solver_budget.py` experiment plan check units, shapes, finiteness, and
   value ranges exactly once at load time. Nothing downstream re-checks.
2. Run-directory isolation: `prepare.py` refuses an existing output directory
   (cold-start only; resumption is unsupported and stated as such).
3. Artifact provenance: writers stamp schema versions and resolved config;
   readers check versions when a silent misread is possible.

Everything else fails fast with raw tracebacks; there is no exception
hierarchy beyond `ValueError` at the data boundary.

## Execution and artifacts

`prepare.py` owns stage-aware dependency, input, grid, weight-audit, and
output-conflict checks. For `run`, it reads catalogue and target exactly once
and hands the prepared arrays to the numerical path before any run directory
is created. Coverage uses a catalogue-only payload and never reads the
target or probes numerical dependencies.

Expensive integration is confined to `optimize.py`. A common trial
loop writes one sample row per evaluated point and replaces only the current best
snapshot. Its fixed wrapper consumes explicit points sequentially and never
imports skopt; its adaptive wrapper alone owns `Optimizer.ask/tell`. Evaluation,
adaptive `tell()`, and persistence receive the same rounded coordinates.

The default `run` lifecycle is validate -> preflight/prepare -> fixed evaluation or
adaptive optimization -> numerical artifact inspection -> managed report -> saved
inspection. Numerical failure preserves partial artifacts. A report failure does
not invalidate completed numerical artifacts.

`inspection.py`, `report.py`, and Marimo consume artifacts. They do not reopen the
source catalogue and never reconstruct missing results through AGAMA. Managed
`report/` publication is staged and validated before an optional directory
replacement. `inspection.json` is a derived cache; resolved configuration,
`sample.dat`, and `best/` remain authoritative.

## Weight-model boundary

Both weight modes share preparation, orbit integration, density comparison,
velocity likelihood, optimization, and artifact code. Only the weight provider
inside one evaluation changes:

- `catalogue_fixed` maps the catalogue `w` column to every orbit sample and
  preserves the historical global density normalization;
- `density_solved` builds a sparse equal-time orbit response, profiles one
  non-negative weight per successful seed against the target density, and
  distributes each solved orbit weight over that orbit's actual finite samples
  for velocity scoring.

The sparse response and solver live in `density.py` and `weights.py`; neither
knows about TOML, paths, AGAMA, optimization, or plotting. No-Fixed uses
density normalization `none` so there is no weight/scale degeneracy. Failed
seed integrations retain zero slots in the persisted full-catalogue weight
vector.

## Extension rules

- A new equation or reusable array diagnostic joins the flat numerical module
  that owns it and must be testable with small in-memory arrays.
- A new survey/file format joins `catalogue.py` and translates into the
  existing array contract at that one boundary.
- A new expensive execution mode is a new flat stage module that writes
  versioned artifacts; source catalogues and plotting scripts never become
  runtime dependencies of numerical modules.
- A new figure is a `plot_*.py` module reading saved artifacts; `optimize.py`
  must not import it.
- The five-parameter corner constraint figure (`plot_constraints.py`,
  `build_parameter_constraints_corner_figure`) reads only the persisted
  `sample.dat` trial table and resolved-config search bounds, and is
  display-only: it never re-runs orbit integration or mutates solver weights.
  It writes the report artifact `parameter_constraints_corner.pdf` plus a
  `parameter_constraints_corner_surfaces.npz` of profiled GP surfaces for
  re-plotting without recomputation.
- A new interactive view reads artifacts from `apps/` and must not trigger
  model evaluation.
- Historical code stays under `archive/` and is never used as a compatibility
  dependency of the active package.
