# Scientific project blueprint

## Purpose and project type

`halo_mw_LMC` is a continuously evolving research repository, not a
general-purpose public astronomy library. Its purpose is to test how well a
Zhu-style empirical orbit-superposition model describes the Milky Way stellar
halo when density and optional velocity information are compared on explicit
azimuth-resolved grids.

The repository should remain small enough for one researcher to maintain while
making every expensive scientific run reproducible. Reusable numerical
behavior belongs in the package; experiment variants belong in checked-in
configuration; reports read persisted artifacts.

## Scientific goals and non-claims

The supported scientific questions are:

1. Whether the static Zhu et al. (2026) potential and catalogue-weighted orbit
   library reproduce the configured density and velocity summaries.
2. Whether an explicitly experimental density-profiled weight model remains
   numerically stable and preserves potential discrimination.
3. How data support, solver tolerance, regularization, and radial masks affect
   those comparisons.

Current results do not by themselves establish:

- a posterior distribution or calibrated uncertainty on the Milky Way
  potential;
- an absolute, selection-corrected stellar density from catalogue coverage
  plots or the synthetic DESI target;
- stable potential recovery merely because density gates pass;
- complete phase-space support alignment while the density-only
  `|z| >= 2 kpc` mask remains;
- validity of warm-start, replay, or historical-point injection.

## Supported model paths

### Conservative baseline: `catalogue_fixed`

The production baseline maps each catalogue star's `w` value to its orbit
samples. The optimizer changes only the configured potential coordinates.
This is the reference path for identifying whether a new experimental method
has changed the scientific meaning of the model.

### Experimental path: `density_solved`

Each trial potential builds a sparse equal-time `(R,z,phi)` orbit response and
profiles non-negative orbit weights against a unit-mass density target. The
same solved weights score the velocity likelihood. The solver determines the
weight scale, so the weights are not post-normalized and a second density scale
is forbidden. The full numerical contract is maintained in
[density_solved_weights.md](density_solved_weights.md).

## Data and execution flow

```text
catalogue + density target
        |
        v
recipe TOML + run TOML
        |
        v
validate + stage-aware preflight/one-pass preparation
        |
        v
orbit integration and trial evaluation
        |
        v
fixed-point schedule or adaptive ask/tell optimizer
        |
        v
versioned numerical artifacts
        |
        v
artifact inspection + managed static report + Marimo/comparison scripts
```

The recipe owns scientific choices; the run file owns paths, identity, seed,
iteration count, and report/coverage settings. Exact array and dependency
contracts live only in [architecture.md](architecture.md).

The daily `run` command follows this sequence without changing the scientific
model: validate → preflight/prepare → evaluate or optimize → numerical artifact
validation → report → inspection. Reports and later inspections read only saved
artifacts. Synthetic target generation remains a separate, low-frequency script
because it changes an input artifact rather than advancing a run lifecycle.

## Experiment contract

Before adding an experiment, record:

- the scientific question and the claim the experiment could support;
- the conservative baseline and the single intended difference;
- fixed inputs, masks, potential points, random seeds, and execution order;
- metrics, gates, numerical diagnostics, and failure interpretation;
- the artifact paths needed for comparison without reintegration;
- an explicit stop/go decision for the next, more expensive stage.

Parameter-only variants should normally be new recipe/run TOML files using an
existing workflow. A new launcher is justified only when execution order,
provenance capture, or failure handling is genuinely different. A new shared
abstraction is justified after a second real consumer exists.

## Validation ladder

| Level | Entry condition | Exit condition | Claim allowed |
| --- | --- | --- | --- |
| Configuration and coverage | Typed config loads and required inputs are identified | Paths, grids, masks, and catalogue support are reviewed | The planned experiment is internally configured |
| Small-array and analytic tests | Equation or reusable numerical behavior is isolated from I/O | Units, shapes, limiting behavior, and known solutions pass | The implementation matches its local mathematical contract |
| One-potential benchmark | Production dependencies and full catalogue are available | Runtime, artifacts, solver status, failed-orbit audit, and gates are reviewed | The workflow can execute one engineering case |
| One-factor sensitivity | A common baseline and exactly one changed factor are defined | Paired artifacts are comparable without reintegration | The measured output is or is not sensitive to that factor |
| Fixed-potential ranking | Sensitivity changes an objective or solved weights materially | Identical potential points preserve or fail the documented ranking criteria | Potential ordering is screened for numerical stability |
| Adaptive scan | All preceding gates pass and open support decisions are resolved | Repeated runs and diagnostics show stable search behavior | The optimizer is ready for a bounded research scan |
| Scientific inference | Mock or held-out validation and selection assumptions are documented | Recovery, failure modes, and uncertainty interpretation are defensible | A scoped astrophysical claim may be considered |

Passing one level does not imply that later levels are satisfied. In
particular, a density gate is not evidence of stable potential recovery.

## Current decision register

| Decision | Status | Owner and consequence |
| --- | --- | --- |
| Package dependency direction and artifact-only reporting | verified | [architecture.md](architecture.md); enforced by architecture and artifact tests |
| Static Zhu et al. (2026) potential is the active model | verified | [zhu_2026_potential.md](zhu_2026_potential.md); changing components requires a fresh production benchmark |
| `catalogue_fixed` remains the conservative baseline | verified | Baseline for comparisons; do not reinterpret catalogue weights as free Schwarzschild coefficients |
| `density_solved` is an experimental alternative | verified | [density_solved_weights.md](density_solved_weights.md); implementation exists, scientific identifiability remains under study |
| Radial support is tested at `8 <= r < 40 kpc` | verified | [density_solved_r8_40_experiment.md](density_solved_r8_40_experiment.md); the density-only vertical mask remains |
| Initial-cell target-derived representative weights enter the optimizer | rejected | They remain diagnostic-only and are distinct from the trial-specific orbit-response solve |
| Tight-tolerance potential ranking is stable | pending | Complete and review the paired fixed-point test before an adaptive scan |
| Adaptive multi-trial search is ready | pending | Blocked on ranking, solver, regularization, and support review |
| Warm-start belongs in the main optimizer | pending | Deferred until after October 2026 and requires a separate experimental design |
| Synthetic DESI density is an absolute measurement | rejected | [desi_density_model.md](desi_density_model.md); treat it only as a relative tracer-density shape with configured synthetic error |

Use `verified` only for behavior supported by the current code, configuration,
tests, or inspected artifacts. Use `hypothesis` for a proposed scientific
explanation and `pending` for a decision awaiting evidence. Do not promote a
status based only on code existence.

## Change routing

- A physical equation, likelihood, selection assumption, support mask,
  objective, or scientific gate changes this blueprint or its owning model/
  experiment document before production implementation.
- A reusable array operation belongs in `halo_mw_lmc/core/` and must be
  testable with in-memory arrays.
- A survey or file-format adapter belongs in `halo_mw_lmc/data/`.
- Expensive orchestration belongs in `halo_mw_lmc/workflows/` and writes
  versioned artifacts.
- Artifact readers and figure builders belong in `artifacts.py`,
  `visualization/`, `apps/`, or thin scripts; they must not rerun AGAMA.
- A parameter-only experiment changes configuration and its experiment
  document, not the package architecture.

Large architecture changes should be proposed with a minimal option, a more
structured option, migration cost, and evidence that the new abstraction has a
real consumer. The accepted design becomes the new blueprint; it is never
silently rewritten by routine maintenance.
