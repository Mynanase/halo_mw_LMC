---
name: scientific-repo-maintainer
description: Maintain the halo_mw_LMC scientific Python repository when changing numerical code, configs, experiments, tests, artifacts, or figures. Route work through the project blueprint and existing layer boundaries; do not use for unrelated repositories or initial major research design.
---

# Scientific Repository Maintainer

Keep daily changes consistent with the accepted scientific design without
adding unnecessary structure.

## Inspect and classify

Before editing:

1. Inspect Git status and preserve unrelated tracked and untracked work.
2. Read `AGENTS.md`, `docs/project_blueprint.md`,
   `docs/architecture.md`, and the document owning the requested behavior.
3. Search for an existing implementation, configuration, test, diagnostic, or
   artifact reader before creating a file.
4. Classify the request:
   - **Scientific model:** equation, potential, likelihood, weights, selection,
     support, objective, gate, or interpretation.
   - **Experiment design:** baseline, varied factor, schedule, acceptance, or
     progression to a more expensive stage.
   - **Implementation:** behavior-preserving code or configuration work.
   - **Diagnostic/display:** analysis or figures derived from saved artifacts.

## Route the change

- Put reusable array-only scientific behavior in `halo_mw_lmc/core/` and
  test it with small in-memory cases.
- Put catalogue or target file adaptation in `halo_mw_lmc/data/`.
- Put expensive orchestration in `halo_mw_lmc/workflows/`.
- Keep configuration parsing strict and keep scientific choices in recipe
  TOML while run identity, paths, seed, and iteration count stay in run TOML.
- Put persistence in `artifacts.py`; put figures and read-only interaction in
  `visualization/`, `apps/`, or a thin script.
- Never let reporting, Marimo, or plotting rerun missing AGAMA integrations.

Prefer a new configuration over a copied script for parameter-only variation.
Extract shared code only after a second real use is visible. Keep experiment
launchers thin and delegate scientific computation to tested package code.

## Protect scientific meaning

If a request changes a scientific assumption or the claim an output supports,
state the affected assumption and update the blueprint or owning document
before implementation. Stop and ask when an unresolved choice would materially
change the result.

An ordinary bug fix or internal refactor that preserves contracts needs focused
tests, not a ceremonial blueprint edit. A display-only request must not mutate
solver weights, model state, configuration, or archived artifacts.

Calibration examples:

- A new adaptive scan is experiment design and must satisfy the documented
  fixed-point and solver gates before launch.
- A density-mask or weight-semantics change is scientific, even if expressed
  as a TOML edit.
- A new orbit-weight figure reads saved artifacts, uses
  `visualization/`, and keeps its script thin.
- A pure numerical bug belongs in `core/` with a regression test and does
  not justify unrelated restructuring.

## Verify and report

Use the requested Conda environment from `AGENTS.md`; probe missing
dependencies and never install or silently switch environments. Run the
smallest relevant scientific and workflow tests, then compilation when Python
changed.

Finish by reporting:

- the design or experiment document consulted or updated;
- why each changed file belongs in its layer;
- checks completed and their environment;
- production, real-data, rendering, or long-run validation still pending.
