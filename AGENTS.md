# Project instructions for agents

## Scope

These instructions apply to the entire repository and must remain suitable for
Git. Machine-specific facts, private papers, unpublished notes, and disposable
benchmarks belong under `.agent-local/`, which is ignored. Agents may read it
when present but must not assume it is portable or upload it without permission.

## Project contract

This repository implements a Zhu-style empirical orbit-superposition model for
the Milky Way stellar halo with explicit `(R,z,phi)` density comparison and
optional velocity likelihood terms. The supported daily entry point is:

```bash
halo-mw-lmc run configs/runs/fix_weight.toml
```

Use `validate`, `preflight`, `coverage`, `evaluate`, `optimize`, `inspect`, and
`report` for isolated lifecycle stages. Historical `python -m halo_mw_lmc`
flags remain compatibility-only. Numerical code belongs in
`halo_mw_lmc/core/`, file adapters in `halo_mw_lmc/data/`, expensive execution
in `halo_mw_lmc/workflows/`, and figures in `halo_mw_lmc/visualization/`.
Reports, inspection, and apps consume persisted artifacts.

## Sources of truth

- `docs/project_blueprint.md`: goals, non-claims, validation ladder,
  decisions, and change process.
- `docs/architecture.md`: arrays, dependencies, configuration, artifacts,
  and extension rules.
- `docs/zhu_2026_potential.md`, `docs/density_solved_weights.md`, and
  `docs/desi_density_model.md`: active model contracts.
- `docs/density_solved_r8_40_experiment.md` and
  `docs/no_fixed_benchmark.md`: active experiment and production benchmark.

Before changing a physical or statistical assumption, state the model and
evidence, then read the blueprint and owning document. Keep mutable thresholds
and experiment status out of this file.

## Runtime policy

- Production/server commands use `conda run -n halo_lmc python <command>`;
  local debugging uses `conda run -n dp-jax python <command>`.
- Never silently switch environments or install/upgrade packages. Probe and
  report missing dependencies; use a disposable environment for build trials.
- AGAMA is vendored at `Agama-master/` with exact casing. Do not manage it
  with pip/Conda; rebuild it with its repository build after an ABI mismatch.
- The repository `.venv` is not a preferred scientific runtime.
- Read `.agent-local/notes/runtime-environments.md` when present for current
  probes, and verify again before expensive runs.

```bash
conda run -n halo_lmc python -m unittest discover -s tests -v
conda run -n halo_lmc python -m compileall -q halo_mw_lmc apps/results.py
```

## Stable scientific invariants

- The main optimizer is cold-start only. Resume, replay, warm-start, or
  historical-point injection requires a separately approved experiment.
- Use a fixed configured seed and a new output directory; never append to an
  existing `sample.dat`.
- Evaluation, `optimizer.tell()`, and persisted samples use identical
  coordinates.
- The 201-bin fitting velocity grid is independent of plot coarsening.
- `catalogue_fixed` uses catalogue `w`; `density_solved` profiles
  non-negative trial weights. Never mix or post-normalize their semantics.
- `density_solved` uses the sparse equal-time response, each orbit's finite
  sample count, density scale one, and the solved weights for velocity. See
  `docs/density_solved_weights.md` for the full solver contract.
- Stars at `r < 8 kpc` are diagnostic-only and never affect the velocity
  objective.
- The active potential is the static Zhu et al. (2026) implementation in
  `halo_mw_lmc/core/potentials.py`; do not restore time-dependent LMC
  integration.
- Coverage plots are raw catalogue number densities, not selection-corrected
  physical densities. Before changing sparse/empty 6D treatment, run
  `halo-mw-lmc coverage configs/runs/fix_weight.toml`.

## Change workflow

1. Inspect the tree, Git status, blueprint, architecture, and owning document.
2. Classify the change as model, experiment, implementation, diagnostic, or
   display work.
3. Update the design first when the question, support, objective, gate, or
   interpretation changes.
4. Prefer configuration over copied scripts; abstract only after a second real
   use exists.
5. Add small-array tests for equations and integration tests for workflow or
   configuration contracts.
6. Separate local evidence from production or real-catalogue validation.

## Repository hygiene

- Preserve unrelated tracked and untracked work; ignored data may still exist.
- Keep `archive/` outside active imports and keep plotting out of optimization.
  Reports and Marimo must not reconstruct missing integrations.
- Do not commit generated data, runs, local environments, `.agent-local/`,
  machine paths, secrets, or copyrighted/private research material.
- Use `.agent-local/papers/`, `notes/`, and `benchmarks/` for local
  material. Index implementation-relevant papers in
  `.agent-local/notes/paper-index.md` with the cited section or equation.
- Before a large scan or potential-component change, rerun the full-catalogue
  production benchmark and keep benchmark changes separate from model changes.
