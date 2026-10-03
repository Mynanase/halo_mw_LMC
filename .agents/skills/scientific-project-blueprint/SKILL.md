---
name: scientific-project-blueprint
description: Design or revise scientific goals, experiments, validation, data flow, and repository boundaries before a major research implementation. Use only when explicitly invoked for a new research direction, consequential experiment, or architecture change; do not use for routine fixes.
---

# Scientific Project Blueprint

Design the scientific and repository plan before consequential implementation.
Keep the design proportional to a single-researcher repository and explain why
each boundary exists.

## Ground the design

Before proposing changes:

1. Read the root `AGENTS.md`, `docs/project_blueprint.md`,
   `docs/architecture.md`, Git status, and the relevant model or experiment
   document.
2. Inspect the current entry points, configuration, artifacts, and tests. Do
   not infer the workflow from directory names alone.
3. Label statements as `verified`, `hypothesis`, or `pending`. Code
   existence is not scientific validation.
4. Read useful `.agent-local/` context when present, but treat it as
   non-portable and never upload it without explicit permission.

## Build the proposal

Make the proposal decision-complete:

- State the scientific question, intended claim, and explicit non-claims.
- Define the conservative baseline, controls, varied quantities, fixed
  quantities, data support, units, masks, random seeds, and computational
  constraints.
- Define a staged validation ladder with entry conditions, exit conditions,
  failure interpretation, and the claim allowed at each stage.
- Map data flow to the existing package boundaries and artifact contracts.
- Prefer configuration variants over copied scripts and require a second real
  consumer before adding a shared abstraction.
- For a genuine architecture choice, compare a minimal option with at most one
  more structured option, then recommend the smallest design supported by
  current evidence.
- Identify migration, compatibility, rollback, and production-only checks.

For `halo_mw_LMC`, density-gate success does not authorize an adaptive scan.
The fixed-potential ranking, solver, regularization, and support decisions in
`docs/density_solved_r8_40_experiment.md` must be reviewed first.

## Authorization boundary

Present the design for review before changing scientific code, configuration,
or experiment schedules. Explicit invocation of this skill authorizes design
work, not unrelated repository mutation.

After the user accepts the design and requests implementation:

1. Update `docs/project_blueprint.md` or the owning experiment/model document.
2. Implement only the accepted scope.
3. Add tests and artifact checks matching the validation ladder.
4. Report local evidence separately from production or real-catalogue checks.

Routine bug fixes that preserve scientific meaning do not require a blueprint
rewrite; route those through the repository maintenance workflow.
