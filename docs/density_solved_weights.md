# Density-solved orbit weights

This document fixes the statistical and numerical contract for the experimental
No-Fixed mode. The fixed-catalogue mode remains available through a different
recipe and shares the same outer workflow.

## Per-potential evaluation

For each trial potential parameter vector `theta`:

1. Integrate all valid seed orbits with equal-time sampling.
2. Build the sparse three-dimensional response

   ```text
   A[j,i] = samples from orbit i in cell j / (finite samples of orbit i * V[j])
   ```

   where rows are C-order flattened `(R,z,phi)` cells and columns are successful
   seed orbits.
3. On the configured density fit mask, solve

   ```text
   w_hat(theta) = argmin(w >= 0) [chi2_density(theta,w) + lambda * ||w||2]
   ```

   through one configured numerical backend. The production default remains
   SciPy `lsq_linear` (TRF/LSMR, explicit `max_iter` and a configured fixed
   inner `lsmr_tol`). Two solver-only benchmark backends preserve the same
   objective and non-negative constraint:

   - `dense_nnls` forms `[A; sqrt(lambda) I]` once and calls SciPy's dense
     NNLS implementation;
   - `dual_ridge` solves the observation-space dual with a semismooth Newton
     step and Armijo line search, then recovers
     `w = max(-A.T @ y, 0) / lambda`. It requires `lambda > 0`.

   Seed orbits whose
   response is strictly zero inside the fit mask are dropped from the solve and
   restored as zero weight afterwards. Failed seed integrations also receive
   zero weight in the full `(N_seed,)` result.

All backends receive the exact same inverse-error-scaled CSR `A`, target `b`,
and L2 strength. A SHA-256 problem fingerprint is persisted so benchmark runs
can reject comparisons that did not solve the same numerical problem. The
normalized primal KKT residual is recorded for every backend. Alternative
backends are converged only when their own termination succeeds and this KKT
residual is no larger than `solver_tolerance`; the historical `lsq_linear`
success rule remains unchanged for backward compatibility.

The default `lsmr_tol = 1e-6` replaces SciPy's `"auto"` rule, which kept the
inner LSMR solve coarse while the optimality residual was large and stalled
outer convergence for tens of thousands of iterations. With the tight fixed
tolerance the TRF outer loop reaches the same accuracy in a few hundred
iterations. Setting `lsmr_tol = null` restores the old `"auto"` behaviour.
4. Reconstruct `rho_model = A @ w_hat`. No additional density amplitude is fit.
   The target is normalized to unit mass inside the fit mask, so the
   least-squares solution itself sets the total weight scale; the solver does
   **not** impose `sum(w) = 1` and no post-solve renormalization is applied.
5. Give every finite sample from orbit `i` the histogram weight `w_hat[i] / N_i`,
   then calculate the existing conditional velocity likelihood for `r >= 8 kpc`.

The density solve and velocity likelihood are therefore simultaneous at the
outer potential level: every trial potential gets its own profiled weights, and
those exact weights are used immediately for that trial's velocity score.

## Outer objectives

Every evaluation computes and persists both scalars:

```text
J_velocity = -log L_velocity
J_density_velocity = 0.5 * chi2_density - log L_velocity
```

The recipe selects which scalar is sent to the optimizer. `velocity_only`
requires a finite `density_max_chi2_per_bin`; a trial that does not converge or
fails this density-quality gate receives a large finite penalty. The joint mode
uses the profiled density chi-square directly and has no separate gate.

The L2 penalty chooses a stable solution among density-compatible orbit
mixtures. It is recorded independently and is not added again to either outer
statistical objective.

## Density normalization

`absolute` treats the target density amplitude as physical and lets it determine
the total orbit weight. `unit_mass` divides target density and uncertainty by
the target mass inside the density fit mask. The check-in experimental recipe
uses `unit_mass`, so the target is a fixed-amplitude shape constraint. The
inverse-error weighted least-squares problem then determines the total orbit
weight from that fixed amplitude directly; no `sum(w) = 1` constraint and no
post-solve renormalization are used, because a uniform rescaling of all weights
would leave the conditional velocity likelihood unchanged and a hard unit-mass
constraint was found to force the wrong total scale against the shape fit.

No-Fixed recipes must set `density_fit.normalization = "none"`. Allowing a
second fitted density scale would introduce the degeneracy `w -> c*w` and
`scale -> scale/c`.

## Persisted diagnostics

Every row in `sample.dat` records density chi-square and chi-square per fitted
bin, velocity-only and joint objectives, regularization penalty, inner solver
objective, effective orbit count, maximum weight fraction, active orbit count,
exact-zero weight fraction, solver convergence/status, normalized KKT residual,
and solve-only wall time.
The backend, explicit `max_iter`, `solver_tolerance`, and backend-specific
`lsmr_tol` are recorded in `resolved_config.json`.
The complete `(N_seed,)` weights are stored only
for the current best trial in `best/evaluation.npz`, together with the density
and velocity arrays needed by reports and Marimo. For the best trial, the
inner-solve iteration count (`nit`), first-order optimality, and solver cost
are persisted together with backend, KKT residual, solve wall time, and problem
fingerprint. Peak RSS is measured externally by GNU `time -v` because an
in-process high-water mark cannot isolate one solver call reliably.

The independent 8--40 kpc benchmark additionally partitions the density
residuals into the velocity-aligned radial shells `[8,10,12,15,20,30,40]` and
persists shell and shell-by-phi chi-square per bin. Its velocity-only objective
requires every shell/phi cell to pass the configured limit; existing recipes
without shell fields retain the global gate only. See
`docs/density_solved_r8_40_experiment.md`.

## Open boundary: velocity floor vs density coverage

The conservative density fit mask constrains only `15 <= r < 40 kpc`, while the
velocity likelihood starts at `r >= 8 kpc`. Orbits that never enter the density
fit region therefore get weight only through the numerical L2 regularization and
the least-squares scale, with no density evidence. The intended resolution is to
align these boundaries: either raise the velocity floor to the density-start
radius, or extend the trustworthy density constraint down to the velocity floor.
This is an open decision and has not been applied to the production recipe.
