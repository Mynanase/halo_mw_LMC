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

   with SciPy `lsq_linear`. Failed seed integrations receive zero weight in the
   full `(N_seed,)` result.
4. Reconstruct `rho_model = A @ w_hat`. No additional density amplitude is fit.
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
the target mass inside the density fit mask and explicitly enforces
`sum(w) = 1`. The checked-in experimental recipe uses `unit_mass`; this makes
the target a shape constraint and gives the solved weights a consistent scale.

No-Fixed recipes must set `density_fit.normalization = "none"`. Allowing a
second fitted density scale would introduce the degeneracy `w -> c*w` and
`scale -> scale/c`.

## Persisted diagnostics

Every row in `sample.dat` records density chi-square and chi-square per fitted
bin, velocity-only and joint objectives, regularization penalty, inner solver
objective, effective orbit count, maximum weight fraction, active orbit count,
and solver convergence/status. The complete `(N_seed,)` weights are stored only
for the current best trial in `best/evaluation.npz`, together with the density
and velocity arrays needed by reports and Marimo.
