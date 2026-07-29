# Zhu et al. (2026) fiducial potential

The active orbit pipeline uses the static, constant-shape potential described
by Zhu et al. (2026), *A vertically orientated dark matter halo marks a flip of
the Galactic disc*, equations 6--8.

## AGAMA mapping

| Physical component | AGAMA type | Fixed parameters |
| --- | --- | --- |
| Barred bulge | `Ferrers` | `mass=1.6e10`, `scaleRadius=3.5`, `p=0.44`, `q=0.31` |
| Thin disc | `Disk` | `mass=3.16e10`, `scaleRadius=2.6`, `scaleHeight=0.3`, `innerCutoffRadius=7`, `sersicIndex=1` |
| Thick disc | `Disk` | `mass=6e9`, `scaleRadius=2.0`, `scaleHeight=0.9`, `innerCutoffRadius=0`, `sersicIndex=1` |
| Dark-matter halo | `Spheroid` | `densityNorm=10**rho0`, `scaleRadius=10**log_rs`, `alpha=1`, `beta=3`, fitted `gamma`, `p`, and `q`, plus `outerCutoffRadius=500`, `cutoffStrength=5` |

The fiducial halo is aligned with the Galactic Cartesian axes. The two exposed
orientation arguments must therefore remain zero. The local optimizer defaults
are centred on the representative best-fitting values
`(rho0, rs, gamma, p, q) = (6.2, 70 kpc, 1.0, 0.8, 0.92)`.
When this point is inside the configured bounds, it is evaluated as the first
new trial before the normal Bayesian `ask`/`tell` loop continues.
The configured bounds are deliberately broader than the reported shape
uncertainties and must not be interpreted as paper confidence intervals.

The paper removes the mean LMC-induced velocity signal from the observational
sample. It does not add a time-dependent LMC potential to every trial in this
fiducial fit, so the abandoned backward/forward LMC integration is not part of
the active implementation.

The source-of-truth implementation is
`halo_mw_lmc.potentials.zhu_2026_component_parameters`; all AGAMA component
dictionaries are supplied in one `agama.Potential(...)` call so AGAMA may group
compatible density components efficiently.

## Measured computational cost

A controlled local benchmark used the same synthetic halo phase points, ten
orbital periods, and equal output sampling for both potentials. At 512 orbits
and 1000 samples per orbit, the previous simplified potential took 1.35 s and
the paper potential took 3.50 s, a factor of 2.59. The multiplier was similar
(2.63) in a smaller 128-orbit benchmark. Potential construction itself remained
about 0.02 s and is negligible.

Linear extrapolation to the paper's 11,415-star catalogue gives approximately
30 s versus 78 s per trial on the benchmark machine. This estimate excludes
input, binning, likelihood, plotting, and optimizer overhead; re-run the
full-catalogue benchmark in the production environment before a large scan.
