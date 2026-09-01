# DESI year-1 K-giant synthetic density target

## Provenance and scope

The local source is `Desi/3D_density_profile.py`. It contains one analytic
triaxial broken-power-law density function and a 17-value fitted vector. The
first value, `-239.106294`, is accepted as an argument named `r_0` but is never
used by the source function. Consequently this integration treats the model as
a **relative tracer-density shape**, not an absolute Milky Way stellar density.

No paper title, DOI, arXiv identifier, parameter covariance, or posterior
samples were supplied with the local folder. Those must be added before this
target is described as a published posterior density estimate.

The accompanying object NPY contains 21,102 DESI K giants with sky position,
distance, velocities, errors, and completeness. It is an input catalogue behind
the modelling work; the gridded target generator does not deserialize or depend
on it at runtime.

## Analytic model

At spherical Galactocentric radius `s`, the intermediate-to-major and
minor-to-major axis ratios and two orientation angles are cubic polynomials:

```text
p(s)     = p0     + kp1 s + kp2 s² + kp3 s³
q(s)     = q0     + kq1 s + kq2 s² + kq3 s³
phi(s)   = phi0   + kphi1 s + kphi2 s² + kphi3 s³
theta(s) = theta0 + kth1 s + kth2 s² + kth3 s³
```

The source rotation is preserved exactly. Its rotated coordinates define

```text
m² = x'² + y'² / p(s)² + z'² / q(s)².
```

The relative density is continuous at `m1=15.8041 kpc` and
`m2=77.1921 kpc`, with slopes `1.2795`, `3.4636`, and `5.189`:

```text
rho = (m/m1)^(-alpha1)                                      m < m1
rho = (m/m1)^(-alpha2)                                 m1 <= m < m2
rho = (m2/m1)^(-alpha2) (m/m2)^(-alpha3)                    m >= m2
```

The tracked transcription and parameter record live in
`halo_mw_lmc/core/tracer_density.py`. The original `Desi/` files remain
unchanged and are never imported because the source script allocates three
40-million-element random arrays at module import time.

## Gridding and uncertainty

The generator volume-averages the model in every configured `(R,z,phi)` cell.
It uses Gauss--Legendre quadrature in `u=R²/2`, `z`, and `phi`, so the
cylindrical Jacobian is included exactly through `du=R dR`. The current local
artifact uses the No-Fixed recipe's `25x25x4` northern-halo grid:

```text
R:   0 .. 50 kpc
z:   0 .. 50 kpc
phi: -pi .. pi
```

Order 6 supplies the saved target; its difference from order 4 is saved as the
quadrature error. Because no model posterior uncertainty was supplied, the
configured target error is

```text
sqrt((0.10 * rho)² + quadrature_error²).
```

The 10% term is a synthetic model-discrepancy choice, not a DESI measurement
error. Change it only through the generator TOML and record sensitivity tests.

## Generated artifact

Run:

```bash
conda run -n dp-jax python scripts/generate_synthetic_density.py \
  configs/synthetic_density/desi_year1_kgiants.toml
```

The script is the recommended low-frequency interface. The installed
`halo-mw-lmc-density` command and `python -m halo_mw_lmc.generate_density`
remain compatibility entry points and call the same strict loader/workflow.

The generated file is
`data_for_model/synthetic/desi_year1_kgiants_25x25x4.npz`. It is ignored by Git
as generated research data. Besides the standard target arrays and grid edges,
the NPZ stores model parameters, source path and SHA-256, quadrature settings,
uncertainty semantics, and the unused fitted value. Generation is cold-start:
an existing output is never overwritten.

The current coordinate assumption is that the source script's Galactocentric
`x,y,z` handedness matches the active catalogue's `x_gc,y_gc,z_gc`. The source
figure places the Sun at `x=-8 kpc`, but the active catalogue contains no frame
metadata that independently proves the convention. Validate this before using
the target for a production inference claim.
