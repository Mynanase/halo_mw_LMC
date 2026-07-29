"""Gravitational-potential definitions used by the orbit pipeline."""

from __future__ import annotations

import math
from typing import Any


ZHU_2026_POTENTIAL_NAME = "zhu_2026_fiducial_constant_shape"

# Representative best-fitting values shown for the constant-shape halo in
# Zhu et al. (2026), including the model illustrated in their Figure 7.
ZHU_2026_BEST_FIT = {
    "rho0": 6.2,
    "log_rs": math.log10(70.0),
    "scale_radius": 70.0,
    "phalo": 0.8,
    "qhalo": 0.92,
    "gamma": 1.0,
}

# These are deliberately broader than the quoted p_DM and q_DM uncertainties.
# They are engineering defaults for a local optimization, not confidence
# intervals reported by the paper.
ZHU_2026_LOCAL_SEARCH_BOUNDS = {
    "qhalo": (0.70, 1.15),
    "phalo": (0.40, 1.20),
    "rho0": (5.50, 7.00),
    "rho0_plus_2logrs": (9.50, 10.30),
    "gamma": (0.50, 1.80),
}


def _require_finite(**values: float) -> None:
    invalid = sorted(name for name, value in values.items() if not math.isfinite(value))
    if invalid:
        raise ValueError(f"potential parameters must be finite: {', '.join(invalid)}")


def zhu_2026_component_parameters(
    rho0: float,
    log_rs: float,
    phalo: float,
    qhalo: float,
    gamma: float,
    alpha_halo: float = 0.0,
    beta_halo: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    """Return the four AGAMA components of Zhu et al. (2026), equations 6--8.

    ``rho0`` and ``log_rs`` are base-10 logarithms in ``Msun/kpc^3`` and kpc.
    The fiducial paper model fixes the two halo-orientation angles to zero.
    """

    _require_finite(
        rho0=rho0,
        log_rs=log_rs,
        phalo=phalo,
        qhalo=qhalo,
        gamma=gamma,
        alpha_halo=alpha_halo,
        beta_halo=beta_halo,
    )
    if phalo <= 0 or qhalo <= 0:
        raise ValueError("halo axis ratios p and q must be positive")
    if not 0 <= gamma < 3:
        raise ValueError("halo inner slope gamma must satisfy 0 <= gamma < 3")
    if not math.isclose(alpha_halo, 0.0, abs_tol=1e-12) or not math.isclose(
        beta_halo, 0.0, abs_tol=1e-12
    ):
        raise NotImplementedError(
            "the Zhu et al. (2026) fiducial halo fixes alpha_halo=beta_halo=0"
        )

    return (
        {
            "type": "Ferrers",
            "mass": 1.6e10,
            "scaleRadius": 3.5,
            "p": 0.44,
            "q": 0.31,
        },
        {
            "type": "Disk",
            "mass": 3.16e10,
            "scaleRadius": 2.6,
            "scaleHeight": 0.3,
            "innerCutoffRadius": 7.0,
            "sersicIndex": 1.0,
        },
        {
            "type": "Disk",
            "mass": 6.0e9,
            "scaleRadius": 2.0,
            "scaleHeight": 0.9,
            "innerCutoffRadius": 0.0,
            "sersicIndex": 1.0,
        },
        {
            "type": "Spheroid",
            "densityNorm": 10.0**rho0,
            "alpha": 1.0,
            "beta": 3.0,
            "gamma": gamma,
            "scaleRadius": 10.0**log_rs,
            "p": phalo,
            "q": qhalo,
            "outerCutoffRadius": 500.0,
            "cutoffStrength": 5.0,
        },
    )


def build_zhu_2026_potential(
    rho0: float,
    log_rs: float,
    phalo: float,
    qhalo: float,
    gamma: float,
    alpha_halo: float = 0.0,
    beta_halo: float = 0.0,
):
    """Construct the validated fiducial Zhu et al. (2026) AGAMA potential."""

    components = zhu_2026_component_parameters(
        rho0,
        log_rs,
        phalo,
        qhalo,
        gamma,
        alpha_halo,
        beta_halo,
    )

    import agama

    agama.setUnits(length=1, velocity=1, mass=1)
    # Supplying all dictionaries in one call lets AGAMA group compatible
    # density components while retaining the analytic Ferrers component.
    return agama.Potential(*components)
