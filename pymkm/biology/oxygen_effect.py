"""
Oxygen effect modeling (OSMK 2021 and 2023) for microdosimetry.

This module centralizes the model-specific oxygen calculations used by pyMKM:

- relative radioresistance;
- OSMK 2021 and OSMK 2023 radioresistance formulations;
- OSMK 2023 scaling of microdosimetric parameters;
- oxygen-corrected linear-quadratic coefficients.

The core model functions expose explicit biological parameters. Compatibility
wrappers accepting a parameter container are retained for the public API used
by pyMKM 1.0.x.
"""

from typing import Optional, Tuple, Union

import numpy as np

Numeric = Union[float, np.ndarray]


def compute_relative_radioresistance(
    K: float,
    pO2: float,
    K_mult: Numeric,
) -> Numeric:
    """
    Compute relative radioresistance from oxygen tension and a model term.

    :param K: Oxygen pressure [mmHg] at which the radioresistance transition is centered.
    :param pO2: Partial pressure of oxygen [mmHg].
    :param K_mult: Model-specific multiplicative term in the denominator.
    :return: Relative radioresistance ``R``.
    """
    denominator = pO2 + K * K_mult
    return (pO2 + K) / denominator


def _compute_scaling_factor_component(
    R: Numeric,
    f_max: float,
    Rmax: float,
) -> Numeric:
    """Compute one OSMK 2023 scaling-factor component from ``R``."""
    return 1 + (R - 1) * (f_max - 1) / (Rmax - 1)


def compute_scaling_factors(
    R: Numeric,
    f_rd_max: float,
    f_z0_max: float,
    Rmax: float,
) -> Tuple[Numeric, Numeric]:
    """
    Compute OSMK 2023 scaling factors for domain radius and ``z0``.

    :param R: Relative radioresistance.
    :param f_rd_max: Maximum domain-radius scaling factor at full hypoxia.
    :param f_z0_max: Maximum saturation-parameter scaling factor at full hypoxia.
    :param Rmax: Maximum radioresistance at ``pO2 = 0``.
    :return: Tuple ``(f_rd, f_z0)``.
    """
    f_rd = _compute_scaling_factor_component(R, f_rd_max, Rmax)
    f_z0 = _compute_scaling_factor_component(R, f_z0_max, Rmax) ** 2
    return f_rd, f_z0


def compute_osmk2021_radioresistance(
    z_bar_domain: Numeric,
    K: float,
    pO2: float,
    zR: float,
    gamma: float,
    Rm: float,
) -> Numeric:
    """
    Compute relative radioresistance according to OSMK 2021.

    :param z_bar_domain: Dose-mean domain specific energy [Gy].
    :param K: Oxygen pressure parameter [mmHg].
    :param pO2: Oxygen partial pressure [mmHg].
    :param zR: Radiation-quality oxygen parameter [Gy].
    :param gamma: Exponent of the OSMK 2021 radiation-quality term.
    :param Rm: Minimum value of the radiation-quality-dependent maximum radioresistance.
    :return: Relative radioresistance ``R``.
    """
    z_ratio = (z_bar_domain / zR) ** gamma
    K_mult = (z_ratio + 1) / (z_ratio + Rm)
    return compute_relative_radioresistance(K=K, pO2=pO2, K_mult=K_mult)


def compute_osmk2023_radioresistance(
    K: float,
    pO2: float,
    Rmax: float,
    f_rd_max: float,
    f_z0_max: float,
) -> Tuple[Numeric, Numeric, Numeric]:
    """
    Compute OSMK 2023 radioresistance and microdosimetric scaling factors.

    :param K: Oxygen pressure parameter [mmHg].
    :param pO2: Oxygen partial pressure [mmHg].
    :param Rmax: Maximum radioresistance at ``pO2 = 0``.
    :param f_rd_max: Maximum domain-radius scaling factor at full hypoxia.
    :param f_z0_max: Maximum saturation-parameter scaling factor at full hypoxia.
    :return: Tuple ``(R, f_rd, f_z0)``.
    """
    R = compute_relative_radioresistance(
        K=K,
        pO2=pO2,
        K_mult=1 / Rmax,
    )
    f_rd, f_z0 = compute_scaling_factors(
        R=R,
        f_rd_max=f_rd_max,
        f_z0_max=f_z0_max,
        Rmax=Rmax,
    )
    return R, f_rd, f_z0


def compute_osmk2023_effective_parameters(
    domain_radius: float,
    z0: float,
    f_rd: float,
    f_z0: float,
) -> Tuple[float, float]:
    """
    Apply OSMK 2023 scaling to the microdosimetric parameters.

    The shared numerical precision is three decimal places for the effective
    domain radius and two decimal places for the effective saturation parameter.

    :param domain_radius: Baseline domain radius [um].
    :param z0: Baseline saturation parameter [Gy].
    :param f_rd: OSMK 2023 domain-radius scaling factor.
    :param f_z0: OSMK 2023 saturation-parameter scaling factor.
    :return: Tuple ``(effective_domain_radius, effective_z0)``.
    """
    effective_domain_radius = round(domain_radius / f_rd, 3)
    effective_z0 = round(z0 * f_z0, 2)
    return effective_domain_radius, effective_z0


def compute_osmk_radioresistance(
    version: str,
    z_bar_domain: Numeric,
    params,
) -> Tuple[Numeric, Optional[Numeric], Optional[Numeric]]:
    """
    Compatibility dispatcher for OSMK 2021/2023 radioresistance.

    This retains the pyMKM 1.0.x container-based API. New internal code uses
    :func:`compute_osmk2021_radioresistance` and
    :func:`compute_osmk2023_radioresistance` directly.

    :param version: OSMK model version (``'2021'`` or ``'2023'``).
    :param z_bar_domain: Dose-mean domain specific energy [Gy].
    :param params: Parameter object with the attributes required by the selected version.
    :return: Tuple ``(R, f_rd, f_z0)``; the scaling factors are ``None`` for OSMK 2021.
    :raises ValueError: If the version is unsupported.
    """
    if version == "2021":
        R = compute_osmk2021_radioresistance(
            z_bar_domain=z_bar_domain,
            K=params.K,
            pO2=params.pO2,
            zR=params.zR,
            gamma=params.gamma,
            Rm=params.Rm,
        )
        return R, None, None

    if version == "2023":
        return compute_osmk2023_radioresistance(
            K=params.K,
            pO2=params.pO2,
            Rmax=params.Rmax,
            f_rd_max=params.f_rd_max,
            f_z0_max=params.f_z0_max,
        )

    raise ValueError(f"Unsupported OSMK version: {version}")


def compute_osmk_alpha(
    z_bar_star_domain: Numeric,
    R: Numeric,
    alphaL: float,
    alphaS: float,
    beta0: float,
) -> Numeric:
    """Compute the oxygen-corrected OSMK linear LQ coefficient."""
    return alphaL + (alphaS / R) + (beta0 * z_bar_star_domain / (R ** 2))


def compute_osmk_beta(
    z_bar_star_domain: Numeric,
    z_bar_domain: Numeric,
    R: Numeric,
    beta0: float,
) -> Numeric:
    """Compute the oxygen-corrected OSMK quadratic LQ coefficient."""
    return (z_bar_star_domain / z_bar_domain) * beta0 / (R ** 2)


def apply_oxygen_correction_alpha(
    z_bar_star_domain: Numeric,
    R: Numeric,
    params,
) -> Numeric:
    """
    Compatibility wrapper for the pyMKM 1.0.x container-based alpha API.
    """
    return compute_osmk_alpha(
        z_bar_star_domain=z_bar_star_domain,
        R=R,
        alphaL=params.alphaL,
        alphaS=params.alphaS,
        beta0=params.beta0,
    )


def apply_oxygen_correction_beta(
    z_bar_star_domain: Numeric,
    z_bar_domain: Numeric,
    R: Numeric,
    params,
) -> Numeric:
    """
    Compatibility wrapper for the pyMKM 1.0.x container-based beta API.
    """
    return compute_osmk_beta(
        z_bar_star_domain=z_bar_star_domain,
        z_bar_domain=z_bar_domain,
        R=R,
        beta0=params.beta0,
    )


def compute_osmk_lq_coefficients(
    z_bar_star_domain: Numeric,
    z_bar_domain: Numeric,
    R: Numeric,
    params=None,
    *,
    alphaL: Optional[float] = None,
    alphaS: Optional[float] = None,
    beta0: Optional[float] = None,
) -> Tuple[Numeric, Numeric]:
    """
    Compute oxygen-corrected OSMK linear-quadratic coefficients.

    The preferred API passes ``alphaL``, ``alphaS`` and ``beta0`` explicitly.
    For backward compatibility, a pyMKM 1.0.x-style ``params`` object may be
    supplied instead. The two forms cannot be mixed.

    :param z_bar_star_domain: Saturation-corrected dose-mean domain specific energy [Gy].
    :param z_bar_domain: Uncorrected dose-mean domain specific energy [Gy].
    :param R: Relative radioresistance.
    :param params: Optional compatibility parameter container.
    :param alphaL: Linear coefficient for lethal lesions [Gy^-1].
    :param alphaS: Linear coefficient for sublethal lesions [Gy^-1].
    :param beta0: Low-LET quadratic LQ coefficient [Gy^-2].
    :return: Tuple ``(alpha_OSMK, beta_OSMK)``.
    :raises ValueError: If explicit parameters are incomplete or mixed with ``params``.
    """
    explicit_values = (alphaL, alphaS, beta0)

    if params is not None:
        if any(value is not None for value in explicit_values):
            raise ValueError(
                "Provide either params or explicit alphaL/alphaS/beta0 values, not both."
            )
        alphaL = params.alphaL
        alphaS = params.alphaS
        beta0 = params.beta0
    elif any(value is None for value in explicit_values):
        raise ValueError(
            "alphaL, alphaS, and beta0 must all be provided when params is omitted."
        )

    z_bar_domain_safe = np.where(
        np.asarray(z_bar_domain) == 0,
        np.finfo(float).eps,
        z_bar_domain,
    )

    alpha_osmk = compute_osmk_alpha(
        z_bar_star_domain=z_bar_star_domain,
        R=R,
        alphaL=alphaL,
        alphaS=alphaS,
        beta0=beta0,
    )
    beta_osmk = compute_osmk_beta(
        z_bar_star_domain=z_bar_star_domain,
        z_bar_domain=z_bar_domain_safe,
        R=R,
        beta0=beta0,
    )
    return alpha_osmk, beta_osmk
