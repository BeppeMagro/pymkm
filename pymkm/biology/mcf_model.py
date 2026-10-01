"""
MCF-MKM biological model utilities.

This module implements functions to compute:

- Nuclear single-event specific energy from domain scaling
- MCF-MKM non-Poisson correction factor c(b)
- Dose-averaged MCF-MKM quantities c̄ and z̄^(c)
- MCF-MKM linear-quadratic coefficients α and β

The functions operate on single-event specific energies provided by the
existing :class:`~pymkm.physics.specific_energy.SpecificEnergy` machinery.
The nuclear specific energy can either be supplied independently or obtained
from the domain specific energy according to the scaling used in the original
MCF-MKM formulation.

Intended for use by the MCF-MKM microdosimetric and survival-modeling pipeline.
"""

import numpy as np
from typing import Tuple
from pymkm.utils.geometry_tools import GeometryTools
from pymkm.utils.integration import integrate_1d


def compute_scaled_nuclear_specific_energy(
    z_domain: np.ndarray,
    domain_radius: float,
    nucleus_radius: float
) -> np.ndarray:
    """
    Compute nuclear single-event specific energy by geometric scaling.

    Following the original MCF-MKM formulation, the same lineal-energy event
    is associated with the domain and nucleus, so that

        z_n,1(b) = z_d,1(b) * (r_d / R_n)^2

    :param z_domain: Domain single-event specific energy z_d,1(b) [Gy].
    :type z_domain: np.ndarray
    :param domain_radius: Radius of the subnuclear domain r_d [μm].
    :type domain_radius: float
    :param nucleus_radius: Radius of the cell nucleus R_n [μm].
    :type nucleus_radius: float

    :return: Nuclear single-event specific energy z_n,1(b) [Gy].
    :rtype: np.ndarray

    :raises ValueError: If domain_radius or nucleus_radius is not strictly positive.
    """
    radius_ratio_squared = GeometryTools.calculate_squared_radius_ratio(
        domain_radius,
        nucleus_radius
    )
    return z_domain * radius_ratio_squared


def compute_mcf_correction_factor(
    z_domain: np.ndarray,
    z_nucleus: np.ndarray,
    alpha0: float,
    beta0: float
) -> np.ndarray:
    """
    Compute the MCF-MKM non-Poisson correction factor c(b).

    The correction factor is evaluated for each impact parameter from
    single-event domain and nucleus specific energies:

        T(b) = alpha0 * z_n,1(b)
             + beta0 * z_d,1(b) * z_n,1(b)
             + beta0 * z_n,1(b)^2

        c(b) = [1 - exp(-T(b))] / T(b)

    :param z_domain: Domain single-event specific energy z_d,1(b) [Gy].
    :type z_domain: np.ndarray
    :param z_nucleus: Nucleus single-event specific energy z_n,1(b) [Gy].
    :type z_nucleus: np.ndarray
    :param alpha0: Linear LQ coefficient in the limit of vanishing lineal energy [Gy⁻¹].
    :type alpha0: float
    :param beta0: Quadratic LQ coefficient in the limit of vanishing lineal energy [Gy⁻²].
    :type beta0: float

    :return: MCF-MKM correction factor c(b).
    :rtype: np.ndarray

    :raises ValueError: If z_domain and z_nucleus do not have the same shape.
    """
    if np.shape(z_domain) != np.shape(z_nucleus):
        raise ValueError("z_domain and z_nucleus must have the same shape.")

    T = (
        alpha0 * z_nucleus
        + beta0 * z_domain * z_nucleus
        + beta0 * z_nucleus ** 2
    )

    c = np.ones_like(T, dtype=float)
    eps = 1e-12
    mask = np.abs(T) > eps
    c[mask] = -np.expm1(-T[mask]) / T[mask]
    return c


def compute_mcf_averaged_quantities(
    z_domain: np.ndarray,
    b_array: np.ndarray,
    c: np.ndarray,
    integration_method: str = "trapz"
) -> Tuple[float, float]:
    """
    Compute dose-averaged MCF-MKM quantities c̄ and z̄^(c).

    For a uniform transverse fluence, the dose-weighted averages are
    calculated in impact-parameter space as

        c̄ = [ ∫ c(b) z_d,1(b) b db ]
             / [ ∫ z_d,1(b) b db ]

    and

        z̄^(c) = [ ∫ z_d,1(b)^2 c(b) b db ]
                 / [ ∫ z_d,1(b) b db ].

    :param z_domain: Domain single-event specific energy z_d,1(b) [Gy].
    :type z_domain: np.ndarray
    :param b_array: Impact parameter values [μm], must be sorted.
    :type b_array: np.ndarray
    :param c: Event-wise MCF-MKM correction factor c(b).
    :type c: np.ndarray
    :param integration_method: Integration rule to use: 'trapz', 'simps', or 'quad'.
    :type integration_method: str

    :return: Tuple (c_bar, z_bar_c), where c_bar is dimensionless and z_bar_c is in Gy.
    :rtype: tuple[float, float]

    :raises ValueError: If input arrays have different shapes or the integration method is invalid.
    """

    if not (np.shape(z_domain) == np.shape(b_array) == np.shape(c)):
        raise ValueError("z_domain, b_array, and c must have the same shape.")

    denom = integrate_1d(
        z_domain * b_array,
        b_array,
        method=integration_method
    )
    if denom == 0:
        return 0.0, 0.0

    c_bar = integrate_1d(
        c * z_domain * b_array,
        b_array,
        method=integration_method
    ) / denom
    z_bar_c = integrate_1d(
        z_domain ** 2 * c * b_array,
        b_array,
        method=integration_method
    ) / denom

    return c_bar, z_bar_c

def compute_mcf_lq_coefficients(
    c_bar: float,
    z_bar_c: float,
    alpha0: float,
    beta0: float
) -> Tuple[float, float]:
    """
    Compute MCF-MKM linear-quadratic coefficients from averaged quantities.

    The MCF-MKM coefficients are reconstructed from the dose-averaged
    quantities stored in the microdosimetric table as

        alpha_MCF = alpha0 * c_bar + beta0 * z_bar_c

        beta_MCF = beta0 * c_bar^2

    :param c_bar: Dose-averaged MCF-MKM correction factor c̄.
    :type c_bar: float
    :param z_bar_c: Dose-averaged corrected domain specific energy z̄^(c) [Gy].
    :type z_bar_c: float
    :param alpha0: Linear LQ coefficient in the limit of vanishing lineal energy [Gy⁻¹].
    :type alpha0: float
    :param beta0: Quadratic LQ coefficient in the limit of vanishing lineal energy [Gy⁻²].
    :type beta0: float

    :return: Tuple (alpha_MCF, beta_MCF) [Gy⁻¹, Gy⁻²].
    :rtype: tuple[float, float]
    """
    alpha_mcf = alpha0 * c_bar + beta0 * z_bar_c
    beta_mcf = beta0 * c_bar ** 2
    return alpha_mcf, beta_mcf

