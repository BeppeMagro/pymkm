"""
MKM and SMK biological model utilities.

This module centralizes the calculation of the linear-quadratic coefficients
used by the classic MKM and its stochastic extension (SMK). Microdosimetric
quantities are supplied by the MKTable pipeline; this module only converts
those quantities into model-specific biological coefficients.
"""

from typing import Tuple, Union
import numpy as np

Numeric = Union[float, np.ndarray]


def compute_mkm_lq_coefficients(
    z_bar_star_domain: Numeric,
    alpha0: float,
    beta0: float
) -> Tuple[Numeric, Numeric]:
    """
    Compute the classic MKM linear-quadratic coefficients.

        alpha_MKM = alpha0 + beta0 * z_bar_star_domain
        beta_MKM = beta0

    :param z_bar_star_domain: Saturation-corrected dose-mean domain specific energy [Gy].
    :param alpha0: Low-LET linear LQ coefficient [Gy^-1].
    :param beta0: Low-LET quadratic LQ coefficient [Gy^-2].
    :return: Tuple ``(alpha_MKM, beta_MKM)``.
    """
    alpha_mkm = alpha0 + beta0 * z_bar_star_domain
    beta_mkm = beta0
    return alpha_mkm, beta_mkm


def compute_smk_lq_coefficients(
    z_bar_star_domain: Numeric,
    z_bar_domain: Numeric,
    alpha0: float,
    beta0: float
) -> Tuple[Numeric, Numeric]:
    """
    Compute the normoxic SMK linear-quadratic coefficients.

    The SMK uses the same linear coefficient as the classic MKM and scales
    the quadratic coefficient by the ratio of saturation-corrected to
    uncorrected dose-mean domain specific energy.

        alpha_SMK = alpha0 + beta0 * z_bar_star_domain
        beta_SMK = (z_bar_star_domain / z_bar_domain) * beta0

    Zero values of ``z_bar_domain`` are replaced by machine epsilon, matching
    the historical numerical safeguard used in ``SFTable.compute``.

    :param z_bar_star_domain: Saturation-corrected dose-mean domain specific energy [Gy].
    :param z_bar_domain: Uncorrected dose-mean domain specific energy [Gy].
    :param alpha0: Low-LET linear LQ coefficient [Gy^-1].
    :param beta0: Low-LET quadratic LQ coefficient [Gy^-2].
    :return: Tuple ``(alpha_SMK, beta_SMK)``.
    """
    alpha_smk, _ = compute_mkm_lq_coefficients(
        z_bar_star_domain=z_bar_star_domain,
        alpha0=alpha0,
        beta0=beta0,
    )
    z_bar_domain_safe = np.where(
        np.asarray(z_bar_domain) == 0,
        np.finfo(float).eps,
        z_bar_domain,
    )
    beta_smk = (z_bar_star_domain / z_bar_domain_safe) * beta0
    return alpha_smk, beta_smk

def compute_smk_gamma(
    alpha_smk: Numeric,
    beta_smk: Numeric,
    z_bar_nucleus: Numeric,
    dose: Numeric
) -> Numeric:
    """
    Compute the dose-dependent SMK correction coefficient ``gamma_SMK``.

    The stochastic MKM survival expression includes the multiplicative
    correction ``1 + gamma_SMK * D``, with

        gamma_SMK(D) = z_bar_nucleus *
            [0.5 * (alpha_SMK + 2 * beta_SMK * D)^2 - beta_SMK]

    This quantity is specific to the SMK formalism and is kept separate from
    :func:`compute_smk_lq_coefficients` because it depends explicitly on dose.

    :param alpha_smk: SMK linear coefficient [Gy^-1].
    :param beta_smk: SMK quadratic coefficient [Gy^-2].
    :param z_bar_nucleus: Dose-mean nucleus specific energy [Gy].
    :param dose: Dose value or dose array [Gy].
    :return: ``gamma_SMK`` with units of Gy^-1.
    """
    return z_bar_nucleus * (
        0.5 * (alpha_smk + 2 * beta_smk * dose) ** 2 - beta_smk
    )

