import numpy as np
import pytest

from pymkm.biology.mkm_model import (
    compute_mkm_lq_coefficients,
    compute_smk_lq_coefficients,
    compute_smk_gamma,
)


def test_compute_mkm_lq_coefficients():
    alpha, beta = compute_mkm_lq_coefficients(
        z_bar_star_domain=2.0,
        alpha0=0.1,
        beta0=0.05,
    )
    assert alpha == pytest.approx(0.2)
    assert beta == pytest.approx(0.05)


def test_compute_smk_lq_coefficients():
    alpha, beta = compute_smk_lq_coefficients(
        z_bar_star_domain=1.5,
        z_bar_domain=2.0,
        alpha0=0.1,
        beta0=0.04,
    )
    assert alpha == pytest.approx(0.16)
    assert beta == pytest.approx((1.5 / 2.0) * 0.04)


def test_compute_smk_lq_coefficients_zero_domain_is_safe():
    alpha, beta = compute_smk_lq_coefficients(
        z_bar_star_domain=0.0,
        z_bar_domain=0.0,
        alpha0=0.1,
        beta0=0.04,
    )
    assert alpha == pytest.approx(0.1)
    assert float(beta) == pytest.approx(0.0)


def test_compute_smk_lq_coefficients_array_input():
    z_star = np.array([0.5, 1.0])
    z = np.array([1.0, 2.0])
    alpha, beta = compute_smk_lq_coefficients(z_star, z, 0.1, 0.04)
    np.testing.assert_allclose(alpha, np.array([0.12, 0.14]))
    np.testing.assert_allclose(beta, np.array([0.02, 0.02]))

def test_compute_smk_gamma_scalar():
    gamma = compute_smk_gamma(
        alpha_smk=0.16,
        beta_smk=0.03,
        z_bar_nucleus=0.2,
        dose=2.0,
    )
    expected = 0.2 * (0.5 * (0.16 + 2 * 0.03 * 2.0) ** 2 - 0.03)
    assert gamma == pytest.approx(expected)


def test_compute_smk_gamma_array_input():
    dose = np.array([0.0, 1.0, 2.0])
    gamma = compute_smk_gamma(
        alpha_smk=0.16,
        beta_smk=0.03,
        z_bar_nucleus=0.2,
        dose=dose,
    )
    expected = 0.2 * (0.5 * (0.16 + 2 * 0.03 * dose) ** 2 - 0.03)
    np.testing.assert_allclose(gamma, expected)

